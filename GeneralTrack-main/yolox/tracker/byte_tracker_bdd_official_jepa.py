"""GeneralTrack association adapter using the official GOT-JEPA filter path.

This module is intentionally separate from the compact-vector adapter.  It
keeps GeneralTrack's detector, Point2InstanceRelation association, Kalman
filter, and lifecycle, but uses the official spatial ToMP head map as follows:

    accepted track history -> frozen filter_predictor -> frozen classifier
    candidate head maps --------------------------------------------------^

The candidate maps are only consumed by the final classifier score.  The
filter predictor receives a candidate-independent test map (the accepted
track's causal map prototype), so the predictor cannot select its own
candidate.  The official model is cached once per process because the
evaluator creates a tracker object at every video boundary.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

from .byte_tracker_bdd_jepa import BYTETracker as VectorBYTETracker


GOT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main").resolve()
GOT_EXPERIMENTS = GOT_ROOT / "experiments"
GOT_PYTRACKING = GOT_ROOT / "pytracking"
GOT_COTRACKER = GOT_PYTRACKING / "ltr" / "cotracker2" / "co_tracker"
HYBRIDSORT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT").resolve()
for _path in (GOT_COTRACKER, GOT_PYTRACKING, GOT_EXPERIMENTS, HYBRIDSORT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


_RUNTIME_CACHE = {}


def _load_runtime(checkpoint: str, device: str, batch: int):
    key = (str(Path(checkpoint).resolve()), str(device))
    if key not in _RUNTIME_CACHE:
        from mot_ocsort_jepa_official_v2 import OfficialLatent

        _RUNTIME_CACHE[key] = OfficialLatent(Path(checkpoint), device, int(batch))
    return _RUNTIME_CACHE[key]


def _map_mean(maps):
    arr = np.asarray(maps, dtype=np.float32)
    return np.mean(arr, axis=0).astype(np.float32)


class BYTETracker(VectorBYTETracker):
    """Drop-in tracker with a causal official JEPA filter score."""

    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate=frame_rate)
        self.jepa_mode = "official_filter"
        self.official_history = int(getattr(args, "jepa_history", 4))
        self.official_checkpoint = str(getattr(
            args, "jepa_official_checkpoint",
            "/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar"))
        self.official_temperature = float(getattr(args, "jepa_official_temperature", 1.0))
        self.official_scope = str(getattr(args, "jepa_official_scope", "lost"))
        self.official = _load_runtime(self.official_checkpoint, "cuda:0",
                                      int(getattr(args, "jepa_official_batch", 8)))
        self.jepa_stats.update({
            "official_filter": True,
            "official_scope": self.official_scope,
            "official_temperature": self.official_temperature,
            "official_filter_calls": 0,
            "official_classifier_calls": 0,
            "official_valid_pairs": 0,
            "official_missing_track_maps": 0,
            "official_missing_detection_maps": 0,
            "official_score_min": None,
            "official_score_max": None,
        })
        self._official_valid = None

    def _make_detections(self, boxes, scores, classes, features=None, feature_maps=None):
        detections = super()._make_detections(boxes, scores, classes, features, feature_maps)
        for det in detections:
            if not hasattr(det, "official_map"):
                det.official_map = getattr(det, "jepa_map", None)
        return detections

    def _commit_feature(self, track, det, allow=True):
        # Keep the compact vector history too; it is useful in diagnostics,
        # while the official path below uses the spatial map history.
        super()._commit_feature(track, det, allow=allow)
        if not hasattr(track, "official_map_history"):
            track.official_map_history = deque(maxlen=max(1, self.official_history))
        if allow:
            fmap = getattr(det, "official_map", None)
            if fmap is not None:
                fmap = np.asarray(fmap, dtype=np.float32)
                if fmap.shape == (256, 27, 27) and np.isfinite(fmap).all():
                    track.official_map_history.append(fmap.copy())

    def _official_similarity(self, tracks, detections):
        """Return [0,1] per-track candidate scores from the official filter."""
        out = np.full((len(tracks), len(detections)), 0.5, dtype=np.float32)
        valid = np.zeros_like(out, dtype=bool)
        if not tracks or not detections:
            self._official_valid = valid
            return out

        candidate_maps = []
        for det in detections:
            fmap = getattr(det, "official_map", None)
            if fmap is None:
                self.jepa_stats["official_missing_detection_maps"] += 1
                candidate_maps.append(None)
            else:
                candidate_maps.append(np.asarray(fmap, dtype=np.float32))
        available_candidates = [x for x in candidate_maps if x is not None]
        if not available_candidates:
            self._official_valid = valid
            return out

        usable = []
        contexts = []
        for i, track in enumerate(tracks):
            hist = getattr(track, "official_map_history", None)
            if not hist:
                self.jepa_stats["official_missing_track_maps"] += 1
                continue
            arr = np.asarray(list(hist)[-max(1, self.official_history):], dtype=np.float32)
            if arr.ndim != 4 or arr.shape[1:] != (256, 27, 27) or not np.isfinite(arr).all():
                self.jepa_stats["official_missing_track_maps"] += 1
                continue
            usable.append(i)
            contexts.append(_map_mean(arr))
        if not usable:
            self._official_valid = valid
            return out

        # The test map is deliberately the same causal prototype for every
        # candidate of this track.  It is not any current detection map.
        omegas = self.official.predict_filter_batch(contexts, contexts)
        self.jepa_stats["official_filter_calls"] += len(usable)
        for row, track_index in enumerate(usable):
            present = [j for j, x in enumerate(candidate_maps) if x is not None]
            scores = self.official.classifier_scores(omegas[row], [candidate_maps[j] for j in present])
            self.jepa_stats["official_classifier_calls"] += 1
            if scores.size == 0 or not np.isfinite(scores).all():
                continue
            center = float(np.median(scores))
            scale = float(np.std(scores))
            scale = max(scale, 1e-4)
            temperature = max(self.official_temperature, 1e-4)
            z = np.clip((scores - center) / (scale * temperature), -12.0, 12.0)
            calibrated = 1.0 / (1.0 + np.exp(-z))
            out[track_index, present] = calibrated.astype(np.float32)
            valid[track_index, present] = True
            self.jepa_stats["official_valid_pairs"] += int(len(present))
            mn, mx = float(np.min(scores)), float(np.max(scores))
            old_min = self.jepa_stats["official_score_min"]
            old_max = self.jepa_stats["official_score_max"]
            self.jepa_stats["official_score_min"] = mn if old_min is None else min(float(old_min), mn)
            self.jepa_stats["official_score_max"] = mx if old_max is None else max(float(old_max), mx)
        self._official_valid = valid
        return out

    def _fuse_jepa(self, base_cost, tracks, detections, branch):
        if base_cost.size == 0:
            return base_cost
        if self.official_scope == "lost" and branch != "lost":
            return base_cost
        if self.official_scope == "high" and branch != "high":
            return base_cost
        if self.official_scope not in ("lost", "high", "both"):
            return base_cost
        self.jepa_stats["candidate_pairs"] += int(base_cost.size)
        official_sim = self._official_similarity(tracks, detections)
        valid = self._official_valid if self._official_valid is not None else np.zeros_like(official_sim, bool)
        base_sim = np.clip(1.0 - np.asarray(base_cost, dtype=np.float32), 0.0, 1.0)
        gate = np.zeros_like(base_sim, dtype=np.float32)
        for i, track in enumerate(tracks):
            if not np.any(valid[i]):
                continue
            gap = max(1, int(self.frame_id) - int(getattr(track, "frame_id", self.frame_id)))
            if gap <= self.jepa_max_gap:
                gate[i, valid[i]] = 1.0
        self.jepa_stats["eligible_pairs"] += int(np.sum(gate > 0))
        self.jepa_stats["fused_pairs"] += int(np.sum((gate > 0) & valid))
        if not np.any(gate > 0):
            return base_cost
        fused_sim = base_sim + self.jepa_lambda * gate * (official_sim - base_sim)
        return 1.0 - np.clip(fused_sim, 0.0, 1.0)
