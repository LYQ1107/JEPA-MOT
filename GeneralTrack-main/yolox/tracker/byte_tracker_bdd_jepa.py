"""GeneralTrack association adapter using frozen causal GOT-JEPA vectors.

This module intentionally keeps the original GeneralTrack detector, Kalman
filter, Point2InstanceRelation model, score fusion, and lifecycle unchanged.
The only extra signal is a candidate-independent temporal prototype built from
features committed by earlier accepted detections.  It is fused into the
existing association cost in a selected branch:

* ``lost``: lost-track reactivation only;
* ``lost_iou_gate``: lost-track fusion restricted to the native spatial gate;
* ``lost_fallback``: native lost assignment plus conservative JEPA recovery;
* ``ambiguous``: short-gap lost reactivation plus ambiguous high-score pairs;
* ``all``: short-gap lost reactivation and every eligible high-score pair.
* ``high_selective``: reward only source-calibrated high-agreement JEPA
  candidates in ambiguous high-score rows; native lost association is
  unchanged.
* ``pair_high``: use a source-trained JEPA/geometry pair probability as a
  positive-only reward in high-score association; lost association is native.
* ``pair_high_ambiguous``: apply that pair reward only to native-viable,
  ambiguous high-score rows; lost association is native.
* ``pair_relative_consensus``: use the source-trained pair score only for a
  relative, veto-only high-score conflict; native assignment and lifecycle
  remain the anchor.
* ``birth_confirm``: keep native association, but delay output of borderline
  new tracks until a causal JEPA-supported continuation confirms the birth.
* ``high_selective_lost_consensus``: combine the source-calibrated positive
  high-score reward with the conservative lost-track veto.
* ``lost_relative_swap``: keep the native lost assignment unless a causal
  JEPA query clearly prefers a different, spatially viable, unused candidate;
  then swap the detection without changing the number of reactivations.
* ``lost_consensus_score``: apply the conservative lost veto only below a
  fixed detector-score ceiling, preserving high-confidence native matches.
* ``high_fallback``: preserve native high-score matches and recover only
  unmatched high-score track/detection pairs under JEPA and spatial gates.
* ``high_fallback_pair``: the same conservative recovery, with a
  source-trained JEPA/geometry pair probability replacing raw cosine.
* ``high_fallback_lost_fallback``: apply the same conservative JEPA recovery
  to both unmatched high-score pairs and unmatched lost-track pairs.
* ``high_fallback_dormant``: additionally reuse an expired track ID only when
  a new birth has a source-calibrated JEPA match to a causal dormant gallery.
* ``high_fallback_swap``: add a two-way JEPA-consensus swap before the same
  conservative recovery, and may reassign an ambiguous row to an otherwise
  unmatched native-viable detection, without changing the native threshold.
* ``high_jepa_reassign``: keep the native high assignment as the anchor, but
  repair a row when a different native-viable detection has a fixed,
  source-calibrated JEPA preference; mutual swaps require evidence from both
  rows and the matched count is preserved.
* ``high_fallback_box``: preserve the native high assignment and conservative
  JEPA recovery, but hold the causal Kalman prediction when a native high
  match is simultaneously inconsistent in JEPA space and spatially distant.
* ``high_fallback_duplicate``: additionally suppress an unmatched high-score
  birth only when a current active track already explains it in both JEPA and
  box space.
* ``high_fallback_duplicate_geom``: geometry-only null control for the above
  birth veto; it uses the same IoU gate but deliberately ignores JEPA.
* ``high_fallback_duplicate_geom_jepa``: keeps the geometry-only birth veto
  while retaining real JEPA features for the high-score fallback association.
* ``high_fallback_duplicate_geom_jepa_birthconfirm``: adds causal JEPA birth
  confirmation to the geometry-only birth-veto control and real JEPA
  high-score fallback.
* ``high_fallback_duplicate_geom_jepa_consensus``: additionally vetoes a
  native high-score match only when its causal JEPA identity is below the
  fixed similarity floor, then runs the same fallback and geometry-only birth
  control.
* ``high_fallback_duplicate_geom_jepa_lostfallback``: combines the
  geometry-only birth control and margin-free high fallback with the
  conservative causal JEPA fallback on lost tracks.
* ``high_fallback_duplicate_geom_jepa_box``: combines the geometry-only birth
  control and margin-free high fallback with a causal JEPA/spatial check that
  can hold a distant native high-score box at its Kalman prediction.
* ``high_fallback_duplicate_geom_jepa_swap``: combines the geometry-only birth
  control with the causal two-way JEPA swap/recovery branch.
* ``high_fallback_duplicate_geom_jepa_reassign``: combines the geometry-only
  birth control with the causal JEPA native-reassignment branch.
* ``high_fallback_duplicate_geom_jepa_unique``: requires the duplicate
  candidate to have a clear causal JEPA identity margin over other local
  active tracks before suppressing its birth.
* ``high_fallback_duplicate_geom_jepa_expand``: keeps every geometry-only
  duplicate veto and lets causal JEPA identity evidence add only low-overlap
  duplicate vetoes inside the existing spatial neighbourhood.
* ``high_fallback_duplicate_geom_jepa_iou_tiebreak``: keeps the same JEPA
  fallback gate but uses predicted-box IoU only to resolve otherwise valid
  JEPA fallback candidates, without changing the assignment threshold.
* ``high_fallback_lost_consensus``: combine that high-score recovery with
  the conservative JEPA veto on native lost-track reactivation.
* ``high_jepa_iou``: on ambiguous high-score rows, replace the native
  appearance cost with a JEPA/IoU hybrid while leaving easy rows native.
* ``high_relative``: on ambiguous high-score rows, reward only the
  target-unlabeled JEPA top-1 candidate when its relative margin is clear;
  no source-domain absolute similarity floor is used.
* ``high_relative_consensus``: preserve the native high-score assignment and
  veto only a relative-JEPA conflict whose preferred candidate is local and
  otherwise unused; JEPA cannot steal an existing native match.
* ``high_selective_spatial``: source-calibrated positive JEPA reward in
  ambiguous high-score rows, additionally restricted to a predicted-box IoU
  neighbourhood.
* ``high_selective_spatial_lost_consensus``: combine the spatial high-score
  branch with the conservative lost-track consistency veto.

The detector rows carry a D-dimensional vector after the normal six GeneralTrack
columns.  If no vector is present, the adapter falls back to the original
association cost for that pair.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .basetrack import TrackState
from .byte_tracker_bdd import (
    BYTETracker as OriginalBYTETracker,
    STrack,
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)
from yolox.tracker import matching_bdd as matching
from .jepa_pair_adapter import load_pair_adapter
from .uadetrac_jepa_temporal_adapter import load_temporal_adapter


FEATURE_DIM = 256


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x))
    return x / max(n, 1e-8)


class BYTETracker(OriginalBYTETracker):
    """Drop-in GeneralTrack tracker with optional JEPA cost fusion."""

    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate=frame_rate)
        self.feature_dim = int(getattr(args, "jepa_feature_dim", None) or FEATURE_DIM)
        self.jepa_mode = str(getattr(args, "jepa_mode", "lost")).lower()
        self.jepa_lambda = float(getattr(args, "jepa_lambda", 0.35))
        self.jepa_max_gap = int(getattr(args, "jepa_max_gap", 8))
        self.jepa_ambiguity_margin = float(getattr(args, "jepa_ambiguity_margin", 0.08))
        self.jepa_relative_margin = float(getattr(args, "jepa_relative_margin", 0.03))
        self.jepa_high_iou = float(getattr(args, "jepa_high_iou", 0.05))
        self.jepa_history = int(getattr(args, "jepa_history", 4))
        self.jepa_history_aggregation = str(
            getattr(args, "jepa_history_aggregation", "mean")).lower()
        self.jepa_momentum_beta = float(getattr(args, "jepa_momentum_beta", 0.25))
        self.jepa_birth_score = float(getattr(args, "jepa_birth_score", 0.60))
        self.jepa_birth_min_sim = float(getattr(args, "jepa_birth_min_sim", 0.90))
        self.jepa_dormant_min_sim = float(getattr(args, "jepa_dormant_min_sim", 0.90))
        self.jepa_dormant_margin = float(getattr(args, "jepa_dormant_margin", 0.03))
        self.jepa_dormant_max_gap = int(getattr(args, "jepa_dormant_max_gap", 200))
        self.jepa_dormant_max_size = int(getattr(args, "jepa_dormant_max_size", 512))
        self.jepa_dormant_gallery = []
        self.jepa_veto_max_score = getattr(args, "jepa_veto_max_score", None)
        if self.jepa_veto_max_score is not None:
            self.jepa_veto_max_score = float(self.jepa_veto_max_score)
        temporal_path = getattr(args, "jepa_temporal_adapter", None)
        self.temporal_adapter = load_temporal_adapter(temporal_path) if temporal_path else None
        pair_path = getattr(args, "jepa_pair_adapter", None)
        self.pair_adapter = load_pair_adapter(pair_path) if pair_path else None
        self.jepa_stats = {
            "mode": self.jepa_mode,
            "feature_dim": self.feature_dim,
            "lambda": self.jepa_lambda,
            "max_gap": self.jepa_max_gap,
            "ambiguity_margin": self.jepa_ambiguity_margin,
            "relative_margin": self.jepa_relative_margin,
            "history_aggregation": self.jepa_history_aggregation,
            "momentum_beta": self.jepa_momentum_beta,
            "high_iou": self.jepa_high_iou,
            "candidate_pairs": 0,
            "eligible_pairs": 0,
            "fused_pairs": 0,
            "missing_track_feature_pairs": 0,
            "missing_detection_feature_pairs": 0,
            "top2_reorder_rows": 0,
            "low_fallback_pairs": 0,
            "temporal_query_count": 0,
            "temporal_fallback_count": 0,
            "lost_consensus_pairs": 0,
            "lost_consensus_vetoes": 0,
            "lost_fallback_pairs": 0,
            "lost_fallback_rejections": 0,
            "high_consensus_pairs": 0,
            "high_consensus_vetoes": 0,
            "high_relative_consensus_pairs": 0,
            "high_relative_consensus_vetoes": 0,
            "lost_relative_rows": 0,
            "lost_relative_switches": 0,
            "lost_score_gate_skips": 0,
            "high_fallback_pairs": 0,
            "high_fallback_rejections": 0,
            "high_swap_pairs": 0,
            "high_reassign_pairs": 0,
            "box_hold_candidates": 0,
            "box_holds": 0,
            "duplicate_candidates": 0,
            "duplicate_suppressed": 0,
            "duplicate_feature_expanded": 0,
            "dormant_gallery_archives": 0,
            "dormant_gallery_matches": 0,
            "dormant_gallery_rejections": 0,
            "consensus_gap_skips": 0,
            "pair_candidate_pairs": 0,
            "pair_eligible_pairs": 0,
            "birth_pending": 0,
            "birth_confirmed": 0,
            "birth_low_similarity": 0,
            "birth_output_suppressed": 0,
            "primary_pairs": 0,
        }

    def _attach_detection_feature(self, det: STrack, feature: np.ndarray | None) -> None:
        if feature is None:
            det.jepa_feature = None
            return
        f = np.asarray(feature, dtype=np.float32).reshape(-1)
        if f.size != self.feature_dim or not np.isfinite(f).all() or np.linalg.norm(f) <= 1e-8:
            det.jepa_feature = None
            return
        det.jepa_feature = _unit(f)

    def _commit_feature(self, track: STrack, det: STrack, allow: bool = True) -> None:
        """Commit only an accepted, sufficiently reliable detector feature."""
        if not hasattr(track, "jepa_history"):
            track.jepa_history = deque(maxlen=max(1, self.jepa_history))
        if allow and getattr(det, "jepa_feature", None) is not None:
            feature = np.asarray(det.jepa_feature, dtype=np.float32).copy()
            track.jepa_history.append(feature)
            if not hasattr(track, "jepa_frame_history"):
                track.jepa_frame_history = deque(maxlen=max(1, self.jepa_history))
            track.jepa_frame_history.append((int(self.frame_id), feature))

    def _jepa_box_update(self, track: STrack, det: STrack):
        """Keep a causal predicted box for a JEPA-inconsistent jump.

        The detector row is still consumed by the native assignment and its
        score is retained.  Only the state measurement and feature memory are
        held when both the frozen JEPA relation and the predicted-box
        neighbourhood reject the row.  This keeps the intervention online and
        avoids changing detector or association thresholds.
        """
        if self.jepa_mode not in ("high_fallback_box", "high_fallback_duplicate_geom_jepa_box"):
            return det, True
        query = self._track_query(track)
        feature = getattr(det, "jepa_feature", None)
        if query is None or feature is None:
            return det, True
        similarity = float(np.dot(query, feature))
        overlap = float(1.0 - matching.iou_distance([track], [det])[0, 0])
        min_similarity = float(getattr(self.args, "jepa_box_min_sim", 0.90))
        min_iou = float(getattr(self.args, "jepa_box_min_iou", 0.05))
        if similarity >= min_similarity or overlap >= min_iou:
            return det, True
        predicted_box = np.asarray(track.tlbr, dtype=np.float32)
        held = STrack(STrack.tlbr_to_tlwh(predicted_box), det.score, det.cls)
        self.jepa_stats["box_hold_candidates"] += 1
        self.jepa_stats["box_holds"] += 1
        return held, False

    def _jepa_duplicate(self, det: STrack, tracks) -> bool:
        """Detect a current unmatched detection already explained by a track."""
        if self.jepa_mode not in ("high_fallback_duplicate", "high_fallback_duplicate_geom", "high_fallback_duplicate_geom_jepa", "high_fallback_duplicate_geom_jepa_birthconfirm", "high_fallback_duplicate_geom_jepa_consensus", "high_fallback_duplicate_geom_jepa_lostfallback", "high_fallback_duplicate_geom_jepa_swap", "high_fallback_duplicate_geom_jepa_reassign", "high_fallback_duplicate_geom_jepa_unique", "high_fallback_duplicate_geom_jepa_expand", "high_fallback_duplicate_geom_jepa_iou_tiebreak"):
            return False
        query_tracks = {}
        for track in list(self.tracked_stracks) + list(tracks):
            if (track.state == TrackState.Tracked and
                    int(getattr(track, "frame_id", -1)) == int(self.frame_id)):
                query_tracks[int(track.track_id)] = track
        feature = getattr(det, "jepa_feature", None)
        geometry_only = self.jepa_mode in ("high_fallback_duplicate_geom", "high_fallback_duplicate_geom_jepa", "high_fallback_duplicate_geom_jepa_birthconfirm", "high_fallback_duplicate_geom_jepa_consensus", "high_fallback_duplicate_geom_jepa_lostfallback", "high_fallback_duplicate_geom_jepa_box", "high_fallback_duplicate_geom_jepa_swap", "high_fallback_duplicate_geom_jepa_reassign", "high_fallback_duplicate_geom_jepa_expand", "high_fallback_duplicate_geom_jepa_iou_tiebreak")
        expand_geometry_with_jepa = self.jepa_mode == "high_fallback_duplicate_geom_jepa_expand"
        if not geometry_only and not expand_geometry_with_jepa and feature is None:
            return False
        min_similarity = float(getattr(self.args, "jepa_duplicate_min_sim", 0.90))
        min_iou = float(getattr(self.args, "jepa_duplicate_iou", 0.30))
        feature_candidates = []
        for track in query_tracks.values():
            overlap = float(1.0 - matching.iou_distance([track], [det])[0, 0])
            self.jepa_stats["duplicate_candidates"] += 1
            feature_expanded = False
            if expand_geometry_with_jepa:
                # Preserve the complete geometry-only null behavior. JEPA
                # can only add a veto for a candidate that missed the
                # geometry gate, so the paired null differs in one place.
                accepted = overlap >= min_iou
                if not accepted and feature is not None:
                    query = self._track_query(track)
                    if query is not None:
                        feature_iou = float(getattr(self.args, "jepa_high_iou", 0.05))
                        similarity = float(np.dot(query, feature))
                        accepted = similarity >= min_similarity and overlap >= feature_iou
                        feature_expanded = bool(accepted)
            elif geometry_only:
                accepted = overlap >= min_iou
            else:
                query = self._track_query(track)
                if query is None:
                    continue
                similarity = float(np.dot(query, feature))
                if overlap >= min_iou:
                    feature_candidates.append((similarity, int(track.track_id)))
                accepted = similarity >= min_similarity and overlap >= min_iou
            if accepted and self.jepa_mode != "high_fallback_duplicate_geom_jepa_unique":
                if feature_expanded:
                    self.jepa_stats["duplicate_feature_expanded"] += 1
                self.jepa_stats["duplicate_suppressed"] += 1
                return True
        if self.jepa_mode == "high_fallback_duplicate_geom_jepa_unique" and feature_candidates:
            feature_candidates.sort(key=lambda item: (-item[0], item[1]))
            best = feature_candidates[0][0]
            second = feature_candidates[1][0] if len(feature_candidates) > 1 else -1.0
            min_margin = float(getattr(self.args, "jepa_duplicate_margin", 0.03))
            if best >= min_similarity and (len(feature_candidates) == 1 or
                                           best - second >= min_margin):
                self.jepa_stats["duplicate_suppressed"] += 1
                return True
        return False

    def _initialize_birth_state(self, track: STrack) -> None:
        """Initialize causal output confirmation for a newly born track."""
        if self.jepa_mode not in ("birth_confirm", "high_fallback_birth_confirm", "high_fallback_duplicate_geom_jepa_birthconfirm"):
            return
        pending = float(track.score) < self.jepa_birth_score
        track.jepa_birth_pending = bool(pending)
        track.jepa_birth_confirmed = not bool(pending)
        if pending:
            self.jepa_stats["birth_pending"] += 1

    def _confirm_birth(self, track: STrack, det: STrack) -> None:
        """Confirm a borderline birth using only its previous feature state."""
        if self.jepa_mode not in ("birth_confirm", "high_fallback_birth_confirm", "high_fallback_duplicate_geom_jepa_birthconfirm") or not getattr(track, "jepa_birth_pending", False):
            return
        proto = self._track_query(track)
        feat = getattr(det, "jepa_feature", None)
        if proto is None or feat is None:
            self.jepa_stats["birth_low_similarity"] += 1
            return
        sim = float(np.dot(proto, feat))
        if sim >= self.jepa_birth_min_sim:
            track.jepa_birth_pending = False
            track.jepa_birth_confirmed = True
            self.jepa_stats["birth_confirmed"] += 1
        else:
            self.jepa_stats["birth_low_similarity"] += 1

    def _output_allowed(self, track: STrack) -> bool:
        if self.jepa_mode not in ("birth_confirm", "high_fallback_birth_confirm", "high_fallback_duplicate_geom_jepa_birthconfirm"):
            return True
        allowed = not bool(getattr(track, "jepa_birth_pending", False))
        if not allowed:
            self.jepa_stats["birth_output_suppressed"] += 1
        return allowed

    def _dormant_enabled(self) -> bool:
        return self.jepa_mode in ("high_fallback_dormant", "dormant_reid")

    def _archive_dormant(self, track: STrack) -> None:
        """Store a removed track's causal JEPA identity for long-gap births."""
        if not self._dormant_enabled():
            return
        query = self._track_query(track)
        if query is None:
            return
        self.jepa_dormant_gallery.append({
            "track_id": int(track.track_id),
            "last_frame": int(track.frame_id),
            "query": np.asarray(query, dtype=np.float32).copy(),
        })
        if len(self.jepa_dormant_gallery) > self.jepa_dormant_max_size:
            self.jepa_dormant_gallery = self.jepa_dormant_gallery[-self.jepa_dormant_max_size:]
        self.jepa_stats["dormant_gallery_archives"] += 1

    def _claim_dormant_id(self, det: STrack) -> int | None:
        """Return a dormant ID using only the current birth feature and gallery."""
        if not self._dormant_enabled() or not self.jepa_dormant_gallery:
            return None
        feature = getattr(det, "jepa_feature", None)
        if feature is None:
            return None
        candidates = []
        for index, entry in enumerate(self.jepa_dormant_gallery):
            gap = int(self.frame_id) - int(entry["last_frame"])
            if gap < 1 or gap > self.jepa_dormant_max_gap:
                continue
            score = float(np.dot(entry["query"], feature))
            candidates.append((score, index))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], x[1]))
        best_score, best_index = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else -1.0
        if (best_score < self.jepa_dormant_min_sim or
                (len(candidates) > 1 and best_score - second_score < self.jepa_dormant_margin)):
            self.jepa_stats["dormant_gallery_rejections"] += 1
            return None
        entry = self.jepa_dormant_gallery.pop(best_index)
        self.jepa_stats["dormant_gallery_matches"] += 1
        return int(entry["track_id"])

    def _track_prototype(self, track: STrack) -> np.ndarray | None:
        hist = getattr(track, "jepa_history", None)
        if not hist:
            return None
        arr = np.asarray(list(hist)[-max(1, self.jepa_history):], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.feature_dim or not np.isfinite(arr).all():
            return None
        return _unit(np.mean(arr, axis=0))

    def _track_query(self, track: STrack) -> np.ndarray | None:
        """Return a causal predicted query, falling back to the mean history."""
        hist = getattr(track, "jepa_frame_history", None)
        if self.temporal_adapter is not None and hist and len(hist) >= 2:
            frames = [int(x[0]) for x in hist]
            feats = [np.asarray(x[1], np.float32) for x in hist]
            query = self.temporal_adapter["predict"](feats, frames, int(self.frame_id))
            if query is not None and np.isfinite(query).all() and np.linalg.norm(query) > 1e-8:
                self.jepa_stats["temporal_query_count"] += 1
                return _unit(query)
        if self.temporal_adapter is not None:
            self.jepa_stats["temporal_fallback_count"] += 1
        if self.jepa_history_aggregation == "last":
            raw_hist = getattr(track, "jepa_history", None)
            if raw_hist:
                return _unit(np.asarray(raw_hist[-1], dtype=np.float32))
        raw_hist = getattr(track, "jepa_history", None)
        if raw_hist:
            arr = np.asarray(list(raw_hist)[-max(1, self.jepa_history):], dtype=np.float32)
            if (arr.ndim == 2 and arr.shape[1] == self.feature_dim and
                    np.isfinite(arr).all()):
                if self.jepa_history_aggregation == "recent_weighted":
                    # Fixed causal decay: the newest accepted observation is
                    # weighted most, while older observations still suppress
                    # one-frame crop noise.
                    weights = np.linspace(0.35, 1.0, len(arr), dtype=np.float32)
                    return _unit(np.sum(arr * weights[:, None], axis=0))
                if self.jepa_history_aggregation == "last_mean":
                    return _unit(0.70 * arr[-1] + 0.30 * np.mean(arr, axis=0))
                if self.jepa_history_aggregation == "momentum" and len(arr) >= 2:
                    query = arr[-1] + self.jepa_momentum_beta * (arr[-1] - arr[-2])
                    return _unit(query)
        return self._track_prototype(track)

    def _jepa_similarity(self, tracks, detections) -> np.ndarray:
        out = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            if self.jepa_history_aggregation == "max":
                raw_hist = getattr(track, "jepa_history", None)
                hist = (np.asarray(list(raw_hist)[-max(1, self.jepa_history):],
                                    dtype=np.float32)
                        if raw_hist else None)
                if hist is None or hist.ndim != 2 or hist.shape[1] != self.feature_dim:
                    proto = None
                    bank = None
                else:
                    bank = np.asarray([_unit(x) for x in hist], dtype=np.float32)
                    proto = bank
            else:
                proto = self._track_query(track)
                bank = None
            if proto is None:
                self.jepa_stats["missing_track_feature_pairs"] += len(detections)
                continue
            for j, det in enumerate(detections):
                feat = getattr(det, "jepa_feature", None)
                if feat is None:
                    self.jepa_stats["missing_detection_feature_pairs"] += 1
                    continue
                if bank is None:
                    value = np.dot(proto, feat)
                else:
                    value = np.max(bank @ np.asarray(feat, dtype=np.float32))
                out[i, j] = float(np.clip(value, -1.0, 1.0))
        return out

    def _pair_probability(self, tracks, detections) -> np.ndarray:
        if self.pair_adapter is None:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)
        scores = self.pair_adapter.score_matrix(
            tracks, detections, self._track_query, int(self.frame_id))
        self.jepa_stats["pair_candidate_pairs"] += int(scores.size)
        self.jepa_stats["pair_eligible_pairs"] += int(
            np.sum(scores >= self.pair_adapter.threshold))
        return scores

    def _fuse_pair_probability(self, base_cost: np.ndarray, tracks, detections) -> np.ndarray:
        """Reward only source-calibrated pair-positive candidates."""
        if base_cost.size == 0 or self.pair_adapter is None:
            return base_cost
        probability = self._pair_probability(tracks, detections)
        base_sim = np.clip(1.0 - np.asarray(base_cost, dtype=np.float32), 0.0, 1.0)
        eligible = probability >= float(self.pair_adapter.threshold)
        # A pair adapter is not allowed to create a match that the native
        # GeneralTrack gate rejected.  This also keeps the positive-only
        # reward from lowering an originally invalid cost below the assignment
        # threshold.
        eligible &= np.asarray(base_cost, dtype=np.float32) <= float(self.args.match_thresh)
        if self.jepa_mode == "pair_high_ambiguous":
            # Only intervene where the learned GeneralTrack relation is
            # genuinely uncertain.  Easy rows remain byte-for-byte native.
            for i in range(base_cost.shape[0]):
                viable = np.where(np.isfinite(base_cost[i]) &
                                  (base_cost[i] <= float(self.args.match_thresh)))[0]
                if viable.size < 2:
                    eligible[i, :] = False
                    continue
                order = viable[np.argsort(base_cost[i, viable], kind="mergesort")]
                margin = float(base_cost[i, order[1]] - base_cost[i, order[0]])
                if margin > self.jepa_ambiguity_margin:
                    eligible[i, :] = False
        # Positive-only gating avoids allowing an out-of-domain pair model to
        # veto a native GeneralTrack match. It can only help when its
        # calibrated probability exceeds the native association similarity.
        boost = float(self.jepa_lambda) * eligible * np.maximum(
            0.0, probability - base_sim)
        return 1.0 - np.clip(base_sim + boost, 0.0, 1.0)

    def _fuse_jepa(self, base_cost: np.ndarray, tracks, detections, branch: str) -> np.ndarray:
        """Residual-fuse JEPA similarity without changing the base score scale."""
        if base_cost.size == 0 or self.jepa_mode in ("none", "off", "disabled"):
            return base_cost
        self.jepa_stats["candidate_pairs"] += int(base_cost.size)
        jepa_sim = self._jepa_similarity(tracks, detections)
        base_sim = np.clip(1.0 - np.asarray(base_cost, dtype=np.float32), 0.0, 1.0)
        gate = np.zeros_like(base_sim, dtype=np.float32)
        for i, track in enumerate(tracks):
            proto = self._track_query(track)
            if proto is None:
                continue
            gap = max(1, int(self.frame_id) - int(getattr(track, "frame_id", self.frame_id)))
            if gap > self.jepa_max_gap:
                continue
            if branch == "lost":
                gate[i, :] = 1.0
            elif branch == "lost_iou_gate":
                # The un-gated lost branch can let a high JEPA score replace
                # a much more plausible spatial candidate.  Keep this
                # variant inside the native IoU neighborhood; the threshold
                # is fixed before target evaluation and defaults to the
                # GeneralTrack match-threshold boundary.
                min_iou = float(getattr(self.args, "jepa_lost_iou", 0.10))
                gate[i, :] = (base_cost[i] <= (1.0 - min_iou)).astype(np.float32)
            elif branch == "high" and self.jepa_mode in (
                    "ambiguous", "all", "temporal_ambiguous", "high_selective",
                    "high_selective_lost_consensus", "high_relative",
                    "high_selective_spatial", "high_selective_spatial_lost_consensus",
                    "high_relative_consensus"):
                # Use the original GeneralTrack relation score to detect an
                # ambiguous row.  A large margin leaves easy assignments
                # untouched, making the intervention attributable to JEPA.
                vals = np.sort(base_sim[i])
                margin = float(vals[-1] - vals[-2]) if vals.size >= 2 else float("inf")
                if self.jepa_mode == "all" or margin <= self.jepa_ambiguity_margin:
                    gate[i, :] = 1.0
        eligible = (gate > 0) & (jepa_sim != 0)
        relative_gain = None
        if branch == "high" and self.jepa_mode == "high_relative":
            # Use only within-row JEPA ordering.  The representation may have
            # been calibrated on unlabeled target candidates, so an absolute
            # source-domain cosine floor would not be valid here.  The native
            # GeneralTrack cost still supplies the assignment scale and the
            # JEPA signal can only add a small positive top-1 bonus.
            eligible[:] = False
            relative_gain = np.zeros_like(jepa_sim, dtype=np.float32)
            rank_margin = float(self.jepa_relative_margin)
            for i in range(len(tracks)):
                if not np.any(gate[i] > 0):
                    continue
                viable = np.where((gate[i] > 0) &
                                  np.isfinite(jepa_sim[i]) &
                                  (base_cost[i] <= float(self.args.match_thresh)))[0]
                if viable.size < 2:
                    continue
                order = viable[np.argsort(jepa_sim[i, viable], kind="mergesort")]
                top1, top2 = int(order[-1]), int(order[-2])
                margin = float(jepa_sim[i, top1] - jepa_sim[i, top2])
                if margin < rank_margin:
                    continue
                eligible[i, top1] = True
                relative_gain[i, top1] = max(0.0, margin)
                self.jepa_stats.setdefault("relative_rank_rows", 0)
                self.jepa_stats["relative_rank_rows"] += 1
                self.jepa_stats.setdefault("relative_rank_pairs", 0)
                self.jepa_stats["relative_rank_pairs"] += 1
        if branch == "high" and self.jepa_mode in (
                "high_selective", "high_selective_lost_consensus",
                "high_selective_spatial", "high_selective_spatial_lost_consensus"):
            # A source-only calibration floor prevents the frozen JEPA
            # representation from perturbing native assignments when its
            # candidate crop is not sufficiently similar to the prototype.
            # This mode only rewards high-agreement pairs; it never turns a
            # low-similarity pair into a penalty.
            high_floor = float(getattr(self.args, "jepa_high_min_sim", 0.90))
            eligible &= jepa_sim >= high_floor
            if self.jepa_mode in ("high_selective_spatial", "high_selective_spatial_lost_consensus"):
                iou_sim = np.clip(1.0 - matching.iou_distance(tracks, detections), 0.0, 1.0)
                eligible &= iou_sim >= self.jepa_high_iou
        self.jepa_stats["eligible_pairs"] += int(np.sum(gate > 0))
        self.jepa_stats["fused_pairs"] += int(np.sum(eligible))
        if not np.any(eligible):
            return base_cost
        if relative_gain is not None:
            fused_sim = base_sim + self.jepa_lambda * relative_gain
        else:
            fused_sim = base_sim + self.jepa_lambda * gate * (jepa_sim - base_sim)
        return 1.0 - np.clip(fused_sim, 0.0, 1.0)

    def _fuse_jepa_iou_primary(self, base_cost: np.ndarray, tracks, detections) -> np.ndarray:
        """Use JEPA plus predicted-box IoU only where native matching is hard."""
        if base_cost.size == 0:
            return base_cost
        self.jepa_stats["candidate_pairs"] += int(base_cost.size)
        jepa_sim = self._jepa_similarity(tracks, detections)
        native_sim = np.clip(1.0 - np.asarray(base_cost, dtype=np.float32), 0.0, 1.0)
        iou_sim = np.clip(1.0 - matching.iou_distance(tracks, detections), 0.0, 1.0)
        gate = np.zeros_like(native_sim, dtype=bool)
        floor = float(getattr(self.args, "jepa_high_min_sim", 0.90))
        for i in range(len(tracks)):
            vals = np.sort(native_sim[i])
            margin = float(vals[-1] - vals[-2]) if vals.size >= 2 else float("inf")
            gap = max(1, int(self.frame_id) - int(getattr(tracks[i], "frame_id", self.frame_id)))
            if gap <= self.jepa_max_gap and margin <= self.jepa_ambiguity_margin:
                gate[i] = jepa_sim[i] >= floor
        self.jepa_stats["eligible_pairs"] += int(np.sum(gate))
        self.jepa_stats["primary_pairs"] += int(np.sum(gate))
        if not np.any(gate):
            return base_cost
        weight = float(np.clip(getattr(self.args, "jepa_primary_weight", 0.50), 0.0, 1.0))
        hybrid = (1.0 - weight) * iou_sim + weight * jepa_sim
        fused_sim = np.where(gate, hybrid, native_sim)
        return 1.0 - np.clip(fused_sim, 0.0, 1.0)

    def _top2_assignment(self, native_cost: np.ndarray, tracks, detections):
        """Conservative JEPA tie-break on an existing native assignment.

        The native GeneralTrack assignment is computed first.  JEPA may only
        replace a native top-1 candidate with that same row's native top-2,
        and only when the row is ambiguous, both candidates are viable, and
        the center-pooled JEPA similarity has a meaningful preference.  This
        preserves the original number of reactivations and prevents a weak
        appearance signal from creating a spatially implausible match.
        """
        native_matches, _, _ = matching.linear_assignment(
            native_cost, thresh=self.args.match_thresh
        )
        if native_cost.size == 0 or len(native_matches) == 0:
            return native_matches, tuple(range(native_cost.shape[0])), tuple(range(native_cost.shape[1]))
        if self.jepa_mode not in ("lost_top2", "high_top2"):
            return native_matches, tuple(np.setdiff1d(np.arange(native_cost.shape[0]), native_matches[:, 0])), tuple(np.setdiff1d(np.arange(native_cost.shape[1]), native_matches[:, 1]))

        jepa_sim = self._jepa_similarity(tracks, detections)
        row_to_det = {int(i): int(j) for i, j in native_matches.tolist()}
        det_to_row = {int(j): int(i) for i, j in native_matches.tolist()}
        proposals = []
        for i, current in list(row_to_det.items()):
            proto = self._track_prototype(tracks[i])
            if proto is None:
                continue
            gap = max(1, int(self.frame_id) - int(getattr(tracks[i], "frame_id", self.frame_id)))
            if gap > self.jepa_max_gap:
                continue
            viable = np.where(np.isfinite(native_cost[i]) &
                              (native_cost[i] <= float(self.args.match_thresh)))[0]
            if viable.size < 2:
                continue
            order = viable[np.argsort(native_cost[i, viable], kind="mergesort")]
            top1, top2 = int(order[0]), int(order[1])
            if current != top1:
                continue
            margin = float(native_cost[i, top2] - native_cost[i, top1])
            if margin > self.jepa_ambiguity_margin:
                continue
            if not np.isfinite(jepa_sim[i, top1]) or not np.isfinite(jepa_sim[i, top2]):
                continue
            # A small fixed preference threshold keeps near-ties from being
            # changed by float16/cache noise; it is not derived from GT.
            jdelta = float(jepa_sim[i, top2] - jepa_sim[i, top1])
            if jdelta <= 0.02:
                continue
            proposals.append((-jdelta, i, top1, top2))
        proposals.sort()
        for _, i, current, preferred in proposals:
            if row_to_det.get(i) != current:
                continue
            owner = det_to_row.get(preferred)
            if owner is not None:
                # A two-way native top-2 swap is allowed only if the other
                # row also has this candidate in its native top-2 and the
                # reverse switch is independently JEPA-preferred.
                owner_current = row_to_det.get(owner)
                if owner_current is None:
                    continue
                owner_viable = np.where(np.isfinite(native_cost[owner]) &
                                        (native_cost[owner] <= float(self.args.match_thresh)))[0]
                if owner_current not in owner_viable:
                    continue
                owner_order = owner_viable[np.argsort(native_cost[owner, owner_viable], kind="mergesort")]
                if preferred not in owner_order[:2] or current not in owner_order[:2]:
                    continue
                if not (jepa_sim[owner, current] > jepa_sim[owner, owner_current] + 0.02):
                    continue
                row_to_det[i], row_to_det[owner] = preferred, current
                det_to_row[preferred], det_to_row[current] = i, owner
                self.jepa_stats["top2_reorder_rows"] += 2
            else:
                row_to_det[i] = preferred
                det_to_row.pop(current, None)
                det_to_row[preferred] = i
                self.jepa_stats["top2_reorder_rows"] += 1

        matches = np.asarray(sorted(row_to_det.items()), dtype=int)
        matched_rows = matches[:, 0] if matches.size else np.empty((0,), dtype=int)
        matched_dets = matches[:, 1] if matches.size else np.empty((0,), dtype=int)
        unmatched_rows = tuple(np.setdiff1d(np.arange(native_cost.shape[0]), matched_rows))
        unmatched_dets = tuple(np.setdiff1d(np.arange(native_cost.shape[1]), matched_dets))
        return matches, unmatched_rows, unmatched_dets

    def _high_fallback_swap_assignment(self, native_cost, tracks, detections):
        """Repair only a mutually supported native high-score swap.

        GeneralTrack's native high-score assignment remains the anchor.  A
        JEPA swap is allowed only for two rows that are both native
        near-ties, whose cross-pairs remain inside the original assignment
        gate, and whose cross JEPA similarities pass fixed appearance and
        spatial gates with a per-row margin.  The subsequent fallback is the
        same unmatched-pair recovery used by ``high_fallback``.
        """
        native_matches, native_ut, native_ud = matching.linear_assignment(
            native_cost, thresh=self.args.match_thresh)
        native_matches = np.asarray(native_matches, dtype=int).reshape(-1, 2)
        if native_cost.size == 0 or len(native_matches) == 0:
            return native_matches, tuple(native_ut), tuple(native_ud)

        sim = self._jepa_similarity(tracks, detections)
        iou_sim = 1.0 - matching.iou_distance(tracks, detections)
        floor = float(getattr(self.args, "jepa_high_min_sim", 0.90))
        min_iou = float(getattr(self.args, "jepa_high_iou", 0.05))
        min_margin = float(getattr(self.args, "jepa_relative_margin", 0.03))
        ambiguity = float(getattr(self.args, "jepa_ambiguity_margin", 0.08))

        row_to_det = {int(i): int(j) for i, j in native_matches.tolist()}
        proposals = []
        rows = sorted(row_to_det)
        for pos, left in enumerate(rows):
            for right in rows[pos + 1:]:
                left_det = row_to_det[left]
                right_det = row_to_det[right]
                left_viable = np.where(np.isfinite(native_cost[left]) &
                                       (native_cost[left] <= float(self.args.match_thresh)))[0]
                right_viable = np.where(np.isfinite(native_cost[right]) &
                                        (native_cost[right] <= float(self.args.match_thresh)))[0]
                if left_viable.size < 2 or right_viable.size < 2:
                    continue
                left_order = left_viable[np.argsort(native_cost[left, left_viable], kind="mergesort")]
                right_order = right_viable[np.argsort(native_cost[right, right_viable], kind="mergesort")]
                if (left_det != int(left_order[0]) or right_det != int(right_order[0]) or
                        float(native_cost[left, left_order[1]] - native_cost[left, left_order[0]]) > ambiguity or
                        float(native_cost[right, right_order[1]] - native_cost[right, right_order[0]]) > ambiguity):
                    continue
                left_gap = max(1, int(self.frame_id) - int(
                    getattr(tracks[left], "frame_id", self.frame_id)))
                right_gap = max(1, int(self.frame_id) - int(
                    getattr(tracks[right], "frame_id", self.frame_id)))
                if left_gap > self.jepa_max_gap or right_gap > self.jepa_max_gap:
                    continue
                if (native_cost[left, right_det] > float(self.args.match_thresh) or
                        native_cost[right, left_det] > float(self.args.match_thresh)):
                    continue
                cross_left = float(sim[left, right_det])
                cross_right = float(sim[right, left_det])
                current_left = float(sim[left, left_det])
                current_right = float(sim[right, right_det])
                if (cross_left < floor or cross_right < floor or
                        float(iou_sim[left, right_det]) < min_iou or
                        float(iou_sim[right, left_det]) < min_iou or
                        cross_left - current_left < min_margin or
                        cross_right - current_right < min_margin):
                    continue
                gain = (cross_left - current_left) + (cross_right - current_right)
                proposals.append((-gain, left, right, left_det, right_det))

        used_rows = set()
        for _, left, right, left_det, right_det in sorted(proposals):
            if left in used_rows or right in used_rows:
                continue
            if row_to_det.get(left) != left_det or row_to_det.get(right) != right_det:
                continue
            row_to_det[left], row_to_det[right] = right_det, left_det
            used_rows.update((left, right))
            self.jepa_stats["high_swap_pairs"] += 1

        matches = np.asarray(sorted(row_to_det.items()), dtype=int).reshape(-1, 2)

        # A one-way replacement is allowed only to an originally unmatched
        # native-viable detection.  The matched count is preserved, while the
        # displaced native detection becomes eligible for the ordinary
        # unmatched fallback/new-track path.
        unmatched_dets = set(int(x) for x in np.asarray(native_ud, dtype=int).tolist())
        reassignments = []
        for row, current in sorted(row_to_det.items()):
            viable = np.where(np.isfinite(native_cost[row]) &
                              (native_cost[row] <= float(self.args.match_thresh)))[0]
            if viable.size < 2:
                continue
            order = viable[np.argsort(native_cost[row, viable], kind="mergesort")]
            if (float(native_cost[row, order[1]] - native_cost[row, order[0]]) > ambiguity or
                    current != int(order[0])):
                continue
            gap = max(1, int(self.frame_id) - int(
                getattr(tracks[row], "frame_id", self.frame_id)))
            if gap > self.jepa_max_gap:
                continue
            candidates = sorted(unmatched_dets)
            if not candidates:
                continue
            best = max(candidates, key=lambda det: float(sim[row, det]))
            gain = float(sim[row, best] - sim[row, current])
            if (float(sim[row, best]) < floor or gain < min_margin or
                    float(iou_sim[row, best]) < min_iou or
                    native_cost[row, best] > float(self.args.match_thresh)):
                continue
            reassignments.append((-gain, row, current, best))
        used_dets = set()
        for _, row, current, preferred in sorted(reassignments):
            if preferred in used_dets or preferred not in unmatched_dets:
                continue
            if row_to_det.get(row) != current:
                continue
            row_to_det[row] = preferred
            unmatched_dets.remove(preferred)
            unmatched_dets.add(current)
            used_dets.add(preferred)
            self.jepa_stats["high_reassign_pairs"] += 1

        matches = np.asarray(sorted(row_to_det.items()), dtype=int).reshape(-1, 2)

        # Keep the original high_fallback recovery for native unmatched rows,
        # plus any displaced native detection from a one-way replacement.
        old_ut = np.asarray(native_ut, dtype=int)
        old_ud = np.asarray(sorted(unmatched_dets), dtype=int)
        if old_ut.size and old_ud.size:
            rem_tracks = [tracks[int(i)] for i in old_ut.tolist()]
            rem_dets = [detections[int(i)] for i in old_ud.tolist()]
            rem_sim = self._jepa_similarity(rem_tracks, rem_dets)
            rem_iou = 1.0 - matching.iou_distance(rem_tracks, rem_dets)
            valid = (rem_sim >= floor) & (rem_iou >= min_iou)
            for ri, track in enumerate(rem_tracks):
                gap = max(1, int(self.frame_id) - int(
                    getattr(track, "frame_id", self.frame_id)))
                if gap > self.jepa_max_gap:
                    valid[ri, :] = False
                if np.any(valid[ri]):
                    vals = np.sort(rem_sim[ri, valid[ri]])[::-1]
                    if vals.size >= 2 and float(vals[0] - vals[1]) < min_margin:
                        valid[ri, :] = False
                        self.jepa_stats["high_fallback_rejections"] += 1
            fallback_cost = 1.0 - rem_sim
            fallback_cost[~valid] = 1e6
            fallback_matches, _, _ = matching.linear_assignment(
                fallback_cost, thresh=1.0 - floor)
            used_t, used_d = set(), set()
            for ri, rj in fallback_matches.tolist():
                if not valid[int(ri), int(rj)]:
                    continue
                matches = np.vstack([matches, [old_ut[int(ri)], old_ud[int(rj)]]])
                used_t.add(int(ri)); used_d.add(int(rj))
            self.jepa_stats["high_fallback_pairs"] += len(used_t)
            u_track = tuple(int(x) for k, x in enumerate(old_ut) if k not in used_t)
            u_det = tuple(int(x) for k, x in enumerate(old_ud) if k not in used_d)
        else:
            u_track = tuple(int(x) for x in native_ut)
            u_det = tuple(int(x) for x in native_ud)
        return matches, u_track, u_det

    def _high_jepa_reassign_assignment(self, native_cost, tracks, detections):
        """Repair a native high assignment with a strong JEPA alternative.

        The native assignment and its gate remain the anchor.  JEPA can only
        choose a candidate that the original GeneralTrack cost already
        considered viable.  An originally unassigned candidate displaces the
        native candidate into the normal new-track path; a candidate owned by
        a different row requires a mutually supported two-way swap.  The
        matched count is therefore unchanged and no detector/lifecycle
        threshold is involved in this intervention.
        """
        native_matches, native_ut, native_ud = matching.linear_assignment(
            native_cost, thresh=self.args.match_thresh)
        native_matches = np.asarray(native_matches, dtype=int).reshape(-1, 2)
        if native_cost.size == 0 or len(native_matches) == 0:
            return native_matches, tuple(native_ut), tuple(native_ud)

        sim = self._jepa_similarity(tracks, detections)
        iou_sim = 1.0 - matching.iou_distance(tracks, detections)
        floor = float(getattr(self.args, "jepa_high_min_sim", 0.90))
        min_iou = float(getattr(self.args, "jepa_high_iou", 0.05))
        min_margin = float(getattr(self.args, "jepa_relative_margin", 0.03))
        row_to_det = {int(i): int(j) for i, j in native_matches.tolist()}
        det_to_row = {int(j): int(i) for i, j in native_matches.tolist()}
        native_unmatched = set(int(x) for x in np.asarray(native_ud, dtype=int).tolist())
        original_unmatched = set(native_unmatched)
        proposals = []

        for row, current in sorted(row_to_det.items()):
            gap = max(1, int(self.frame_id) - int(
                getattr(tracks[row], "frame_id", self.frame_id)))
            if gap > self.jepa_max_gap:
                continue
            viable = np.where(np.isfinite(native_cost[row]) &
                              (native_cost[row] <= float(self.args.match_thresh)))[0]
            if viable.size < 2:
                continue
            order = viable[np.argsort(sim[row, viable], kind="mergesort")]
            preferred = int(order[-1])
            if preferred == current:
                continue
            preference = float(sim[row, preferred] - sim[row, current])
            if (float(sim[row, preferred]) < floor or preference < min_margin or
                    float(iou_sim[row, preferred]) < min_iou):
                continue

            owner = det_to_row.get(preferred)
            if owner is None:
                if preferred not in original_unmatched:
                    continue
                proposals.append((-preference, "one_way", row, current, preferred))
                continue
            if owner == row:
                continue
            owner_current = row_to_det.get(owner)
            if owner_current is None or owner_current == preferred:
                continue
            owner_viable = np.where(
                np.isfinite(native_cost[owner]) &
                (native_cost[owner] <= float(self.args.match_thresh)))[0]
            if current not in owner_viable:
                continue
            owner_preference = float(sim[owner, current] - sim[owner, owner_current])
            if (float(sim[owner, current]) < floor or
                    owner_preference < min_margin or
                    float(iou_sim[owner, current]) < min_iou):
                continue
            proposals.append((-(preference + owner_preference), "swap",
                              row, current, preferred))

        used_rows = set()
        for _, kind, row, current, preferred in sorted(proposals):
            if row in used_rows or row_to_det.get(row) != current:
                continue
            owner = det_to_row.get(preferred)
            if kind == "one_way":
                if preferred not in native_unmatched:
                    continue
                row_to_det[row] = preferred
                det_to_row.pop(current, None)
                det_to_row[preferred] = row
                native_unmatched.remove(preferred)
                native_unmatched.add(current)
                used_rows.add(row)
                self.jepa_stats["high_reassign_pairs"] += 1
                continue
            if owner is None or owner in used_rows:
                continue
            owner_current = row_to_det.get(owner)
            if owner_current != preferred:
                continue
            # Recheck the reverse preference in case an earlier proposal
            # changed this row or detection.
            if (float(sim[owner, current]) < floor or
                    float(sim[owner, current] - sim[owner, owner_current]) < min_margin):
                continue
            row_to_det[row], row_to_det[owner] = preferred, current
            det_to_row[preferred], det_to_row[current] = row, owner
            used_rows.update((row, owner))
            self.jepa_stats["high_swap_pairs"] += 1

        matches = np.asarray(sorted(row_to_det.items()), dtype=int).reshape(-1, 2)
        return matches, tuple(int(x) for x in native_ut), tuple(sorted(native_unmatched))

    def _make_detections(self, boxes, scores, classes, features=None, feature_maps=None):
        if len(boxes) == 0:
            return []
        out = []
        for idx, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            det = STrack(STrack.tlbr_to_tlwh(box), score, cls)
            feat = None if features is None else features[idx]
            fmap = None if feature_maps is None else np.asarray(feature_maps[idx], dtype=np.float32)
            if feat is None and fmap is not None and fmap.shape == (256, 27, 27):
                start = (27 - 3) // 2
                feat = fmap[:, start:start + 3, start:start + 3].mean(axis=(1, 2))
            self._attach_detection_feature(det, feat)
            if fmap is not None and fmap.shape == (256, 27, 27) and np.isfinite(fmap).all():
                det.jepa_map = fmap.copy()
            else:
                det.jepa_map = None
            out.append(det)
        return out

    def update(self, output_results, img_info, img_size, imgs, jepa_maps=None):
        """Original ByteTrack update with JEPA only in association costs."""
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            clses = np.ones(len(scores), dtype=np.float32)
            jepa_features = None
        else:
            output_results = output_results.cpu().numpy() if torch.is_tensor(output_results) else np.asarray(output_results)
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
            clses = output_results[:, 5]
            jepa_features = output_results[:, 6:6 + self.feature_dim] if output_results.shape[1] >= 6 + self.feature_dim else None

        if jepa_maps is not None:
            jepa_maps = jepa_maps.detach().cpu().numpy() if torch.is_tensor(jepa_maps) else np.asarray(jepa_maps)
            if jepa_maps.ndim != 4 or jepa_maps.shape[0] != len(scores) or tuple(jepa_maps.shape[1:]) != (256, 27, 27):
                raise ValueError("JEPA map rows must be [N,256,27,27], got {} for N={}".format(
                    tuple(jepa_maps.shape), len(scores)))

        img_h, img_w = img_info[0], img_info[1]
        _ = min(img_size[0] / float(img_h), img_size[1] / float(img_w))

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        cls_keep = clses[remain_inds]
        cls_second = clses[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        feat_keep = jepa_features[remain_inds] if jepa_features is not None else None
        feat_second = jepa_features[inds_second] if jepa_features is not None else None
        map_keep = jepa_maps[remain_inds] if jepa_maps is not None else None
        map_second = jepa_maps[inds_second] if jepa_maps is not None else None

        detections = self._make_detections(dets, scores_keep, cls_keep, feat_keep, map_keep)

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        if self.frame_id == 1:
            self.image_key.append(imgs)
            self.image_key.append(imgs)
        else:
            self.image_key[0] = self.image_key[1]
            self.image_key[1] = imgs

        # Step 2-1: preserve GeneralTrack's learned high-score relation and
        # optionally add a gated JEPA residual for hard/ambiguous rows.
        strack_pool1 = tracked_stracks
        dists1 = matching.classification_distance(
            strack_pool1, detections, self.image_key, _, self.model, img_h, img_w
        )
        STrack.multi_predict(strack_pool1)
        if self.jepa_mode == "high_top2":
            if not self.args.mot20:
                dists1 = matching.fuse_score(dists1, detections)
            matches, u_track1, u_detection = self._top2_assignment(
                dists1, strack_pool1, detections
            )
        elif self.jepa_mode == "high_consensus":
            # Preserve the native high-score assignment and use JEPA only as
            # a source-calibrated veto.  A veto returns both rows to the
            # normal unmatched/new-track path; it never performs a JEPA
            # replacement assignment.
            if not self.args.mot20:
                dists1 = matching.fuse_score(dists1, detections)
            native_matches, native_ut, native_ud = matching.linear_assignment(
                dists1, thresh=self.args.match_thresh
            )
            sim = self._jepa_similarity(strack_pool1, detections)
            min_sim = float(getattr(self.args, "jepa_lost_min_sim", 0.90))
            min_gap = int(getattr(self.args, "jepa_consensus_min_gap", 1))
            accepted = []
            veto_tracks = set()
            veto_dets = set()
            for itracked, idet in native_matches.tolist():
                self.jepa_stats["high_consensus_pairs"] += 1
                if float(sim[int(itracked), int(idet)]) >= min_sim:
                    accepted.append((int(itracked), int(idet)))
                else:
                    veto_tracks.add(int(itracked))
                    veto_dets.add(int(idet))
                    self.jepa_stats["high_consensus_vetoes"] += 1
            matches = np.asarray(accepted, dtype=int).reshape(-1, 2)
            u_track1 = tuple(sorted(set(int(x) for x in native_ut) | veto_tracks))
            u_detection = tuple(sorted(set(int(x) for x in native_ud) | veto_dets))
        elif self.jepa_mode in ("high_relative_consensus", "pair_relative_consensus"):
            # Preserve the native high-score assignment unless a causal JEPA
            # query has a clear relative preference for a local, currently
            # unused candidate.  This is a veto-only safeguard: it never
            # steals a candidate already assigned to another track and never
            # creates a replacement assignment from appearance alone.
            if not self.args.mot20:
                dists1 = matching.fuse_score(dists1, detections)
            native_matches, native_ut, native_ud = matching.linear_assignment(
                dists1, thresh=self.args.match_thresh)
            sim = (self._pair_probability(strack_pool1, detections)
                   if self.jepa_mode == "pair_relative_consensus"
                   else self._jepa_similarity(strack_pool1, detections))
            min_margin = float(getattr(self.args, "jepa_relative_margin", 0.03))
            min_iou = float(getattr(self.args, "jepa_high_iou", 0.05))
            native_det_owner = {int(j): int(i) for i, j in native_matches.tolist()}
            accepted = []
            veto_tracks = set()
            veto_dets = set()
            for itracked, idet in native_matches.tolist():
                itracked, idet = int(itracked), int(idet)
                self.jepa_stats["high_relative_consensus_pairs"] += 1
                viable = np.where(np.isfinite(dists1[itracked]) &
                                  (dists1[itracked] <= float(self.args.match_thresh)))[0]
                gap = max(1, int(self.frame_id) - int(
                    getattr(strack_pool1[itracked], "frame_id", self.frame_id)))
                if viable.size < 2 or gap > self.jepa_max_gap:
                    accepted.append((itracked, idet))
                    continue
                native_order = viable[np.argsort(dists1[itracked, viable], kind="mergesort")]
                native_margin = float(dists1[itracked, native_order[1]] -
                                       dists1[itracked, native_order[0]])
                if native_margin > self.jepa_ambiguity_margin:
                    accepted.append((itracked, idet))
                    continue
                best = int(viable[np.argmax(sim[itracked, viable])])
                preference = float(sim[itracked, best] - sim[itracked, idet])
                local_iou = float(1.0 - matching.iou_distance(
                    [strack_pool1[itracked]], [detections[best]])[0, 0])
                if (best != idet and preference >= min_margin and
                        local_iou >= min_iou and best not in native_det_owner):
                    veto_tracks.add(itracked)
                    veto_dets.add(idet)
                    self.jepa_stats["high_relative_consensus_vetoes"] += 1
                else:
                    accepted.append((itracked, idet))
            matches = np.asarray(accepted, dtype=int).reshape(-1, 2)
            u_track1 = tuple(sorted(set(int(x) for x in native_ut) | veto_tracks))
            u_detection = tuple(sorted(set(int(x) for x in native_ud) | veto_dets))
        elif self.jepa_mode in ("high_fallback_swap", "high_fallback_duplicate_geom_jepa_swap"):
            if not self.args.mot20:
                dists1 = matching.fuse_score(dists1, detections)
            matches, u_track1, u_detection = self._high_fallback_swap_assignment(
                dists1, strack_pool1, detections)
        elif self.jepa_mode in ("high_jepa_reassign", "high_fallback_duplicate_geom_jepa_reassign"):
            if not self.args.mot20:
                dists1 = matching.fuse_score(dists1, detections)
            matches, u_track1, u_detection = self._high_jepa_reassign_assignment(
                dists1, strack_pool1, detections)
        elif self.jepa_mode in ("high_fallback", "high_fallback_pair", "high_fallback_low", "high_fallback_box", "high_fallback_duplicate", "high_fallback_duplicate_geom", "high_fallback_duplicate_geom_jepa", "high_fallback_duplicate_geom_jepa_birthconfirm", "high_fallback_duplicate_geom_jepa_consensus", "high_fallback_duplicate_geom_jepa_lostfallback", "high_fallback_duplicate_geom_jepa_box", "high_fallback_duplicate_geom_jepa_swap", "high_fallback_duplicate_geom_jepa_reassign", "high_fallback_duplicate_geom_jepa_unique", "high_fallback_duplicate_geom_jepa_expand", "high_fallback_duplicate_geom_jepa_iou_tiebreak", "high_fallback_birth_confirm", "high_fallback_lost_consensus", "high_fallback_lost_fallback", "high_fallback_dormant"):
            if not self.args.mot20:
                dists1 = matching.fuse_score(dists1, detections)
            native_matches, native_ut, native_ud = matching.linear_assignment(
                dists1, thresh=self.args.match_thresh)
            fallback_ut = tuple(int(x) for x in native_ut)
            fallback_ud = tuple(int(x) for x in native_ud)
            if self.jepa_mode == "high_fallback_duplicate_geom_jepa_consensus":
                # Keep native GeneralTrack as the anchor.  Reopen only a
                # native pair whose causal JEPA identity is below the same
                # fixed floor used by the residual fallback.  Missing state
                # remains native rather than becoming an artificial veto.
                native_sim = self._jepa_similarity(strack_pool1, detections)
                min_sim = float(getattr(self.args, "jepa_high_min_sim", 0.90))
                accepted = []
                veto_tracks = set()
                veto_dets = set()
                for itracked, idet in native_matches.tolist():
                    itracked, idet = int(itracked), int(idet)
                    self.jepa_stats["high_consensus_pairs"] += 1
                    query = self._track_query(strack_pool1[itracked])
                    feature = getattr(detections[idet], "jepa_feature", None)
                    if (query is None or feature is None or
                            float(native_sim[itracked, idet]) >= min_sim):
                        accepted.append((itracked, idet))
                    else:
                        veto_tracks.add(itracked)
                        veto_dets.add(idet)
                        self.jepa_stats["high_consensus_vetoes"] += 1
                matches = np.asarray(accepted, dtype=int).reshape(-1, 2)
                fallback_ut = tuple(sorted(set(fallback_ut) | veto_tracks))
                fallback_ud = tuple(sorted(set(fallback_ud) | veto_dets))
            else:
                matches = np.asarray(native_matches, dtype=int).reshape(-1, 2)
            old_ut = np.asarray(fallback_ut, dtype=int)
            old_ud = np.asarray(fallback_ud, dtype=int)
            if old_ut.size and old_ud.size:
                rem_tracks = [strack_pool1[int(i)] for i in old_ut.tolist()]
                rem_dets = [detections[int(i)] for i in old_ud.tolist()]
                sim = (self._pair_probability(rem_tracks, rem_dets)
                       if self.jepa_mode == "high_fallback_pair"
                       else self._jepa_similarity(rem_tracks, rem_dets))
                iou_sim = 1.0 - matching.iou_distance(rem_tracks, rem_dets)
                min_sim = (float(self.pair_adapter.threshold)
                           if self.jepa_mode == "high_fallback_pair"
                           else float(getattr(self.args, "jepa_high_min_sim", 0.90)))
                min_iou = float(getattr(self.args, "jepa_high_iou", 0.20))
                valid = (sim >= min_sim) & (iou_sim >= min_iou)
                for ri, track in enumerate(rem_tracks):
                    gap = max(1, int(self.frame_id) - int(
                        getattr(track, "frame_id", self.frame_id)))
                    if gap > self.jepa_max_gap:
                        valid[ri, :] = False
                    if np.any(valid[ri]):
                        vals = np.sort(sim[ri, valid[ri]])[::-1]
                        if vals.size >= 2 and float(vals[0] - vals[1]) < self.jepa_relative_margin:
                            valid[ri, :] = False
                            self.jepa_stats["high_fallback_rejections"] += 1
                if self.jepa_mode == "high_fallback_duplicate_geom_jepa_iou_tiebreak":
                    # JEPA remains the acceptance signal and primary score;
                    # IoU is only a deterministic tie-break among accepted
                    # candidates, so this branch cannot create a match that
                    # the JEPA floor rejected.
                    fallback_cost = 1.0 - (sim + 0.05 * np.clip(iou_sim, 0.0, 1.0))
                else:
                    fallback_cost = 1.0 - sim
                fallback_cost[~valid] = 1e6
                fallback_matches, _, _ = matching.linear_assignment(
                    fallback_cost, thresh=1.0 - min_sim)
                used_t, used_d = set(), set()
                for ri, rj in fallback_matches.tolist():
                    if not valid[int(ri), int(rj)]:
                        continue
                    matches = np.vstack([matches, [old_ut[int(ri)], old_ud[int(rj)]]])
                    used_t.add(int(ri)); used_d.add(int(rj))
                self.jepa_stats["high_fallback_pairs"] += len(used_t)
                u_track1 = tuple(int(x) for k, x in enumerate(old_ut) if k not in used_t)
                u_detection = tuple(int(x) for k, x in enumerate(old_ud) if k not in used_d)
            else:
                u_track1 = fallback_ut
                u_detection = fallback_ud
        elif self.jepa_mode in ("pair_high", "pair_high_ambiguous"):
            # Keep the native GeneralTrack assignment scale and use the
            # source-trained pair adapter only as a positive residual.
            if not self.args.mot20:
                dists1 = matching.fuse_score(dists1, detections)
            dists = self._fuse_pair_probability(dists1, strack_pool1, detections)
            matches, u_track1, u_detection = matching.linear_assignment(
                dists, thresh=self.args.match_thresh)
        elif self.jepa_mode == "high_jepa_iou":
            dists = self._fuse_jepa_iou_primary(dists1, strack_pool1, detections)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            matches, u_track1, u_detection = matching.linear_assignment(
                dists, thresh=self.args.match_thresh)
        elif self.jepa_mode == "birth_confirm":
            # Birth confirmation is deliberately orthogonal to association:
            # it only changes which newly born tracks are emitted.
            if not self.args.mot20:
                dists1 = matching.fuse_score(dists1, detections)
            matches, u_track1, u_detection = matching.linear_assignment(
                dists1, thresh=self.args.match_thresh)
        else:
            dists1 = self._fuse_jepa(dists1, strack_pool1, detections, "high")
            if not self.args.mot20:
                dists = matching.fuse_score(dists1, detections)
            else:
                dists = dists1
            matches, u_track1, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track, det = strack_pool1[itracked], detections[idet]
            track.low = 0
            self._confirm_birth(track, det)
            update_det, commit_feature = self._jepa_box_update(track, det)
            track.update(update_det, self.frame_id)
            self._commit_feature(track, det, allow=commit_feature)
            activated_starcks.append(track)

        # Step 2-2: lost-track reactivation.  The original branch is IoU;
        # JEPA is fused here first, so detector-score fusion and assignment
        # thresholds remain exactly those of GeneralTrack.
        strack_pool2 = self.lost_stracks
        STrack.multi_predict(strack_pool2)
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(strack_pool2, detections)
        if self.jepa_mode == "lost_top2":
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            matches, u_track2, u_detection = self._top2_assignment(
                dists, strack_pool2, detections
            )
        elif self.jepa_mode == "lost_relative_swap":
            # Native lost association is the anchor.  JEPA may exchange a
            # native detection for an otherwise-unassigned candidate only
            # when its within-row preference is clear and the predicted-box
            # neighbourhood remains plausible.  This preserves lifecycle
            # and match count while allowing appearance to repair a wrong
            # native choice.
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            native_matches, native_ut, native_ud = matching.linear_assignment(
                dists, thresh=self.args.match_thresh)
            sim = self._jepa_similarity(strack_pool2, detections)
            min_margin = float(getattr(self.args, "jepa_relative_margin", 0.03))
            min_iou = float(getattr(self.args, "jepa_lost_iou", 0.10))
            row_to_det = {int(i): int(j) for i, j in native_matches.tolist()}
            det_to_row = {int(j): int(i) for i, j in native_matches.tolist()}
            proposals = []
            for itracked, idet in native_matches.tolist():
                itracked, idet = int(itracked), int(idet)
                gap = max(1, int(self.frame_id) - int(
                    getattr(strack_pool2[itracked], "frame_id", self.frame_id)))
                if gap > self.jepa_max_gap:
                    continue
                viable = np.where(np.isfinite(dists[itracked]) &
                                  (dists[itracked] <= float(self.args.match_thresh)))[0]
                if viable.size < 2:
                    continue
                best = int(viable[np.argmax(sim[itracked, viable])])
                preference = float(sim[itracked, best] - sim[itracked, idet])
                local_iou = float(1.0 - matching.iou_distance(
                    [strack_pool2[itracked]], [detections[best]])[0, 0])
                if best == idet or preference < min_margin or local_iou < min_iou:
                    continue
                if best in det_to_row:
                    continue
                self.jepa_stats["lost_relative_rows"] += 1
                proposals.append((-preference, itracked, idet, best))
            proposals.sort()
            for _, itracked, current, preferred in proposals:
                if row_to_det.get(itracked) != current or preferred in det_to_row:
                    continue
                row_to_det[itracked] = preferred
                det_to_row.pop(current, None)
                det_to_row[preferred] = itracked
                self.jepa_stats["lost_relative_switches"] += 1
            matches = np.asarray(sorted(row_to_det.items()), dtype=int).reshape(-1, 2)
            matched_dets = {int(j) for _, j in matches.tolist()}
            old_matched = {int(j) for _, j in native_matches.tolist()}
            u_track2 = tuple(int(x) for x in native_ut)
            u_detection = tuple(sorted((set(int(x) for x in native_ud) - matched_dets) |
                                       (old_matched - matched_dets)))
        elif self.jepa_mode in ("lost_consensus", "lost_consensus_score",
                                 "high_fallback_lost_consensus",
                                 "high_selective_lost_consensus",
                                 "high_selective_spatial_lost_consensus"):
            # Keep the native IoU/score assignment and use JEPA only as a
            # conservative veto.  A low-confidence appearance match is left
            # unmatched so the normal new-track path can handle the row; JEPA
            # never creates a distant replacement assignment here.
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            native_matches, native_ut, native_ud = matching.linear_assignment(
                dists, thresh=self.args.match_thresh
            )
            jepa_sim = self._jepa_similarity(strack_pool2, detections)
            min_sim = float(getattr(self.args, "jepa_lost_min_sim", 0.90))
            min_gap = int(getattr(self.args, "jepa_consensus_min_gap", 1))
            accepted = []
            veto_tracks = set()
            veto_dets = set()
            for itracked, idet in native_matches.tolist():
                self.jepa_stats["lost_consensus_pairs"] += 1
                gap = max(1, int(self.frame_id) - int(getattr(strack_pool2[int(itracked)], "frame_id", self.frame_id)))
                if gap < min_gap:
                    accepted.append((int(itracked), int(idet)))
                    self.jepa_stats["consensus_gap_skips"] += 1
                elif (self.jepa_mode == "lost_consensus_score" and
                      self.jepa_veto_max_score is not None and
                      float(detections[int(idet)].score) > self.jepa_veto_max_score):
                    accepted.append((int(itracked), int(idet)))
                    self.jepa_stats["lost_score_gate_skips"] += 1
                elif float(jepa_sim[int(itracked), int(idet)]) >= min_sim:
                    accepted.append((int(itracked), int(idet)))
                else:
                    veto_tracks.add(int(itracked))
                    veto_dets.add(int(idet))
                    self.jepa_stats["lost_consensus_vetoes"] += 1
            matches = np.asarray(accepted, dtype=int).reshape(-1, 2)
            u_track2 = tuple(sorted(set(int(x) for x in native_ut) | veto_tracks))
            u_detection = tuple(sorted(set(int(x) for x in native_ud) | veto_dets))
        elif self.jepa_mode in ("lost_fallback", "high_fallback_lost_fallback", "high_fallback_duplicate_geom_jepa_lostfallback"):
            # First preserve the native GeneralTrack lost assignment.  JEPA
            # may only recover an otherwise-unmatched lost track/detection
            # pair, and only under source-calibrated appearance, spatial, and
            # per-track margin gates.  This prevents JEPA from stealing an
            # already plausible native match.
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            native_matches, native_ut, native_ud = matching.linear_assignment(
                dists, thresh=self.args.match_thresh
            )
            matches = np.asarray(native_matches, dtype=int).reshape(-1, 2)
            old_ut = np.asarray(native_ut, dtype=int)
            old_ud = np.asarray(native_ud, dtype=int)
            if old_ut.size and old_ud.size:
                rem_tracks = [strack_pool2[int(i)] for i in old_ut.tolist()]
                rem_dets = [detections[int(i)] for i in old_ud.tolist()]
                sim = self._jepa_similarity(rem_tracks, rem_dets)
                iou = 1.0 - matching.iou_distance(rem_tracks, rem_dets)
                min_sim = float(getattr(self.args, "jepa_lost_min_sim", 0.90))
                min_iou = float(getattr(self.args, "jepa_fallback_iou", 0.05))
                margin = float(getattr(self.args, "jepa_fallback_margin", 0.02))
                valid = (sim >= min_sim) & (iou >= min_iou)
                for ri, track in enumerate(rem_tracks):
                    gap = max(1, int(self.frame_id) - int(getattr(track, "frame_id", self.frame_id)))
                    if gap > self.jepa_max_gap:
                        valid[ri, :] = False
                    if np.any(valid[ri]):
                        vals = np.sort(sim[ri, valid[ri]])[::-1]
                        if vals.size >= 2 and float(vals[0] - vals[1]) < margin:
                            valid[ri, :] = False
                            self.jepa_stats["lost_fallback_rejections"] += 1
                fallback_cost = 1.0 - sim
                fallback_cost[~valid] = 1e6
                fallback_matches, _, _ = matching.linear_assignment(
                    fallback_cost, thresh=1.0 - min_sim
                )
                used_t, used_d = set(), set()
                for ri, rj in fallback_matches.tolist():
                    if not valid[int(ri), int(rj)]:
                        continue
                    matches = np.vstack([matches, [old_ut[int(ri)], old_ud[int(rj)]]])
                    used_t.add(int(ri))
                    used_d.add(int(rj))
                self.jepa_stats["lost_fallback_pairs"] += len(used_t)
                u_track2 = tuple(int(x) for k, x in enumerate(old_ut) if k not in used_t)
                u_detection = tuple(int(x) for k, x in enumerate(old_ud) if k not in used_d)
            else:
                u_track2 = tuple(int(x) for x in native_ut)
                u_detection = tuple(int(x) for x in native_ud)
        elif self.jepa_mode in ("high_fallback", "high_fallback_pair", "high_fallback_low", "high_fallback_box", "high_fallback_duplicate", "high_fallback_duplicate_geom", "high_fallback_duplicate_geom_jepa", "high_fallback_duplicate_geom_jepa_birthconfirm", "high_fallback_duplicate_geom_jepa_consensus", "high_fallback_duplicate_geom_jepa_lostfallback", "high_fallback_duplicate_geom_jepa_box", "high_fallback_duplicate_geom_jepa_swap", "high_fallback_duplicate_geom_jepa_reassign", "high_fallback_duplicate_geom_jepa_unique", "high_fallback_duplicate_geom_jepa_expand", "high_fallback_swap", "high_jepa_reassign", "high_fallback_birth_confirm", "high_fallback_lost_fallback", "high_fallback_dormant", "high_consensus", "high_selective", "high_relative",
                                 "high_relative_consensus", "pair_relative_consensus",
                                 "high_selective_spatial", "birth_confirm", "high_jepa_iou",
                                 "pair_high", "pair_high_ambiguous"):
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            matches, u_track2, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        else:
            lost_branch = "lost_iou_gate" if self.jepa_mode == "lost_iou_gate" else "lost"
            dists = self._fuse_jepa(dists, strack_pool2, detections, lost_branch)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections)
            matches, u_track2, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track, det = strack_pool2[itracked], detections[idet]
            track.low = 0
            self._confirm_birth(track, det)
            track.re_activate(det, self.frame_id, new_id=False)
            self._commit_feature(track, det, allow=True)
            refind_stracks.append(track)

        # Step 3: unchanged low-score association; low detections are not
        # committed to JEPA memory to prevent contaminated updates.
        if len(dets_second) > 0:
            detections_second = self._make_detections(dets_second, scores_second, cls_second, feat_second, map_second)
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool1[i] for i in u_track1 if strack_pool1[i].state == TrackState.Tracked]
        dists1 = matching.classification_distance(
            r_tracked_stracks, detections_second, self.image_key, _, self.model, img_h, img_w
        )
        matches, u_track3, u_detection_second = matching.linear_assignment(dists1, thresh=0.01)

        # Optional JEPA fallback for low-score detections that the native
        # relation head rejected.  It is gated by both center-pooled
        # appearance and predicted-box IoU, so this branch can recover only a
        # local existing track and cannot create an extra output.
        if self.jepa_mode in ("low_fallback", "high_fallback_low") and r_tracked_stracks and detections_second:
            rem_tracks = [r_tracked_stracks[int(i)] for i in u_track3]
            rem_dets = [detections_second[int(i)] for i in u_detection_second]
            if rem_tracks and rem_dets:
                rem_sim = self._jepa_similarity(rem_tracks, rem_dets)
                rem_iou = 1.0 - matching.iou_distance(rem_tracks, rem_dets)
                low_sim = float(getattr(self.args, "jepa_low_sim", 0.85))
                low_iou = float(getattr(self.args, "jepa_low_iou", 0.05))
                fallback_cost = 1.0 - rem_sim
                fallback_cost[(rem_sim < low_sim) | (rem_iou < low_iou)] = 1e6
                low_matches, _, _ = matching.linear_assignment(fallback_cost, thresh=1.0 - low_sim)
                old_u_tracks = np.asarray(u_track3, dtype=int)
                old_u_dets = np.asarray(u_detection_second, dtype=int)
                used_t, used_d = set(), set()
                for li, lj in low_matches.tolist():
                    matches = np.vstack([matches, [old_u_tracks[li], old_u_dets[lj]]]) if len(matches) else np.asarray([[old_u_tracks[li], old_u_dets[lj]]], dtype=int)
                    used_t.add(int(li)); used_d.add(int(lj))
                u_track3 = tuple(int(x) for k, x in enumerate(old_u_tracks) if k not in used_t)
                u_detection_second = tuple(int(x) for k, x in enumerate(old_u_dets) if k not in used_d)
                self.jepa_stats["low_fallback_pairs"] += len(low_matches)
        for itracked, idet in matches:
            track, det = r_tracked_stracks[itracked], detections_second[idet]
            track.low = 1
            self._confirm_birth(track, det)
            track.update(det, self.frame_id)
            self._commit_feature(track, det, allow=False)
            activated_starcks.append(track)
        for it in u_track3:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Unconfirmed branch is byte-for-byte equivalent in decision logic.
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            track, det = unconfirmed[itracked], detections[idet]
            self._confirm_birth(track, det)
            track.update(det, self.frame_id)
            self._commit_feature(track, det, allow=True)
            activated_starcks.append(track)
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 4: new tracks.
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            if self._jepa_duplicate(track, activated_starcks):
                continue
            dormant_id = self._claim_dormant_id(track)
            track.activate(self.kalman_filter, self.frame_id)
            if dormant_id is not None:
                track.track_id = dormant_id
            track.jepa_history = deque(maxlen=max(1, self.jepa_history))
            self._initialize_birth_state(track)
            self._commit_feature(track, track, allow=True)
            activated_starcks.append(track)

        # Step 5: unchanged lifecycle expiry and list maintenance.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                self._archive_dormant(track)
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        return [track for track in self.tracked_stracks
                if track.is_activated and self._output_allowed(track)]
