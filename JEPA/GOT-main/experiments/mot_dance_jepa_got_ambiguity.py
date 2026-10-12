#!/usr/bin/env python3
"""Conservative ambiguity-only fusion on the frozen GOT-JEPA representation.

This experiment is intentionally separate from :mod:`mot_dance_jepa_got`.
The public GOT-JEPA checkpoint is *not* re-trained or changed here; the
already extracted 256-D pooled vectors are loaded from their immutable cache.
The native/raw association cost remains the primary decision.  A history-only
JEPA predictor is allowed to resolve an event only when (a) the native row's
top-1/top-2 margin is in a threshold fixed from the training suffix (p25),
(b) both native candidates are otherwise admissible, and (c) predictor
quality/uncertainty gates pass.  The predictor is evaluated once per track,
before any current candidate is inspected, so the tie-break is
candidate-independent.

The runner writes ``train_only_diagnostic.json`` before it reads validation
annotations, then tracks all validation sequences without GT and invokes the
vendored official TrackEval audit.  No post-hoc track repair is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# The vendored motmetrics/TrackEval code predates NumPy 1.24.  Keep aliases
# process-local; no dependency or production checkout is edited.
for _name, _value in {"float": float, "int": int, "bool": bool, "object": object, "str": str}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mot_dance_jepa_got as got  # noqa: E402
import mot_dance_jepa_real as dance  # noqa: E402


CHECKPOINT_DEFAULT = "/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar"
CACHE_DEFAULT = "/data2/usr_for_deadline/mot_dance_jepa_got_120/feature_cache"
TRAIN_DEFAULT = "dancetrack0001,dancetrack0002,dancetrack0006"
EVAL_DEFAULT = "dancetrack0004,dancetrack0007,dancetrack0014"
TRACKEVAL_DEFAULT = "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/TrackEval-official"

# These values are part of the pre-registration for this bounded run.  They
# are deliberately not command-line knobs, preventing an eval-driven sweep.
METHODS = ("raw", "ambiguity_jepa", "no_teacher", "random_frozen")
PREDICTOR_METHODS = ("ambiguity_jepa", "no_teacher", "random_frozen")
MAX_ASSIGN_COST = 0.82
DETECTION_THRESHOLD = 0.10
QUALITY_THRESHOLD = 0.35
UNCERTAINTY_PROB_THRESHOLD = 0.65
TRAIN_SUFFIX_FRACTION = 0.50
MARGIN_QUANTILE = 25.0
HISTORY_K = 4


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def hash_json(value: object) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), default=float).encode()
    return hashlib.sha256(blob).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def list_names(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -40.0, 40.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def _records_features(records, z, expected_dim: int = 256) -> Dict[Tuple[int, int], np.ndarray]:
    arr = np.asarray(z, dtype=np.float32)
    if arr.shape != (len(records), expected_dim) or not np.isfinite(arr).all():
        raise RuntimeError("invalid pooled cache array {} expected ({}, {})".format(arr.shape, len(records), expected_dim))
    arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
    return {(int(fr), int(key)): arr[i] for i, (fr, key, _, _) in enumerate(records)}


def load_cached_features(
    cache_root: Path,
    cache_name: str,
    records,
    sequence: str,
    role: str,
    max_frames: int,
    checkpoint_sha: str,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[str, object]]:
    """Load and strictly validate one immutable pooled GOT-JEPA cache."""
    path = cache_root / (cache_name + ".npz")
    meta_path = path.with_suffix(".json")
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError("pooled cache or sidecar missing: {}".format(path))
    meta = json.loads(meta_path.read_text())
    expected_record_sha = got.hash_records(records)
    checks = {
        "schema": meta.get("schema") == "got-jepa-pooled-cache-v1",
        "sequence": meta.get("sequence") == sequence,
        "role": meta.get("role") == role,
        "max_frames": int(meta.get("max_frames", -1)) == int(max_frames),
        "record_sha256": meta.get("record_sha256") == expected_record_sha,
        "checkpoint_sha256": meta.get("checkpoint_sha256") == checkpoint_sha,
        "source_feature_shape": meta.get("source_feature_shape") == [256, 27, 27],
        "pooled_dim": int(meta.get("pooled_dim", -1)) == 256,
    }
    if not all(checks.values()):
        raise RuntimeError("cache provenance mismatch {}: {}".format(path, checks))
    with np.load(path, allow_pickle=False) as z:
        if "features" not in z or "frames" not in z or "keys" not in z:
            raise RuntimeError("cache schema missing arrays: {}".format(path))
        if tuple(z["frames"].shape) != (len(records),) or tuple(z["keys"].shape) != (len(records),):
            raise RuntimeError("cache index shape mismatch {}".format(path))
        # Verify index ordering as well as the sidecar hash.  This catches a
        # cache copied from a different D-CTRL stream even if row counts match.
        fr = np.asarray(z["frames"], dtype=np.int64)
        keys = np.asarray(z["keys"], dtype=np.int64)
        exp_fr = np.asarray([r[0] for r in records], dtype=np.int64)
        exp_keys = np.asarray([r[1] for r in records], dtype=np.int64)
        if not np.array_equal(fr, exp_fr) or not np.array_equal(keys, exp_keys):
            raise RuntimeError("cache frame/key ordering mismatch {}".format(path))
        feats = _records_features(records, z["features"])
    observed_sha = sha256(path)
    if meta.get("cache_sha256") not in (None, observed_sha):
        raise RuntimeError("cache file hash mismatch {}".format(path))
    public = dict(meta)
    public.update({"cache": str(path), "cache_sha256_observed": observed_sha,
                   "cache_hit": True, "validation_checks": checks})
    return feats, public


def attach_detector_features(det: Dict[int, List[dance.Det]], feats: Dict[Tuple[int, int], np.ndarray]) -> None:
    for fr, rows in det.items():
        for j, d in enumerate(rows):
            f = feats.get((int(fr), int(j)))
            if f is None or f.shape != (256,) or not np.isfinite(f).all():
                raise RuntimeError("missing/invalid detector pooled feature {}:{}".format(fr, j))
            d.feat = got.norm(f)


def load_sequence_detector(seq: str, split: str, max_frames: int, cache_root: Path, checkpoint_sha: str):
    seq_dir = dance.DANCE_ROOT / split / seq
    det = dance.parse_frame_det_cache(dance.DANCE_CTRL_CACHE / seq, max_frames)
    records = got.detector_records(det)
    feats, meta = load_cached_features(cache_root, seq + "_detector", records, seq, "detector", max_frames, checkpoint_sha)
    attach_detector_features(det, feats)
    return seq_dir, det, meta


def load_train_item(seq: str, max_frames: int, cache_root: Path, checkpoint_sha: str, train_gt):
    seq_dir, det, dm = load_sequence_detector(seq, "train", max_frames, cache_root, checkpoint_sha)
    records = got.gt_records(train_gt[seq])
    teacher, tm = load_cached_features(cache_root, seq + "_teacher", records, seq, "clean_train_teacher", max_frames, checkpoint_sha)
    return (seq, seq_dir, train_gt[seq], det, teacher), {"detector": dm, "teacher": tm}


def native_cost_matrix(tracker: dance.SimpleMOT, frame: int, dets: Sequence[dance.Det]):
    """Reproduce the raw SimpleMOT cost matrix without changing tracker state."""
    active = [d for d in dets if d.score >= tracker.det_thr and d.feat is not None]
    n, m = len(tracker.tracks), len(active)
    cost = np.full((n, m), 1e6, dtype=np.float32)
    pred_boxes: List[np.ndarray] = []
    for tr in tracker.tracks:
        pb = tr.box.copy()
        pb[[0, 2]] += tr.velocity[0]
        pb[[1, 3]] += tr.velocity[1]
        pred_boxes.append(pb)
    for i, tr in enumerate(tracker.tracks):
        pb = pred_boxes[i]
        for j, d in enumerate(active):
            iou = dance.box_iou(pb, d.box)
            app = 1.0 - dance.cosine(tr.feat, d.feat)
            dcx = ((d.box[0] + d.box[2]) - (pb[0] + pb[2])) / 2.0
            dcy = ((d.box[1] + d.box[3]) - (pb[1] + pb[3])) / 2.0
            scale = max(10.0, math.sqrt(max(1.0, (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))))
            dist = math.sqrt(float(dcx * dcx + dcy * dcy)) / scale
            if dist > 8.0 and iou < 0.01:
                continue
            value = 0.58 * app + 0.32 * (1.0 - iou) + 0.10 * min(dist / 4.0, 1.0)
            cost[i, j] = float(value) if np.isfinite(value) else 1e6
    return active, cost, pred_boxes


def predictor_cost(
    tracker: dance.SimpleMOT,
    tr: dance.Track,
    pred_tuple: Tuple[np.ndarray, float, float],
    d: dance.Det,
    pred_box: np.ndarray,
) -> float:
    """Candidate cost using a predictor already computed from history only."""
    proj, quality, uncertainty = pred_tuple
    cand = d.feat if d.feat is not None else np.zeros_like(tr.feat)
    hist = tracker._hist(tr)
    up = sigmoid(float(uncertainty))
    app = float(quality) * (1.0 - dance.cosine(cand, proj)) + (1.0 - float(quality)) * (1.0 - dance.cosine(cand, hist)) + 0.05 * up
    iou = dance.box_iou(pred_box, d.box)
    dcx = ((d.box[0] + d.box[2]) - (pred_box[0] + pred_box[2])) / 2.0
    dcy = ((d.box[1] + d.box[3]) - (pred_box[1] + pred_box[3])) / 2.0
    scale = max(10.0, math.sqrt(max(1.0, (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))))
    dist = math.sqrt(float(dcx * dcx + dcy * dcy)) / scale
    if dist > 8.0 and iou < 0.01:
        return 1e6
    value = 0.58 * app + 0.32 * (1.0 - iou) + 0.10 * min(dist / 4.0, 1.0)
    return float(value) if np.isfinite(value) else 1e6


def gate_for(tr: dance.Track, pred_tuple: Tuple[np.ndarray, float, float]) -> Tuple[bool, float, float]:
    _, q, u = pred_tuple
    up = sigmoid(float(u))
    # Every term is track/history-only.  In particular, no current candidate
    # score is consulted by this gate.
    ok = bool(float(q) >= QUALITY_THRESHOLD and up <= UNCERTAINTY_PROB_THRESHOLD and int(tr.misses) <= 4)
    return ok, float(q), float(up)


class AmbiguityMOT(dance.SimpleMOT):
    """Raw lifecycle with a strictly gated, local top-two tie-break."""

    def __init__(self, predictor: Optional[dance.TemporalPredictor], method: str,
                 margin_threshold: float, seed: int, image_width: float, image_height: float):
        # Set the inherited method to raw so all memory/lifecycle updates are
        # exactly the native baseline.  ``variant`` only controls the tie.
        super().__init__("raw", None, max_age=30, det_thr=DETECTION_THRESHOLD,
                         seed=seed, image_width=image_width, image_height=image_height)
        self.predictor = predictor
        self.variant = method
        self.margin_threshold = float(margin_threshold)
        self.native_assignment_pairs: Dict[int, List[Tuple[int, int]]] = {}
        self.used_assignment_pairs: Dict[int, List[Tuple[int, int]]] = {}
        self.event_rows: List[Dict[str, object]] = []
        self.tie_eligible_events = 0
        self.tie_applied_events = 0
        self.assignment_changed_rows = 0
        self.assignment_changed_outside_tie = 0
        self.native_match_count = 0
        self.used_match_count = 0

    def _reset_audit(self) -> None:
        self.native_assignment_pairs = {}
        self.used_assignment_pairs = {}
        self.event_rows = []
        self.tie_eligible_events = 0
        self.tie_applied_events = 0
        self.assignment_changed_rows = 0
        self.assignment_changed_outside_tie = 0
        self.native_match_count = 0
        self.used_match_count = 0

    def step(self, frame: int, dets: Sequence[dance.Det]) -> None:
        active, native, pred_boxes = native_cost_matrix(self, frame, dets)
        n, m = native.shape
        pred_cache: Dict[int, Tuple[np.ndarray, float, float]] = {}
        gate_cache: Dict[int, Tuple[bool, float, float]] = {}
        jcost = np.full((n, m), 1e6, dtype=np.float32)

        # Compute each predictor output before entering the candidate loop.
        # This is the candidate-order independence invariant.
        if self.predictor is not None:
            for i, tr in enumerate(self.tracks):
                p = self.predictor.predict(self._temporal_tokens(tr, frame))
                pred_cache[i] = p
                gate_cache[i] = gate_for(tr, p)
                self.quality_values.append(float(p[1]))
                self.uncertainty_values.append(float(p[2]))
                self.temporal_prediction_count += 1
                for j, d in enumerate(active):
                    jcost[i, j] = predictor_cost(self, tr, p, d, pred_boxes[i])

        # Native Hungarian assignment is always computed first.
        native_matches: List[Tuple[int, int]] = []
        if n and m:
            rr, cc = linear_sum_assignment(native)
            for i, j in zip(rr.tolist(), cc.tolist()):
                if native[i, j] < MAX_ASSIGN_COST:
                    native_matches.append((i, j))
        native_by_row = {i: j for i, j in native_matches}
        native_by_det = {j: i for i, j in native_matches}
        matches = list(native_matches)
        matched_by_row = dict(native_by_row)
        matched_by_det = dict(native_by_det)
        tie_rows: Dict[int, Dict[str, object]] = {}

        # Determine the registered ambiguity set for every track row.
        for i in range(n):
            finite = np.where(np.isfinite(native[i]) & (native[i] < 1e5))[0]
            if finite.size < 2:
                continue
            order = finite[np.argsort(native[i, finite], kind="mergesort")]
            j1, j2 = int(order[0]), int(order[1])
            margin = float(native[i, j2] - native[i, j1])
            ambiguous = bool(margin <= self.margin_threshold)
            gate_ok, q, up = gate_cache.get(i, (False, 0.0, 1.0))
            both_viable = bool(native[i, j1] < MAX_ASSIGN_COST and native[i, j2] < MAX_ASSIGN_COST)
            pref = j1
            jdelta = None
            if i in pred_cache and np.isfinite(jcost[i, j1]) and np.isfinite(jcost[i, j2]):
                jdelta = float(jcost[i, j1] - jcost[i, j2])
                if jcost[i, j2] + 1e-7 < jcost[i, j1]:
                    pref = j2
            eligible = bool(self.predictor is not None and ambiguous and gate_ok and both_viable and pref != j1)
            if eligible:
                self.tie_eligible_events += 1
                tie_rows[i] = {"top1": j1, "top2": j2, "preferred": pref}
            # Store a compact event trace; it is useful for auditing without
            # exposing image/GT content.
            self.event_rows.append({
                "frame": int(frame), "track_index": int(i),
                "track_id": int(self.tracks[i].tid), "top1": j1, "top2": j2,
                "native_top1": float(native[i, j1]), "native_top2": float(native[i, j2]),
                "native_margin": margin, "ambiguous": ambiguous,
                "both_viable": both_viable, "quality": q, "uncertainty_prob": up,
                "gate_pass": bool(gate_ok), "predictor_preferred": int(pref),
                "predictor_delta_top1_minus_top2": jdelta, "tie_eligible": eligible,
            })

        # Resolve only already matched rows and only with an unused/top-two
        # candidate.  This conservative local operation preserves the number
        # of native matches and cannot create a new track or repair a miss.
        # Processing largest predictor gain first makes conflicts deterministic.
        proposals = []
        for i, info in tie_rows.items():
            cur = matched_by_row.get(i)
            pref = int(info["preferred"])
            if cur is None or pref == cur:
                continue
            gain = float(jcost[i, cur] - jcost[i, pref]) if jcost.shape[1] else 0.0
            proposals.append((-gain, int(i), int(pref), int(cur)))
        proposals.sort()
        for _, i, pref, cur in proposals:
            # State may have changed after an earlier proposal.
            if matched_by_row.get(i) != cur:
                continue
            if native[i, pref] >= MAX_ASSIGN_COST or not np.isfinite(jcost[i, pref]):
                continue
            owner = matched_by_det.get(pref)
            if owner is None:
                # The preferred candidate is free; replacing the native top-2
                # choice is exactly the registered tie-break operation.
                matched_by_det.pop(cur, None)
                matched_by_det[pref] = i
                matched_by_row[i] = pref
                self.tie_applied_events += 1
            else:
                # A two-way swap is allowed only when the other row is itself
                # registered ambiguous and its current candidate is in its
                # native top-two set.  This retains one-to-one assignments and
                # keeps all changes within the predeclared ambiguity budget.
                other_info = tie_rows.get(owner)
                other_cur = matched_by_row.get(owner)
                if other_info is None or other_cur is None:
                    continue
                other_top = {int(other_info["top1"]), int(other_info["top2"])}
                if cur not in other_top or not np.isfinite(jcost[owner, cur]):
                    continue
                # Find the candidate currently held by i and require it to be
                # viable for the other row before swapping.
                if native[owner, cur] >= MAX_ASSIGN_COST or not np.isfinite(jcost[owner, cur]):
                    continue
                before = float(jcost[i, cur] + jcost[owner, pref])
                after = float(jcost[i, pref] + jcost[owner, cur])
                if after + 1e-7 >= before:
                    continue
                matched_by_row[i], matched_by_row[owner] = pref, cur
                matched_by_det[pref], matched_by_det[cur] = i, owner
                self.tie_applied_events += 1

        matches = sorted((int(i), int(j)) for i, j in matched_by_row.items())
        self.native_assignment_pairs.setdefault(int(frame), []).extend((int(self.tracks[i].tid), int(j)) for i, j in native_matches)
        self.used_assignment_pairs.setdefault(int(frame), []).extend((int(self.tracks[i].tid), int(j)) for i, j in matches)
        self.native_match_count += len(native_matches)
        self.used_match_count += len(matches)
        native_map = {i: j for i, j in native_matches}
        for i, j in matches:
            if native_map.get(i) != j:
                self.assignment_changed_rows += 1
                if i not in tie_rows:
                    self.assignment_changed_outside_tie += 1

        # The following lifecycle is copied from the raw SimpleMOT path.  It
        # deliberately does not use predictor output, ensuring that this
        # experiment isolates only the ambiguity tie-break.
        used_t, used_d = set(), set()
        for i, j in matches:
            tr, d = self.tracks[i], active[j]
            oldc = np.asarray([(tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2])
            newc = np.asarray([(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2])
            tr.velocity = (0.7 * tr.velocity + 0.3 * (newc - oldc)).astype(np.float32)
            tr.box = d.box.copy(); tr.last_frame = frame; tr.misses = 0; tr.hits += 1; tr.last_score = d.score
            tr.feat = dance.normalize(.85 * tr.feat + .15 * d.feat).astype(np.float32)
            self._record_obs(tr, frame, d.feat, d.box, tr.velocity, d.score, missing=not (d.score >= .35))
            tr.history.append(d.feat.copy())
            if len(tr.history) > 8:
                tr.history = tr.history[-8:]
            self.outputs.setdefault(frame, []).append((tr.tid, tr.box.copy(), d.score))
            self.assignment_pairs.setdefault(frame, []).append((int(tr.tid), int(j)))
            used_t.add(i); used_d.add(j)
        survivors = []
        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                pred_box = tr.box.copy()
                pred_box[[0, 2]] += tr.velocity[0]; pred_box[[1, 3]] += tr.velocity[1]
                self._record_obs(tr, frame, np.zeros_like(tr.feat), pred_box, tr.velocity, 0.0, missing=True)
                tr.misses += 1
            if tr.misses <= self.max_age:
                survivors.append(tr)
        self.tracks = survivors
        for j, d in enumerate(active):
            if j in used_d:
                continue
            tr_new = dance.Track(self.next_id, d.box.copy(), d.feat.copy(), frame,
                                 last_score=d.score, history=[d.feat.copy()])
            tr_new.obs_history = [dance.HistoryObs(frame, dance.normalize(d.feat).astype(np.float32),
                                                   d.box.copy(), np.zeros(2, np.float32), d.score, False)]
            self.tracks.append(tr_new)
            self.outputs.setdefault(frame, []).append((self.next_id, d.box.copy(), d.score))
            self.assignment_pairs.setdefault(frame, []).append((int(self.next_id), int(j)))
            self.next_id += 1

    def run(self, dets_by_frame: Dict[int, List[dance.Det]]) -> Dict[int, List[Tuple[int, np.ndarray, float]]]:
        self.outputs = {}; self.assignment_pairs = {}
        self.cost_margins = []; self.quality_values = []; self.uncertainty_values = []
        self.memory_gate_accept = 0; self.memory_gate_reject = 0; self.temporal_prediction_count = 0
        self._reset_audit()
        for fr in sorted(dets_by_frame):
            self.step(int(fr), dets_by_frame[fr])
        return self.outputs

    def audit_stats(self) -> Dict[str, object]:
        return {
            "method": self.variant,
            "margin_threshold": float(self.margin_threshold),
            "event_rows": len(self.event_rows),
            "event_trace_sha256": hash_json(self.event_rows),
            "tie_eligible_events": int(self.tie_eligible_events),
            "tie_applied_events": int(self.tie_applied_events),
            "assignment_changed_rows": int(self.assignment_changed_rows),
            "assignment_changed_outside_tie": int(self.assignment_changed_outside_tie),
            "native_match_count": int(self.native_match_count),
            "used_match_count": int(self.used_match_count),
            "candidate_independent_predictor": True,
            "native_cost_primary": True,
            "no_repair": True,
            "quality_mean": float(np.mean(self.quality_values)) if self.quality_values else None,
            "uncertainty_prob_mean": float(np.mean(self.uncertainty_values)) if self.uncertainty_values else None,
            "temporal_prediction_count": int(self.temporal_prediction_count),
        }


def collect_train_diagnostic(train_items, predictors, max_frames: int, seed: int) -> Dict[str, object]:
    """Derive p25 and affected-event counts from train suffix only."""
    suffix_start = max(1, int(max_frames * (1.0 - TRAIN_SUFFIX_FRACTION)) + 1)
    margins: List[float] = []
    event_rows: List[Dict[str, object]] = []
    per_method = {m: {"gate_pass": 0, "gate_fail": 0, "potential_tie_events": 0,
                      "predictor_prefers_native_top2": 0, "q": [], "u": []}
                  for m in PREDICTOR_METHODS}
    for seq, seq_dir, gt, det, teacher in train_items:
        iw, ih = dance.sequence_wh(seq_dir)
        probe = dance.SimpleMOT("raw", None, seed=seed, image_width=iw, image_height=ih)
        for fr in sorted(det):
            active, native, pred_boxes = native_cost_matrix(probe, int(fr), det[fr])
            if int(fr) >= suffix_start:
                for i, tr in enumerate(probe.tracks):
                    finite = np.where(np.isfinite(native[i]) & (native[i] < 1e5))[0]
                    if finite.size < 2:
                        continue
                    order = finite[np.argsort(native[i, finite], kind="mergesort")]
                    j1, j2 = int(order[0]), int(order[1])
                    margin = float(native[i, j2] - native[i, j1])
                    margins.append(margin)
                    row = {"sequence": seq, "frame": int(fr), "track_index": int(i),
                           "track_id": int(tr.tid), "top1": j1, "top2": j2,
                           "native_top1": float(native[i, j1]), "native_top2": float(native[i, j2]),
                           "native_margin": margin}
                    for method, pred in predictors.items():
                        p = pred.predict(probe._temporal_tokens(tr, int(fr)))
                        ok, q, up = gate_for(tr, p)
                        pc1 = predictor_cost(probe, tr, p, active[j1], pred_boxes[i])
                        pc2 = predictor_cost(probe, tr, p, active[j2], pred_boxes[i])
                        pref2 = bool(pc2 + 1e-7 < pc1)
                        st = per_method[method]
                        st["q"].append(q); st["u"].append(up)
                        if ok: st["gate_pass"] += 1
                        else: st["gate_fail"] += 1
                        if ok and pref2 and native[i, j1] < MAX_ASSIGN_COST and native[i, j2] < MAX_ASSIGN_COST:
                            st["potential_tie_events"] += 1
                        if pref2: st["predictor_prefers_native_top2"] += 1
                        row[method] = {"quality": q, "uncertainty_prob": up, "gate_pass": ok,
                                       "predictor_cost_top1": float(pc1), "predictor_cost_top2": float(pc2),
                                       "prefers_top2": pref2}
                    event_rows.append(row)
            probe.step(int(fr), det[fr])
    if not margins:
        raise RuntimeError("train suffix produced no native top1/top2 margins")
    margin_arr = np.asarray(margins, dtype=np.float64)
    threshold = float(np.percentile(margin_arr, MARGIN_QUANTILE))
    for m, st in per_method.items():
        for k in ("q", "u"):
            arr = np.asarray(st.pop(k), dtype=np.float64)
            st[k + "_mean"] = float(np.mean(arr)) if arr.size else None
            st[k + "_p10"] = float(np.percentile(arr, 10)) if arr.size else None
            st[k + "_p90"] = float(np.percentile(arr, 90)) if arr.size else None
        st["ambiguous_rows_at_threshold"] = int(np.sum(margin_arr <= threshold))
    return {
        "schema": "dance-got-jepa-ambiguity-train-diagnostic-v1",
        "source": "train sequences only; raw/native tracker state; no validation GT",
        "train_sequences": [x[0] for x in train_items], "max_frames": int(max_frames),
        "suffix_fraction": TRAIN_SUFFIX_FRACTION, "suffix_start_frame": suffix_start,
        "margin_quantile": MARGIN_QUANTILE, "margin_threshold_p25": threshold,
        "margin_count": int(margin_arr.size),
        "margin_summary": {k: float(v) for k, v in zip(("min", "p10", "p25", "p50", "p75", "p90", "max"), np.percentile(margin_arr, [0, 10, 25, 50, 75, 90, 100]))},
        "affected_event_diagnostic": per_method,
        "event_rows": event_rows,
        "event_rows_sha256": hash_json(event_rows),
        "eval_gt_parsed": False,
        "threshold_fixed_before_eval": True,
    }


def tracks_hash(outputs: Dict[int, List[Tuple[int, np.ndarray, float]]]) -> str:
    h = hashlib.sha256()
    for fr in sorted(outputs):
        for tid, box, score in outputs[fr]:
            h.update(np.asarray([int(fr), int(tid)], np.int64).tobytes())
            h.update(np.asarray(box, np.float32).tobytes())
            h.update(np.asarray([float(score)], np.float32).tobytes())
    return h.hexdigest()


def run_trackers(eval_items, predictors, threshold: float, out: Path, seed: int):
    all_records: Dict[str, Dict[str, object]] = {}
    no_gt = {"eval_uses_gt": False, "gt_argument_in_tracker": False,
             "tracker_completed_before_eval_gt_parse": True, "sequence_hashes": {}}
    for seq, seq_dir, det, cache_meta in eval_items:
        iw, ih = dance.sequence_wh(seq_dir)
        all_records[seq] = {}; no_gt["sequence_hashes"][seq] = {}
        for method in METHODS:
            pred = predictors.get(method)
            tr = AmbiguityMOT(pred, method, threshold, seed, iw, ih)
            t0 = time.perf_counter(); result = tr.run(det); elapsed = time.perf_counter() - t0
            path = out / "dance" / "tracks" / method / (seq + ".txt")
            dance.save_track_file(path, result)
            stats = tr.audit_stats()
            rec = {"outputs": result, "track_path": str(path), "track_sha256": sha256(path),
                   "semantic_sha256": tracks_hash(result), "track_rows": len(dance.output_rows(result)),
                   "association_seconds": elapsed, "cache": cache_meta, **stats}
            all_records[seq][method] = rec
            no_gt["sequence_hashes"][seq][method] = rec["semantic_sha256"]
    no_gt["tracker_completed_unix"] = time.time()
    return all_records, no_gt


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


def invoke_trackeval(out: Path, eval_names: Sequence[str], methods: Sequence[str], max_frames: int, trackeval_root: Path) -> Dict[str, object]:
    te_dir = out / "trackeval_official"
    te_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "audit_trackeval_real.py"), "--result-dir", str(out),
           "--dataset", "dance", "--max-frames", str(max_frames), "--sequences", ",".join(eval_names),
           "--methods", ",".join(methods), "--trackeval-root", str(trackeval_root), "--out-dir", str(te_dir)]
    log = out / "trackeval_official_run.log"
    try:
        with log.open("w") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, check=False)
        audit_path = te_dir / "audit.json"
        audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"status": "missing"}
        audit["runner_returncode"] = int(proc.returncode)
        if proc.returncode != 0 or audit.get("status") != "complete":
            raise RuntimeError("official TrackEval failed; see {}".format(log))
        return audit
    except Exception:
        fail = {"command": cmd, "traceback": traceback.format_exc(), "log": str(log)}
        write_json(out / "failed_attempts" / "trackeval_failure.json", fail)
        raise


def combined_from_audit(audit: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    return dict(audit.get("combined", {})) if isinstance(audit.get("combined"), dict) else {}


def build_summary(audit: Dict[str, object], methods: Sequence[str]) -> Dict[str, object]:
    combined = combined_from_audit(audit)
    raw = combined.get("raw", {})
    rows: Dict[str, object] = {}
    metric_names = ("HOTA_official_mean", "HOTA_official_alpha0.50", "AssA_official_alpha0.50",
                   "DetA_official_alpha0.50", "IDF1_official", "MOTA_official", "IDSW_official")
    for method in methods:
        cur = combined.get(method, {})
        delta = {}
        for k in metric_names:
            try:
                delta[k] = float(cur[k]) - float(raw[k])
            except Exception:
                delta[k] = None
        rows[method] = {"combined": cur, "delta_vs_raw": delta}
    cand_delta = rows.get("ambiguity_jepa", {}).get("delta_vs_raw", {})
    positive = bool(cand_delta.get("HOTA_official_mean") is not None and cand_delta.get("HOTA_official_mean") > 0 and cand_delta.get("IDF1_official", 0) > 0)
    return {"schema": "dance-got-jepa-ambiguity-summary-v1", "status": "positive" if positive else "no_positive_gain",
            "metric_definition": "Official vendored TrackEval HOTA/CLEAR/Identity on first 120 DanceTrack validation frames; HOTA/AssA/DetA at alpha=.50, IDF1/MOTA/IDSW.",
            "methods": list(methods), "combined": combined, "rows": rows,
            "candidate_positive_hota_and_idf1": positive,
            "no_eval_threshold_tuning": True, "no_posthoc_track_repair": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    ap.add_argument("--cache-dir", default=CACHE_DEFAULT)
    ap.add_argument("--dance-train", default=TRAIN_DEFAULT)
    ap.add_argument("--dance-eval", default=EVAL_DEFAULT)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--max-samples", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trackeval-root", default=TRACKEVAL_DEFAULT)
    args = ap.parse_args()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    (out / "failed_attempts").mkdir(exist_ok=True)
    seed_all(args.seed)
    checkpoint = Path(args.checkpoint).resolve(); cache_root = Path(args.cache_dir).resolve()
    payload: Dict[str, object] = {
        "schema": "dance-got-jepa-ambiguity-v1", "status": "started",
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
        "platform": platform.platform(), "seed": int(args.seed), "max_frames": int(args.max_frames),
        "methods": list(METHODS), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "cache_dir": str(cache_root), "encoder_retrained": False, "production_tracker_modified": False,
        "old_crop80_script_modified": False, "fixed_config": {
            "train_suffix_fraction": TRAIN_SUFFIX_FRACTION, "margin_quantile": MARGIN_QUANTILE,
            "quality_threshold": QUALITY_THRESHOLD, "uncertainty_prob_threshold": UNCERTAINTY_PROB_THRESHOLD,
            "max_assign_cost": MAX_ASSIGN_COST, "detector_threshold": DETECTION_THRESHOLD,
            "history_k": HISTORY_K, "tie_rule": "already-matched native top-two only; free candidate or quality-improving two-way swap",
        },
    }
    t0 = time.perf_counter()
    try:
        train_names, eval_names = list_names(args.dance_train), list_names(args.dance_eval)
        if args.max_frames != 120 or args.seed != 17 or args.epochs != 8:
            raise ValueError("this bounded audit is fixed to max_frames=120, seed=17, epochs=8")
        train_gt = dance.limit_frame_dicts(dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/train.json", "train", train_names), args.max_frames)
        train_items = []; feature_meta = {}
        for seq in train_names:
            item, meta = load_train_item(seq, args.max_frames, cache_root, payload["checkpoint_sha256"], train_gt)
            train_items.append(item); feature_meta[seq] = meta
        samples, sample_meta = got.build_temporal_samples_got(train_items, args.max_frames, args.seed, args.max_samples)
        predictors: Dict[str, Optional[dance.TemporalPredictor]] = {"raw": None}
        for method in ("ambiguity_jepa", "no_teacher"):
            pred = dance.TemporalPredictor(256, "jepa_temporal" if method == "ambiguity_jepa" else "no_teacher",
                                           args.seed, args.epochs, args.device)
            pred.fit(samples); predictors[method] = pred
        rnd = dance.TemporalPredictor(256, "random_frozen", args.seed, 0, "cpu"); rnd.fit_random(); predictors["random_frozen"] = rnd
        # This file is intentionally emitted before any validation data/GT is
        # loaded.  It is the immutable threshold source for the eval run.
        train_diag = collect_train_diagnostic(train_items,
                                              {m: predictors[m] for m in PREDICTOR_METHODS},
                                              args.max_frames, args.seed)
        write_json(out / "train_only_diagnostic.json", train_diag)
        threshold = float(train_diag["margin_threshold_p25"])
        payload["train_only_diagnostic"] = {k: v for k, v in train_diag.items() if k != "event_rows"}
        payload["margin_threshold"] = threshold

        # Only detector files and cached features are loaded before tracking;
        # validation annotations are parsed after all tracker rows are saved.
        eval_items = []
        for seq in eval_names:
            seq_dir, det, dm = load_sequence_detector(seq, "val", args.max_frames, cache_root, payload["checkpoint_sha256"])
            eval_items.append((seq, seq_dir, det, dm)); feature_meta[seq] = {"detector": dm, "teacher": None}
        tracked, no_gt = run_trackers(eval_items, predictors, threshold, out, args.seed)
        tracker_completed = no_gt["tracker_completed_unix"]
        # Save all track files and hashes before touching validation GT.
        eval_gt = dance.limit_frame_dicts(dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/val.json", "val", eval_names), args.max_frames)
        no_gt["eval_gt_parsed_unix"] = time.time(); no_gt["tracking_precedes_gt_parse"] = bool(tracker_completed <= no_gt["eval_gt_parsed_unix"])
        rows = []; sequences = {}
        for seq, seq_dir, det, dm in eval_items:
            sequences[seq] = {}
            for method in METHODS:
                rec = tracked[seq][method]
                metrics = dance.evaluate_mot(eval_gt[seq], rec["outputs"])
                public = {k: v for k, v in rec.items() if k != "outputs"}; public.update(metrics)
                public.update({"detector_digest": got.hash_records(got.detector_records(det)),
                               "detector_recall_iou50": dance.detector_recall(eval_gt[seq], det)})
                sequences[seq][method] = public; rows.append({"dataset": "dance", "sequence": seq, "method": method, **public})
        payload["dance"] = {"dataset": "dance", "train_sequences": train_names, "eval_sequences": eval_names,
                             "feature_cache": feature_meta, "train_samples": {k: list(v.shape) for k, v in samples.items()},
                             "sample_audit": sample_meta,
                             "predictor_stats": {m: (predictors[m].train_stats if predictors.get(m) is not None else None) for m in METHODS},
                             "margin_threshold": threshold, "sequences": sequences, "rows": rows,
                             "no_gt_audit": no_gt,
                             "training_protocol": {"teacher_gt_only": True, "teacher_scope": "train sequences only",
                                                   "teacher_encoder": "frozen GOT-JEPA_378 pooled 256-D cache",
                                                   "detector_encoder": "same frozen GOT-JEPA_378 pooled 256-D cache",
                                                   "student_context_gt": False, "gt_geometry_in_student": False,
                                                   "gt_visibility_in_student": False, "eval_uses_gt": False,
                                                   "candidate_in_predictor": False, "threshold_source": "train suffix native margins p25",
                                                   "tie_break": "candidate-independent history prediction on native top-two ambiguity only",
                                                   "no_posthoc_track_repair": True}}
        payload["status"] = "complete_tracker_pending_trackeval"
        payload["elapsed_seconds"] = time.perf_counter() - t0
        payload["script_sha256"] = sha256(Path(__file__).resolve())
        write_json(out / "results.json", payload)
        keys = sorted({k for r in rows for k in r})
        with (out / "per_sequence.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        # Official TrackEval is a separate process and reads only saved tracks.
        audit = invoke_trackeval(out, eval_names, METHODS, args.max_frames, Path(args.trackeval_root).resolve())
        summary = build_summary(audit, METHODS)
        write_json(out / "summary.json", summary)
        candidate_delta = summary["rows"].get("ambiguity_jepa", {}).get("delta_vs_raw", {})
        reflection_status = "positive" if summary.get("candidate_positive_hota_and_idf1") else "no_positive_gain"
        reflection = {
            "schema": "dance-got-jepa-ambiguity-reflection-v1", "status": reflection_status,
            "question": "Does ambiguity-gated candidate-independent GOT-JEPA improve association over native raw assignment?",
            "fixed_threshold": threshold, "threshold_source": "train suffix p25 only",
            "candidate_delta_vs_raw": candidate_delta, "official_trackeval_status": audit.get("status"),
            "interpretation": ("The ambiguity-only candidate improves both combined HOTA mean and IDF1 under the preregistered rule."
                               if reflection_status == "positive" else
                               "The preregistered ambiguity-only candidate does not improve both combined HOTA mean and IDF1; retain raw/native assignment and do not tune on eval."),
            "next_step": ("Confirm on a second fixed seed/split before integration." if reflection_status == "positive" else
                          "Try a separately preregistered ambiguity confidence calibration or train a MOT-specific representation; do not alter this run."),
            "no_eval_threshold_tuning": True, "no_posthoc_track_repair": True,
        }
        write_json(out / "reflection.json", reflection)
        payload["status"] = "complete"; payload["trackeval_status"] = audit.get("status")
        payload["elapsed_seconds"] = time.perf_counter() - t0
        write_json(out / "results.json", payload)
        # Immutable source copies make the run independently auditable.
        prov = out / "provenance"; prov.mkdir(exist_ok=True)
        for src in (Path(__file__).resolve(), HERE / "mot_dance_jepa_got.py", HERE / "mot_dance_jepa_real.py", HERE / "audit_trackeval_real.py"):
            shutil.copy2(src, prov / src.name)
        provenance = {"schema": "dance-got-jepa-ambiguity-provenance-v1", "results_sha256": sha256(out / "results.json"),
                      "train_diagnostic_sha256": sha256(out / "train_only_diagnostic.json"),
                      "summary_sha256": sha256(out / "summary.json"), "reflection_sha256": sha256(out / "reflection.json"),
                      "trackeval_audit_sha256": sha256(out / "trackeval_official" / "audit.json"),
                      "checkpoint": str(checkpoint), "checkpoint_sha256": payload["checkpoint_sha256"],
                      "cache_dir": str(cache_root), "cache_files": sorted(str(p) for p in cache_root.glob("*.npz")),
                      "source_files": {src.name: sha256(prov / src.name) for src in (Path(__file__).resolve(), HERE / "mot_dance_jepa_got.py", HERE / "mot_dance_jepa_real.py", HERE / "audit_trackeval_real.py")},
                      "failed_attempts": sorted(str(p) for p in (out / "failed_attempts").glob("*")),
                      "encoder_retrained": False, "eval_gt_used_for_threshold": False, "no_posthoc_track_repair": True}
        write_json(out / "provenance.json", provenance)
        print(json.dumps({"status": "complete", "out": str(out), "threshold": threshold,
                          "combined": summary.get("combined"), "reflection": reflection_status}, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        payload["status"] = "failure"; payload["error_type"] = type(exc).__name__; payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(); payload["elapsed_seconds"] = time.perf_counter() - t0
        write_json(out / "failure.json", payload); write_json(out / "results.json", payload)
        print(json.dumps({"status": "failure", "out": str(out), "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
