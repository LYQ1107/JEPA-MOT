#!/usr/bin/env python3
"""Reappearance-only GOT-JEPA association experiment on DanceTrack.

The frozen public GOT-JEPA checkpoint is represented by the existing 256-D
pooled feature cache.  This runner tests the MOT-specific hypothesis that a
history predictor is useful after an observation gap, while the native/raw
appearance remains untouched for continuously visible tracks.  A predictor is
computed once per track from detector history (before looking at current
candidates), and is used only when ``1 <= misses == gap <= 4`` and the fixed
quality/uncertainty gates pass.  Track lifecycle and geometry are otherwise
the raw SimpleMOT lifecycle.  Validation GT is parsed only after all tracker
files have been written and hashed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# The isolated GOT and ambiguity helpers only read existing code/cache here;
# no encoder is constructed by this experiment.
import mot_dance_jepa_got as got  # noqa: E402
import mot_dance_jepa_got_ambiguity as amb  # noqa: E402
import mot_dance_jepa_real as dance  # noqa: E402


CHECKPOINT_DEFAULT = got.CHECKPOINT_DEFAULT
CACHE_DEFAULT = "/data2/usr_for_deadline/mot_dance_jepa_got_120/feature_cache"
TRAIN_DEFAULT = "dancetrack0001,dancetrack0002,dancetrack0006"
EVAL_DEFAULT = "dancetrack0004,dancetrack0007,dancetrack0014"
TRACKEVAL_DEFAULT = "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/TrackEval-official"
METHODS = ("raw", "reappearance_jepa", "no_teacher", "random_frozen")
PREDICTOR_METHODS = ("reappearance_jepa", "no_teacher", "random_frozen")

# Fixed task-semantic gates.  They are not command-line sweep parameters.
MAX_ASSIGN_COST = 0.82
DETECTION_THRESHOLD = 0.10
QUALITY_THRESHOLD = 0.35
UNCERTAINTY_PROB_THRESHOLD = 0.65
GAP_MIN = 1
GAP_MAX = 4
HISTORY_K = 4


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def hash_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=float, separators=(",", ":")).encode()).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=float) + "\n")


def list_names(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -40.0, 40.0))
    return float(1.0 / (1.0 + math.exp(-x)))


def gate_for(tr: dance.Track, prediction: Tuple[np.ndarray, float, float]) -> Tuple[bool, float, float]:
    _, quality, uncertainty = prediction
    up = sigmoid(float(uncertainty))
    ok = bool(float(quality) >= QUALITY_THRESHOLD and up <= UNCERTAINTY_PROB_THRESHOLD and int(tr.misses) <= GAP_MAX)
    return ok, float(quality), float(up)


def tracks_hash(outputs: Dict[int, List[Tuple[int, np.ndarray, float]]]) -> str:
    h = hashlib.sha256()
    for fr in sorted(outputs):
        for tid, box, score in outputs[fr]:
            h.update(np.asarray([int(fr), int(tid)], np.int64).tobytes())
            h.update(np.asarray(box, np.float32).tobytes())
            h.update(np.asarray([float(score)], np.float32).tobytes())
    return h.hexdigest()


def load_train_predictors(payload: Dict[str, object], cache_root: Path, device: str):
    seed = int(payload["seed"]); max_frames = int(payload["max_frames"]); epochs = int(payload["epochs"])
    train_names = list_names(str(payload["dance_train"]))
    checkpoint_sha = str(payload["checkpoint_sha256"])
    train_gt = dance.limit_frame_dicts(
        dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/train.json", "train", train_names), max_frames)
    train_items = []; feature_meta = {}
    for seq in train_names:
        item, meta = amb.load_train_item(seq, max_frames, cache_root, checkpoint_sha, train_gt)
        train_items.append(item); feature_meta[seq] = meta
    samples, sample_meta = got.build_temporal_samples_got(train_items, max_frames, seed, int(payload["max_samples"]))
    predictors: Dict[str, Optional[dance.TemporalPredictor]] = {"raw": None}
    for method, mode in (("reappearance_jepa", "jepa_temporal"), ("no_teacher", "no_teacher")):
        pred = dance.TemporalPredictor(256, mode, seed, epochs, device)
        pred.fit(samples); predictors[method] = pred
    rnd = dance.TemporalPredictor(256, "random_frozen", seed, 0, "cpu"); rnd.fit_random(); predictors["random_frozen"] = rnd
    return train_items, predictors, feature_meta, sample_meta, samples


def native_cost_matrix(tracker: dance.SimpleMOT, frame: int, dets):
    # Keep one implementation of the raw/native cost to make replay hashes
    # comparable with the prior ambiguity audit.
    return amb.native_cost_matrix(tracker, frame, dets)


def predictor_cost(tracker, tr, pred_tuple, d, pred_box):
    return amb.predictor_cost(tracker, tr, pred_tuple, d, pred_box)


def detector_gid_map(gt_rows, active):
    """IoU=.5 one-to-one labels used only by the train correctness audit."""
    out = {}
    if not gt_rows or not active:
        return out
    mat = np.asarray([[dance.box_iou(g.box, d.box) for d in active] for g in gt_rows], dtype=np.float32)
    rr, cc = linear_sum_assignment(-mat)
    for i, j in zip(rr.tolist(), cc.tolist()):
        if float(mat[i, j]) >= 0.5:
            out[int(j)] = int(gt_rows[i].gid)
    return out


class ReappearanceMOT(dance.SimpleMOT):
    """Raw lifecycle whose appearance branch activates only after a gap."""

    def __init__(self, predictor: Optional[dance.TemporalPredictor], method: str,
                 seed: int, image_width: float, image_height: float):
        # Inherited ``method`` is raw so box/lifecycle defaults are shared;
        # ``variant`` selects only whether a gap predictor is available.
        super().__init__("raw", None, max_age=30, det_thr=DETECTION_THRESHOLD,
                         seed=seed, image_width=image_width, image_height=image_height)
        self.predictor = predictor
        self.variant = method
        self.event_rows: List[Dict[str, object]] = []
        self.lifecycle_rows: List[Dict[str, object]] = []
        self.native_assignment_pairs: Dict[int, List[Tuple[int, int]]] = {}
        self.used_assignment_pairs: Dict[int, List[Tuple[int, int]]] = {}
        self.reappearance_rows = 0
        self.gate_pass_rows = 0
        self.jepa_used_rows = 0
        self.continuous_raw_rows = 0
        self.memory_update_accept = 0
        self.memory_update_reject = 0
        self.assignment_changed_rows = 0
        self.assignment_changed_outside_reappearance = 0
        self.native_match_count = 0
        self.used_match_count = 0

    def _reset_audit(self) -> None:
        self.event_rows = []
        self.lifecycle_rows = []
        self.native_assignment_pairs = {}
        self.used_assignment_pairs = {}
        self.reappearance_rows = 0; self.gate_pass_rows = 0; self.jepa_used_rows = 0
        self.continuous_raw_rows = 0; self.memory_update_accept = 0; self.memory_update_reject = 0
        self.assignment_changed_rows = 0; self.assignment_changed_outside_reappearance = 0
        self.native_match_count = 0; self.used_match_count = 0

    def step(self, frame: int, dets: Sequence[dance.Det]) -> None:
        active, native, pred_boxes = native_cost_matrix(self, frame, dets)
        n, m = native.shape
        pred_cache: Dict[int, Tuple[np.ndarray, float, float]] = {}
        gate_cache: Dict[int, Tuple[bool, float, float]] = {}
        gap_cache: Dict[int, Dict[str, object]] = {}
        jcost = np.full((n, m), 1e6, dtype=np.float32)

        # Gap state is an observable tracker property and is recorded for the
        # raw control as well as predictor methods.  Computing it outside the
        # predictor branch keeps the baseline event denominator honest.
        for i, tr in enumerate(self.tracks):
            gap = int(tr.misses)
            frame_gap = max(0, int(frame) - int(tr.last_frame) - 1)
            gap_ok = bool(GAP_MIN <= gap <= GAP_MAX and frame_gap == gap)
            gap_cache[i] = {"misses": gap, "frame_gap": frame_gap, "gap_ok": gap_ok}
            if gap_ok:
                self.reappearance_rows += 1

        # Predictor output is produced once per track, before candidate costs.
        # This is the candidate-independent reappearance invariant.
        if self.predictor is not None:
            for i, tr in enumerate(self.tracks):
                pred = self.predictor.predict(self._temporal_tokens(tr, frame))
                pred_cache[i] = pred
                gate_cache[i] = gate_for(tr, pred)
                self.quality_values.append(float(pred[1]))
                self.uncertainty_values.append(float(gate_cache[i][2]))
                if gap_cache[i]["gap_ok"] and gate_cache[i][0]:
                    self.gate_pass_rows += 1
                gap_ok = bool(gap_cache[i]["gap_ok"])
                if gap_ok and gate_cache[i][0]:
                    self.jepa_used_rows += 1
                    for j, d in enumerate(active):
                        jcost[i, j] = predictor_cost(self, tr, pred, d, pred_boxes[i])
        # Raw/continuous rows keep the native cost exactly.  Rows with a
        # valid gap and predictor gate use only the history-only JEPA cost.
        used_cost = native.copy()
        use_jepa: Dict[int, bool] = {}
        for i, tr in enumerate(self.tracks):
            if self.predictor is None:
                use_jepa[i] = False
                self.continuous_raw_rows += int(int(tr.misses) == 0)
                continue
            info = gap_cache.get(i, {"misses": int(tr.misses), "frame_gap": max(0, int(frame) - int(tr.last_frame) - 1), "gap_ok": False})
            use = bool(info["gap_ok"] and gate_cache.get(i, (False, 0.0, 1.0))[0])
            use_jepa[i] = use
            if not use and int(info["misses"]) == 0:
                self.continuous_raw_rows += 1
            if use and m:
                used_cost[i, :] = jcost[i, :]

        # Record native and used Hungarian assignments separately; there is no
        # post-hoc repair or tie manipulation in this branch.
        native_matches: List[Tuple[int, int]] = []
        used_matches: List[Tuple[int, int]] = []
        if n and m:
            rr, cc = linear_sum_assignment(native)
            native_matches = [(i, j) for i, j in zip(rr.tolist(), cc.tolist()) if native[i, j] < MAX_ASSIGN_COST]
            rr, cc = linear_sum_assignment(used_cost)
            used_matches = [(i, j) for i, j in zip(rr.tolist(), cc.tolist()) if used_cost[i, j] < MAX_ASSIGN_COST]
        native_by_row = {i: j for i, j in native_matches}; used_by_row = {i: j for i, j in used_matches}
        self.native_match_count += len(native_matches); self.used_match_count += len(used_matches)
        self.native_assignment_pairs.setdefault(int(frame), []).extend((int(self.tracks[i].tid), int(j)) for i, j in native_matches)
        self.used_assignment_pairs.setdefault(int(frame), []).extend((int(self.tracks[i].tid), int(j)) for i, j in used_matches)

        # Event trace includes every track row that has at least two finite
        # native candidates, plus its gap/gate state.  It is kept separately
        # from results.json to make causal audits easy without GT leakage.
        for i, tr in enumerate(self.tracks):
            finite = np.where(np.isfinite(native[i]) & (native[i] < 1e5))[0]
            order = finite[np.argsort(native[i, finite], kind="mergesort")] if finite.size else np.asarray([], dtype=int)
            top1 = int(order[0]) if order.size else None
            top2 = int(order[1]) if order.size >= 2 else None
            margin = float(native[i, top2] - native[i, top1]) if top2 is not None else None
            jbest = None
            if use_jepa.get(i, False) and m:
                jf = np.where(np.isfinite(jcost[i]) & (jcost[i] < 1e5))[0]
                if jf.size:
                    jbest = int(jf[np.argmin(jcost[i, jf])])
            info = gap_cache.get(i, {"misses": int(tr.misses), "frame_gap": max(0, int(frame) - int(tr.last_frame) - 1), "gap_ok": False})
            q, up = gate_cache.get(i, (False, 0.0, 1.0))[1:]
            self.event_rows.append({
                "frame": int(frame), "track_index": int(i), "track_id": int(tr.tid),
                "misses_before": int(info["misses"]), "gap_frames": int(info["frame_gap"]),
                "gap_ok": bool(info["gap_ok"]), "quality": float(q), "uncertainty_prob": float(up),
                "gate_pass": bool(gate_cache.get(i, (False, 0.0, 1.0))[0]), "jepa_used": bool(use_jepa.get(i, False)),
                "native_top1": top1, "native_top2": top2, "native_margin": margin,
                "native_top1_cost": float(native[i, top1]) if top1 is not None else None,
                "native_top2_cost": float(native[i, top2]) if top2 is not None else None,
                "jepa_best": jbest, "native_assignment": native_by_row.get(i), "used_assignment": used_by_row.get(i),
            })
            if top2 is not None:
                self.cost_margins.append(float(margin))
        native_map = {i: j for i, j in native_matches}
        for i, j in used_matches:
            if native_map.get(i) != j:
                self.assignment_changed_rows += 1
                if not use_jepa.get(i, False):
                    self.assignment_changed_outside_reappearance += 1

        # Raw lifecycle with the only deliberate exception being the memory
        # gate on a reappearance row.  No predicted latent is written to tr.feat.
        used_t, used_d = set(), set()
        for i, j in used_matches:
            tr, d = self.tracks[i], active[j]
            oldc = np.asarray([(tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2])
            newc = np.asarray([(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2])
            tr.velocity = (0.7 * tr.velocity + 0.3 * (newc - oldc)).astype(np.float32)
            tr.box = d.box.copy(); tr.last_frame = frame; tr.misses = 0; tr.hits += 1; tr.last_score = d.score
            legal = bool(d.score >= .35 and (1.0 - float(used_cost[i, j])) >= .08)
            gap_info = gap_cache.get(i, {"gap_ok": False})
            gate_ok = gate_cache.get(i, (False, 0.0, 1.0))[0]
            # Lifecycle-invariant rule: a predictor method is allowed to
            # change memory semantics only on the row where its prototype was
            # actually used.  Continuous-visible rows, out-of-range gaps, and
            # predictor-gate failures copy raw SimpleMOT exactly (raw always
            # commits an observed crop; the missing bit depends only on score).
            if self.predictor is None or not bool(use_jepa.get(i, False)):
                update_ok = True
                missing = not (d.score >= .35)
                memory_source = "raw_exact"
            else:
                update_ok = bool(legal and gate_ok)
                missing = not update_ok
                memory_source = "jepa_gate"
            self.lifecycle_rows.append({
                "frame": int(frame), "track_index": int(i), "track_id": int(tr.tid),
                "jepa_used": bool(use_jepa.get(i, False)), "score": float(d.score),
                "legal_raw_observation": bool(legal), "gate_pass": bool(gate_ok),
                "update_ok": bool(update_ok), "missing": bool(missing),
                "memory_source": memory_source,
            })
            if update_ok:
                self.memory_update_accept += 1
                tr.feat = dance.normalize(.85 * tr.feat + .15 * d.feat).astype(np.float32)
            else:
                self.memory_update_reject += 1
            self._record_obs(tr, frame, d.feat, d.box, tr.velocity, d.score, missing=missing)
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
            "method": self.variant, "event_rows": len(self.event_rows),
            "event_trace_sha256": hash_json(self.event_rows),
            "lifecycle_rows": len(self.lifecycle_rows),
            "lifecycle_trace_sha256": hash_json(self.lifecycle_rows),
            "reappearance_rows": int(self.reappearance_rows), "gate_pass_rows": int(self.gate_pass_rows),
            "jepa_used_rows": int(self.jepa_used_rows), "continuous_raw_rows": int(self.continuous_raw_rows),
            "memory_update_accept": int(self.memory_update_accept), "memory_update_reject": int(self.memory_update_reject),
            "native_match_count": int(self.native_match_count), "used_match_count": int(self.used_match_count),
            "assignment_changed_rows": int(self.assignment_changed_rows),
            "assignment_changed_outside_reappearance": int(self.assignment_changed_outside_reappearance),
            "candidate_independent_predictor": True, "reappearance_rule": "misses>=1, frame_gap==misses, gap<=4",
            "native_continuous_cost_unchanged": True, "no_posthoc_repair": True,
            "quality_mean": float(np.mean(self.quality_values)) if self.quality_values else None,
            "uncertainty_prob_mean": float(np.mean(self.uncertainty_values)) if self.uncertainty_values else None,
            "temporal_prediction_count": int(self.temporal_prediction_count),
        }


def collect_train_diagnostic(train_items, predictors, max_frames: int, seed: int) -> Dict[str, object]:
    """Count task-semantic gap events and train-only causal correctness."""
    events: List[Dict[str, object]] = []
    stats = {m: collections.Counter() for m in PREDICTOR_METHODS}
    stats["raw"] = collections.Counter()
    for seq, seq_dir, gt, det, teacher in train_items:
        iw, ih = dance.sequence_wh(seq_dir)
        probe = dance.SimpleMOT("raw", None, seed=seed, image_width=iw, image_height=ih)
        tid_gid: Dict[int, int] = {}
        for fr in sorted(det):
            active, native, pred_boxes = native_cost_matrix(probe, int(fr), det[fr])
            gids = detector_gid_map(gt.get(int(fr), []), active)
            # Native state before this frame determines the reappearance gap.
            for i, tr in enumerate(probe.tracks):
                gap = int(tr.misses); frame_gap = max(0, int(fr) - int(tr.last_frame) - 1)
                gap_ok = bool(GAP_MIN <= gap <= GAP_MAX and frame_gap == gap)
                if not gap_ok:
                    continue
                stats["raw"]["gap_rows"] += 1
                finite = np.where(np.isfinite(native[i]) & (native[i] < 1e5))[0]
                if finite.size == 0:
                    continue
                order = finite[np.argsort(native[i, finite], kind="mergesort")]
                native_best = int(order[0])
                target_gid = tid_gid.get(int(tr.tid))
                if target_gid is None and gt.get(int(fr)):
                    vals = [(dance.box_iou(g.box, tr.box), int(g.gid)) for g in gt[int(fr)]]
                    if vals and max(vals)[0] >= .5:
                        target_gid = max(vals)[1]
                event = {"sequence": seq, "frame": int(fr), "track_id": int(tr.tid),
                         "misses": gap, "frame_gap": frame_gap, "native_best": native_best,
                         "native_best_cost": float(native[i, native_best]), "target_gid": target_gid}
                for method in PREDICTOR_METHODS:
                    pred = predictors[method].predict(probe._temporal_tokens(tr, int(fr)))
                    ok, q, up = gate_for(tr, pred)
                    st = stats[method]; st["gap_rows"] += 1; st["gate_pass"] += int(ok)
                    jcost = np.asarray([predictor_cost(probe, tr, pred, d, pred_boxes[i]) for d in active], np.float32)
                    jf = np.where(np.isfinite(jcost) & (jcost < 1e5))[0]
                    jbest = int(jf[np.argmin(jcost[jf])]) if jf.size else None
                    st["predictor_best_changes"] += int(jbest is not None and jbest != native_best)
                    eligible = bool(ok and jbest is not None)
                    st["eligible_gap_events"] += int(eligible)
                    if eligible:
                        g_native, g_jepa = gids.get(native_best), gids.get(jbest)
                        st["label_available"] += int(target_gid is not None)
                        n_ok = bool(target_gid is not None and g_native == target_gid)
                        j_ok = bool(target_gid is not None and g_jepa == target_gid)
                        st["native_correct"] += int(n_ok); st["jepa_correct"] += int(j_ok)
                        st["improvement"] += int((not n_ok) and j_ok); st["regression"] += int(n_ok and (not j_ok))
                    event[method] = {"quality": q, "uncertainty_prob": up, "gate_pass": ok,
                                     "jepa_best": jbest, "jepa_best_cost": float(jcost[jbest]) if jbest is not None else None,
                                     "native_correct_gid": gids.get(native_best), "jepa_correct_gid": gids.get(jbest) if jbest is not None else None}
                events.append(event)
            # Advance raw state and maintain train-only identity labels.
            native_pairs = []
            if native.size:
                rr, cc = linear_sum_assignment(native)
                native_pairs = [(i, j) for i, j in zip(rr.tolist(), cc.tolist()) if native[i, j] < MAX_ASSIGN_COST]
            for i, j in native_pairs:
                if j in gids:
                    tid_gid[int(probe.tracks[i].tid)] = int(gids[j])
            before_ids = {int(t.tid) for t in probe.tracks}
            probe.step(int(fr), det[fr])
            for tid, j in probe.assignment_pairs.get(int(fr), []):
                if int(tid) not in before_ids and int(j) in gids:
                    tid_gid[int(tid)] = int(gids[int(j)])
        # Remove transient values before serialisation below.
    for method, st in stats.items():
        for key in ("quality", "uncertainty_prob"):
            pass
        if method != "raw":
            st["native_accuracy"] = float(st["native_correct"] / max(st["label_available"], 1))
            st["jepa_accuracy"] = float(st["jepa_correct"] / max(st["label_available"], 1))
            st["correctness_gain"] = int(st["improvement"] - st["regression"])
    return {
        "schema": "dance-got-jepa-reappearance-train-diagnostic-v1", "source": "train sequences only; no eval GT",
        "train_sequences": [x[0] for x in train_items], "max_frames": int(max_frames),
        "gap_rule": {"misses_min": GAP_MIN, "gap_max": GAP_MAX, "frame_gap_equals_misses": True},
        "quality_threshold": QUALITY_THRESHOLD, "uncertainty_measure": "sigmoid(log_variance)",
        "uncertainty_prob_threshold": UNCERTAINTY_PROB_THRESHOLD,
        "event_count": {m: int(st.get("gap_rows", 0)) for m, st in stats.items()},
        "affected_event_diagnostic": {m: dict(st) for m, st in stats.items()},
        "events": events, "events_sha256": hash_json(events), "eval_gt_read": False,
        "threshold_tuning_on_eval": False, "causal_correctness_is_descriptive_only": True,
    }


def write_event_traces(out: Path, seq: str, method: str, event_rows: List[Dict[str, object]]) -> Dict[str, object]:
    path = out / "event_traces" / (seq + "__" + method + ".json")
    dump(path, {"schema": "dance-got-jepa-reappearance-event-trace-v1", "sequence": seq, "method": method,
               "rows": event_rows, "rows_sha256": hash_json(event_rows)})
    return {"path": str(path), "sha256": sha256(path), "rows": len(event_rows)}


def write_lifecycle_traces(out: Path, seq: str, method: str, lifecycle_rows: List[Dict[str, object]]) -> Dict[str, object]:
    path = out / "event_traces" / (seq + "__" + method + "__lifecycle.json")
    dump(path, {"schema": "dance-got-jepa-reappearance-lifecycle-trace-v1", "sequence": seq,
               "method": method, "rows": lifecycle_rows, "rows_sha256": hash_json(lifecycle_rows)})
    return {"path": str(path), "sha256": sha256(path), "rows": len(lifecycle_rows)}


def run_trackers(eval_items, predictors, out: Path, seed: int):
    tracked = {}; no_gt = {"eval_uses_gt": False, "gt_argument_in_tracker": False,
                           "tracker_completed_before_eval_gt_parse": True, "sequence_hashes": {}}
    for seq, seq_dir, det, cache_meta in eval_items:
        iw, ih = dance.sequence_wh(seq_dir); tracked[seq] = {}; no_gt["sequence_hashes"][seq] = {}
        for method in METHODS:
            tr = ReappearanceMOT(predictors.get(method), method, seed, iw, ih)
            t0 = time.perf_counter(); result = tr.run(det); elapsed = time.perf_counter() - t0
            track_path = out / "dance" / "tracks" / method / (seq + ".txt")
            dance.save_track_file(track_path, result)
            trace = write_event_traces(out, seq, method, tr.event_rows)
            lifecycle_trace = write_lifecycle_traces(out, seq, method, tr.lifecycle_rows)
            st = tr.audit_stats()
            rec = {"outputs": result, "track_path": str(track_path), "track_sha256": sha256(track_path),
                   "semantic_sha256": tracks_hash(result), "track_rows": len(dance.output_rows(result)),
                   "association_seconds": elapsed, "cache": cache_meta, "event_trace": trace,
                   "lifecycle_trace": lifecycle_trace, **st}
            tracked[seq][method] = rec; no_gt["sequence_hashes"][seq][method] = rec["semantic_sha256"]
    no_gt["tracker_completed_unix"] = time.time()
    return tracked, no_gt


def invoke_trackeval(out: Path, eval_names: Sequence[str], trackeval_root: Path) -> Dict[str, object]:
    te = out / "trackeval_official"; te.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "audit_trackeval_real.py"), "--result-dir", str(out), "--dataset", "dance",
           "--max-frames", "120", "--sequences", ",".join(eval_names), "--methods", ",".join(METHODS),
           "--trackeval-root", str(trackeval_root), "--out-dir", str(te)]
    log = out / "trackeval_official_run.log"
    with log.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, text=True)
    path = te / "audit.json"; audit = json.loads(path.read_text()) if path.exists() else {"status": "missing"}
    audit["runner_returncode"] = int(proc.returncode)
    if proc.returncode != 0 or audit.get("status") != "complete":
        dump(out / "failed_attempts" / "trackeval_failure.json", {"command": cmd, "audit": audit,
                                                                      "log": str(log)})
        raise RuntimeError("official TrackEval failed; see {}".format(log))
    return audit


def summary_from_audit(audit: Dict[str, object]) -> Dict[str, object]:
    combined = audit.get("combined", {}) if isinstance(audit.get("combined"), dict) else {}
    raw = combined.get("raw", {})
    names = ("HOTA_official_mean", "HOTA_official_alpha0.50", "AssA_official_alpha0.50",
             "DetA_official_alpha0.50", "IDF1_official", "MOTA_official", "IDSW_official")
    rows = {}
    for method in METHODS:
        cur = combined.get(method, {}); delta = {}
        for name in names:
            try: delta[name] = float(cur[name]) - float(raw[name])
            except Exception: delta[name] = None
        rows[method] = {"combined": cur, "delta_vs_raw": delta}
    d = rows.get("reappearance_jepa", {}).get("delta_vs_raw", {})
    positive = bool(d.get("HOTA_official_mean") is not None and d.get("HOTA_official_mean") > 0 and d.get("IDF1_official", 0) > 0)
    return {"schema": "dance-got-jepa-reappearance-summary-v1", "status": "positive" if positive else "no_positive_gain",
            "metric_definition": "Official vendored TrackEval HOTA/CLEAR/Identity on first 120 DanceTrack val frames; HOTA/AssA/DetA alpha=.50, IDF1/MOTA/IDSW.",
            "methods": list(METHODS), "combined": combined, "rows": rows,
            "candidate_positive_hota_and_idf1": positive, "no_eval_threshold_tuning": True, "no_posthoc_track_repair": True}


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
    ap.add_argument("--diagnostic-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True); (out / "failed_attempts").mkdir(exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve(); cache_root = Path(args.cache_dir).resolve()
    payload: Dict[str, object] = {
        "schema": "dance-got-jepa-reappearance-v1", "status": "started", "command": " ".join(sys.argv),
        "cwd": os.getcwd(), "python": sys.version, "platform": platform.platform(), "seed": int(args.seed),
        "max_frames": int(args.max_frames), "max_samples": int(args.max_samples), "epochs": int(args.epochs),
        "dance_train": args.dance_train, "dance_eval": args.dance_eval, "methods": list(METHODS),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "cache_dir": str(cache_root),
        "encoder_retrained": False, "production_tracker_modified": False, "old_crop80_script_modified": False,
        "diagnostic_only": bool(args.diagnostic_only),
        "fixed_protocol": {"gap_rule": "misses>=1 and frame_gap==misses and gap<=4",
                            "quality_threshold": QUALITY_THRESHOLD, "uncertainty_measure": "sigmoid(log_variance)",
                            "uncertainty_prob_threshold": UNCERTAINTY_PROB_THRESHOLD,
                            "continuous_visible": "raw appearance+geometry exactly", "memory_update": "legal observation and gap gate",
                            "candidate_independent_predictor": True, "native_lifecycle_shared": True,
                            "no_posthoc_repair": True, "threshold_source": "task semantics; no eval-derived threshold"},
    }
    t0 = time.perf_counter()
    try:
        # Keep the experiment protocol fixed while allowing the pre-registered
        # second replication seed.  No other CLI sweep is permitted here.
        if args.max_frames != 120 or args.seed not in (17, 23) or args.epochs != 8:
            raise ValueError("bounded run is fixed to max_frames=120, seed=17 or 23, epochs=8")
        amb.seed_all(args.seed)
        train_items, predictors, feature_meta, sample_meta, samples = load_train_predictors(payload, cache_root, args.device)
        train_diag = collect_train_diagnostic(train_items, {m: predictors[m] for m in PREDICTOR_METHODS}, args.max_frames, args.seed)
        dump(out / "train_only_diagnostic.json", train_diag)
        protocol = {"schema": "dance-got-jepa-reappearance-protocol-v1", "dataset": "DanceTrack",
                    "train_sequences": list_names(args.dance_train), "eval_sequences": list_names(args.dance_eval),
                    "max_frames": args.max_frames, "detector": "LocateMOT D-CTRL six-column cache (no IDs)",
                    "feature_encoder": "frozen GOT_JEPA_378 256x27x27 mean-pooled L2 256-D cache",
                    "teacher_gt_only_train": True, "student_context_gt": False, "eval_gt_used_for_tracking": False,
                    "gap_rule": "track misses>=1, frame_gap==misses, <=4; task-semantic not tuned",
                    "continuous_visible_rule": "misses==0 uses native raw appearance/geometry",
                    "quality_gate": "quality>=0.35", "uncertainty_gate": "sigmoid(log_variance)<=0.65",
                    "predictor_candidate_independent": True, "memory_update_gate": "legal observed detection + gap gate",
                    "lifecycle": "raw SimpleMOT max_age=30; no posthoc repair", "controls": list(METHODS),
                    "train_only_diagnostic_sha256": sha256(out / "train_only_diagnostic.json")}
        dump(out / "protocol.json", protocol)
        payload["train_only_diagnostic"] = {k: v for k, v in train_diag.items() if k != "events"}
        payload["feature_cache"] = feature_meta; payload["sample_meta"] = sample_meta
        payload["predictor_stats"] = {m: (predictors[m].train_stats if predictors.get(m) is not None else None) for m in METHODS}
        if args.diagnostic_only:
            payload["status"] = "complete_diagnostic_only"; payload["elapsed_seconds"] = time.perf_counter() - t0
            dump(out / "results.json", payload)
            dump(out / "provenance.json", {"schema": "dance-got-jepa-reappearance-provenance-v1",
                                             "results_sha256": sha256(out / "results.json"),
                                             "diagnostic_sha256": sha256(out / "train_only_diagnostic.json"),
                                             "protocol_sha256": sha256(out / "protocol.json"),
                                             "checkpoint_sha256": payload["checkpoint_sha256"], "encoder_retrained": False,
                                             "eval_gt_read": False, "eval_tracking_run": False,
                                             "source_script_sha256": sha256(Path(__file__).resolve())})
            print(json.dumps({"status": payload["status"], "out": str(out),
                              "event_count": train_diag["event_count"],
                              "affected": train_diag["affected_event_diagnostic"]}, indent=2, sort_keys=True, default=float))
            return 0

        # Validation detections/features are loaded only after the train-only
        # diagnostic and protocol files have been committed.
        eval_items = []
        for seq in list_names(args.dance_eval):
            seq_dir, det, cm = amb.load_sequence_detector(seq, "val", args.max_frames, cache_root, payload["checkpoint_sha256"])
            eval_items.append((seq, seq_dir, det, cm))
        tracked, no_gt = run_trackers(eval_items, predictors, out, args.seed)
        tracker_completed = no_gt["tracker_completed_unix"]
        eval_gt = dance.limit_frame_dicts(dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/val.json", "val", list_names(args.dance_eval)), args.max_frames)
        no_gt["eval_gt_parsed_unix"] = time.time(); no_gt["tracking_precedes_gt_parse"] = bool(tracker_completed <= no_gt["eval_gt_parsed_unix"])
        rows = []; sequences = {}
        for seq, seq_dir, det, cm in eval_items:
            sequences[seq] = {}
            for method in METHODS:
                rec = tracked[seq][method]; metrics = dance.evaluate_mot(eval_gt[seq], rec["outputs"])
                public = {k: v for k, v in rec.items() if k != "outputs"}; public.update(metrics)
                public.update({"detector_digest": got.hash_records(got.detector_records(det)),
                               "detector_recall_iou50": dance.detector_recall(eval_gt[seq], det)})
                sequences[seq][method] = public; rows.append({"dataset": "dance", "sequence": seq, "method": method, **public})
        payload["dance"] = {"dataset": "dance", "train_sequences": list_names(args.dance_train),
                             "eval_sequences": list_names(args.dance_eval), "feature_cache": feature_meta,
                             "train_samples": {k: list(v.shape) for k, v in samples.items()}, "sample_meta": sample_meta,
                             "predictor_stats": {m: (predictors[m].train_stats if predictors.get(m) is not None else None) for m in METHODS},
                             "train_only_diagnostic": {k: v for k, v in train_diag.items() if k != "events"},
                             "protocol": protocol, "no_gt_audit": no_gt, "sequences": sequences, "rows": rows}
        payload["status"] = "complete_tracker_pending_trackeval"; payload["elapsed_seconds"] = time.perf_counter() - t0
        dump(out / "results.json", payload)
        keys = sorted({k for r in rows for k in r})
        with (out / "per_sequence.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        audit = invoke_trackeval(out, list_names(args.dance_eval), Path(args.trackeval_root).resolve())
        summary = summary_from_audit(audit); dump(out / "summary.json", summary)
        cand_delta = summary["rows"].get("reappearance_jepa", {}).get("delta_vs_raw", {})
        ref_status = "positive" if summary.get("candidate_positive_hota_and_idf1") else "no_positive_gain"
        reflection = {"schema": "dance-got-jepa-reappearance-reflection-v1", "status": ref_status,
                      "train_gap_diagnostic": train_diag["affected_event_diagnostic"],
                      "candidate_delta_vs_raw": cand_delta, "official_trackeval_status": audit.get("status"),
                      "interpretation": ("Reappearance-only GOT-JEPA improves both combined HOTA and IDF1 under the fixed task-semantic gate; verify on a second fixed seed before integration." if ref_status == "positive" else
                                         "The fixed reappearance-only gate does not improve both combined HOTA and IDF1; retain raw for this pooled feature and test a spatial-token/track-level target."),
                      "no_eval_threshold_tuning": True, "no_posthoc_track_repair": True}
        dump(out / "reflection.json", reflection)
        payload["status"] = "complete"; payload["trackeval_status"] = audit.get("status"); payload["elapsed_seconds"] = time.perf_counter() - t0
        dump(out / "results.json", payload)
        prov_dir = out / "provenance"; prov_dir.mkdir(exist_ok=True)
        for src in (Path(__file__).resolve(), HERE / "mot_dance_jepa_got.py", HERE / "mot_dance_jepa_real.py", HERE / "audit_trackeval_real.py"):
            shutil.copy2(src, prov_dir / src.name)
        dump(out / "provenance.json", {"schema": "dance-got-jepa-reappearance-provenance-v1",
                                       "results_sha256": sha256(out / "results.json"),
                                       "diagnostic_sha256": sha256(out / "train_only_diagnostic.json"),
                                       "protocol_sha256": sha256(out / "protocol.json"),
                                       "summary_sha256": sha256(out / "summary.json"), "reflection_sha256": sha256(out / "reflection.json"),
                                       "trackeval_audit_sha256": sha256(out / "trackeval_official" / "audit.json"),
                                       "checkpoint_sha256": payload["checkpoint_sha256"], "encoder_retrained": False,
                                       "eval_gt_used_for_threshold": False, "no_posthoc_track_repair": True,
                                       "source_files": {p.name: sha256(prov_dir / p.name) for p in (Path(__file__).resolve(), HERE / "mot_dance_jepa_got.py", HERE / "mot_dance_jepa_real.py", HERE / "audit_trackeval_real.py")}})
        print(json.dumps({"status": "complete", "out": str(out), "combined": summary.get("combined"),
                          "reflection": ref_status, "train_gap_events": train_diag["event_count"]}, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        payload["status"] = "failure"; payload["error_type"] = type(exc).__name__; payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(); payload["elapsed_seconds"] = time.perf_counter() - t0
        dump(out / "failure.json", payload); dump(out / "results.json", payload)
        print(json.dumps({"status": "failure", "out": str(out), "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
