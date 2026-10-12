#!/usr/bin/env python3
"""Explain why the ambiguity-only GOT-JEPA events did (or did not) fire.

This is a read-only replay audit for
``mot_dance_jepa_got_ambiguity.py``.  It uses the saved pooled feature cache
and detector streams, replays the deterministic tracker, and classifies every
eligible top-two event (row unmatched, preferred candidate occupied, rejected
swap, or applied).  It never loads validation annotations and never invokes
TrackEval.  Training annotations are used only for the separately reported
train-suffix correctness audit, not to alter a threshold.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mot_dance_jepa_got_ambiguity as amb  # noqa: E402
import mot_dance_jepa_real as dance  # noqa: E402


METHODS = ("raw", "ambiguity_jepa", "no_teacher", "random_frozen")
PRED_METHODS = ("ambiguity_jepa", "no_teacher", "random_frozen")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


def load_predictors(result_payload: Dict[str, object], cache_root: Path):
    """Recreate the exact tiny predictors from train-only cached samples."""
    args_seed = int(result_payload.get("seed", 17)); max_frames = int(result_payload.get("max_frames", 120))
    epochs = int(result_payload.get("dance", {}).get("predictor_stats", {}).get("ambiguity_jepa", {}).get("epochs", 8))
    train_names = list(result_payload.get("dance", {}).get("train_sequences", ["dancetrack0001", "dancetrack0002", "dancetrack0006"]))
    checkpoint_sha = str(result_payload["checkpoint_sha256"])
    train_gt = dance.limit_frame_dicts(dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/train.json", "train", train_names), max_frames)
    train_items = []
    for seq in train_names:
        item, _ = amb.load_train_item(seq, max_frames, cache_root, checkpoint_sha, train_gt)
        train_items.append(item)
    samples, _ = amb.got.build_temporal_samples_got(train_items, max_frames, args_seed, int(result_payload.get("dance", {}).get("train_samples", {}).get("x", [0])[0] or 12000))
    predictors = {"raw": None}
    for method, mode in (("ambiguity_jepa", "jepa_temporal"), ("no_teacher", "no_teacher")):
        p = dance.TemporalPredictor(256, mode, args_seed, epochs, "cuda")
        p.fit(samples); predictors[method] = p
    p = dance.TemporalPredictor(256, "random_frozen", args_seed, 0, "cpu"); p.fit_random(); predictors["random_frozen"] = p
    return train_items, predictors


def snapshot(tracker: amb.AmbiguityMOT, frame: int, dets, predictor):
    """Mirror the pre-assignment part of AmbiguityMOT.step for replay."""
    active, native, pred_boxes = amb.native_cost_matrix(tracker, frame, dets)
    n, m = native.shape
    pred_cache = {}; gate_cache = {}
    jcost = np.full((n, m), 1e6, dtype=np.float32)
    if predictor is not None:
        for i, tr in enumerate(tracker.tracks):
            pp = predictor.predict(tracker._temporal_tokens(tr, frame))
            pred_cache[i] = pp; gate_cache[i] = amb.gate_for(tr, pp)
            for j, d in enumerate(active):
                jcost[i, j] = amb.predictor_cost(tracker, tr, pp, d, pred_boxes[i])
    native_matches = []
    if n and m:
        rr, cc = linear_sum_assignment(native)
        native_matches = [(i, j) for i, j in zip(rr.tolist(), cc.tolist()) if native[i, j] < amb.MAX_ASSIGN_COST]
    native_by_row = {i: j for i, j in native_matches}; native_by_det = {j: i for i, j in native_matches}
    tie_rows = {}; events = []
    for i in range(n):
        finite = np.where(np.isfinite(native[i]) & (native[i] < 1e5))[0]
        if finite.size < 2:
            continue
        order = finite[np.argsort(native[i, finite], kind="mergesort")]
        j1, j2 = int(order[0]), int(order[1]); margin = float(native[i, j2] - native[i, j1])
        ambiguous = bool(margin <= tracker.margin_threshold)
        gate_ok, q, up = gate_cache.get(i, (False, 0.0, 1.0))
        both = bool(native[i, j1] < amb.MAX_ASSIGN_COST and native[i, j2] < amb.MAX_ASSIGN_COST)
        pref = j1; jdelta = None
        if i in pred_cache and np.isfinite(jcost[i, j1]) and np.isfinite(jcost[i, j2]):
            jdelta = float(jcost[i, j1] - jcost[i, j2])
            if jcost[i, j2] + 1e-7 < jcost[i, j1]:
                pref = j2
        eligible = bool(predictor is not None and ambiguous and gate_ok and both and pref != j1)
        if eligible:
            tie_rows[i] = {"top1": j1, "top2": j2, "preferred": pref}
        events.append({"frame": int(frame), "track_index": int(i), "track_id": int(tracker.tracks[i].tid),
                       "top1": j1, "top2": j2, "native_top1": float(native[i, j1]),
                       "native_top2": float(native[i, j2]), "native_margin": margin,
                       "ambiguous": ambiguous, "both_viable": both, "quality": q,
                       "uncertainty_prob": up, "gate_pass": bool(gate_ok),
                       "predictor_preferred": int(pref), "predictor_delta_top1_minus_top2": jdelta,
                       "tie_eligible": eligible})
    return {"active": active, "native": native, "pred_boxes": pred_boxes, "jcost": jcost,
            "native_matches": native_matches, "native_by_row": native_by_row,
            "native_by_det": native_by_det, "tie_rows": tie_rows, "events": events}


def classify(ctx: Dict[str, object], predictor_present: bool):
    """Simulate the exact conservative proposal loop and give each reason."""
    native = ctx["native"]; jcost = ctx["jcost"]; tie_rows = ctx["tie_rows"]
    matched_by_row = dict(ctx["native_by_row"]); matched_by_det = dict(ctx["native_by_det"])
    reasons = collections.Counter(); statuses = {}
    # Non-proposal eligible rows can be explained before sorting proposals.
    proposals = []
    for i, info in tie_rows.items():
        cur = matched_by_row.get(i); pref = int(info["preferred"])
        if cur is None:
            reasons["native_row_unmatched"] += 1; statuses[i] = "native_row_unmatched"; continue
        if pref == cur:
            reasons["already_native_assignment_preferred"] += 1; statuses[i] = "already_native_assignment_preferred"; continue
        gain = float(jcost[i, cur] - jcost[i, pref]) if jcost.shape[1] else 0.0
        proposals.append((-gain, int(i), int(pref), int(cur)))
    proposals.sort()
    for _, i, pref, cur in proposals:
        if matched_by_row.get(i) != cur:
            reasons["superseded_by_prior_tie"] += 1; statuses[i] = "superseded_by_prior_tie"; continue
        if native[i, pref] >= amb.MAX_ASSIGN_COST or not np.isfinite(jcost[i, pref]):
            reasons["preferred_candidate_not_viable"] += 1; statuses[i] = "preferred_candidate_not_viable"; continue
        owner = matched_by_det.get(pref)
        if owner is None:
            matched_by_det.pop(cur, None); matched_by_det[pref] = i; matched_by_row[i] = pref
            reasons["applied_free_preferred"] += 1; statuses[i] = "applied_free_preferred"; continue
        other_info = tie_rows.get(owner); other_cur = matched_by_row.get(owner)
        if other_info is None or other_cur is None:
            reasons["preferred_candidate_occupied_non_tie"] += 1; statuses[i] = "preferred_candidate_occupied_non_tie"; continue
        other_top = {int(other_info["top1"]), int(other_info["top2"])}
        if cur not in other_top or not np.isfinite(jcost[owner, cur]):
            reasons["occupied_tie_row_current_not_top2"] += 1; statuses[i] = "occupied_tie_row_current_not_top2"; continue
        if native[owner, cur] >= amb.MAX_ASSIGN_COST or not np.isfinite(jcost[owner, cur]):
            reasons["occupied_tie_row_native_not_viable"] += 1; statuses[i] = "occupied_tie_row_native_not_viable"; continue
        before = float(jcost[i, cur] + jcost[owner, pref]); after = float(jcost[i, pref] + jcost[owner, cur])
        if after + 1e-7 >= before:
            reasons["two_way_swap_rejected_jepa_not_better"] += 1; statuses[i] = "two_way_swap_rejected_jepa_not_better"; continue
        matched_by_row[i], matched_by_row[owner] = pref, cur
        matched_by_det[pref], matched_by_det[cur] = i, owner
        reasons["applied_two_way_swap"] += 1; statuses[i] = "applied_two_way_swap"; statuses[owner] = "applied_two_way_swap_peer"
    # A defensive consistency marker; the replay caller compares this map to
    # the actual AmbiguityMOT assignment trace after step().
    return matched_by_row, reasons, statuses


def reset_tracker(tracker: amb.AmbiguityMOT) -> None:
    tracker.outputs = {}; tracker.assignment_pairs = {}
    tracker.cost_margins = []; tracker.quality_values = []; tracker.uncertainty_values = []
    tracker.memory_gate_accept = 0; tracker.memory_gate_reject = 0; tracker.temporal_prediction_count = 0
    tracker._reset_audit()


def replay_eval(result_payload: Dict[str, object], result_dir: Path, eval_names: Sequence[str], predictors, threshold: float, cache_root: Path):
    summaries = {}; hash_checks = {}
    for seq in eval_names:
        seq_dir, det, dm = amb.load_sequence_detector(seq, "val", int(result_payload["max_frames"]), cache_root, result_payload["checkpoint_sha256"])
        summaries[seq] = {}; hash_checks[seq] = {}
        iw, ih = dance.sequence_wh(seq_dir)
        for method in METHODS:
            tr = amb.AmbiguityMOT(predictors.get(method), method, threshold, int(result_payload["seed"]), iw, ih)
            reset_tracker(tr); reason_counts = collections.Counter(); status_counts = collections.Counter(); replay_event_rows = []
            assignment_checks = []; native_checks = []
            for fr in sorted(det):
                ctx = snapshot(tr, int(fr), det[fr], predictors.get(method))
                expected_map, rs, sts = classify(ctx, predictors.get(method) is not None)
                reason_counts.update(rs); status_counts.update(sts.values()); replay_event_rows.extend(ctx["events"])
                expected_used = sorted((int(tr.tracks[i].tid), int(j)) for i, j in expected_map.items())
                expected_native = sorted((int(tr.tracks[i].tid), int(j)) for i, j in ctx["native_matches"])
                tr.step(int(fr), det[fr])
                actual_used = sorted(tr.used_assignment_pairs.get(int(fr), [])); actual_native = sorted(tr.native_assignment_pairs.get(int(fr), []))
                assignment_checks.append(bool(actual_used == expected_used)); native_checks.append(bool(actual_native == expected_native))
            result = tr.outputs
            saved = result_payload["dance"]["sequences"][seq][method]
            replay_hash = amb.tracks_hash(result); saved_hash = saved.get("semantic_sha256")
            event_hash = amb.hash_json(replay_event_rows); saved_event_hash = saved.get("event_trace_sha256")
            hash_checks[seq][method] = {
                "track_semantic_hash_matches": bool(replay_hash == saved_hash), "replay_semantic_sha256": replay_hash,
                "saved_semantic_sha256": saved_hash, "event_trace_hash_matches": bool(event_hash == saved_event_hash),
                "replay_event_trace_sha256": event_hash, "saved_event_trace_sha256": saved_event_hash,
                "native_assignment_trace_matches": bool(all(native_checks)), "used_assignment_trace_matches": bool(all(assignment_checks)),
                "track_rows_replay": len(dance.output_rows(result)), "track_rows_saved": int(saved.get("track_rows", -1)),
            }
            summaries[seq][method] = {"reason_counts": dict(reason_counts), "status_counts": dict(status_counts),
                                      "tie_eligible_events_replay": int(sum(reason_counts.values())),
                                      "tie_applied_events_replay": int(reason_counts.get("applied_free_preferred", 0) + reason_counts.get("applied_two_way_swap", 0)),
                                      "saved_tie_eligible_events": int(saved.get("tie_eligible_events", -1)),
                                      "saved_tie_applied_events": int(saved.get("tie_applied_events", -1)),
                                      "saved_assignment_changed_rows": int(saved.get("assignment_changed_rows", -1))}
    return summaries, hash_checks


def detector_gid_map(gt_rows, active):
    """One-to-one IoU=.5 map used only by the train correctness audit."""
    out = {}
    if not gt_rows or not active:
        return out
    mat = np.asarray([[dance.box_iou(g.box, d.box) for d in active] for g in gt_rows], dtype=np.float32)
    rr, cc = linear_sum_assignment(-mat)
    for i, j in zip(rr.tolist(), cc.tolist()):
        if float(mat[i, j]) >= 0.5:
            out[int(j)] = int(gt_rows[i].gid)
    return out


def train_correctness(train_items, train_diag: Dict[str, object], threshold: float):
    """Compare native top1 and JEPA-preferred top2 against train GT IDs."""
    rows = train_diag.get("event_rows", []); by_key = {(str(r["sequence"]), int(r["frame"]), int(r["track_id"])): r for r in rows}
    counts = {m: collections.Counter() for m in PRED_METHODS}
    # Replay raw state and retain a detector-index -> GT-ID map per frame.
    for seq, seq_dir, gt, det, teacher in train_items:
        iw, ih = dance.sequence_wh(seq_dir); probe = dance.SimpleMOT("raw", None, seed=17, image_width=iw, image_height=ih)
        tid_gid = {}
        for fr in sorted(det):
            active, native, _ = amb.native_cost_matrix(probe, int(fr), det[fr])
            gids = detector_gid_map(gt.get(int(fr), []), active)
            if int(fr) >= int(train_diag.get("suffix_start_frame", 61)):
                for i, tr in enumerate(probe.tracks):
                    row = by_key.get((seq, int(fr), int(tr.tid)))
                    if row is None:
                        continue
                    target_gid = tid_gid.get(int(tr.tid))
                    # If the track has no previous label, infer it from its
                    # current observable box solely for this train audit.
                    if target_gid is None and gt.get(int(fr)):
                        vals = [(dance.box_iou(g.box, tr.box), int(g.gid)) for g in gt[int(fr)]]
                        if vals and max(vals)[0] >= 0.5:
                            target_gid = max(vals)[1]
                    for method in PRED_METHODS:
                        x = row[method]
                        st = counts[method]; st["rows"] += 1
                        # The saved train diagnostic predates the replay
                        # audit's nested ``both_viable`` field.  Recompute
                        # that purely native condition from its recorded
                        # top-two costs rather than silently dropping every
                        # eligible event.
                        both_viable = bool(float(row["native_top1"]) < amb.MAX_ASSIGN_COST and
                                           float(row["native_top2"]) < amb.MAX_ASSIGN_COST)
                        if not (float(row["native_margin"]) <= threshold and bool(x.get("gate_pass")) and both_viable and bool(x.get("prefers_top2"))):
                            continue
                        st["eligible"] += 1
                        j1, j2 = int(row["top1"]), int(row["top2"])
                        g1, g2 = gids.get(j1), gids.get(j2)
                        n_ok = bool(target_gid is not None and g1 == target_gid)
                        j_ok = bool(target_gid is not None and g2 == target_gid)
                        st["label_available"] += int(target_gid is not None)
                        st["native_top1_correct"] += int(n_ok); st["jepa_top2_correct"] += int(j_ok)
                        st["improvement"] += int((not n_ok) and j_ok); st["regression"] += int(n_ok and (not j_ok))
                        st["same_correctness"] += int(n_ok == j_ok)
            # Advance native tracker and update the observable track->GT map.
            native_pairs = []
            if native.size:
                rr, cc = linear_sum_assignment(native)
                native_pairs = [(i, j) for i, j in zip(rr.tolist(), cc.tolist()) if native[i, j] < amb.MAX_ASSIGN_COST]
            for i, j in native_pairs:
                if j in gids:
                    tid_gid[int(probe.tracks[i].tid)] = gids[j]
            before_ids = set(t.tid for t in probe.tracks)
            probe.step(int(fr), det[fr])
            # Spawned tracks are recorded in assignment_pairs; map only those
            # with a training detector label.
            for tid, j in probe.assignment_pairs.get(int(fr), []):
                if int(tid) not in before_ids and int(j) in gids:
                    tid_gid[int(tid)] = gids[int(j)]
        for method, st in counts.items():
            st["correctness_gain"] = int(st["improvement"] - st["regression"])
            st["native_accuracy"] = float(st["native_top1_correct"] / max(st["label_available"], 1))
            st["jepa_accuracy"] = float(st["jepa_top2_correct"] / max(st["label_available"], 1))
    return {m: dict(c) for m, c in counts.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=amb.CACHE_DEFAULT)
    args = ap.parse_args()
    result_dir = Path(args.result_dir).resolve(); out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    payload = json.loads((result_dir / "results.json").read_text())
    base = {"schema": "dance-got-jepa-ambiguity-event-reason-audit-v1", "status": "started",
            "result_dir": str(result_dir), "result_sha256": sha256(result_dir / "results.json"),
            "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
            "platform": platform.platform(), "eval_gt_read": False, "trackeval_invoked": False,
            "train_gt_read_for_correctness_only": True}
    try:
        threshold = float(payload["margin_threshold"]); cache_root = Path(args.cache_dir).resolve()
        train_items, predictors = load_predictors(payload, cache_root)
        eval_names = list(payload["dance"]["eval_sequences"])
        reason_summary, hash_checks = replay_eval(payload, result_dir, eval_names, predictors, threshold, cache_root)
        diag = json.loads((result_dir / "train_only_diagnostic.json").read_text())
        train_corr = train_correctness(train_items, diag, threshold)
        all_hash_pass = all(vv for seq in hash_checks.values() for method in seq.values() for vv in (
            method["track_semantic_hash_matches"], method["event_trace_hash_matches"],
            method["native_assignment_trace_matches"], method["used_assignment_trace_matches"]))
        result = dict(base); result.update({"status": "complete", "threshold": threshold,
                       "threshold_source": "saved train suffix p25; not recomputed from eval",
                       "reason_summary": reason_summary, "replay_hash_checks": hash_checks,
                       "replay_all_hash_checks_pass": bool(all_hash_pass),
                       "train_suffix_correctness": train_corr,
                       "train_diagnostic_sha256": sha256(result_dir / "train_only_diagnostic.json"),
                       "elapsed_seconds": time.perf_counter() - t0,
                       "interpretation": "Classifies every eligible event without changing saved tracks or invoking TrackEval."})
        dump(out / "event_reason_audit.json", result)
        # Reflection is deliberately descriptive; it cannot alter a metric or
        # trigger a second eval run.
        cand = reason_summary
        aggregate = collections.Counter()
        for seq in cand.values():
            aggregate.update(seq.get("ambiguity_jepa", {}).get("reason_counts", {}))
        reflection = {
            "schema": "dance-got-jepa-ambiguity-event-reflection-v1", "status": "complete",
            "candidate": "ambiguity_jepa", "eligible_events": int(sum(aggregate.values())),
            "reason_counts": dict(aggregate), "track_replay_hashes_verified": bool(all_hash_pass),
            "train_correctness": train_corr.get("ambiguity_jepa"),
            "conclusion": ("No candidate assignment changed because eligible proposals were blocked by native matching/occupancy; inspect reason counts before changing the fusion rule." if aggregate.get("applied_free_preferred", 0) + aggregate.get("applied_two_way_swap", 0) == 0 else
                           "At least one registered tie was applied; use the train-only correctness table to decide whether a pair-swap residual is warranted."),
            "next_candidate": "If train-only JEPA top2 correctness exceeds native top1 on eligible events, preregister a bounded candidate-independent pair-swap residual; otherwise stop pooled-256D fusion and test spatial-token/track-level targets.",
            "no_eval_gt_or_trackeval": True,
        }
        dump(out / "event_reason_reflection.json", reflection)
        dump(out / "provenance.json", {"schema": "dance-got-jepa-ambiguity-event-audit-provenance-v1",
                                       "audit_sha256": sha256(out / "event_reason_audit.json"),
                                       "reflection_sha256": sha256(out / "event_reason_reflection.json"),
                                       "result_sha256": base["result_sha256"], "eval_gt_read": False,
                                       "trackeval_invoked": False, "checkpoint_sha256": payload.get("checkpoint_sha256"),
                                       "source_script_sha256": sha256(HERE / Path(__file__).name)})
        print(json.dumps({"status": "complete", "out": str(out), "replay_all_hash_checks_pass": all_hash_pass,
                          "candidate_reason_counts": dict(aggregate), "train_correctness": train_corr.get("ambiguity_jepa")}, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        dump(out / "event_reason_audit.json", {**base, "status": "failure", "error": repr(exc), "elapsed_seconds": time.perf_counter() - t0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
