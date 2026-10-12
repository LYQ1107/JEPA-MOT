#!/usr/bin/env python3
"""No-GT lifecycle/assignment preflight for the reappearance branch.

The preflight runs the corrected reappearance tracker on detector/cache data,
compares its raw control with the plain ``SimpleMOT`` raw semantics, and
checks that non-JEPA rows use the exact raw memory update.  It never reads
validation annotations and never invokes TrackEval; official evaluation is
launched only after ``preflight_pass`` is true.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mot_dance_jepa_got_reappearance as run  # noqa: E402
import mot_dance_jepa_got_ambiguity as amb  # noqa: E402
import mot_dance_jepa_real as dance  # noqa: E402


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=float) + "\n")


def reset_tracker(tr: run.ReappearanceMOT) -> None:
    tr.outputs = {}; tr.assignment_pairs = {}
    tr.cost_margins = []; tr.quality_values = []; tr.uncertainty_values = []
    tr.memory_gate_accept = 0; tr.memory_gate_reject = 0; tr.temporal_prediction_count = 0
    tr._reset_audit()


def changed_rows_by_frame(tr: run.ReappearanceMOT):
    out = collections.defaultdict(list)
    for fr in sorted(set(tr.native_assignment_pairs) | set(tr.used_assignment_pairs)):
        native = {tid: j for tid, j in tr.native_assignment_pairs.get(fr, [])}
        used = {tid: j for tid, j in tr.used_assignment_pairs.get(fr, [])}
        for tid in sorted(set(native) | set(used)):
            if native.get(tid) != used.get(tid):
                out[int(fr)].append(int(tid))
    return out


def audit_tracker(seq: str, seq_dir: Path, det, predictors, out: Path, seed: int):
    iw, ih = dance.sequence_wh(seq_dir)
    method_rows = {}
    for method in run.METHODS:
        tr = run.ReappearanceMOT(predictors.get(method), method, seed, iw, ih)
        reset_tracker(tr); result = tr.run(det)
        # Plain raw shadow is used only for the raw-control identity check.
        raw_shadow = dance.SimpleMOT("raw", None, seed=seed, image_width=iw, image_height=ih)
        raw_result = raw_shadow.run(det)
        raw_equal = bool(run.tracks_hash(result) == run.tracks_hash(raw_result)) if method == "raw" else None
        lifecycle_fail = []
        for row in tr.lifecycle_rows:
            score = float(row["score"]); use = bool(row["jepa_used"])
            if not use:
                if row.get("memory_source") != "raw_exact" or not bool(row.get("update_ok")) or bool(row.get("missing")) != bool(score < .35):
                    lifecycle_fail.append({"kind": "non_jepa_not_raw_exact", **row})
            else:
                if row.get("memory_source") != "jepa_gate" or not bool(row.get("gate_pass")):
                    lifecycle_fail.append({"kind": "jepa_gate_contract", **row})
                if bool(row.get("update_ok")) and not bool(row.get("legal_raw_observation")):
                    lifecycle_fail.append({"kind": "illegal_jepa_memory_update", **row})
                if not bool(row.get("update_ok")) and not bool(row.get("missing")):
                    lifecycle_fail.append({"kind": "rejected_jepa_not_marked_missing", **row})
        event_map = {(int(e["frame"]), int(e["track_id"])): e for e in tr.event_rows}
        event_fail = []
        for e in tr.event_rows:
            if int(e["misses_before"]) == 0 and bool(e["jepa_used"]):
                event_fail.append({"kind": "continuous_row_used_jepa", **e})
            if bool(e["jepa_used"]) and not (bool(e["gap_ok"]) and bool(e["gate_pass"])):
                event_fail.append({"kind": "jepa_used_without_gap_gate", **e})
        changed = changed_rows_by_frame(tr)
        direct = 0; cascade = 0; changed_fail = []
        for fr, tids in changed.items():
            frame_events = [e for e in tr.event_rows if int(e["frame"]) == int(fr)]
            any_jepa = any(bool(e["jepa_used"]) for e in frame_events)
            for tid in tids:
                e = event_map.get((int(fr), int(tid)))
                if e is not None and bool(e.get("jepa_used")):
                    direct += 1
                elif any_jepa:
                    cascade += 1
                else:
                    changed_fail.append({"frame": fr, "track_id": tid, "kind": "change_without_jepa_row"})
        # If a frame has no active reappearance row, used/native assignments
        # must be identical.  This catches accidental lifecycle drift.
        for fr in sorted(set(tr.native_assignment_pairs) | set(tr.used_assignment_pairs)):
            frame_events = [e for e in tr.event_rows if int(e["frame"]) == int(fr)]
            if not any(bool(e.get("jepa_used")) for e in frame_events):
                n = sorted(tr.native_assignment_pairs.get(fr, [])); u = sorted(tr.used_assignment_pairs.get(fr, []))
                if n != u:
                    changed_fail.append({"frame": fr, "kind": "native_used_mismatch_without_jepa"})
        trace_dir = out / "event_traces"
        run.write_event_traces(out, seq + "__preflight", method, tr.event_rows)
        run.write_lifecycle_traces(out, seq + "__preflight", method, tr.lifecycle_rows)
        method_rows[method] = {
            "track_semantic_sha256": run.tracks_hash(result), "track_rows": len(dance.output_rows(result)),
            "raw_shadow_semantic_sha256": run.tracks_hash(raw_result), "raw_control_exact": raw_equal,
            "audit_stats": tr.audit_stats(), "lifecycle_contract_failures": lifecycle_fail,
            "event_contract_failures": event_fail, "assignment_changed_direct": direct,
            "assignment_changed_global_cascade": cascade, "assignment_change_failures": changed_fail,
            "preflight_method_pass": bool(not lifecycle_fail and not event_fail and not changed_fail and (raw_equal is not False)),
        }
    return method_rows


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=run.CHECKPOINT_DEFAULT)
    ap.add_argument("--cache-dir", default=run.CACHE_DEFAULT)
    ap.add_argument("--dance-train", default=run.TRAIN_DEFAULT)
    ap.add_argument("--dance-eval", default=run.EVAL_DEFAULT)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--max-samples", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reference-raw-dir", default=None,
                    help="optional prior invariant run whose raw semantic hashes must match")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve(); pre = out / "preflight"; pre.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve(); cache_root = Path(args.cache_dir).resolve(); t0 = time.perf_counter()
    payload = {"seed": args.seed, "max_frames": args.max_frames, "max_samples": args.max_samples,
               "epochs": args.epochs, "dance_train": args.dance_train, "dance_eval": args.dance_eval,
               "checkpoint_sha256": sha256(checkpoint)}
    base = {"schema": "dance-got-jepa-reappearance-invariant-preflight-v1", "status": "started",
            "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
            "platform": platform.platform(), "out_dir": str(out), "checkpoint": str(checkpoint),
            "checkpoint_sha256": payload["checkpoint_sha256"], "cache_dir": str(cache_root),
            "eval_gt_read": False, "trackeval_invoked": False, "seed": args.seed, "max_frames": args.max_frames}
    try:
        # The protocol is fixed apart from the explicitly requested audit
        # replication seeds.  Keeping this allow-list prevents accidental
        # parameter sweeps while permitting the second independent seed.
        if args.seed not in (17, 23) or args.max_frames != 120 or args.epochs != 8:
            raise ValueError("preflight fixed to seed17 or seed23/max_frames120/epochs8")
        run.amb.seed_all(args.seed)
        train_items, predictors, feature_meta, sample_meta, samples = run.load_train_predictors(payload, cache_root, args.device)
        train_diag = run.collect_train_diagnostic(train_items, {m: predictors[m] for m in run.PREDICTOR_METHODS}, args.max_frames, args.seed)
        dump(pre / "train_only_diagnostic.json", train_diag)
        protocol = {"schema": "dance-got-jepa-reappearance-invariant-preflight-protocol-v1",
                    "gap_rule": "misses>=1 and frame_gap==misses and gap<=4",
                    "continuous_visible": "native raw appearance/geometry and raw memory update exactly",
                    "predictor_gate": "quality>=0.35, sigmoid(log_variance)<=0.65",
                    "candidate_independent": True, "no_posthoc_repair": True,
                    "eval_gt_read": False, "trackeval_invoked": False,
                    "train_only_diagnostic_sha256": sha256(pre / "train_only_diagnostic.json")}
        dump(pre / "protocol.json", protocol)
        # A second-seed replication must use exactly the same raw lifecycle;
        # load only the prior report (never images or annotations) to compare
        # its semantic hashes after the fresh no-GT replay.
        reference_raw = {}
        if args.reference_raw_dir:
            ref = Path(args.reference_raw_dir).resolve() / "preflight" / "invariant_audit.json"
            if not ref.exists():
                raise FileNotFoundError(f"reference raw audit missing: {ref}")
            ref_report = json.loads(ref.read_text())
            for seq, vals in (ref_report.get("eval", {}) or {}).items():
                raw_rec = (vals or {}).get("raw", {}) or {}
                if raw_rec.get("raw_shadow_semantic_sha256"):
                    reference_raw[str(seq)] = str(raw_rec["raw_shadow_semantic_sha256"])
        base["reference_raw_dir"] = str(Path(args.reference_raw_dir).resolve()) if args.reference_raw_dir else None
        base["reference_raw_semantic_sha256"] = reference_raw
        eval_rows = {}; all_pass = True
        for seq in run.list_names(args.dance_eval):
            seq_dir, det, _ = amb.load_sequence_detector(seq, "val", args.max_frames, cache_root, payload["checkpoint_sha256"])
            eval_rows[seq] = audit_tracker(seq, seq_dir, det, predictors, pre, args.seed)
            if reference_raw:
                observed = (eval_rows[seq].get("raw", {}) or {}).get("raw_shadow_semantic_sha256")
                match = bool(observed == reference_raw.get(seq))
                eval_rows[seq]["raw"]["raw_shadow_matches_reference"] = match
                if not match:
                    eval_rows[seq]["raw"]["preflight_method_pass"] = False
            all_pass = all_pass and all(v["preflight_method_pass"] for v in eval_rows[seq].values())
        report = dict(base); report.update({"status": "complete", "preflight_pass": bool(all_pass),
                                            "train_only_diagnostic": {k: v for k, v in train_diag.items() if k != "events"},
                                            "train_event_count": train_diag["event_count"], "eval": eval_rows,
                                            "elapsed_seconds": time.perf_counter() - t0,
                                            "next_action": "run official TrackEval only if preflight_pass=true"})
        dump(pre / "invariant_audit.json", report)
        dump(pre / "provenance.json", {"schema": "dance-got-jepa-reappearance-invariant-preflight-provenance-v1",
                                        "audit_sha256": sha256(pre / "invariant_audit.json"),
                                        "protocol_sha256": sha256(pre / "protocol.json"),
                                        "checkpoint_sha256": payload["checkpoint_sha256"],
                                        "source_script_sha256": sha256(HERE / "mot_dance_jepa_got_reappearance.py"),
                                        "eval_gt_read": False, "trackeval_invoked": False,
                                        "reference_raw_dir": base.get("reference_raw_dir"),
                                        "reference_raw_semantic_sha256": base.get("reference_raw_semantic_sha256", {})})
        print(json.dumps({"status": "complete", "preflight_pass": all_pass, "out": str(pre),
                          "train_event_count": train_diag["event_count"],
                          "eval_methods": {s: {m: r["preflight_method_pass"] for m, r in x.items()} for s, x in eval_rows.items()}}, indent=2, sort_keys=True, default=float))
        return 0 if all_pass else 2
    except Exception as exc:
        dump(pre / "invariant_audit.json", {**base, "status": "failure", "error": repr(exc), "elapsed_seconds": time.perf_counter() - t0})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
