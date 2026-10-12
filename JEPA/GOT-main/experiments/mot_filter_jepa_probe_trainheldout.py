#!/usr/bin/env python3
"""Train-only held-out gate for the lagged ridge-filter JEPA probe.

This wrapper deliberately never asks the MOT loader for evaluation sequences.
It builds the existing oracle replay records from MOT17 train sequences, fits the
student only on the first (temporal) portion, and measures the frozen ridge
teacher and controls on the last portion at t+1.  The output is a diagnostic
quality gate, not an online tracker result.  The imported probe implementation
is hashed in the artifact so that a later source edit cannot silently make the
numbers look reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mot_filter_jepa_probe as probe  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_temporal(records: Sequence[dict], fraction: float) -> Tuple[List[dict], List[dict], dict]:
    """Split each sequence by frame, never by random rows.

    A frame and all identities in that frame stay in one side.  Thus the
    held-out gate is genuinely after the fit frames and cannot be changed by a
    random seed.  The one-frame edge case is rejected rather than silently
    reporting a fit-on-test score.
    """
    fit: List[dict] = []
    hold: List[dict] = []
    detail: Dict[str, object] = {}
    for seq in sorted({str(r["sequence"]) for r in records}):
        rows = [r for r in records if str(r["sequence"]) == seq]
        frames = sorted({int(r["frame"]) for r in rows})
        if len(frames) < 2:
            raise RuntimeError(f"sequence {seq} has <2 lagged frames")
        cut = int(np.floor(len(frames) * float(fraction)))
        cut = min(max(cut, 1), len(frames) - 1)
        fit_frames = set(frames[:cut])
        hold_frames = set(frames[cut:])
        assert fit_frames and hold_frames and max(fit_frames) < min(hold_frames)
        fit.extend(r for r in rows if int(r["frame"]) in fit_frames)
        hold.extend(r for r in rows if int(r["frame"]) in hold_frames)
        detail[seq] = {
            "all_frames": frames,
            "fit_frames": sorted(fit_frames),
            "holdout_frames": sorted(hold_frames),
            "fit_samples": int(sum(int(r["frame"]) in fit_frames for r in rows)),
            "holdout_samples": int(sum(int(r["frame"]) in hold_frames for r in rows)),
        }
    if not fit or not hold:
        raise RuntimeError("empty train-fit or train-heldout split")
    return fit, hold, detail


def _metric(fn, rows):
    out, arr = probe.metrics(fn, rows)
    if not bool(out.get("finite", False)):
        raise FloatingPointError("non-finite held-out metric")
    if not bool(out.get("candidate_order_invariant", False)):
        raise AssertionError("candidate permutation audit failed")
    return out, arr


def run(args: argparse.Namespace) -> dict:
    # Passing an empty eval list is intentional: load_items then does not open
    # any MOT17 eval gt/det file.  This is the key distinction from the old
    # probe's eval `teacher_t1_validation` oracle diagnostic.
    loader_args = SimpleNamespace(
        mot_train=args.mot_train,
        mot_eval="",
        max_frames=args.max_frames,
    )
    train_items, eval_items, detector_meta = probe.base.load_items("mot17", loader_args)
    if eval_items:
        raise AssertionError("train-only wrapper unexpectedly loaded eval items")
    arr, records, meta = probe.build_filter_records(
        train_items, args.max_frames, args.seed, args.max_samples
    )
    fit_rows, hold_rows, split = split_temporal(records, args.fit_fraction)
    fit_arr = probe._arrays(fit_rows)
    if int(meta.get("duplicate_count", -1)) != 0:
        raise AssertionError("duplicate one-to-one match in train records")
    for name, value in fit_arr.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite fit field {name}")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    models = {}
    stats = {}
    for method in methods:
        if method in ("raw", "history"):
            continue
        model = probe.FilterStudent(method, args.seed, args.epochs)
        if method == "random_frozen":
            model.fit_random()
        else:
            model.fit(fit_arr)
        models[method] = model
        stats[method] = model.stats

    def predict(method: str, row: dict, clean: bool = False) -> np.ndarray:
        if method == "raw":
            return row["candidate"]
        if method == "history":
            return row["hist"]
        return models[method].predict(row, clean)

    # The teacher is evaluated on t+1 candidates in the held-out frames.  Its
    # clean current crop and GT-derived negatives are used only to construct an
    # offline held-out diagnostic target; no eval split is touched here.
    teacher_holdout, teacher_arr = _metric(
        lambda row, clean=False: row["teacher_w"], hold_rows
    )
    candidate_chance = float(
        np.mean([1.0 / max(1, len(r["next_candidates"])) for r in hold_rows])
    )

    heldout = {}
    arrays = {}
    for method in methods:
        heldout[method], arrays[method] = _metric(
            lambda row, clean=False, mm=method: predict(mm, row, clean), hold_rows
        )

    paired = {}
    if "current_jepa" in arrays:
        for control in ("no_teacher", "random_frozen", "raw", "history"):
            if control in arrays:
                paired["current_jepa_minus_" + control] = {
                    metric: probe.paired(
                        arrays["current_jepa"][metric], arrays[control][metric], args.seed + i
                    )
                    for i, metric in enumerate(("r1", "margin", "cos"))
                }

    # This is intentionally a two-level gate.  A good teacher does not imply a
    # useful student.  The integration gate requires a predeclared positive
    # margin and a meaningful delta above chance; tiny wins over weak controls
    # are not enough to call an online improvement.
    teacher_gate = {
        "finite": bool(teacher_holdout.get("finite", False)),
        "recall_at_1": float(teacher_holdout.get("recall_at_1", float("nan"))),
        "margin_mean": float(teacher_holdout.get("margin_mean", float("nan"))),
        "chance_recall": candidate_chance,
        "min_recall_at_1": float(args.teacher_min_r1),
        "min_delta_over_chance": float(args.teacher_delta),
        "useful": bool(
            teacher_holdout.get("finite", False)
            and teacher_holdout.get("recall_at_1", -1.0) >= args.teacher_min_r1
            and teacher_holdout.get("recall_at_1", -1.0) >= candidate_chance + args.teacher_delta
            and teacher_holdout.get("margin_mean", -1.0) > 0.0
        ),
    }
    student_row = heldout.get("current_jepa", {})
    student_delta = float(student_row.get("recall_at_1", -1.0) - candidate_chance)
    student_gate = {
        "current_r1": float(student_row.get("recall_at_1", float("nan"))),
        "current_margin": float(student_row.get("margin_mean", float("nan"))),
        "delta_over_chance": student_delta,
        "min_delta_over_chance": float(args.student_delta),
        "positive_margin": bool(student_row.get("margin_mean", -1.0) > 0.0),
        "beats_no_teacher": bool(
            "no_teacher" in heldout
            and student_row.get("recall_at_1", -1.0) > heldout["no_teacher"].get("recall_at_1", -1.0)
        ),
        "beats_random": bool(
            "random_frozen" in heldout
            and student_row.get("recall_at_1", -1.0) > heldout["random_frozen"].get("recall_at_1", -1.0)
        ),
    }
    student_gate["useful"] = bool(
        student_gate["positive_margin"]
        and student_delta >= args.student_delta
        and student_gate["beats_no_teacher"]
        and student_gate["beats_random"]
    )

    ridge_diag = {
        key: meta.get(key)
        for key in (
            "ridge_dtype",
            "ridge_condition_number_median",
            "ridge_condition_number_max",
            "ridge_residual_l2_median",
            "ridge_residual_l2_max",
            "ridge_invalid_count",
        )
    }
    numeric_gate = {
        "dtype_float64": ridge_diag.get("ridge_dtype") == "float64",
        "invalid_count_zero": ridge_diag.get("ridge_invalid_count") == 0,
        "condition_max_finite": bool(np.isfinite(ridge_diag.get("ridge_condition_number_max", np.nan))),
        "residual_max_finite": bool(np.isfinite(ridge_diag.get("ridge_residual_l2_max", np.nan))),
        "condition_max_limit": float(args.max_condition),
        "condition_ok": bool(
            np.isfinite(ridge_diag.get("ridge_condition_number_max", np.nan))
            and ridge_diag.get("ridge_condition_number_max", np.inf) <= args.max_condition
        ),
    }
    numeric_gate["pass"] = bool(all(numeric_gate[k] for k in ("dtype_float64", "invalid_count_zero", "condition_max_finite", "residual_max_finite", "condition_ok")))

    return {
        "protocol": {
            "schema": "mot-filter-jepa-trainheldout-v1",
            "teacher": "closed-form dual ridge from clean GT current + same-frame negatives, train-only offline diagnostic",
            "student": "causal history + corrupted detector current predicts frozen teacher filter",
            "lag": 1,
            "score_frame": "t+1 only",
            "history_oracle_for_probe": True,
            "oracle_semantics": "train-heldout oracle-current+oracle-history diagnostic; not online",
            "eval_gt_read": False,
            "eval_items_loaded": 0,
            "quality_gate_split": "within-MOT17-train temporal heldout",
        },
        "detector": detector_meta,
        "train_sequences": [x[0] for x in train_items],
        "eval_sequences": [],
        "train_samples": len(records),
        "train_fit_samples": len(fit_rows),
        "train_holdout_samples": len(hold_rows),
        "train_split": split,
        "train_meta": meta,
        "ridge_diagnostics": ridge_diag,
        "teacher_train_heldout_t1": teacher_holdout,
        "candidate_chance_recall": candidate_chance,
        "predictor_stats": stats,
        "train_holdout_validation": heldout,
        "paired_bootstrap": paired,
        "quality_gate": {
            "teacher": teacher_gate,
            "numeric": numeric_gate,
            "student": student_gate,
            "proceed_lagged_tracker": bool(teacher_gate["useful"] and numeric_gate["pass"] and student_gate["useful"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot-train", default="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-09-FRCNN")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fit-fraction", type=float, default=0.80)
    parser.add_argument("--teacher-min-r1", type=float, default=0.80)
    parser.add_argument("--teacher-delta", type=float, default=0.20)
    parser.add_argument("--student-delta", type=float, default=0.10)
    parser.add_argument("--max-condition", type=float, default=1e8)
    parser.add_argument(
        "--methods",
        default="raw,history,current_jepa,current_only,full_mask,no_teacher,random_frozen,both_clean",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/results/mot_filter_jepa_trainheldout_mot20_v1",
    )
    args = parser.parse_args()
    probe.base.seed_all(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wrapper_sha = sha256_file(Path(__file__))
    base_sha = sha256_file(HERE / "mot_filter_jepa_probe.py")
    payload = {
        "schema": "mot-filter-jepa-trainheldout-v1",
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "generated_unix": time.time(),
        "wrapper_sha256": wrapper_sha,
        "base_probe_sha256": base_sha,
        "seed": args.seed,
        "max_frames": args.max_frames,
        "fit_fraction": args.fit_fraction,
        "eval_gt_read": False,
    }
    try:
        payload["mot17_train_only"] = run(args)
        payload["status"] = "complete"
    except Exception as exc:
        payload["status"] = "failed"
        payload["failure"] = type(exc).__name__ + ": " + str(exc)
        (out / "failure.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    gate = payload["mot17_train_only"]["quality_gate"]
    print(json.dumps({"status": payload["status"], "quality_gate": gate}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

