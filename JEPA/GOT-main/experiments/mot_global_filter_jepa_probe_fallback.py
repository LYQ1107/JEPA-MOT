#!/usr/bin/env python3
"""Minimal train-only frozen-global-filter JEPA smoke.

This is an independent fallback while the per-window ridge probe is being
audited.  It never loads MOT17 evaluation items.  Per-window ridge filters are
used *only as train-fit labels*; a single global ridge mapping from
``[history, clean-current, geometry]`` to that filter is then fit on train-fit
rows, frozen, and applied to train-heldout rows.  Heldout teacher inference uses
the uncorrupted detector crop (not heldout GT), and both teacher/student outputs
are scored only on the next-frame candidate set.  GT history remains an oracle
replay aid and metrics remain diagnostic, not an online tracker claim.
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
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def split_temporal(rows: Sequence[dict], fraction: float = 0.8):
    fit: List[dict] = []
    hold: List[dict] = []
    detail = {}
    for seq in sorted({str(r["sequence"]) for r in rows}):
        rr = [r for r in rows if str(r["sequence"]) == seq]
        frames = sorted({int(r["frame"]) for r in rr})
        if len(frames) < 2:
            raise RuntimeError(f"not enough frames for heldout split: {seq}")
        cut = min(max(int(np.floor(len(frames) * fraction)), 1), len(frames) - 1)
        ff, hh = set(frames[:cut]), set(frames[cut:])
        assert max(ff) < min(hh)
        fit.extend(r for r in rr if int(r["frame"]) in ff)
        hold.extend(r for r in rr if int(r["frame"]) in hh)
        detail[seq] = {
            "fit_frames": sorted(ff),
            "heldout_frames": sorted(hh),
            "fit_samples": int(sum(int(r["frame"]) in ff for r in rr)),
            "heldout_samples": int(sum(int(r["frame"]) in hh for r in rr)),
        }
    if not fit or not hold:
        raise RuntimeError("empty global teacher fit/heldout split")
    return fit, hold, detail


def context(row: dict, current_key: str) -> np.ndarray:
    return np.concatenate(
        [np.asarray(row["hist"], np.float64), np.asarray(row[current_key], np.float64), np.asarray(row["geom"], np.float64)]
    )


def fit_global_mapping(rows: Sequence[dict], lam: float):
    """Fit one frozen global map to train-fit per-window filter labels."""
    x = np.asarray([context(r, "target") for r in rows], np.float64)
    y = probe.base.normalize(np.asarray([r["teacher_w"] for r in rows], np.float64), axis=1).astype(np.float64)
    mu = x.mean(0)
    sd = x.std(0)
    zero = sd < 1e-8
    sd[zero] = 1.0
    xs = (x - mu) / sd
    gram = xs.T @ xs + float(lam) * np.eye(xs.shape[1], dtype=np.float64)
    rhs = xs.T @ y
    cond = float(np.linalg.cond(gram))
    try:
        w = np.linalg.solve(gram, rhs)
        solver = "float64_solve"
    except np.linalg.LinAlgError:
        u, s, vh = np.linalg.svd(gram, full_matrices=False)
        w = (vh.T * (1.0 / np.maximum(s, 1e-12))) @ (u.T @ rhs)
        solver = "float64_svd_fallback"
    residual = float(np.linalg.norm(gram @ w - rhs))
    pred = probe.base.normalize(xs @ w, axis=1)
    y_norm = probe.base.normalize(y, axis=1)
    target_cos = np.sum(pred * y_norm, axis=1)
    centered = y_norm - y_norm.mean(0, keepdims=True)
    sv = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.sum(sv > max(sv[0] if len(sv) else 0.0, 1e-12) * 1e-6)) if len(sv) else 0
    diag = {
        "dtype": "float64",
        "solver": solver,
        "lambda": float(lam),
        "condition_number": cond,
        "residual_l2": residual,
        "finite": bool(np.isfinite(w).all() and np.isfinite(pred).all()),
        "input_dim": int(xs.shape[1]),
        "output_dim": int(w.shape[1]),
        "fit_rows": int(len(rows)),
        "zero_scale_columns": int(zero.sum()),
        "target_mean_cosine_fit": float(target_cos.mean()),
        "target_variance_mean": float(np.var(y_norm, axis=0).mean()),
        "target_centered_rank": rank,
        "parameter_sha256": hashlib.sha256(np.asarray(w, np.float64).tobytes()).hexdigest(),
    }
    return {"w": w, "mu": mu, "sd": sd}, diag


def global_predict(model: dict, row: dict, current_key: str) -> np.ndarray:
    x = (context(row, current_key) - model["mu"]) / model["sd"]
    return probe.base.normalize((x @ model["w"]).astype(np.float32)).astype(np.float32)


def metric(proto_fn, rows):
    out, arr = probe.metrics(proto_fn, rows)
    if not out.get("finite", False) or not out.get("candidate_order_invariant", False):
        raise AssertionError("global probe metric/permutation audit failed")
    return out, arr


def run(args):
    # Empty mot_eval is deliberate and is checked before any record build.
    la = SimpleNamespace(mot_train=args.mot_train, mot_eval="", max_frames=args.max_frames)
    train_items, eval_items, detector_meta = probe.base.load_items("mot17", la)
    if eval_items:
        raise AssertionError("fallback unexpectedly loaded eval items")
    _arr, rows, meta = probe.build_filter_records(train_items, args.max_frames, args.seed, args.max_samples)
    fit_rows, hold_rows, split = split_temporal(rows, args.fit_fraction)
    if int(meta.get("duplicate_count", -1)) != 0:
        raise AssertionError("duplicate assignment")
    global_model, global_diag = fit_global_mapping(fit_rows, args.global_lambda)
    fit_targets = np.asarray([global_predict(global_model, r, "target") for r in fit_rows], np.float32)
    hold_targets = np.asarray([global_predict(global_model, r, "candidate") for r in hold_rows], np.float32)
    target_var = float(np.var(hold_targets, axis=0).mean())
    target_rank = int(np.linalg.matrix_rank(hold_targets - hold_targets.mean(0, keepdims=True))) if len(hold_targets) > 1 else 0

    fit_arr = probe._arrays(fit_rows)
    # Replace only the JEPA target; no per-window heldout label is passed to the
    # student.  no_teacher retains the clean target control by design.
    fit_arr["teacher_w"] = fit_targets
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

    def pred(method: str, row: dict, clean: bool = False):
        if method == "raw":
            return row["candidate"]
        if method == "history":
            return row["hist"]
        return models[method].predict(row, clean)

    teacher_hold, teacher_arr = metric(lambda r, clean=False: global_predict(global_model, r, "candidate"), hold_rows)

    def evaluate_student(rows):
        values, arrays = {}, {}
        for method in methods:
            values[method], arrays[method] = metric(lambda r, clean=False, mm=method: pred(mm, r, clean), rows)
        return values, arrays

    # ``FilterStudent.predict`` consumes ``row['corrupt']`` by default.  Keep
    # that stress view, but also create a detector-clean view by replacing only
    # the student corruption field with the uncorrupted detector candidate.  We
    # do not use heldout GT to make either input; the clean GT is retained only
    # inside the offline teacher labels produced for train-fit.
    heldout_stress, arrays_stress = evaluate_student(hold_rows)
    hold_clean_rows = []
    for row in hold_rows:
        rr = dict(row)
        rr["corrupt"] = np.asarray(row["candidate"], np.float32).copy()
        hold_clean_rows.append(rr)
    heldout_clean, arrays_clean = evaluate_student(hold_clean_rows)

    def paired_controls(arrays, prefix):
        out = {}
        anchor = "student_jepa_global" if "student_jepa_global" in arrays else ("current_jepa" if "current_jepa" in arrays else None)
        if anchor:
            for ctrl in ("raw", "history", "no_teacher", "random_frozen"):
                if ctrl in arrays:
                    out[prefix + "/" + anchor + "_minus_" + ctrl] = {
                        k: probe.paired(arrays[anchor][k], arrays[ctrl][k], args.seed + i)
                        for i, k in enumerate(("r1", "margin", "cos"))
                    }
        return out

    paired = {}
    paired.update(paired_controls(arrays_clean, "deployment_clean"))
    paired.update(paired_controls(arrays_stress, "stress_corrupt"))
    chance = float(np.mean([1.0 / max(1, len(r["next_candidates"])) for r in hold_rows]))
    # Strict, pre-registered gate: teacher must beat chance; student must have
    # positive margin and meaningful chance delta.  This smoke intentionally
    # does not authorize a tracker even if all diagnostics pass.
    teacher_pass = bool(
        teacher_hold.get("finite", False)
        and teacher_hold.get("recall_at_1", -1.0) >= args.teacher_min_r1
        and teacher_hold.get("recall_at_1", -1.0) >= chance + args.teacher_delta
        and teacher_hold.get("margin_mean", -1.0) > 0.0
    )
    # Deployment gate is based on the clean-detector view.  Stress-corrupt
    # results remain a separate robustness diagnostic and cannot be substituted
    # for ordinary deployment quality.
    heldout = heldout_clean
    arrays = arrays_clean
    student_key = "student_jepa_global" if "student_jepa_global" in heldout else ("current_jepa" if "current_jepa" in heldout else None)
    student_row = heldout.get(student_key, {}) if student_key else {}
    student_pass = bool(
        student_key
        and student_row.get("margin_mean", -1.0) > 0.0
        and student_row.get("recall_at_1", -1.0) >= chance + args.student_delta
    )
    numeric_pass = bool(
        global_diag["finite"]
        and np.isfinite(global_diag["condition_number"])
        and global_diag["condition_number"] <= args.max_condition
        and np.isfinite(global_diag["residual_l2"])
        and target_var > args.min_target_variance
        and target_rank > 0
    )
    gate = {
        "teacher": {"pass": teacher_pass, "metrics": teacher_hold, "chance_recall": chance, "min_r1": args.teacher_min_r1, "min_delta": args.teacher_delta},
        "student": {"pass": student_pass, "method": student_key, "metrics": student_row, "min_delta": args.student_delta},
        "numeric_nonconstant": {"pass": numeric_pass, "target_variance_mean_heldout": target_var, "target_rank_heldout": target_rank, "max_condition": args.max_condition, "diagnostics": global_diag},
        "proceed_lagged_tracker": False,
        "decision": "NO-GO (sanity/diagnostic only)",
    }
    return {
        "protocol": {
            "schema": "mot-global-filter-jepa-fallback-v2",
            "teacher": "one global ridge mapping fit on train-fit per-window labels, then frozen",
            "teacher_fit_split": "MOT17 train temporal fit only",
            "heldout_teacher_refit": False,
            "heldout_teacher_current": "uncorrupted detector crop; no heldout GT crop",
            "student": "corrupted detector current predicts frozen global teacher output",
            "student_eval_views": ["deployment_clean (corrupt field replaced by detector candidate)", "stress_corrupt (synthetic mask+noise)"],
            "lag": 1,
            "score_frame": "t+1 only",
            "history_oracle_for_probe": True,
            "oracle_semantics": "GT history replay only; not online",
            "eval_gt_read": False,
            "teacher_frozen": True,
        },
        "detector": detector_meta,
        "train_sequences": [x[0] for x in train_items],
        "eval_sequences": [],
        "train_samples": len(rows),
        "train_fit_samples": len(fit_rows),
        "train_heldout_samples": len(hold_rows),
        "train_split": split,
        "train_meta": meta,
        "global_teacher_diagnostics": global_diag,
        "teacher_trainheldout_t1": teacher_hold,
        "heldout_validation": heldout,
        "heldout_validation_deployment_clean": heldout_clean,
        "heldout_validation_stress_corrupt": heldout_stress,
        "paired_bootstrap": paired,
        "predictor_stats": stats,
        "quality_gate": gate,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mot-train", default="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-09-FRCNN")
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--fit-fraction", type=float, default=0.80)
    p.add_argument("--global-lambda", type=float, default=1.0)
    p.add_argument("--teacher-min-r1", type=float, default=0.80)
    p.add_argument("--teacher-delta", type=float, default=0.20)
    p.add_argument("--student-delta", type=float, default=0.10)
    p.add_argument("--max-condition", type=float, default=1e8)
    p.add_argument("--min-target-variance", type=float, default=1e-8)
    p.add_argument("--methods", default="raw,history,student_jepa_global,no_teacher,random_frozen")
    p.add_argument("--out-dir", default="experiments/results/mot_global_filter_jepa_trainonly_mot20_v2")
    args = p.parse_args()
    probe.base.seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mot-global-filter-jepa-fallback-v2",
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "generated_unix": time.time(),
        "script_sha256": sha256_file(Path(__file__)),
        "base_probe_sha256": sha256_file(HERE / "mot_filter_jepa_probe.py"),
        "eval_gt_read": False,
        "seed": args.seed,
        "max_frames": args.max_frames,
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
    print(json.dumps(payload["mot17_train_only"]["quality_gate"], indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
