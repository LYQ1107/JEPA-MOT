"""Probe a ranking-regularized predictive update after the H2 Wave-B result.

This is a small, pre-registered follow-up to ``h2_fixed_teacher``.  It keeps
the same BDD seed17 pseudo-pair bank, source snapshots, schedule, optimizer,
and CPU budget, and changes only Update-P from prediction-only to the existing
equal-weight ``predictive_ranking`` objective.  It is mechanism evidence, not
the registered full-video H2 factorial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .h2_fixed_teacher import (
    _batch,
    _forward,
    _load_model,
    _load_teacher_bank,
    _make_schedule,
    _metrics,
    _parameter_distance,
    _source_validation,
)
from .train_source import sha256


def _gradient_vector(model, batch, method):
    model.zero_grad(set_to_none=True)
    model.train()
    _, _, loss, parts, _ = _forward(model, batch, method)
    loss.backward()
    vector = torch.cat([
        parameter.grad.detach().float().reshape(-1)
        for parameter in model.parameters() if parameter.grad is not None
    ]).cpu()
    return vector, loss, parts


def _run_cell(source_path, examples, schedule, device, source_cache, args):
    model, source_payload = _load_model(source_path, device)
    initial = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    before = _metrics(model, examples, np.arange(len(examples)), device)
    source_before = (
        _source_validation(model, source_cache, device, args.seed, args.source_eval_limit)
        if source_cache else None
    )
    alignment_rows = []
    for indices in schedule[: min(16, len(schedule))]:
        batch = _batch(examples, indices, device)
        d, _, _ = _gradient_vector(model, batch, "discriminative")
        p, _, _ = _gradient_vector(model, batch, "predictive")
        h, _, _ = _gradient_vector(model, batch, "predictive_ranking")
        alignment_rows.append({
            "hybrid_vs_discriminative_cosine": float(torch.dot(h, d) / max(float(h.norm() * d.norm()), 1e-12)),
            "hybrid_vs_predictive_cosine": float(torch.dot(h, p) / max(float(h.norm() * p.norm()), 1e-12)),
            "hybrid_norm": float(h.norm()),
            "discriminative_norm": float(d.norm()),
            "predictive_norm": float(p.norm()),
        })
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    losses = []
    for step, indices in enumerate(schedule, start=1):
        batch = _batch(examples, indices, device)
        _, _, loss, parts, _ = _forward(model, batch, "predictive_ranking")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        losses.append({
            "step": step,
            "loss": float(loss.detach()),
            "prediction": float(parts["prediction"]),
            "ranking": float(parts["ranking"]),
            "gradient_norm": float(norm),
        })
    after = _metrics(model, examples, np.arange(len(examples)), device)
    source_after = (
        _source_validation(model, source_cache, device, args.seed, args.source_eval_limit)
        if source_cache else None
    )
    return {
        "source_checkpoint_sha256": sha256(source_path),
        "source_method": source_payload.get("method"),
        "source_initialization": "D" if source_payload.get("method") == "discriminative" else "P",
        "online_update": "predictive_ranking",
        "steps": len(schedule),
        "batch_size": int(schedule.shape[1]),
        "fixed_teacher_before": before,
        "fixed_teacher_after": after,
        "fixed_teacher_delta": {key: after[key] - before[key] for key in ["top1", "mrr", "prediction_error", "teacher_kl"]},
        "source_validation_before": source_before,
        "source_validation_after": source_after,
        "parameter_update_l2": _parameter_distance(model, initial),
        "gradient_alignment": {
            "batches": len(alignment_rows),
            "hybrid_vs_discriminative_cosine_mean": float(np.mean([x["hybrid_vs_discriminative_cosine"] for x in alignment_rows])),
            "hybrid_vs_predictive_cosine_mean": float(np.mean([x["hybrid_vs_predictive_cosine"] for x in alignment_rows])),
            "hybrid_norm_mean": float(np.mean([x["hybrid_norm"] for x in alignment_rows])),
            "discriminative_norm_mean": float(np.mean([x["discriminative_norm"] for x in alignment_rows])),
            "predictive_norm_mean": float(np.mean([x["predictive_norm"] for x in alignment_rows])),
            "rows": alignment_rows,
        },
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "gradient_norm_mean": float(np.mean([x["gradient_norm"] for x in losses])),
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "weight_decay": 1e-4},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-d", type=Path, required=True)
    parser.add_argument("--source-p", type=Path, required=True)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--source-eval-limit", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    examples, teacher_payload = _load_teacher_bank(args.teacher_bank)
    schedule = _make_schedule(len(examples), args.steps, args.batch, args.seed)
    schedule_sha = hashlib.sha256(schedule.tobytes()).hexdigest()
    baseline = json.loads(args.baseline_report.read_text())
    cells = {}
    for source_name, source_path in (("D", args.source_d), ("P", args.source_p)):
        name = f"{source_name}_to_predictive_ranking"
        print("H2_MECHANISM_CELL_START", name, flush=True)
        cells[name] = _run_cell(source_path, examples, schedule, args.device, args.source_cache, args)
        print("H2_MECHANISM_CELL_COMPLETE", name, flush=True)
    report = {
        "status": "COMPLETE_H2_WAVE_D_PREDICTIVE_RANKING_MECHANISM_PROBE",
        "scope": "one-seed BDD fixed-teacher mechanism probe after Wave-B gradient diagnosis",
        "dataset": "bdd100k",
        "seed": args.seed,
        "device": args.device,
        "objective": "predictive_ranking_equal_weight_prediction_plus_ranking",
        "cells": cells,
        "same_teacher_examples_across_cells": True,
        "same_sample_index_schedule_across_cells": True,
        "schedule": {"steps": args.steps, "batch": args.batch, "example_count": len(examples), "sha256": schedule_sha},
        "teacher_bank": {
            "path": str(args.teacher_bank.resolve()), "sha256": sha256(args.teacher_bank),
            "artifact_kind": "recorded_frozen_pseudo_pair_bank_from_one_prior_unlabeled_stream",
            "online_config": teacher_payload.get("online_config"),
            "memory_count": len(teacher_payload.get("memory", [])),
            "recent_count": len(teacher_payload.get("recent", [])),
            "target_gt_read": False,
        },
        "baseline_report": {"path": str(args.baseline_report.resolve()), "sha256": sha256(args.baseline_report), "status": baseline.get("status")},
        "source_checkpoint_paths": {"D": str(args.source_d.resolve()), "P": str(args.source_p.resolve())},
        "source_cache": str(args.source_cache.resolve()) if args.source_cache else None,
        "target_gt_used_for_training_or_gate": False,
        "sealed_test_used": False,
        "full_video_metrics": False,
        "closed_loop_online": False,
        "hota_claim": False,
        "interpretation_limit": "This tests whether adding a ranking term rescues prediction-only updates on one shared recorded pseudo-pair bank; it is not a full-video continual-learning result and does not replace the registered 2x2 factorial.",
        "code_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1]).decode().strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("H2_MECHANISM_PROBE_COMPLETE", args.out, flush=True)


if __name__ == "__main__":
    main()
