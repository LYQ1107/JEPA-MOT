"""Run a bounded, fixed-teacher H2 objective comparison on recorded pairs.

This is H2-Wave-B evidence only.  A single prior stream learner checkpoint
provides a frozen bank of unlabeled pseudo-pairs.  The same bank and the same
sample-index schedule are replayed from the Source-D and Source-P snapshots
under Update-D and Update-P.  No target labels are read and no live tracker,
HOTA matrix, or closed-loop pseudo-label generation is claimed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from .models import HistoryPredictor, PredictorConfig, objective
from .train_source import SourcePairs, sha256


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_model(path: Path, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HistoryPredictor(PredictorConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


def _load_teacher_bank(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    examples = list(payload.get("memory", [])) + list(payload.get("recent", []))
    if len(examples) < 8:
        raise ValueError("fixed teacher bank must contain at least eight examples")
    for index, example in enumerate(examples):
        if len(example) < 7:
            raise ValueError(f"teacher example {index} has no complete recorded fields")
        if not all(torch.is_tensor(value) for value in example[:7]):
            raise ValueError(f"teacher example {index} contains a non-tensor field")
        if tuple(example[0].shape) != (4, 1024) or tuple(example[1].shape) != (4, 10):
            raise ValueError("fixed teacher bank is not the audited BDD history-4/1024-D format")
        if tuple(example[2].shape) != (4,) or tuple(example[3].shape) != (16, 1024):
            raise ValueError("fixed teacher bank has an unexpected candidate shape")
    return examples, payload


def _make_schedule(count: int, steps: int, batch: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    schedule = rng.randint(0, count, size=(steps, batch), dtype=np.int64)
    return schedule


def _batch(examples, indices, device):
    values = default_collate([examples[int(index)] for index in indices])
    return tuple(value.to(device) for value in values[:7])


def _forward(model, batch, method):
    features, metadata, mask, candidates, valid, teacher, teacher_query = batch
    query = model(features, metadata, mask)
    positive = torch.zeros(len(query), dtype=torch.long, device=query.device)
    loss, parts = objective(query, candidates, positive, method, candidate_mask=valid)
    logits = torch.einsum("bd,bnd->bn", query, F.normalize(candidates.detach(), dim=-1)) / 0.07
    logits = logits.masked_fill(~valid, -1e4)
    return query, logits, loss, parts, teacher


def _metrics(model, examples, indices, device, batch_size=64):
    model.eval()
    top1 = 0.0
    reciprocal = 0.0
    prediction_error = 0.0
    teacher_kl = 0.0
    total = 0
    for start in range(0, len(indices), batch_size):
        batch = _batch(examples, indices[start : start + batch_size], device)
        with torch.no_grad():
            query, logits, _, _, teacher = _forward(model, batch, "discriminative")
            valid = batch[4]
            order = torch.argsort(logits, dim=1, descending=True)
            positive_rank = (order == 0).nonzero(as_tuple=False)[:, 1] + 1
            top1 += float((positive_rank == 1).sum())
            reciprocal += float((1.0 / positive_rank.float()).sum())
            prediction_error += float(
                (1.0 - F.cosine_similarity(query, batch[3][:, 0], dim=-1)).sum()
            )
            teacher_kl += float(
                F.kl_div(F.log_softmax(logits, dim=-1), teacher, reduction="sum").detach()
            )
            total += len(query)
    return {
        "examples": total,
        "top1": top1 / total if total else None,
        "mrr": reciprocal / total if total else None,
        "prediction_error": prediction_error / total if total else None,
        "teacher_kl": teacher_kl / total if total else None,
    }


def _source_validation(model, cache: Path, device: str, seed: int, limit: int):
    with np.load(cache, allow_pickle=False) as data:
        videos = sorted(set(int(value) for value in data["video_ids"]))
    heldout = videos[-max(1, int(len(videos) * 0.20)) :]
    valid = SourcePairs(cache, 4, heldout, 960, 540, prediction_gaps=(1, 2, 4, 8))
    if not len(valid):
        return {"examples": 0, "video_split": heldout}
    rng = np.random.RandomState(seed + 701)
    count = min(int(limit), len(valid))
    selected = rng.choice(len(valid), size=count, replace=False)
    model.eval()
    top1 = 0.0
    prediction_error = 0.0
    for start in range(0, count, 64):
        batch = default_collate([valid[int(index)] for index in selected[start : start + 64]])
        features, metadata, mask, candidates, valid_mask = [value.to(device) for value in batch]
        with torch.no_grad():
            query = model(features, metadata, mask)
            logits = torch.einsum("bd,bnd->bn", query, F.normalize(candidates, dim=-1))
            logits = logits.masked_fill(~valid_mask, -1e4)
            top1 += float((logits.argmax(dim=1) == 0).sum())
            prediction_error += float(
                (1.0 - F.cosine_similarity(query, candidates[:, 0], dim=-1)).sum()
            )
    return {
        "examples": count,
        "video_split": heldout,
        "top1": top1 / count,
        "prediction_error": prediction_error / count,
        "source_cache_sha256": sha256(cache),
        "target_gt_used_for_training_or_gate": False,
    }


def _parameter_distance(model, initial):
    total = torch.zeros((), dtype=torch.float64)
    for parameter, reference in zip(model.parameters(), initial):
        total += (parameter.detach().double() - reference.double()).pow(2).sum()
    return float(total.sqrt())


def _run_cell(source_path, update_method, examples, schedule, device, source_cache, args):
    model, source_payload = _load_model(source_path, device)
    initial = [parameter.detach().cpu().clone() for parameter in model.parameters()]
    frozen = _metrics(model, examples, np.arange(len(examples)), device)
    source_before = (
        _source_validation(model, source_cache, device, args.seed, args.source_eval_limit)
        if source_cache
        else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    losses = []
    gradient_norms = []
    for step, indices in enumerate(schedule, start=1):
        batch = _batch(examples, indices, device)
        model.train()
        _, _, loss, parts, _ = _forward(model, batch, update_method)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step} ({update_method})")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"non-finite gradient at step {step} ({update_method})")
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
        if source_cache
        else None
    )
    return {
        "source_checkpoint_sha256": sha256(source_path),
        "source_method": source_payload.get("method"),
        "source_initialization": "D" if source_payload.get("method") == "discriminative" else "P",
        "online_update": "D" if update_method == "discriminative" else "P",
        "steps": len(schedule),
        "batch_size": int(schedule.shape[1]),
        "frozen_reference": frozen,
        "fixed_teacher_before": frozen,
        "fixed_teacher_after": after,
        "fixed_teacher_delta": {
            key: (after[key] - frozen[key]) if isinstance(after.get(key), (int, float)) else None
            for key in ["top1", "mrr", "prediction_error", "teacher_kl"]
        },
        "source_validation_before": source_before,
        "source_validation_after": source_after,
        "parameter_update_l2": _parameter_distance(model, initial),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "gradient_norm_mean": float(np.mean([row["gradient_norm"] for row in losses])),
        "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "weight_decay": 1e-4},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-d", type=Path, required=True)
    parser.add_argument("--source-p", type=Path, required=True)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--source-eval-limit", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1:
        raise ValueError("steps and batch must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    examples, teacher_payload = _load_teacher_bank(args.teacher_bank)
    schedule = _make_schedule(len(examples), args.steps, args.batch, args.seed)
    schedule_sha = _hash_bytes(schedule.tobytes())
    cells = {}
    for source_name, source_path in (("D", args.source_d), ("P", args.source_p)):
        for update_name, update_method in (("D", "discriminative"), ("P", "predictive")):
            cell_name = f"{source_name}_to_{update_name}"
            print("H2_FIXED_TEACHER_CELL_START", cell_name, flush=True)
            cells[cell_name] = _run_cell(
                source_path, update_method, examples, schedule, args.device,
                args.source_cache, args,
            )
            print("H2_FIXED_TEACHER_CELL_COMPLETE", cell_name, flush=True)
    report = {
        "status": "COMPLETE_H2_WAVE_B_FIXED_TEACHER_DIAGNOSTIC",
        "scope": "one-seed BDD fixed-teacher recorded-pair objective comparison",
        "dataset": "bdd100k",
        "seed": args.seed,
        "device": args.device,
        "cells": cells,
        "same_initialization_within_source": True,
        "same_teacher_examples_across_cells": True,
        "same_sample_index_schedule_across_cells": True,
        "schedule": {"steps": args.steps, "batch": args.batch, "example_count": len(examples), "sha256": schedule_sha},
        "teacher_bank": {
            "path": str(args.teacher_bank.resolve()),
            "sha256": sha256(args.teacher_bank),
            "artifact_kind": "recorded_frozen_pseudo_pair_bank_from_one_prior_unlabeled_stream",
            "online_config": teacher_payload.get("online_config"),
            "memory_count": len(teacher_payload.get("memory", [])),
            "recent_count": len(teacher_payload.get("recent", [])),
            "target_gt_read": False,
        },
        "source_checkpoint_paths": {"D": str(args.source_d.resolve()), "P": str(args.source_p.resolve())},
        "source_cache": str(args.source_cache.resolve()) if args.source_cache else None,
        "target_gt_used_for_training_or_gate": False,
        "sealed_test_used": False,
        "full_video_metrics": False,
        "closed_loop_online": False,
        "hota_claim": False,
        "interpretation_limit": "This isolates the loss objective on one shared recorded pseudo-pair bank; it is not a full-video continual-learning result and does not replace the registered 2x2 closed-loop factorial.",
        "code_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1]).decode().strip(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("H2_FIXED_TEACHER_COMPLETE", args.out, flush=True)


if __name__ == "__main__":
    main()
