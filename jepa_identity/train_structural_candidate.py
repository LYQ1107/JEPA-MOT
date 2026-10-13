"""Train one structural H1 candidate on an audited source-only cache.

This entry point intentionally mirrors the existing source-history training
split and provenance checks.  It is not an online target-domain updater.  A
future target evaluation must load the frozen checkpoint and use the common
state evaluator; target labels are never read here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from jepa_cmot.train_source import SourcePairs, sha256

from .structural_candidates import (
    CandidateRelativeConfig,
    CandidateRelativePredictor,
    IdentityResidualConfig,
    IdentityResidualPredictor,
    candidate_relative_objective,
    identity_residual_objective,
)


def _load_cache_metadata(cache: Path, source_domain: str, engineering_smoke: bool) -> dict:
    metadata_path = cache.with_suffix(cache.suffix + ".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("source_domain") != source_domain or metadata.get("target_domains_read"):
        raise ValueError("cache is not an audited source-only cache")
    if not metadata.get("gt_read"):
        raise ValueError("structural source training requires source-supervised cache provenance")
    if metadata.get("engineering_smoke_only") and not engineering_smoke:
        raise ValueError("engineering cache cannot become a formal research run")
    return metadata


def _build_model(candidate: str, feature_dim: int, history: int, metadata_dim: int,
                 hidden: int, relation_dim: int):
    if candidate == "identity_residual":
        config = IdentityResidualConfig(
            feature_dim=feature_dim, history=history, metadata_dim=metadata_dim,
            hidden_dim=hidden,
        )
        return IdentityResidualPredictor(config), config
    if candidate == "candidate_relative":
        config = CandidateRelativeConfig(
            feature_dim=feature_dim, history=history, metadata_dim=metadata_dim,
            hidden_dim=hidden, relation_dim=relation_dim,
        )
        return CandidateRelativePredictor(config), config
    raise ValueError("unknown structural candidate: %s" % candidate)


def _forward_loss(model, candidate: str, method: str, batch, device: str):
    features, metadata, history_mask, candidates, candidate_mask = [tensor.to(device) for tensor in batch]
    positive_index = torch.zeros(features.shape[0], dtype=torch.long, device=device)
    if candidate == "identity_residual":
        query, anchor = model.history_state(features, metadata, history_mask)
        loss, components = identity_residual_objective(
            query, anchor, candidates, positive_index, method,
            candidate_mask=candidate_mask,
        )
        scores = torch.einsum("bd,bnd->bn", query, F.normalize(candidates, dim=-1))
    else:
        relation_query = model(features, metadata, history_mask)
        loss, components = candidate_relative_objective(
            model, relation_query, candidates, positive_index, method,
            candidate_mask=candidate_mask,
        )
        scores = model.score(relation_query, candidates, candidate_mask)
    return loss, components, scores, positive_index


def _validation(model, candidate: str, valid: SourcePairs, batch_size: int, device: str):
    loader = DataLoader(valid, batch_size=batch_size, num_workers=0)
    hits = 0.0
    total = 0
    prediction_error = 0.0
    with torch.no_grad():
        for batch in loader:
            loss, components, scores, positive_index = _forward_loss(
                model, candidate, "predictive_ranking", batch, device
            )
            del loss, components
            tied = torch.isclose(scores, scores.max(dim=1, keepdim=True).values,
                                  rtol=0.0, atol=1e-6) & batch[4].to(device).bool()
            counts = tied.sum(dim=1).clamp_min(1)
            hits += float((tied[:, 0].float() / counts).sum())
            total += len(positive_index)
            prediction_error += float((1.0 - F.softmax(scores, dim=-1)[:, 0]).sum())
    return {
        "source_validation_rank1": hits / total if total else None,
        "source_validation_rank1_tie_rule": "fractional-credit-within-1e-6",
        "source_validation_prediction_error_proxy": prediction_error / total if total else None,
        "validation_samples": total,
    }


def train(args) -> dict:
    metadata = _load_cache_metadata(args.cache, args.source_domain, args.engineering_smoke)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    with np.load(args.cache, allow_pickle=False) as cache:
        videos = sorted(set(int(value) for value in cache["video_ids"]))
        feature_dim = int(cache["features"].shape[1])
    if len(videos) < 2:
        raise ValueError("need separate fitting and validation source videos")
    heldout = videos[-max(1, int(len(videos) * 0.20)):]
    fitting = sorted(set(videos) - set(heldout))
    train_set = SourcePairs(
        args.cache, args.history, fitting, args.width, args.height,
        max_candidates=args.max_candidates, prediction_gaps=tuple(args.prediction_gaps),
    )
    valid_set = SourcePairs(
        args.cache, args.history, heldout, args.width, args.height,
        max_candidates=args.max_candidates, prediction_gaps=tuple(args.prediction_gaps),
    )
    if not len(train_set) or not len(valid_set):
        raise ValueError("no temporal source examples in one of the video-disjoint splits")
    model, config = _build_model(
        args.candidate, feature_dim, args.history, 10, args.hidden, args.relation_dim,
    )
    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.out.mkdir(parents=True, exist_ok=True)
    final = args.out / "final.pt"
    experiment = {
        "candidate": args.candidate,
        "method": args.method,
        "seed": args.seed,
        "steps": args.steps,
        "batch": args.batch,
        "history": args.history,
        "hidden": args.hidden,
        "relation_dim": args.relation_dim,
        "source_domain": args.source_domain,
        "width": args.width,
        "height": args.height,
        "max_candidates": args.max_candidates,
        "prediction_gaps": args.prediction_gaps,
        "cache_sha256": sha256(args.cache),
        "engineering_smoke_only": args.engineering_smoke,
        "target_gt_used_for_training_or_gate": False,
        "candidate_input_to_history_encoder": False,
        "source_split": "video_disjoint_last_20_percent_v1",
        "code_sha256": {
            path.name: sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("structural_candidates.py"),
                Path(__file__).parents[1] / "jepa_cmot" / "train_source.py",
                Path(__file__).parents[1] / "jepa_cmot" / "history.py",
            )
        },
    }
    fingerprint = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    report_path = args.out / "metrics.json"
    if final.exists() and report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get("experiment_fingerprint") != fingerprint:
            raise ValueError("existing completed structural experiment has different configuration")
        print("STRUCTURAL_TRAIN_ALREADY_COMPLETE", report_path, flush=True)
        return report
    rng = np.random.RandomState(args.seed)
    losses = []
    start = time.perf_counter()
    for step in range(args.steps):
        batch = default_collate([train_set[int(index)] for index in rng.randint(0, len(train_set), args.batch)])
        loss, components, _, _ = _forward_loss(model, args.candidate, args.method, batch, args.device)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite structural objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite structural gradient")
        optimizer.step()
        losses.append({
            "step": step + 1,
            "loss": float(loss.detach()),
            "prediction": float(components["prediction"]),
            "ranking": float(components["ranking"]),
            "relation": float(components.get("relation", 0.0)),
            "gradient_norm": float(norm),
        })
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            print(args.candidate, args.method, step + 1, losses[-1], flush=True)
    model.eval()
    validation = _validation(model, args.candidate, valid_set, args.batch, args.device)
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "candidate": args.candidate,
        "method": args.method,
        "seed": args.seed,
        "source_domain": args.source_domain,
        "source_feature_metadata": metadata,
        "training_code_sha256": experiment["code_sha256"],
        "experiment_fingerprint": fingerprint,
    }
    torch.save(payload, final)
    report = {
        "status": "complete",
        "engineering_smoke_only": args.engineering_smoke,
        "candidate": args.candidate,
        "method": args.method,
        "config": asdict(config),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch,
        "prediction_gaps": args.prediction_gaps,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "fit_videos": fitting,
        "validation_videos": heldout,
        "fit_samples": len(train_set),
        "validation_samples": len(valid_set),
        **validation,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "checkpoint_sha256": sha256(final),
        "cache_sha256": experiment["cache_sha256"],
        "experiment_fingerprint": fingerprint,
        "training_code_sha256": experiment["code_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1]).decode().strip(),
        "seconds": time.perf_counter() - start,
        "target_gt_used_for_training_or_gate": False,
        "candidate_input_to_history_encoder": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "losses.json").write_text(json.dumps(losses) + "\n")
    print("STRUCTURAL_TRAIN_COMPLETE", json.dumps(report), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate", choices=["identity_residual", "candidate_relative"], required=True)
    parser.add_argument("--method", choices=["predictive", "discriminative", "predictive_ranking"], required=True)
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--relation-dim", type=int, default=128)
    parser.add_argument("--prediction-gaps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--width", type=float, default=960)
    parser.add_argument("--height", type=float, default=540)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--engineering-smoke", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1 or args.max_candidates < 2:
        raise ValueError("steps, batch, and max-candidates must be positive")
    if min(args.prediction_gaps) < 1:
        raise ValueError("prediction gaps must be positive")
    train(args)


if __name__ == "__main__":
    main()

