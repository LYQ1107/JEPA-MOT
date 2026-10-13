"""Preflight and evaluate one structural checkpoint without changing H1 rules.

The preflight rejects provenance/configuration mismatches before the common
state evaluator is run.  It is designed for a formal GPU-produced checkpoint,
but can also validate an explicitly allowed engineering smoke checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate_structural_candidate import run_evaluation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def static_checks(checkpoint: Path, metrics_path: Path, cache: Path,
                  candidate: str, expected_source_domain: str,
                  allow_engineering_smoke: bool) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metrics = json.loads(metrics_path.read_text())
    cache_metadata = json.loads(cache.with_suffix(cache.suffix + ".json").read_text())
    checkpoint_sha = sha256(checkpoint)
    cache_sha = sha256(cache)
    _assert(payload.get("candidate") == candidate,
            "checkpoint candidate does not match requested structural candidate")
    _assert(metrics.get("candidate") == candidate,
            "metrics candidate does not match requested structural candidate")
    _assert(metrics.get("checkpoint_sha256") == checkpoint_sha,
            "checkpoint hash does not match metrics provenance")
    _assert(metrics.get("cache_sha256") == cache_sha,
            "cache hash does not match metrics provenance")
    _assert(payload.get("source_domain") == expected_source_domain,
            "checkpoint source domain does not match requested source domain")
    _assert(metrics.get("target_gt_used_for_training_or_gate") is False,
            "checkpoint metrics permit target GT in training or gate")
    _assert(metrics.get("candidate_input_to_history_encoder") is False,
            "checkpoint metrics permit candidate input to history encoder")
    _assert(not cache_metadata.get("target_domains_read"),
            "cache metadata includes target domains")
    _assert(cache_metadata.get("gt_read") is True,
            "source cache lacks source GT provenance")
    _assert(cache_metadata.get("causal") is True and cache_metadata.get("future_frames_read") is False,
            "source cache is not marked causal")
    smoke = bool(metrics.get("engineering_smoke_only"))
    _assert(allow_engineering_smoke or not smoke,
            "engineering smoke checkpoint requires --allow-engineering-smoke")
    config = payload.get("config", {})
    _assert(int(config.get("history", -1)) == 4,
            "structural checkpoint history must be 4 for common-state evaluation")
    _assert(int(config.get("metadata_dim", -1)) == 10,
            "structural checkpoint metadata_dim must be 10")
    with np.load(cache, allow_pickle=False) as arrays:
        feature_dim = int(arrays["features"].shape[1])
        video_count = len(set(int(value) for value in arrays["video_ids"]))
    _assert(int(config.get("feature_dim", -1)) == feature_dim,
            "checkpoint feature dimension does not match source cache")
    _assert(video_count >= 2, "source cache cannot form a video-disjoint validation split")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "cache_sha256": cache_sha,
        "checkpoint_candidate": payload.get("candidate"),
        "checkpoint_method": payload.get("method"),
        "checkpoint_seed": payload.get("seed"),
        "engineering_smoke_only": smoke,
        "source_domain": expected_source_domain,
        "feature_dim": feature_dim,
        "source_video_count": video_count,
        "target_gt_used_for_training_or_gate": False,
        "candidate_input_to_history_encoder": False,
    }


def run_preflight(checkpoint: Path, metrics_path: Path, cache: Path, candidate: str,
                  source_domain: str, cases: Path, inputs_dir: Path,
                  out_cases: Path, max_events: int | None,
                  allow_engineering_smoke: bool) -> dict:
    checks = static_checks(
        checkpoint, metrics_path, cache, candidate, source_domain,
        allow_engineering_smoke,
    )
    evaluation = run_evaluation(
        checkpoint, candidate, cases, inputs_dir, out_cases, max_events,
    )
    return {
        "status": "PASS_STRUCTURAL_CANDIDATE_PREFLIGHT",
        "static_checks": checks,
        "evaluation": evaluation,
        "identity_evidence_claim": False,
        "limitation": "preflight validates provenance and common-state execution; it does not establish an H1 gain",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--candidate", choices=["identity_residual", "candidate_relative"], required=True)
    parser.add_argument("--source-domain", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--out-cases", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--allow-engineering-smoke", action="store_true")
    args = parser.parse_args()
    report = run_preflight(
        args.checkpoint, args.metrics, args.cache, args.candidate,
        args.source_domain, args.cases, args.inputs_dir, args.out_cases,
        args.max_events, args.allow_engineering_smoke,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("STRUCTURAL_PREFLIGHT_PASS", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

