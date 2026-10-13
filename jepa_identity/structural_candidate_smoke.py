"""Run a small real-cache smoke for the structural H1 candidates.

This is an engineering integration check, not a training or identity result.
It maps existing common-state cases to the existing dense feature cache, forms
past-only histories, and verifies both structural models can score the same
candidate set and compute their offline source loss when a positive label is
present.  Ground-truth IDs are read only to select the loss index after query
and score tensors have already been constructed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from jepa_cmot.history import pack_history

from .common_state_audit import VideoFeatureIndex
from .structural_candidates import (
    CandidateRelativeConfig,
    CandidateRelativePredictor,
    IdentityResidualConfig,
    IdentityResidualPredictor,
    candidate_relative_objective,
    identity_residual_objective,
    stable_candidate_order,
    validate_causal_history,
)


def _load_entries(inputs_dir: Path) -> dict[str, dict]:
    entries = {}
    for path in sorted(inputs_dir.glob("*.json")):
        rows = json.loads(path.read_text())
        if rows:
            entries[str(rows[0]["video"])] = rows[0]
    return entries


def _history_and_candidates(case: dict, index: VideoFeatureIndex):
    history_matches = []
    native_class = int(case.get("native_class", 0))
    for row in case.get("history", []):
        match = index.match(int(row["frame"]), row["box"], native_class)
        if match is None:
            return None
        history_matches.append((row, match))
    candidate_matches = []
    for candidate in case.get("candidate_gt_ids", []):
        match = index.match(
            int(case["frame"]), candidate["box"],
            int(candidate.get("native_class", native_class)),
        )
        if match is None:
            return None
        candidate_matches.append((candidate, match))
    if not history_matches or not candidate_matches:
        return None
    image = index.images[int(case["frame"])]
    birth = min(int(row["frame"]) for row, _ in history_matches)
    observations = [
        {
            "feature": np.asarray(match["feature"], dtype=np.float32),
            "box": np.asarray(row["box"], dtype=np.float32),
            "frame": int(row["frame"]),
            "birth": birth,
            "confidence": float(row.get("confidence", 1.0)),
        }
        for row, match in history_matches
    ]
    features, metadata, mask = pack_history(
        observations,
        int(case["frame"]),
        history=4,
        feature_dim=int(history_matches[0][1]["feature"].shape[-1]),
        width=float(image["width"]),
        height=float(image["height"]),
    )
    candidates = np.asarray([match["feature"] for _, match in candidate_matches], dtype=np.float32)
    candidate_ids = [int(candidate.get("index", position)) for position, (candidate, _) in enumerate(candidate_matches)]
    candidate_gt_ids = [candidate.get("gt_id") for candidate, _ in candidate_matches]
    return {
        "features": features,
        "metadata": metadata,
        "mask": mask,
        "candidate_features": candidates,
        "candidate_ids": candidate_ids,
        "candidate_gt_ids": candidate_gt_ids,
        "history_frames": np.asarray([int(row["frame"]) for row, _ in history_matches], dtype=np.int64),
        "current_frame": int(case["frame"]),
        "min_history_iou": min(float(match["match_iou"]) for _, match in history_matches),
        "min_candidate_iou": min(float(match["match_iou"]) for _, match in candidate_matches),
    }


def run_smoke(cases_path: Path, inputs_dir: Path, max_events: int) -> dict:
    cases = [
        json.loads(line)
        for line in cases_path.read_text().splitlines()
        if line.strip()
    ]
    cases = [
        case for case in cases
        if case.get("method") == "predictive_residual_audit"
        and case.get("diagnosis") in {
            "corrected_local_identity", "harmful_local_identity", "both_correct", "neither_correct",
        }
    ][:max_events]
    entries = _load_entries(inputs_dir)
    indices: dict[str, VideoFeatureIndex] = {}
    models: dict[str, torch.nn.Module] = {}
    feature_dim = None
    mapped = 0
    positive = 0
    min_history_iou = []
    min_candidate_iou = []
    skip_reasons = Counter()
    causal_checks = 0
    residual_loss_checks = 0
    relation_loss_checks = 0
    query_checks = 0
    order_checks = 0

    torch.set_num_threads(2)
    torch.manual_seed(17)
    for case in cases:
        video = str(case["video"])
        if video not in entries:
            skip_reasons["missing_input_entry"] += 1
            continue
        if video not in indices:
            indices[video] = VideoFeatureIndex(entries[video])
        try:
            batch = _history_and_candidates(case, indices[video])
        except (KeyError, IndexError, ValueError, StopIteration):
            batch = None
        if batch is None:
            skip_reasons["feature_box_mapping_failed"] += 1
            continue
        if feature_dim is None:
            feature_dim = int(batch["features"].shape[-1])
            residual = IdentityResidualPredictor(IdentityResidualConfig(
                feature_dim=feature_dim, history=4, metadata_dim=int(batch["metadata"].shape[-1]),
                hidden_dim=32))
            relative = CandidateRelativePredictor(CandidateRelativeConfig(
                feature_dim=feature_dim, history=4, metadata_dim=int(batch["metadata"].shape[-1]),
                hidden_dim=32, relation_dim=32))
            models = {"identity_residual": residual, "candidate_relative": relative}
            for model in models.values():
                model.eval()
        elif int(batch["features"].shape[-1]) != feature_dim:
            skip_reasons["feature_dimension_changed"] += 1
            continue

        history_frames = np.zeros((1, 4), dtype=np.int64)
        history_tail = batch["history_frames"][-4:]
        history_frames[0, -len(history_tail):] = history_tail
        validate_causal_history(history_frames, np.asarray([batch["current_frame"]]), batch["mask"][None])
        causal_checks += 1

        candidate_count = len(batch["candidate_ids"])
        if stable_candidate_order(batch["candidate_ids"]) != tuple(range(candidate_count)):
            skip_reasons["nondeterministic_candidate_order"] += 1
            continue
        order_checks += 1
        features = torch.from_numpy(batch["features"])[None]
        metadata = torch.from_numpy(batch["metadata"])[None]
        mask = torch.from_numpy(batch["mask"])[None]
        candidates = torch.from_numpy(batch["candidate_features"])[None]
        candidate_mask = torch.ones((1, candidate_count), dtype=torch.bool)
        with torch.no_grad():
            residual_query, anchor = models["identity_residual"].history_state(features, metadata, mask)
            relation_query = models["candidate_relative"](features, metadata, mask)
            residual_scores = torch.einsum(
                "bd,bnd->bn", residual_query, torch.nn.functional.normalize(candidates, dim=-1)
            )
            relation_scores = models["candidate_relative"].score(
                relation_query, candidates, candidate_mask
            )
            residual_again = models["identity_residual"](features, metadata, mask)
            relation_again = models["candidate_relative"](features, metadata, mask)
        if not torch.allclose(residual_query, residual_again, atol=1e-7, rtol=0):
            raise AssertionError("identity residual query changed across identical history calls")
        if not torch.allclose(relation_query, relation_again, atol=1e-7, rtol=0):
            raise AssertionError("candidate-relative query changed across identical history calls")
        if not torch.isfinite(residual_scores).all() or not torch.isfinite(relation_scores).all():
            raise AssertionError("structural candidate score is non-finite")
        query_checks += 1

        expected = case.get("expected_gt_id")
        if expected in batch["candidate_gt_ids"]:
            positive_index = torch.tensor([batch["candidate_gt_ids"].index(expected)], dtype=torch.long)
            with torch.no_grad():
                residual_loss, _ = identity_residual_objective(
                    residual_query, anchor, candidates, positive_index, "predictive_ranking",
                    candidate_mask=candidate_mask,
                )
                relation_loss, _ = candidate_relative_objective(
                    models["candidate_relative"], relation_query, candidates,
                    positive_index, "predictive_ranking", candidate_mask=candidate_mask,
                )
            if not torch.isfinite(residual_loss) or not torch.isfinite(relation_loss):
                raise AssertionError("structural candidate loss is non-finite")
            residual_loss_checks += 1
            relation_loss_checks += 1
            positive += 1
        mapped += 1
        min_history_iou.append(batch["min_history_iou"])
        min_candidate_iou.append(batch["min_candidate_iou"])

    if not cases:
        raise ValueError("no eligible common-state cases selected")
    if mapped == 0:
        raise RuntimeError("no real common-state case could be mapped")
    return {
        "status": "COMPLETE_STRUCTURAL_CANDIDATE_REAL_CACHE_SMOKE",
        "source_cases": str(cases_path),
        "inputs_dir": str(inputs_dir),
        "requested_events": len(cases),
        "mapped_events": mapped,
        "skipped_events": sum(skip_reasons.values()),
        "skip_reasons": dict(skip_reasons),
        "candidate_present_events": positive,
        "causal_history_checks": causal_checks,
        "candidate_order_checks": order_checks,
        "candidate_independent_query_checks": query_checks,
        "identity_residual_loss_checks": residual_loss_checks,
        "candidate_relative_loss_checks": relation_loss_checks,
        "feature_dim": feature_dim,
        "parameter_counts": {name: sum(parameter.numel() for parameter in model.parameters())
                             for name, model in models.items()},
        "min_history_match_iou": min(min_history_iou),
        "min_candidate_match_iou": min(min_candidate_iou),
        "target_gt_used_for_model_forward": False,
        "target_gt_used_for_loss_index_only": True,
        "identity_evidence_claim": False,
        "limitation": "engineering smoke on existing residual common-state cases; no training or held-out identity result",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=16)
    args = parser.parse_args()
    report = run_smoke(args.cases, args.inputs_dir, args.max_events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("STRUCTURAL_CANDIDATE_REAL_CACHE_SMOKE", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

