"""Evaluate a structural checkpoint on the frozen common-state H1 cases.

This evaluator is intentionally separate from source training.  It reuses the
recorded history and candidate subset, computes structural scores first, and
only then calls the existing offline identity metric summarizer.  It supports
both structural candidates and therefore is ready for a formal checkpoint
without changing the H1 audit protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common_state_audit import EVALUABLE, VideoFeatureIndex, summarize
from .structural_candidate_smoke import _history_and_candidates, _load_entries
from .structural_candidates import (
    CandidateRelativeConfig,
    CandidateRelativePredictor,
    IdentityResidualConfig,
    IdentityResidualPredictor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_model(checkpoint: Path, candidate: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    if candidate == "identity_residual":
        model = IdentityResidualPredictor(IdentityResidualConfig(**config))
    elif candidate == "candidate_relative":
        model = CandidateRelativePredictor(CandidateRelativeConfig(**config))
    else:
        raise ValueError("unknown structural candidate: %s" % candidate)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    if int(model.config.history) != 4 or int(model.config.metadata_dim) != 10:
        raise ValueError("common-state evaluator requires history=4 and metadata_dim=10")
    return model, payload


def _score_case(case: dict, index: VideoFeatureIndex, model, candidate: str):
    batch = _history_and_candidates(case, index)
    if batch is None:
        return None
    features = torch.from_numpy(batch["features"])[None]
    metadata = torch.from_numpy(batch["metadata"])[None]
    history_mask = torch.from_numpy(batch["mask"])[None]
    candidates = torch.from_numpy(batch["candidate_features"])[None]
    candidate_mask = torch.ones((1, candidates.shape[1]), dtype=torch.bool)
    with torch.no_grad():
        if candidate == "identity_residual":
            query = model(features, metadata, history_mask)
            similarity = torch.einsum(
                "bd,bnd->bn", query, F.normalize(candidates, dim=-1)
            )
            costs = (1.0 - similarity)[0].numpy()
        else:
            query = model(features, metadata, history_mask)
            similarity = model.score(query, candidates, candidate_mask)
            costs = (-similarity)[0].numpy()
    if not np.isfinite(costs).all():
        raise FloatingPointError("structural common-state score is non-finite")
    # The candidate ordering and candidate set are copied from the recorded
    # state.  Offline labels stay attached to the case and are not consumed by
    # the model before this point.
    enriched = dict(case)
    enriched["scores"] = {"structural_candidate": [float(value) for value in costs]}
    enriched["common_state_metadata"] = {
        "history_rows": int(batch["mask"].sum()),
        "candidate_rows": int(candidates.shape[1]),
        "min_history_match_iou": float(batch["min_history_iou"]),
        "min_candidate_match_iou": float(batch["min_candidate_iou"]),
    }
    return enriched


def run_evaluation(checkpoint: Path, candidate: str, cases_path: Path,
                   inputs_dir: Path, out_cases: Path, max_events: int | None = None) -> dict:
    cases = [
        json.loads(line)
        for line in cases_path.read_text().splitlines()
        if line.strip()
    ]
    cases = [
        case for case in cases
        if case.get("method") == "predictive_residual_audit"
        and case.get("diagnosis") in EVALUABLE
    ]
    if max_events is not None:
        cases = cases[:max_events]
    model, payload = _load_model(checkpoint, candidate)
    entries = _load_entries(inputs_dir)
    indices: dict[str, VideoFeatureIndex] = {}
    scored = []
    skipped = Counter()
    mapping_iou = []
    for case in cases:
        video = str(case["video"])
        if video not in entries:
            skipped["missing_input_entry"] += 1
            continue
        if video not in indices:
            indices[video] = VideoFeatureIndex(entries[video])
        try:
            enriched = _score_case(case, indices[video], model, candidate)
        except (KeyError, IndexError, ValueError, StopIteration):
            enriched = None
        if enriched is None:
            skipped["feature_box_mapping_failed"] += 1
            continue
        scored.append(enriched)
        mapping_iou.append(float(enriched["common_state_metadata"]["min_candidate_match_iou"]))
    if not scored:
        raise RuntimeError("no structural common-state case could be scored")
    out_cases.parent.mkdir(parents=True, exist_ok=True)
    with out_cases.open("w") as stream:
        for case in scored:
            stream.write(json.dumps(case, separators=(",", ":"), ensure_ascii=False) + "\n")
    metrics = summarize(scored, "structural_candidate", seed=1701)
    return {
        "status": "COMPLETE_STRUCTURAL_COMMON_STATE_CPU_EVALUATION",
        "candidate": candidate,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_payload_method": payload.get("method"),
        "checkpoint_payload_seed": payload.get("seed"),
        "source_cases": str(cases_path),
        "source_case_count": len(cases),
        "scored_case_count": len(scored),
        "skipped_cases": dict(skipped),
        "candidate_coverage": metrics["candidate_coverage_rate"],
        "metrics": metrics,
        "mean_min_candidate_match_iou": sum(mapping_iou) / len(mapping_iou),
        "target_GT_used_for_training_or_gate": False,
        "offline_GT_used_for_diagnosis": True,
        "sealed_used": False,
        "identity_evidence_claim": False,
        "limitations": [
            "engineering or formal checkpoint is evaluated on the recorded predictive-residual common state",
            "candidate subset and coverage are fixed by the existing audit",
            "GT labels enter only the post-score event summarizer",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", choices=["identity_residual", "candidate_relative"], required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--out-cases", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    report = run_evaluation(
        args.checkpoint, args.candidate, args.cases, args.inputs_dir,
        args.out_cases, args.max_events,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("STRUCTURAL_COMMON_STATE_EVALUATION", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

