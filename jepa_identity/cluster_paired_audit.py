"""Video-cluster paired uncertainty audit for fixed identity cases.

Frame-level cases within a video are correlated.  This module therefore uses
the video as the resampling unit and reports a cluster-balanced mean delta.
It does not replace the event-level report; it prevents event counts from
being mistaken for independent videos.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common_state_audit import event_metric
from .paired_common_state import DEFAULT_COMPARISONS, DEFAULT_CONTROLS, _control_results, _winner_signal, load_cases


METRICS = ("top1", "mrr", "pairwise_auc", "net_correction")


def _value(metric: str, case: dict[str, Any], result: dict[str, Any]) -> float | None:
    if metric == "top1":
        return float(bool(result["top1"]))
    if metric == "net_correction":
        return float(_winner_signal(case, result)[0])
    value = result.get(metric)
    return None if value is None or not np.isfinite(float(value)) else float(value)


def _bootstrap(values: np.ndarray, seed: int, draws: int = 5000) -> list[float] | None:
    if values.size == 0:
        return None
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 256):
        stop = min(draws, start + 256)
        means[start:stop] = values[rng.integers(0, values.size, size=(stop - start, values.size))].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def _permutation(values: np.ndarray, seed: int, draws: int = 10000) -> float | None:
    if values.size == 0:
        return None
    observed = abs(float(values.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, draws, 256):
        stop = min(draws, start + 256)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(stop - start, values.size))
        extreme += int(np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed))
    return float((extreme + 1) / (draws + 1))


def audit(cases: list[dict[str, Any]], *, dataset: str,
          controls: Iterable[str] = DEFAULT_CONTROLS,
          comparisons: Iterable[tuple[str, str]] = DEFAULT_COMPARISONS,
          seed: int = 17) -> dict[str, Any]:
    controls = tuple(controls)
    comparisons = tuple(comparisons)
    results = _control_results(cases, controls)
    present_mask = np.asarray([bool(row["candidate_present"]) for row in results[controls[0]]], dtype=bool)
    present_cases = [case for case, present in zip(cases, present_mask) if present]
    videos = sorted({str(case.get("video")) for case in cases})
    groups: dict[str, list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        if present_mask[index]:
            groups[str(case.get("video"))].append(index)

    reports = []
    for comparison_index, (candidate, reference) in enumerate(comparisons):
        if candidate not in results or reference not in results:
            raise ValueError(f"unknown comparison controls: {candidate}, {reference}")
        metric_reports: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            cluster_means = []
            cluster_names = []
            for video in sorted(groups):
                deltas = []
                for index in groups[video]:
                    candidate_value = _value(metric, cases[index], results[candidate][index])
                    reference_value = _value(metric, cases[index], results[reference][index])
                    if candidate_value is not None and reference_value is not None:
                        deltas.append(candidate_value - reference_value)
                if deltas:
                    cluster_names.append(video)
                    cluster_means.append(float(np.mean(deltas)))
            values = np.asarray(cluster_means, dtype=np.float64)
            metric_reports[metric] = {
                "clusters": int(values.size),
                "mean_improvement": float(values.mean()) if values.size else None,
                "bootstrap_95ci": _bootstrap(values, seed + comparison_index * 50 + metric_index),
                "permutation_p_two_sided": _permutation(values, seed + 1000 + comparison_index * 50 + metric_index),
                "positive_clusters": int(np.count_nonzero(values > 0)),
                "negative_clusters": int(np.count_nonzero(values < 0)),
                "zero_clusters": int(np.count_nonzero(values == 0)),
                "cluster_names": cluster_names,
            }
        reports.append({"candidate": candidate, "reference": reference, "metrics": metric_reports})

    return {
        "status": "COMPLETE_VIDEO_CLUSTER_PAIRED_AUDIT_SEED17",
        "dataset": dataset,
        "scope": "video-cluster-balanced paired diagnostics on fixed event and candidate cases",
        "source_case_count": len(cases),
        "candidate_present_case_count": int(present_mask.sum()),
        "candidate_absent_case_count": int((~present_mask).sum()),
        "video_count": len(videos),
        "videos_with_candidate_present_cases": len(groups),
        "controls": list(controls),
        "comparisons": reports,
        "bootstrap_draws": 5000,
        "permutation_draws": 10000,
        "seed": seed,
        "target_GT_used_for_training_or_gate": False,
        "offline_GT_used_for_diagnosis": True,
        "sealed_used": False,
        "interpretation": "Each video contributes one cluster mean; positive means candidate improves the metric over reference. This is a video-balanced diagnostic, not a multi-seed claim.",
        "limitations": [
            "video cluster means change the estimand from event-weighted to video-balanced and are reported alongside, not instead of, event-level results",
            "the cases inherit the recorded rollout and candidate-subset limits of the source audit",
            "UA uses the existing legacy-track replay and reconstructed geometry candidates",
            "one seed and no multiple-comparison correction are used",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(load_cases(args.cases), dataset=args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("VIDEO_CLUSTER_PAIRED_COMPLETE", args.dataset, report["video_count"], report["candidate_present_case_count"], flush=True)


if __name__ == "__main__":
    main()
