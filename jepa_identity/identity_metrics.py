"""Event-level identity ranking metrics for the Phase-1 offline audit.

The input is the JSONL emitted by :mod:`identity_event_audit`.  Metrics are
conditioned on the recorded changed-assignment events and are reported with
candidate coverage so an absent correct candidate is not mislabelled as a
ranking failure.  Costs are treated as lower-is-better.  The softmax-derived
NLL/Brier/ECE values are explicitly uncalibrated diagnostics because the
recorded matcher costs were not fit as probabilities in this audit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable


EVALUABLE = {
    "corrected_local_identity",
    "harmful_local_identity",
    "both_correct",
    "neither_correct",
}
SCORE_KEYS = {
    "native": "native_cost",
    "intervention": "fused_cost",
}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def bootstrap_interval(values: Iterable[float], seed: int = 17, draws: int = 500) -> list[float] | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return [means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]]


def softmax_negative(costs: list[float]) -> list[float]:
    if not costs:
        return []
    finite = [value if math.isfinite(value) else 1e6 for value in costs]
    pivot = min(finite)
    weights = [math.exp(max(-80.0, min(80.0, -(value - pivot)))) for value in finite]
    total = sum(weights)
    return [weight / total for weight in weights] if total else [1.0 / len(weights)] * len(weights)


def ranking_case(case: dict[str, Any], score_key: str) -> dict[str, Any] | None:
    if case.get("diagnosis") not in EVALUABLE:
        return None
    expected = case.get("expected_gt_id")
    candidates = [row for row in case.get("candidate_gt_ids", []) if row.get(score_key) is not None]
    if expected is None:
        return None
    if not candidates:
        return {
            "candidate_present": False,
            "top1": None,
            "rank": None,
            "mrr": None,
            "recall_at": {"1": False, "3": False, "5": False},
            "pairwise_auc": None,
            "average_precision": None,
            "margin": None,
            "nll": None,
            "brier": None,
            "winner_confidence": None,
            "winner_correct": None,
        }
    ordered = sorted(candidates, key=lambda row: (float(row[score_key]), int(row.get("index", 0))))
    positive_positions = [index for index, row in enumerate(ordered) if row.get("gt_id") == expected]
    if not positive_positions:
        return {
            "candidate_present": False,
            "top1": None,
            "rank": None,
            "mrr": None,
            "recall_at": {"1": False, "3": False, "5": False},
            "pairwise_auc": None,
            "average_precision": None,
            "margin": None,
            "nll": None,
            "brier": None,
            "winner_confidence": None,
            "winner_correct": None,
        }
    rank = positive_positions[0] + 1
    positive = [row for row in ordered if row.get("gt_id") == expected]
    negatives = [row for row in ordered if row.get("gt_id") != expected]
    pair_values = []
    for positive_row in positive:
        positive_cost = float(positive_row[score_key])
        for negative_row in negatives:
            negative_cost = float(negative_row[score_key])
            pair_values.append(1.0 if positive_cost < negative_cost else 0.5 if positive_cost == negative_cost else 0.0)
    pairwise_auc = sum(pair_values) / len(pair_values) if pair_values else None
    precision_values = []
    positive_seen = 0
    for position, row in enumerate(ordered, start=1):
        if row.get("gt_id") == expected:
            positive_seen += 1
            precision_values.append(positive_seen / position)
    average_precision = sum(precision_values) / len(precision_values) if precision_values else None
    best_positive = min(float(row[score_key]) for row in positive)
    best_negative = min((float(row[score_key]) for row in negatives), default=None)
    margin = None if best_negative is None else best_negative - best_positive
    probabilities = softmax_negative([float(row[score_key]) for row in ordered])
    positive_probability = sum(probability for probability, row in zip(probabilities, ordered) if row.get("gt_id") == expected)
    winner_correct = ordered[0].get("gt_id") == expected
    winner_confidence = probabilities[0]
    return {
        "candidate_present": True,
        "top1": bool(winner_correct),
        "rank": rank,
        "mrr": 1.0 / rank,
        "recall_at": {str(k): rank <= k for k in (1, 3, 5)},
        "pairwise_auc": pairwise_auc,
        "average_precision": average_precision,
        "margin": margin,
        "nll": -math.log(max(1e-12, positive_probability)),
        "brier": (positive_probability - 1.0) ** 2,
        "winner_confidence": winner_confidence,
        "winner_correct": bool(winner_correct),
    }


def summarize_cases(cases: list[dict[str, Any]], seed: int = 17) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        if case.get("diagnosis") in EVALUABLE:
            grouped[str(case.get("method"))][str(case.get("domain"))].append(case)

    def summarize(method_cases: list[dict[str, Any]], score_name: str, local_seed: int) -> dict[str, Any]:
        score_key = SCORE_KEYS[score_name]
        all_results = [ranking_case(case, score_key) for case in method_cases]
        all_results = [result for result in all_results if result is not None]
        eligible = [result for result in all_results if result["candidate_present"]]
        coverage_total = len(all_results)
        coverage_count = len(eligible)

        def values(key: str) -> list[float]:
            return [float(result[key]) for result in eligible if result.get(key) is not None]

        top1_count = sum(bool(result["top1"]) for result in eligible)
        output: dict[str, Any] = {
            "evaluable_changed_events": len(method_cases),
            "events_with_scored_candidates": coverage_total,
            "correct_candidate_present": coverage_count,
            "correct_candidate_absent": coverage_total - coverage_count,
            "candidate_coverage_rate": coverage_count / coverage_total if coverage_total else None,
            "candidate_coverage_95ci": wilson_interval(coverage_count, coverage_total),
            "top1_identity_accuracy_conditional": top1_count / coverage_count if coverage_count else None,
            "top1_identity_accuracy_95ci": wilson_interval(top1_count, coverage_count),
            "mrr_conditional": sum(values("mrr")) / len(values("mrr")) if values("mrr") else None,
            "mrr_95ci": bootstrap_interval(values("mrr"), local_seed + 1),
            "recall_at_k_conditional": {
                str(k): sum(bool(result["recall_at"][str(k)]) for result in eligible) / coverage_count if coverage_count else None
                for k in (1, 3, 5)
            },
            "pairwise_auc": sum(values("pairwise_auc")) / len(values("pairwise_auc")) if values("pairwise_auc") else None,
            "pairwise_auc_95ci": bootstrap_interval(values("pairwise_auc"), local_seed + 2),
            "average_precision": sum(values("average_precision")) / len(values("average_precision")) if values("average_precision") else None,
            "average_precision_95ci": bootstrap_interval(values("average_precision"), local_seed + 3),
            "positive_vs_best_negative_margin": sum(values("margin")) / len(values("margin")) if values("margin") else None,
            "positive_vs_best_negative_margin_95ci": bootstrap_interval(values("margin"), local_seed + 4),
            "candidate_nll_uncalibrated": sum(values("nll")) / len(values("nll")) if values("nll") else None,
            "candidate_nll_uncalibrated_95ci": bootstrap_interval(values("nll"), local_seed + 5),
            "candidate_brier_uncalibrated": sum(values("brier")) / len(values("brier")) if values("brier") else None,
            "candidate_brier_uncalibrated_95ci": bootstrap_interval(values("brier"), local_seed + 6),
            "winner_ece_uncalibrated": None,
            "score_definition": "lower matcher cost is better; candidate list is the tracer's recorded top subset",
            "probability_warning": "NLL/Brier/ECE use softmax(-recorded cost) without a fresh calibration fit and are diagnostic only",
        }
        confidence_bins = [[] for _ in range(10)]
        for result in eligible:
            confidence = float(result["winner_confidence"])
            confidence_bins[min(9, int(confidence * 10))].append(result)
        ece = 0.0
        for bin_results in confidence_bins:
            if not bin_results:
                continue
            confidence_mean = sum(float(result["winner_confidence"]) for result in bin_results) / len(bin_results)
            accuracy_mean = sum(bool(result["winner_correct"]) for result in bin_results) / len(bin_results)
            ece += len(bin_results) / coverage_count * abs(confidence_mean - accuracy_mean)
        output["winner_ece_uncalibrated"] = ece if coverage_count else None
        return output

    result: dict[str, Any] = {
        "status": "COMPLETE_OFFLINE_IDENTITY_RANKING_METRICS",
        "scope": "existing changed-assignment events only; not all association opportunities",
        "evaluable_diagnoses": sorted(EVALUABLE),
        "methods": {},
        "limitations": [
            "a correct target absent from the recorded candidate subset is reported as coverage loss, not ranking loss",
            "the event set is conditioned on changed assignments and each method's own rollout state",
            "native and intervention scores are read from the tracer; no common-state rerun or independent calibration was performed",
            "random_rank_reward has no changed-event rows in this audit and therefore has no event-level ranking estimate",
        ],
    }
    for method, by_domain in sorted(grouped.items()):
        method_cases = [case for rows in by_domain.values() for case in rows]
        method_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(method)) % 1000
        result["methods"][method] = {
            "event_count_by_domain": {domain: len(rows) for domain, rows in sorted(by_domain.items())},
            "native": summarize(method_cases, "native", method_seed),
            "intervention": summarize(method_cases, "intervention", method_seed + 1000),
            "local_diagnosis_counts": {
                diagnosis: sum(1 for case in method_cases if case.get("diagnosis") == diagnosis)
                for diagnosis in sorted(EVALUABLE)
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    report = summarize_cases(cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("IDENTITY_RANKING_METRICS_COMPLETE", len(report["methods"]), flush=True)


if __name__ == "__main__":
    main()
