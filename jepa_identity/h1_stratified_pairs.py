"""Domain-stratified paired comparisons for the fixed BDD H1 cases."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .paired_common_state import DEFAULT_COMPARISONS, DEFAULT_CONTROLS, compare, load_cases


def compact(report: dict[str, Any]) -> dict[str, Any]:
    metrics = ("top1", "mrr", "pairwise_auc", "net_correction")
    comparisons = []
    for comparison in report["comparisons"]:
        comparisons.append({
            "candidate": comparison["candidate"],
            "reference": comparison["reference"],
            "paired_cases": comparison["paired_cases"],
            "metrics": {
                name: {
                    "paired_cases": comparison["metrics"][name]["paired_cases"],
                    "mean_improvement": comparison["metrics"][name]["mean_improvement"],
                    "bootstrap_95ci": comparison["metrics"][name]["bootstrap_95ci"],
                    "permutation_p_two_sided": comparison["metrics"][name]["permutation_p_two_sided"],
                }
                for name in metrics
            },
        })
    return {
        "source_case_count": report["source_case_count"],
        "candidate_present_case_count": report["candidate_present_case_count"],
        "candidate_absent_case_count": report["candidate_absent_case_count"],
        "comparisons": comparisons,
    }


def stratify(cases: list[dict[str, Any]], *, controls: Iterable[str] = DEFAULT_CONTROLS,
             comparisons: Iterable[tuple[str, str]] = DEFAULT_COMPARISONS,
             seed: int = 17) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case.get("domain", "unknown"))].append(case)
    reports = {}
    for index, domain in enumerate(sorted(groups)):
        reports[domain] = compact(compare(groups[domain], controls=controls,
                                          comparisons=comparisons, seed=seed + index * 100))
    return {
        "status": "COMPLETE_BDD_DOMAIN_STRATIFIED_PAIRED_COMPARISON_SEED17",
        "scope": "same-event paired uncertainty diagnostics within each BDD development domain",
        "source_cases": "runs/JEPA_IDENTITY/h1_wave_b/common_state_cases.jsonl",
        "source_case_count": len(cases),
        "domains": reports,
        "controls": list(controls),
        "comparisons": [list(pair) for pair in comparisons],
        "bootstrap_draws": 5000,
        "permutation_draws": 10000,
        "seed": seed,
        "target_GT_used_for_training_or_gate": False,
        "offline_GT_used_for_diagnosis": True,
        "sealed_used": False,
        "limitations": [
            "the domain strata inherit the recorded predictive-residual rollout state and top-candidate subset",
            "candidate-absent events are retained as coverage counts and excluded from ranking deltas",
            "small rainy and overcast strata and all rare-gap strata should not be read as standalone confirmatory tests",
            "one seed and no multiple-comparison correction are used",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = stratify(load_cases(args.cases))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("H1_DOMAIN_STRATIFIED_COMPLETE", report["source_case_count"], sorted(report["domains"]), flush=True)


if __name__ == "__main__":
    main()
