"""Paired uncertainty audit for the fixed BDD common-state cases.

Every control in this audit receives the same recorded history and candidate
set.  This module therefore compares per-event outcomes directly instead of
comparing independently aggregated means.  It is an offline identity audit:
labels are consumed only after the score arrays and the candidate-present
filter have been fixed in the input JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .common_state_audit import event_metric


DEFAULT_CONTROLS = (
    "history_last",
    "mlp_ce",
    "mlp_jepa_predictive",
    "mlp_jepa_ranking",
    "transformer_ce",
    "transformer_jepa_predictive",
    "transformer_jepa_ranking",
)

DEFAULT_COMPARISONS = (
    ("mlp_jepa_predictive", "mlp_ce"),
    ("mlp_jepa_predictive", "history_last"),
    ("mlp_jepa_predictive", "mlp_jepa_ranking"),
    ("transformer_jepa_predictive", "transformer_ce"),
    ("transformer_jepa_predictive", "history_last"),
    ("transformer_jepa_predictive", "transformer_jepa_ranking"),
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not cases:
        raise ValueError(f"no cases in {path}")
    keys = []
    for case in cases:
        key = (case.get("video"), int(case.get("frame", -1)),
               int(case.get("track_id", -1)), case.get("branch"))
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError("common-state input contains duplicate event keys")
    return cases


def _finite(values: Iterable[float | None]) -> np.ndarray:
    return np.asarray([float(value) for value in values if value is not None and np.isfinite(value)], dtype=np.float64)


def paired_bootstrap(values: np.ndarray, *, seed: int, draws: int = 5000) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    if values.size == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    chunk = 256
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def paired_permutation_pvalue(values: np.ndarray, *, seed: int, draws: int = 10000) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    observed = abs(float(values.mean()))
    if observed == 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 256
    for start in range(0, draws, chunk):
        stop = min(draws, start + chunk)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(stop - start, values.size))
        null_means = (signs * values).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(null_means) >= observed))
    return float((extreme + 1) / (draws + 1))


def _winner_signal(case: dict[str, Any], result: dict[str, Any]) -> tuple[float, float, float]:
    """Return net correction, recovered error, and introduced error."""
    expected = case.get("expected_gt_id")
    native = case.get("native_gt_id")
    winner = result.get("winner_gt_id")
    recovered = float(native != expected and winner == expected)
    harmful = float(native == expected and winner != expected)
    return recovered - harmful, recovered, harmful


def _control_results(cases: list[dict[str, Any]], controls: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    for control in controls:
        values = [event_metric(case, control) for case in cases]
        results[control] = values
    present = {
        control: tuple(bool(result["candidate_present"]) for result in values)
        for control, values in results.items()
    }
    reference = next(iter(present.values()))
    if any(mask != reference for mask in present.values()):
        raise ValueError("candidate-present masks differ across controls")
    return results


def _summary(values: np.ndarray, *, seed: int, higher_is_better: bool = True) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"paired_cases": 0, "mean_improvement": None, "median_improvement": None,
                "bootstrap_95ci": None, "permutation_p_two_sided": None}
    if not higher_is_better:
        values = -values
    return {
        "paired_cases": int(values.size),
        "mean_improvement": float(values.mean()),
        "median_improvement": float(np.median(values)),
        "bootstrap_95ci": paired_bootstrap(values, seed=seed),
        "permutation_p_two_sided": paired_permutation_pvalue(values, seed=seed + 1000),
    }


def _mean_finite(rows: Iterable[dict[str, Any]], key: str) -> tuple[float | None, int]:
    values = _finite(row.get(key) for row in rows)
    return (float(values.mean()), int(values.size)) if values.size else (None, 0)


def compare(cases: list[dict[str, Any]], *, controls: Iterable[str] = DEFAULT_CONTROLS,
            comparisons: Iterable[tuple[str, str]] = DEFAULT_COMPARISONS,
            seed: int = 17) -> dict[str, Any]:
    controls = tuple(controls)
    comparison_list = tuple(comparisons)
    results = _control_results(cases, controls)
    present_mask = np.asarray([
        bool(result["candidate_present"])
        for result in results[controls[0]]
    ], dtype=bool)
    present_cases = [case for case, present in zip(cases, present_mask) if present]
    case_keys = [
        f"{case.get('video')}:{int(case.get('frame', -1))}:{int(case.get('track_id', -1))}:{case.get('branch')}"
        for case, present in zip(cases, present_mask) if present
    ]
    key_digest = hashlib.sha256("\n".join(case_keys).encode()).hexdigest()

    control_summary: dict[str, Any] = {}
    for offset, control in enumerate(controls):
        selected = [result for result, present in zip(results[control], present_mask) if present]
        signals = [_winner_signal(case, result) for case, result in zip(present_cases, selected)]
        mrr, mrr_count = _mean_finite(selected, "mrr")
        auc, auc_count = _mean_finite(selected, "pairwise_auc")
        nll, nll_count = _mean_finite(selected, "nll")
        brier, brier_count = _mean_finite(selected, "brier")
        control_summary[control] = {
            "candidate_present_cases": len(selected),
            "top1": (float(np.mean([bool(result["top1"]) for result in selected]))
                     if selected else None),
            "mrr": mrr,
            "mrr_finite_cases": mrr_count,
            "pairwise_auc": auc,
            "pairwise_auc_finite_cases": auc_count,
            "nll_uncalibrated": nll,
            "nll_finite_cases": nll_count,
            "brier_uncalibrated": brier,
            "brier_finite_cases": brier_count,
            "net_correction": float(sum(signal[0] for signal in signals)),
            "recovered_errors": int(sum(signal[1] for signal in signals)),
            "introduced_errors": int(sum(signal[2] for signal in signals)),
        }

    comparison_reports = []
    for index, (candidate, reference) in enumerate(comparison_list):
        if candidate not in results or reference not in results:
            raise ValueError(f"unknown comparison control: {candidate}, {reference}")
        candidate_rows = [row for row, present in zip(results[candidate], present_mask) if present]
        reference_rows = [row for row, present in zip(results[reference], present_mask) if present]
        candidate_signals = [_winner_signal(case, row) for case, row in zip(present_cases, candidate_rows)]
        reference_signals = [_winner_signal(case, row) for case, row in zip(present_cases, reference_rows)]

        def metric_delta(name: str, *, higher_is_better: bool = True) -> dict[str, Any]:
            raw = np.asarray([
                float(candidate_row[name]) - float(reference_row[name])
                for candidate_row, reference_row in zip(candidate_rows, reference_rows)
                if candidate_row.get(name) is not None and reference_row.get(name) is not None
                and np.isfinite(float(candidate_row[name])) and np.isfinite(float(reference_row[name]))
            ], dtype=np.float64)
            return _summary(raw, seed=seed + index * 20 + len(name), higher_is_better=higher_is_better)

        top1_delta = np.asarray([
            float(bool(candidate_row["top1"])) - float(bool(reference_row["top1"]))
            for candidate_row, reference_row in zip(candidate_rows, reference_rows)
        ], dtype=np.float64)
        net_delta = np.asarray([
            candidate_signal[0] - reference_signal[0]
            for candidate_signal, reference_signal in zip(candidate_signals, reference_signals)
        ], dtype=np.float64)
        recovered_delta = np.asarray([
            candidate_signal[1] - reference_signal[1]
            for candidate_signal, reference_signal in zip(candidate_signals, reference_signals)
        ], dtype=np.float64)
        harmful_delta = np.asarray([
            candidate_signal[2] - reference_signal[2]
            for candidate_signal, reference_signal in zip(candidate_signals, reference_signals)
        ], dtype=np.float64)
        comparison_reports.append({
            "candidate": candidate,
            "reference": reference,
            "paired_cases": int(len(candidate_rows)),
            "improvement_convention": "positive means candidate is better; NLL/Brier use reference minus candidate",
            "metrics": {
                "top1": _summary(top1_delta, seed=seed + index * 20 + 1),
                "mrr": metric_delta("mrr"),
                "pairwise_auc": metric_delta("pairwise_auc"),
                "nll_uncalibrated": metric_delta("nll", higher_is_better=False),
                "brier_uncalibrated": metric_delta("brier", higher_is_better=False),
                "net_correction": _summary(net_delta, seed=seed + index * 20 + 6),
                "recovered_errors": _summary(recovered_delta, seed=seed + index * 20 + 7),
                "introduced_errors": _summary(harmful_delta, seed=seed + index * 20 + 8, higher_is_better=False),
            },
        })

    return {
        "status": "COMPLETE_PAIRED_COMMON_STATE_COMPARISON_SEED17",
        "scope": "paired per-event comparison on the fixed BDD development common-state audit; no new training or rollout",
        "source_cases": "runs/JEPA_IDENTITY/h1_wave_b/common_state_cases.jsonl",
        "source_case_count": len(cases),
        "candidate_present_case_count": int(present_mask.sum()),
        "candidate_absent_case_count": int((~present_mask).sum()),
        "paired_event_key_sha256": key_digest,
        "controls": list(controls),
        "control_summary_candidate_present": control_summary,
        "comparisons": comparison_reports,
        "bootstrap_draws": 5000,
        "permutation_draws": 10000,
        "seed": seed,
        "target_GT_used_for_training_or_gate": False,
        "offline_GT_used_for_diagnosis": True,
        "sealed_used": False,
        "limitations": [
            "the cases are recorded predictive-residual rollout states and not an independent all-opportunity baseline",
            "candidate-absent cases are excluded from ranking metrics and are reported separately rather than imputed",
            "paired permutation p-values and bootstrap intervals are descriptive uncertainty diagnostics for one seed, not a multi-seed confirmatory claim",
            "NLL and Brier use the existing uncalibrated softmax(-cost) diagnostic",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = compare(load_cases(args.cases))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("PAIRED_COMMON_STATE_COMPLETE", report["source_case_count"], report["candidate_present_case_count"], flush=True)


if __name__ == "__main__":
    main()
