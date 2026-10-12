#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PHASE1_ROOT = REPO / "work_dirs/limit_search_gpu23/phase1_threshold"
FULL_ROOT = REPO / "work_dirs/limit_search_gpu23/full17_threshold"
DEFAULT_OUTPUT = REPO / "work_dirs/limit_search_gpu23/darth_4dataset_best_thresholds.csv"

CANDIDATES = {
    "base": (0.05, 0.20, 0.20),
    "det01": (0.01, 0.20, 0.20),
    "det03": (0.03, 0.20, 0.20),
    "det10": (0.10, 0.20, 0.20),
    "init01": (0.05, 0.10, 0.20),
    "init03": (0.05, 0.30, 0.20),
    "init07": (0.05, 0.70, 0.20),
    "obj01": (0.05, 0.20, 0.10),
    "obj05": (0.05, 0.20, 0.50),
}

DATASETS = {
    "bdd": ["overcast", "rainy", "dawn_dusk", "night"],
    "motsynth": ["cloudy", "fog", "rain", "snow", "dark"],
    "ua_detrac": ["cloudy", "rainy", "night"],
    "vkitti2": ["overcast", "morning", "rain", "fog", "sunset"],
}

SOURCES = {
    "track": (
        "track_metrics.csv",
        ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn", "MT", "PT", "ML", "FM"],
    ),
    "hota": (
        "hota_metrics.csv",
        ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA", "HOTA", "Dets", "IDs"],
    ),
    "bbox": (
        "bbox_metrics.csv",
        ["mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l"],
    ),
}

# Metrics with a well-defined optimization direction. PT, Dets and IDs are
# retained in source artifacts but excluded because they are not monotonic
# quality objectives on their own.
METRIC_DIRECTIONS = [
    ("track/IDF1", "max"),
    ("track/MOTA", "max"),
    ("track/MOTP", "max"),
    ("track/FP", "min"),
    ("track/FN", "min"),
    ("track/IDSw", "min"),
    ("track/Rcll", "max"),
    ("track/Prcn", "max"),
    ("track/MT", "max"),
    ("track/ML", "min"),
    ("track/FM", "min"),
    ("hota/HOTA", "max"),
    ("hota/DetA", "max"),
    ("hota/AssA", "max"),
    ("hota/LocA", "max"),
    ("hota/DetRe", "max"),
    ("hota/DetPr", "max"),
    ("hota/AssRe", "max"),
    ("hota/AssPr", "max"),
    ("bbox/mAP", "max"),
    ("bbox/mAP_50", "max"),
    ("bbox/mAP_75", "max"),
    ("bbox/mAP_s", "max"),
    ("bbox/mAP_m", "max"),
    ("bbox/mAP_l", "max"),
]

SUM_METRICS = {
    "track/FP",
    "track/FN",
    "track/IDSw",
    "track/MT",
    "track/PT",
    "track/ML",
    "track/FM",
    "hota/Dets",
    "hota/IDs",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def read_overall(path):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("") or row.get("class") or row.get("name")
            if label == "OVERALL":
                return row
    raise ValueError(f"OVERALL row missing: {path}")


def find_run_dir(dataset, domain, candidate):
    for root in (FULL_ROOT, PHASE1_ROOT):
        task_dir = root / dataset / domain / candidate
        if not (task_dir / "done.marker").is_file():
            continue
        summaries = sorted(
            task_dir.glob("*/*/summary_metrics.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for summary in summaries:
            run_dir = summary.parent
            if all((run_dir / filename).is_file() for filename, _ in SOURCES.values()):
                return run_dir
    return None


def read_metrics(run_dir):
    metrics = {}
    for prefix, (filename, names) in SOURCES.items():
        row = read_overall(run_dir / filename)
        for name in names:
            raw = row.get(name, "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = math.nan
            metrics[f"{prefix}/{name}"] = value
    return metrics


def aggregate(values, metric):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan
    if metric in SUM_METRICS:
        return sum(finite)
    return sum(finite) / len(finite)


def format_value(value):
    if not math.isfinite(value):
        return ""
    return f"{value:.10g}"


def main():
    args = parse_args()
    domain_metrics = {dataset: {} for dataset in DATASETS}
    missing = []

    for dataset, domains in DATASETS.items():
        for candidate in CANDIDATES:
            domain_metrics[dataset][candidate] = {}
            for domain in domains:
                run_dir = find_run_dir(dataset, domain, candidate)
                if run_dir is None:
                    missing.append((dataset, domain, candidate))
                    continue
                domain_metrics[dataset][candidate][domain] = read_metrics(run_dir)

    if missing and not args.allow_partial:
        print(f"INCOMPLETE: {len(missing)} of {sum(len(v) for v in DATASETS.values()) * len(CANDIDATES)} tasks missing")
        for item in missing[:20]:
            print("  missing=" + "/".join(item))
        return 2

    rows = []
    for dataset, domains in DATASETS.items():
        dataset_complete = all(
            len(domain_metrics[dataset][candidate]) == len(domains)
            for candidate in CANDIDATES
        )
        eligible = [
            candidate
            for candidate in CANDIDATES
            if domain_metrics[dataset][candidate]
            and (args.allow_partial or len(domain_metrics[dataset][candidate]) == len(domains))
        ]
        if not eligible:
            continue

        for metric, direction in METRIC_DIRECTIONS:
            aggregate_values = {}
            for candidate in eligible:
                values = [
                    metrics.get(metric, math.nan)
                    for metrics in domain_metrics[dataset][candidate].values()
                ]
                aggregate_values[candidate] = aggregate(values, metric)
            finite = {
                candidate: value
                for candidate, value in aggregate_values.items()
                if math.isfinite(value)
            }
            if not finite:
                continue
            best_value = max(finite.values()) if direction == "max" else min(finite.values())
            ties = [
                candidate
                for candidate in CANDIDATES
                if candidate in finite
                and math.isclose(finite[candidate], best_value, rel_tol=1e-12, abs_tol=1e-9)
            ]
            best_candidate = ties[0]
            det, init, obj = CANDIDATES[best_candidate]
            best_domains = domain_metrics[dataset][best_candidate]
            domain_values = ";".join(
                f"{domain}:{format_value(best_domains[domain].get(metric, math.nan))}"
                for domain in domains
                if domain in best_domains
            )
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "direction": direction,
                "best_candidate": best_candidate,
                "det_score_thr": det,
                "init_score_thr": init,
                "obj_score_thr": obj,
                "best_value": format_value(best_value),
                "unit": "count" if metric in SUM_METRICS else "percent",
                "aggregation": "sum_domains" if metric in SUM_METRICS else "macro_mean_domains",
                "domain_values": domain_values,
                "domains_evaluated": len(best_domains),
                "domains_expected": len(domains),
                "candidates_evaluated": len(eligible),
                "tied_candidates": ";".join(ties),
                "complete": str(dataset_complete).lower(),
            })

    if not rows:
        print("No metric rows available")
        return 3

    fieldnames = list(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_output.replace(args.output)
    print(f"WROTE {args.output} rows={len(rows)} missing={len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
