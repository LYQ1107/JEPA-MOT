#!/usr/bin/env python3
"""Summarize fixed-protocol DanceTrack temporal-JEPA TrackEval runs.

This is a read-only aggregation utility.  It never selects thresholds or
rewrites tracker files; it only joins the official ``official_metrics.csv``
files from the requested seed directories and computes deltas against the
raw/no-teacher/random-frozen controls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


METHODS = ["raw", "jepa_temporal", "no_teacher", "random_frozen"]
METRICS = [
    "HOTA_official_mean", "HOTA_official_alpha0.50", "AssA_official_alpha0.50",
    "DetA_official_alpha0.50", "IDF1_official", "MOTA_official", "IDSW_official",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(root: Path, seed: int) -> list[dict[str, object]]:
    path = root / "trackeval_official" / "official_metrics.csv"
    rows: list[dict[str, object]] = []
    for row in csv.DictReader(path.open(newline="")):
        if row.get("method") not in METHODS:
            continue
        out: dict[str, object] = {"seed": seed, "sequence": row["sequence"], "method": row["method"]}
        out.update({k: float(row[k]) for k in METRICS})
        rows.append(out)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", nargs=2, metavar=("SEED", "DIR"), required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    source: list[dict[str, object]] = []
    for seed_s, root_s in args.run:
        seed = int(seed_s); root = Path(root_s).resolve()
        csv_path = root / "trackeval_official" / "official_metrics.csv"
        rows = read_rows(root, seed)
        all_rows.extend(rows)
        source.append({
            "seed": seed, "run_dir": str(root),
            "results_sha256": sha256(root / "results.json"),
            "protocol_audit_sha256": sha256(root / "protocol_audit.json"),
            "trackeval_audit_sha256": sha256(root / "trackeval_official" / "audit.json"),
            "official_metrics_sha256": sha256(csv_path),
            "protocol_audit_all_pass": bool(json.loads((root / "protocol_audit.json").read_text())["checks"]["all_pass"]),
            "trackeval_status": json.loads((root / "trackeval_official" / "audit.json").read_text()).get("status"),
        })
    # Stable long-form copy of official rows.
    long_path = out / "official_metrics_long.csv"
    with long_path.open("w", newline="") as f:
        fields = ["seed", "sequence", "method"] + METRICS
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)

    index = {(int(r["seed"]), str(r["sequence"]), str(r["method"])): r for r in all_rows}
    seqs = sorted({str(r["sequence"]) for r in all_rows})
    comparisons: list[dict[str, object]] = []
    for seed_s, _ in args.run:
        seed = int(seed_s)
        for seq in seqs:
            base = index[(seed, seq, "raw")]
            for metric in METRICS:
                row: dict[str, object] = {"seed": seed, "sequence": seq, "metric": metric}
                for method in METHODS:
                    row[method] = float(index[(seed, seq, method)][metric])
                for method in ["raw", "no_teacher", "random_frozen"]:
                    row["delta_jepa_vs_" + method] = float(index[(seed, seq, "jepa_temporal")][metric]) - float(index[(seed, seq, method)][metric])
                comparisons.append(row)
    comp_path = out / "comparisons.csv"
    with comp_path.open("w", newline="") as f:
        fields = ["seed", "sequence", "metric"] + METHODS + ["delta_jepa_vs_" + m for m in ["raw", "no_teacher", "random_frozen"]]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(comparisons)

    # Aggregate mean/std over seeds, retaining each sequence separately in the
    # long CSV.  IDSW is reported as a count (and its seed std), not a percent.
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for seq in seqs:
        aggregate[seq] = {}
        for method in METHODS:
            aggregate[seq][method] = {}
            for metric in METRICS:
                vals = [float(index[(int(seed_s), seq, method)][metric]) for seed_s, _ in args.run]
                aggregate[seq][method][metric + "_mean"] = float(statistics.mean(vals))
                aggregate[seq][method][metric + "_std_pop"] = float(statistics.pstdev(vals))
    delta_mean: dict[str, dict[str, dict[str, float]]] = {}
    for seq in seqs:
        delta_mean[seq] = {}
        for control in ["raw", "no_teacher", "random_frozen"]:
            delta_mean[seq][control] = {}
            for metric in METRICS:
                vals = [float(index[(int(seed_s), seq, "jepa_temporal")][metric]) - float(index[(int(seed_s), seq, control)][metric]) for seed_s, _ in args.run]
                delta_mean[seq][control][metric] = float(statistics.mean(vals))
    payload = {
        "schema": "dance-temporal-repro-summary-v1", "methods": METHODS, "metrics": METRICS,
        "runs": source, "sequences": seqs, "aggregate_by_sequence": aggregate,
        "delta_jepa_mean_by_sequence": delta_mean,
        "definition": "Values are vendored TrackEval official HOTA/CLEAR/Identity on the fixed 300-frame DanceTrack staging copy; HOTA mean and alpha=0.50, AssA/DetA alpha=0.50, IDF1/MOTA/IDSW.",
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_dir": str(out), "rows": len(all_rows), "comparisons": len(comparisons), "runs": source}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
