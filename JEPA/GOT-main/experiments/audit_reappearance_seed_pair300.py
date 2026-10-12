#!/usr/bin/env python3
"""Aggregate the two fixed-seed invariant reappearance runs.

Read-only: this script consumes the committed JSON/CSV reports and never
opens images or ground-truth annotations.  It verifies that both runs passed
their protocol audits, that the raw semantic hashes agree, and computes
paired mean/std summaries for the official metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List


METHODS = ("raw", "reappearance_jepa", "no_teacher", "random_frozen")
METRICS = ("HOTA_official_mean", "HOTA_official_alpha0.50", "AssA_official_alpha0.50",
           "DetA_official_alpha0.50", "IDF1_official", "MOTA_official", "IDSW_official")
SEQUENCES = ("dancetrack0004", "dancetrack0007", "dancetrack0014", "COMBINED_SEQ")


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def num(x: Any) -> float:
    return float(x)


def stats(values: List[float]) -> Dict[str, Any]:
    vals = [float(x) for x in values]
    return {"values": vals, "mean": statistics.fmean(vals),
            "std_population": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "std_sample": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed17-dir", required=True)
    ap.add_argument("--seed23-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    dirs = {17: Path(args.seed17_dir).resolve(), 23: Path(args.seed23_dir).resolve()}
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    runs: Dict[int, Dict[str, Any]] = {}
    for seed, root in dirs.items():
        required = ["summary.json", "results.json", "reflection.json", "protocol.json",
                    "provenance.json", "reappearance_protocol_audit.json",
                    "trackeval_official/audit.json"]
        for rel in required:
            if not (root / rel).exists():
                failures.append(f"seed{seed}:missing:{rel}")
        if failures and not all((root / rel).exists() for rel in required):
            continue
        summary = load(root / "summary.json"); results = load(root / "results.json")
        reflection = load(root / "reflection.json"); protocol = load(root / "protocol.json")
        provenance = load(root / "provenance.json"); audit = load(root / "reappearance_protocol_audit.json")
        te = load(root / "trackeval_official/audit.json")
        if summary.get("status") not in ("positive", "no_positive_gain"):
            failures.append(f"seed{seed}:summary_status_invalid:{summary.get('status')}")
        if audit.get("status") != "pass": failures.append(f"seed{seed}:protocol_audit_not_pass")
        if te.get("status") != "complete" or te.get("failures") != []: failures.append(f"seed{seed}:trackeval_not_complete")
        runs[seed] = {"root": str(root), "summary": summary, "results": results,
                      "reflection": reflection, "protocol": protocol,
                      "provenance": provenance, "audit": audit, "trackeval": te}

    # Raw is deterministic by design; this equality also guards against a
    # silent seed-dependent lifecycle change.
    raw_hashes: Dict[str, Dict[str, Any]] = {}
    for seq in SEQUENCES[:-1]:
        vals = []
        for seed in (17, 23):
            try:
                vals.append((runs[seed]["results"]["dance"]["sequences"][seq]["raw"]["semantic_sha256"]))
            except Exception:
                vals.append(None)
        raw_hashes[seq] = {"values": vals, "match": vals[0] is not None and vals[0] == vals[1]}
        if not raw_hashes[seq]["match"]: failures.append(f"raw_semantic_mismatch:{seq}")

    combined_by_seed: Dict[str, Any] = {}
    aggregate: Dict[str, Any] = {}
    for method in METHODS:
        combined_by_seed[method] = {}
        aggregate[method] = {}
        for metric in METRICS:
            vals = []
            for seed in (17, 23):
                vals.append(num(runs[seed]["summary"]["combined"][method][metric]))
                combined_by_seed[method].setdefault(str(seed), {})[metric] = vals[-1]
            aggregate[method][metric] = stats(vals)

    paired: Dict[str, Any] = {}
    for metric in METRICS:
        vals = [num(runs[s]["summary"]["combined"]["reappearance_jepa"][metric]) -
                num(runs[s]["summary"]["combined"]["raw"][metric]) for s in (17, 23)]
        paired[metric] = stats(vals)

    per_sequence: Dict[str, Any] = {}
    for seq in SEQUENCES:
        per_sequence[seq] = {}
        for metric in METRICS:
            vals = [num(runs[s]["trackeval"]["combined"]["reappearance_jepa"][metric]) -
                    num(runs[s]["trackeval"]["combined"]["raw"][metric]) for s in (17, 23)] if seq == "COMBINED_SEQ" else [
                        num(next(x for x in runs[s]["trackeval"]["official_rows"] if x["sequence"] == seq and x["method"] == "reappearance_jepa")[metric]) -
                        num(next(x for x in runs[s]["trackeval"]["official_rows"] if x["sequence"] == seq and x["method"] == "raw")[metric]) for s in (17, 23)]
            per_sequence[seq][metric] = stats(vals)

    # Ensure raw protocol/checkpoint identity across seeds.
    ckpts = [runs[s]["results"].get("checkpoint_sha256") for s in (17, 23)]
    a_caches = [runs[s]["results"].get("cache_dir") for s in (17, 23)]
    if ckpts[0] != ckpts[1]: failures.append("checkpoint_mismatch")
    if a_caches[0] != a_caches[1]: failures.append("cache_dir_mismatch")
    candidate_positive_by_seed = {
        str(seed): bool(num(runs[seed]["summary"]["combined"]["reappearance_jepa"]["HOTA_official_mean"]) > num(runs[seed]["summary"]["combined"]["raw"]["HOTA_official_mean"]) and
                           num(runs[seed]["summary"]["combined"]["reappearance_jepa"]["IDF1_official"]) > num(runs[seed]["summary"]["combined"]["raw"]["IDF1_official"]))
        for seed in (17, 23)
    }
    positive_both = all(candidate_positive_by_seed.values())

    candidate_delta_by_seed = {
        str(seed): {metric: num(runs[seed]["summary"]["combined"]["reappearance_jepa"][metric]) -
                    num(runs[seed]["summary"]["combined"]["raw"][metric]) for metric in METRICS}
        for seed in (17, 23)
    }
    report = {"schema": "dance-got-jepa-reappearance300-seed-pair-audit-v1",
              "status": "pass" if not failures else "failure", "seeds": [17, 23],
              "run_dirs": {str(k): v["root"] for k, v in runs.items()},
              "dataset": "DanceTrack", "max_frames": 300, "gt_read": False,
              "failures": failures, "checkpoint_sha256": ckpts,
              "cache_dirs": a_caches, "raw_semantic_hashes": raw_hashes,
              "combined_by_seed": combined_by_seed, "aggregate": aggregate,
              "paired_reappearance_vs_raw": paired, "candidate_delta_by_seed": candidate_delta_by_seed,
              "per_sequence_reappearance_vs_raw": per_sequence,
              "candidate_positive_by_seed": candidate_positive_by_seed,
              "candidate_positive_both_seeds": positive_both,
              "controls": {"no_teacher": "required control; teacher attribution is not established", "random_frozen": "negative/control feature sanity check; not an attribution baseline"},
              "interpretation": ("The fixed reappearance-only predictor is positive for HOTA mean and IDF1 in both 300-frame seeds; effect magnitude and safety are reported without threshold tuning." if positive_both else "The fixed reappearance-only predictor is not positive for both primary endpoints across 300-frame seeds; stop further sweep/expansion and retain the 120-frame result as boundary evidence."),
              "python": sys.version, "platform": platform.platform()}
    (out / "seed_pair_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    # Flat CSV companions make the per-seed/per-sequence values easy to plot
    # without changing or re-reading any experiment artifact.
    metric_csv = out / "seed_pair_metrics.csv"
    with metric_csv.open("w", newline="") as f:
        fields = ["seed", "sequence", "method", *METRICS]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for seed in (17, 23):
            for seq in SEQUENCES:
                rows = runs[seed]["trackeval"].get("official_rows", [])
                row = next(x for x in rows if x.get("sequence") == seq and x.get("method") == "raw") if seq != "COMBINED_SEQ" else runs[seed]["trackeval"]["combined"]["raw"]
                for method in METHODS:
                    src = next(x for x in rows if x.get("sequence") == seq and x.get("method") == method) if seq != "COMBINED_SEQ" else runs[seed]["trackeval"]["combined"][method]
                    w.writerow({"seed": seed, "sequence": seq, "method": method, **{k: num(src[k]) for k in METRICS}})
    delta_csv = out / "seed_pair_paired_delta.csv"
    with delta_csv.open("w", newline="") as f:
        fields = ["sequence", *METRICS]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for seq in SEQUENCES:
            w.writerow({"sequence": seq, **{k: per_sequence[seq][k]["mean"] for k in METRICS}})
    report["csv"] = {"metrics": str(metric_csv), "paired_delta_mean": str(delta_csv)}
    (out / "seed_pair_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    (out / "reflection.json").write_text(json.dumps({"schema": "dance-got-jepa-reappearance300-seed-pair-reflection-v1", "status": report["status"], "candidate_positive_by_seed": candidate_positive_by_seed, "candidate_positive_both_seeds": positive_both, "paired_delta": paired, "controls": report["controls"], "teacher_attribution": "not established; no_teacher is a required control", "next_step": "stop expansion and retain 120f as boundary evidence" if not positive_both else "no further sweep; consider integration only after external validation", "no_gt": True}, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"status": report["status"], "out": str(out / "seed_pair_audit.json"), "failures": failures, "paired_delta": paired}, indent=2, sort_keys=True, default=float))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
