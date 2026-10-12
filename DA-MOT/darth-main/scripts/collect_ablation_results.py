#!/usr/bin/env python3
"""Collect first-round ablation results into per-method CSVs + a compare CSV.

Outputs (under work_dirs/bdd_4domain_ablation/summaries/):
  task2_spa_only_results.csv
  task2_causal_only_results.csv
  task2_spa_causal_results.csv
  task2_ablation_compare.csv     (all methods incl. baseline_task2)
  task2_ablation_summary.md

CSV fields (metrics in canonical reporting order):
  method, stage, epoch, eval_domain,
  MOTA, IDF1, HOTA, DetA, AssA,                # primary tracking metrics
  DetRe, DetPr, AssRe, AssPr, LocA,            # HOTA sub-metrics
  MOTP, Rcll, Prcn, MT, PT, ML,                # CLEAR-MOT rates/counts
  FP, FN, IDSw, FM,                            # CLEAR-MOT error counts
  Dets, IDs,                                   # HOTA detection/id counts
  eval_dir, checkpoint

Parses the OVERALL rows of the CLEAR-MOT table and the HOTA table from each
eval.log. No baseline CSV format is modified; this writes only new files.
"""
import csv
import glob
import os
import os.path as osp
import re

ROOT = "work_dirs/bdd_4domain_ablation"
SUMMARY_DIR = osp.join(ROOT, "summaries")

# {method: (eval_root, checkpoint_root)}
METHODS = {
    "baseline_task2": ("work_dirs/bdd_4domain/task2_full_episode/eval",
                       "work_dirs/bdd_4domain/task2_full_episode/episode/adapt"),
    "spa_only": (osp.join(ROOT, "task2_spa_only/eval"),
                 osp.join(ROOT, "task2_spa_only/episode/adapt")),
    "causal_only": (osp.join(ROOT, "task2_causal_only/eval"),
                    osp.join(ROOT, "task2_causal_only/episode/adapt")),
    "spa_causal": (osp.join(ROOT, "task2_spa_causal/eval"),
                   osp.join(ROOT, "task2_spa_causal/episode/adapt")),
}

META_FIELDS = ["method", "stage", "epoch", "eval_domain"]
# Canonical metric reporting order: primary tracking metrics first, then HOTA
# sub-metrics, then CLEAR-MOT rates, then error counts, then raw counts.
METRIC_FIELDS = [
    "MOTA", "IDF1", "HOTA", "DetA", "AssA",
    "DetRe", "DetPr", "AssRe", "AssPr", "LocA",
    "MOTP", "Rcll", "Prcn", "MT", "PT", "ML",
    "FP", "FN", "IDSw", "FM",
    "Dets", "IDs",
]
TAIL_FIELDS = ["eval_dir", "checkpoint"]
FIELDS = META_FIELDS + METRIC_FIELDS + TAIL_FIELDS

CLEARMOT_HDR = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CLEARMOT_COLS = ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn",
                 "MT", "PT", "ML", "FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS = ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA",
             "HOTA", "Dets", "IDs"]


def _norm(s):
    return re.sub(r"\s+", " ", s.strip())


def parse_eval_log(path):
    """Return dict of metrics from the OVERALL rows, or None."""
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return None
    res = {}
    for hdr, cols in ((CLEARMOT_HDR, CLEARMOT_COLS), (HOTA_HDR, HOTA_COLS)):
        for i, line in enumerate(lines):
            if _norm(line) == hdr:
                for j in range(i + 1, min(i + 15, len(lines))):
                    if lines[j].strip().startswith("OVERALL"):
                        vals = lines[j].split()[1:]
                        for c, v in zip(cols, vals):
                            try:
                                res[c] = float(v)
                            except ValueError:
                                res[c] = v
                        break
                break
    return res if res else None


def find_checkpoint(ckpt_root, stage, epoch):
    p = osp.join(ckpt_root, stage, f"epoch_{epoch}.pth")
    return p if osp.isfile(p) else ""


def collect_method(method):
    eval_root, ckpt_root = METHODS[method]
    rows = []
    for log in sorted(glob.glob(f"{eval_root}/**/eval.log", recursive=True)):
        rel = log[len(eval_root) + 1:].replace("/eval.log", "")
        parts = rel.split("/")
        # expected: <stage>/epoch_<N>/<domain>
        stage = parts[0]
        epoch = ""
        domain = parts[-1]
        for p in parts:
            m = re.match(r"epoch_(\d+)", p)
            if m:
                epoch = m.group(1)
        if "source_only" in stage:
            # not part of the ablation comparison (no adaptation), skip
            continue
        m = parse_eval_log(log)
        if not m:
            continue
        row = {
            "method": method,
            "stage": stage,
            "epoch": epoch,
            "eval_domain": domain,
            "eval_dir": osp.dirname(log),
            "checkpoint": find_checkpoint(ckpt_root, stage, epoch),
        }
        for k in METRIC_FIELDS:
            row[k] = m.get(k, "")
        rows.append(row)
    return rows


def write_csv(path, rows):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def diagonal_rows(rows):
    """Keep only rows where eval_domain == the domain just adapted (diagonal)."""
    stage_to_dom = {
        "stage01_after_overcast": "overcast",
        "stage02_after_rainy": "rainy",
        "stage03_after_dawn_dusk": "dawn_dusk",
        "stage04_after_night": "night",
    }
    out = []
    for r in rows:
        if stage_to_dom.get(r["stage"]) == r["eval_domain"]:
            out.append(r)
    return out


def write_summary_md(path, all_rows):
    domains = ["overcast", "rainy", "dawn_dusk", "night"]
    methods = ["baseline_task2", "spa_only", "causal_only", "spa_causal"]
    lines = ["# Task2 ablation summary (epoch_10 diagonal)", ""]
    lines.append("Diagonal = evaluate each stage on the domain it just adapted "
                 "to, at epoch 10.")
    lines.append("")
    # index: (method, domain) -> row at epoch 10 diagonal
    idx = {}
    for r in diagonal_rows(all_rows):
        if r["epoch"] == "10":
            idx[(r["method"], r["eval_domain"])] = r
    for metric in ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "IDSw"]:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| domain | " + " | ".join(methods) + " |")
        lines.append("|" + "---|" * (len(methods) + 1))
        for dom in domains:
            cells = []
            for mth in methods:
                r = idx.get((mth, dom))
                cells.append(f"{r[metric]}" if r else "N/A")
            lines.append(f"| {dom} | " + " | ".join(cells) + " |")
        # averages
        avg_cells = []
        for mth in methods:
            vals = [idx[(mth, d)][metric] for d in domains
                    if (mth, d) in idx and isinstance(idx[(mth, d)][metric], float)]
            avg_cells.append(f"{sum(vals)/len(vals):.2f}" if vals else "N/A")
        lines.append("| **avg** | " + " | ".join(avg_cells) + " |")
        lines.append("")
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")


def main():
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    all_rows = []
    for method in METHODS:
        rows = collect_method(method)
        all_rows.extend(rows)
        if method != "baseline_task2":
            out = osp.join(SUMMARY_DIR, f"task2_{method}_results.csv")
            write_csv(out, rows)
            print(f"[WRITE] {out}  ({len(rows)} rows)")
        else:
            print(f"[READ ] baseline_task2  ({len(rows)} rows)")

    compare = osp.join(SUMMARY_DIR, "task2_ablation_compare.csv")
    write_csv(compare, all_rows)
    print(f"[WRITE] {compare}  ({len(all_rows)} rows)")

    md = osp.join(SUMMARY_DIR, "task2_ablation_summary.md")
    write_summary_md(md, all_rows)
    print(f"[WRITE] {md}")


if __name__ == "__main__":
    main()
