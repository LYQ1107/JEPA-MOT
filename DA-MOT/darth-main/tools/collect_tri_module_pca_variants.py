#!/usr/bin/env python3
import csv
import os
import os.path as osp
import re
from collections import defaultdict

OUT_ROOT = "work_dirs/bdd_4domain/tri_module_pca_variants"
BASELINE_LONG = "work_dirs/bdd_4domain/continual_forgetting_eval/continual_all_domain_eval_long.csv"
LONG_CSV = osp.join(OUT_ROOT, "tri_module_pca_variants_long.csv")
SUMMARY_CSV = osp.join(OUT_ROOT, "tri_module_pca_variants_summary.csv")
COMPARE_MD = osp.join(OUT_ROOT, "tri_module_pca_variants_compare.md")
MATRIX_DIR = osp.join(OUT_ROOT, "matrices")

DOMS = ["overcast", "rainy", "dawn_dusk", "night"]
STAGES = [
    "stage01_after_overcast",
    "stage02_after_rainy",
    "stage03_after_dawn_dusk",
    "stage04_after_night",
]
STAGE_DOM = dict(zip(STAGES, DOMS))
NEW_METHODS = [
    "spaP1_causalProtoCenter_roiV1Weak_4domain_5e",
    "spaP1_causalShrinkage_roiV1Weak_4domain_5e",
    "spaP1_causalProtoShrink_roiV1Weak_4domain_5e",
]
BASELINE_METHODS = [
    "spaP1_causalOld_4domain_5e",
    "spaP1_causalOld_roiV1Weak_4domain_5e",
]
METHODS = BASELINE_METHODS + NEW_METHODS

CM_HDR = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CM_COLS = ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn", "MT", "PT", "ML", "FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS = ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA", "HOTA", "Dets", "IDs"]
VALUE_FIELDS = [
    "MOTA", "IDF1", "HOTA", "DetA", "AssA", "LocA", "FP", "FN", "IDSw",
    "Dets", "IDs", "Rcll", "Prcn", "DetPr", "DetRe", "FM", "MOTP", "MT", "PT", "ML",
]
FORGET_METRICS = ["HOTA", "IDF1", "AssA", "IDSw", "MOTA", "DetA", "FN", "FP"]
MATRIX_METRICS = ["HOTA", "IDF1", "MOTA", "AssA", "DetA", "IDSw"]


def read_text(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def parse_log(path):
    if not osp.isfile(path):
        return {}
    lines = open(path, errors="ignore").read().splitlines()
    res = {}
    for hdr, cols in ((CM_HDR, CM_COLS), (HOTA_HDR, HOTA_COLS)):
        for i, line in enumerate(lines):
            if re.sub(r"\s+", " ", line.strip()) == hdr:
                for j in range(i + 1, min(i + 30, len(lines))):
                    if lines[j].strip().upper().startswith("OVERALL"):
                        for c, v in zip(cols, lines[j].split()[1:]):
                            try:
                                res[c] = float(v)
                            except Exception:
                                pass
                        break
                break
    return res


def collect_new_rows():
    rows = []
    for method in NEW_METHODS:
        for sidx, stage in enumerate(STAGES, 1):
            for didx, dom in enumerate(DOMS, 1):
                outdir = osp.join(OUT_ROOT, method, "forgetting_eval", stage, "epoch_5", dom)
                logpath = osp.join(outdir, "eval.log")
                m = parse_log(logpath)
                row = dict(
                    method=method,
                    stage=stage,
                    stage_idx=sidx,
                    adapted_domain=read_text(osp.join(outdir, "adapted_domain.txt")) or STAGE_DOM[stage],
                    eval_domain=dom,
                    eval_domain_idx=didx,
                    eval_epoch=5,
                    thr_override=read_text(osp.join(outdir, "cfg_options.txt")),
                    checkpoint=read_text(osp.join(outdir, "checkpoint_path.txt")),
                    eval_log=logpath if osp.isfile(logpath) else "NOT_FOUND",
                )
                for k in VALUE_FIELDS:
                    row[k] = m.get(k, "")
                rows.append(row)
    return rows


def collect_baseline_rows():
    if not osp.isfile(BASELINE_LONG):
        return []
    rows = []
    with open(BASELINE_LONG, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("method") in BASELINE_METHODS:
                rows.append(r)
    return rows


def as_float(v):
    if v in ("", None, "NOT_FOUND"):
        return None
    try:
        return float(v)
    except Exception:
        return None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v):
    return "" if v is None else f"{v:.4f}"


def build_lookup(rows, metric):
    d = {}
    for r in rows:
        v = as_float(r.get(metric))
        if v is not None:
            d[(r["method"], r["eval_domain"], r["stage"])] = v
    return d


def metric_summary(rows, metric):
    lookup = build_lookup(rows, metric)
    out = []
    for method in METHODS:
        A = lambda dom, stage: lookup.get((method, dom, stage))
        diag = [A(DOMS[i], STAGES[i]) for i in range(4)]
        final = [A(dom, STAGES[-1]) for dom in DOMS]
        old_final = [A(dom, STAGES[-1]) for dom in DOMS[:3]]
        bwt_terms, forget_diag, forget_best = [], [], []
        for i in range(3):
            cur = A(DOMS[i], STAGES[i])
            fin = A(DOMS[i], STAGES[-1])
            hist = [A(DOMS[i], STAGES[j]) for j in range(i, 4)]
            hist = [v for v in hist if v is not None]
            if cur is not None and fin is not None:
                bwt_terms.append(fin - cur)
                forget_diag.append(max(0.0, cur - fin))
            if hist and fin is not None:
                forget_best.append(max(0.0, max(hist) - fin))
        stage_seen = [mean([A(DOMS[i], STAGES[j]) for i in range(j + 1)]) for j in range(4)]
        out.append(dict(
            method=method,
            metric=metric,
            diag_avg=mean(diag),
            final_avg_all_domains=mean(final),
            final_avg_seen_domains=mean(final),
            BWT_diag=mean(bwt_terms),
            AvgForgetting_diag=mean(forget_diag),
            AvgForgetting_best=mean(forget_best),
            old_domain_final_avg=mean(old_final),
            stage_seen_avg_1=stage_seen[0],
            stage_seen_avg_2=stage_seen[1],
            stage_seen_avg_3=stage_seen[2],
            stage_seen_avg_4=stage_seen[3],
        ))
    return out


def write_long(rows):
    os.makedirs(OUT_ROOT, exist_ok=True)
    fields = [
        "method", "stage", "stage_idx", "adapted_domain", "eval_domain",
        "eval_domain_idx", "eval_epoch", "thr_override",
    ] + VALUE_FIELDS + ["checkpoint", "eval_log"]
    with open(LONG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[WRITE] {LONG_CSV} ({len(rows)} rows)")


def write_summary(rows):
    fields = [
        "method", "metric", "diag_avg", "final_avg_all_domains", "final_avg_seen_domains",
        "BWT_diag", "AvgForgetting_diag", "AvgForgetting_best", "old_domain_final_avg",
        "stage_seen_avg_1", "stage_seen_avg_2", "stage_seen_avg_3", "stage_seen_avg_4",
    ]
    metric_rows = []
    for metric in FORGET_METRICS:
        metric_rows.extend(metric_summary(rows, metric))
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in metric_rows:
            w.writerow({k: fmt(r[k]) if k not in ("method", "metric") else r[k] for k in fields})
    print(f"[WRITE] {SUMMARY_CSV} ({len(metric_rows)} rows)")
    return metric_rows


def write_matrices(rows):
    os.makedirs(MATRIX_DIR, exist_ok=True)
    for method in METHODS:
        for metric in MATRIX_METRICS:
            lookup = build_lookup(rows, metric)
            path = osp.join(MATRIX_DIR, f"{method}_{metric}_matrix.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["eval_domain"] + STAGES)
                for dom in DOMS:
                    w.writerow([dom] + [lookup.get((method, dom, stage), "") for stage in STAGES])
            print(f"[WRITE] {path}")


def matrix_md(rows, method, metric):
    lookup = build_lookup(rows, metric)
    lines = [f"### {method} — {metric}", "", "| eval domain \\ stage | " + " | ".join(STAGES) + " |", "|---|" + "---|" * len(STAGES)]
    for dom in DOMS:
        vals = []
        for stage in STAGES:
            v = lookup.get((method, dom, stage))
            vals.append("—" if v is None else f"{v:.2f}")
        lines.append(f"| {dom} | " + " | ".join(vals) + " |")
    return lines


def write_md(rows, metric_rows):
    by = {(r["method"], r["metric"]): r for r in metric_rows}
    lines = [
        "# Tri-module PCA Variants vs causalOld",
        "",
        "Baselines are read from `work_dirs/bdd_4domain/continual_forgetting_eval/continual_all_domain_eval_long.csv`.",
        "New variants are trained/evaluated under `work_dirs/bdd_4domain/tri_module_pca_variants/`.",
        "",
        "## Key summary",
        "",
        "| method | HOTA final | HOTA BWT | HOTA forget-best | IDF1 final | AssA final | MOTA final | IDSw final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        h = by.get((method, "HOTA"), {})
        i = by.get((method, "IDF1"), {})
        a = by.get((method, "AssA"), {})
        m = by.get((method, "MOTA"), {})
        s = by.get((method, "IDSw"), {})
        lines.append(
            f"| {method} | {fmt(h.get('final_avg_all_domains'))} | {fmt(h.get('BWT_diag'))} | "
            f"{fmt(h.get('AvgForgetting_best'))} | {fmt(i.get('final_avg_all_domains'))} | "
            f"{fmt(a.get('final_avg_all_domains'))} | {fmt(m.get('final_avg_all_domains'))} | "
            f"{fmt(s.get('final_avg_all_domains'))} |"
        )
    lines += ["", "## HOTA matrices", ""]
    for method in METHODS:
        lines += matrix_md(rows, method, "HOTA") + [""]
    with open(COMPARE_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"[WRITE] {COMPARE_MD}")


def main():
    rows = collect_baseline_rows() + collect_new_rows()
    write_long(rows)
    write_matrices(rows)
    metric_rows = write_summary(rows)
    write_md(rows, metric_rows)


if __name__ == "__main__":
    main()
