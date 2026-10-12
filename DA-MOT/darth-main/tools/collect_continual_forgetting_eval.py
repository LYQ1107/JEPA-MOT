#!/usr/bin/env python3
import csv
import math
import os
import os.path as osp
import re
from collections import defaultdict

ROOT = "work_dirs/bdd_4domain/continual_forgetting_eval"
LONG_CSV = osp.join(ROOT, "continual_all_domain_eval_long.csv")
MATRIX_DIR = osp.join(ROOT, "matrices")
METRICS_CSV = osp.join(ROOT, "continual_forgetting_metrics.csv")
MD_OUT = osp.join(ROOT, "continual_forgetting_compare.md")

DOMS = ["overcast", "rainy", "dawn_dusk", "night"]
STAGES = [
    "stage01_after_overcast",
    "stage02_after_rainy",
    "stage03_after_dawn_dusk",
    "stage04_after_night",
]
STAGE_DOM = dict(zip(STAGES, DOMS))
METHODS = [
    "spaP1_causalOld_4domain_5e",
    "spaP1_causalOld_roiV1Weak_4domain_5e",
]

CM_HDR = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CM_COLS = ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn", "MT", "PT", "ML", "FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS = ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA", "HOTA", "Dets", "IDs"]
VALUE_FIELDS = [
    "MOTA", "IDF1", "HOTA", "DetA", "AssA", "LocA", "FP", "FN", "IDSw",
    "Dets", "IDs", "Rcll", "Prcn", "DetPr", "DetRe", "FM", "MOTP", "MT", "PT", "ML",
]
MATRIX_METRICS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSw", "Dets", "Prcn", "Rcll"]
FORGET_METRICS = ["HOTA", "IDF1", "AssA", "IDSw", "MOTA", "DetA", "FN", "FP"]


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


def collect_rows():
    rows = []
    for method in METHODS:
        for sidx, stage in enumerate(STAGES, 1):
            for didx, dom in enumerate(DOMS, 1):
                outdir = osp.join(ROOT, method, stage, "epoch_5", dom)
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


def write_long(rows):
    os.makedirs(ROOT, exist_ok=True)
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


def as_float(v):
    if v in ("", None, "NOT_FOUND"):
        return None
    try:
        return float(v)
    except Exception:
        return None


def build_lookup(rows, metric):
    d = {}
    for r in rows:
        v = as_float(r.get(metric))
        if v is not None:
            d[(r["method"], r["eval_domain"], r["stage"])] = v
    return d


def write_matrices(rows):
    os.makedirs(MATRIX_DIR, exist_ok=True)
    paths = []
    for method in METHODS:
        for metric in MATRIX_METRICS:
            lookup = build_lookup(rows, metric)
            path = osp.join(MATRIX_DIR, f"{method}_{metric}_matrix.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["eval_domain"] + STAGES)
                for dom in DOMS:
                    vals = [lookup.get((method, dom, stage), "") for stage in STAGES]
                    w.writerow([dom] + vals)
            paths.append(path)
            print(f"[WRITE] {path}")
    return paths


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def metric_summary(rows, metric):
    lookup = build_lookup(rows, metric)
    out = []
    for method in METHODS:
        A = lambda dom, stage: lookup.get((method, dom, stage))
        diag = [A(DOMS[i], STAGES[i]) for i in range(4)]
        final = [A(dom, STAGES[-1]) for dom in DOMS]
        old_final = [A(dom, STAGES[-1]) for dom in DOMS[:3]]
        bwt_terms = []
        forget_diag = []
        forget_best = []
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
        future_parts = []
        future_map = {
            0: [A("rainy", STAGES[0]), A("dawn_dusk", STAGES[0]), A("night", STAGES[0])],
            1: [A("dawn_dusk", STAGES[1]), A("night", STAGES[1])],
            2: [A("night", STAGES[2])],
        }
        for vals in future_map.values():
            mv = mean(vals)
            if mv is not None:
                future_parts.append(mv)
        stage_seen = []
        for j in range(4):
            stage_seen.append(mean([A(DOMS[i], STAGES[j]) for i in range(j + 1)]))
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
            future_domain_avg=mean(future_parts),
            stage_seen_avg_1=stage_seen[0],
            stage_seen_avg_2=stage_seen[1],
            stage_seen_avg_3=stage_seen[2],
            stage_seen_avg_4=stage_seen[3],
        ))
    return out


def fmt(v):
    return "" if v is None else f"{v:.4f}"


def write_forgetting_metrics(rows):
    fields = [
        "method", "metric", "diag_avg", "final_avg_all_domains", "final_avg_seen_domains",
        "BWT_diag", "AvgForgetting_diag", "AvgForgetting_best", "old_domain_final_avg",
        "future_domain_avg", "stage_seen_avg_1", "stage_seen_avg_2", "stage_seen_avg_3", "stage_seen_avg_4",
    ]
    all_rows = []
    for metric in FORGET_METRICS:
        all_rows.extend(metric_summary(rows, metric))
    with open(METRICS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: fmt(r[k]) if k not in ("method", "metric") else r[k] for k in fields})
    print(f"[WRITE] {METRICS_CSV} ({len(all_rows)} rows)")
    return all_rows


def matrix_md(rows, method, metric):
    lookup = build_lookup(rows, metric)
    lines = [f"### {method} — {metric} matrix", "", "| eval domain \\ stage | " + " | ".join(STAGES) + " |", "|---|" + "---|" * len(STAGES)]
    for dom in DOMS:
        vals = []
        for stage in STAGES:
            v = lookup.get((method, dom, stage))
            vals.append("—" if v is None else f"{v:.2f}")
        lines.append(f"| {dom} | " + " | ".join(vals) + " |")
    return lines


def write_md(rows, metric_rows, matrix_paths):
    by = {(r["method"], r["metric"]): r for r in metric_rows}
    lines = [
        "# Continual Forgetting / All-Domain Evaluation",
        "",
        "Rows are evaluation domains; columns are adapted stages. Higher is better for MOTA/IDF1/HOTA/AssA/DetA; lower is better for FP/FN/IDSw.",
        "",
        "## HOTA matrices", "",
    ]
    for method in METHODS:
        lines += matrix_md(rows, method, "HOTA") + [""]
    lines += ["## IDF1 matrices", ""]
    for method in METHODS:
        lines += matrix_md(rows, method, "IDF1") + [""]
    lines += ["## MOTA matrices", ""]
    for method in METHODS:
        lines += matrix_md(rows, method, "MOTA") + [""]

    lines += [
        "## Forgetting metrics", "",
        "| metric | method | BWT_diag | AvgForgetting_diag | AvgForgetting_best | old_domain_final_avg | final_avg_all_domains | diag_avg |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in FORGET_METRICS:
        for method in METHODS:
            r = by.get((method, metric), {})
            lines.append(
                f"| {metric} | {method} | {fmt(r.get('BWT_diag'))} | {fmt(r.get('AvgForgetting_diag'))} | "
                f"{fmt(r.get('AvgForgetting_best'))} | {fmt(r.get('old_domain_final_avg'))} | "
                f"{fmt(r.get('final_avg_all_domains'))} | {fmt(r.get('diag_avg'))} |"
            )
    lines += [""]

    lines += ["## Initial interpretation", ""]
    for metric in ["HOTA", "IDF1", "AssA", "IDSw"]:
        a = by.get((METHODS[0], metric), {})
        b = by.get((METHODS[1], metric), {})
        if not a or not b:
            continue
        better_bwt = (b.get("BWT_diag") or -math.inf) > (a.get("BWT_diag") or -math.inf)
        lower_forget = (b.get("AvgForgetting_diag") or math.inf) < (a.get("AvgForgetting_diag") or math.inf)
        lines.append(
            f"- {metric}: RoI-V1Weak BWT {'higher' if better_bwt else 'not higher'}; "
            f"AvgForgetting_diag {'lower' if lower_forget else 'not lower'}."
        )
    lines += [
        "",
        "Final conclusion should be based on HOTA/IDF1/AssA/IDSw, because RoI prototype memory primarily targets association stability and forgetting mitigation.",
        "",
        "## Output files", "",
        f"- Long table: `{LONG_CSV}`",
        f"- Forgetting metrics: `{METRICS_CSV}`",
        f"- Matrices: `{MATRIX_DIR}/`",
        "",
    ]
    with open(MD_OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"[WRITE] {MD_OUT}")


def main():
    rows = collect_rows()
    write_long(rows)
    matrix_paths = write_matrices(rows)
    metric_rows = write_forgetting_metrics(rows)
    write_md(rows, metric_rows, matrix_paths)


if __name__ == "__main__":
    main()
