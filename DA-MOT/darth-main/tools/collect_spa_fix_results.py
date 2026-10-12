#!/usr/bin/env python3
"""Collect SPA-fix results -> summary CSVs + compare markdown.

Scans:
  - work_dirs/bdd_4domain/spa_fix/threshold_sweep/<sName>/<domain>/
  - work_dirs/bdd_4domain/spa_fix/<exp>/eval/stage*/epoch_5/<domain>/
Parses each eval.log OVERALL row (ClearMOT + HOTA), and computes deltas vs
baseline (task2_full_episode) and vs original spa_only_4domain_5e.
"""
import csv
import glob
import os.path as osp
import re

SPA_FIX = "work_dirs/bdd_4domain/spa_fix"
BASELINE_ROOT = "work_dirs/bdd_4domain/task2_full_episode/eval"
ORIG_SPA_ROOT = "work_dirs/bdd_4domain/spa_only_4domain_5e/eval"
OUT_DIR = f"{SPA_FIX}/summary"

CLEARMOT_HDR = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CLEARMOT_COLS = ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn",
                 "MT", "PT", "ML", "FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS = ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA",
             "HOTA", "Dets", "IDs"]

METRIC_KEYS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "LocA", "FP", "FN",
               "IDSw", "Dets", "IDs", "FM", "Prcn", "Rcll", "DetPr", "DetRe"]
DELTA_KEYS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSw"]


def _norm(s):
    return re.sub(r"\s+", " ", s.strip())


def parse_eval_log(path):
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


def _ref_metrics(root, dom):
    """Find the epoch_5 eval.log for a given domain under a stage tree."""
    pats = [
        f"{root}/stage*_after_{dom}/epoch_5/{dom}/eval.log",
        f"{root}/stage*_after_{dom}/epoch_5/{dom}/qdtrack*/*/*.log",
    ]
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            m = parse_eval_log(f)
            if m:
                return m
    return None


def _add_delta(row, ref, tag):
    for k in DELTA_KEYS:
        col = f"delta_vs_{tag}_{k}"
        if ref and row.get(k) not in ("", None) and ref.get(k) not in ("", None):
            try:
                row[col] = round(float(row[k]) - float(ref[k]), 3)
            except (ValueError, TypeError):
                row[col] = ""
        else:
            row[col] = ""


def _mkrow(method, scheme, order, dom, ep, m, work_dir, ckpt, elog):
    row = dict(method=method, scheme=scheme, target_order=order,
               eval_domain=dom, eval_epoch=ep, work_dir=work_dir,
               checkpoint=ckpt, eval_log=elog)
    for k in METRIC_KEYS:
        row[k] = m.get(k, "")
    base = _ref_metrics(BASELINE_ROOT, dom)
    orig = _ref_metrics(ORIG_SPA_ROOT, dom)
    _add_delta(row, base, "baseline")
    _add_delta(row, orig, "original_spa")
    return row


def collect_sweep():
    rows = []
    for sdir in sorted(glob.glob(f"{SPA_FIX}/threshold_sweep/*")):
        sname = osp.basename(sdir)
        for edir in sorted(glob.glob(f"{sdir}/*")):
            dom = osp.basename(edir)
            elog = f"{edir}/eval.log"
            m = parse_eval_log(elog)
            if not m:
                continue
            rows.append(_mkrow("threshold_sweep", sname, "spa_only_night",
                               dom, 5, m, edir, "", elog))
    return rows


def collect_trained():
    rows = []
    for exp_dir in sorted(glob.glob(f"{SPA_FIX}/*")):
        exp = osp.basename(exp_dir)
        if exp in ("threshold_sweep", "summary") or not osp.isdir(exp_dir):
            continue
        order = "4domain" if "4domain" in exp else "clear_to_night"
        for elog in sorted(glob.glob(f"{exp_dir}/eval/stage*/epoch_5/*/eval.log")):
            m = parse_eval_log(elog)
            if not m:
                continue
            dom = osp.basename(osp.dirname(elog))
            stage = elog.split("/eval/")[1].split("/")[0]
            ckpt = f"{exp_dir}/episode/adapt/{stage}/epoch_5.pth"
            rows.append(_mkrow(exp, exp, order, dom, 5, m, exp_dir, ckpt, elog))
    return rows


def _write(path, rows):
    if not rows:
        print(f"[SKIP] no rows for {path}")
        return
    base_fields = ["method", "scheme", "target_order", "eval_domain",
                   "eval_epoch"] + METRIC_KEYS + ["checkpoint", "eval_log",
                                                  "work_dir"]
    delta_fields = [f"delta_vs_baseline_{k}" for k in DELTA_KEYS] + \
                   [f"delta_vs_original_spa_{k}" for k in DELTA_KEYS]
    fields = base_fields + delta_fields
    import os
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[WRITE] {path}  ({len(rows)} rows)")


def main():
    sweep = collect_sweep()
    trained = collect_trained()
    c2n = [r for r in trained if r["target_order"] == "clear_to_night"]
    d4 = [r for r in trained if r["target_order"] == "4domain"]
    _write(f"{OUT_DIR}/spa_threshold_sweep_summary.csv", sweep)
    _write(f"{OUT_DIR}/spa_fix_clear_to_night_summary.csv", c2n)
    _write(f"{OUT_DIR}/spa_fix_4domain_summary.csv", d4)

    # compact compare markdown
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    md = ["# SPA-fix compare", ""]
    for title, rows in (("Threshold sweep (night)", sweep),
                        ("Trained clear->night", c2n),
                        ("Trained 4domain", d4)):
        md += [f"## {title}", ""]
        if not rows:
            md += ["_no results yet_", ""]
            continue
        md += ["| scheme | domain | MOTA | IDF1 | HOTA | DetA | AssA | FP | FN | "
               "IDSw | dBase_MOTA | dBase_FP | dBase_IDSw | dSPA_MOTA | dSPA_FP | "
               "dSPA_IDSw |", "|" + "---|" * 16]
        for r in rows:
            md.append(
                "| {scheme} | {eval_domain} | {MOTA} | {IDF1} | {HOTA} | {DetA} | "
                "{AssA} | {FP} | {FN} | {IDSw} | {delta_vs_baseline_MOTA} | "
                "{delta_vs_baseline_FP} | {delta_vs_baseline_IDSw} | "
                "{delta_vs_original_spa_MOTA} | {delta_vs_original_spa_FP} | "
                "{delta_vs_original_spa_IDSw} |".format(
                    **{k: r.get(k, "") for k in r}))
        md.append("")
    with open(f"{OUT_DIR}/spa_fix_compare.md", "w") as fp:
        fp.write("\n".join(md) + "\n")
    print(f"[WRITE] {OUT_DIR}/spa_fix_compare.md")


if __name__ == "__main__":
    main()
