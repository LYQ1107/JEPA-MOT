#!/usr/bin/env python3
"""Collect causal PCA ablation results -> summary.csv + compare.md.

Reads:
  work_dirs/bdd_4domain/causal_pca_ablation/v*/eval/stage*/epoch_5/*/eval.log

Reference baselines (hardcoded from prior experiments):
  baseline_no_method:  AVG MOTA 23.78, IDF1 41.34, HOTA 36.40, DetA 30.11,
                       AssA 44.80, IDSw 2498, FP 8313, FN 35490
  current_causal_only: AVG MOTA 24.24, IDF1 41.67, HOTA 36.45, DetA 30.49,
                       AssA 44.29  (per-domain from task2_causal_only)
"""
import csv
import glob
import os
import os.path as osp
import re

OUT_DIR = "work_dirs/bdd_4domain/causal_pca_ablation"
CSV_OUT = f"{OUT_DIR}/summary.csv"
MD_OUT  = f"{OUT_DIR}/summary.md"

CLEARMOT_HDR = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CLEARMOT_COLS = ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn",
                 "MT", "PT", "ML", "FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS = ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA",
             "HOTA", "Dets", "IDs"]

METRIC_KEYS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "LocA",
               "FP", "FN", "IDSw", "Dets", "IDs", "FM", "Prcn", "Rcll",
               "DetPr", "DetRe"]
AVG_KEYS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSw"]
DELTA_KEYS = AVG_KEYS

# Reference baselines
REF_BASELINE = dict(MOTA=23.78, IDF1=41.34, HOTA=36.40, DetA=30.11,
                    AssA=44.80, IDSw=2498.0, FP=8313.0, FN=35490.0)
REF_CAUSAL   = dict(MOTA=24.24, IDF1=41.67, HOTA=36.45, DetA=30.49,
                    AssA=44.29, IDSw=None,   FP=None,    FN=None)

DOM_ORDER = ["overcast", "rainy", "dawn_dusk", "night"]


def _norm(s):
    return re.sub(r"\s+", " ", s.strip())


def parse_log(path):
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


def collect():
    rows = []
    for vdir in sorted(glob.glob(f"{OUT_DIR}/v*")):
        method = osp.basename(vdir)
        for elog in sorted(glob.glob(
                f"{vdir}/eval/stage*/epoch_5/*/eval.log")):
            m = parse_log(elog)
            if not m:
                continue
            dom   = osp.basename(osp.dirname(elog))
            stage = elog.split("/eval/")[1].split("/")[0]
            ckpt  = f"{vdir}/episode/adapt/{stage}/epoch_5.pth"
            row = dict(method=method, stage=stage, target_domain=dom,
                       eval_epoch=5, work_dir=vdir, checkpoint=ckpt,
                       eval_log=elog)
            for k in METRIC_KEYS:
                row[k] = m.get(k, "")
            rows.append(row)
    return rows


def avg_per_method(rows):
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)
    avgs = {}
    for method, mrs in by_method.items():
        dom_rows = {r["target_domain"]: r for r in mrs}
        vals = {k: [] for k in AVG_KEYS}
        for dom in DOM_ORDER:
            dr = dom_rows.get(dom)
            if dr:
                for k in AVG_KEYS:
                    v = dr.get(k)
                    if v not in ("", None):
                        try:
                            vals[k].append(float(v))
                        except (ValueError, TypeError):
                            pass
        avgs[method] = {k: (round(sum(v)/len(v), 2) if v else "")
                        for k, v in vals.items()}
    return avgs


def _delta(val, ref):
    if val in ("", None) or ref in (None, ""):
        return ""
    try:
        return round(float(val) - float(ref), 3)
    except (ValueError, TypeError):
        return ""


def write_csv(rows, avgs):
    delta_b_fields  = [f"delta_vs_baseline_{k}"    for k in DELTA_KEYS]
    delta_c_fields  = [f"delta_vs_causal_only_{k}" for k in DELTA_KEYS]
    avg_fields      = [f"avg_{k}" for k in AVG_KEYS]
    fields = (["method", "stage", "target_domain", "eval_epoch"]
              + METRIC_KEYS + avg_fields + delta_b_fields + delta_c_fields
              + ["checkpoint", "eval_log", "work_dir"])

    all_rows = []
    for r in rows:
        method = r["method"]
        av     = avgs.get(method, {})
        out    = dict(r)
        for k in AVG_KEYS:
            out[f"avg_{k}"] = av.get(k, "")
        for k in DELTA_KEYS:
            out[f"delta_vs_baseline_{k}"]    = _delta(av.get(k), REF_BASELINE.get(k))
            out[f"delta_vs_causal_only_{k}"] = _delta(av.get(k), REF_CAUSAL.get(k))
        all_rows.append(out)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[WRITE] {CSV_OUT}  ({len(all_rows)} rows)")


def write_md(rows, avgs):
    def _f(v, w=7):
        if v in ("", None): return "—".rjust(w)
        try: return f"{float(v):.2f}".rjust(w)
        except: return str(v).rjust(w)

    def _fd(v, w=7):
        if v in ("", None): return "—".rjust(w)
        try:
            f = float(v)
            return (f"+{f:.2f}" if f > 0 else f"{f:.2f}").rjust(w)
        except: return str(v).rjust(w)

    lines = ["# Causal PCA Ablation — 4-domain 5e", ""]

    # per-domain table
    lines += ["## Per-domain results", "",
              "| method | domain | MOTA | IDF1 | HOTA | DetA | AssA | "
              "FP | FN | IDSw |",
              "|--------|--------|------|------|------|------|------|"
              "----|----|------|"]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['target_domain']} "
            f"| {_f(r.get('MOTA'))} | {_f(r.get('IDF1'))} "
            f"| {_f(r.get('HOTA'))} | {_f(r.get('DetA'))} "
            f"| {_f(r.get('AssA'))} "
            f"| {_f(r.get('FP'),8)} | {_f(r.get('FN'),8)} "
            f"| {_f(r.get('IDSw'),6)} |")

    # 4-domain average compare table
    lines += ["", "## 4-domain averages vs baselines", "",
              "| method | avgMOTA | avgIDF1 | avgHOTA | avgDetA | avgAssA | "
              "avgFP | avgFN | avgIDSw | "
              "dBase_MOTA | dBase_AssA | dBase_IDSw | "
              "dCausal_MOTA | dCausal_AssA | dCausal_IDSw |",
              "|--------|---------|---------|---------|---------|---------|"
              "-------|-------|---------|"
              "-----------|------------|------------|"
              "-------------|--------------|--------------|"]

    # hardcoded reference rows
    lines.append(
        f"| **baseline_no_method** "
        f"| {_f(23.78)} | {_f(41.34)} | {_f(36.40)} | {_f(30.11)} | {_f(44.80)} "
        f"| {_f(8313,7)} | {_f(35490,7)} | {_f(2498,7)} "
        f"| — | — | — | — | — | — |")
    lines.append(
        f"| **current_causal_only** "
        f"| {_f(24.24)} | {_f(41.67)} | {_f(36.45)} | {_f(30.49)} | {_f(44.29)} "
        f"| — | — | — "
        f"| {_fd(24.24-23.78)} | {_fd(44.29-44.80)} | — "
        f"| — | — | — |")

    for method, av in sorted(avgs.items()):
        db_mota  = _delta(av.get("MOTA"),  REF_BASELINE["MOTA"])
        db_assA  = _delta(av.get("AssA"),  REF_BASELINE["AssA"])
        db_idsw  = _delta(av.get("IDSw"),  REF_BASELINE["IDSw"])
        dc_mota  = _delta(av.get("MOTA"),  REF_CAUSAL["MOTA"])
        dc_assA  = _delta(av.get("AssA"),  REF_CAUSAL["AssA"])
        dc_idsw  = _delta(av.get("IDSw"),  None)
        lines.append(
            f"| **{method}** "
            f"| {_f(av.get('MOTA'))} | {_f(av.get('IDF1'))} "
            f"| {_f(av.get('HOTA'))} | {_f(av.get('DetA'))} "
            f"| {_f(av.get('AssA'))} "
            f"| {_f(av.get('FP'),7)} | {_f(av.get('FN'),7)} "
            f"| {_f(av.get('IDSw'),7)} "
            f"| {_fd(db_mota)} | {_fd(db_assA)} | {_fd(db_idsw)} "
            f"| {_fd(dc_mota)} | {_fd(dc_assA)} | {_fd(dc_idsw)} |")

    lines += ["", "## Judgment criteria",
              "",
              "Priority (causal PCA goal = stable association, reduce weather nuisance):",
              "1. AssA ↑ vs current_causal_only",
              "2. IDSw ↓ vs baseline and current_causal_only",
              "3. IDF1 ↑, HOTA ↑",
              "4. night MOTA ↑",
              "5. DetA / FN not significantly worse",
              ""]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MD_OUT, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"[WRITE] {MD_OUT}")


def main():
    rows = collect()
    avgs = avg_per_method(rows)
    write_csv(rows, avgs)
    write_md(rows, avgs)

    # terminal summary
    print()
    print(f"{'method':<42} {'dom':<12} {'MOTA':>6} {'IDF1':>6} "
          f"{'HOTA':>6} {'AssA':>6} {'IDSw':>6} {'FP':>7} {'FN':>7}")
    print("-" * 110)
    for r in rows:
        print(f"{r['method']:<42} {r['target_domain']:<12} "
              f"{str(r.get('MOTA','')):>6} {str(r.get('IDF1','')):>6} "
              f"{str(r.get('HOTA','')):>6} {str(r.get('AssA','')):>6} "
              f"{str(r.get('IDSw','')):>6} "
              f"{str(r.get('FP','')):>7} {str(r.get('FN','')):>7}")
    print()
    print("4-domain averages:")
    for method, av in sorted(avgs.items()):
        print(f"  {method}: MOTA={av.get('MOTA','?')} IDF1={av.get('IDF1','?')} "
              f"HOTA={av.get('HOTA','?')} AssA={av.get('AssA','?')} "
              f"IDSw={av.get('IDSw','?')} FP={av.get('FP','?')} FN={av.get('FN','?')}")


if __name__ == "__main__":
    main()
