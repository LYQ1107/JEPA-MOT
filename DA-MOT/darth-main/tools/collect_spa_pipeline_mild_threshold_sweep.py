#!/usr/bin/env python3
"""Collect spa_pipeline_mild threshold sweep results -> CSV + MD.

Reads:
  work_dirs/bdd_4domain/spa_fix/pipeline_mild_threshold_sweep/night/P*/eval.log

Reference baselines (hardcoded):
  baseline default night      : MOTA 14.0 IDF1 33.6 HOTA 29.3 DetA 25.1 AssA 34.9
                                FP 13904 FN 51260 IDSw 3976
  baseline calibrated (B1)    : MOTA 20.3 IDF1 35.7 HOTA 30.5 DetA 24.9 AssA 37.9
                                FP 8433  FN 53067 IDSw 2600
  pipeline_mild default night : MOTA 17.5 IDF1 38.2 HOTA 33.0 DetA 29.1 AssA 38.1
                                FP 15263 FN 46228 IDSw 4821
"""
import csv
import glob
import os
import os.path as osp
import re

OUT_DIR = "work_dirs/bdd_4domain/spa_fix/pipeline_mild_threshold_sweep"
IN_DIR  = f"{OUT_DIR}/night"
CSV_OUT = f"{OUT_DIR}/pipeline_mild_threshold_sweep_night_summary.csv"
MD_OUT  = f"{OUT_DIR}/pipeline_mild_threshold_sweep_night_summary.md"

CLEARMOT_HDR  = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CLEARMOT_COLS = ["IDF1","MOTA","MOTP","FP","FN","IDSw","Rcll","Prcn","MT","PT","ML","FM"]
HOTA_HDR      = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS     = ["AssRe","AssPr","AssA","LocA","DetRe","DetPr","DetA","HOTA","Dets","IDs"]

METRIC_KEYS = ["MOTA","IDF1","HOTA","DetA","AssA","LocA",
               "FP","FN","IDSw","Dets","IDs","FM","Rcll","Prcn","DetPr","DetRe","MOTP",
               "MT","PT","ML"]
DELTA_KEYS  = ["MOTA","IDF1","HOTA","DetA","AssA","FP","FN","IDSw","Dets","IDs","Rcll","Prcn"]

SWEEP_ORDER = ["P0_default","P1_init030","P2_obj030","P3_score015_obj030",
               "P4_init030_obj030","P5_init025","P6_obj025","P7_score010_obj030"]

THR_LABELS = {
    "P0_default":         "default (init=0.20 obj=0.20)",
    "P1_init030":         "init_score_thr=0.30",
    "P2_obj030":          "obj_score_thr=0.30",
    "P3_score015_obj030": "score_thr=0.15 + obj=0.30",
    "P4_init030_obj030":  "init=0.30 + obj=0.30",
    "P5_init025":         "init_score_thr=0.25",
    "P6_obj025":          "obj_score_thr=0.25",
    "P7_score010_obj030": "score_thr=0.10 + obj=0.30",
}

REF = {
    "baseline_default":    dict(MOTA=14.0,IDF1=33.6,HOTA=29.3,DetA=25.1,AssA=34.9,FP=13904,FN=51260,IDSw=3976),
    "baseline_B1_init030": dict(MOTA=20.3,IDF1=35.7,HOTA=30.5,DetA=24.9,AssA=37.9,FP=8433, FN=53067,IDSw=2600),
    "pipeline_default":    dict(MOTA=17.5,IDF1=38.2,HOTA=33.0,DetA=29.1,AssA=38.1,FP=15263,FN=46228,IDSw=4821),
}

CKPT = ("work_dirs/bdd_4domain/spa_fix/spa_pipeline_mild_4domain_5e/"
        "episode/adapt/stage04_after_night/epoch_5.pth")


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
                            try:   res[c] = float(v)
                            except: res[c] = v
                        break
                break
    return res if res else None


def _delta(a, b):
    if a in ("", None) or b in ("", None):
        return ""
    try:   return round(float(a) - float(b), 3)
    except: return ""


def collect():
    rows = []
    for pname in SWEEP_ORDER:
        elog = osp.join(IN_DIR, pname, "eval.log")
        opts_file = osp.join(IN_DIR, pname, "cfg_options.txt")
        m = parse_log(elog)
        opts = open(opts_file).read().strip() if osp.isfile(opts_file) else ""
        row = dict(
            group="pipeline_mild_thr_sweep",
            method=f"spa_pipeline_mild_{pname}",
            eval_domain="night",
            eval_epoch=5,
            stage="stage04_after_night",
            thr_variant=pname,
            thr_override=THR_LABELS.get(pname, ""),
            cfg_options=opts,
            checkpoint=CKPT,
            eval_log=elog if osp.isfile(elog) else "NOT_FOUND",
            work_dir=osp.join(IN_DIR, pname),
        )
        for k in METRIC_KEYS:
            row[k] = m.get(k, "") if m else "NOT_FOUND"
        rows.append(row)
    return rows


def write_csv(rows):
    p0 = {k: rows[0].get(k) for k in DELTA_KEYS} if rows else {}
    delta_fields = [f"delta_vs_P0_{k}" for k in DELTA_KEYS]
    fields = (["group","method","eval_domain","eval_epoch","stage",
               "thr_variant","thr_override","cfg_options"]
              + METRIC_KEYS + delta_fields
              + ["checkpoint","eval_log","work_dir"])

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CSV_OUT, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            for k in DELTA_KEYS:
                out[f"delta_vs_P0_{k}"] = _delta(r.get(k), p0.get(k))
            w.writerow({k: out.get(k,"") for k in fields})
    print(f"[WRITE] {CSV_OUT}  ({len(rows)} rows)")


def write_md(rows):
    def _f(v, w=7):
        if v in ("","NOT_FOUND",None): return "—".rjust(w)
        try: return f"{float(v):.1f}".rjust(w)
        except: return str(v).rjust(w)

    def _fd(v, w=7):
        if v in ("","NOT_FOUND",None): return "—".rjust(w)
        try:
            f = float(v)
            return (f"+{f:.1f}" if f > 0 else f"{f:.1f}").rjust(w)
        except: return str(v).rjust(w)

    p0 = {k: rows[0].get(k) for k in DELTA_KEYS} if rows else {}

    lines = [
        "# spa_pipeline_mild Threshold Sweep — Night (epoch 5, stage04_after_night)",
        "",
        f"Checkpoint: `{CKPT}`",
        "Eval config: `configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_night.py`",
        "Night eval config defaults: `init_score_thr=0.20  obj_score_thr=0.20  score_thr=0.05`",
        "",
        "## Reference baselines",
        "",
        "| ref | MOTA | IDF1 | HOTA | DetA | AssA | FP | FN | IDSw |",
        "|-----|------|------|------|------|------|-----|-----|------|",
        "| baseline default | 14.0 | 33.6 | 29.3 | 25.1 | 34.9 | 13,904 | 51,260 | 3,976 |",
        "| baseline B1 init=0.30 | 20.3 | 35.7 | 30.5 | 24.9 | 37.9 | 8,433 | 53,067 | 2,600 |",
        "| pipeline_mild default | 17.5 | 38.2 | 33.0 | 29.1 | 38.1 | 15,263 | 46,228 | 4,821 |",
        "",
        "## Sweep Results",
        "",
        "| variant | threshold | MOTA | IDF1 | HOTA | DetA | AssA | FP | FN | IDSw | "
        "dMOTA | dIDF1 | dHOTA | dAssA | dFP | dFN | dIDSw |",
        "|---------|-----------|------|------|------|------|------|-----|-----|------|"
        "-------|-------|-------|-------|-----|-----|-------|",
    ]

    for r in rows:
        d = {k: _delta(r.get(k), p0.get(k)) for k in DELTA_KEYS}
        lines.append(
            f"| {r['thr_variant']} | {r['thr_override']} "
            f"| {_f(r.get('MOTA'))} | {_f(r.get('IDF1'))} | {_f(r.get('HOTA'))} "
            f"| {_f(r.get('DetA'))} | {_f(r.get('AssA'))} "
            f"| {_f(r.get('FP'),8)} | {_f(r.get('FN'),8)} | {_f(r.get('IDSw'),6)} "
            f"| {_fd(d.get('MOTA'))} | {_fd(d.get('IDF1'))} | {_fd(d.get('HOTA'))} "
            f"| {_fd(d.get('AssA'))} | {_fd(d.get('FP'),8)} | {_fd(d.get('FN'),8)} "
            f"| {_fd(d.get('IDSw'))} |")

    lines += [
        "",
        "## vs baseline calibrated (B1 init=0.30)",
        "",
        "| variant | MOTA | IDF1 | HOTA | DetA | AssA | FP | FN | IDSw | "
        "dMOTA | dIDF1 | dHOTA | dFP | dFN | dIDSw |",
        "|---------|------|------|------|------|------|-----|-----|------|"
        "-------|-------|-------|-----|-----|-------|",
    ]
    b1 = REF["baseline_B1_init030"]
    for r in rows:
        lines.append(
            f"| {r['thr_variant']} | {r['thr_override']} "
            f"| {_f(r.get('MOTA'))} | {_f(r.get('IDF1'))} | {_f(r.get('HOTA'))} "
            f"| {_f(r.get('DetA'))} | {_f(r.get('AssA'))} "
            f"| {_f(r.get('FP'),8)} | {_f(r.get('FN'),8)} | {_f(r.get('IDSw'),6)} "
            f"| {_fd(_delta(r.get('MOTA'), b1['MOTA']))} "
            f"| {_fd(_delta(r.get('IDF1'), b1['IDF1']))} "
            f"| {_fd(_delta(r.get('HOTA'), b1['HOTA']))} "
            f"| {_fd(_delta(r.get('FP'),   b1['FP']),8)} "
            f"| {_fd(_delta(r.get('FN'),   b1['FN']),8)} "
            f"| {_fd(_delta(r.get('IDSw'), b1['IDSw']))} |")

    lines += [
        "",
        "## Judgment criteria",
        "",
        "Priority (pipeline_mild goal: keep IDF1/HOTA/DetA/FN gains, reduce FP/IDSw):",
        "1. FP ↓ and IDSw ↓ vs pipeline_mild default",
        "2. MOTA ↑ or at least not lower than pipeline_mild default",
        "3. IDF1 / HOTA / DetA / AssA not significantly lower than pipeline_mild default",
        "4. FN should not spike back to baseline calibrated levels (≤53,067)",
        "5. vs baseline B1: IDF1 ↑, HOTA ↑, DetA ↑, FN ↓",
        "",
    ]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MD_OUT, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"[WRITE] {MD_OUT}")


def main():
    rows = collect()
    write_csv(rows)
    write_md(rows)

    # terminal table
    print()
    p0 = {k: rows[0].get(k) for k in DELTA_KEYS} if rows else {}
    print(f"{'variant':<22} {'threshold':<32} {'MOTA':>6} {'IDF1':>6} {'HOTA':>6} "
          f"{'FP':>7} {'FN':>7} {'IDSw':>5}  {'dMOTA':>7}")
    print("-" * 110)
    for r in rows:
        dm = _delta(r.get("MOTA"), p0.get("MOTA"))
        sign = ("+" if isinstance(dm, float) and dm > 0 else "") if dm != "" else ""
        dm_s = f"{sign}{dm:.1f}" if dm != "" else "—"
        print(f"{r['thr_variant']:<22} {r['thr_override']:<32} "
              f"{str(r.get('MOTA','')):>6} {str(r.get('IDF1','')):>6} "
              f"{str(r.get('HOTA','')):>6} "
              f"{str(r.get('FP','')):>7} {str(r.get('FN','')):>7} "
              f"{str(r.get('IDSw','')):>5}  {dm_s:>7}")


if __name__ == "__main__":
    main()
