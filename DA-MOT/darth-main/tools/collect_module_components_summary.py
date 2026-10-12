#!/usr/bin/env python3
"""Collect module combination results -> module_components_summary.csv + avg + compare.md

Usage:
    python tools/collect_module_components_summary.py
"""
import csv
import os
import os.path as osp
import re
import shutil
from collections import defaultdict
from datetime import datetime

SUMMARY_CSV = "work_dirs/bdd_4domain/module_components_summary.csv"
AVG_CSV     = "work_dirs/bdd_4domain/module_components_summary_avg.csv"
MD_OUT      = "work_dirs/bdd_4domain/module_components_summary_compare.md"
COMBO_ROOT  = "work_dirs/bdd_4domain/module_combinations"

FIELDS = [
    "group","method","train_setting","eval_domain","eval_epoch",
    "stage","thr_override",
    "MOTA","IDF1","HOTA","DetA","AssA","LocA",
    "FP","FN","IDSw","Dets","IDs",
    "Rcll","Prcn","DetPr","DetRe",
    "FM","MOTP","MT","PT","ML",
    "eval_log",
]
AVG_METRICS = ["MOTA","IDF1","HOTA","DetA","AssA","LocA",
               "FP","FN","IDSw","Dets","IDs","Rcll","Prcn","DetPr","DetRe"]
AVG_FIELDS  = ["group","method","train_setting","thr_override"] + \
              [f"AVG_{k}" for k in AVG_METRICS]

DOMS = ["overcast","rainy","dawn_dusk","night"]

CM_HDR   = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CM_COLS  = ["IDF1","MOTA","MOTP","FP","FN","IDSw","Rcll","Prcn","MT","PT","ML","FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS= ["AssRe","AssPr","AssA","LocA","DetRe","DetPr","DetA","HOTA","Dets","IDs"]

COMBOS = [
    ("combo","spaP1_roiV1_4domain_5e",         "4stage_sequential_5e","model.tracker.init_score_thr=0.30"),
    ("combo","spaP1_causalOld_4domain_5e",      "4stage_sequential_5e","model.tracker.init_score_thr=0.30"),
    ("combo","spaP1_causalOld_roiV1_4domain_5e","4stage_sequential_5e","model.tracker.init_score_thr=0.30"),
    ("tri_module_soft","spaP1_causalOld_roiV1_scorewired_4domain_5e","4stage_sequential_5e","model.tracker.init_score_thr=0.30"),
    ("tri_module_soft","spaP1_causalOld_roiV1Weak_4domain_5e","4stage_sequential_5e","model.tracker.init_score_thr=0.30"),
    ("tri_module_soft","spaP1_causalOldWeak_roiV1_4domain_5e","4stage_sequential_5e","model.tracker.init_score_thr=0.30"),
    ("tri_module_gate","spaP1_causalOld_roiV1Gate_4domain_5e","4stage_sequential_5e","model.tracker.init_score_thr=0.30"),
]


def parse_log(path):
    if not osp.isfile(path):
        return {}
    lines = open(path).read().splitlines()
    res = {}
    for hdr, cols in ((CM_HDR, CM_COLS), (HOTA_HDR, HOTA_COLS)):
        for i, line in enumerate(lines):
            if re.sub(r"\s+", " ", line.strip()) == hdr:
                for j in range(i+1, min(i+20, len(lines))):
                    if lines[j].strip().upper().startswith("OVERALL"):
                        for c, v in zip(cols, lines[j].split()[1:]):
                            try: res[c] = float(v)
                            except: pass
                        break
                break
    return res


def build_combo_rows():
    rows = []
    for group, method, train_setting, thr in COMBOS:
        work_dir = osp.join(COMBO_ROOT, method)
        for idx, dom in enumerate(DOMS, 1):
            stage = f"stage{idx:02d}_after_{dom}"
            logpath = osp.join(work_dir, "eval", stage, "epoch_5", dom, "eval.log")
            m = parse_log(logpath)
            r = {f: "" for f in FIELDS}
            r.update(dict(
                group=group, method=method, train_setting=train_setting,
                eval_domain=dom, eval_epoch=5, stage=stage, thr_override=thr,
                eval_log=logpath if osp.isfile(logpath) else "NOT_FOUND",
            ))
            for k in ["MOTA","IDF1","HOTA","DetA","AssA","LocA","FP","FN","IDSw",
                      "Dets","IDs","Rcll","Prcn","DetPr","DetRe","FM","MOTP","MT","PT","ML"]:
                if k in m:
                    r[k] = m[k]
            rows.append(r)
    rows.extend(build_sweep_rows())
    return rows


def _read_text(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def build_sweep_rows():
    rows = []
    if not osp.isdir(COMBO_ROOT):
        return rows
    seen = set()
    for root, _, files in os.walk(COMBO_ROOT):
        if "eval.log" not in files:
            continue
        marker = osp.join(root, "eval_done.marker")
        if not osp.isfile(marker):
            continue
        logpath = osp.join(root, "eval.log")
        method = _read_text(osp.join(root, "method.txt"))
        group = _read_text(osp.join(root, "group.txt"))
        stage = _read_text(osp.join(root, "stage.txt"))
        dom = _read_text(osp.join(root, "eval_domain.txt"))
        thr = _read_text(osp.join(root, "cfg_options.txt"))
        if not (method or group or thr):
            continue
        rel = osp.relpath(root, COMBO_ROOT)
        parts = rel.split(os.sep)
        if not method:
            method = parts[-1]
        if not group:
            group = "tri_module_thr_sweep"
        if not dom:
            for d in DOMS:
                if d in parts or d == osp.basename(root):
                    dom = d
                    break
        if not stage and dom in DOMS:
            stage = f"stage{DOMS.index(dom)+1:02d}_after_{dom}"
        key = (method, dom, thr, logpath)
        if key in seen:
            continue
        seen.add(key)
        m = parse_log(logpath)
        r = {f: "" for f in FIELDS}
        r.update(dict(
            group=group, method=method, train_setting="4stage_sequential_5e",
            eval_domain=dom, eval_epoch=5, stage=stage, thr_override=thr,
            eval_log=logpath,
        ))
        for k in ["MOTA","IDF1","HOTA","DetA","AssA","LocA","FP","FN","IDSw",
                  "Dets","IDs","Rcll","Prcn","DetPr","DetRe","FM","MOTP","MT","PT","ML"]:
            if k in m:
                r[k] = m[k]
        rows.append(r)
    return rows


def load_existing():
    if not osp.isfile(SUMMARY_CSV):
        return [], []
    with open(SUMMARY_CSV, newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = reader.fieldnames
        rows = list(reader)
    return rows, existing_fields


def save_summary(rows, fields):
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k,"") for k in fields})


def compute_avgs(rows):
    by_method = defaultdict(list)
    for r in rows:
        by_method[(r["method"], r.get("train_setting",""), r.get("thr_override",""),
                   r.get("group",""))].append(r)
    avgs = []
    for (method, ts, thr, group), rs in sorted(by_method.items()):
        a = dict(group=group, method=method, train_setting=ts, thr_override=thr)
        for k in AVG_METRICS:
            vals = [float(r[k]) for r in rs if r.get(k) not in ("","NOT_FOUND",None)]
            a[f"AVG_{k}"] = round(sum(vals)/len(vals), 2) if vals else ""
        avgs.append(a)
    with open(AVG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AVG_FIELDS)
        w.writeheader()
        w.writerows(avgs)
    print(f"[WRITE] {AVG_CSV}  ({len(avgs)} rows)")
    return avgs


def _fs(v, w=7):
    if v in ("","NOT_FOUND",None): return "—".rjust(w)
    try: return f"{float(v):.1f}".rjust(w)
    except: return str(v).rjust(w)


def _fd(v, w=7):
    if v in ("","NOT_FOUND",None,"0.0",0.0,0): return "0.0".rjust(w)
    try:
        f = float(v)
        return (f"+{f:.1f}" if f > 0 else f"{f:.1f}").rjust(w)
    except: return str(v).rjust(w)


def write_md(avgs, all_rows):
    base_methods = [
        "baseline_4domain_5e",
        "causal_only_4domain_5e",
        "spa_pipeline_mild_4domain_5e",
        "roi_proto_v1_topscore_car_2stage_5e",
        "spaP1_roiV1_4domain_5e",
        "spaP1_causalOld_4domain_5e",
        "spaP1_causalOld_roiV1_4domain_5e",
    ]
    extra_methods = sorted(a["method"] for a in avgs
                           if a["method"] not in base_methods)
    COMPARE_METHODS = base_methods + extra_methods
    avg_by_method = {a["method"]: a for a in avgs}
    rows_by_method = defaultdict(dict)
    for r in all_rows:
        rows_by_method[r["method"]][r.get("eval_domain","")] = r

    lines = [
        "# Module Combination Results — Summary",
        "",
        "## Setup",
        "",
        "All 4-stage experiments: clear → overcast → rainy → dawn_dusk → night, 5 epochs/stage.",
        "eval @ epoch 5, default threshold unless noted.",
        "SPA-P1 eval threshold: `model.tracker.init_score_thr=0.30`",
        "",
        "> **Note**: `roi_proto_v1_topscore_car_2stage_5e` is a 2-stage experiment",
        "> (clear→overcast→night) and is **not directly comparable** to 4-stage results.",
        "> It is included only to identify the best RoI memory configuration.",
        "",
        "## 4-Domain Average",
        "",
        "| method | thr | MOTA | IDF1 | HOTA | DetA | AssA | FP | FN | IDSw |",
        "|--------|-----|------|------|------|------|------|-----|-----|------|",
    ]
    for m in COMPARE_METHODS:
        if m not in avg_by_method: continue
        a = avg_by_method[m]
        thr = "(default)" if not a.get("thr_override") else "P1"
        note = " *(2-stage)*" if "2stage" in m else ""
        lines.append(
            f"| {m}{note} | {thr}"
            f" | {_fs(a.get('AVG_MOTA'))} | {_fs(a.get('AVG_IDF1'))}"
            f" | {_fs(a.get('AVG_HOTA'))} | {_fs(a.get('AVG_DetA'))}"
            f" | {_fs(a.get('AVG_AssA'))} | {_fs(a.get('AVG_FP'),8)}"
            f" | {_fs(a.get('AVG_FN'),8)} | {_fs(a.get('AVG_IDSw'),6)} |")

    # delta vs spa_pipeline_mild
    ref_m = "spa_pipeline_mild_4domain_5e"
    if ref_m in avg_by_method:
        ref = avg_by_method[ref_m]
        lines += [
            "",
            "## Delta vs spa_pipeline_mild_4domain_5e (default thr)",
            "",
            "| method | dMOTA | dIDF1 | dHOTA | dDetA | dAssA | dFP | dFN | dIDSw |",
            "|--------|-------|-------|-------|-------|-------|-----|-----|-------|",
        ]
        for m in COMPARE_METHODS:
            if m not in avg_by_method or m == ref_m: continue
            a = avg_by_method[m]
            def d(k):
                try: return round(float(a.get(f"AVG_{k}",""))-float(ref.get(f"AVG_{k}","")),1)
                except: return ""
            lines.append(
                f"| {m} | {_fd(d('MOTA'))} | {_fd(d('IDF1'))} | {_fd(d('HOTA'))}"
                f" | {_fd(d('DetA'))} | {_fd(d('AssA'))} | {_fd(d('FP'),7)}"
                f" | {_fd(d('FN'),7)} | {_fd(d('IDSw'),6)} |")

    # per-domain night spotlight
    lines += [
        "",
        "## Night Domain Detail",
        "",
        "| method | thr | MOTA | IDF1 | HOTA | DetA | AssA | FP | FN | IDSw |",
        "|--------|-----|------|------|------|------|------|-----|-----|------|",
    ]
    for m in COMPARE_METHODS:
        r = rows_by_method.get(m, {}).get("night")
        if not r: continue
        thr = "(default)" if not r.get("thr_override") else "P1"
        note = " *(2-stage)*" if "2stage" in m else ""
        lines.append(
            f"| {m}{note} | {thr}"
            f" | {_fs(r.get('MOTA'))} | {_fs(r.get('IDF1'))}"
            f" | {_fs(r.get('HOTA'))} | {_fs(r.get('DetA'))}"
            f" | {_fs(r.get('AssA'))} | {_fs(r.get('FP'),8)}"
            f" | {_fs(r.get('FN'),8)} | {_fs(r.get('IDSw'),6)} |")

    lines += ["", "---", f"*Generated by tools/collect_module_components_summary.py*", ""]

    with open(MD_OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"[WRITE] {MD_OUT}")


def main():
    existing, fields = load_existing()
    if not fields:
        fields = FIELDS

    combo_rows = build_combo_rows()

    # merge/update rows. Back up when an existing row will be replaced.
    exist_key = lambda r: (r.get("method",""), r.get("eval_domain",""),
                           r.get("thr_override",""))
    combo_keys = {exist_key(r) for r in combo_rows}
    will_replace = any(exist_key(r) in combo_keys for r in existing)
    if will_replace and osp.isfile(SUMMARY_CSV):
        bak = SUMMARY_CSV + ".bak_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(SUMMARY_CSV, bak)
        print(f"[BACKUP] {bak}")
    kept = [r for r in existing if exist_key(r) not in combo_keys]
    all_rows = kept + [{k: r.get(k,"") for k in fields} for r in combo_rows]

    save_summary(all_rows, fields)
    print(f"[WRITE] {SUMMARY_CSV}  ({len(kept)} kept + {len(combo_rows)} combo = {len(all_rows)} total)")

    avgs = compute_avgs(all_rows)
    write_md(avgs, all_rows)

    # terminal preview of combo rows
    print()
    print(f"{'method':<36} {'dom':<10} {'MOTA':>6} {'IDF1':>6} {'HOTA':>6} {'FP':>7} {'FN':>7} {'IDSw':>5}")
    print("-" * 90)
    for r in combo_rows:
        print(f"{r['method']:<36} {r['eval_domain']:<10}"
              f" {_fs(r.get('MOTA'))} {_fs(r.get('IDF1'))} {_fs(r.get('HOTA'))}"
              f" {_fs(r.get('FP'),7)} {_fs(r.get('FN'),7)} {_fs(r.get('IDSw'),5)}")


if __name__ == "__main__":
    main()
