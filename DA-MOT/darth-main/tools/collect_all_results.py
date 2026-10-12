#!/usr/bin/env python3
"""Collect ALL results into one unified CSV.

Covers:
  - baseline (task2_full_episode, epoch5, each domain's own stage)
  - original_spa (spa_only_4domain_5e, epoch5)
  - threshold_sweep (on original_spa night ckpt, 7 variants)
  - spa_fix trained: spa_mild_A_4domain_5e, spa_mild_conf70_4domain_5e
    (and any future spa_fix/* that have eval_done.markers)

Output: work_dirs/bdd_4domain/all_results_summary.csv
"""
import csv
import glob
import os
import os.path as osp
import re

OUT_PATH = "work_dirs/bdd_4domain/all_results_summary.csv"

CLEARMOT_HDR = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CLEARMOT_COLS = ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn",
                 "MT", "PT", "ML", "FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS = ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA",
             "HOTA", "Dets", "IDs"]

METRICS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSw",
           "Dets", "IDs", "Rcll", "Prcn", "DetPr", "DetRe", "LocA",
           "FM", "MOTP", "MT", "PT", "ML"]

FIELDS = ["group", "method", "eval_domain", "eval_epoch", "stage",
          "thr_override"] + METRICS + ["eval_log"]


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


def mkrow(group, method, domain, epoch, stage, thr, metrics, log):
    r = dict(group=group, method=method, eval_domain=domain,
             eval_epoch=epoch, stage=stage, thr_override=thr, eval_log=log)
    for k in METRICS:
        r[k] = metrics.get(k, "")
    return r


rows = []

# ── 1. Baseline (task2_full_episode) – epoch5, per-domain own stage ──────────
BASELINE = "work_dirs/bdd_4domain/task2_full_episode/eval"
STAGE_DOM = {
    "stage01_after_overcast": "overcast",
    "stage02_after_rainy":    "rainy",
    "stage03_after_dawn_dusk": "dawn_dusk",
    "stage04_after_night":    "night",
}
for stage, dom in STAGE_DOM.items():
    log = f"{BASELINE}/{stage}/epoch_5/{dom}/eval.log"
    m = parse_log(log)
    if m:
        rows.append(mkrow("baseline", "baseline_4domain_5e", dom, 5,
                          stage, "", m, log))

# ── 2. Original SPA (spa_only_4domain_5e) – epoch5 ───────────────────────────
SPA_ORIG = "work_dirs/bdd_4domain/spa_only_4domain_5e/eval"
for stage, dom in STAGE_DOM.items():
    log = f"{SPA_ORIG}/{stage}/epoch_5/{dom}/eval.log"
    m = parse_log(log)
    if m:
        rows.append(mkrow("original_spa", "spa_only_4domain_5e", dom, 5,
                          stage, "", m, log))

# ── 3. Threshold sweep (on original_spa night ckpt) ──────────────────────────
THR_LABELS = {
    "s0_original":     "default (init=0.2 obj=0.2)",
    "s1_det_plus005":  "score_thr=0.10",
    "s2_det_plus010":  "score_thr=0.15",
    "s3_init_plus010": "init_score_thr=0.30",
    "s4_track_plus010":"obj_score_thr=0.30",
    "s5_det_init_plus010": "score_thr=0.15+init=0.30",
    "s6_det_track_plus010":"score_thr=0.15+obj=0.30",
}
THR_ROOT = "work_dirs/bdd_4domain/spa_fix/threshold_sweep"
for sname, label in THR_LABELS.items():
    log = f"{THR_ROOT}/{sname}/night/eval.log"
    m = parse_log(log)
    if m:
        rows.append(mkrow("threshold_sweep", sname, "night", 5,
                          "stage04_after_night", label, m, log))

# ── 4. SPA-fix trained experiments ───────────────────────────────────────────
SPA_FIX = "work_dirs/bdd_4domain/spa_fix"
SKIP_DIRS = {"threshold_sweep", "summary"}
for exp_dir in sorted(glob.glob(f"{SPA_FIX}/*")):
    exp = osp.basename(exp_dir)
    if exp in SKIP_DIRS or not osp.isdir(exp_dir):
        continue
    for marker in sorted(glob.glob(
            f"{exp_dir}/eval/stage*/epoch_5/*/eval_done.marker")):
        elog = osp.join(osp.dirname(marker), "eval.log")
        m = parse_log(elog)
        if not m:
            continue
        dom = osp.basename(osp.dirname(marker))
        stage = marker.split("/eval/")[1].split("/")[0]
        rows.append(mkrow("spa_fix", exp, dom, 5, stage, "", m, elog))

# ── write ─────────────────────────────────────────────────────────────────────
os.makedirs(osp.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=FIELDS)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in FIELDS})

print(f"[WRITE] {OUT_PATH}  ({len(rows)} rows)")

# ── pretty print to terminal ──────────────────────────────────────────────────
GROUP_ORDER = ["baseline", "original_spa", "threshold_sweep", "spa_fix"]
DOM_ORDER   = ["overcast", "rainy", "dawn_dusk", "night"]

print()
print(f"{'group':<18} {'method':<32} {'domain':<12} {'ep':>3}  "
      f"{'MOTA':>6} {'IDF1':>6} {'HOTA':>6} {'DetA':>6} {'AssA':>6}  "
      f"{'FP':>7} {'FN':>7} {'IDSw':>5}  {'thr_override'}")
print("-" * 140)

for g in GROUP_ORDER:
    group_rows = [r for r in rows if r["group"] == g]
    # sort: baseline/orig by domain order; sweep by sname; spa_fix by method+domain
    if g == "threshold_sweep":
        key = list(THR_LABELS.keys())
        group_rows.sort(key=lambda r: key.index(r["method"]) if r["method"] in key else 99)
    elif g == "spa_fix":
        group_rows.sort(key=lambda r: (r["method"],
                                       DOM_ORDER.index(r["eval_domain"])
                                       if r["eval_domain"] in DOM_ORDER else 9))
    else:
        group_rows.sort(key=lambda r: DOM_ORDER.index(r["eval_domain"])
                        if r["eval_domain"] in DOM_ORDER else 9)
    for r in group_rows:
        print(f"{r['group']:<18} {r['method']:<32} {r['eval_domain']:<12} "
              f"{str(r['eval_epoch']):>3}  "
              f"{str(r.get('MOTA','')):>6} {str(r.get('IDF1','')):>6} "
              f"{str(r.get('HOTA','')):>6} {str(r.get('DetA','')):>6} "
              f"{str(r.get('AssA','')):>6}  "
              f"{str(r.get('FP','')):>7} {str(r.get('FN','')):>7} "
              f"{str(r.get('IDSw','')):>5}  {r.get('thr_override','')}")
    print()
