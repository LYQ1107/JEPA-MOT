#!/usr/bin/env python3
"""Collect baseline threshold sweep results and compare with SPA sweep.

Reads:
  work_dirs/bdd_4domain/baseline_threshold_sweep/night/B*/eval.log
  work_dirs/bdd_4domain/spa_fix/threshold_sweep/s*/night/eval.log  (SPA reference)

Outputs:
  work_dirs/bdd_4domain/baseline_threshold_sweep/baseline_threshold_sweep_night_summary.csv
  work_dirs/bdd_4domain/baseline_threshold_sweep/baseline_threshold_sweep_night_summary.md
"""
import csv
import os
import os.path as osp
import re

OUT_DIR = "work_dirs/bdd_4domain/baseline_threshold_sweep"
CSV_OUT = f"{OUT_DIR}/baseline_threshold_sweep_night_summary.csv"
MD_OUT  = f"{OUT_DIR}/baseline_threshold_sweep_night_summary.md"

CLEARMOT_HDR = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CLEARMOT_COLS = ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn",
                 "MT", "PT", "ML", "FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS = ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA",
             "HOTA", "Dets", "IDs"]

METRIC_KEYS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "LocA",
               "FP", "FN", "IDSw", "Dets", "IDs", "FM", "Prcn", "Rcll",
               "DetPr", "DetRe", "MOTP"]
DELTA_KEYS  = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSw",
               "Dets", "IDs"]

# threshold settings recorded in cfg_options.txt per variant
BASELINE_THR_LABELS = {
    "B0_default":         {"score_thr": 0.05,  "init_score_thr": 0.20, "obj_score_thr": 0.20,
                           "match_score_thr": 0.50, "label": "default (init=0.2 obj=0.2)"},
    "B1_init030":         {"score_thr": 0.05,  "init_score_thr": 0.30, "obj_score_thr": 0.20,
                           "match_score_thr": 0.50, "label": "init_score_thr=0.30"},
    "B2_obj030":          {"score_thr": 0.05,  "init_score_thr": 0.20, "obj_score_thr": 0.30,
                           "match_score_thr": 0.50, "label": "obj_score_thr=0.30"},
    "B3_score015_obj030": {"score_thr": 0.15,  "init_score_thr": 0.20, "obj_score_thr": 0.30,
                           "match_score_thr": 0.50, "label": "score_thr=0.15+obj=0.30"},
}

SPA_THR_LABELS = {
    "s0_original":        {"label": "SPA default (init=0.2 obj=0.2)"},
    "s3_init_plus010":    {"label": "SPA + init_score_thr=0.30"},
    "s4_track_plus010":   {"label": "SPA + obj_score_thr=0.30"},
    "s6_det_track_plus010": {"label": "SPA + score_thr=0.15+obj=0.30"},
}


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


def _delta(row, ref):
    for k in DELTA_KEYS:
        col = f"delta_vs_baseline_default_{k}"
        if ref and row.get(k) not in ("", None) and ref.get(k) not in ("", None):
            try:
                row[col] = round(float(row[k]) - float(ref[k]), 3)
            except (ValueError, TypeError):
                row[col] = ""
        else:
            row[col] = ""


# ── collect baseline rows ─────────────────────────────────────────────────────
baseline_rows = []
for bname, thr_info in BASELINE_THR_LABELS.items():
    elog = f"{OUT_DIR}/night/{bname}/eval.log"
    m = parse_log(elog)
    ckpt_file = f"{OUT_DIR}/night/{bname}/checkpoint_path.txt"
    ckpt = open(ckpt_file).read().strip() if osp.isfile(ckpt_file) else \
        "work_dirs/bdd_4domain/task2_full_episode/episode/adapt/stage04_after_night/epoch_5.pth"
    row = dict(
        method="baseline", run="baseline_4domain_5e",
        eval_domain="night", stage="stage04_after_night", epoch=5,
        threshold_setting=thr_info["label"],
        score_thr=thr_info["score_thr"],
        init_score_thr=thr_info["init_score_thr"],
        obj_score_thr=thr_info["obj_score_thr"],
        match_score_thr=thr_info["match_score_thr"],
        track_score_thr="",
        checkpoint=ckpt,
        eval_log=elog if osp.isfile(elog) else "NOT_FOUND",
    )
    for k in METRIC_KEYS:
        row[k] = m.get(k, "") if m else "NOT_FOUND"
    baseline_rows.append(row)

# compute deltas vs B0
ref = {k: baseline_rows[0].get(k) for k in DELTA_KEYS} if baseline_rows else {}
for r in baseline_rows:
    _delta(r, ref)

# ── collect SPA rows ──────────────────────────────────────────────────────────
spa_rows = []
SPA_ROOT = "work_dirs/bdd_4domain/spa_fix/threshold_sweep"
for sname, info in SPA_THR_LABELS.items():
    elog = f"{SPA_ROOT}/{sname}/night/eval.log"
    m = parse_log(elog)
    row = dict(
        method="original_spa", run="spa_only_4domain_5e",
        eval_domain="night", stage="stage04_after_night", epoch=5,
        threshold_setting=info["label"],
        score_thr="", init_score_thr="", obj_score_thr="",
        match_score_thr="", track_score_thr="",
        checkpoint="work_dirs/bdd_4domain/spa_only_4domain_5e/episode/adapt/stage04_after_night/epoch_5.pth",
        eval_log=elog if osp.isfile(elog) else "NOT_FOUND",
    )
    for k in METRIC_KEYS:
        row[k] = m.get(k, "") if m else "NOT_FOUND"
    # delta vs baseline B0 default
    _delta(row, ref)
    spa_rows.append(row)

# ── write CSV ─────────────────────────────────────────────────────────────────
FIELDS = (["method", "run", "eval_domain", "stage", "epoch",
           "threshold_setting", "score_thr", "init_score_thr",
           "obj_score_thr", "match_score_thr", "track_score_thr"]
          + METRIC_KEYS
          + [f"delta_vs_baseline_default_{k}" for k in DELTA_KEYS]
          + ["checkpoint", "eval_log"])

os.makedirs(OUT_DIR, exist_ok=True)
all_rows = baseline_rows + spa_rows
with open(CSV_OUT, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=FIELDS)
    w.writeheader()
    for r in all_rows:
        w.writerow({k: r.get(k, "") for k in FIELDS})
print(f"[WRITE] {CSV_OUT}  ({len(all_rows)} rows)")

# ── write markdown ────────────────────────────────────────────────────────────
def _fmt(v, width=7):
    if v in ("", None, "NOT_FOUND"):
        return ("—").rjust(width)
    try:
        return f"{float(v):.1f}".rjust(width)
    except (ValueError, TypeError):
        return str(v).rjust(width)

def _fmt_delta(v, width=7):
    if v in ("", None, "NOT_FOUND"):
        return "—".rjust(width)
    try:
        f = float(v)
        sign = "+" if f > 0 else ""
        return f"{sign}{f:.1f}".rjust(width)
    except (ValueError, TypeError):
        return str(v).rjust(width)

md_lines = [
    "# Baseline vs SPA Threshold Sweep — Night (epoch 5, stage04_after_night)",
    "",
    "Eval config: `configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_night.py`",
    "Night eval config default: `init_score_thr=0.20  obj_score_thr=0.20  score_thr=0.05`",
    "",
    "## Results Table",
    "",
    "| # | method | threshold_setting | MOTA | IDF1 | HOTA | DetA | AssA | FP | FN | IDSw | "
    "dMOTA | dIDF1 | dHOTA | dFP | dFN | dIDSw |",
    "|---|--------|-------------------|------|------|------|------|------|-----|-----|------|"
    "-------|-------|-------|-----|-----|-------|",
]

for i, r in enumerate(all_rows):
    md_lines.append(
        f"| {i} "
        f"| {r['method']} "
        f"| {r['threshold_setting']} "
        f"| {_fmt(r.get('MOTA'))} "
        f"| {_fmt(r.get('IDF1'))} "
        f"| {_fmt(r.get('HOTA'))} "
        f"| {_fmt(r.get('DetA'))} "
        f"| {_fmt(r.get('AssA'))} "
        f"| {_fmt(r.get('FP'),8)} "
        f"| {_fmt(r.get('FN'),8)} "
        f"| {_fmt(r.get('IDSw'),6)} "
        f"| {_fmt_delta(r.get('delta_vs_baseline_default_MOTA'))} "
        f"| {_fmt_delta(r.get('delta_vs_baseline_default_IDF1'))} "
        f"| {_fmt_delta(r.get('delta_vs_baseline_default_HOTA'))} "
        f"| {_fmt_delta(r.get('delta_vs_baseline_default_FP'),8)} "
        f"| {_fmt_delta(r.get('delta_vs_baseline_default_FN'),8)} "
        f"| {_fmt_delta(r.get('delta_vs_baseline_default_IDSw'))} |"
    )

md_lines += [
    "",
    "## Key Questions",
    "",
    "| Question | Answer |",
    "|----------|--------|",
]

# auto-answer key questions if data is available
b0_mota = baseline_rows[0].get("MOTA", "") if baseline_rows else ""
b_best  = max((r for r in baseline_rows if r.get("MOTA") not in ("", None, "NOT_FOUND")),
              key=lambda r: float(r["MOTA"]), default=None)
s_best  = max((r for r in spa_rows if r.get("MOTA") not in ("", None, "NOT_FOUND")),
              key=lambda r: float(r["MOTA"]), default=None)

def _v(r, k):
    return r.get(k, "—") if r else "—"

b_best_mota = float(_v(b_best, "MOTA")) if b_best and _v(b_best,"MOTA") != "—" else None
s_best_mota = float(_v(s_best, "MOTA")) if s_best and _v(s_best,"MOTA") != "—" else None
b0_mota_f   = float(b0_mota) if b0_mota not in ("", None, "NOT_FOUND") else None

if b_best_mota is not None and b0_mota_f is not None:
    b_gain = round(b_best_mota - b0_mota_f, 1)
    q1 = f"YES — baseline calibrated best MOTA={b_best_mota} (+{b_gain} vs default)" if b_gain > 1 \
         else f"SMALL — baseline calibrated best MOTA={b_best_mota} (only +{b_gain} vs default)"
else:
    q1 = "NOT_FOUND (eval not complete)"

if b_best_mota is not None:
    q2 = f"Baseline calibrated best: MOTA={b_best_mota} ({_v(b_best,'threshold_setting')})"
else:
    q2 = "NOT_FOUND"

if b_best_mota is not None and s_best_mota is not None:
    diff = round(s_best_mota - b_best_mota, 1)
    q3 = (f"YES — SPA calibrated MOTA={s_best_mota} vs Baseline calibrated MOTA={b_best_mota} (+{diff})"
          if diff > 0 else
          f"NO  — SPA calibrated MOTA={s_best_mota} <= Baseline calibrated MOTA={b_best_mota} ({diff})")
else:
    q3 = "NOT_FOUND"

md_lines += [
    f"| Q1: Baseline calibrated also improves significantly? | {q1} |",
    f"| Q2: Best baseline calibrated result | {q2} |",
    f"| Q3: SPA calibrated still better than baseline calibrated? | {q3} |",
    "",
    "## Interpretation",
    "",
    "- If **Baseline calibrated ≈ Baseline default**: threshold is SPA-specific; SPA genuinely helps.",
    "- If **Baseline calibrated also gains significantly**: threshold is a general test-time factor;",
    "  calibrated baseline must be used as the fair comparison for all SPA results.",
    "- If **SPA calibrated > Baseline calibrated**: SPA still adds value above calibrated baseline.",
    "- If **SPA calibrated ≤ Baseline calibrated**: SPA noise exceeds its benefit even after calibration.",
]

with open(MD_OUT, "w") as fp:
    fp.write("\n".join(md_lines) + "\n")
print(f"[WRITE] {MD_OUT}")

# ── terminal summary ──────────────────────────────────────────────────────────
print()
print(f"{'#':<3} {'method':<14} {'threshold':<35} {'MOTA':>6} {'IDF1':>6} {'HOTA':>6} "
      f"{'FP':>7} {'FN':>7} {'IDSw':>5}  {'dMOTA':>7}")
print("-" * 105)
for i, r in enumerate(all_rows):
    print(f"{i:<3} {r['method']:<14} {r['threshold_setting']:<35} "
          f"{_fmt(r.get('MOTA')):>6} {_fmt(r.get('IDF1')):>6} {_fmt(r.get('HOTA')):>6} "
          f"{_fmt(r.get('FP'),7):>7} {_fmt(r.get('FN'),7):>7} {_fmt(r.get('IDSw'),5):>5}  "
          f"{_fmt_delta(r.get('delta_vs_baseline_default_MOTA')):>7}")
    if i == len(baseline_rows) - 1:
        print("-" * 105)

print()
print(f"[Q1] {q1}")
print(f"[Q2] {q2}")
print(f"[Q3] {q3}")


if __name__ == "__main__":
    pass
