#!/usr/bin/env python3
"""Collect spa_style_replace_clear_to_night_5e results into CSV + MD.

Outputs:
  work_dirs/bdd_4domain/spa_style_replace_clear_to_night_5e/metrics_epoch3_5.csv
  work_dirs/bdd_4domain/spa_style_replace_clear_to_night_5e/summary_epoch3_5.md
"""
import csv
import glob
import os.path as osp
import re

ROOT = "work_dirs/bdd_4domain/spa_style_replace_clear_to_night_5e"
EVAL_ROOT = osp.join(ROOT, "eval")

CLEARMOT_HDR = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CLEARMOT_COLS = ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn",
                 "MT", "PT", "ML", "FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS = ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA",
             "HOTA", "Dets", "IDs"]

FIELDS = ["epoch", "eval_domain", "MOTA", "IDF1", "HOTA", "DetA", "AssA",
          "DetRe", "DetPr", "Rcll", "Prcn", "FP", "FN", "IDSw", "FM",
          "Dets", "IDs", "eval_dir"]


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


def main():
    rows = []
    for log in sorted(glob.glob(f"{EVAL_ROOT}/**/eval.log", recursive=True)):
        rel = log[len(EVAL_ROOT) + 1:].replace("/eval.log", "")
        parts = rel.split("/")
        epoch = ""
        for p in parts:
            m = re.match(r"epoch_(\d+)", p)
            if m:
                epoch = m.group(1)
        domain = parts[-1]
        m = parse_eval_log(log)
        if not m:
            continue
        row = {"epoch": epoch, "eval_domain": domain, "eval_dir": osp.dirname(log)}
        for k in FIELDS:
            if k in ("epoch", "eval_domain", "eval_dir"):
                continue
            row[k] = m.get(k, "")
        rows.append(row)

    csv_path = osp.join(ROOT, "metrics_epoch3_5.csv")
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[WRITE] {csv_path}  ({len(rows)} rows)")

    md = [f"# {osp.basename(ROOT)} — night eval (epoch 3 & 5)", ""]
    md.append("| epoch | MOTA | IDF1 | HOTA | DetA | AssA | DetPr | Prcn | "
              "FP | FN | IDSw | Dets | IDs |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: x["epoch"]):
        md.append("| {epoch} | {MOTA} | {IDF1} | {HOTA} | {DetA} | {AssA} | "
                  "{DetPr} | {Prcn} | {FP} | {FN} | {IDSw} | {Dets} | {IDs} |"
                  .format(**r))
    md_path = osp.join(ROOT, "summary_epoch3_5.md")
    with open(md_path, "w") as fp:
        fp.write("\n".join(md) + "\n")
    print(f"[WRITE] {md_path}")


if __name__ == "__main__":
    main()
