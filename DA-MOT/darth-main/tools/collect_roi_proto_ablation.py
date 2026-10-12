#!/usr/bin/env python3
"""Collect RoI Prototype Memory ablation results -> summary.csv + summary.md.

Scans work_dirs/bdd_4domain/roi_proto_ablation/, parses each version's night
epoch3 eval log, pulls RoI proto memory stats from the train log, and computes
deltas relative to V0 control.
"""
import csv
import glob
import os.path as osp
import re

ROOT = "work_dirs/bdd_4domain/roi_proto_ablation"

CLEARMOT_HDR = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CLEARMOT_COLS = ["IDF1", "MOTA", "MOTP", "FP", "FN", "IDSw", "Rcll", "Prcn",
                 "MT", "PT", "ML", "FM"]
HOTA_HDR = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLS = ["AssRe", "AssPr", "AssA", "LocA", "DetRe", "DetPr", "DetA",
             "HOTA", "Dets", "IDs"]

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


def parse_proto_stats(train_log):
    """Pull last ROI_PROTO_LOSS stats + finalize prototype counts."""
    stats = dict(roi_proto_candidates="", roi_proto_prototypes="",
                 roi_proto_loss_mean="", roi_proto_mean_topk_sim="")
    try:
        txt = open(train_log).read().splitlines()
    except OSError:
        return stats
    losses, sims = [], []
    for line in txt:
        m = re.search(r"ROI_PROTO_LOSS.*prototypes=(\d+).*"
                      r"mean_topk_sim=([\d.]+).*loss=([\d.]+)", line)
        if m:
            stats["roi_proto_prototypes"] = m.group(1)
            sims.append(float(m.group(2)))
            losses.append(float(m.group(3)))
        f = re.search(r"Finalize domain \w+: prototypes=\{.*?(\d+)\}", line)
        if f:
            stats["roi_proto_candidates"] = f.group(1)
    if losses:
        stats["roi_proto_loss_mean"] = round(sum(losses) / len(losses), 5)
    if sims:
        stats["roi_proto_mean_topk_sim"] = round(sum(sims) / len(sims), 5)
    return stats


FIELDS = ["version", "target_order", "eval_domain", "eval_epoch", "MOTA",
          "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSw", "Dets", "IDs",
          "FM", "roi_proto_candidates", "roi_proto_prototypes",
          "roi_proto_loss_mean", "roi_proto_mean_topk_sim", "work_dir",
          "checkpoint", "eval_log"]
DELTA_KEYS = ["MOTA", "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSw",
              "Dets", "IDs"]


def _find_eval(vdir, epochs=(5, 3)):
    """Return (eval_log, ckpt, epoch) for the first epoch that has an eval."""
    for ep in epochs:
        elog = f"{vdir}/eval/epoch_{ep}/night/eval.log"
        if osp.isfile(elog):
            ckpt = f"{vdir}/episode/adapt/stage02_after_night/epoch_{ep}.pth"
            return elog, ckpt, ep
    return None, None, None


def collect_rows():
    """Scan all version dirs; auto-detect epoch5 (V0-V3) or epoch3 (V1a/V1b)."""
    rows = []
    for vdir in sorted(glob.glob(f"{ROOT}/*")):
        if not osp.isdir(vdir):
            continue
        version = osp.basename(vdir)
        elog, ckpt, ep = _find_eval(vdir)
        if elog is None:
            continue
        m = parse_eval_log(elog)
        if not m:
            continue
        tlog = f"{vdir}/logs/train_episode.log"
        row = dict(version=version, target_order="overcast->night",
                   eval_domain="night", eval_epoch=ep,
                   work_dir=vdir, checkpoint=ckpt, eval_log=elog)
        for k in ("MOTA", "IDF1", "HOTA", "DetA", "AssA", "FP", "FN", "IDSw",
                  "Dets", "IDs", "FM"):
            row[k] = m.get(k, "")
        row.update(parse_proto_stats(tlog))
        rows.append(row)
    return rows


def _add_deltas(rows, ref_prefix, tag):
    # ref_prefix like "v0_" or "v1_" matches only the exact control/main
    # version, not v1a_/v1b_.
    ref = next((r for r in rows if r["version"].startswith(ref_prefix)), None)
    for r in rows:
        for k in DELTA_KEYS:
            col = f"delta_vs_{tag}_{k}"
            if ref and r.get(k) not in ("", None) and ref.get(k) not in ("", None):
                try:
                    r[col] = round(float(r[k]) - float(ref[k]), 3)
                except (ValueError, TypeError):
                    r[col] = ""
            else:
                r[col] = ""
    return ref


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1ab", action="store_true",
                    help="write summary_v1ab_quick_verify.* with vs_v0 + vs_v1 deltas")
    args = ap.parse_args()

    rows = collect_rows()

    if args.v1ab:
        _add_deltas(rows, "v0_", "v0")
        _add_deltas(rows, "v1_", "v1")
        delta_fields = [f"delta_vs_v0_{k}" for k in DELTA_KEYS] + \
                       [f"delta_vs_v1_{k}" for k in DELTA_KEYS]
        all_fields = FIELDS + delta_fields
        csv_path = f"{ROOT}/summary_v1ab_quick_verify.csv"
        md_path = f"{ROOT}/summary_v1ab_quick_verify.md"
        title = "# RoI Prototype Memory quick-verify (V1a/V1b) — night"
    else:
        _add_deltas(rows, "v0_", "v0")
        all_fields = FIELDS + [f"delta_vs_v0_{k}" for k in DELTA_KEYS]
        csv_path = f"{ROOT}/summary.csv"
        md_path = f"{ROOT}/summary.md"
        title = "# RoI Prototype Memory ablation — night"

    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=all_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_fields})
    print(f"[WRITE] {csv_path}  ({len(rows)} versions)")

    md = [title, "",
          "| version | ep | MOTA | IDF1 | HOTA | DetA | AssA | IDSw | FP | FN | "
          "Dets | dV0_IDSw | dV0_AssA |"]
    md.append("|" + "---|" * 13)
    for r in sorted(rows, key=lambda x: x["version"]):
        r.setdefault("delta_vs_v0_IDSw", "")
        r.setdefault("delta_vs_v0_AssA", "")
        md.append(
            "| {version} | {eval_epoch} | {MOTA} | {IDF1} | {HOTA} | {DetA} | "
            "{AssA} | {IDSw} | {FP} | {FN} | {Dets} | {delta_vs_v0_IDSw} | "
            "{delta_vs_v0_AssA} |".format(**r))
    with open(md_path, "w") as fp:
        fp.write("\n".join(md) + "\n")
    print(f"[WRITE] {md_path}")


if __name__ == "__main__":
    main()
