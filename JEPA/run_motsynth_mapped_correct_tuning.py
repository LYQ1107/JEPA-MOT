#!/usr/bin/env python3
"""Correct-protocol MOTSynth V-JEPA 2.1 tuning and audit.

This run is deliberately separate from the earlier dense-preparation run.
It uses the mapped source_detector_thr02 preparation (detector-covered frames
only), reuses the verified mapped GeneralTrack baseline, selects JEPA-only
thresholds on disjoint tune videos, and reports untouched holdout plus full
mapped-domain results.
"""

import gc
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np


JEPA_ROOT = Path("/data1/liuyeqiang/JEPA_MOT_resume_20261012/JEPA")
REPO = Path("/data1/liuyeqiang/JEPA_MOT_resume_20261012/GeneralTrack-main")
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
DATASET = "motsynth"
DOMAINS = ("cloudy", "dark", "fog", "rain", "snow")
GPU_IDS = (5, 6)

FULL_PREPARED_ROOT = REPO / "datasets/generaltrack_prepared_mapped/source_detector_thr02/motsynth"
SPLIT_ROOT = JEPA_ROOT / "outputs/jepa_motsynth_mapped_splits"
CACHE_ROOT = Path(os.environ.get(
    "JEPA_CORRECT_CACHE_ROOT",
    str(JEPA_ROOT / "outputs/official_vjepa21_cache_mapped_source_detector_thr02_score0"),
))
BASELINE_ROOT = REPO / "outputs/reference_archive/generaltrack_mapped/source_detector_thr02/motsynth"
RUN_ROOT = Path(os.environ.get(
    "JEPA_CORRECT_RUN_ROOT",
    str(REPO / "outputs/jepa_motsynth_mapped_correct_tuning_vjepa21_score0"),
))
TUNE_ROOT = RUN_ROOT / "tune"
HOLDOUT_TUNED_ROOT = RUN_ROOT / "holdout_tuned"
HOLDOUT_FROZEN_ROOT = RUN_ROOT / "holdout_frozen"
HOLDOUT_NULL_ROOT = RUN_ROOT / "holdout_geometry_null"
HOLDOUT_BASELINE_ROOT = RUN_ROOT / "holdout_baseline"
FULL_TUNED_ROOT = RUN_ROOT / "full_tuned"
FULL_FROZEN_ROOT = RUN_ROOT / "full_frozen"
FULL_NULL_ROOT = RUN_ROOT / "full_geometry_null"

LOG_ROOT = Path("/tmp/jepa_motsynth_mapped_correct_logs")
STATE_LOG = LOG_ROOT / "coordinator.log"
REPORT = JEPA_ROOT / "reports/JEPA_MOTSynth_mapped_correct_tuning_audit.md"
SUMMARY_JSON = JEPA_ROOT / "reports/JEPA_MOTSynth_mapped_correct_tuning_summary.json"

CACHE_NAME = "motsynth_{domain}_vjepa21_l384_w16_causalbox_predictor.npz"
FIELDS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN")
CANDIDATES = (
    {"name": "s085_iou000", "sim": 0.85, "iou": 0.00},
    {"name": "s085_iou005", "sim": 0.85, "iou": 0.05},
    {"name": "s090_iou000", "sim": 0.90, "iou": 0.00},
    {"name": "s090_iou005", "sim": 0.90, "iou": 0.05},
    {"name": "s095_iou000", "sim": 0.95, "iou": 0.00},
    {"name": "s095_iou005", "sim": 0.95, "iou": 0.05},
)
FROZEN_CANDIDATE = next(c for c in CANDIDATES if c["name"] == "s090_iou005")


def log(message):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}\n"
    with STATE_LOG.open("a") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def load_summary(path):
    try:
        data = json.loads(Path(path).read_text())
        summary = data.get("summary", data)
        if all(field in summary for field in FIELDS):
            return summary
    except (OSError, ValueError, TypeError):
        pass
    return None


def metric_done(path):
    return load_summary(path) is not None


def split_dir(domain, split):
    if split == "full":
        return FULL_PREPARED_ROOT / domain
    return SPLIT_ROOT / DATASET / domain / split


def cache_path(domain, split, geometry_null=False):
    if split == "full":
        name = CACHE_NAME.format(domain=domain)
        if geometry_null:
            name = name.replace(".npz", "_geometry_null.npz")
        return CACHE_ROOT / name
    if geometry_null:
        return split_dir(domain, split) / "vjepa21_features_geometry_null.npz"
    return split_dir(domain, split) / "vjepa21_features.npz"


def ensure_null_split_caches():
    for domain in DOMAINS:
        for split in ("tune", "holdout"):
            source = split_dir(domain, split) / "vjepa21_features.npz"
            target = cache_path(domain, split, geometry_null=True)
            with np.load(source, allow_pickle=False) as data:
                features = np.zeros(data["features"].shape, dtype=np.float32)
                offsets = np.asarray(data["offsets"], dtype=np.int64)
                image_ids = np.asarray(data["image_ids"], dtype=np.int64)
            target_is_current = False
            if target.is_file():
                try:
                    with np.load(target, allow_pickle=False) as old:
                        target_is_current = (
                            tuple(old["features"].shape) == tuple(features.shape)
                            and np.array_equal(old["offsets"], offsets)
                            and np.array_equal(old["image_ids"], image_ids)
                        )
                except (OSError, ValueError, KeyError):
                    target_is_current = False
            if target_is_current:
                del features, offsets, image_ids
                gc.collect()
                continue
            np.savez_compressed(target, features=features, offsets=offsets,
                                image_ids=image_ids)
            del features, offsets, image_ids
            gc.collect()
            log(f"NULL_CACHE_READY {domain}/{split}")


def candidate_output(split, domain, candidate):
    if split == "tune":
        return TUNE_ROOT / candidate["name"] / domain
    if split == "holdout_tuned":
        return HOLDOUT_TUNED_ROOT / domain
    if split == "holdout_frozen":
        return HOLDOUT_FROZEN_ROOT / domain
    if split == "full_tuned":
        return FULL_TUNED_ROOT / domain
    if split == "full_frozen":
        return FULL_FROZEN_ROOT / domain
    raise ValueError(split)


def variant_output(variant, domain, full=False):
    roots = {
        "geometry_null": FULL_NULL_ROOT if full else HOLDOUT_NULL_ROOT,
    }
    return roots[variant] / domain


def tracker_command(domain, prepared, output, candidate, cache):
    return [
        PYTHON, "-u", str(REPO / "tools/track_damot_domain.py"),
        "--dataset", DATASET, "--domain", domain,
        "--prepared-dir", str(prepared), "--output-dir", str(output),
        "--track_thresh", "0.35", "--det_thresh", "0.45",
        "--track_buffer", "50", "--match_thresh", "0.90",
        "--matchlost_thresh", "0.90",
        "--restore_ckpt", str(REPO / "checkpoints/generaltrack_bdd.pth"),
        "--resize", "720", "1280",
        "--jepa-cache", str(cache), "--jepa-feature-dim", "1664",
        "--jepa-mode", "high_fallback_duplicate_geom_jepa",
        "--jepa-high-min-sim", str(candidate["sim"]),
        "--jepa-high-iou", str(candidate["iou"]),
        "--jepa-relative-margin", "0", "--jepa-max-gap", "8",
        "--jepa-history-aggregation", "last",
        "--jepa-duplicate-min-sim", "0.90", "--jepa-duplicate-iou", "0.20",
    ]


def eval_command(domain, prepared, track_results, output, method, protocol):
    return [
        PYTHON, "-u", str(REPO / "tools/eval_damot_generaltrack.py"),
        "--dataset", DATASET, "--domain", domain,
        "--prepared-dir", str(prepared),
        "--track-results", str(track_results),
        "--output-dir", str(output / "eval"),
        "--summary-csv", str(output / "eval/summary.csv"),
        "--method", method, "--protocol", protocol,
    ]


def run_tracker_variant(domain, prepared, output, candidate, cache, method,
                        protocol, gpu, key):
    metric = output / "eval/metrics.json"
    if metric_done(metric):
        log(f"SKIP {key} {domain}")
        return True
    output.mkdir(parents=True, exist_ok=True)
    track_cmd = tracker_command(domain, prepared, output, candidate, cache)
    eval_cmd = eval_command(domain, prepared, output / "track_results", output,
                            method, protocol)
    log_path = LOG_ROOT / f"{key}_{domain}.log"
    for attempt in (1, 2):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a") as handle:
            handle.write(f"\n=== attempt={attempt} gpu={gpu} ===\n")
            handle.write("TRACK_CMD=" + " ".join(track_cmd) + "\n")
            handle.flush()
            track_rc = subprocess.run(track_cmd, cwd=str(REPO), env=env,
                                      stdout=handle, stderr=subprocess.STDOUT).returncode
            handle.write(f"TRACK_EXIT_CODE={track_rc}\n")
            if track_rc == 0:
                handle.write("EVAL_CMD=" + " ".join(eval_cmd) + "\n")
                handle.flush()
                eval_rc = subprocess.run(eval_cmd, cwd=str(REPO), env=env,
                                         stdout=handle, stderr=subprocess.STDOUT).returncode
            else:
                eval_rc = -1
            handle.write(f"EVAL_EXIT_CODE={eval_rc}\n")
            handle.flush()
        if track_rc == 0 and eval_rc == 0 and metric_done(metric):
            log(f"DONE {key} {domain} gpu={gpu}")
            return True
        log(f"RETRY {key} {domain} attempt={attempt} track_rc={track_rc} eval_rc={eval_rc}")
        time.sleep(10)
    log(f"FAILED {key} {domain}")
    return False


def run_tune_queue():
    jobs = [(domain, candidate) for domain in DOMAINS for candidate in CANDIDATES]
    log(f"TUNE_QUEUE_START jobs={len(jobs)} gpus={GPU_IDS}")

    def worker(item):
        index, (domain, candidate) = item
        return run_tracker_variant(
            domain, split_dir(domain, "tune"), candidate_output("tune", domain, candidate),
            candidate, cache_path(domain, "tune"),
            "MOTSynth_VJEPA21_TUNED_" + candidate["name"],
            "target-video-split-tuning-mapped", GPU_IDS[index % len(GPU_IDS)], "tune",
        )

    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as pool:
        results = list(pool.map(worker, enumerate(jobs)))
    log(f"TUNE_QUEUE_FINISHED ok={sum(results)}/{len(results)}")
    return all(results)


def select_candidate():
    table = {}
    for candidate in CANDIDATES:
        rows = []
        for domain in DOMAINS:
            metric = load_summary(
                candidate_output("tune", domain, candidate) / "eval/metrics.json"
            )
            if metric is None:
                raise RuntimeError(f"missing tune metric {candidate['name']} {domain}")
            rows.append(metric)
        values = {
            field: float(np.mean([row[field] for row in rows]))
            for field in FIELDS
        }
        values["score"] = (values["HOTA"] + values["MOTA"]) / 2.0
        table[candidate["name"]] = values
    selected = max(table, key=lambda name: (
        table[name]["score"], table[name]["HOTA"], table[name]["MOTA"]
    ))
    log(f"SELECTED candidate={selected} score={table[selected]['score']:.4f}")
    return candidate_by_name(selected), table


def candidate_by_name(name):
    return next(candidate for candidate in CANDIDATES if candidate["name"] == name)


def run_holdout_variants(selected):
    jobs = []
    for domain in DOMAINS:
        jobs.append((domain, "tuned", selected, False))
        jobs.append((domain, "frozen", FROZEN_CANDIDATE, False))
        jobs.append((domain, "geometry_null", FROZEN_CANDIDATE, False))

    def worker(item):
        index, (domain, variant, candidate, _) = item
        if variant == "geometry_null":
            output = variant_output(variant, domain, full=False)
            cache = cache_path(domain, "holdout", geometry_null=True)
            key = "holdout_geometry_null"
            method = "MOTSynth_VJEPA21_GEOMETRY_NULL"
        else:
            output = candidate_output("holdout_" + variant, domain, candidate)
            cache = cache_path(domain, "holdout")
            key = "holdout_" + variant
            method = "MOTSynth_VJEPA21_" + variant.upper()
        return run_tracker_variant(
            domain, split_dir(domain, "holdout"), output, candidate, cache, method,
            "target-video-split-holdout-mapped", GPU_IDS[index % len(GPU_IDS)], key,
        )

    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as pool:
        results = list(pool.map(worker, enumerate(jobs)))
    log(f"HOLDOUT_VARIANTS_FINISHED ok={sum(results)}/{len(results)}")
    return all(results)


def run_full_variants(selected):
    jobs = []
    for domain in DOMAINS:
        jobs.append((domain, "tuned", selected))
        jobs.append((domain, "frozen", FROZEN_CANDIDATE))
        jobs.append((domain, "geometry_null", FROZEN_CANDIDATE))

    def worker(item):
        index, (domain, variant, candidate) = item
        if variant == "geometry_null":
            output = variant_output(variant, domain, full=True)
            cache = cache_path(domain, "full", geometry_null=True)
            key = "full_geometry_null"
            method = "MOTSynth_VJEPA21_GEOMETRY_NULL"
        else:
            output = candidate_output("full_" + variant, domain, candidate)
            cache = cache_path(domain, "full")
            key = "full_" + variant
            method = "MOTSynth_VJEPA21_" + variant.upper()
        return run_tracker_variant(
            domain, split_dir(domain, "full"), output, candidate, cache, method,
            "mapped-full-correct", GPU_IDS[index % len(GPU_IDS)], key,
        )

    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as pool:
        results = list(pool.map(worker, enumerate(jobs)))
    log(f"FULL_VARIANTS_FINISHED ok={sum(results)}/{len(results)}")
    return all(results)


def run_baseline_holdout():
    def worker(item):
        index, domain = item
        output = HOLDOUT_BASELINE_ROOT / domain
        metric = output / "eval/metrics.json"
        if metric_done(metric):
            log(f"SKIP holdout_baseline {domain}")
            return True
        output.mkdir(parents=True, exist_ok=True)
        command = eval_command(
            domain, split_dir(domain, "holdout"),
            BASELINE_ROOT / domain / "track_results", output,
            "GeneralTrack", "target-video-split-holdout-control-mapped",
        )
        log_path = LOG_ROOT / f"holdout_baseline_{domain}.log"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        with log_path.open("a") as handle:
            handle.write("EVAL_CMD=" + " ".join(command) + "\n")
            rc = subprocess.run(command, cwd=str(REPO), env=env,
                                stdout=handle, stderr=subprocess.STDOUT).returncode
            handle.write(f"EVAL_EXIT_CODE={rc}\n")
        ok = rc == 0 and metric_done(metric)
        log(f"{'DONE' if ok else 'FAILED'} holdout_baseline {domain} rc={rc}")
        return ok

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, enumerate(DOMAINS)))
    log(f"HOLDOUT_BASELINE_FINISHED ok={sum(results)}/{len(results)}")
    return all(results)


def read_variant_metric(root, domain, direct=False):
    path = root / domain / ("metrics.json" if direct else "eval/metrics.json")
    return load_summary(path)


def macro(records):
    return {
        field: float(np.mean([row[field] for row in records]))
        for field in FIELDS
    }


def fmt(value):
    return str(int(value)) if isinstance(value, (int, np.integer)) else f"{float(value):.4f}"


def write_report(selected, tune_table):
    full_records = []
    holdout_records = []
    for domain in DOMAINS:
        full_records.append({
            "domain": domain,
            "baseline": read_variant_metric(BASELINE_ROOT, domain, direct=True),
            "geometry_null": read_variant_metric(FULL_NULL_ROOT, domain),
            "frozen": read_variant_metric(FULL_FROZEN_ROOT, domain),
            "tuned": read_variant_metric(FULL_TUNED_ROOT, domain),
        })
        holdout_records.append({
            "domain": domain,
            "baseline": read_variant_metric(HOLDOUT_BASELINE_ROOT, domain),
            "geometry_null": read_variant_metric(HOLDOUT_NULL_ROOT, domain),
            "frozen": read_variant_metric(HOLDOUT_FROZEN_ROOT, domain),
            "tuned": read_variant_metric(HOLDOUT_TUNED_ROOT, domain),
        })
    for records in (full_records, holdout_records):
        for record in records:
            if any(record[label] is None for label in
                   ("baseline", "geometry_null", "frozen", "tuned")):
                raise RuntimeError("missing metric while writing report")

    lines = [
        "# MOTSynth mapped source_detector_thr02 + V-JEPA 2.1 正确协议调参审计报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 协议修正",
        "",
        "本报告只使用 `generaltrack_prepared_mapped/source_detector_thr02`。每个 MOTSynth 视频只保留 detector 实际覆盖的帧（通常 450 帧/视频）；snow 为 9 个视频、4050 帧。之前 1798 帧/视频的 dense preparation 结果不纳入本报告。",
        "Baseline 复用已核实的 `outputs/generaltrack_mapped/source_detector_thr02/motsynth`，其五域宏平均 HOTA 为 16.7432。",
        "V-JEPA 缓存直接从本报告的 mapped annotation 重提取；每个 image 的 offsets 已逐帧核对为与 detector 行数完全一致。缓存对 detector score > 0.35 的行计算官方 predictor 表示，低于该值的行保留零向量但不删除，从而不改变 GeneralTrack/ByteTrack 的检测输入协议。",
        "候选只搜索 JEPA `high-min-sim` 与预测框 IoU；detector、GeneralTrack checkpoint、生命周期和所有非 JEPA 阈值固定。候选选择只看 tune videos，holdout 不参与选择。",
        "",
        "## 候选与选择",
        "",
        "| candidate | high-min-sim | predicted-box IoU | tune HOTA | tune MOTA | joint score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        row = tune_table[candidate["name"]]
        mark = " **selected**" if candidate["name"] == selected["name"] else ""
        lines.append(
            f"| {candidate['name']}{mark} | {candidate['sim']:.2f} | "
            f"{candidate['iou']:.2f} | {row['HOTA']:.4f} | {row['MOTA']:.4f} | {row['score']:.4f} |"
        )

    def add_macro(title, records):
        lines.extend([
            "", f"## {title}", "",
            "宏平均为五个 MOTSynth 域等权平均；IDSW/FP/FN 为域平均计数。",
            "",
            "| method | HOTA | DetA | AssA | MOTA | IDF1 | avg IDSW | avg FP | avg FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label, display in (("baseline", "GeneralTrack baseline"),
                               ("geometry_null", "Geometry-null"),
                               ("frozen", "Frozen V-JEPA21"),
                               ("tuned", "Tuned V-JEPA21")):
            values = macro([record[label] for record in records])
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                display, *(fmt(values[field]) for field in FIELDS)))
        tuned = macro([record["tuned"] for record in records])
        null = macro([record["geometry_null"] for record in records])
        baseline = macro([record["baseline"] for record in records])
        frozen = macro([record["frozen"] for record in records])
        lines.extend([
            "", "JEPA-specific 与 baseline 对比：",
            f"- tuned − geometry-null：HOTA {tuned['HOTA'] - null['HOTA']:+.4f}，MOTA {tuned['MOTA'] - null['MOTA']:+.4f}",
            f"- tuned − baseline：HOTA {tuned['HOTA'] - baseline['HOTA']:+.4f}，MOTA {tuned['MOTA'] - baseline['MOTA']:+.4f}",
            f"- frozen − baseline：HOTA {frozen['HOTA'] - baseline['HOTA']:+.4f}，MOTA {frozen['MOTA'] - baseline['MOTA']:+.4f}",
        ])

    add_macro("Full mapped-domain 结果", full_records)
    add_macro("Untouched holdout 结果", holdout_records)

    lines.extend(["", "## Full mapped-domain 逐域结果", "",
                  "| domain | method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for record in full_records:
        for label, display in (("baseline", "Baseline"), ("geometry_null", "Geometry-null"),
                               ("frozen", "Frozen V-JEPA21"), ("tuned", "Tuned V-JEPA21")):
            values = record[label]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                record["domain"], display, *(fmt(values[field]) for field in FIELDS)))

    lines.extend(["", "## Holdout 逐域 HOTA/MOTA", "",
                  "| domain | baseline | geometry-null | frozen JEPA | tuned JEPA | tuned-null |",
                  "|---|---:|---:|---:|---:|---:|"])
    for record in holdout_records:
        b, n, f, t = (record["baseline"], record["geometry_null"],
                      record["frozen"], record["tuned"])
        lines.append(
            f"| {record['domain']} | {b['HOTA']:.4f}/{b['MOTA']:.4f} | "
            f"{n['HOTA']:.4f}/{n['MOTA']:.4f} | {f['HOTA']:.4f}/{f['MOTA']:.4f} | "
            f"{t['HOTA']:.4f}/{t['MOTA']:.4f} | "
            f"{t['HOTA'] - n['HOTA']:+.4f}/{t['MOTA'] - n['MOTA']:+.4f} |"
        )

    lines.extend([
        "", "## 路径与边界", "",
        f"- mapped prepared: `{FULL_PREPARED_ROOT}`",
        f"- mapped V-JEPA cache: `{CACHE_ROOT}`",
        f"- tuning/holdout/full outputs: `{RUN_ROOT}`",
        "- 本轮 MOTSynth 结果不与之前 dense 1798-frame preparation 的结果混合。",
    ])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    SUMMARY_JSON.write_text(json.dumps({
        "protocol": "mapped_source_detector_thr02",
        "selection": selected,
        "tuning": tune_table,
        "full_records": full_records,
        "holdout_records": holdout_records,
        "full_macro": {label: macro([r[label] for r in full_records])
                       for label in ("baseline", "geometry_null", "frozen", "tuned")},
        "holdout_macro": {label: macro([r[label] for r in holdout_records])
                          for label in ("baseline", "geometry_null", "frozen", "tuned")},
    }, indent=2, default=float))
    log(f"REPORT_WRITTEN {REPORT}")


def main():
    log("MOTSYNTH_MAPPED_COORDINATOR_START")
    ensure_null_split_caches()
    if not run_tune_queue():
        log("MOTSYNTH_STOPPED_AFTER_TUNE_FAILURE")
        return 2
    selected, tune_table = select_candidate()
    (RUN_ROOT / "selection.json").write_text(json.dumps({
        "selected": selected, "table": tune_table
    }, indent=2))
    if not run_holdout_variants(selected):
        log("MOTSYNTH_STOPPED_AFTER_HOLDOUT_FAILURE")
        return 3
    if not run_full_variants(selected):
        log("MOTSYNTH_STOPPED_AFTER_FULL_FAILURE")
        return 4
    if not run_baseline_holdout():
        log("MOTSYNTH_STOPPED_AFTER_BASELINE_CONTROL_FAILURE")
        return 5
    write_report(selected, tune_table)
    log("MOTSYNTH_MAPPED_COORDINATOR_FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
