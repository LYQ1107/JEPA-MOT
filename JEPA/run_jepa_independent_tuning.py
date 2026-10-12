#!/usr/bin/env python3
"""Bounded JEPA-only tuning on disjoint video splits for three datasets.

The frozen full-domain experiment is never overwritten.  Candidate parameters
are selected on the deterministic tuning videos, then evaluated on untouched
holdout videos together with baseline, frozen V-JEPA, and geometry-null
controls.
"""

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
SPLIT_ROOT = JEPA_ROOT / "outputs/jepa_tuning_splits"
FROZEN_ROOT = REPO / "outputs/official_vjepa21_other_datasets_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast"
NULL_ROOT = REPO / "outputs/official_vjepa21_other_datasets_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_geometry_null_iou020"
BASELINE_ROOT = REPO / "outputs/generaltrack_thr02"
TUNE_ROOT = REPO / "outputs/jepa_independent_tuning_vjepa21"
HOLDOUT_ROOT = TUNE_ROOT / "holdout_tuned"
CONTROL_ROOT = TUNE_ROOT / "holdout_controls"
LOG_ROOT = Path("/tmp/jepa_independent_tuning_logs")
STATE_LOG = LOG_ROOT / "coordinator.log"
REPORT = JEPA_ROOT / "reports/JEPA_independent_tuning_audit.md"
SUMMARY_JSON = JEPA_ROOT / "reports/JEPA_independent_tuning_summary.json"

GPU_IDS = (5, 6)
DOMAINS = {
    "bdd100k": ("dawn_dusk", "night", "overcast", "rainy"),
    "vkitti2": ("fog", "morning", "overcast", "rain", "sunset"),
    "motsynth": ("cloudy", "dark", "fog", "rain", "snow"),
}

# JEPA-only candidate grid.  Detector, GeneralTrack relation model, lifecycle,
# geometry duplicate gate, history policy, and all non-JEPA thresholds remain
# fixed.  The current frozen setting is s090_iou005.
CANDIDATES = (
    {"name": "s085_iou000", "sim": 0.85, "iou": 0.00},
    {"name": "s085_iou005", "sim": 0.85, "iou": 0.05},
    {"name": "s090_iou000", "sim": 0.90, "iou": 0.00},
    {"name": "s090_iou005", "sim": 0.90, "iou": 0.05},
    {"name": "s095_iou000", "sim": 0.95, "iou": 0.00},
    {"name": "s095_iou005", "sim": 0.95, "iou": 0.05},
)
FIELDS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN")


def log(message):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = "{} {}\n".format(datetime.now().astimezone().isoformat(timespec="seconds"), message)
    with STATE_LOG.open("a") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def split_dir(dataset, domain, split):
    return SPLIT_ROOT / dataset / domain / split


def load_summary(path):
    try:
        data = json.loads(path.read_text())
        summary = data.get("summary", data)
        if all(field in summary for field in FIELDS):
            return summary
    except (OSError, ValueError, TypeError):
        pass
    return None


def metric_done(path):
    return load_summary(path) is not None


def tune_output(dataset, candidate, domain):
    return TUNE_ROOT / "tune" / dataset / candidate / domain


def candidate_by_name(name):
    return next(c for c in CANDIDATES if c["name"] == name)


def tracker_command(dataset, domain, candidate, split, output):
    split_path = split_dir(dataset, domain, split)
    return [
        PYTHON, "-u", str(REPO / "tools/track_damot_domain.py"),
        "--dataset", dataset, "--domain", domain,
        "--prepared-dir", str(split_path), "--output-dir", str(output),
        "--track_thresh", "0.35", "--det_thresh", "0.45", "--track_buffer", "50",
        "--match_thresh", "0.90", "--matchlost_thresh", "0.90",
        "--restore_ckpt", str(REPO / "checkpoints/generaltrack_bdd.pth"),
        "--resize", "720", "1280",
        "--jepa-cache", str(split_path / "vjepa21_features.npz"),
        "--jepa-feature-dim", "1664",
        "--jepa-mode", "high_fallback_duplicate_geom_jepa",
        "--jepa-high-min-sim", str(candidate["sim"]),
        "--jepa-high-iou", str(candidate["iou"]),
        "--jepa-relative-margin", "0",
        "--jepa-max-gap", "8",
        "--jepa-history-aggregation", "last",
        "--jepa-duplicate-min-sim", "0.90",
        "--jepa-duplicate-iou", "0.20",
    ]


def eval_command(dataset, domain, split, track_results, output, method, protocol):
    return [
        PYTHON, "-u", str(REPO / "tools/eval_damot_generaltrack.py"),
        "--dataset", dataset, "--domain", domain,
        "--prepared-dir", str(split_dir(dataset, domain, split)),
        "--track-results", str(track_results),
        "--output-dir", str(output / "eval"),
        "--summary-csv", str(output / "eval/summary.csv"),
        "--method", method, "--protocol", protocol,
    ]


def run_candidate(dataset, domain, candidate, split, gpu):
    output = (tune_output(dataset, candidate["name"], domain)
              if split == "tune" else HOLDOUT_ROOT / dataset / candidate["name"] / domain)
    metric = output / "eval/metrics.json"
    if metric_done(metric):
        log(f"SKIP {split} {dataset}/{domain}/{candidate['name']}")
        return True
    output.mkdir(parents=True, exist_ok=True)
    track_cmd = tracker_command(dataset, domain, candidate, split, output)
    eval_cmd = eval_command(dataset, domain, split, output / "track_results", output,
                            "VJEPA21_TUNED_" + candidate["name"],
                            "target-video-split-tuning" if split == "tune" else "target-video-split-holdout")
    log_path = LOG_ROOT / f"{split}_{dataset}_{domain}_{candidate['name']}.log"
    for attempt in (1, 2):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a") as handle:
            handle.write("\n=== attempt={} gpu={} started={} ===\n".format(
                attempt, gpu, datetime.now().astimezone().isoformat(timespec="seconds")))
            handle.write("TRACK_CMD={}\n".format(" ".join(track_cmd)))
            handle.flush()
            track_rc = subprocess.run(track_cmd, cwd=str(REPO), env=env,
                                      stdout=handle, stderr=subprocess.STDOUT).returncode
            handle.write(f"TRACK_EXIT_CODE={track_rc}\n")
            handle.flush()
            if track_rc == 0:
                handle.write("EVAL_CMD={}\n".format(" ".join(eval_cmd)))
                handle.flush()
                eval_rc = subprocess.run(eval_cmd, cwd=str(REPO), env=env,
                                         stdout=handle, stderr=subprocess.STDOUT).returncode
            else:
                eval_rc = -1
            handle.write(f"EVAL_EXIT_CODE={eval_rc}\n")
            handle.flush()
        if track_rc == 0 and eval_rc == 0 and metric_done(metric):
            log(f"DONE {split} {dataset}/{domain}/{candidate['name']} gpu={gpu}")
            return True
        log(f"RETRY {split} {dataset}/{domain}/{candidate['name']} attempt={attempt} "
            f"track_rc={track_rc} eval_rc={eval_rc}")
        time.sleep(10)
    log(f"FAILED {split} {dataset}/{domain}/{candidate['name']}")
    return False


def run_tune_queue():
    jobs = [
        (dataset, domain, candidate)
        for dataset, domains in DOMAINS.items()
        for domain in domains
        for candidate in CANDIDATES
    ]
    log(f"TUNE_QUEUE_START jobs={len(jobs)} gpus={GPU_IDS}")

    def worker(item):
        index, (dataset, domain, candidate) = item
        return run_candidate(dataset, domain, candidate, "tune", GPU_IDS[index % len(GPU_IDS)])

    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as pool:
        results = list(pool.map(worker, enumerate(jobs)))
    log(f"TUNE_QUEUE_FINISHED ok={sum(results)}/{len(results)}")
    return all(results)


def wait_for_splits():
    expected = sum(len(domains) for domains in DOMAINS.values()) * 2
    while True:
        ready = sum(1 for dataset, domains in DOMAINS.items() for domain in domains
                    for split in ("tune", "holdout")
                    if (split_dir(dataset, domain, split) / "vjepa21_features.npz").is_file())
        log(f"SPLITS_READY {ready}/{expected}")
        if ready == expected:
            return
        time.sleep(30)


def select_candidates():
    selection = {}
    table = {}
    for dataset, domains in DOMAINS.items():
        table[dataset] = {}
        for candidate in CANDIDATES:
            rows = []
            for domain in domains:
                metric = load_summary(tune_output(dataset, candidate["name"], domain) / "eval/metrics.json")
                if metric is None:
                    raise RuntimeError(f"missing tuning metric {dataset}/{domain}/{candidate['name']}")
                rows.append(metric)
            values = {field: float(np.mean([row[field] for row in rows])) for field in FIELDS}
            values["score"] = (values["HOTA"] + values["MOTA"]) / 2.0
            table[dataset][candidate["name"]] = values
        selected = max(table[dataset], key=lambda name: (
            table[dataset][name]["score"],
            table[dataset][name]["HOTA"],
            table[dataset][name]["MOTA"],
        ))
        selection[dataset] = selected
        log(f"SELECTED {dataset} candidate={selected} score={table[dataset][selected]['score']:.4f}")
    return selection, table


def eval_existing_control(label, dataset, domain, split, track_root):
    output = CONTROL_ROOT / label / dataset / domain
    metric = output / "eval/metrics.json"
    if metric_done(metric):
        return True
    output.mkdir(parents=True, exist_ok=True)
    command = eval_command(
        dataset, domain, split, track_root / dataset / domain / "track_results", output,
        "GeneralTrack" if label == "baseline" else ("VJEPA21" if label == "frozen" else "VJEPA21_GEOMETRY_NULL"),
        "target-video-split-holdout-control",
    )
    log_path = LOG_ROOT / f"holdout_control_{label}_{dataset}_{domain}.log"
    with log_path.open("a") as handle:
        handle.write("EVAL_CMD={}\n".format(" ".join(command)))
        rc = subprocess.run(command, cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT).returncode
        handle.write(f"EVAL_EXIT_CODE={rc}\n")
    if rc != 0 or not metric_done(metric):
        log(f"FAILED holdout control {label} {dataset}/{domain} rc={rc}")
        return False
    log(f"DONE holdout control {label} {dataset}/{domain}")
    return True


def run_holdout(selection):
    tuned_jobs = []
    for dataset, domains in DOMAINS.items():
        candidate = candidate_by_name(selection[dataset])
        tuned_jobs.extend((dataset, domain, candidate) for domain in domains)

    def worker(item):
        index, (dataset, domain, candidate) = item
        return run_candidate(dataset, domain, candidate, "holdout", GPU_IDS[index % len(GPU_IDS)])

    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as pool:
        tuned_results = list(pool.map(worker, enumerate(tuned_jobs)))

    controls = []
    for dataset, domains in DOMAINS.items():
        for domain in domains:
            controls.append(("baseline", dataset, domain, BASELINE_ROOT))
            controls.append(("frozen", dataset, domain, FROZEN_ROOT))
            controls.append(("geometry_null", dataset, domain, NULL_ROOT))

    def control_worker(item):
        label, dataset, domain, track_root = item
        return eval_existing_control(label, dataset, domain, "holdout", track_root)

    with ThreadPoolExecutor(max_workers=4) as pool:
        control_results = list(pool.map(control_worker, controls))
    return all(tuned_results) and all(control_results)


def read_holdout_metric(label, dataset, domain, candidate=None):
    if label == "tuned":
        path = HOLDOUT_ROOT / dataset / candidate / domain / "eval/metrics.json"
    else:
        path = CONTROL_ROOT / label / dataset / domain / "eval/metrics.json"
    return load_summary(path)


def fmt(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def write_report(selection, tune_table):
    records = []
    for dataset, domains in DOMAINS.items():
        for domain in domains:
            candidate = selection[dataset]
            records.append({
                "dataset": dataset,
                "domain": domain,
                "selected_candidate": candidate,
                "baseline": read_holdout_metric("baseline", dataset, domain),
                "frozen": read_holdout_metric("frozen", dataset, domain),
                "geometry_null": read_holdout_metric("geometry_null", dataset, domain),
                "tuned": read_holdout_metric("tuned", dataset, domain, candidate),
            })
    if any(any(record[label] is None for label in ("baseline", "frozen", "geometry_null", "tuned"))
           for record in records):
        raise RuntimeError("missing holdout metric while writing tuning report")

    lines = [
        "# JEPA 独立调参实验审计报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 实验协议",
        "",
        "冻结配置的 full-domain 结果已先封存；本实验只在 BDD100K、VKITTI2、MOTSynth 上调 JEPA 参数。每个域按视频确定性切分为约 20% tuning videos 和 80% holdout videos。",
        "候选选择只使用 tuning videos 的 HOTA/MOTA 等权平均；最终 tuned 结果只在未参与选参的 holdout videos 上报告。",
        "",
        "仅搜索 JEPA fallback 的 `high-min-sim` 和 `predicted-box IoU`：",
        "",
        "| candidate | high-min-sim | predicted-box IoU |",
        "|---|---:|---:|",
    ]
    for candidate in CANDIDATES:
        lines.append(f"| {candidate['name']} | {candidate['sim']:.2f} | {candidate['iou']:.2f} |")
    lines += [
        "",
        "Detector rows、GeneralTrack checkpoint、track/det threshold、native lifecycle、history=last、geometry duplicate IoU=0.20 和评测脚本均固定；没有调 detector 或 tracker 阈值。",
        "",
        "## 每个数据集的 tuning 选择",
        "",
        "| dataset | candidate | tuning HOTA | tuning MOTA | joint score | frozen candidate score |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DOMAINS:
        selected = selection[dataset]
        frozen = tune_table[dataset]["s090_iou005"]
        chosen = tune_table[dataset][selected]
        lines.append("| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            dataset, selected, chosen["HOTA"], chosen["MOTA"], chosen["score"], frozen["score"]))

    lines += [
        "",
        "## Holdout 宏平均",
        "",
        "宏平均为同一数据集 holdout domains 等权平均；JEPA-specific 为 tuned - geometry-null。",
        "",
        "| scope | method | HOTA | DetA | AssA | MOTA | IDF1 | avg IDSW | avg FP | avg FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    scopes = [(dataset, [r for r in records if r["dataset"] == dataset]) for dataset in DOMAINS]
    scopes.append(("ALL", records))
    macro = {}
    for scope, scope_records in scopes:
        macro[scope] = {}
        for label, display in (("baseline", "Baseline"), ("frozen", "Frozen V-JEPA21"),
                               ("geometry_null", "Geometry-null"), ("tuned", "Tuned V-JEPA21")):
            rows = [record[label] for record in scope_records]
            values = {field: float(np.mean([row[field] for row in rows])) for field in FIELDS}
            macro[scope][label] = values
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                scope, display, *(fmt(values[field]) for field in FIELDS)))
    lines += [
        "",
        "## 调参后的 JEPA-specific 增益",
        "",
        "| scope | tuned − geometry-null HOTA | tuned − geometry-null MOTA | tuned − frozen HOTA | tuned − frozen MOTA |",
        "|---|---:|---:|---:|---:|"]
    for scope, _ in scopes:
        tuned = macro[scope]["tuned"]
        null = macro[scope]["geometry_null"]
        frozen = macro[scope]["frozen"]
        lines.append("| {} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} |".format(
            scope, tuned["HOTA"] - null["HOTA"], tuned["MOTA"] - null["MOTA"],
            tuned["HOTA"] - frozen["HOTA"], tuned["MOTA"] - frozen["MOTA"]))

    lines += [
        "",
        "## Holdout 逐域结果",
        "",
        "| dataset | domain | selected | frozen HOTA/MOTA | tuned HOTA/MOTA | tuned-null HOTA/MOTA |",
        "|---|---|---|---:|---:|---:|"]
    for record in records:
        frozen, tuned, null = record["frozen"], record["tuned"], record["geometry_null"]
        lines.append("| {} | {} | {} | {:.4f}/{:.4f} | {:.4f}/{:.4f} | {:+.4f}/{:+.4f} |".format(
            record["dataset"], record["domain"], record["selected_candidate"],
            frozen["HOTA"], frozen["MOTA"], tuned["HOTA"], tuned["MOTA"],
            tuned["HOTA"] - null["HOTA"], tuned["MOTA"] - null["MOTA"]))

    lines += [
        "",
        "## 输出与解释边界",
        "",
        f"- tuning outputs: `{TUNE_ROOT}`",
        f"- holdout tuned outputs: `{HOLDOUT_ROOT}`",
        f"- holdout controls: `{CONTROL_ROOT}`",
        "- 这是一组 target-domain video-split adaptation 结果，不应与冻结零样本主结果混写；最终 holdout 仍未参与候选选择。",
        "- 如果 tuned 相对 frozen 改善，只能说明在该数据集内部视频划分上调 JEPA 阈值有效；不能把它表述成无需调参的跨域泛化。",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    SUMMARY_JSON.write_text(json.dumps({"selection": selection, "tuning": tune_table,
                                        "records": records, "macro": macro}, indent=2, default=float))
    log(f"REPORT_WRITTEN {REPORT}")


def main():
    log("TUNING_COORDINATOR_START")
    wait_for_splits()
    if not run_tune_queue():
        log("TUNING_STOPPED_AFTER_TUNE_FAILURE")
        return 2
    selection, table = select_candidates()
    (TUNE_ROOT / "selection.json").parent.mkdir(parents=True, exist_ok=True)
    (TUNE_ROOT / "selection.json").write_text(json.dumps({"selection": selection, "table": table}, indent=2))
    if not run_holdout(selection):
        log("TUNING_STOPPED_AFTER_HOLDOUT_FAILURE")
        return 3
    write_report(selection, table)
    log("TUNING_COORDINATOR_FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
