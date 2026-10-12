#!/usr/bin/env python3
"""Persistent single-coordinator runner for the remaining V-JEPA21 evaluations.

This is intentionally one local coordinator, not a sub-agent.  It resumes
completed jobs, retries interrupted jobs, and writes the final audit report
after all final/null evaluations are complete.
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
PREPARED_ROOT = REPO / "datasets" / "generaltrack_prepared_thr02"
CACHE_ROOT = JEPA_ROOT / "outputs" / "official_vjepa21_cache"
CHECKPOINT = JEPA_ROOT / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"
LOG_ROOT = Path("/tmp/vjepa_other_tracker_persistent_logs")
STATE_LOG = LOG_ROOT / "coordinator.log"
FINAL_ROOT = REPO / "outputs/official_vjepa21_other_datasets_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast"
NULL_ROOT = REPO / "outputs/official_vjepa21_other_datasets_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_geometry_null_iou020"
REPORT = JEPA_ROOT / "reports/ALL_DATASETS_VJEPA21_final_audit.md"
SUMMARY_JSON = JEPA_ROOT / "reports/ALL_DATASETS_VJEPA21_summary.json"

# Four GPUs, all within the user's limit.  These were free at recovery time.
GPU_IDS = (5, 6, 7, 8)

OTHER_DOMAINS = (
    ("bdd100k", "dawn_dusk"),
    ("bdd100k", "night"),
    ("bdd100k", "overcast"),
    ("bdd100k", "rainy"),
    ("vkitti2", "fog"),
    ("vkitti2", "morning"),
    ("vkitti2", "overcast"),
    ("vkitti2", "rain"),
    ("vkitti2", "sunset"),
    ("motsynth", "cloudy"),
    ("motsynth", "dark"),
    ("motsynth", "fog"),
    ("motsynth", "rain"),
    ("motsynth", "snow"),
)
UA_DOMAINS = (("ua_detrac", "cloudy"), ("ua_detrac", "night"), ("ua_detrac", "rainy"))
ALL_DOMAINS = UA_DOMAINS + OTHER_DOMAINS
METRIC_FIELDS = ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN")


def log(message):
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    line = "{} {}\n".format(datetime.now().astimezone().isoformat(timespec="seconds"), message)
    with STATE_LOG.open("a") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def cache_path(dataset, domain):
    return CACHE_ROOT / f"{dataset}_{domain}_vjepa21_l384_w16_causalbox_predictor.npz"


def null_cache_path(dataset, domain):
    return CACHE_ROOT / f"{dataset}_{domain}_vjepa21_l384_w16_causalbox_predictor_geometry_null.npz"


def output_root(kind):
    return FINAL_ROOT if kind == "final" else NULL_ROOT


def metric_path(dataset, domain, kind):
    return output_root(kind) / dataset / domain / "eval" / "metrics.json"


def load_summary(path):
    try:
        data = json.loads(path.read_text())
        summary = data.get("summary", data)
        if all(field in summary for field in METRIC_FIELDS):
            return summary
    except (OSError, ValueError, TypeError):
        pass
    return None


def metric_done(dataset, domain, kind):
    return load_summary(metric_path(dataset, domain, kind)) is not None


def cache_done(dataset, domain):
    sidecar = Path(str(cache_path(dataset, domain)) + ".json")
    try:
        info = json.loads(sidecar.read_text())
        shape = info.get("feature_shape", [None, None])
        return (
            info.get("status") == "complete"
            and info.get("feature_score_min") == 0.35
            and info.get("crop_backend") == "gpu_roi_align"
            and info.get("frame_batch_size") == 16
            and info.get("roi_mode") == "causal_box_crop_predictor"
            and len(shape) == 2
            and shape[1] == 1664
        )
    except (OSError, ValueError, TypeError):
        return False


def make_null_cache(dataset, domain):
    source = cache_path(dataset, domain)
    target = null_cache_path(dataset, domain)
    sidecar = Path(str(target) + ".json")
    if target.exists() and sidecar.exists():
        return
    with np.load(str(source), allow_pickle=False) as data:
        features = np.zeros_like(data["features"], dtype=np.float32)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        image_ids = np.asarray(data["image_ids"], dtype=np.int64)
    np.savez_compressed(str(target), features=features, offsets=offsets, image_ids=image_ids)
    source_info = json.loads(Path(str(source) + ".json").read_text())
    source_info.update({
        "status": "complete",
        "feature_source": "geometry_null_zero_features",
        "feature_shape": [int(x) for x in features.shape],
        "feature_rows_computed": 0,
        "feature_rows_zero_filled": int(features.shape[0]),
        "geometry_null_for": str(source),
    })
    sidecar.write_text(json.dumps(source_info, indent=2))


def commands(dataset, domain, kind):
    cache = cache_path(dataset, domain) if kind == "final" else null_cache_path(dataset, domain)
    output = output_root(kind) / dataset / domain
    prepared = PREPARED_ROOT / dataset / domain
    track_cmd = [
        PYTHON, "-u", str(REPO / "tools/track_damot_domain.py"),
        "--dataset", dataset, "--domain", domain,
        "--prepared-dir", str(prepared), "--output-dir", str(output),
        "--track_thresh", "0.35", "--det_thresh", "0.45", "--track_buffer", "50",
        "--match_thresh", "0.90", "--matchlost_thresh", "0.90",
        "--restore_ckpt", str(REPO / "checkpoints/generaltrack_bdd.pth"),
        "--resize", "720", "1280",
        "--jepa-cache", str(cache), "--jepa-feature-dim", "1664",
        "--jepa-mode", "high_fallback_duplicate_geom_jepa",
        "--jepa-high-min-sim", "0.90", "--jepa-high-iou", "0.05",
        "--jepa-relative-margin", "0", "--jepa-max-gap", "8",
        "--jepa-history-aggregation", "last", "--jepa-duplicate-min-sim", "0.90",
        "--jepa-duplicate-iou", "0.20",
    ]
    method = "VJEPA21" if kind == "final" else "VJEPA21_GEOMETRY_NULL"
    eval_cmd = [
        PYTHON, "-u", str(REPO / "tools/eval_damot_generaltrack.py"),
        "--dataset", dataset, "--domain", domain,
        "--prepared-dir", str(prepared), "--track-results", str(output / "track_results"),
        "--output-dir", str(output / "eval"), "--summary-csv", str(output / "eval/summary.csv"),
        "--method", method, "--protocol", "source-only-no-adaptation",
    ]
    return output, track_cmd, eval_cmd


def run_job(dataset, domain, kind, gpu):
    if metric_done(dataset, domain, kind):
        log(f"SKIP {dataset}/{domain}/{kind}: metrics already complete")
        return True
    output, track_cmd, eval_cmd = commands(dataset, domain, kind)
    output.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{dataset}_{domain}_{kind}.log"
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
            handle.write("TRACK_EXIT_CODE={}\n".format(track_rc))
            handle.flush()
            if track_rc == 0:
                handle.write("EVAL_CMD={}\n".format(" ".join(eval_cmd)))
                handle.flush()
                eval_rc = subprocess.run(eval_cmd, cwd=str(REPO), env=env,
                                         stdout=handle, stderr=subprocess.STDOUT).returncode
            else:
                eval_rc = -1
            handle.write("EVAL_EXIT_CODE={}\n".format(eval_rc))
            handle.flush()
        if track_rc == 0 and eval_rc == 0 and metric_done(dataset, domain, kind):
            log(f"DONE {dataset}/{domain}/{kind} gpu={gpu}")
            return True
        log(f"RETRY {dataset}/{domain}/{kind} gpu={gpu} attempt={attempt} track_rc={track_rc} eval_rc={eval_rc}")
        time.sleep(10)
    log(f"FAILED {dataset}/{domain}/{kind} gpu={gpu}; leaving evidence in {log_path}")
    return False


def wait_for_caches():
    while True:
        ready = sum(cache_done(*job) for job in OTHER_DOMAINS)
        log(f"CACHE_READY {ready}/{len(OTHER_DOMAINS)}")
        if ready == len(OTHER_DOMAINS):
            return
        time.sleep(60)


def run_all_jobs():
    wait_for_caches()
    for dataset, domain in OTHER_DOMAINS:
        make_null_cache(dataset, domain)
    jobs = [(dataset, domain, kind)
            for dataset, domain in OTHER_DOMAINS
            for kind in ("final", "null")
            if not metric_done(dataset, domain, kind)]
    log(f"TRACK_QUEUE_START pending={len(jobs)} gpus={GPU_IDS}")

    def worker(item):
        dataset, domain, kind, gpu = item
        return run_job(dataset, domain, kind, gpu)

    # The iterator is consumed lazily so no second agent/process is created.
    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as pool:
        items = ((dataset, domain, kind, GPU_IDS[index % len(GPU_IDS)])
                 for index, (dataset, domain, kind) in enumerate(jobs))
        results = list(pool.map(worker, items))
    log(f"TRACK_QUEUE_FINISHED ok={sum(results)}/{len(results)}")
    return all(results)


def read_metric(dataset, domain, kind):
    if dataset == "ua_detrac":
        if kind == "baseline":
            path = REPO / "outputs/generaltrack_thr02/ua_detrac" / domain / "metrics.json"
        elif kind == "final":
            path = REPO / "outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_jepa_margin000_histlast" / domain / "eval/metrics.json"
        else:
            path = REPO / "outputs/official_vjepa21_l384_w16_causalbox_predictor_high_fallback_duplicate_geom_iou020" / domain / "eval/metrics.json"
    else:
        if kind == "baseline":
            path = REPO / "outputs/generaltrack_thr02" / dataset / domain / "metrics.json"
        elif kind == "final":
            path = FINAL_ROOT / dataset / domain / "eval/metrics.json"
        else:
            path = NULL_ROOT / dataset / domain / "eval/metrics.json"
    return load_summary(path)


def fmt(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.4f}"


def mean_metric(rows, field):
    return float(np.mean([float(row[field]) for row in rows]))


def write_report():
    records = []
    for dataset, domain in ALL_DOMAINS:
        baseline = read_metric(dataset, domain, "baseline")
        null = read_metric(dataset, domain, "null")
        final = read_metric(dataset, domain, "final")
        if not (baseline and null and final):
            raise RuntimeError(f"missing report metrics for {dataset}/{domain}")
        records.append({"dataset": dataset, "domain": domain,
                        "baseline": baseline, "geometry_null": null, "VJEPA21": final})

    report_lines = [
        "# V-JEPA 2.1 Online GeneralTrack 多数据集完成报告",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 实验结论",
        "",
        "本报告汇总 UA-DETRAC、BDD100K、VKITTI2 和 MOTSynth 的固定 GeneralTrack/TBD 在线评测。",
        "所有最终方法与 geometry-only null 使用相同 detector rows、checkpoint、tracker lifecycle、阈值和评测协议；null 只将 V-JEPA 特征置零，因此 `VJEPA21 - geometry_null` 是本实验中 JEPA-specific 的配对增益估计。",
        "",
        "## 固定配置",
        "",
        "- V-JEPA 2.1 官方 ViT-L/384 encoder + 官方 predictor，16 帧 causal box crop，context=1.5，最后层预测 token mean-pool，1664-D L2-normalized feature。",
        "- 在线接入 GeneralTrack 原生 relation association；V-JEPA 只处理未匹配高置信轨迹/检测候选，并保留 geometry-only duplicate birth gate。",
        "- detector rows、score threshold、track lifecycle、匹配阈值和 checkpoint 均未因数据集改变。",
        "- 缓存只使用 detector 输入和过去帧，未读取 GT；GT 只用于最终评测。",
        "",
        "## 逐域完整指标",
        "",
        "数值均为 TrackEval evaluator 输出；HOTA、DetA、AssA、MOTA、IDF1 为百分点，IDSW/FP/FN 为计数。",
        "",
        "| dataset | domain | method | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FP | FN |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        for kind in ("baseline", "geometry_null", "VJEPA21"):
            row = record[kind]
            label = {"baseline": "GeneralTrack", "geometry_null": "Geometry-null", "VJEPA21": "V-JEPA21"}[kind]
            report_lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                record["dataset"], record["domain"], label,
                *(fmt(row[field]) for field in METRIC_FIELDS)))

    report_lines += ["", "## 逐域增益审计", "",
                     "| dataset | domain | ΔHOTA final-baseline | ΔMOTA final-baseline | JEPA ΔHOTA final-null | JEPA ΔMOTA final-null |",
                     "|---|---|---:|---:|---:|---:|"]
    for record in records:
        b, n, f = record["baseline"], record["geometry_null"], record["VJEPA21"]
        report_lines.append("| {} | {} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} |".format(
            record["dataset"], record["domain"],
            f["HOTA"] - b["HOTA"], f["MOTA"] - b["MOTA"],
            f["HOTA"] - n["HOTA"], f["MOTA"] - n["MOTA"]))

    report_lines += ["", "## 数据集宏平均", "",
                     "宏平均为同一数据集各域等权平均；`ALL` 为 17 个域等权平均，不按帧数或检测数加权。",
                     "",
                     "| scope | method | HOTA | DetA | AssA | MOTA | IDF1 | avg IDSW | avg FP | avg FN |",
                     "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    scopes = [("ua_detrac", [r for r in records if r["dataset"] == "ua_detrac"]),
              ("bdd100k", [r for r in records if r["dataset"] == "bdd100k"]),
              ("vkitti2", [r for r in records if r["dataset"] == "vkitti2"]),
              ("motsynth", [r for r in records if r["dataset"] == "motsynth"]),
              ("ALL", records)]
    labels = {"baseline": "GeneralTrack", "geometry_null": "Geometry-null", "VJEPA21": "V-JEPA21"}
    macro = {}
    for scope, scope_records in scopes:
        macro[scope] = {}
        for kind in ("baseline", "geometry_null", "VJEPA21"):
            rows = [r[kind] for r in scope_records]
            values = {field: mean_metric(rows, field) for field in METRIC_FIELDS}
            macro[scope][kind] = values
            report_lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                scope, labels[kind], *(fmt(values[field]) for field in METRIC_FIELDS)))

    report_lines += ["", "## JEPA-specific 宏平均增益", "",
                     "| scope | JEPA ΔHOTA (V-JEPA21 - null) | JEPA ΔMOTA (V-JEPA21 - null) | total ΔHOTA vs baseline | total ΔMOTA vs baseline |",
                     "|---|---:|---:|---:|---:|"]
    for scope, _ in scopes:
        n = macro[scope]["geometry_null"]
        f = macro[scope]["VJEPA21"]
        b = macro[scope]["baseline"]
        report_lines.append("| {} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} |".format(
            scope, f["HOTA"] - n["HOTA"], f["MOTA"] - n["MOTA"],
            f["HOTA"] - b["HOTA"], f["MOTA"] - b["MOTA"]))

    report_lines += ["", "## 结果文件", "",
                     f"- V-JEPA21 final outputs: `{FINAL_ROOT}`",
                     f"- Geometry-null outputs: `{NULL_ROOT}`",
                     f"- V-JEPA cache and sidecars: `{CACHE_ROOT}`",
                     f"- UA-DETRAC prior audit: `{JEPA_ROOT / 'reports/UA-DETRAC_VJEPA21_Online_TBD_ICLR_audit.md'}`",
                     "",
                     "## 审计边界",
                     "",
                     "geometry-null 只能隔离本次 paired protocol 中的 JEPA 特征增益，不能证明所有总体增益都来自 JEPA；若作为 ICLR 正式论文，还应补充多 seed、更多公开 tracker baseline 和统计显著性。",
                     ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report_lines))
    SUMMARY_JSON.write_text(json.dumps({"records": records, "macro": macro}, indent=2, default=float))
    log(f"REPORT_WRITTEN {REPORT}")


def main():
    log("COORDINATOR_START persistent_single_agent")
    success = run_all_jobs()
    if not success:
        log("COORDINATOR_STOPPED_WITH_FAILURE; no report generated")
        return 2
    write_report()
    log("COORDINATOR_FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
