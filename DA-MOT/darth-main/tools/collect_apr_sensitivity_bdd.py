#!/usr/bin/env python3
"""Collect BDD APR sensitivity metrics from complete forgetting matrices."""

import csv
import re
from pathlib import Path


ROOT = Path("work_dirs/bdd_4domain/apr_sensitivity")
DEFAULT_ROOT = Path(
    "work_dirs/bdd_4domain/tri_module_pca_variants/"
    "spaP1_causalProtoShrink_roiV1Weak_4domain_5e"
)
DOMAINS = ["overcast", "rainy", "dawn_dusk", "night"]
STAGES = [
    "stage01_after_overcast",
    "stage02_after_rainy",
    "stage03_after_dawn_dusk",
    "stage04_after_night",
]
CONFIGS = {
    "kp50_kn3": dict(kp=50, kn=3, root=ROOT / "kp50_kn3"),
    "kp100_kn3": dict(kp=100, kn=3, root=DEFAULT_ROOT),
    "kp200_kn3": dict(kp=200, kn=3, root=ROOT / "kp200_kn3"),
    "kp100_kn1": dict(kp=100, kn=1, root=ROOT / "kp100_kn1"),
    "kp100_kn5": dict(kp=100, kn=5, root=ROOT / "kp100_kn5"),
}
CM_HEADER = "IDF1 MOTA MOTP FP FN IDSw Rcll Prcn MT PT ML FM"
CM_COLUMNS = CM_HEADER.split()
HOTA_HEADER = "AssRe AssPr AssA LocA DetRe DetPr DetA HOTA Dets IDs"
HOTA_COLUMNS = HOTA_HEADER.split()


def parse_log(path):
    lines = path.read_text(errors="ignore").splitlines()
    result = {}
    for header, columns in ((CM_HEADER, CM_COLUMNS), (HOTA_HEADER, HOTA_COLUMNS)):
        for index, line in enumerate(lines):
            if re.sub(r"\s+", " ", line.strip()) != header:
                continue
            for candidate in lines[index + 1:index + 31]:
                if candidate.strip().upper().startswith("OVERALL"):
                    for column, value in zip(columns, candidate.split()[1:]):
                        result[column] = float(value)
                    break
            break
    return result


def metric_matrix(root, metric):
    values = {}
    for stage in STAGES:
        for domain in DOMAINS:
            log = root / "forgetting_eval" / stage / "epoch_5" / domain / "eval.log"
            if not log.is_file():
                raise FileNotFoundError(log)
            metrics = parse_log(log)
            if metric not in metrics:
                raise RuntimeError("{} missing from {}".format(metric, log))
            values[(domain, stage)] = metrics[metric]
    return values


def mean(values):
    return sum(values) / len(values)


def forgetting_metrics(values):
    final_stage = STAGES[-1]
    final = [values[(domain, final_stage)] for domain in DOMAINS]
    bwt = []
    forget_best = []
    for index, domain in enumerate(DOMAINS[:-1]):
        diagonal = values[(domain, STAGES[index])]
        final_value = values[(domain, final_stage)]
        history = [values[(domain, stage)] for stage in STAGES[index:]]
        bwt.append(final_value - diagonal)
        forget_best.append(max(0.0, max(history) - final_value))
    return mean(final), mean(bwt), mean(forget_best)


def adaptation_fps(log_path, samples_per_gpu=16):
    times = []
    for line in log_path.read_text(errors="ignore").splitlines():
        if "loss_roi_proto" not in line:
            continue
        match = re.search(r"\btime:\s*([0-9.]+)", line)
        if match:
            times.append(float(match.group(1)))
    if not times:
        return None
    return samples_per_gpu / mean(times)


def storage_mib(kp):
    return 4 * kp * 1 * 256 * 4 / (1024.0 ** 2)


def collect_unique():
    rows = []
    for name, config in CONFIGS.items():
        hota = metric_matrix(config["root"], "HOTA")
        assa = metric_matrix(config["root"], "AssA")
        hota_final, bwt_hota, avgf_hota = forgetting_metrics(hota)
        assa_final, _, _ = forgetting_metrics(assa)
        train_log = config["root"] / "logs" / "train_episode.log"
        rows.append(dict(
            config=name,
            prototype_number=config["kp"],
            neighbor_number=config["kn"],
            HOTA=hota_final,
            AssA=assa_final,
            AvgF_HOTA=avgf_hota,
            BWT_HOTA=bwt_hota,
            Memory_MiB=storage_mib(config["kp"]),
            Adapt_FPS_per_GPU=adaptation_fps(train_log),
            thresholds="det=0.05;init=0.30;obj=0.20",
            source_root=str(config["root"].resolve()),
        ))
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    unique = collect_unique()
    write_csv(ROOT / "apr_sensitivity_unique.csv", unique)
    by_name = {row["config"]: row for row in unique}
    table_rows = []
    for group, names in (
        ("Prototype number Kp", ["kp50_kn3", "kp100_kn3", "kp200_kn3"]),
        ("Neighbor number Kn", ["kp100_kn1", "kp100_kn3", "kp100_kn5"]),
    ):
        for name in names:
            row = dict(by_name[name])
            row = {"parameter_group": group, **row}
            table_rows.append(row)
    write_csv(ROOT / "apr_sensitivity_table.csv", table_rows)
    print("WROTE", (ROOT / "apr_sensitivity_unique.csv").resolve())
    print("WROTE", (ROOT / "apr_sensitivity_table.csv").resolve())


if __name__ == "__main__":
    main()
