#!/usr/bin/env python
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work_dirs/vkitti2/vkitti2_darth_baseline_results.csv'

KEY_MAP = [
    ('bbox/OVERALL/mAP', 'mAP'),
    ('bbox/OVERALL/mAP_50', 'mAP_50'),
    ('track/OVERALL/MOTA', 'MOTA_OVERALL'),
    ('track/AVERAGE/MOTA', 'MOTA_AVG'),
    ('track/OVERALL/IDF1', 'IDF1_OVERALL'),
    ('track/AVERAGE/IDF1', 'IDF1_AVG'),
    ('hota/OVERALL/HOTA', 'HOTA_OVERALL'),
    ('hota/AVERAGE/HOTA', 'HOTA_AVG'),
    ('hota/OVERALL/DetA', 'DetA_OVERALL'),
    ('hota/OVERALL/AssA', 'AssA_OVERALL'),
    ('hota/OVERALL/LocA', 'LocA_OVERALL'),
    ('track/OVERALL/Rcll', 'Rcll_OVERALL'),
    ('track/OVERALL/Prcn', 'Prcn_OVERALL'),
    ('track/OVERALL/IDSw', 'IDSw_OVERALL'),
    ('track/OVERALL/FP', 'FP_OVERALL'),
    ('track/OVERALL/FN', 'FN_OVERALL'),
    ('track/car/MOTA', 'MOTA_car'),
    ('track/truck/MOTA', 'MOTA_truck'),
    ('track/car/IDF1', 'IDF1_car'),
    ('track/truck/IDF1', 'IDF1_truck'),
    ('hota/car/HOTA', 'HOTA_car'),
    ('hota/truck/HOTA', 'HOTA_truck'),
]

HEADER = ['run_id', 'description', 'epochs', 'status'] + [v for _, v in KEY_MAP]


def parse_metrics(line):
    metrics = {}
    for m in re.finditer(r'([\w/]+):\s*([-+]?\d+(?:\.\d+)?)', line):
        metrics[m.group(1)] = float(m.group(2))
    return metrics


def parse_epoch_val(log_path, epoch):
    if not log_path.exists():
        return None
    for line in log_path.read_text(errors='replace').splitlines():
        if 'Epoch(val)' not in line:
            continue
        m = re.search(r'Epoch\(val\) \[(\d+)\]', line)
        if m and int(m.group(1)) == epoch:
            return parse_metrics(line)
    return None


def parse_last_epoch_val(log_path):
    if not log_path.exists():
        return None
    val_lines = [line for line in log_path.read_text(errors='replace').splitlines()
                 if 'Epoch(val)' in line]
    return parse_metrics(val_lines[-1]) if val_lines else None


def parse_latest_test_log(domain):
    log_dir = ROOT / 'work_dirs/vkitti2/test/logs'
    logs = sorted(log_dir.glob(f'clone_to_{domain}_5e_eval_*.log'),
                  key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    lines = logs[-1].read_text(errors='replace').splitlines()
    candidates = [line for line in lines if 'track/OVERALL/IDF1:' in line and 'bbox/OVERALL/mAP:' in line]
    if not candidates:
        return None
    return parse_metrics(candidates[-1])


def row(run_id, desc, epochs, status, metrics):
    values = [run_id, desc, epochs, status]
    for key, _ in KEY_MAP:
        values.append(round(metrics[key], 4) if metrics and key in metrics else '')
    return values


def main():
    overcast_log = ROOT / 'work_dirs/vkitti2/adapt/logs/rerun_srcinit_10e_20260707_204818.log'
    queue_logs = sorted((ROOT / 'work_dirs/vkitti2/adapt/logs').glob('clone_to_multi_domains_5e_queue_*.log'),
                        key=lambda p: p.stat().st_mtime)
    morning_metrics = None
    if queue_logs:
        val_lines = [line for line in queue_logs[-1].read_text(errors='replace').splitlines()
                     if 'Epoch(val)' in line]
        if len(val_lines) >= 5:
            morning_metrics = parse_metrics(val_lines[4])
    if morning_metrics is None:
        morning_metrics = parse_latest_test_log('morning')

    rows = [
        row('adapt_overcast_e5', 'clone→overcast (epoch5)', 5, 'done', parse_epoch_val(overcast_log, 5)),
        row('adapt_overcast_e10', 'clone→overcast (epoch10)', 10, 'done', parse_last_epoch_val(overcast_log)),
        row('adapt_morning_e5', 'clone→morning (epoch5)', 5, 'done' if morning_metrics else 'missing', morning_metrics),
    ]

    for domain in ['fog', 'rain', 'sunset']:
        metrics = parse_latest_test_log(domain)
        ckpts = list((ROOT / f'work_dirs/vkitti2/adapt/darth_clone_to_{domain}_5e').glob(
            f'darth_qdtrack_faster-rcnn_r50_fpn_5e_vkitti2_{domain}_from_clone/*/epoch_5.pth'))
        if metrics:
            status = 'done'
        elif ckpts:
            status = 'trained_no_eval'
        else:
            status = 'pending'
        rows.append(row(f'adapt_{domain}_e5', f'clone→{domain} (epoch5)', 5, status, metrics))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f'Written: {OUT}')
    print(f"{'run_id':<22} {'status':<15} {'mAP':>8} {'MOTA':>8} {'IDF1':>8} {'HOTA':>8}")
    print('-' * 78)
    with OUT.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            print(f"{r['run_id']:<22} {r['status']:<15} {r['mAP']:>8} {r['MOTA_OVERALL']:>8} {r['IDF1_OVERALL']:>8} {r['HOTA_OVERALL']:>8}")


if __name__ == '__main__':
    main()
