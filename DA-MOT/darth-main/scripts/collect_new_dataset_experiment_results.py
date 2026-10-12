#!/usr/bin/env python
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = [
    ('bbox/OVERALL/mAP', 'mAP'), ('bbox/OVERALL/mAP_50', 'mAP_50'),
    ('track/OVERALL/MOTA', 'MOTA_OVERALL'), ('track/AVERAGE/MOTA', 'MOTA_AVG'),
    ('track/OVERALL/IDF1', 'IDF1_OVERALL'), ('track/AVERAGE/IDF1', 'IDF1_AVG'),
    ('hota/OVERALL/HOTA', 'HOTA_OVERALL'), ('hota/AVERAGE/HOTA', 'HOTA_AVG'),
    ('hota/OVERALL/DetA', 'DetA_OVERALL'), ('hota/OVERALL/AssA', 'AssA_OVERALL'),
    ('hota/OVERALL/LocA', 'LocA_OVERALL'),
    ('track/pedestrian/MOTA', 'MOTA_pedestrian'), ('track/car/MOTA', 'MOTA_car'), ('track/truck/MOTA', 'MOTA_truck'),
    ('track/pedestrian/IDF1', 'IDF1_pedestrian'), ('track/car/IDF1', 'IDF1_car'), ('track/truck/IDF1', 'IDF1_truck'),
    ('hota/pedestrian/HOTA', 'HOTA_pedestrian'), ('hota/car/HOTA', 'HOTA_car'), ('hota/truck/HOTA', 'HOTA_truck'),
]
FIELDS = (
    ['dataset', 'method', 'stage', 'eval_domain', 'status']
    + [value for _, value in KEYS]
    + ['checkpoint', 'eval_log']
)
CORE_METRICS = ['mAP', 'mAP_50', 'MOTA_OVERALL', 'IDF1_OVERALL', 'HOTA_OVERALL']

DATASETS = {
    'bdd_da': {
        'source': 'clear',
        'baseline_domains': ['overcast', 'rainy', 'dawn_dusk', 'night'],
        'trimodule_domains': ['overcast', 'rainy', 'dawn_dusk', 'night'],
        'layout': 'bdd_da',
    },
    'motsynth': {
        'source': 'normal',
        'baseline_domains': ['cloudy', 'fog', 'rain', 'snow', 'dark'],
        'trimodule_domains': ['cloudy', 'fog', 'rain', 'snow', 'dark'],
    },
    'ua_detrac': {
        'source': 'sunny',
        'baseline_domains': ['cloudy', 'rainy', 'night'],
        'trimodule_domains': ['cloudy', 'rainy', 'night'],
    },
    'vkitti2': {
        'source': 'clone',
        'baseline_domains': ['overcast', 'morning', 'rain', 'fog', 'sunset'],
        'trimodule_domains': ['overcast', 'morning', 'rain', 'fog', 'sunset'],
    },
}


def parse_keyvals(line):
    return {m.group(1): float(m.group(2)) for m in re.finditer(r'([\w/]+):\s*([-+]?\d+(?:\.\d+)?)', line)}


def parse_log(path):
    path = Path(path)
    if not path.exists():
        return None
    lines = path.read_text(errors='replace').splitlines()
    metrics = {}
    for line in lines:
        if 'bbox/OVERALL/mAP:' in line or 'track/OVERALL/MOTA:' in line:
            metrics.update(parse_keyvals(line))
        if 'Average Precision' in line and 'IoU=0.50:0.95' in line and 'area=   all' in line:
            metrics['bbox/OVERALL/mAP'] = float(line.rsplit('=', 1)[1].strip()) * 100
        if 'Average Precision' in line and 'IoU=0.50      ' in line and 'area=   all' in line:
            metrics['bbox/OVERALL/mAP_50'] = float(line.rsplit('=', 1)[1].strip()) * 100
        s = line.strip()
        if s.startswith(('AVERAGE ', 'OVERALL ', 'pedestrian ', 'car ', 'truck ')):
            parts = s.split()
            if len(parts) == 7:
                name = parts[0]
                loca, deta, mota, hota, idf1, assa = map(float, parts[1:])
                key = name
                for prefix in ('track', 'hota'):
                    metrics[f'{prefix}/{key}/LocA'] = loca
                    metrics[f'{prefix}/{key}/DetA'] = deta
                    metrics[f'{prefix}/{key}/MOTA'] = mota
                    metrics[f'{prefix}/{key}/HOTA'] = hota
                    metrics[f'{prefix}/{key}/IDF1'] = idf1
                    metrics[f'{prefix}/{key}/AssA'] = assa
    return metrics or None


def latest_log(pattern):
    logs = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def row(dataset, method, stage, domain, ckpt, log):
    ckpt = Path(ckpt) if ckpt else None
    log = Path(log) if log else None
    metrics = parse_log(log) if log and log.is_file() else None
    if ckpt is None or not ckpt.is_file():
        status = 'missing_checkpoint'
    elif log is None or not log.is_file():
        status = 'missing_log'
    elif not metrics:
        status = 'parse_failed'
    else:
        status = 'done'
    out = {
        'dataset': dataset,
        'method': method,
        'stage': stage,
        'eval_domain': domain,
        'status': status,
        'checkpoint': str(ckpt) if ckpt else '',
        'eval_log': str(log) if log else '',
    }
    for key, name in KEYS:
        out[name] = round(metrics[key], 4) if metrics and key in metrics else ''
    return out


def collect_bdd_da(dataset, info):
    rows = []
    src = info['source']
    baseline_root = ROOT / 'work_dirs/bdd_4domain/task1_single_target'
    for d in info['baseline_domains']:
        adapt_name = f'darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_{d}'
        ckpt = baseline_root / f'tta_{src}_to_{d}/{adapt_name}/epoch_5.pth'
        log = baseline_root / f'eval/tta_{src}_to_{d}/epoch_5/{d}/eval.log'
        rows.append(row(
            dataset, 'baseline_single_target', f'{src}_to_{d}_epoch5', d,
            ckpt, log,
        ))

    method = 'trimodule_spaP1_causalProtoShrink_roiV1Weak'
    experiment = (
        ROOT / 'work_dirs/bdd_4domain/tri_module_pca_variants'
        / 'spaP1_causalProtoShrink_roiV1Weak_4domain_5e'
    )
    for idx, d in enumerate(info['trimodule_domains'], 1):
        stage = f'stage{idx:02d}_after_{d}'
        ckpt = experiment / f'episode/adapt/{stage}/epoch_5.pth'
        log = experiment / f'forgetting_eval/{stage}/epoch_5/{d}/eval.log'
        rows.append(row(dataset, method, stage, d, ckpt, log))
    return rows


def collect_dataset(dataset, info):
    if info.get('layout') == 'bdd_da':
        return collect_bdd_da(dataset, info)

    rows = []
    src = info['source']
    for d in info['baseline_domains']:
        log = (latest_log(f'work_dirs/{dataset}/test/source_{src}_to_{d}_direct/**/**/*.log')
               or latest_log(f'work_dirs/{dataset}/test/logs/source_{src}_to_{d}_direct*.log'))
        ckpt = latest_log(f'work_dirs/{dataset}/source/**/epoch_20.pth')
        rows.append(row(dataset, 'baseline_source_direct', 'source_epoch20', d, ckpt, log))
        log = (latest_log(f'work_dirs/{dataset}/test/baseline_{src}_to_{d}_5e/**/**/*.log')
               or latest_log(f'work_dirs/{dataset}/baseline_single_target/logs/eval_{d}.log')
               or latest_log(f'work_dirs/{dataset}/test/logs/baseline_{src}_to_{d}_5e_eval*.log'))
        ckpt = (latest_log(f'work_dirs/{dataset}/baseline_single_target/adapt/{src}_to_{d}_5e/epoch_5.pth')
                or latest_log(f'work_dirs/{dataset}/baseline_single_target/adapt/{src}_to_{d}_5e/**/epoch_5.pth'))
        rows.append(row(dataset, 'baseline_single_target', f'{src}_to_{d}_epoch5', d, ckpt, log))
    for idx, d in enumerate(info['trimodule_domains'], 1):
        stage = f'stage{idx:02d}_after_{d}'
        log = ROOT / f'work_dirs/{dataset}/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/{stage}/epoch_5/{d}/eval.log'
        ckpt = ROOT / f'work_dirs/{dataset}/trimodule_spaP1_causalProtoShrink_roiV1Weak/episode/adapt/{stage}/epoch_5.pth'
        rows.append(row(dataset, 'trimodule_spaP1_causalProtoShrink_roiV1Weak', stage, d, ckpt if ckpt.exists() else None, log if log.exists() else None))
    return rows


def validate_rows(rows):
    seen = set()
    for index, result in enumerate(rows, 1):
        extra = set(result) - set(FIELDS)
        if extra:
            raise ValueError(f'row {index} has unexpected fields: {sorted(extra)}')
        key = tuple(result[name] for name in ('dataset', 'method', 'stage', 'eval_domain'))
        if key in seen:
            raise ValueError(f'duplicate result row: {key}')
        seen.add(key)
        if result['status'] != 'done':
            continue
        for name in ('checkpoint', 'eval_log'):
            if not result[name] or not Path(result[name]).is_file():
                raise ValueError(f'done row {key} has invalid {name}: {result[name]}')
        missing = [name for name in CORE_METRICS if result[name] == '']
        if missing:
            raise ValueError(f'done row {key} is missing metrics: {missing}')


def backup_existing(path):
    if not path.exists():
        return None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = path.with_name(f'{path.name}.bak.{timestamp}')
    shutil.copy2(str(path), str(backup))
    print(f'Backed up {path} -> {backup}')
    return backup


def write_csv(dataset, rows):
    out = ROOT / f'work_dirs/{dataset}/{dataset}_new_dataset_experiment_results.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    validate_rows(rows)
    backup_existing(out)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Written {out} ({len(rows)} rows)')
    for r in rows:
        print(f"{dataset:9s} {r['method'][:28]:28s} {r['eval_domain']:8s} {r['status']:12s} mAP={r['mAP']} MOTA={r['MOTA_OVERALL']} IDF1={r['IDF1_OVERALL']} HOTA={r['HOTA_OVERALL']}")


def main():
    all_rows = []
    for dataset, info in DATASETS.items():
        rows = collect_dataset(dataset, info)
        write_csv(dataset, rows)
        all_rows.extend(rows)
    out = ROOT / 'work_dirs/new_dataset_experiment_results_all.csv'
    validate_rows(all_rows)
    backup_existing(out)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'Written {out} ({len(all_rows)} rows)')


if __name__ == '__main__':
    main()
