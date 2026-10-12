#!/usr/bin/env python3
"""Re-evaluate archived MOTSynth results with the archived TrackEval code."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('/data1/liuyeqiang/nas/A100/data1/LWR/vranlee/SERVER_ONLY/avis')
REPO = ARCHIVE / 'GeneralTrack-main'
RUN = REPO / 'outputs/jepa_motsynth_mapped_correct_tuning_vjepa21_score0'
FIELDS = ('HOTA', 'DetA', 'AssA', 'MOTA', 'IDF1', 'IDSW', 'FP', 'FN')
sys.path.insert(0, str(ROOT / 'GeneralTrack-main/tools'))
from eval_damot_generaltrack import evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(data, path):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domains', nargs='+', default=['cloudy', 'dark', 'fog', 'rain', 'snow'])
    parser.add_argument('--splits', nargs='+', choices=['full', 'holdout'], default=['full', 'holdout'])
    args = parser.parse_args()
    output = ROOT / 'results/metric_reproduction.json'
    output.parent.mkdir(exist_ok=True)
    report = {
        'purpose': 'Metric reproduction from archived predictions; no tracker rerun or model update.',
        'archive': str(ARCHIVE),
        'target_gt_use': 'evaluation only',
        'evaluator_sha256': {
            str(p.relative_to(ROOT)): digest(p)
            for p in (ROOT / 'GeneralTrack-main').rglob('*.py')
        },
        'rows': [],
        'complete': False,
    }
    for split in args.splits:
        for domain in args.domains:
            prepared = (
                REPO / 'datasets/generaltrack_prepared_mapped/source_detector_thr02/motsynth' / domain
                if split == 'full' else
                ARCHIVE / 'JEPA/outputs/jepa_motsynth_mapped_splits/motsynth' / domain / split
            )
            annotation_path = prepared / 'annotations/val.json'
            annotation = json.loads(annotation_path.read_text())
            annotation_hash = digest(annotation_path)
            for variant in ('baseline', 'geometry_null', 'frozen', 'tuned'):
                archived_output = (
                    REPO / 'outputs/generaltrack_mapped/source_detector_thr02/motsynth' / domain
                    if split == 'full' and variant == 'baseline' else
                    RUN / (split + '_' + variant) / domain
                )
                predictions = (
                    REPO / 'outputs/generaltrack_mapped/source_detector_thr02/motsynth' / domain / 'track_results'
                    if variant == 'baseline' else archived_output / 'track_results'
                )
                metric_path = archived_output / (
                    'metrics.json' if split == 'full' and variant == 'baseline' else 'eval/metrics.json'
                )
                expected = json.loads(metric_path.read_text())['summary']
                before = time.monotonic()
                actual = evaluate(annotation, predictions)
                delta = {f: actual['metrics'][f] - expected[f] for f in FIELDS}
                passed = all(abs(v) <= 0.000051 for v in delta.values())
                row = {
                    'split': split, 'domain': domain, 'variant': variant,
                    'annotation_sha256': annotation_hash,
                    'prediction_sha256': {p.name: digest(p) for p in sorted(predictions.glob('*.txt'))},
                    'num_videos': actual['sequence_count'], 'num_frames': actual['frame_count'],
                    'metrics': actual['metrics'], 'expected': {f: expected[f] for f in FIELDS},
                    'delta': delta, 'pass': passed,
                    'seconds': round(time.monotonic() - before, 2),
                }
                report['rows'].append(row)
                save(report, output)
                print(f'{split}/{domain}/{variant}: {"PASS" if passed else "FAIL"} '
                      f'HOTA={actual["metrics"]["HOTA"]:.4f} seconds={row["seconds"]}', flush=True)
                if not passed:
                    raise RuntimeError(f'Archived metric mismatch: {split}/{domain}/{variant}: {delta}')
            del annotation
    report['complete'] = True
    report['all_pass'] = all(row['pass'] for row in report['rows'])
    report['macro'] = {}
    for split in args.splits:
        report['macro'][split] = {}
        for variant in ('baseline', 'geometry_null', 'frozen', 'tuned'):
            rows = [r for r in report['rows'] if r['split'] == split and r['variant'] == variant]
            report['macro'][split][variant] = {
                f: sum(r['metrics'][f] for r in rows) / len(rows) for f in FIELDS
            }
    save(report, output)
    print(f'Completed {len(report["rows"])} evaluations: {output}', flush=True)


if __name__ == '__main__':
    main()
