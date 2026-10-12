"""Paired video bootstrap of aggregate TrackEval metrics, preserving TP weights."""
import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_stream import HOTA, CLEAR, Identity


def restore(raw):
    return {metric: {key: np.asarray(value) if isinstance(value, list) else value
                     for key, value in fields.items()} for metric, fields in raw.items()}


def aggregate(videos, indices):
    results = {}
    for metric in [HOTA(), CLEAR({'PRINT_CONFIG': False}), Identity({'PRINT_CONFIG': False})]:
        results[metric.get_name()] = metric.combine_sequences(
            {str(slot): videos[index][metric.get_name()] for slot, index in enumerate(indices)})
    return np.asarray([100 * np.mean(results['HOTA']['HOTA']),
                       100 * np.mean(results['HOTA']['AssA']),
                       100 * results['Identity']['IDF1']], float)


def compare(reference, candidate, repeats=2000, seed=17):
    names = sorted(reference['videos'])
    if names != sorted(candidate['videos']):
        raise ValueError('Bootstrap requires exactly matched videos')
    for name in names:
        for key in ['native_frames', 'scored_frames']:
            if reference['videos'][name][key] != candidate['videos'][name][key]:
                raise ValueError('Bootstrap frame coverage differs')
    if len(names) < 2:
        raise ValueError('A single video cannot estimate between-video uncertainty')
    left = [restore(reference['videos'][name]['raw']) for name in names]
    right = [restore(candidate['videos'][name]['raw']) for name in names]
    all_indices = np.arange(len(names))
    point = aggregate(right, all_indices) - aggregate(left, all_indices)
    rng = np.random.RandomState(seed)
    samples = []
    for _ in range(repeats):
        selected = rng.randint(0, len(names), len(names))
        samples.append(aggregate(right, selected) - aggregate(left, selected))
    lower, upper = np.percentile(samples, [2.5, 97.5], axis=0)
    return dict(videos=names, repeats=repeats, seed=seed,
                estimator='paired video resampling; recompute aggregate TrackEval from raw counts',
                difference='candidate minus reference',
                metrics={name: dict(delta=float(point[i]), percentile_95=[float(lower[i]), float(upper[i])])
                         for i, name in enumerate(['HOTA', 'AssA', 'IDF1'])},
                scope='between-video uncertainty for this fixed seed; not cross-seed uncertainty')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    for field in ['dataset', 'domain', 'split', 'protocol']:
        if reference[field] != candidate[field]:
            raise ValueError('Bootstrap evaluation protocols differ')
    report = compare(reference, candidate, args.repeats, args.seed)
    report.update(dataset=reference['dataset'], domain=reference['domain'], split=reference['split'],
                  reference=str(args.reference), candidate=str(args.candidate))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print('PAIRED_BOOTSTRAP_COMPLETE', report['metrics'], flush=True)


if __name__ == '__main__':
    main()
