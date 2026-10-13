"""Audit physical source examples used by official paired readout training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .train_source import SourcePairs, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=['bdd100k', 'ua_detrac'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    caches = [args.root / 'source' / args.dataset / f'features_{r}_w4.npz'
              for r in ['encoder', 'predictor']]
    metadata = [json.loads(p.with_suffix('.npz.json').read_text()) for p in caches]
    for m in metadata:
        assert m['status'] == 'complete' and m['gt_read']
        assert not m['engineering_smoke_only'] and not m['target_domains_read']
    with np.load(caches[0]) as encoder, np.load(caches[1]) as predictor:
        physical_fields = ['boxes', 'frame_ids', 'video_ids', 'track_ids',
                           'segment_ids', 'image_ids', 'offsets']
        assert all(np.array_equal(encoder[k], predictor[k]) for k in physical_fields)
        assert np.array_equal(encoder['features'], predictor['candidate_ranking_features'])
        videos = sorted(set(int(v) for v in encoder['video_ids']))
    heldout = videos[-max(1, int(len(videos) * .20)):]
    fit = sorted(set(videos) - set(heldout))
    width, height = (1280, 720) if args.dataset == 'bdd100k' else (960, 540)
    reports = []
    for split, selected in [('fit', fit), ('validation', heldout)]:
        datasets = [SourcePairs(p, 4, selected, width, height,
                                prediction_gaps=(1, 2, 4, 8)) for p in caches]
        assert np.array_equal(datasets[0].ranking_features, datasets[1].ranking_features)
        assert datasets[0].samples == datasets[1].samples, 'Physical history/positive/negative differs'
        digest = hashlib.sha256()
        for sample in datasets[0].samples:
            digest.update((json.dumps(sample, separators=(',', ':')) + '\n').encode())
        reports.append(dict(split=split, videos=selected, samples=len(datasets[0]),
                            physical_examples_sha256=digest.hexdigest(),
                            history_positive_and_ordered_negatives_equal=True))
    training = {}
    for readout in ['encoder', 'predictor']:
        training[readout] = {}
        for method in ['discriminative', 'predictive', 'predictive_ranking']:
            model_root = args.root / 'models' / args.dataset / readout / 'seed17' / method
            report = json.loads((model_root / 'metrics.json').read_text())
            assert report['status'] == 'complete' and report['seed'] == 17
            assert report['fit_videos'] == fit and report['validation_videos'] == heldout
            assert report['prediction_gaps'] == [1, 2, 4, 8]
            assert report['checkpoint_sha256'] == sha256(model_root / 'final.pt')
            assert report['steps'] == 2000 and report['batch_size'] == 64
            assert report['fit_samples'] == reports[0]['samples']
            assert report['validation_samples'] == reports[1]['samples']
            assert report['cache_sha256'] == sha256(caches[['encoder', 'predictor'].index(readout)])
            training[readout][method] = {k: report[k] for k in
                ['parameters', 'source_validation_rank1', 'source_validation_prediction_error',
                 'checkpoint_sha256', 'cache_sha256', 'experiment_fingerprint']}
    # Same sample list plus the recorded seed/budget yields the same sampler indices.
    sampler = np.random.RandomState(17).randint(0, reports[0]['samples'], (2000, 64))
    result = dict(status='VERIFIED_ACTUAL_SOURCE_POOLS_NOT_TARGET_MOT_RESULT', dataset=args.dataset,
                  source_GT_read=True, target_GT_read=False, history=4, prediction_gaps=[1, 2, 4, 8],
                  equal_physical_fields=physical_fields, splits=reports, source_training=training,
                  reconstructed_seed17_sampler_indices_sha256=hashlib.sha256(
                      sampler.astype('<i8').tobytes()).hexdigest(),
                  sampler_scope='Reconstructed uniform-with-replacement indices from seed and '
                                'actual sample list; not a saved per-update training trace.',
                  limitation='Predictor readout masks current pixels and has 1664 dimensions; '
                             'encoder observes them and has 1024. Frozen parameters, information '
                             'and compute are different. Prediction-error scales across readouts '
                             'do not measure identity benefit.',
                  invocation=[sys.executable, '-m', 'jepa_cmot.audit_paired_source_pools'] + sys.argv[1:],
                  audit_code_sha256=sha256(__file__))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('ACTUAL_PAIRED_SOURCE_POOLS_EQUAL', [(r['split'], r['samples']) for r in reports])


if __name__ == '__main__':
    main()
