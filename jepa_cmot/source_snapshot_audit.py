"""Evaluate continual snapshots on the original, complete source validation pairs."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .models import HistoryPredictor, PredictorConfig
from .train_source import SourcePairs, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    snapshots = json.loads((args.sequence/'snapshots.json').read_text())
    protocol = json.loads((args.sequence/'protocol.json').read_text())
    reference = {method: json.loads((args.source_root/method/'metrics.json').read_text())
                 for method in ['discriminative', 'predictive_ranking']}
    cache_sha = sha256(args.cache)
    common = reference['predictive_ranking']
    if any(r['cache_sha256'] != cache_sha or r['validation_videos'] != common['validation_videos']
           or r['prediction_gaps'] != common['prediction_gaps'] for r in reference.values()):
        raise ValueError('Source reference inputs differ')
    for r in reference.values():
        original = dict(config=r['config'], method=r['method'], seed=r['seed'], steps=r['steps'],
            batch=r['batch_size'], source_domain=protocol['order'][0],
            width=float(args.width), height=float(args.height), cache_sha256=cache_sha,
            prediction_gaps=r['prediction_gaps'], engineering_smoke_only=False,
            sampler='uniform-with-replacement-v1', geometry='shared-causal-history-v2',
            code_sha256=r['training_code_sha256'])
        fingerprint = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
        if fingerprint != r['experiment_fingerprint']:
            raise ValueError('Geometry/input configuration differs from original source experiment')
    valid = SourcePairs(args.cache, common['config']['history'], common['validation_videos'],
                        args.width, args.height, prediction_gaps=common['prediction_gaps'])
    if len(valid) != common['validation_samples']:
        raise ValueError('Source validation pair coverage differs')
    models = {}
    records = {}
    for stage, weights in enumerate(snapshots):
        for control, path in weights.items():
            payload = torch.load(path, map_location='cpu')
            model = HistoryPredictor(PredictorConfig(**payload['config']))
            model.load_state_dict(payload['state_dict'])
            model.eval()
            key = (control, stage)
            models[key] = model
            records[key] = dict(checkpoint=str(path), checkpoint_sha256=sha256(path),
                                updates=payload.get('updates', 0), samples=0, hits=0.,
                                prediction_sum=0., tied_examples=0, hard_negative_margin_sum=0.,
                                query_sum=torch.zeros(model.config.feature_dim, dtype=torch.float64),
                                query_square_sum=0.)
    with torch.no_grad():
        for batch in DataLoader(valid, batch_size=64, num_workers=0):
            features, metadata, mask, candidates, candidate_mask = batch
            for key, model in models.items():
                query = model(features, metadata, mask)
                scores = torch.einsum('bd,bnd->bn', query, candidates).masked_fill(~candidate_mask, -1e4)
                tied = torch.isclose(scores, scores.max(1).values[:, None], rtol=0, atol=1e-6) & candidate_mask
                counts = tied.sum(1)
                record = records[key]
                record['hits'] += float((tied[:, 0].float()/counts).sum())
                record['samples'] += len(query)
                record['tied_examples'] += int((counts > 1).sum())
                record['prediction_sum'] += float((1-F.cosine_similarity(query, candidates[:, 0])).sum())
                record['hard_negative_margin_sum'] += float((scores[:, 0]-scores[:, 1:].max(1).values).sum())
                record['query_sum'] += query.double().sum(0)
                record['query_square_sum'] += float(query.double().square().sum())
    values = {}
    for (control, stage), record in records.items():
        record['rank1'] = record.pop('hits')/record['samples']
        record['prediction_error'] = record.pop('prediction_sum')/record['samples']
        record['mean_positive_minus_hard_negative_cosine'] = record.pop('hard_negative_margin_sum')/record['samples']
        query_sum = record.pop('query_sum')
        square_sum = record.pop('query_square_sum')
        count = record['samples']
        record['query_covariance_trace'] = square_sum/count-float((query_sum/count).square().sum())
        record['global_mean_distinct_pair_query_cosine'] = (
            float(query_sum.square().sum())-square_sum)/(count*(count-1))
        method = 'discriminative' if control.startswith('discriminative') else 'predictive_ranking'
        if stage == 0:
            record['rank1_delta_from_original_source_GPU_validation'] = (
                record['rank1']-reference[method]['source_validation_rank1'])
            record['prediction_error_delta_from_original_source_GPU_validation'] = (
                record['prediction_error']-reference[method]['source_validation_prediction_error'])
        values.setdefault(control, {})[str(stage)] = record
    for stages in values.values():
        for record in stages.values():
            record['rank1_delta_from_same_CPU_stage0'] = record['rank1']-stages['0']['rank1']
            record['prediction_error_delta_from_same_CPU_stage0'] = (
                record['prediction_error']-stages['0']['prediction_error'])
    result = dict(dataset=protocol['dataset'], cache_sha256=cache_sha,
        scope='all original source-internal validation pairs; no optimization or target labels',
        device='cpu-float32', pair_count=len(valid), validation_videos=common['validation_videos'],
        prediction_gaps=common['prediction_gaps'], width=args.width, height=args.height,
        rank_tie_rule='fractional-credit-within-1e-6', source_GT_read_for_evaluation=True,
        target_GT_read=False, snapshots=values,
        original_source_experiment_fingerprints_verified=True,
        limitation='source descriptor prediction/ranking is not MOT HOTA/AssA or final continual forgetting')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('SOURCE_SNAPSHOT_AUDIT_COMPLETE', len(models), len(valid), flush=True)


if __name__ == '__main__':
    main()
