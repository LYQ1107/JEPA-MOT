"""Frozen source identity summaries and native time-gap diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .train_source import SourcePairs, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--reference-metrics', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    reference = json.loads(args.reference_metrics.read_text())
    metadata = json.loads(Path(str(args.cache) + '.json').read_text())
    if not metadata['gt_read'] or reference['target_gt_used']:
        raise ValueError('Source baseline requires source-only GT features')
    if sha256(args.cache) != reference['cache_sha256']:
        raise ValueError('Source baseline cache differs from matched training')
    width, height = (960, 540) if metadata['dataset'] == 'ua_detrac' else (1280, 720)
    valid = SourcePairs(args.cache, reference['config']['history'], reference['validation_videos'],
                        width, height, prediction_gaps=reference['prediction_gaps'])
    methods = {name: dict(hits=0., error=0., tied=0) for name in
               ['last', 'mean', 'recent_weighted', 'current_GT_latent_oracle']}
    total = 0
    with torch.no_grad():
        for features, geometry, mask, candidates, candidate_mask in DataLoader(valid, batch_size=128):
            weighted = mask * torch.arange(1, mask.shape[1] + 1)
            queries = dict(last=features[:, -1],
                           mean=F.normalize(features.sum(1), dim=-1),
                           recent_weighted=F.normalize((features * weighted[..., None]).sum(1), dim=-1),
                           current_GT_latent_oracle=candidates[:, 0])
            for name, query in queries.items():
                scores = torch.einsum('bd,bnd->bn', query, candidates).masked_fill(~candidate_mask, -1e4)
                tied = torch.isclose(scores, scores.max(1).values[:, None], rtol=0, atol=1e-6) & candidate_mask
                count = tied.sum(1)
                methods[name]['hits'] += float((tied[:, 0].float() / count).sum())
                methods[name]['tied'] += int((count > 1).sum())
                methods[name]['error'] += float((1 - F.cosine_similarity(query, candidates[:, 0])).sum())
            total += len(features)
    gaps = np.asarray([int(valid.frames[target] - valid.frames[previous[-1]])
                       for previous, target, _ in valid.samples])
    report = dict(cache_sha256=reference['cache_sha256'], dataset=metadata['dataset'],
        roi_mode=metadata['roi_mode'], validation_videos=reference['validation_videos'],
        samples=total, prediction_gaps_in_observation_indices=reference['prediction_gaps'],
        native_frame_prediction_gap=dict(minimum=int(gaps.min()), median=float(np.median(gaps)),
            p90=float(np.percentile(gaps, 90)), maximum=int(gaps.max()), fraction_delta_t_1=float((gaps == 1).mean())),
        methods={name: dict(rank1=values['hits'] / total, prediction_error=values['error'] / total,
                            tied_examples=values['tied']) for name, values in methods.items()},
        tie_rule='fractional-credit-within-1e-6', target_GT_used=False,
        oracle_scope='source diagnostic only; current identity GT never available to tracker')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print('SOURCE_BASELINES_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
