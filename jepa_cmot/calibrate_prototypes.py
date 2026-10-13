"""Original source validation candidates; matched two-parameter calibration."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .prototypes import ema_step, unit
from .train_source import SourcePairs, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--reference-metrics', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    metadata = json.loads(args.cache.with_suffix('.npz.json').read_text())
    reference = json.loads(args.reference_metrics.read_text())
    if not metadata['gt_read'] or metadata['target_domains_read'] or reference['target_gt_used']:
        raise ValueError('Prototype calibration is source-only')
    if sha256(args.cache) != reference['cache_sha256']:
        raise ValueError('Prototype source cache differs from the matched reference')
    width, height = (960, 540) if metadata['dataset'] == 'ua_detrac' else (1280, 720)
    data = SourcePairs(args.cache, 4, reference['validation_videos'], width, height,
                       prediction_gaps=reference['prediction_gaps'])
    groups = {}
    for index in range(len(data.features)):
        key = (int(data.videos[index]), int(data.tracks[index]), int(data.segments[index]))
        groups.setdefault(key, []).append(index)
    prototypes = np.empty_like(data.features)
    for indices in groups.values():
        previous = None
        for index in sorted(indices, key=lambda i: int(data.frames[i])):
            previous = ema_step(previous, data.features[index])
            prototypes[index] = previous
    args.out.mkdir(parents=True, exist_ok=True)
    for method in ['mean4', 'ema']:
        values = []
        labels = []
        for previous, target, candidates in data.samples:
            query = unit(data.features[previous].sum(0)) if method == 'mean4' else prototypes[previous[-1]]
            values.extend((data.features[candidates] @ query).tolist())
            labels.extend([1.] + [0.]*(len(candidates)-1))
        scores = torch.tensor(values)
        target = torch.tensor(labels)
        slope = torch.nn.Parameter(torch.tensor(8.))
        intercept = torch.nn.Parameter((scores[target > 0].mean()+scores[target == 0].mean())/2)
        optimizer = torch.optim.Adam([slope, intercept], lr=.05)
        positive_weight = (target == 0).sum()/(target > 0).sum()
        for _ in range(200):
            loss = F.binary_cross_entropy_with_logits(F.softplus(slope)*(scores-intercept), target,
                                                     pos_weight=positive_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        report = dict(method='frozen_'+method, slope=float(F.softplus(slope).detach()),
            intercept=float(intercept.detach()), loss=float(loss.detach()), steps=200,
            trainable_parameters=2, identity_predictor_parameters=0, target_gt_used=False,
            fit_scope='source-validation-only', videos=reference['validation_videos'],
            samples=len(data), cache_sha256=reference['cache_sha256'], checkpoint_sha256=None,
            source_feature_readout=metadata['roi_mode'], beta=.25 if method == 'ema' else None,
            history='last four observations' if method == 'mean4' else 'all strictly previous observations in same video/segment; normalize after each EMA update')
        (args.out/(method+'.json')).write_text(json.dumps(report, indent=2)+'\n')
        print('SOURCE_PROTOTYPE_CALIBRATION_COMPLETE', method, len(data), flush=True)


if __name__ == '__main__':
    main()
