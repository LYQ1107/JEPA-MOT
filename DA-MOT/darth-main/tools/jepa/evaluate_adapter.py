"""Evaluate an adapter on a labelled source replay anchor.

This is an audit utility for continual runs.  It never reads target
annotations: the input replay file is the source-domain cache exported by
``train_temporal_adapter.py``.  Comparing the source loss before and after an
online run gives a small, explicit retention signal alongside target MOT
metrics.
"""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from darth.models.jepa import TemporalJEPAAdapter, temporal_jepa_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(payload):
    state = payload.get('state_dict', payload.get('adapter', payload))
    cleaned = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        new_key = key
        for prefix in ('module.', 'jepa.', 'adapter.'):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def make_batch(arrays, start, end, device):
    batch = dict(
        history_embeds=torch.from_numpy(
            arrays['history_embeds'][start:end]).float(),
        history_meta=torch.from_numpy(
            arrays['history_meta'][start:end]).float(),
        history_mask=torch.from_numpy(
            arrays['history_mask'][start:end]).float(),
        target_embed=torch.from_numpy(
            arrays['target_embed'][start:end]).float(),
        target_motion=torch.from_numpy(
            arrays['target_motion'][start:end]).float())
    if 'quality_target' in arrays:
        batch['quality_target'] = torch.from_numpy(
            arrays['quality_target'][start:end]).float()
    return {key: value.to(device) for key, value in batch.items()}


def main():
    args = parse_args()
    adapter_path = os.path.abspath(args.adapter)
    replay_path = os.path.abspath(args.replay)
    payload = torch.load(adapter_path, map_location='cpu')
    config = payload.get('config', {}) if isinstance(payload, dict) else {}
    arrays = np.load(replay_path, allow_pickle=False)
    count = int(arrays['target_embed'].shape[0])
    if count < 1:
        raise RuntimeError('source replay is empty')
    embed_dim = int(config.get('embed_dim', arrays['target_embed'].shape[1]))
    history_len = int(config.get('history_len', arrays['history_embeds'].shape[1]))
    meta_dim = int(config.get('meta_dim', arrays['history_meta'].shape[2]))
    hidden_dim = int(config.get('hidden_dim', 128))
    latent_dim = int(config.get('latent_dim', embed_dim))
    device = torch.device(args.device)
    adapter = TemporalJEPAAdapter(
        embed_dim=embed_dim,
        history_len=history_len,
        meta_dim=meta_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim).to(device)
    missing, unexpected = adapter.load_state_dict(load_state(payload),
                                                  strict=False)
    adapter.eval()
    totals = []
    component_values = {}
    with torch.no_grad():
        for start in range(0, count, max(1, args.batch_size)):
            end = min(count, start + max(1, args.batch_size))
            losses = temporal_jepa_loss(
                adapter, make_batch(arrays, start, end, device),
                use_teacher=True)
            for key, value in losses.items():
                if key == 'predictions':
                    continue
                component_values.setdefault(key, []).append(
                    float(value.detach().cpu()))
            totals.append(end - start)
    result = dict(
        adapter=adapter_path,
        adapter_sha256=sha256(adapter_path),
        replay=replay_path,
        replay_sha256=sha256(replay_path),
        samples=sum(totals),
        device=str(device),
        missing_keys=list(missing),
        unexpected_keys=list(unexpected),
        gt_used_for_this_evaluation=True,
        target_gt_used=False,
        losses={key: float(np.average(values))
                for key, values in component_values.items()})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
