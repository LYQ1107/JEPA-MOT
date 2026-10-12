"""Offline source training for the MOT temporal JEPA adapter.

This trainer uses real QDTrack track-head features at labelled source boxes.
The history for each target observation is assembled from strictly earlier
frames of the same source identity.  The resulting replay file is source
only and is explicitly marked as unsuitable for target inference.
"""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from darth.models.jepa import TOKEN_META_DIM, TemporalJEPAAdapter, temporal_jepa_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('cache')
    parser.add_argument('--output', required=True)
    parser.add_argument('--replay-output', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--history-len', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--latent-dim', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--val-sequences', default='')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def geometry(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    return torch.tensor([
        (x1 + x2) / (2.0 * width), (y1 + y2) / (2.0 * height),
        max(x2 - x1, 0.0) / width, max(y2 - y1, 0.0) / height])


def make_meta(record, previous, first_frame):
    current_geometry = geometry(
        record['box'], record['width'], record['height'])
    if previous is None:
        velocity = torch.zeros(4)
        gap = 0.0
    else:
        delta = max(int(record['frame']) - int(previous['frame']), 1)
        velocity = (current_geometry - previous['geometry']) / float(delta)
        gap = float(max(int(record['frame']) - int(previous['frame']) - 1, 0))
    age = float(max(int(record['frame']) - int(first_frame) + 1, 1))
    visibility_proxy = 1.0 / (1.0 + gap)
    return torch.cat([
        current_geometry, velocity,
        torch.tensor([record['score'], age, gap, visibility_proxy])])


def make_sample(history_records, target_record, history_len):
    history_records = history_records[-history_len:]
    embed_dim = int(target_record['embed'].numel())
    history_embeds = torch.zeros(history_len, embed_dim)
    history_meta = torch.zeros(history_len, TOKEN_META_DIM)
    history_mask = torch.zeros(history_len)
    first_frame = history_records[0]['frame']
    start = history_len - len(history_records)
    previous = None
    for offset, record in enumerate(history_records):
        slot = start + offset
        record['geometry'] = geometry(
            record['box'], record['width'], record['height'])
        history_embeds[slot] = record['embed']
        history_meta[slot] = make_meta(record, previous, first_frame)
        history_mask[slot] = 1.0
        previous = record
    previous = history_records[-1]
    target_geometry = geometry(
        target_record['box'], target_record['width'], target_record['height'])
    delta = max(int(target_record['frame']) - int(previous['frame']), 1)
    target_motion = (target_geometry - previous['geometry']) / float(delta)
    return dict(
        history_embeds=history_embeds,
        history_meta=history_meta,
        history_mask=history_mask,
        target_embed=target_record['embed'].float(),
        target_motion=target_motion.float(),
        quality_target=torch.tensor(1.0),
        source=True,
        created_frame=int(target_record['frame']),
        target_score=1.0)


def prepare_samples(arrays, history_len):
    groups = {}
    count = int(arrays['embeddings'].shape[0])
    for index in range(count):
        key = (int(arrays['sequence_ids'][index]),
               int(arrays['instance_ids'][index]))
        groups.setdefault(key, []).append(index)
    samples = []
    sample_groups = []
    for key, indices in sorted(groups.items()):
        indices.sort(key=lambda i: int(arrays['frame_ids'][i]))
        records = []
        for index in indices:
            record = dict(
                embed=torch.from_numpy(arrays['embeddings'][index]).float(),
                box=arrays['boxes'][index],
                frame=int(arrays['frame_ids'][index]),
                width=int(arrays['image_widths'][index]),
                height=int(arrays['image_heights'][index]),
                score=1.0)
            if records:
                samples.append(make_sample(records, record, history_len))
                sample_groups.append(key)
            records.append(record)
    return samples, sample_groups


def collate(samples, device):
    return dict(
        history_embeds=torch.stack(
            [sample['history_embeds'] for sample in samples]).to(device),
        history_meta=torch.stack(
            [sample['history_meta'] for sample in samples]).to(device),
        history_mask=torch.stack(
            [sample['history_mask'] for sample in samples]).to(device),
        target_embed=torch.stack(
            [sample['target_embed'] for sample in samples]).to(device),
        target_motion=torch.stack(
            [sample['target_motion'] for sample in samples]).to(device),
        quality_target=torch.stack(
            [sample['quality_target'] for sample in samples]).to(device))


def corrupt_student(batch, generator):
    """Mask/drop/noise/identity-mix only the student history."""
    history = batch['history_embeds'].clone()
    mask = batch['history_mask'].clone()
    drop = torch.rand(mask.shape, device=mask.device, generator=generator) < 0.15
    mask = mask * (~drop).float()
    noise = 0.03 * torch.randn(
        history.shape, device=history.device, generator=generator)
    history = history * mask.unsqueeze(-1) + noise * mask.unsqueeze(-1)
    if history.size(0) > 1:
        mix = torch.rand(
            (history.size(0), history.size(1), 1),
            device=history.device, generator=generator) < 0.05
        permutation = torch.roll(torch.arange(history.size(0), device=history.device), 1)
        history = torch.where(mix, history[permutation], history)
    corrupted = dict(batch)
    corrupted['history_embeds'] = history
    corrupted['history_mask'] = mask
    return corrupted


def mean_loss(adapter, samples, device, use_teacher=True):
    if not samples:
        return None
    batch = collate(samples, device)
    losses = temporal_jepa_loss(adapter, batch, use_teacher=use_teacher)
    return losses


def main():
    args = parse_args()
    output = os.path.abspath(args.output)
    replay_output = os.path.abspath(args.replay_output or (output + '.replay.npz'))
    if not args.overwrite and (os.path.exists(output) or
                               os.path.exists(replay_output)):
        raise IOError('refusing to overwrite existing adapter/replay')
    for path in (output, replay_output):
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    arrays = np.load(args.cache, allow_pickle=False)
    samples, sample_groups = prepare_samples(arrays, args.history_len)
    if len(samples) < 2:
        raise RuntimeError('at least two temporal source samples are needed')
    all_sequences = sorted(set(key[0] for key in sample_groups))
    requested_val = set(
        int(item) for item in args.val_sequences.split(',') if item.strip())
    if requested_val:
        val_indices = [i for i, key in enumerate(sample_groups)
                       if key[0] in requested_val]
    elif len(all_sequences) > 1:
        val_sequence = all_sequences[-1]
        val_indices = [i for i, key in enumerate(sample_groups)
                       if key[0] == val_sequence]
    else:
        # A one-sequence smoke cache cannot support sequence-disjoint split;
        # hold out identities instead and report that limitation explicitly.
        identities = sorted(set(key[1] for key in sample_groups))
        val_identity = identities[-1]
        val_indices = [i for i, key in enumerate(sample_groups)
                       if key[1] == val_identity]
    val_set = set(val_indices)
    train_samples = [sample for i, sample in enumerate(samples) if i not in val_set]
    val_samples = [sample for i, sample in enumerate(samples) if i in val_set]
    if not train_samples or not val_samples:
        split = max(1, len(samples) // 5)
        val_samples = samples[-split:]
        train_samples = samples[:-split]
        val_indices = list(range(len(samples) - split, len(samples)))

    device = torch.device(args.device)
    embed_dim = int(arrays['embeddings'].shape[1])
    latent_dim = args.latent_dim or embed_dim
    adapter = TemporalJEPAAdapter(
        embed_dim=embed_dim,
        history_len=args.history_len,
        hidden_dim=args.hidden_dim,
        latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(
        [p for p in adapter.parameters() if p.requires_grad],
        lr=args.learning_rate)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    best_val = None
    history = []
    for epoch in range(max(1, args.epochs)):
        order = list(range(len(train_samples)))
        random.Random(args.seed + epoch).shuffle(order)
        adapter.train()
        train_values = []
        for start in range(0, len(order), max(1, args.batch_size)):
            batch_samples = [train_samples[i]
                             for i in order[start:start + args.batch_size]]
            batch = corrupt_student(collate(batch_samples, device), generator)
            optimizer.zero_grad()
            losses = temporal_jepa_loss(adapter, batch, use_teacher=True)
            if not bool(torch.isfinite(losses['total']).item()):
                raise FloatingPointError('non-finite offline JEPA loss')
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in adapter.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            adapter.update_teacher(0.99)
            train_values.append(float(losses['total'].detach().cpu()))
        adapter.eval()
        with torch.no_grad():
            val_losses = []
            for start in range(0, len(val_samples), max(1, args.batch_size)):
                losses = mean_loss(
                    adapter, val_samples[start:start + args.batch_size],
                    device, use_teacher=True)
                val_losses.append(float(losses['total'].detach().cpu()))
            train_loss = float(np.mean(train_values))
            val_loss = float(np.mean(val_losses))
        history.append(dict(epoch=epoch + 1, train_loss=train_loss,
                            val_loss=val_loss))
        if best_val is None or val_loss < best_val:
            best_val = val_loss

    replay_arrays = dict(
        history_embeds=np.stack([s['history_embeds'].numpy() for s in samples]),
        history_meta=np.stack([s['history_meta'].numpy() for s in samples]),
        history_mask=np.stack([s['history_mask'].numpy() for s in samples]),
        target_embed=np.stack([s['target_embed'].numpy() for s in samples]),
        target_motion=np.stack([s['target_motion'].numpy() for s in samples]),
        quality_target=np.ones(len(samples), dtype=np.float32))
    np.savez_compressed(replay_output, **replay_arrays)
    state = {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
    payload = dict(
        state_dict=state,
        config=dict(
            embed_dim=embed_dim,
            history_len=args.history_len,
            meta_dim=TOKEN_META_DIM,
            hidden_dim=args.hidden_dim,
            latent_dim=latent_dim),
        train_history=history,
        provenance=dict(
            cache=os.path.abspath(args.cache),
            cache_sha256=sha256(args.cache),
            gt_used=True,
            future_used_for_target=True,
            target_gt_never_in_inference=True,
            student_corruption=['mask', 'drop', 'noise', 'identity_mix'],
            split_type=('sequence_disjoint' if len(all_sequences) > 1
                        else 'identity_holdout_one_sequence'),
            train_count=len(train_samples),
            val_count=len(val_samples),
            replay=os.path.abspath(replay_output)),
    )
    torch.save(payload, output)
    manifest = dict(
        adapter=os.path.abspath(output),
        replay=os.path.abspath(replay_output),
        adapter_sha256=sha256(output),
        replay_sha256=sha256(replay_output),
        cache_sha256=sha256(args.cache),
        samples=len(samples),
        train_count=len(train_samples),
        val_count=len(val_samples),
        sequences=all_sequences,
        val_indices=val_indices,
        best_val=best_val,
        history=history,
        gt_used=True,
        future_used_for_target=True,
        target_gt_never_in_inference=True)
    with open(output + '.manifest.json', 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
