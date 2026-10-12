"""Build a source-only track feature cache for temporal JEPA training.

The cache is intentionally produced in a separate, labelled source step.  It
contains detector track-head features extracted at source GT boxes and is
never read by the target streaming tracker.  Target inference therefore does
not inherit this script's GT access.
"""

from __future__ import absolute_import, division, print_function

import argparse
import hashlib
import json
import os
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from darth.datasets import build_dataset
from darth.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--max-videos', type=int, default=0)
    parser.add_argument('--max-frames-per-video', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    output = os.path.abspath(args.output)
    if os.path.exists(output) and not args.overwrite:
        raise IOError('refusing to overwrite existing cache: {}'.format(
            output))
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    if cfg.get('test_cfg', False):
        model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    else:
        model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Keep deterministic ordering: dataset order is the source sequence order
    # used to reconstruct causal histories in the trainer.
    seen_videos = set()
    per_video_counts = {}
    embeddings = []
    boxes = []
    labels = []
    instance_ids = []
    frame_ids = []
    sequence_ids = []
    image_ids = []
    image_widths = []
    image_heights = []
    start = time.time()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    for index, info in enumerate(dataset.data_infos):
        video_id = int(info.get('video_id', info.get('id', index)))
        if video_id not in seen_videos:
            if args.max_videos and len(seen_videos) >= args.max_videos:
                break
            seen_videos.add(video_id)
            per_video_counts[video_id] = 0
        if (args.max_frames_per_video and
                per_video_counts[video_id] >= args.max_frames_per_video):
            continue
        if args.max_images and len(frame_ids) >= args.max_images:
            break

        ann = dataset.get_ann_info(info)
        if ann['bboxes'].shape[0] == 0:
            per_video_counts[video_id] += 1
            continue
        sample = dataset.prepare_test_img(index)
        image = sample['img'][0].unsqueeze(0).to(device)
        img_meta = sample['img_metas'][0].data
        scale_factor = torch.as_tensor(
            img_meta['scale_factor'], dtype=torch.float32, device=device)
        gt_boxes = torch.as_tensor(
            ann['bboxes'], dtype=torch.float32, device=device)
        scaled_boxes = gt_boxes * scale_factor
        with torch.no_grad():
            feats = model.detector.extract_feat(image)
            track_features = model.track_head.extract_bbox_feats(
                feats, [scaled_boxes])
        track_features = track_features.detach().cpu().float().numpy()

        n = int(gt_boxes.size(0))
        embeddings.append(track_features)
        boxes.append(np.asarray(ann['bboxes'], dtype=np.float32))
        labels.append(np.asarray(ann['labels'], dtype=np.int64))
        instance_ids.append(np.asarray(ann['instance_ids'], dtype=np.int64))
        frame_ids.append(np.full(n, int(info['frame_id']), dtype=np.int64))
        sequence_ids.append(np.full(n, video_id, dtype=np.int64))
        image_ids.append(np.full(n, int(info['id']), dtype=np.int64))
        image_widths.append(np.full(n, int(info['width']), dtype=np.int64))
        image_heights.append(np.full(n, int(info['height']), dtype=np.int64))
        per_video_counts[video_id] += 1

    if not embeddings:
        raise RuntimeError('no labelled source observations were cached')
    elapsed = time.time() - start
    arrays = dict(
        embeddings=np.concatenate(embeddings, axis=0),
        boxes=np.concatenate(boxes, axis=0),
        labels=np.concatenate(labels, axis=0),
        instance_ids=np.concatenate(instance_ids, axis=0),
        frame_ids=np.concatenate(frame_ids, axis=0),
        sequence_ids=np.concatenate(sequence_ids, axis=0),
        image_ids=np.concatenate(image_ids, axis=0),
        image_widths=np.concatenate(image_widths, axis=0),
        image_heights=np.concatenate(image_heights, axis=0))
    np.savez_compressed(output, **arrays)

    manifest = dict(
        cache=os.path.abspath(output),
        config=os.path.abspath(args.config),
        checkpoint=os.path.abspath(args.checkpoint),
        config_sha256=sha256(args.config),
        checkpoint_sha256=sha256(args.checkpoint),
        count=int(arrays['embeddings'].shape[0]),
        embedding_dim=int(arrays['embeddings'].shape[1]),
        sequence_ids=sorted(int(v) for v in set(arrays['sequence_ids'])),
        frames=int(len(set(zip(arrays['sequence_ids'], arrays['frame_ids'])))),
        elapsed_seconds=float(elapsed),
        peak_cuda_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == 'cuda' else 0),
        gt_used=True,
        future_used_for_target=True,
        target_gt_used_in_inference=False,
        purpose='source-labelled-offline-temporal-jepa-training-only',
        per_video_frames={str(k): int(v) for k, v in per_video_counts.items()},
    )
    manifest_path = output + '.manifest.json'
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
