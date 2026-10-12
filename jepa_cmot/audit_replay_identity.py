"""Offline identity audit of completed fixed-teacher graph replay snapshots."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from eval_damot_generaltrack import box_iou_matrix
from .audit_counterfactual import group
from .train_source import sha256


def audit_snapshot(sequence, manifest, detections, stage, control):
    protocol = json.loads((sequence / 'protocol.json').read_text())
    snapshots = json.loads((sequence / 'snapshots.json').read_text())
    if not protocol.get('actual_update_counts_matched_each_frame'):
        raise ValueError('This audit requires the registered fixed-teacher sequence')
    checkpoint = Path(snapshots[stage][control])
    payload = torch.load(checkpoint, map_location='cpu')
    order = protocol['order'][1:stage + 1]
    videos = [v for domain in order for v in manifest['videos']
              if v['dataset'] == protocol['dataset'] and v['domain'] == domain
              and v['split'] == 'adaptation']
    boundaries = np.cumsum([v['frames'] for v in videos])
    if int(boundaries[-1]) != payload['frames']:
        raise ValueError('Snapshot frame count differs from registered full videos')
    groups = defaultdict(list)
    for pool in ['memory', 'recent']:
        for index, example in enumerate(payload[pool]):
            if len(example) != 9:
                raise ValueError('Replay has no recorded physical frame/candidate keys')
            global_frame = int(example[7][0]) // (1 << 32)
            video_index = int(np.searchsorted(boundaries, global_frame, side='left'))
            start = int(boundaries[video_index - 1]) if video_index else 0
            local_frame = global_frame - start
            groups[video_index].append((pool, index, local_frame, example))
    cases = []
    feature_errors = []
    history_feature_errors = []
    history_box_errors = []
    mapping = [0, 2, 3, 4, 6, 7]
    for video_index, examples in groups.items():
        video = videos[video_index]
        directory = detections / video['domain'] / 'adaptation' / video['name']
        prepared = json.loads((directory / 'prepared/annotations/val.json').read_text())
        images = sorted(prepared['images'], key=lambda row: row['frame_id'])
        labels = json.loads(Path(video['labels']).read_text())
        gt_images = {i['id']: i for i in labels['images']}
        known = set(labels['evaluation_frame_ids'])
        annotations = defaultdict(list)
        for a in labels['annotations']:
            if not a.get('iscrowd'):
                annotations[gt_images[a['image_id']]['frame_id']].append(a)
        with np.load(directory / 'features_dense_current.npz', allow_pickle=False) as cache:
            features, offsets = cache['features'], cache['offsets']
            lookup = {int(value): i for i, value in enumerate(cache['image_ids'])}

            def identity(image, box, native_group):
                frame = image['frame_id']
                if frame not in known:
                    return None
                rows = [a for a in annotations[frame] if protocol['dataset'] == 'ua_detrac'
                        or group(mapping[a['category_id'] - 1]) == native_group]
                if not rows:
                    return None
                x1, y1, x2, y2 = box
                ious = box_iou_matrix([[x1, y1, x2-x1, y2-y1]], [a['bbox'] for a in rows])[0]
                best = int(np.argmax(ious))
                return int(rows[best]['instance_id']) if ious[best] >= .5 else None

            for pool, index, local_frame, example in examples:
                image = images[local_frame - 1]
                raw_index = int(example[7][0]) % (1 << 32) - 1
                positive = np.asarray(image['detection'][raw_index], np.float32)
                native_group = group(mapping[int(positive[5])])
                j = lookup[int(image['id'])]
                raw_feature = features[offsets[j] + raw_index].astype(np.float32)
                raw_feature /= max(float(np.linalg.norm(raw_feature)), 1e-8)
                error = float(np.max(np.abs(raw_feature - example[3][0].numpy())))
                feature_errors.append(error)
                if error > 2e-5:
                    raise ValueError('Physical key does not reproduce stored positive feature')
                past = []
                clipped = 0
                for past_feature, metadata, valid in zip(example[0].numpy(), example[1].numpy(), example[2].numpy()):
                    if not valid:
                        continue
                    gap = int(round(float(metadata[5]) * 128))
                    if gap == 128:
                        clipped += 1
                        continue
                    if not 0 < gap < local_frame:
                        raise ValueError('Replay history is not strictly within the same past video')
                    cx, cy, width, height = metadata[:4]
                    box = np.asarray([cx-width/2, cy-height/2, cx+width/2, cy+height/2])
                    box *= [image['width'], image['height'], image['width'], image['height']]
                    past_image = images[local_frame-gap-1]
                    past_rows = np.asarray(past_image['detection'], np.float32).reshape(-1, 6)
                    best = int(np.argmin(np.linalg.norm(past_rows[:, :4] - box, axis=1)))
                    box_error = float(np.max(np.abs(past_rows[best, :4] - box)))
                    history_box_errors.append(box_error)
                    past_j = lookup[int(past_image['id'])]
                    raw_past = features[offsets[past_j] + best].astype(np.float32)
                    raw_past /= max(float(np.linalg.norm(raw_past)), 1e-8)
                    past_error = float(np.max(np.abs(raw_past - past_feature)))
                    history_feature_errors.append(past_error)
                    if box_error > .01 or past_error > 2e-5:
                        raise ValueError('Recovered past frame/box does not reproduce stored history feature')
                    past.append(identity(past_image, box, native_group))
                matched = [v for v in past if v is not None]
                positive_identity = identity(image, positive[:4], native_group)
                dominant = None
                purity = None
                if matched:
                    dominant, count = Counter(matched).most_common(1)[0]
                    purity = count / len(matched)
                if not matched:
                    status = 'unmatched_history'
                elif purity < .75:
                    status = 'mixed_identity_history'
                elif positive_identity is None:
                    status = 'unmatched_current_detection'
                elif positive_identity == dominant:
                    status = 'identity_consistent'
                else:
                    status = 'confirmed_identity_mismatch'
                query = example[6].numpy()
                positive_feature = example[3][0].numpy()
                cases.append(dict(pool=pool, index=index, video=video['name'], domain=video['domain'],
                    frame=image['frame_id'], status=status, history_identity_coverage=len(matched),
                    history_slots=int(example[2].sum()), clipped_history_slots=clipped,
                    history_purity=purity, dominant_history_GT_id=dominant,
                    positive_GT_id=positive_identity, prediction_need=1-float(query @ positive_feature),
                    appearance_hardness=1-float(example[0][-1].numpy() @ positive_feature)))
    counts = {pool: dict(Counter(c['status'] for c in cases if c['pool'] == pool))
              for pool in ['memory', 'recent']}
    discrepancies = {}
    for status in {c['status'] for c in cases}:
        rows = [c for c in cases if c['status'] == status]
        discrepancies[status] = {k: float(np.mean([c[k] for c in rows]))
                                for k in ['prediction_need', 'appearance_hardness']}
    matched_controls = {}
    for other_control, other_path in snapshots[stage].items():
        other = torch.load(other_path, map_location='cpu')
        equal = all(other[key] == payload[key] for key in ['frames', 'updates', 'seen'])
        for pool in ['memory', 'recent']:
            equal = equal and len(other[pool]) == len(payload[pool])
            equal = equal and all(torch.equal(a[k], b[k]) for a, b in zip(payload[pool], other[pool])
                                  for k in [0, 1, 2, 3, 4, 7, 8])
        if not equal:
            raise ValueError('Fixed-teacher retained samples differ across graph controls')
        matched_controls[other_control] = dict(sample_fields_equal=True, updates=other['updates'],
                                              checkpoint_sha256=sha256(other_path))
    return dict(stage=stage, checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        native_frames=payload['frames'], updates=payload['updates'],
        all_delayed_confirmed=payload['stats']['delayed_confirmed'], counts=counts,
        maximum_positive_feature_reconstruction_error=max(feature_errors, default=0),
        reconstructed_history_observations=len(history_feature_errors),
        maximum_history_feature_reconstruction_error=max(history_feature_errors, default=0),
        maximum_history_box_reconstruction_error_pixels=max(history_box_errors, default=0),
        matched_controls=matched_controls,
        discrepancy_by_status=discrepancies, cases=cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--control', default='predictive_ranking_none_graph')
    parser.add_argument('--stages', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = dict(scope='offline completed-domain reservoir and remaining fresh buffer identity audit',
        target_GT_fed_to_training_or_gates=False,
        definition='nearest IoU05 group-compatible GT; dominant matched-history identity purity >=.75',
        limitation='not all proposed/confirmed pseudo pairs or frequency-weighted training batches; '
                   'unmatched boxes are separate from confirmed identity mismatches; '
                   'clipped historical gaps cannot be recovered and are excluded',
        sequence_protocol_sha256=hashlib.sha256((args.sequence/'protocol.json').read_bytes()).hexdigest(),
        snapshots=[audit_snapshot(args.sequence, manifest, args.detections, stage, args.control)
                   for stage in args.stages])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('REPLAY_IDENTITY_AUDIT_COMPLETE', [(s['stage'], s['counts']) for s in result['snapshots']], flush=True)


if __name__ == '__main__':
    main()
