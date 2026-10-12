#!/usr/bin/env python3
"""Select the best 8-frame GT-aligned window per domain and render ours only."""
import argparse
import colorsys
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main')

DOMAINS = [
    # dataset, domain, annotation, image root, ours track result
    ('bdd_da', 'overcast', 'data/bdd_da/annotations/box_track_20/box_track_overcast_all_cocofmt.json',
     '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast',
     'work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e/forgetting_eval/stage01_after_overcast/epoch_5/overcast/qdtrack_faster-rcnn_r50_fpn_20e_bdd_overcast/20260708_032738/track_coco.json'),
    ('bdd_da', 'rainy', 'data/bdd_da/annotations/box_track_20/box_track_rainy_all_cocofmt.json',
     '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_rainy',
     'work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e/forgetting_eval/stage02_after_rainy/epoch_5/rainy/qdtrack_faster-rcnn_r50_fpn_20e_bdd_rainy/20260708_040453/track_coco.json'),
    ('bdd_da', 'dawn_dusk', 'data/bdd_da/annotations/box_track_20/box_track_dawn_dusk_all_cocofmt.json',
     '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_dawn_dusk',
     'work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e/forgetting_eval/stage03_after_dawn_dusk/epoch_5/dawn_dusk/qdtrack_faster-rcnn_r50_fpn_20e_bdd_dawn_dusk/20260708_044008/track_coco.json'),
    ('bdd_da', 'night', 'data/bdd_da/annotations/box_track_20/box_track_night_all_cocofmt.json',
     '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_night',
     'work_dirs/bdd_4domain/tri_module_pca_variants/spaP1_causalProtoShrink_roiV1Weak_4domain_5e/forgetting_eval/stage04_after_night/epoch_5/night/qdtrack_faster-rcnn_r50_fpn_20e_bdd_night/20260708_051808/track_coco.json'),
    ('ua_detrac', 'cloudy', 'data/ua_detrac/annotations/box_track_cloudy_all_cocofmt.json', 'data/ua_detrac/images/cloudy',
     'work_dirs/ua_detrac/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage01_after_cloudy/epoch_5/cloudy/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_cloudy_eval/20260709_105947/track_coco.json'),
    ('ua_detrac', 'rainy', 'data/ua_detrac/annotations/box_track_rainy_all_cocofmt.json', 'data/ua_detrac/images/rainy',
     'work_dirs/ua_detrac/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage02_after_rainy/epoch_5/rainy/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_rainy_eval/20260709_110429/track_coco.json'),
    ('ua_detrac', 'night', 'data/ua_detrac/annotations/box_track_night_all_cocofmt.json', 'data/ua_detrac/images/night',
     'work_dirs/ua_detrac/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage03_after_night/epoch_5/night/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_night_eval/20260709_110840/track_coco.json'),
    ('motsynth', 'cloudy', 'data/motsynth/annotations/motsynth_cloudy_cocofmt.json', 'data/motsynth/images',
     'work_dirs/motsynth/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage01_after_cloudy/epoch_5/cloudy/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_cloudy_eval/20260709_051616/track_coco.json'),
    ('motsynth', 'fog', 'data/motsynth/annotations/motsynth_fog_cocofmt.json', 'data/motsynth/images',
     'work_dirs/motsynth/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage02_after_fog/epoch_5/fog/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_fog_eval/20260709_052636/track_coco.json'),
    ('motsynth', 'rain', 'data/motsynth/annotations/motsynth_rain_cocofmt.json', 'data/motsynth/images',
     'work_dirs/motsynth/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage03_after_rain/epoch_5/rain/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_rain_eval/20260709_053558/track_coco.json'),
    ('motsynth', 'snow', 'data/motsynth/annotations/motsynth_snow_cocofmt.json', 'data/motsynth/images',
     'work_dirs/motsynth/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage04_after_snow/epoch_5/snow/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_snow_eval/20260709_054430/track_coco.json'),
    ('motsynth', 'dark', 'data/motsynth/annotations/motsynth_dark_cocofmt.json', 'data/motsynth/images',
     'work_dirs/motsynth/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage05_after_dark/epoch_5/dark/qdtrack_faster-rcnn_r50_fpn_20e_motsynth_dark_eval/20260709_055202/track_coco.json'),
    ('vkitti2', 'overcast', 'data/vkitti2/annotations/vkitti2_overcast_cocofmt.json', 'data/vkitti2/images',
     'work_dirs/vkitti2/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage01_after_overcast/epoch_5/overcast/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_overcast_eval/20260709_150026/track_coco.json'),
    ('vkitti2', 'morning', 'data/vkitti2/annotations/vkitti2_morning_cocofmt.json', 'data/vkitti2/images',
     'work_dirs/vkitti2/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage02_after_morning/epoch_5/morning/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_morning_eval/20260709_150308/track_coco.json'),
    ('vkitti2', 'rain', 'data/vkitti2/annotations/vkitti2_rain_cocofmt.json', 'data/vkitti2/images',
     'work_dirs/vkitti2/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage03_after_rain/epoch_5/rain/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_rain_eval/20260709_150545/track_coco.json'),
    ('vkitti2', 'fog', 'data/vkitti2/annotations/vkitti2_fog_cocofmt.json', 'data/vkitti2/images',
     'work_dirs/vkitti2/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage04_after_fog/epoch_5/fog/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_fog_eval/20260709_150821/track_coco.json'),
    ('vkitti2', 'sunset', 'data/vkitti2/annotations/vkitti2_sunset_cocofmt.json', 'data/vkitti2/images',
     'work_dirs/vkitti2/test/trimodule_spaP1_causalProtoShrink_roiV1Weak/stage05_after_sunset/epoch_5/sunset/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_sunset_eval/20260709_151058/track_coco.json'),
]


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def frame_quality(gt, pred, thr):
    if not gt:
        return 0.0, 0, 0, 0.0
    if not pred:
        return 0.0, 0, 0, 0.0
    cost = np.ones((len(gt), len(pred)), dtype=np.float32)
    valid_iou = {}
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if g['category_id'] != p['category_id']:
                continue
            v = iou(g['bbox'], p['bbox'])
            if v >= thr:
                cost[i, j] = 1 - v
                valid_iou[(i, j)] = v
    r, c = linear_sum_assignment(cost)
    vals = [valid_iou[(i, j)] for i, j in zip(r, c) if (i, j) in valid_iou]
    matched = len(vals)
    recall = matched / len(gt)
    precision = matched / len(pred)
    miou = sum(vals) / matched if matched else 0.0
    # prioritize GT coverage + localization; penalize false-positive clutter
    return recall * miou * (0.6 + 0.4 * precision), matched, len(pred), miou


def color_for_id(tid):
    r, g, b = colorsys.hsv_to_rgb((tid * 0.61803398875) % 1.0, 0.84, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def render(image, preds):
    for p in preds:
        x, y, w, h = map(int, p['bbox'])
        color = color_for_id(p['track_id'])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"ID:{p['track_id']}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        top = max(0, y - th - base - 3)
        cv2.rectangle(image, (x, top), (x + tw + 4, top + th + base + 3), color, -1)
        cv2.putText(image, label, (x + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def process(entry, output_root, window, interval, min_gt, score_thr, iou_thr):
    dataset, domain, ann_rel, image_root, track_rel = entry
    ann_path, track_path = ROOT / ann_rel, ROOT / track_rel
    if not ann_path.exists() or not track_path.exists():
        raise FileNotFoundError(f'{ann_path} or {track_path}')
    ann = json.load(open(ann_path))
    pred_raw = json.load(open(track_path))
    images = {x['id']: x for x in ann['images']}
    gt_by_image = defaultdict(list)
    for x in ann['annotations']:
        gt_by_image[x['image_id']].append(x)
    pred_by_image = defaultdict(list)
    for x in pred_raw:
        if x.get('score', 1.0) >= score_thr:
            pred_by_image[x['image_id']].append(x)
    videos = {v['id']: v for v in ann['videos']}
    by_video = defaultdict(list)
    for x in ann['images']:
        by_video[x['video_id']].append(x)
    candidates = []
    for video_id, ims in by_video.items():
        ims.sort(key=lambda x: x['frame_id'])
        stats = []
        for im in ims:
            q, m, n, miou = frame_quality(gt_by_image[im['id']], pred_by_image[im['id']], iou_thr)
            stats.append((im, q, m, len(gt_by_image[im['id']]), n, miou))
        span = 1 + (window - 1) * interval
        for start in range(0, len(stats) - span + 1):
            chunk = stats[start:start + span:interval]
            avg_q = sum(s[1] for s in chunk) / window
            avg_m = sum(s[2] for s in chunk) / window
            avg_iou = sum(s[5] for s in chunk) / window
            # Every output image must show at least three GT objects.
            if min(s[3] for s in chunk) >= min_gt and min(s[2] for s in chunk) >= 1:
                candidates.append((avg_q, avg_m, avg_iou, video_id, chunk))
    if not candidates:
        raise RuntimeError(f'No eligible window for {dataset}/{domain}')
    avg_q, avg_m, avg_iou, video_id, chosen = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    seq = videos[video_id].get('name', str(video_id))
    out_dir = output_root / dataset / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for rank, (im, q, m, gt_n, pred_n, miou) in enumerate(chosen, 1):
        file_path = Path(image_root) / im['file_name']
        if not file_path.is_absolute():
            file_path = ROOT / file_path
        canvas = cv2.imread(str(file_path))
        if canvas is None:
            raise FileNotFoundError(file_path)
        canvas = render(canvas, pred_by_image[im['id']])
        out_name = f'{rank:02d}_frame{im["frame_id"] + 1:05d}.jpg'
        cv2.imwrite(str(out_dir / out_name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 96])
        records.append(dict(dataset=dataset, domain=domain, sequence=seq, rank=rank,
                            frame_id=im['frame_id'], file_name=im['file_name'],
                            image_path=str(file_path), quality=round(q, 6), matched=m,
                            gt_boxes=gt_n, pred_boxes=pred_n, mean_iou=round(miou, 6),
                            output=str(out_dir / out_name)))
    return records, dict(dataset=dataset, domain=domain, sequence=seq,
                         video_id=video_id, start_frame=chosen[0][0]['frame_id'],
                         end_frame=chosen[-1][0]['frame_id'], mean_quality=round(avg_q, 6),
                         mean_matches=round(avg_m, 3), mean_iou=round(avg_iou, 6),
                         track_json=str(track_path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='work_dirs/visualization/all_datasets_ours_best')
    p.add_argument('--window', type=int, default=12)
    p.add_argument('--interval', type=int, default=2,
                   help='Frame spacing between exported images.')
    p.add_argument('--min-gt', type=int, default=3,
                   help='Minimum GT targets required in every exported image.')
    p.add_argument('--score-thr', type=float, default=0.2)
    p.add_argument('--iou-thr', type=float, default=0.5)
    args = p.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    all_frames, summaries = [], []
    for entry in DOMAINS:
        print(f'Processing {entry[0]}/{entry[1]} ...', flush=True)
        rows, summary = process(entry, out, args.window, args.interval, args.min_gt,
                                args.score_thr, args.iou_thr)
        all_frames.extend(rows); summaries.append(summary)
        print(f'  {summary["sequence"]} frames {summary["start_frame"]}-{summary["end_frame"]}; quality={summary["mean_quality"]}', flush=True)
    with open(out / 'selection_summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    with open(out / 'selected_frames.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_frames[0])); w.writeheader(); w.writerows(all_frames)
    with open(out / 'README.txt', 'w') as f:
        f.write('Only ours is rendered. Frames are selected using GT-only matching.\n')
        f.write('Score = matched-GT recall × mean IoU × (0.6 + 0.4 × prediction precision).\n')
        f.write('The highest-scoring consecutive 8-frame window is selected for each non-source domain.\n')
    print(f'Completed {len(summaries)} domains / {len(all_frames)} images: {out}')


if __name__ == '__main__':
    main()
