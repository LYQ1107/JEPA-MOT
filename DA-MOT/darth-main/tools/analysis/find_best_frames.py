#!/usr/bin/env python3
"""Find frames where ours beats baseline most (vs GT), and render them.

Per-frame identity-consistency score:
  1. Match predictions to GT per frame (Hungarian, IoU>thr, same category).
  2. For each method, compute each GT track's dominant predicted id
     (majority vote across the whole video).
  3. Per frame, count GT boxes whose matched pred id == that GT track's
     dominant id (i.e. identity is consistent). Also count matched GT boxes.
  4. Frame advantage = (ours_consistent - baseline_consistent).

Renders GT | Baseline | Ours panels for the top-advantage frames.
No video output; images only.
"""
import argparse
import colorsys
import json
import os
from collections import defaultdict, Counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ann-file', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--baseline-json', required=True)
    p.add_argument('--ours-json', required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--score-thr', type=float, default=0.2)
    p.add_argument('--iou-thr', type=float, default=0.5)
    p.add_argument('--top-k', type=int, default=6)
    p.add_argument('--output-dir', required=True)
    return p.parse_args()


def color_for_id(track_id):
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_gt(ann_file, sequence):
    data = json.load(open(ann_file))
    vid = next(v for v in data['videos'] if v['name'] == sequence)
    imgs = [im for im in data['images'] if im['video_id'] == vid['id']]
    imgs.sort(key=lambda x: x['frame_id'])
    gt_by_img = defaultdict(list)  # image_id -> [(inst_id, bbox, cat)]
    for a in data['annotations']:
        if a['video_id'] != vid['id']:
            continue
        gt_by_img[a['image_id']].append(
            (a['instance_id'], a['bbox'], a['category_id']))
    return vid, imgs, gt_by_img


def load_pred(track_json, video_id, score_thr):
    dets = json.load(open(track_json))
    by_img = defaultdict(list)  # image_id -> [(track_id, bbox, cat)]
    for d in dets:
        if d['video_id'] != video_id:
            continue
        if d.get('score', 1.0) < score_thr:
            continue
        by_img[d['image_id']].append(
            (d['track_id'], d['bbox'], d['category_id']))
    return by_img


def match_frame(gt_list, pred_list, iou_thr):
    """Return dict gt_inst_id -> pred_track_id for matched pairs."""
    if not gt_list or not pred_list:
        return {}
    cost = np.ones((len(gt_list), len(pred_list)), dtype=np.float32)
    for i, (_, gb, gc) in enumerate(gt_list):
        for j, (_, pb, pc) in enumerate(pred_list):
            if gc != pc:
                continue
            iou = iou_xywh(gb, pb)
            if iou > iou_thr:
                cost[i, j] = 1.0 - iou
    ri, ci = linear_sum_assignment(cost)
    out = {}
    for i, j in zip(ri, ci):
        if cost[i, j] < 1.0:
            out[gt_list[i][0]] = pred_list[j][0]
    return out


def dominant_ids(imgs, gt_by_img, pred_by_img, iou_thr):
    """gt_inst_id -> dominant predicted track id across video."""
    votes = defaultdict(Counter)
    per_frame_match = {}
    for im in imgs:
        m = match_frame(gt_by_img.get(im['id'], []),
                        pred_by_img.get(im['id'], []), iou_thr)
        per_frame_match[im['id']] = m
        for gt_inst, pred_tid in m.items():
            votes[gt_inst][pred_tid] += 1
    dominant = {g: c.most_common(1)[0][0] for g, c in votes.items()}
    return dominant, per_frame_match


def draw(img, items, banner_text, banner_color=(40, 40, 40)):
    for tid, bbox in items:
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = color_for_id(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'ID:{tid}'
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - bl - 2)
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 2, ty1 + th + bl + 2), color, -1)
        cv2.putText(img, label, (x1 + 1, ty1 + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    h, w = img.shape[:2]
    banner = np.full((30, w, 3), banner_color, dtype=np.uint8)
    cv2.putText(banner, banner_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([banner, img])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seq = args.sequence

    vid, imgs, gt_by_img = load_gt(args.ann_file, seq)
    W, H = vid['width'], vid['height']
    base_by_img = load_pred(args.baseline_json, vid['id'], args.score_thr)
    ours_by_img = load_pred(args.ours_json, vid['id'], args.score_thr)

    base_dom, base_match = dominant_ids(imgs, gt_by_img, base_by_img, args.iou_thr)
    ours_dom, ours_match = dominant_ids(imgs, gt_by_img, ours_by_img, args.iou_thr)

    # per-frame consistency advantage
    rows = []
    for idx, im in enumerate(imgs, start=1):
        gt_list = gt_by_img.get(im['id'], [])
        bm = base_match[im['id']]
        om = ours_match[im['id']]
        b_cons = sum(1 for g, p in bm.items() if base_dom.get(g) == p)
        o_cons = sum(1 for g, p in om.items() if ours_dom.get(g) == p)
        rows.append({
            'idx': idx, 'image_id': im['id'], 'file_name': im['file_name'],
            'n_gt': len(gt_list), 'base_matched': len(bm), 'ours_matched': len(om),
            'base_consistent': b_cons, 'ours_consistent': o_cons,
            'advantage': o_cons - b_cons,
        })

    # rank by advantage, then by ours_consistent
    ranked = sorted(rows, key=lambda r: (r['advantage'], r['ours_consistent']),
                    reverse=True)
    top = ranked[:args.top_k]

    # write per-frame csv
    csv_path = os.path.join(args.output_dir, f'{seq}_per_frame_advantage.csv')
    with open(csv_path, 'w') as f:
        f.write('frame,image_id,file_name,n_gt,base_matched,ours_matched,'
                'base_consistent,ours_consistent,advantage\n')
        for r in rows:
            f.write(f"{r['idx']},{r['image_id']},{r['file_name']},{r['n_gt']},"
                    f"{r['base_matched']},{r['ours_matched']},"
                    f"{r['base_consistent']},{r['ours_consistent']},{r['advantage']}\n")

    best_dir = os.path.join(args.output_dir, f'{seq}_best_frames')
    os.makedirs(best_dir, exist_ok=True)

    sep = np.full((H + 30, 10, 3), 255, dtype=np.uint8)
    for rank, r in enumerate(top, start=1):
        im_id = r['image_id']
        frame = cv2.imread(os.path.join(args.image_root, r['file_name']))
        if frame is None:
            continue
        gt_items = [(g, b) for g, b, c in gt_by_img.get(im_id, [])]
        base_items = [(t, b) for t, b, c in base_by_img.get(im_id, [])]
        ours_items = [(t, b) for t, b, c in ours_by_img.get(im_id, [])]

        g_panel = draw(frame.copy(), gt_items,
                       f'GT  {seq}  frame {r["idx"]}', (80, 80, 80))
        b_panel = draw(frame.copy(), base_items,
                       f'Baseline  cons={r["base_consistent"]}', (30, 30, 120))
        o_panel = draw(frame.copy(), ours_items,
                       f'Ours  cons={r["ours_consistent"]}', (30, 110, 30))

        combo = np.hstack([g_panel, sep, b_panel, sep, o_panel])
        out_path = os.path.join(
            best_dir, f'rank{rank:02d}_frame{r["idx"]:04d}_adv{r["advantage"]}.jpg')
        cv2.imwrite(out_path, combo)
        print(f'rank{rank} frame {r["idx"]} advantage={r["advantage"]} '
              f'base_cons={r["base_consistent"]} ours_cons={r["ours_consistent"]} '
              f'-> {out_path}')

    print('\nper-frame csv:', csv_path)
    print('best frames dir:', best_dir)


if __name__ == '__main__':
    main()
