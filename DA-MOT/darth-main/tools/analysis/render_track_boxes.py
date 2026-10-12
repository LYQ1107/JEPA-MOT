#!/usr/bin/env python3
"""Render tracking boxes onto original frames.

Each track_id gets a unique, stable color. Only the ID is drawn (e.g. "ID:10"),
no category. Produces per-method videos, a side-by-side comparison video, and
a few sample frames for baseline vs ours.
"""
import argparse
import colorsys
import json
import os
from collections import defaultdict

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ann-file', required=True)
    p.add_argument('--image-root', required=True)
    p.add_argument('--baseline-json', required=True)
    p.add_argument('--ours-json', required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--score-thr', type=float, default=0.2)
    p.add_argument('--fps', type=int, default=25)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--sample-frames', type=int, nargs='+',
                   default=[1, 80, 160, 240, 320])
    return p.parse_args()


def color_for_id(track_id):
    """Stable vivid BGR color from track_id via golden-ratio hue hashing."""
    h = (track_id * 0.61803398875) % 1.0
    s = 0.85
    v = 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))


def load_video_meta(ann_file, sequence):
    data = json.load(open(ann_file))
    vid = None
    for v in data['videos']:
        if v['name'] == sequence:
            vid = v
            break
    if vid is None:
        raise ValueError(f'sequence {sequence} not found')
    imgs = [im for im in data['images'] if im['video_id'] == vid['id']]
    imgs.sort(key=lambda x: x['frame_id'])
    # image_id -> (file_name, frame_id)
    id2img = {im['id']: im for im in imgs}
    return vid, imgs, id2img


def build_frame_dets(track_json, video_id, score_thr):
    """image_id -> list of (track_id, bbox_xywh)."""
    dets = json.load(open(track_json))
    frame_dets = defaultdict(list)
    for d in dets:
        if d['video_id'] != video_id:
            continue
        if d.get('score', 1.0) < score_thr:
            continue
        frame_dets[d['image_id']].append((d['track_id'], d['bbox']))
    return frame_dets


def draw_boxes(img, det_list):
    for track_id, bbox in det_list:
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = color_for_id(track_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f'ID:{track_id}'
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - bl - 2)
        cv2.rectangle(img, (x1, ty1), (x1 + tw + 2, ty1 + th + bl + 2), color, -1)
        cv2.putText(img, label, (x1 + 1, ty1 + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return img


def add_banner(img, text):
    h, w = img.shape[:2]
    banner = np.zeros((30, w, 3), dtype=np.uint8)
    cv2.putText(banner, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([banner, img])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seq = args.sequence

    vid, imgs, id2img = load_video_meta(args.ann_file, seq)
    video_id = vid['id']
    W, H = vid['width'], vid['height']

    base_dets = build_frame_dets(args.baseline_json, video_id, args.score_thr)
    ours_dets = build_frame_dets(args.ours_json, video_id, args.score_thr)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    base_vid_path = os.path.join(args.output_dir, f'{seq}_baseline_boxes.mp4')
    ours_vid_path = os.path.join(args.output_dir, f'{seq}_ours_boxes.mp4')
    cmp_vid_path = os.path.join(args.output_dir, f'{seq}_compare_boxes.mp4')

    vw_base = cv2.VideoWriter(base_vid_path, fourcc, args.fps, (W, H + 30))
    vw_ours = cv2.VideoWriter(ours_vid_path, fourcc, args.fps, (W, H + 30))
    vw_cmp = cv2.VideoWriter(cmp_vid_path, fourcc, args.fps, (W * 2 + 10, H + 30))

    sample_set = set(args.sample_frames)
    sample_dir = os.path.join(args.output_dir, f'{seq}_sample_frames')
    os.makedirs(sample_dir, exist_ok=True)

    for idx, im in enumerate(imgs, start=1):
        fpath = os.path.join(args.image_root, im['file_name'])
        frame = cv2.imread(fpath)
        if frame is None:
            continue

        b_frame = draw_boxes(frame.copy(), base_dets.get(im['id'], []))
        o_frame = draw_boxes(frame.copy(), ours_dets.get(im['id'], []))

        b_banner = add_banner(b_frame, f'Baseline  {seq}  frame {idx}')
        o_banner = add_banner(o_frame, f'Ours (DAMOT)  {seq}  frame {idx}')

        vw_base.write(b_banner)
        vw_ours.write(o_banner)

        sep = np.full((H + 30, 10, 3), 255, dtype=np.uint8)
        vw_cmp.write(np.hstack([b_banner, sep, o_banner]))

        if idx in sample_set:
            cv2.imwrite(os.path.join(sample_dir, f'frame{idx:04d}_baseline.jpg'),
                        b_banner)
            cv2.imwrite(os.path.join(sample_dir, f'frame{idx:04d}_ours.jpg'),
                        o_banner)
            cv2.imwrite(os.path.join(sample_dir, f'frame{idx:04d}_compare.jpg'),
                        np.hstack([b_banner, sep, o_banner]))

    vw_base.release()
    vw_ours.release()
    vw_cmp.release()

    print('baseline video:', base_vid_path)
    print('ours video:', ours_vid_path)
    print('compare video:', cmp_vid_path)
    print('sample frames dir:', sample_dir)
    print('baseline tracks:', len(set(t for dl in base_dets.values() for t, _ in dl)))
    print('ours tracks:', len(set(t for dl in ours_dets.values() for t, _ in dl)))


if __name__ == '__main__':
    main()
