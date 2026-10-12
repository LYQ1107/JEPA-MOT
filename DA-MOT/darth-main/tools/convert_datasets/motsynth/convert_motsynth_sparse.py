#!/usr/bin/env python3
"""Generate stride-sampled (sparse) MOTSynth COCO-video JSONs.

Reuses existing symlinks from convert_motsynth_to_cocofmt.py.
Only generates new annotation JSONs with every STRIDE-th frame kept.
gt.txt frames are 1-based; we keep frames where (frame - 1) % STRIDE == 0.
frame_id in output is reassigned as contiguous 0-based index within the video.

Usage:
  python convert_motsynth_sparse.py [--stride 4] [--dry-run]
"""

import configparser
import json
import os
import os.path as osp
import sys

MOTSYNTH_BASE = '/data3/testdata/vranlee/.MOTSynth.partial'
OUT_ROOT      = '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/motsynth'
OUT_ANN_ROOT  = osp.join(OUT_ROOT, 'annotations')

# exclude indoor and all; only these 6 domains
DOMAINS = [
    'MOTSynth_Cloudy',
    'MOTSynth_Dark',
    'MOTSynth_Fog',
    'MOTSynth_Normal',
    'MOTSynth_Rain',
    'MOTSynth_Snow',
]

CATEGORIES = [
    {'id': 1, 'name': 'pedestrian'},
    {'id': 2, 'name': 'car'},
    {'id': 3, 'name': 'truck'},
    {'id': 4, 'name': 'bus'},
    {'id': 5, 'name': 'motorcycle'},
    {'id': 6, 'name': 'bicycle'},
]


def read_seqinfo(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    s = cfg['Sequence']
    return {
        'name':       s.get('name', ''),
        'fps':        int(s.get('frameRate', 20)),
        'width':      int(s.get('imWidth',  1920)),
        'height':     int(s.get('imHeight', 1080)),
        'seq_length': int(s.get('seqLength', 0)),
        'im_ext':     s.get('imExt', '.jpg'),
    }


def load_gt(gt_path):
    """Return {frame(1-based): [(track_id, bbox, vis), ...]}"""
    from collections import defaultdict
    by_frame = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 9:
                continue
            conf = int(parts[6])
            cls  = int(parts[7])
            if conf != 1 or cls != 1:
                continue
            frame    = int(parts[0])
            track_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w  = float(parts[4])
            h  = float(parts[5])
            vis = float(parts[8])
            if w <= 0 or h <= 0:
                continue
            by_frame[frame].append((track_id, [x1, y1, w, h], vis))
    return by_frame


def build_domain_json(domain_name, stride, dry_run=False):
    train_dir = osp.join(MOTSYNTH_BASE, domain_name, 'train')
    short     = domain_name.replace('MOTSynth_', '').lower()

    seqs = sorted(s for s in os.listdir(train_dir)
                  if osp.isdir(osp.join(train_dir, s)))

    videos, images, annotations = [], [], []
    vid_id = img_id = ann_id = 1

    for seq in seqs:
        seq_dir   = osp.join(train_dir, seq)
        gt_path   = osp.join(seq_dir, 'gt', 'gt.txt')
        info_path = osp.join(seq_dir, 'seqinfo.ini')

        if not osp.exists(gt_path):
            continue

        info = read_seqinfo(info_path) if osp.exists(info_path) else {
            'name': seq, 'fps': 20, 'width': 1920, 'height': 1080,
            'seq_length': 0, 'im_ext': '.jpg'}

        seq_len = info['seq_length']
        ext     = info['im_ext']
        if seq_len == 0:
            continue

        # kept 1-based frame numbers: 1, 1+stride, 1+2*stride, ...
        kept_frames_1based = list(range(1, seq_len + 1, stride))

        gt_by_frame = load_gt(gt_path)

        # map 1-based gt frame → new contiguous frame_id (0-based)
        frame1_to_fidx = {f1: fidx for fidx, f1 in enumerate(kept_frames_1based)}

        # effective fps after striding
        eff_fps = max(1, round(info['fps'] / stride))

        for fidx, f1 in enumerate(kept_frames_1based):
            fname = f'{f1 - 1:04d}{ext}'   # 0-based filename
            iid   = img_id
            img_id += 1
            images.append({
                'id':        iid,
                'file_name': f'{short}/{seq}/{fname}',
                'height':    info['height'],
                'width':     info['width'],
                'frame_id':  fidx,
                'video_id':  vid_id,
            })
            for (track_id, bbox, vis) in gt_by_frame.get(f1, []):
                x1, y1, w, h = bbox
                x1 = max(0.0, x1)
                y1 = max(0.0, y1)
                w  = min(w,  info['width']  - x1)
                h  = min(h,  info['height'] - y1)
                if w <= 0 or h <= 0:
                    continue
                annotations.append({
                    'id':          ann_id,
                    'image_id':    iid,
                    'video_id':    vid_id,
                    'category_id': 1,
                    'instance_id': track_id,
                    'bbox':        [round(x1,2), round(y1,2), round(w,2), round(h,2)],
                    'area':        round(w * h, 2),
                    'iscrowd':     0,
                    'visibility':  vis,
                })
                ann_id += 1

        videos.append({
            'id':     vid_id,
            'name':   f'{short}/{seq}',
            'fps':    eff_fps,
            'width':  info['width'],
            'height': info['height'],
        })
        vid_id += 1

    return {'videos': videos, 'images': images,
            'annotations': annotations, 'categories': CATEGORIES}


def main():
    stride  = 4
    dry_run = False
    for a in sys.argv[1:]:
        if a.startswith('--stride='):
            stride = int(a.split('=')[1])
        elif a == '--dry-run':
            dry_run = True

    os.makedirs(OUT_ANN_ROOT, exist_ok=True)

    for domain in DOMAINS:
        print(f'[{domain}] stride={stride} ...')
        coco = build_domain_json(domain, stride=stride, dry_run=dry_run)
        short = domain.replace('MOTSynth_', '').lower()
        out_path = osp.join(OUT_ANN_ROOT,
                            f'motsynth_{short}_s{stride}_cocofmt.json')
        if not dry_run:
            with open(out_path, 'w') as f:
                json.dump(coco, f)
        print(f'  → {out_path}  '
              f'videos={len(coco["videos"])} '
              f'images={len(coco["images"])} '
              f'anns={len(coco["annotations"])}')

    print('[DONE]')


if __name__ == '__main__':
    main()
