#!/usr/bin/env python3
"""Convert MOTSynth weather-domain splits to COCO-video tracking format (BDD_DA style).

gt.txt format (MOT Challenge):
  frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility, ...
  - bbox origin: TOP-LEFT corner  →  COCO [x1, y1, w, h] = [bb_left, bb_top, bb_width, bb_height]
  - conf=1: valid annotation  (conf=0 = ignore region)
  - class=1: pedestrian (only class we keep)

Output JSON schema matches BDD_DA box_track_*_cocofmt.json:
  {videos, images, annotations, categories}
  image.file_name:  <seq_name>/0000.jpg
  video.name:       <seq_name>
  ann.bbox:         [x1, y1, w, h]
  ann.instance_id:  track id
"""

import configparser
import json
import os
import os.path as osp
import sys

MOTSYNTH_BASE = '/data3/testdata/vranlee/.MOTSynth.partial'
OUT_ROOT      = '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/motsynth'
OUT_IMG_ROOT  = osp.join(OUT_ROOT, 'images')
OUT_ANN_ROOT  = osp.join(OUT_ROOT, 'annotations')

DOMAINS = [
    'MOTSynth_Cloudy',
    'MOTSynth_Dark',
    'MOTSynth_Fog',
    'MOTSynth_Indoor',
    'MOTSynth_Normal',
    'MOTSynth_Rain',
    'MOTSynth_Snow',
]

# Only pedestrian kept (class=1 in MOT Challenge)
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
        'name':       s.get('name', osp.basename(osp.dirname(osp.dirname(path)))),
        'fps':        int(s.get('frameRate', 20)),
        'width':      int(s.get('imWidth',  1920)),
        'height':     int(s.get('imHeight', 1080)),
        'seq_length': int(s.get('seqLength', 0)),
        'im_ext':     s.get('imExt', '.jpg'),
    }


def load_gt(gt_path):
    """Return list of dicts: frame(1-based), track_id, bbox=[x1,y1,w,h], vis."""
    rows = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 9:
                continue
            conf  = int(parts[6])
            cls   = int(parts[7])
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
            rows.append(dict(frame=frame, track_id=track_id,
                             bbox=[x1, y1, w, h], vis=vis))
    return rows


def make_symlinks(domain_name, seq_name, src_img1_dir, seq_len, im_ext='.jpg'):
    """Symlink images: OUT_IMG_ROOT/<short>/<seq>/XXXX.jpg → src."""
    short = domain_name.replace('MOTSynth_', '').lower()
    dst_dir = osp.join(OUT_IMG_ROOT, short, seq_name)
    os.makedirs(dst_dir, exist_ok=True)
    for fidx in range(seq_len):
        fname = f'{fidx:04d}{im_ext}'
        src = osp.join(src_img1_dir, fname)
        dst = osp.join(dst_dir, fname)
        if not osp.lexists(dst):
            os.symlink(src, dst)
    return short


def build_domain_json(domain_name, dry_run=False):
    train_dir = osp.join(MOTSYNTH_BASE, domain_name, 'train')
    if not osp.isdir(train_dir):
        print(f'  [SKIP] no train dir: {train_dir}')
        return None

    seqs = sorted(s for s in os.listdir(train_dir)
                  if osp.isdir(osp.join(train_dir, s)))

    short = domain_name.replace('MOTSynth_', '').lower()

    videos, images, annotations = [], [], []
    vid_id = img_id = ann_id = 1

    for seq in seqs:
        seq_dir   = osp.join(train_dir, seq)
        img1_dir  = osp.join(seq_dir, 'img1')
        gt_path   = osp.join(seq_dir, 'gt', 'gt.txt')
        info_path = osp.join(seq_dir, 'seqinfo.ini')

        if not osp.isdir(img1_dir) or not osp.exists(gt_path):
            continue

        info = read_seqinfo(info_path) if osp.exists(info_path) else {
            'name': seq, 'fps': 20, 'width': 1920, 'height': 1080,
            'seq_length': 0, 'im_ext': '.jpg'}

        # create symlinks
        if not dry_run:
            make_symlinks(domain_name, seq, img1_dir,
                          seq_len=info['seq_length'], im_ext=info['im_ext'])

        # Build image list from seqLength (avoid slow listdir on remote FS).
        # MOTSynth img files: 0-based 4-digit, e.g. 0000.jpg
        # gt.txt frames are 1-based → fidx = frame - 1 → fname = f'{fidx:04d}{ext}'
        ext = info['im_ext']
        seq_len = info['seq_length']
        if seq_len == 0:
            # fallback: count files only if seqLength missing
            seq_len = len([f for f in os.listdir(img1_dir)
                           if f.endswith(('.jpg', '.png'))])

        fname2imgid = {}
        for fidx in range(seq_len):
            fname = f'{fidx:04d}{ext}'
            iid = img_id
            img_id += 1
            fname2imgid[fidx] = (iid, fname)
            images.append({
                'id':        iid,
                'file_name': f'{short}/{seq}/{fname}',
                'height':    info['height'],
                'width':     info['width'],
                'frame_id':  fidx,
                'video_id':  vid_id,
            })

        # gt frame is 1-based → fidx = frame - 1
        gt_rows = load_gt(gt_path)
        for r in gt_rows:
            fidx = r['frame'] - 1
            if fidx not in fname2imgid:
                continue
            iid, _ = fname2imgid[fidx]
            x1, y1, w, h = r['bbox']
            # clamp to image
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
                'category_id': 1,   # pedestrian
                'instance_id': r['track_id'],
                'bbox':        [round(x1, 2), round(y1, 2),
                                round(w,  2), round(h,  2)],
                'area':        round(w * h, 2),
                'iscrowd':     0,
                'visibility':  r['vis'],
            })
            ann_id += 1

        videos.append({
            'id':     vid_id,
            'name':   f'{short}/{seq}',
            'fps':    info['fps'],
            'width':  info['width'],
            'height': info['height'],
        })
        vid_id += 1

    return {'videos': videos, 'images': images,
            'annotations': annotations, 'categories': CATEGORIES}


def main(dry_run=False):
    os.makedirs(OUT_ANN_ROOT, exist_ok=True)
    os.makedirs(OUT_IMG_ROOT, exist_ok=True)

    all_videos, all_images, all_anns = [], [], []
    gvid = gimg = gann = 1

    for domain in DOMAINS:
        print(f'[{domain}] processing ...')
        coco = build_domain_json(domain, dry_run=dry_run)
        if coco is None:
            continue

        short = domain.replace('MOTSynth_', '').lower()
        out_path = osp.join(OUT_ANN_ROOT, f'motsynth_{short}_cocofmt.json')
        if not dry_run:
            with open(out_path, 'w') as f:
                json.dump(coco, f)
        print(f'  → {out_path}  '
              f'videos={len(coco["videos"])} '
              f'images={len(coco["images"])} '
              f'anns={len(coco["annotations"])}')

        # accumulate for global JSON
        iv = gvid - 1
        ii = gimg - 1
        ia = gann - 1
        imgid_remap = {}
        for vv in coco['videos']:
            vv2 = dict(vv); vv2['id'] += iv
            all_videos.append(vv2)
        for im in coco['images']:
            im2 = dict(im)
            old = im2['id']
            im2['id']       += ii
            im2['video_id'] += iv
            imgid_remap[old] = im2['id']
            all_images.append(im2)
        for an in coco['annotations']:
            an2 = dict(an)
            an2['id']       += ia
            an2['image_id']  = imgid_remap[an2['image_id']]
            an2['video_id'] += iv
            all_anns.append(an2)
        gvid += len(coco['videos'])
        gimg += len(coco['images'])
        gann += len(coco['annotations'])

    all_path = osp.join(OUT_ANN_ROOT, 'motsynth_all_cocofmt.json')
    if not dry_run:
        with open(all_path, 'w') as f:
            json.dump({'videos': all_videos, 'images': all_images,
                       'annotations': all_anns, 'categories': CATEGORIES}, f)
    print(f'\n[ALL] → {all_path}  '
          f'videos={len(all_videos)} images={len(all_images)} '
          f'anns={len(all_anns)}')


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    main(dry_run=dry_run)
