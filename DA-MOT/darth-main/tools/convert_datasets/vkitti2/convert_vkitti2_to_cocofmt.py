#!/usr/bin/env python3
"""Convert Virtual KITTI 2 to COCO-video tracking format (BDD_DA style).

bbox.txt columns: frame cameraID trackID left right top bottom ...
  - left/right/top/bottom are pixel coordinates (top-left origin)
  - COCO bbox = [x1, y1, w, h] = [left, top, right-left, bottom-top]

info.txt columns: trackID label model color
  - label in {Car, Van, Truck}

Output JSON keys: videos, images, annotations, categories
  - Same schema as BDD_DA box_track_*_cocofmt.json
  - video name: Scene01_clone_Camera_0
  - file_name:  Scene01_clone_Camera_0/rgb_00000.jpg
"""

import argparse
import json
import os
import os.path as osp

VKITTI2_ROOT = '/data3/testdata/vranlee/.MOTSynth.partial/VKITTI2'
RGB_ROOT     = osp.join(VKITTI2_ROOT, 'vkitti_2.0.3_rgb')
GT_ROOT      = osp.join(VKITTI2_ROOT, 'vkitti_2.0.3_textgt')

OUT_IMG_ROOT = '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/vkitti2/images'
OUT_ANN_ROOT = '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/vkitti2/annotations'

IMG_WIDTH  = 1242
IMG_HEIGHT = 375
FPS        = 10

# VKITTI2 label → BDD-style category name + id
# Only Car/Van/Truck appear in the data
LABEL_MAP = {
    'Car':   ('car',   2),
    'Van':   ('truck', 3),
    'Truck': ('truck', 3),
}

CATEGORIES = [
    {'id': 1, 'name': 'pedestrian'},
    {'id': 2, 'name': 'car'},
    {'id': 3, 'name': 'truck'},
    {'id': 4, 'name': 'bus'},
    {'id': 5, 'name': 'motorcycle'},
    {'id': 6, 'name': 'bicycle'},
]


def load_info(path):
    """Return {trackID: category_id} from info.txt."""
    tid2cat = {}
    with open(path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if i == 0 or len(parts) < 2:
                continue
            try:
                tid = int(parts[0])
            except ValueError:
                continue
            label = parts[1]
            if label in LABEL_MAP:
                tid2cat[tid] = LABEL_MAP[label][1]
    return tid2cat


def load_bbox(path):
    """Return list of dicts with frame, cameraID, trackID, bbox=[x1,y1,w,h]."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            frame     = int(parts[0])
            camera_id = int(parts[1])
            track_id  = int(parts[2])
            left      = float(parts[3])
            right     = float(parts[4])
            top       = float(parts[5])
            bottom    = float(parts[6])
            x1, y1    = left, top
            w, h      = right - left, bottom - top
            if w <= 0 or h <= 0:
                continue
            rows.append(dict(frame=frame, camera_id=camera_id,
                             track_id=track_id,
                             bbox=[x1, y1, w, h]))
    return rows


def make_symlinks(scene, variant, camera_id):
    """Create symlinks: OUT_IMG_ROOT/Scene01_clone_Camera_0/rgb_00000.jpg
                      → RGB_ROOT/Scene01/clone/frames/rgb/Camera_0/rgb_00000.jpg
    Returns the video_name string.
    """
    video_name = f'{scene}_{variant}_Camera_{camera_id}'
    src_dir = osp.join(RGB_ROOT, scene, variant, 'frames', 'rgb',
                       f'Camera_{camera_id}')
    dst_dir = osp.join(OUT_IMG_ROOT, video_name)
    os.makedirs(dst_dir, exist_ok=True)
    if not osp.exists(src_dir):
        return None
    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith('.jpg'):
            continue
        src = osp.join(src_dir, fname)
        dst = osp.join(dst_dir, fname)
        if not osp.lexists(dst):
            os.symlink(src, dst)
    return video_name


def build_sequence_data(scene, variant, camera_id,
                        video_id_start, image_id_start, ann_id_start):
    """Build videos/images/annotations lists for one (scene, variant, camera)."""
    gt_dir  = osp.join(GT_ROOT, scene, variant)
    bbox_path = osp.join(gt_dir, 'bbox.txt')
    info_path = osp.join(gt_dir, 'info.txt')

    if not osp.exists(bbox_path) or not osp.exists(info_path):
        return None, None, None, video_id_start, image_id_start, ann_id_start

    tid2cat = load_info(info_path)
    raw     = load_bbox(bbox_path)

    # filter to requested camera
    raw = [r for r in raw if r['camera_id'] == camera_id]
    if not raw:
        return None, None, None, video_id_start, image_id_start, ann_id_start

    # build symlinks
    video_name = make_symlinks(scene, variant, camera_id)
    if video_name is None:
        return None, None, None, video_id_start, image_id_start, ann_id_start

    # all frames present in image dir
    img_src_dir = osp.join(RGB_ROOT, scene, variant, 'frames', 'rgb',
                           f'Camera_{camera_id}')
    frame_files = sorted(f for f in os.listdir(img_src_dir)
                         if f.endswith('.jpg'))

    video = {'id': video_id_start, 'name': video_name,
             'fps': FPS, 'width': IMG_WIDTH, 'height': IMG_HEIGHT}

    # frame index → image_id
    frame2imgid = {}
    images = []
    for fidx, fname in enumerate(frame_files):
        # parse frame number from filename  rgb_00000.jpg → 0
        fnum = int(osp.splitext(fname)[0].split('_')[1])
        img_id = image_id_start
        image_id_start += 1
        frame2imgid[fnum] = img_id
        images.append({
            'id':       img_id,
            'file_name': f'{video_name}/{fname}',
            'height':   IMG_HEIGHT,
            'width':    IMG_WIDTH,
            'frame_id': fidx,
            'video_id': video_id_start - (fidx + 1 - (img_id - (image_id_start - len(frame_files))))
        })
    # fix video_id references
    for im in images:
        im['video_id'] = video['id']

    annotations = []
    for r in raw:
        fnum = r['frame']
        if fnum not in frame2imgid:
            continue
        tid = r['track_id']
        cat_id = tid2cat.get(tid)
        if cat_id is None:
            continue
        x1, y1, w, h = r['bbox']
        # clamp to image bounds
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        w  = min(w,  IMG_WIDTH  - x1)
        h  = min(h,  IMG_HEIGHT - y1)
        if w <= 0 or h <= 0:
            continue
        annotations.append({
            'id':          ann_id_start,
            'image_id':    frame2imgid[fnum],
            'video_id':    video['id'],
            'category_id': cat_id,
            'instance_id': tid,
            'bbox':        [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            'area':        round(w * h, 2),
            'iscrowd':     0,
        })
        ann_id_start += 1

    video_id_start += 1
    return [video], images, annotations, video_id_start, image_id_start, ann_id_start


def build_variant_json(variant, scenes):
    """Build and return COCO dict for a single variant across all scenes."""
    all_videos, all_images, all_anns = [], [], []
    vid_id, img_id, ann_id = 1, 1, 1

    for scene in scenes:
        for cam in [0, 1]:
            v, imgs, anns, vid_id, img_id, ann_id = build_sequence_data(
                scene, variant, cam, vid_id, img_id, ann_id)
            if v is None:
                continue
            all_videos.extend(v)
            all_images.extend(imgs)
            all_anns.extend(anns)

    return {'videos': all_videos, 'images': all_images,
            'annotations': all_anns, 'categories': CATEGORIES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    scenes   = sorted(os.listdir(GT_ROOT))
    variants = sorted({v for s in scenes
                       for v in os.listdir(osp.join(GT_ROOT, s))})

    os.makedirs(OUT_ANN_ROOT, exist_ok=True)
    os.makedirs(OUT_IMG_ROOT, exist_ok=True)

    all_videos_global, all_images_global, all_anns_global = [], [], []
    gvid, gimg, gann = 1, 1, 1

    for variant in variants:
        print(f'[{variant}] processing ...')
        vid_id, img_id, ann_id = 1, 1, 1
        v_videos, v_images, v_anns = [], [], []

        for scene in scenes:
            for cam in [0, 1]:
                v, imgs, anns, vid_id, img_id, ann_id = build_sequence_data(
                    scene, variant, cam, vid_id, img_id, ann_id)
                if v is None:
                    continue
                v_videos.extend(v)
                v_images.extend(imgs)
                v_anns.extend(anns)

        coco = {'videos': v_videos, 'images': v_images,
                'annotations': v_anns, 'categories': CATEGORIES}

        out_path = osp.join(OUT_ANN_ROOT, f'vkitti2_{variant}_cocofmt.json')
        if not args.dry_run:
            with open(out_path, 'w') as f:
                json.dump(coco, f)
        print(f'  → {out_path}  '
              f'videos={len(v_videos)} images={len(v_images)} anns={len(v_anns)}')

        # accumulate for global JSON (re-index IDs to avoid collision)
        id_offset_v   = gvid - 1
        id_offset_img = gimg - 1
        id_offset_ann = gann - 1
        for vv in v_videos:
            vv2 = dict(vv); vv2['id'] += id_offset_v
            all_videos_global.append(vv2)
        imgid_remap = {}
        for im in v_images:
            im2 = dict(im)
            old_id = im2['id']
            im2['id']       += id_offset_img
            im2['video_id'] += id_offset_v
            imgid_remap[old_id] = im2['id']
            all_images_global.append(im2)
        for an in v_anns:
            an2 = dict(an)
            an2['id']       += id_offset_ann
            an2['image_id']  = imgid_remap[an2['image_id']]
            an2['video_id'] += id_offset_v
            all_anns_global.append(an2)
        gvid += len(v_videos)
        gimg += len(v_images)
        gann += len(v_anns)

    all_path = osp.join(OUT_ANN_ROOT, 'vkitti2_all_cocofmt.json')
    if not args.dry_run:
        with open(all_path, 'w') as f:
            json.dump({'videos': all_videos_global, 'images': all_images_global,
                       'annotations': all_anns_global, 'categories': CATEGORIES}, f)
    print(f'\n[ALL] → {all_path}  '
          f'videos={len(all_videos_global)} images={len(all_images_global)} '
          f'anns={len(all_anns_global)}')


if __name__ == '__main__':
    main()
