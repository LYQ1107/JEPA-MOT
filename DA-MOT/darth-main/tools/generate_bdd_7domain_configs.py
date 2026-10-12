#!/usr/bin/env python3
from pathlib import Path
import json

ROOT = Path('/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main')
ANN_ROOT = ROOT / 'data/bdd_da/annotations/box_track_20'
DOMAINS = ['clear', 'dawn_dusk', 'night', 'partly_cloudy', 'overcast', 'rainy', 'snowy']
DOMAIN_DIRS = {
    'clear': '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_clear/',
    'dawn_dusk': '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_dawn_dusk/',
    'night': '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_night/',
    'partly_cloudy': '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_partly_cloudy/',
    'overcast': '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast/',
    'rainy': '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_rainy/',
    'snowy': '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_snowy/',
}

DATASET_TEMPLATE = """# Auto-generated BDD100K weather-domain dataset config
# domain: {domain}
dataset_type = 'BDDTrackingDataset'
img_prefix = '{img_prefix}'
ann_root = 'data/bdd_da/annotations/box_track_20/'
classes = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqResize', img_scale=(1296, 720), share_params=True, keep_ratio=True, bbox_clip_border=False),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices', 'gt_instance_ids']),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='MultiScaleFlipAug', img_scale=(1296, 720), flip=False, transforms=[
        dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'), dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32), dict(type='ImageToTensor', keys=['img']), dict(type='VideoCollect', keys=['img'])])]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[dict(type=dataset_type, ann_file=ann_root + '{train_ann}', img_prefix=img_prefix, key_img_sampler=dict(interval=1), ref_img_sampler=dict(num_ref_imgs=1, frame_range=3, method='uniform'), pipeline=train_pipeline, classes=classes)],
    val=dict(type=dataset_type, ann_file=ann_root + '{eval_ann}', img_prefix=img_prefix, ref_img_sampler=None, pipeline=test_pipeline, classes=classes),
    test=dict(type=dataset_type, ann_file=ann_root + '{eval_ann}', img_prefix=img_prefix, ref_img_sampler=None, pipeline=test_pipeline, classes=classes))
"""

SOURCE_CFG_TEMPLATE = """_base_ = [
    '../qdtrack_faster-rcnn_r50_fpn_12e_base.py',
    '../../../_base_/datasets/bdd_da_{domain}.py',
]
model = dict(detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))))
total_epochs = 20
checkpoint_config = dict(interval=5)
evaluation = dict(metric=['bbox', 'track'], interval=5,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False, nproc=1, majority_voting=False))
"""

TTA_CFG_TEMPLATE = """_base_ = [
    '../../../_base_/target_datasets/bdd_da_{domain}.py',
    '../darth_qdtrack_faster-rcnn_r50_fpn_12e_base.py',
]
model = dict(detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))))
total_epochs = 12
checkpoint_config = dict(interval=2)
evaluation = dict(metric=['bbox', 'track'], interval=2,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False, nproc=1, majority_voting=False))
"""

def json_count(path):
    p = ANN_ROOT / path
    if not p.exists():
        return 0
    try:
        return len(json.loads(p.read_text()).get('images', []))
    except Exception:
        return 0

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f'wrote {path}')

def main():
    notes = []
    for d in DOMAINS:
        train_ann = f'box_track_{d}_train_cocofmt.json'
        val_ann = f'box_track_{d}_val_cocofmt.json'
        all_ann = f'box_track_{d}_all_cocofmt.json'
        source_eval_ann = val_ann if json_count(val_ann) > 0 else all_ann
        target_train_ann = val_ann if json_count(val_ann) > 0 else all_ann
        if source_eval_ann == all_ann:
            notes.append(f'{d}: val empty/missing, using all for val/test')
        source_text = DATASET_TEMPLATE.format(domain=d, img_prefix=DOMAIN_DIRS[d], train_ann=train_ann, eval_ann=source_eval_ann)
        target_text = DATASET_TEMPLATE.format(domain=d, img_prefix=DOMAIN_DIRS[d], train_ann=target_train_ann, eval_ann=target_train_ann)
        write(ROOT / f'configs/_base_/datasets/bdd_da_{d}.py', source_text)
        write(ROOT / f'configs/_base_/target_datasets/bdd_da_{d}.py', target_text)
        write(ROOT / f'configs/mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_{d}.py', SOURCE_CFG_TEMPLATE.format(domain=d))
        write(ROOT / f'configs/adapters/darth/bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_{d}.py', TTA_CFG_TEMPLATE.format(domain=d))
    out = ROOT / 'work_dirs/bdd_7domain/configs/generation_notes.txt'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(notes) + ('\n' if notes else 'All domains use val for eval.\n'))
    print(out.read_text())

if __name__ == '__main__':
    main()
