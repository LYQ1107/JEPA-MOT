# Task5: mixed-domain DARTH adaptation config
# Concatenate train sets of overcast + rainy + dawn_dusk + night into ONE dataset.
# The DataLoader sampler shuffles across all domains -> mixed sampling order.
_base_ = [
    '../darth_qdtrack_faster-rcnn_r50_fpn_12e_base.py',
]

dataset_type = 'BDDTrackingDataset'
ann_root = 'data/bdd_da/annotations/box_track_20/'
classes = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqResize', img_scale=(1296, 720), share_params=True,
         keep_ratio=True, bbox_clip_border=False),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(type='VideoCollect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices',
               'gt_instance_ids']),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='MultiScaleFlipAug', img_scale=(1296, 720), flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip'),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='VideoCollect', keys=['img']),
         ]),
]


def _ds(domain):
    return dict(
        type=dataset_type,
        ann_file=ann_root + f'box_track_{domain}_all_cocofmt.json',
        img_prefix=f'/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_{domain}/',
        key_img_sampler=dict(interval=1),
        ref_img_sampler=dict(num_ref_imgs=1, frame_range=3, method='uniform'),
        pipeline=train_pipeline,
        classes=classes,
    )


model = dict(detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))))

# 6-class BDD; mix 4 target domains into one concatenated train set
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=2,
    persistent_workers=True,
    train=[
        _ds('overcast'),
        _ds('rainy'),
        _ds('dawn_dusk'),
        _ds('night'),
    ],
    val=dict(
        type=dataset_type,
        ann_file=ann_root + 'box_track_overcast_all_cocofmt.json',
        img_prefix='/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast/',
        ref_img_sampler=None, pipeline=test_pipeline, classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=ann_root + 'box_track_overcast_all_cocofmt.json',
        img_prefix='/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast/',
        ref_img_sampler=None, pipeline=test_pipeline, classes=classes),
)

total_epochs = 10
# 10e schedule: same milestone ratio as base [8,11]/12 -> [7,9]/10
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 1000,
    step=[7, 9],
)
checkpoint_config = dict(interval=1)
evaluation = dict(metric=["bbox", "track"], interval=41,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
