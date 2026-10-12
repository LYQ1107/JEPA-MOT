_base_ = [
    './qdtrack_faster-rcnn_r50_fpn_12e_base.py',
    '../../_base_/datasets/bdd_da_clear.py'
]

model = dict(
    detector=dict(
        roi_head=dict(bbox_head=dict(num_classes=6)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
        ),
    ),
    tracker=dict(
        type='QuasiDenseTracker',
        init_score_thr=0.5,
        obj_score_thr=0.3,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.5,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'
    ),
)

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[8, 11])
total_epochs = 12
evaluation = dict(metric=['bbox', 'track'], interval=4)
