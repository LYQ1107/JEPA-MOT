_base_ = [
    '../../qdtrack_faster-rcnn_r50_fpn_12e_base.py',
    '../../../../_base_/target_datasets/vkitti2_sunset.py',
]

model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/vkitti2/source/clone_20e/qdtrack_vkitti2_clone_20e/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone/20260707_171736/epoch_20.pth'),
        roi_head=dict(bbox_head=dict(
            num_classes=6,
            bbox_coder=dict(clip_border=True)))),
    tracker=dict(
        _delete_=True,
        type='QuasiDenseTracker',
        init_score_thr=0.2,
        obj_score_thr=0.2,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.35,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax'))

total_epochs = 5
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[4])
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=['bbox', 'track'], interval=999,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
