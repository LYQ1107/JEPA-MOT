_base_ = [
    '../qdtrack_faster-rcnn_r50_fpn_12e_base.py',
    '../../../_base_/datasets/vkitti2_rain_eval.py',
]

model = dict(
    detector=dict(
        init_cfg=None,
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

total_epochs = 20
evaluation = dict(
    metric=['bbox', 'track'], interval=20,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
