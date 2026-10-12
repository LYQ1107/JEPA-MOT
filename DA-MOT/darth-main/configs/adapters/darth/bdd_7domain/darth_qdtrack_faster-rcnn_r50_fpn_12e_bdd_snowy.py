_base_ = [
    '../../../_base_/target_datasets/bdd_da_snowy.py',
    '../darth_qdtrack_faster-rcnn_r50_fpn_12e_base.py',
]
model = dict(detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))))
total_epochs = 12
checkpoint_config = dict(interval=1)
evaluation = dict(metric=["bbox", "track"], interval=6,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False, nproc=1, majority_voting=False))
