_base_ = [
    '../../../_base_/target_datasets/vkitti2_fog.py',
    '../darth_qdtrack_faster-rcnn_r50_fpn_12e_base.py',
]

model = dict(
    init_cfg=dict(
        type='KDPretrained',
        checkpoint='work_dirs/vkitti2/source/clone_20e/qdtrack_vkitti2_clone_20e/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone/20260707_171736/epoch_20.pth',
        map_location='cpu'),
    detector=dict(
        init_cfg=None,
        roi_head=dict(bbox_head=dict(num_classes=6))),
)

find_unused_parameters = True
total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=['bbox', 'track'], interval=999,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
