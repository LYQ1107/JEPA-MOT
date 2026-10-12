# RoI Prototype Memory ablation — V0 Control (no RoI memory).
# clear -> overcast -> night, 5 epochs/stage. causal=OFF, SPA=OFF.
# Uses ProtoQuasiDenseTrackHead with memory disabled so V0/V1/V2/V3 differ
# ONLY by the memory switch and its params.
_base_ = [
    '../../bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_overcast.py',
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    track_head=dict(
        type='ProtoQuasiDenseTrackHead',
        use_roi_proto_memory=False,
    ),
)

total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=["bbox", "track"], interval=6,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
