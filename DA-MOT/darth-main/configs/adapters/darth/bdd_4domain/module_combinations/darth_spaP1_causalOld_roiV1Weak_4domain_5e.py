# Soft three-module: RoI-V1 weak, score-wired full stage-end memory.
_base_ = [
    './darth_spaP1_causalOld_roiV1_scorewired_4domain_5e.py',
]

model = dict(
    track_head=dict(
        roi_proto_loss_weight=0.005,
        roi_proto_max_rois_per_batch=64,
        roi_proto_score_thr=0.75,
    ),
)
