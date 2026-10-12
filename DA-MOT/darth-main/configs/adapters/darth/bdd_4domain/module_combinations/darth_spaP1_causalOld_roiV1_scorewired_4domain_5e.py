# Strict three-module with score-wired full stage-end RoI-V1.
# Scores are collected through out_pipeline as gt_scores/c_gt_scores and
# consumed by CausalQuasiDenseTrackHead for score-thresholded RoI-V1 memory.
_base_ = [
    './darth_spaP1_causalOld_roiV1_4domain_5e.py',
]

norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    out_pipeline=[
        dict(type='SeqNormalize', **norm_cfg),
        dict(type='SeqPad', size_divisor=32),
        dict(type='MatchInstances', skip_nomatch=False),
        dict(
            type='VideoCollect',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_scores',
                'gt_match_indices', 'gt_instance_ids'
            ]),
    ],
)
