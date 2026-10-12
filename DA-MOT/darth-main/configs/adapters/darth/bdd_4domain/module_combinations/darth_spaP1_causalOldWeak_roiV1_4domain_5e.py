# Soft three-module: causal weak, score-wired full stage-end RoI-V1.
_base_ = [
    './darth_spaP1_causalOld_roiV1_scorewired_4domain_5e.py',
]

model = dict(
    track_head=dict(
        causal_trim_alpha=0.15,
        causal_loss_weight=0.05,
    ),
)
