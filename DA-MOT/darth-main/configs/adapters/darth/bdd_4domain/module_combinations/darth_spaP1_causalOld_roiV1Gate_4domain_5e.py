# Similarity-gated three-module: score-wired full stage-end RoI-V1 with
# prototype loss applied only to RoIs already similar to history prototypes.
_base_ = [
    './darth_spaP1_causalOld_roiV1_scorewired_4domain_5e.py',
]

model = dict(
    track_head=dict(
        roi_proto_use_sim_gate=True,
        roi_proto_sim_gate_thr=0.50,
    ),
)
