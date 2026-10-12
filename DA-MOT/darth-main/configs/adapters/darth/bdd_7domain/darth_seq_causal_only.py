# Experiment B: TACT-style causal direction bank only (trimmed PCL).
# SPA OFF, causal bank ON, memory OFF.
_base_ = [
    './darth_seq_ablation_base.py',
]

model = dict(
    use_spa_frequency_aug=False,
    track_head=dict(
        use_causal_direction_bank=True,
        causal_num_pcs=1,
        causal_start_pc=0,
        causal_trim_alpha=0.25,
        causal_loss_weight=0.1,
        causal_min_samples=32,
        causal_update_momentum=0.9,
        causal_normalize_feats=True,
        causal_bank_scope='global',
        use_roi_proto_memory=False,
    ),
)
