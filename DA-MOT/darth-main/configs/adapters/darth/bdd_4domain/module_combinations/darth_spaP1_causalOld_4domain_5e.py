# Combo 2: pre-Normalize pipeline-level mild SPA + old causal_only.
# Base: spa_pipeline_mild (pipeline SPA, use_spa_frequency_aug=False).
# Added: CausalQuasiDenseTrackHead with original TACT-style global_pc_ema bank.
# Eval: init_score_thr=0.30 (P1) applied at test-time via cfg-options only.
# No RoI prototype memory. No cov_ema / car_weighted / pairwise PCA variants.
_base_ = [
    '../spa_fix/darth_spa_pipeline_mild_4domain_5e.py',
]

model = dict(
    track_head=dict(
        type='CausalQuasiDenseTrackHead',
        use_causal_direction_bank=True,
        causal_pca_mode='global_pc_ema',
        causal_num_pcs=1,
        causal_start_pc=0,
        causal_trim_alpha=0.25,
        causal_loss_weight=0.1,
        causal_min_samples=32,
        causal_update_momentum=0.9,
        causal_normalize_feats=True,
        causal_bank_scope='global',
        causal_feat_dim=256,
        use_roi_proto_memory=False,
    ),
)
