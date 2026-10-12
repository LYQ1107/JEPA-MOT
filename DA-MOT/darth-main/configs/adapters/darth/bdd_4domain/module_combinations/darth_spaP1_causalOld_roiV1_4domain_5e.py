# Combo 3 STRICT: pre-Normalize pipeline-level mild SPA + old causal_only
# + full stage-end RoI-V1.
#
# Base: spa_pipeline_mild (pipeline SPA, use_spa_frequency_aug=False).
# Causal: original old causal_only / TACT-style global_pc_ema bank.
# RoI-V1: history-guided stage-end top-score car prototype memory using
#         roi_proto_memory.py (history_only=True, stage_end_update=True).
# Eval: init_score_thr=0.30 (P1) applied at test-time via cfg-options only.
_base_ = [
    '../spa_fix/darth_spa_pipeline_mild_4domain_5e.py',
]

model = dict(
    track_head=dict(
        type='CausalQuasiDenseTrackHead',
        # old causal_only / original global PC EMA direction bank
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
        # full RoI-V1, not FIFO fallback
        use_roi_proto_memory=True,
        roi_proto_memory_mode='stage',
        roi_proto_class_ids=(1,),
        roi_proto_score_thr=0.70,
        roi_proto_k_per_domain_class=100,
        roi_proto_max_candidates_per_domain=1000,
        roi_proto_topk=3,
        roi_proto_loss_weight=0.01,
        roi_proto_max_rois_per_batch=128,
        roi_proto_diverse_select=False,
        roi_proto_feature_source='student',
        roi_proto_log_interval=50,
    ),
)
