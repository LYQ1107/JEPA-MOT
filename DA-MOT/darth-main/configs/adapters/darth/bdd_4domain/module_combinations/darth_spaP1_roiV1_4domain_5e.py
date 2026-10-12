# Combo 1: pre-Normalize pipeline-level mild SPA + RoI prototype memory V1.
# Base: spa_pipeline_mild (pipeline SPA, use_spa_frequency_aug=False).
# Added: ProtoQuasiDenseTrackHead with RoI-V1 top-score car memory.
# Eval: init_score_thr=0.30 (P1) applied at test-time via cfg-options only.
# No causal direction bank. No normalized-tensor SPA.
_base_ = [
    '../spa_fix/darth_spa_pipeline_mild_4domain_5e.py',
]

model = dict(
    track_head=dict(
        type='ProtoQuasiDenseTrackHead',
        use_roi_proto_memory=True,
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
