# V1: Stable Cov-EMA PCA.
# Replaces per-batch PC-EMA with EMA covariance -> eigh each step.
# Fixes PC sign-flip / direction-drift instability of the original.
# All other settings identical to current causal_only.
_base_ = [
    '../../bdd_7domain/darth_seq_ablation_base.py',
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    use_spa_frequency_aug=False,
    track_head=dict(
        type='CausalQuasiDenseTrackHead',
        use_causal_direction_bank=True,
        causal_pca_mode='cov_ema',
        causal_update_by_covariance=True,
        causal_cov_momentum=0.95,
        causal_qr_orthogonalize=True,
        causal_num_pcs=1,
        causal_start_pc=0,
        causal_trim_alpha=0.25,
        causal_loss_weight=0.10,
        causal_min_samples=64,
        causal_update_momentum=0.9,
        causal_normalize_feats=True,
        causal_bank_scope='global',
        causal_feat_dim=256,
        causal_log_interval=50,
        causal_debug_stats=True,
        causal_class_ids=None,
        causal_use_scores=False,
        use_roi_proto_memory=False,
        roi_memory_loss_weight=0.0,
    ),
)

total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=['bbox', 'track'], interval=6,
    results_kwargs=dict(results_path=None, scalabel=False,
                        append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5,
                      ignore_by_classes=False, nproc=1,
                      majority_voting=False))
