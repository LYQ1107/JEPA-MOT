# V3: Pairwise Residual PCA.
# PCA is estimated on (key_feat - ref_feat) residuals of matched GT pairs,
# NOT on raw RoI features. Hypothesis: pair residuals isolate temporal/weather
# variation from identity/class variance, giving a cleaner non-causal direction.
# Conservative alpha/loss_weight because residual PCs are more sensitive.
# car class_id = 1  (BDD: pedestrian=0, car=1, truck=2, bus=3, motorcycle=4, bicycle=5)
_base_ = [
    '../../bdd_7domain/darth_seq_ablation_base.py',
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    use_spa_frequency_aug=False,
    track_head=dict(
        type='CausalQuasiDenseTrackHead',
        use_causal_direction_bank=True,
        causal_pca_mode='pairwise_residual',
        causal_residual_source='key_ref_match',
        causal_residual_normalize_before_delta=True,
        causal_update_by_covariance=True,
        causal_cov_momentum=0.95,
        causal_qr_orthogonalize=True,
        causal_class_ids=(1,),
        causal_num_pcs=1,
        causal_start_pc=0,
        causal_trim_alpha=0.10,
        causal_loss_weight=0.05,
        causal_min_samples=64,
        causal_update_momentum=0.9,
        causal_normalize_feats=True,
        causal_bank_scope='global',
        causal_feat_dim=256,
        causal_log_interval=50,
        causal_debug_stats=True,
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
