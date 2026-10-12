_base_ = [
    './darth_spaP1_causalOld_roiV1Weak_4domain_5e.py',
]

model = dict(
    track_head=dict(
        causal_pca_mode='proto_shrinkage_cov_ema',
        causal_update_by_covariance=True,
        causal_cov_momentum=0.98,
        causal_qr_orthogonalize=True,
        causal_num_pcs=2,
        causal_trim_alpha=0.10,
        causal_loss_weight=0.025,
        causal_min_samples=96,
        causal_class_ids=None,
        causal_covariance_shrinkage=0.15,
        causal_proto_center_momentum=0.95,
        causal_proto_center_min_count=8,
    ),
)
