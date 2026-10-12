# Shared base for SPA / causal ablation experiments.
# Inherits the bdd_overcast DARTH config (same --base-config the baseline uses
# for the episode trainer; the runner swaps data.train per stage). Sets 6-class
# BDD head and swaps the track head to CausalQuasiDenseTrackHead. All new
# modules default OFF here; concrete experiment configs flip individual switches.
_base_ = [
    './darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_overcast.py',
]

model = dict(
    # 6-class BDD (already set by base, repeated for clarity)
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    # SPA frequency augmentation (default OFF; flipped by experiment configs)
    use_spa_frequency_aug=False,
    spa_apply_to_student=True,
    spa_apply_to_contrastive=False,
    spa_low_ratio=0.2,
    spa_keep_prob=0.8,
    spa_noise_ratio=0.0,
    spa_apply_noise=False,
    # swap to causal track head; bank/memory default OFF -> baseline behavior
    track_head=dict(
        type='CausalQuasiDenseTrackHead',
        use_causal_direction_bank=False,
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
        roi_memory_loss_weight=0.0,
    ),
)

total_epochs = 10
checkpoint_config = dict(interval=1)
evaluation = dict(metric=["bbox", "track"], interval=11,
    results_kwargs=dict(results_path=None, scalabel=False,
                        append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5,
                      ignore_by_classes=False, nproc=1, majority_voting=False))
