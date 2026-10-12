_base_ = [
    '../../../_base_/target_datasets/bdd_da_overcast.py',
    '../darth_qdtrack_faster-rcnn_r50_fpn_12e_base.py',
]

# Three views keep the original DARTH pipeline unchanged.
# SPA is applied inside forward_train on normalized GPU tensors via
# torch.fft (use_spa_frequency_aug=True, spa_forward_postprocess=True),
# exactly as in the original SPA paper (Prompt.masking + LearnablePatchErasing).
#
# Parameters follow SPA paper ResNet50-GN setting (closest to our backbone):
#   freq_mask_ratio = 0.1  →  keep_prob = 0.9
#   noise_ratio     = 0.1  (high-freq noise, applied to contrastive view only)
#
# View assignment (teacher never augmented, keeps pseudo-labels stable):
#   student/key view    ← low_freq_mask  (keep_prob=0.9)
#   contrastive view    ← high_freq_noise (noise_ratio=0.1)

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    # SPA: original-paper torch.fft path, applied on normalized GPU tensors
    use_spa_frequency_aug=True,
    spa_apply_to_student=True,
    spa_apply_to_contrastive=False,
    spa_low_ratio=0.2,       # prompt_alpha from SPA paper
    spa_keep_prob=0.9,       # 1 - freq_mask_ratio=0.1 (R50-GN setting)
    spa_noise_ratio=0.0,
    spa_apply_noise=False,
    # one-time view stats log to confirm SPA acts on normalized tensors
    log_view_stats=True,
)

total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=["bbox", "track"], interval=6,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
