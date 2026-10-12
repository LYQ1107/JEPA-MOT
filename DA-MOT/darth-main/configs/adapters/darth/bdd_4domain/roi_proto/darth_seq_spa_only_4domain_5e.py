# SPA-only four-domain quick validation (5 epochs/stage).
# clear -> overcast -> rainy -> dawn_dusk -> night.
# Reproduces the "00:32" SPA setting: original DARTH three-view augmentation
# kept unchanged; SPA low-frequency amplitude mask added ON TOP of the student
# view only, computed on GPU via torch.fft inside forward_train (post-Normalize
# path). Contrastive view is NOT touched (protects ID embedding).
# causal=OFF, RoI memory=OFF.
_base_ = [
    '../../bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_overcast.py',
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    # SPA frequency augmentation (00:32 setting): GPU torch.fft path.
    use_spa_frequency_aug=True,
    spa_apply_to_student=True,
    spa_apply_to_contrastive=False,
    spa_low_ratio=0.2,       # prompt_alpha from SPA paper
    spa_keep_prob=0.9,       # 1 - mask_ratio(0.1); R50-GN setting, gentle
    spa_noise_ratio=0.0,
    spa_apply_noise=False,
    log_view_stats=True,
)

total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=["bbox", "track"], interval=6,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
