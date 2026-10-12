# Scheme 2A: SPA-Mild (low_ratio=0.10, keep_prob=0.90). student-only, GPU
# torch.fft normalized-tensor path (same code as 00:32), just weaker strength.
# No contrastive SPA, no high-freq noise, no RoI, no causal.
_base_ = [
    '../../bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_overcast.py',
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    use_spa_frequency_aug=True,
    spa_apply_to_student=True,
    spa_apply_to_contrastive=False,
    spa_low_ratio=0.10,
    spa_keep_prob=0.90,
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
