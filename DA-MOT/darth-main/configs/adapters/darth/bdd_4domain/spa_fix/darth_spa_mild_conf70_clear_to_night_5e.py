# Scheme 3: SPA-Mild (0.10/0.90) + confidence gate (spa_conf_thr=0.70).
# Higher pseudo-label reliability so SPA student does not over-learn low-conf
# teacher boxes. student-only, GPU normalized-tensor SPA. No RoI, no causal.
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
    use_spa_conf_gate=True,
    spa_conf_thr=0.70,
    log_view_stats=True,
)

total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=["bbox", "track"], interval=6,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
