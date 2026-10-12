# Scheme 5: Pipeline-level pre-Normalize SPA. SPA runs as a PIPELINE transform
# on the RGB image BEFORE out_pipeline's SeqNormalize (image space), NOT on the
# normalized tensor inside forward_train. Reuses the existing, registered
# SeqSPAFrequencyTransform (numpy FFT low_freq_mask; geometry-preserving, no box
# change). forward_train SPA is disabled to avoid double application.
# student-only (contrastive unchanged), mild strength. No RoI, no causal.
_base_ = [
    '../../bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_overcast.py',
]

norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# student view: original DARTH photometric aug, then SPA low-freq mask, all
# BEFORE out_pipeline's SeqNormalize (RGB image space).
spa_student_pipeline = [
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(type='SeqSPAFrequencyTransform', mode='low_freq_mask',
         low_ratio=0.10, keep_prob=0.90, apply_noise=False, noise_ratio=0.0),
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    # forward_train normalized-tensor SPA OFF (pipeline SPA is used instead)
    use_spa_frequency_aug=False,
    log_view_stats=True,
    student_pipeline=spa_student_pipeline,
)

total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=["bbox", "track"], interval=6,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
