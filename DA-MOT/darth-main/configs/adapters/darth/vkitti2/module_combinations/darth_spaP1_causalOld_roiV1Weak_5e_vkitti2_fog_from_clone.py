_base_ = [
    '../../../../_base_/target_datasets/vkitti2_fog.py',
    '../../darth_qdtrack_faster-rcnn_r50_fpn_12e_base.py',
]

# Our method: continuous adaptation + SPA(P1) + causalOld + RoI-V1Weak.
# Dataset-specific version migrated from
# configs/adapters/darth/bdd_4domain/module_combinations/darth_spaP1_causalOld_roiV1Weak_4domain_5e.py
# without inheriting BDD dataset paths.

norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

spa_student_pipeline = [
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    dict(type='SeqSPAFrequencyTransform', mode='low_freq_mask',
         low_ratio=0.10, keep_prob=0.90, apply_noise=False, noise_ratio=0.0),
]

model = dict(
    init_cfg=dict(
        type='KDPretrained',
        checkpoint='work_dirs/vkitti2/source/clone_20e/qdtrack_vkitti2_clone_20e/qdtrack_faster-rcnn_r50_fpn_20e_vkitti2_clone/20260707_171736/epoch_20.pth',
        map_location='cpu'),
    detector=dict(
        init_cfg=None,
        roi_head=dict(bbox_head=dict(num_classes=6))),
    use_spa_frequency_aug=False,
    log_view_stats=True,
    student_pipeline=spa_student_pipeline,
    out_pipeline=[
        dict(type='SeqNormalize', **norm_cfg),
        dict(type='SeqPad', size_divisor=32),
        dict(type='MatchInstances', skip_nomatch=False),
        dict(
            type='VideoCollect',
            keys=[
                'img', 'gt_bboxes', 'gt_labels', 'gt_scores',
                'gt_match_indices', 'gt_instance_ids'
            ]),
    ],
    track_head=dict(
        type='CausalQuasiDenseTrackHead',
        use_causal_direction_bank=True,
        causal_pca_mode='global_pc_ema',
        causal_num_pcs=1,
        causal_start_pc=0,
        causal_trim_alpha=0.25,
        causal_loss_weight=0.1,
        causal_min_samples=32,
        causal_update_momentum=0.9,
        causal_normalize_feats=True,
        causal_bank_scope='global',
        causal_feat_dim=256,
        use_roi_proto_memory=True,
        roi_proto_memory_mode='stage',
        roi_proto_class_ids=(1,),
        roi_proto_score_thr=0.75,
        roi_proto_k_per_domain_class=100,
        roi_proto_max_candidates_per_domain=1000,
        roi_proto_topk=3,
        roi_proto_loss_weight=0.005,
        roi_proto_max_rois_per_batch=64,
        roi_proto_diverse_select=False,
        roi_proto_feature_source='student',
        roi_proto_log_interval=50,
    ),
)

find_unused_parameters = True
total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=['bbox', 'track'], interval=999,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
