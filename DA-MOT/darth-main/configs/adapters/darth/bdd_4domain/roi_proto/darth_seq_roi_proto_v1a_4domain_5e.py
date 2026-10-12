# RoI Prototype Memory — V1a four-domain 5e (weaker loss 0.005).
# clear -> overcast -> rainy -> dawn_dusk -> night, 5 epochs/stage.
_base_ = [
    '../../bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_overcast.py',
]
model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    track_head=dict(
        type='ProtoQuasiDenseTrackHead',
        use_roi_proto_memory=True,
        roi_proto_class_ids=(1,),
        roi_proto_score_thr=0.70,
        roi_proto_k_per_domain_class=100,
        roi_proto_max_candidates_per_domain=1000,
        roi_proto_topk=3,
        roi_proto_loss_weight=0.005,
        roi_proto_max_rois_per_batch=128,
        roi_proto_diverse_select=False,
        roi_proto_feature_source='student',
        roi_proto_log_interval=50,
    ),
)
total_epochs = 5
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=["bbox", "track"], interval=6,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
