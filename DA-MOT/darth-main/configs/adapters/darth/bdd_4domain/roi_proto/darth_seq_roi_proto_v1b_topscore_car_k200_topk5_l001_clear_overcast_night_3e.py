# RoI Prototype Memory quick-verify — V1b: V1 with wider prototype coverage.
# Identical to V1 EXCEPT k_per_domain_class 100 -> 200 and topk 3 -> 5.
# Purpose: test whether V1's DetA drop is caused by too-few prototypes over-
# constraining current RoI embeddings. NOT V3 (V3 uses diversity selection;
# V1b keeps top-score, only widens coverage). loss_weight stays 0.01.
# clear -> overcast -> night, 3 epochs/stage. causal=OFF, SPA=OFF.
_base_ = [
    '../../bdd_7domain/darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_overcast.py',
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
    track_head=dict(
        type='ProtoQuasiDenseTrackHead',
        use_roi_proto_memory=True,
        roi_proto_class_ids=(1,),          # car (classes idx: pedestrian=0, car=1)
        roi_proto_score_thr=0.70,
        roi_proto_k_per_domain_class=200,  # V1b: wider than V1 (100)
        roi_proto_max_candidates_per_domain=1000,
        roi_proto_topk=5,                  # V1b: wider than V1 (3)
        roi_proto_loss_weight=0.01,
        roi_proto_max_rois_per_batch=128,
        roi_proto_diverse_select=False,
        roi_proto_feature_source='student',
        roi_proto_log_interval=50,
    ),
)

total_epochs = 3
checkpoint_config = dict(interval=1)
evaluation = dict(
    metric=["bbox", "track"], interval=6,
    results_kwargs=dict(results_path=None, scalabel=False, append_empty_preds=True),
    track_kwargs=dict(iou_thr=0.5, ignore_iof_thr=0.5, ignore_by_classes=False,
                      nproc=1, majority_voting=False))
