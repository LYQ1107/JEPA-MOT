"""12-frame real-pipeline JEPA smoke configuration."""

_base_ = [
    '../mot/qdtrack/bdd_7domain/qdtrack_faster-rcnn_r50_fpn_20e_bdd_clear.py'
]

model = dict(
    tracker=dict(
        _delete_=True,
        type='TemporalJEPATracker',
        init_score_thr=0.2,
        obj_score_thr=0.2,
        match_score_thr=0.5,
        memo_tracklet_frames=10,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_conf_thr=0.35,
        nms_backdrop_iou_thr=0.3,
        nms_class_iou_thr=0.7,
        with_cats=True,
        match_metric='bisoftmax',
        jepa_enabled=True,
        jepa_checkpoint='/data2/usr_for_deadline/damot_jepa_20260831/source_sample/temporal_jepa_source_v2.pth',
        jepa_embed_dim=256,
        jepa_history_len=4,
        jepa_hidden_dim=128,
        jepa_lambda=0.20,
        jepa_online=False,
        jepa_use_teacher=True,
        jepa_use_replay=True,
        jepa_source_anchor_weight=1.0,
        jepa_domain='bdd_sample'))

data = dict(
    test=dict(
        ann_file='/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/test/testcases/bdd100k-sample/annotations/box_track_debug_cocofmt.json',
        img_prefix='/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/test/testcases/bdd100k-sample/images/'))
