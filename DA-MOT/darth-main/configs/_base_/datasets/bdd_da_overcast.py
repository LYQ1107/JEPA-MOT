# Auto-generated BDD100K weather-domain dataset config
# domain: overcast
dataset_type = 'BDDTrackingDataset'
img_prefix = '/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_overcast/'
ann_root = 'data/bdd_da/annotations/box_track_20/'
classes = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqResize', img_scale=(1296, 720), share_params=True, keep_ratio=True, bbox_clip_border=False),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(type='VideoCollect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_match_indices', 'gt_instance_ids']),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='MultiScaleFlipAug', img_scale=(1296, 720), flip=False, transforms=[
        dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'), dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32), dict(type='ImageToTensor', keys=['img']), dict(type='VideoCollect', keys=['img'])])]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    persistent_workers=True,
    train=[dict(type=dataset_type, ann_file=ann_root + 'box_track_overcast_all_cocofmt.json', img_prefix=img_prefix, key_img_sampler=dict(interval=1), ref_img_sampler=dict(num_ref_imgs=1, frame_range=3, method='uniform'), pipeline=train_pipeline, classes=classes)],
    val=dict(type=dataset_type, ann_file=ann_root + 'box_track_overcast_all_cocofmt.json', img_prefix=img_prefix, ref_img_sampler=None, pipeline=test_pipeline, classes=classes),
    test=dict(type=dataset_type, ann_file=ann_root + 'box_track_overcast_all_cocofmt.json', img_prefix=img_prefix, ref_img_sampler=None, pipeline=test_pipeline, classes=classes))
