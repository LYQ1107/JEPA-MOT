# UA-DETRAC rainy domain — eval-only dataset config
dataset_type = 'BDDTrackingDataset'
img_prefix = 'data/ua_detrac/images/rainy/'
ann_root = 'data/ua_detrac/annotations/'
classes = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='MultiScaleFlipAug', img_scale=(960, 540), flip=False, transforms=[
        dict(type='Resize', keep_ratio=True), dict(type='RandomFlip'), dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32), dict(type='ImageToTensor', keys=['img']), dict(type='VideoCollect', keys=['img'])])]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val=dict(type=dataset_type, ann_file=ann_root + 'box_track_rainy_all_cocofmt.json', img_prefix=img_prefix,
             ref_img_sampler=None, pipeline=test_pipeline, classes=classes),
    test=dict(type=dataset_type, ann_file=ann_root + 'box_track_rainy_all_cocofmt.json', img_prefix=img_prefix,
              ref_img_sampler=None, pipeline=test_pipeline, classes=classes))
