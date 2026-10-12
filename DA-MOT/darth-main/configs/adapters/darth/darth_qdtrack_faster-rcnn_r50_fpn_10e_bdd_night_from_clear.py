_base_ = [
    '../../_base_/target_datasets/bdd_da_night.py',
    './darth_qdtrack_faster-rcnn_r50_fpn_12e_base.py',
]

model = dict(
    detector=dict(roi_head=dict(bbox_head=dict(num_classes=6))),
)
total_epochs = 10
