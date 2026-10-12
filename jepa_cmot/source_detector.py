"""Train a source-only Faster R-CNN; never initialize from legacy all-video models."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ('pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle')
COCO_INDICES = [0, 2, 7, 5, 3, 1]
COCO_WEIGHT = Path('/home/liuyeqiang/.cache/torch/hub/checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth')


def source_annotation(manifest, dataset, destination, frames_per_video):
    selected = [v for v in manifest['videos'] if v['dataset']==dataset and v['split']=='source_train']
    images, annotations, video_records = [], [], []
    for video in selected:
        labels = json.loads(Path(video['labels']).read_text())
        video_records.extend(labels['videos'])
        known = set(labels['evaluation_frame_ids'])
        candidates = sorted((i for i in labels['images'] if i['frame_id'] in known),key=lambda i:i['frame_id'])
        # A fixed training budget may sample source frames; evaluation never does.
        indices = np.unique(np.linspace(0, len(candidates)-1,
                              min(frames_per_video,len(candidates)),dtype=int))
        remap = {}
        for index in indices:
            image = candidates[int(index)]; new_id = len(images)+1
            remap[image['id']] = new_id; images.append(dict(image,id=new_id))
        for ann in labels['annotations']:
            if ann['image_id'] in remap:
                annotations.append(dict(ann,id=len(annotations)+1,image_id=remap[ann['image_id']]))
    destination.write_text(json.dumps(dict(videos=video_records,images=images,annotations=annotations,
        categories=[dict(id=i+1,name=n) for i,n in enumerate(CLASSES)])))
    return [v['name'] for v in selected],len(images)


def configuration(annotation, out, dataset, steps, engineering):
    base = mmcv.Config.fromfile(str(ROOT/'DA-MOT/darth-main/configs/_base_/models/faster_rcnn_r50_fpn.py'))
    model = copy.deepcopy(base.model.detector)
    model.backbone.init_cfg = None
    model.roi_head.bbox_head.num_classes = 6
    size = (960,540) if dataset=='ua_detrac' else (1280,720)
    normalize = dict(mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375],to_rgb=True)
    pipeline = [dict(type='LoadImageFromFile'),dict(type='LoadAnnotations',with_bbox=True),
        dict(type='Resize',img_scale=size,keep_ratio=True),dict(type='RandomFlip',flip_ratio=.5),
        dict(type='Normalize',**normalize),dict(type='Pad',size_divisor=32),
        dict(type='DefaultFormatBundle'),dict(type='Collect',keys=['img','gt_bboxes','gt_labels'])]
    test_pipeline = [dict(type='LoadImageFromFile'),dict(type='MultiScaleFlipAug',img_scale=size,
        flip=False,transforms=[dict(type='Resize',keep_ratio=True),dict(type='RandomFlip'),
        dict(type='Normalize',**normalize),dict(type='Pad',size_divisor=32),
        dict(type='ImageToTensor',keys=['img']),dict(type='Collect',keys=['img'])])]
    cfg = mmcv.Config(dict(model=model,
        data=dict(samples_per_gpu=1 if engineering else 2,workers_per_gpu=0 if engineering else 2,
            train=dict(type='CocoDataset',ann_file=str(annotation),img_prefix='/',classes=CLASSES,pipeline=pipeline),
            test=dict(type='CocoDataset',ann_file=str(annotation),img_prefix='/',classes=CLASSES,pipeline=test_pipeline)),
        optimizer=dict(type='SGD',lr=.0025,momentum=.9,weight_decay=.0001),
        optimizer_config=dict(grad_clip=dict(max_norm=10,norm_type=2)),
        lr_config=dict(policy='step',warmup='linear',warmup_iters=min(100,steps//4),warmup_ratio=.1,
                       step=[max(1,int(steps*.7)),max(2,int(steps*.9))],by_epoch=False),
        runner=dict(type='IterBasedRunner',max_iters=steps),checkpoint_config=dict(interval=500,by_epoch=False),
        log_config=dict(interval=50,hooks=[dict(type='TextLoggerHook',by_epoch=False)]),
        workflow=[('train',1)],gpu_ids=[0],device='cuda',work_dir=str(out),seed=17,
        dist_params=dict(backend='nccl'),log_level='INFO',load_from=None,resume_from=None))
    # MMDetection's loader compatibility adapter expects a val declaration even
    # when validate=False. It points only to source fitting data and is unused.
    cfg.data.val = copy.deepcopy(cfg.data.test)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['ua_detrac','bdd100k'],required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--frames-per-video',type=int,default=128)
    parser.add_argument('--engineering-smoke',action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    annotation = args.out/'source_annotation.json'
    videos,count = source_annotation(manifest,args.dataset,annotation,args.frames_per_video)
    cfg = configuration(annotation,args.out,args.dataset,args.steps,args.engineering_smoke)
    cfg.dump(str(args.out/'detector_config.py'))
    set_random_seed(17,deterministic=True);torch.set_num_threads(4)
    model = build_detector(cfg.model);model.init_weights()
    state = torch.load(COCO_WEIGHT,map_location='cpu')['state_dict']
    # Transfer only public COCO class rows, including its background class.
    for suffix in ['weight','bias']:
        key='roi_head.bbox_head.fc_cls.'+suffix
        state[key]=state[key][COCO_INDICES+[80]]
        key='roi_head.bbox_head.fc_reg.'+suffix
        shape=state[key].shape
        state[key]=state[key].reshape(80,4,*shape[1:])[COCO_INDICES].reshape(24,*shape[1:])
    model.load_state_dict(state,strict=True)
    dataset = build_dataset(cfg.data.train);model.CLASSES=CLASSES
    latest = args.out/'latest.pth'
    if latest.exists():cfg.resume_from=str(latest)
    provenance=dict(dataset=args.dataset,source_video_names=videos,source_frames=count,steps=args.steps,
        seed=17,initialization='public-COCO-2x-only',target_gt_used=False,
        engineering_smoke_only=args.engineering_smoke,classes=CLASSES,
        annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest())
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    train_detector(model,[dataset],cfg,distributed=False,validate=False,
                   meta=dict(CLASSES=CLASSES,cmot_provenance=provenance))
    print('SOURCE_DETECTOR_COMPLETE',args.dataset,args.steps,flush=True)


if __name__=='__main__':
    main()
