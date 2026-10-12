"""Frozen current-image readout: one official encoder pass, object ROI pooling.

Two identical current images use the pretrained video tubelet projection. The
identity predictor is responsible for temporal prediction from track history.
Every detection remains offset-aligned; no label is read in detection mode.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from build_generaltrack_vjepa21_cache import (_letterbox_rgb,_normalize_frame,
    _temporal_maps,_pool_temporal_box_features,sha256_file,DEFAULT_MODEL)
from vjepa21_encoder import load_official_vjepa21_encoder,forward_tokens

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--source-domain',required=True)
    parser.add_argument('--source-gt',action='store_true')
    parser.add_argument('--feature-score-min',type=float,default=.35)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'JEPA/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt')
    args=parser.parse_args();torch.set_num_threads(4)
    data=json.loads(args.annotation.read_text())
    by_image=defaultdict(list)
    if args.source_gt:
        splits=json.loads((ROOT/'docs/JEPA_CMOT/VIDEO_SPLITS.json').read_text())
        allowed=set(splits['datasets'][args.dataset][args.source_domain]['source_train'])
        if not set(v['name'] for v in data['videos']).issubset(allowed):
            raise ValueError('Source cache contains a non-fitting video')
        for ann in data['annotations']:
            if not ann.get('iscrowd'):by_image[ann['image_id']].append(ann)
    elif data.get('annotations'):
        raise ValueError('Target detection cache must not read GT')
    images=sorted(data['images'],key=lambda i:(i['video_id'],i['frame_id'],i['id']))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    encoder=load_official_vjepa21_encoder(args.checkpoint,model_name=DEFAULT_MODEL,
        img_size=384,num_frames=2,patch_size=16,tubelet_size=2,device=torch.device('cuda'))
    features=[];offsets=[0];image_ids=[];frames=[];videos=[];tracks=[];boxes=[]
    start=time.perf_counter()
    for index,image in enumerate(images):
        rgb=cv2.cvtColor(cv2.imread(image['file_name']),cv2.COLOR_BGR2RGB)
        height,width=rgb.shape[:2]
        canvas,scale,left,top=_letterbox_rgb(rgb,384)
        frame=_normalize_frame(canvas);clip=torch.stack([frame,frame],dim=1).unsqueeze(0).cuda()
        rows=by_image[image['id']] if args.source_gt else image.get('detection',[])
        if args.source_gt:
            detections=np.asarray([[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],
                a['bbox'][1]+a['bbox'][3],1.,a['category_id']-1] for a in rows],np.float32).reshape(-1,6)
        else:detections=np.asarray(rows,np.float32).reshape(-1,6)
        with torch.inference_mode():
            tokens=forward_tokens(encoder,clip)
            maps=_temporal_maps(tokens,2,384,16)
            pooled=_pool_temporal_box_features(maps,detections,temporal_pool='last',image_width=width,
                image_height=height,scale=scale,pad_left=left,pad_top=top,patch_size=16,
                layer_count=1,roi_pool='box_mean')
        if not args.source_gt:
            pooled[detections[:,4]<args.feature_score_min]=0.
        features.append(np.asarray(pooled,np.float32));offsets.append(offsets[-1]+len(detections))
        if args.source_gt:
            image_ids.extend([image['id']]*len(rows));frames.extend([image['frame_id']]*len(rows))
            videos.extend([image['video_id']]*len(rows));tracks.extend([a['instance_id'] for a in rows])
            boxes.extend(detections[:,:4].tolist())
        else:image_ids.append(image['id'])
        if (index+1)%128==0:print('DENSE_PROGRESS',index+1,len(images),offsets[-1],flush=True)
    arrays=dict(features=np.concatenate(features),offsets=np.asarray(offsets),image_ids=np.asarray(image_ids))
    if args.source_gt:arrays.update(frame_ids=np.asarray(frames),video_ids=np.asarray(videos),
                                   track_ids=np.asarray(tracks),boxes=np.asarray(boxes,np.float32))
    np.savez_compressed(args.out,**arrays)
    metadata=dict(status='complete',roi_mode='dense_current',roi_context=1.,
        feature_shape=list(arrays['features'].shape),checkpoint_sha256=sha256_file(args.checkpoint),
        annotation_sha256=sha256_file(args.annotation),gt_read=args.source_gt,
        source_domain=args.source_domain,dataset=args.dataset,target_domains_read=[],
        causal=True,future_frames_read=False,physical_input='two-identical-current-images',
        seconds=time.perf_counter()-start,frames=len(images),engineering_smoke_only=False)
    metadata['feature_score_min']=args.feature_score_min
    args.out.with_suffix(args.out.suffix+'.json').write_text(json.dumps(metadata,indent=2))
    print('DENSE_CACHE_COMPLETE',json.dumps(metadata),flush=True)


if __name__=='__main__':main()
