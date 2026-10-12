"""Label-free current object crops, aligned to all detector rows and native frames."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from build_generaltrack_vjepa21_cache import _extract_box_crop_features,DEFAULT_MODEL
from vjepa21_encoder import load_official_vjepa21_encoder
from .run_stream import ROOT
from .train_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotation',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--source-domain',required=True)
    parser.add_argument('--domain')
    parser.add_argument('--batch',type=int,default=16)
    args=parser.parse_args();torch.set_num_threads(4)
    data=json.loads(args.annotation.read_text())
    if data.get('annotations') or data.get('gt_read'):raise ValueError('Target crop cache cannot read GT')
    images=sorted(data['images'],key=lambda i:(i['video_id'],i['frame_id'],i['id']))
    checkpoint=ROOT/'JEPA/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt'
    encoder=load_official_vjepa21_encoder(checkpoint,model_name=DEFAULT_MODEL,img_size=384,
        num_frames=2,patch_size=16,tubelet_size=2,device=torch.device('cuda'))
    features=[];offsets=[0];ids=[];start=time.perf_counter();encoded=0
    for index,image in enumerate(images):
        detections=np.asarray(image['detection'],np.float32).reshape(-1,6)
        valid=detections[:,4]>=.35
        rows=np.zeros((len(detections),1024),np.float32)
        if valid.any():
            rgb=cv2.cvtColor(cv2.imread(image['file_name']),cv2.COLOR_BGR2RGB)
            rows[valid]=_extract_box_crop_features(encoder,rgb,detections[valid],context=1.5,
                resolution=384,device=torch.device('cuda'),feature_layers='final',layer_count=1,
                batch_size=args.batch,temporal_replicates=2)
            encoded+=int(valid.sum())
        features.append(rows);offsets.append(offsets[-1]+len(rows));ids.append(image['id'])
        if (index+1)%50==0:print('CROP_PROGRESS',index+1,len(images),encoded,flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    arrays=dict(features=np.concatenate(features),offsets=np.asarray(offsets),image_ids=np.asarray(ids))
    np.savez_compressed(args.out,**arrays)
    report=dict(status='complete',roi_mode='box_crop',roi_context=1.5,feature_shape=list(arrays['features'].shape),
        checkpoint_sha256=sha256(checkpoint),annotation_sha256=sha256(args.annotation),gt_read=False,
        source_training_domain=args.source_domain,observed_video_domain=args.domain,
        dataset=args.dataset,target_GT_read=False,
        causal=True,future_frames_read=False,physical_input='two-identical-current-object-crops',
        seconds=time.perf_counter()-start,frames=len(images),encoded_objects=encoded,
        crop_batch_size=args.batch,feature_score_min=.35,
        native_full_frames=bool(data.get('native_full_frames')),
        engineering_smoke_only=bool(data.get('engineering_smoke_only')))
    args.out.with_suffix('.npz.json').write_text(json.dumps(report,indent=2))
    print('CURRENT_CROP_CACHE_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
