"""Source-only contiguous clips with the same per-video image budget as uniform sampling."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def select_clips(images, count=8, length=16):
    images=sorted(images,key=lambda i:i['frame_id'])
    if len(images)<=count*length:
        starts=[0];length=len(images)
    else:
        starts=np.linspace(0,len(images)-length,count,dtype=int).tolist()
    selected=[];segment=0
    for start in starts:
        segment+=1;previous=None
        for original in images[start:start+length]:
            frame=int(original['frame_id'])
            if previous is not None and frame!=previous+1:segment+=1
            selected.append(dict(original,segment_id=segment))
            previous=frame
    return selected


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--reference-annotation',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());reference=json.loads(args.reference_annotation.read_text())
    source='sunny' if args.dataset=='ua_detrac' else 'clear'
    by_name={v['name']:v for v in manifest['videos'] if v['dataset']==args.dataset and
             v['domain']==source and v['split']=='source_train'}
    images=[];annotations=[];counts={}
    for video in reference['videos']:
        native=json.loads(Path(by_name[video['name']]['labels']).read_text())
        known=set(native['evaluation_frame_ids'])
        available=[i for i in native['images'] if i['frame_id'] in known]
        selected=select_clips(available)
        original_annotations=defaultdict(list)
        for annotation in native['annotations']:
            original_annotations[annotation['image_id']].append(annotation)
        for image in selected:
            old_id=image['id'];image=dict(image,id=len(images)+1,video_id=video['id']);images.append(image)
            for annotation in original_annotations[old_id]:
                annotations.append(dict(annotation,id=len(annotations)+1,image_id=image['id'],video_id=video['id']))
        counts[video['name']]=dict(frames=len(selected),segments=len(set(i['segment_id'] for i in selected)))
    result=dict(videos=reference['videos'],images=images,annotations=annotations,categories=reference['categories'],
                frame_sampling='8 evenly spaced contiguous clips of 16 known native frames per source video; reset histories between clips',
                gt_scope='source_train_only',source_domain=source)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result))
    report=dict(dataset=args.dataset,source_domain=source,frames=len(images),objects=len(annotations),
                video_counts=counts,selection_uses_target_GT=False,source_video_ids_match_reference=True,
                temporal_history_crosses_clip_boundary=False,source_image_budget_per_video=128)
    args.out.with_suffix('.selection.json').write_text(json.dumps(report,indent=2))
    print('SOURCE_CLIPS_PREPARED',json.dumps(report),flush=True)


if __name__=='__main__':main()
