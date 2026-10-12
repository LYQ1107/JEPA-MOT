"""Generate one immutable source-detector stream for all tracking controls."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from mmdet.apis import init_detector,inference_detector


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--split',nargs='+',default=['source_train','development','adaptation'])
    parser.add_argument('--video',default=None)
    parser.add_argument('--score',type=float,default=.1)
    parser.add_argument('--engineering-smoke',action='store_true')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    videos=[v for v in manifest['videos'] if v['dataset']==args.dataset and v['split'] in args.split
            and (args.video is None or v['name']==args.video)]
    if not videos:raise ValueError('No selected native videos')
    model=init_detector(str(args.config),str(args.checkpoint),device='cuda')
    digest=hashlib.sha256()
    with args.checkpoint.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):digest.update(b)
    checkpoint_sha=digest.hexdigest()
    for video in videos:
        out=args.out/video['domain']/video['split']/video['name']
        out.mkdir(parents=True,exist_ok=True)
        complete=out/'complete.json'
        if complete.exists():
            report=json.loads(complete.read_text())
            if report['checkpoint_sha256']!=checkpoint_sha or report['score_threshold']!=args.score:
                raise ValueError('Completed stream uses a different detector')
            continue
        stream=json.loads(Path(video['stream']).read_text())
        if stream['annotations'] or stream.get('gt_read'):
            raise ValueError('Detector inference input contains labels')
        start=time.perf_counter();frames=stream['images']
        if args.engineering_smoke:frames=frames[:6]
        for index,image in enumerate(frames):
            results=inference_detector(model,image['file_name'])
            rows=[]
            for category,detections in enumerate(results,1):
                if args.dataset=='ua_detrac' and category not in [2,3,4]:continue
                for detection in detections:
                    if float(detection[4])>=args.score:
                        rows.append(np.append(detection.astype(float),category-1).tolist())
            image['detection']=rows
            if (index+1)%100==0:print('DETECT_PROGRESS',video['name'],index+1,len(frames),flush=True)
        prepared=out/'prepared';(prepared/'annotations').mkdir(parents=True,exist_ok=True)
        annotation=prepared/'annotations/val.json'
        annotation.write_text(json.dumps(dict(stream,images=frames,
            native_full_frames=not args.engineering_smoke,
            engineering_smoke_only=args.engineering_smoke),separators=(',',':')))
        (prepared/'metadata.json').write_text(json.dumps(dict(image_root='/',dataset=args.dataset,
            domain=video['domain'],detector='new-source-only-FasterRCNN',gt_in_inference=False)))
        report=dict(status='complete',dataset=args.dataset,domain=video['domain'],split=video['split'],
            video=video['name'],frames=len(frames),native_full_frames=not args.engineering_smoke,
            engineering_smoke_only=args.engineering_smoke,checkpoint_sha256=checkpoint_sha,
            score_threshold=args.score,detections=sum(len(i['detection']) for i in frames),
            stream_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
            input_stream_sha256=video['stream_sha256'],seconds=time.perf_counter()-start,gt_read=False)
        complete.write_text(json.dumps(report,indent=2));print('DETECT_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
