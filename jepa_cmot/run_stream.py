"""Run causal full-video GeneralTrack streams without opening evaluation labels."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize

if not hasattr(np,'float'):np.float=float
if not hasattr(np,'int'):np.int=int

from track_damot_domain import make_parser
from yolox.tracker.basetrack import BaseTrack
from .causal_tracker import CausalHistoryTracker
from .online import ContinualLearner,OnlineConfig

ROOT=Path(__file__).resolve().parents[1]
COCO6_TO_NATIVE8=np.asarray([0,2,3,4,6,7],np.int64)
NATIVE8_TO_COCO6={int(native):index+1 for index,native in enumerate(COCO6_TO_NATIVE8)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True,help='JSON list of prepared/cache paths only')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--method',choices=['native','frozen_last','trained','online'],required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--online-config',type=Path)
    parser.add_argument('--source-calibration',type=Path)
    parser.add_argument('--resume-learner',type=Path)
    parser.add_argument('--geometry-duplicates',action='store_true')
    parser.add_argument('--engineering-smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17);random.seed(17)
    entries=json.loads(args.inputs.read_text());args.out.mkdir(parents=True,exist_ok=True)
    learner=None
    if args.method in ['trained','online']:
        config=OnlineConfig(**json.loads(args.online_config.read_text())) if args.online_config else OnlineConfig()
        if args.method=='trained' and config.enabled:raise ValueError('Frozen evaluation cannot adapt')
        learner=ContinualLearner(args.checkpoint,config)
        if args.resume_learner:learner.resume(args.resume_learner)
    experiment=dict(method=args.method,inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        checkpoint=str(args.checkpoint),online_config=str(args.online_config),
        source_calibration=str(args.source_calibration),
        resume_learner=str(args.resume_learner),
        geometry_duplicates=args.geometry_duplicates,
        class_mapping='coco6-to-author-bdd8-v1',
        engineering_smoke_only=args.engineering_smoke)
    report_path=args.out/'complete.json'
    if report_path.exists():
        if json.loads(report_path.read_text())['experiment']!=experiment:raise ValueError('Completed run differs')
        print('STREAM_ALREADY_COMPLETE',report_path,flush=True);return
    sequence_reports=[];total_start=time.perf_counter()
    for entry in entries:
        name=entry['video'];directory=args.out/name;directory.mkdir(exist_ok=True)
        sequence_done=directory/'complete.json'
        if sequence_done.exists():
            sequence_reports.append(json.loads(sequence_done.read_text()))
            if learner is not None:learner.resume(directory/'learner.pt')
            continue
        prepared=Path(entry['prepared']);annotation=json.loads((prepared/'annotations/val.json').read_text())
        if annotation.get('annotations') or annotation.get('gt_read'):
            raise ValueError('Tracking input contains GT labels')
        images=sorted(annotation['images'],key=lambda i:i['frame_id'])
        if not args.engineering_smoke and (not annotation.get('native_full_frames') or len(images)!=entry['frames']):
            raise ValueError('Formal tracking requires every native image')
        cache=None;dimension=1664
        if args.method!='native':
            cache=np.load(entry['features'],allow_pickle=False)
            dimension=int(cache['features'].shape[1]);offsets=cache['offsets']
            if learner is not None and learner.model.config.feature_dim!=dimension:
                raise ValueError('Source predictor and target feature readout dimensions differ; use matching caches')
            if learner is not None and not args.engineering_smoke:
                source=learner.source_feature_metadata
                if source is None:raise ValueError('Source checkpoint is missing its feature provenance')
                target=json.loads(Path(entry['features']+'.json').read_text())
                for key in ['roi_mode','roi_context','checkpoint_sha256']:
                    if source[key]!=target[key]:raise ValueError('Source/target feature readout differs: '+key)
            indices={int(image_id):i for i,image_id in enumerate(cache['image_ids'])}
        native=make_parser().parse_args(['--dataset',entry['dataset'],'--domain',entry['domain'],
            '--prepared-dir',str(prepared),'--output-dir',str(directory),
            '--restore_ckpt',str(ROOT/'GeneralTrack-main/checkpoints/generaltrack_bdd.pth'),
            '--jepa-mode','none' if args.method=='native' else 'all',
            '--jepa-feature-dim',str(dimension),'--jepa-lambda','0.15','--jepa-history',
            str(4 if learner is None else learner.model.config.history)])
        native.roialign_size=tuple(native.roialign_size)
        native.geometry_duplicates=args.geometry_duplicates
        native.jepa_duplicate_iou=.2
        if args.source_calibration:
            native.source_calibration=json.loads(args.source_calibration.read_text())
            if native.source_calibration['target_gt_used']:raise ValueError('Calibration used target GT')
        BaseTrack._count=0
        if learner is not None:learner.reset_video()
        tracker=CausalHistoryTracker(native,learner)
        lines=[];start=time.perf_counter()
        with torch.no_grad():
            # The main frozen matcher uses no_grad; online updates explicitly
            # re-enable gradients after all current assignments have completed.
            for index,image in enumerate(images):
                raw=np.asarray(Image.open(image['file_name']).convert('RGB')).copy()
                tensor=torch.from_numpy(raw).permute(2,0,1).float()
                tensor=resize(tensor,[720,1280],antialias=False).unsqueeze(0).cuda()
                detections=np.asarray(image.get('detection',[]),np.float32).reshape(-1,6)
                detections[:,5]=COCO6_TO_NATIVE8[detections[:,5].astype(np.int64)]
                if cache is not None:
                    key=indices[int(image['id'])];features=cache['features'][offsets[key]:offsets[key+1]]
                    if len(features)!=len(detections):raise ValueError('Detection/feature offsets changed')
                    detections=np.concatenate([detections,np.asarray(features,np.float32)],axis=1)
                tracks=tracker.update(torch.from_numpy(detections),[image['height'],image['width']],
                                      [image['height'],image['width']],tensor)
                for track in tracks:
                    x,y,w,h=track.tlwh
                    if w*h<=native.min_box_area:continue
                    lines.append(f"{image['frame_id']},{track.track_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{track.score:.6f},{NATIVE8_TO_COCO6[int(track.cls)]},-1,-1\n")
                if (index+1)%100==0:print('TRACK_PROGRESS',name,index+1,len(images),flush=True)
        (directory/'tracks.txt').write_text(''.join(lines))
        if learner is not None:learner.save(directory/'learner.pt',video_complete=True)
        report=dict(video=name,frames=len(images),seconds=time.perf_counter()-start,
                    tracker_stats=tracker.jepa_stats,causal_stats=tracker.causal_stats,
                    output_sha256=hashlib.sha256((directory/'tracks.txt').read_bytes()).hexdigest(),
                    updates=0 if learner is None else learner.updates,gt_read=False)
        sequence_done.write_text(json.dumps(report,indent=2));sequence_reports.append(report)
        if cache is not None:cache.close()
        print('TRACK_VIDEO_COMPLETE',name,flush=True)
    report_path.write_text(json.dumps(dict(experiment=experiment,videos=sequence_reports,
        seconds=time.perf_counter()-total_start,learner_stats=None if learner is None else learner.stats,
        status='ENGINEERING_ONLY' if args.engineering_smoke else 'COMPLETE'),indent=2))
    print('STREAM_COMPLETE',report_path,flush=True)


if __name__=='__main__':main()
