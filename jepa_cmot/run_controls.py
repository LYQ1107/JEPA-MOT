"""Compare controls on one causal stream, reusing only frozen image-pair work."""
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

from track_damot_domain import make_parser
from yolox.tracker.basetrack import BaseTrack
from .causal_tracker import CausalHistoryTracker
from .online import ContinualLearner,OnlineConfig
from .shared_matcher import SharedMatcher
from .run_stream import ROOT,COCO6_TO_NATIVE8,NATIVE8_TO_COCO6


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    torch.manual_seed(17);np.random.seed(17);random.seed(17)
    entries=json.loads(args.inputs.read_text());specs=json.loads(args.controls.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),controls=specs,
        class_mapping='coco6-to-original-bdd8',reuse='frozen-fnet-correlation-compression-only',
        decision_and_learner_state='independent-per-control',GT_read=False)
    protocol_path=args.out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Existing controls experiment differs')
    protocol_path.write_text(json.dumps(protocol,indent=2))
    learners={};reports={s['name']:[] for s in specs};shared=None
    for spec in specs:
        learner=None
        if spec['method'] in ['trained','online']:
            config=OnlineConfig(**spec.get('online_config',{}))
            if spec['method']=='trained' and config.enabled:raise ValueError('Frozen evaluation cannot update')
            learner=ContinualLearner(spec['checkpoint'],config)
            if spec.get('resume_learner'):learner.resume(spec['resume_learner'])
        learners[spec['name']]=learner
    for entry in entries:
        name=entry['video'];complete=args.out/'video_completion'/name/'complete.json'
        if complete.exists():
            saved=json.loads(complete.read_text())
            for spec in specs:
                key=spec['name'];reports[key].append(saved[key])
                if learners[key]:learners[key].resume(args.out/key/name/'learner.pt')
            continue
        annotation=json.loads((Path(entry['prepared'])/'annotations/val.json').read_text())
        if annotation.get('annotations') or annotation.get('gt_read'):raise ValueError('Control inputs contain GT')
        images=sorted(annotation['images'],key=lambda i:i['frame_id'])
        if len(images)!=entry['frames'] or not annotation['native_full_frames']:
            raise ValueError('Controls require a complete native video')
        with np.load(entry['features'],allow_pickle=False) as cache:
            features=cache['features'];offsets=cache['offsets'];image_ids=cache['image_ids']
        dimension=int(features.shape[1]);lookup={int(i):j for j,i in enumerate(image_ids)}
        trackers={};lines={s['name']:[] for s in specs};counters={s['name']:0 for s in specs}
        for spec in specs:
            key=spec['name'];learner=learners[key]
            if learner:
                if learner.model.config.feature_dim!=dimension:raise ValueError('Control feature dimensions differ')
                target=json.loads(Path(entry['features']+'.json').read_text());source=learner.source_feature_metadata
                for field in ['roi_mode','roi_context','checkpoint_sha256']:
                    if source[field]!=target[field]:raise ValueError('Control source/target readout differs')
                learner.reset_video()
            native=make_parser().parse_args(['--dataset',entry['dataset'],'--domain',entry['domain'],
                '--prepared-dir',entry['prepared'],'--output-dir',str(args.out/key/name),
                '--restore_ckpt',str(ROOT/'GeneralTrack-main/checkpoints/generaltrack_bdd.pth'),
                '--jepa-mode','none' if spec['method']=='native' else 'all',
                '--jepa-feature-dim',str(dimension),'--jepa-lambda','0.15','--jepa-history','4'])
            native.roialign_size=tuple(native.roialign_size);native.geometry_duplicates=spec.get('geometry_duplicates',True)
            native.jepa_duplicate_iou=.2
            native.history_control=spec.get('history_control','full')
            if spec.get('calibration'):
                native.source_calibration=json.loads(Path(spec['calibration']).read_text())
                if native.source_calibration['target_gt_used']:raise ValueError('Calibration used target GT')
            tracker=CausalHistoryTracker(native,learner)
            if shared is None:shared=SharedMatcher(tracker.model).eval()
            tracker.model=shared;trackers[key]=tracker
        start=time.perf_counter();initial_forwards=shared.image_pair_forwards
        with torch.no_grad():
            for index,image in enumerate(images):
                raw=np.asarray(Image.open(image['file_name']).convert('RGB')).copy()
                tensor=resize(torch.from_numpy(raw).permute(2,0,1).float(),[720,1280],antialias=False).unsqueeze(0).cuda()
                detections=np.asarray(image['detection'],np.float32).reshape(-1,6)
                detections[:,5]=COCO6_TO_NATIVE8[detections[:,5].astype(int)]
                j=lookup[int(image['id'])];f=np.asarray(features[offsets[j]:offsets[j+1]],np.float32)
                if len(f)!=len(detections):raise ValueError('Feature/detection offsets changed')
                plain=torch.from_numpy(detections)
                shared.begin_frame()
                for spec in specs:
                    key=spec['name'];tracker=trackers[key];BaseTrack._count=counters[key]
                    control_features=f
                    if spec.get('feature_control')=='shuffle':
                        seed=int(spec.get('seed',17))+int(image['id'])
                        control_features=f[np.random.RandomState(seed).permutation(len(f))]
                    elif spec.get('feature_control')=='random':
                        seed=int(spec.get('seed',17))+int(image['id'])
                        control_features=np.random.RandomState(seed).normal(size=f.shape).astype(np.float32)
                        control_features/=np.maximum(np.linalg.norm(control_features,axis=1,keepdims=True),1e-8)
                        control_features[np.linalg.norm(f,axis=1)<=1e-8]=0
                    augmented=torch.from_numpy(np.concatenate([detections,control_features],axis=1))
                    tracks=tracker.update(plain if spec['method']=='native' else augmented,
                        [image['height'],image['width']],[image['height'],image['width']],tensor)
                    counters[key]=BaseTrack._count
                    for track in tracks:
                        x,y,w,h=track.tlwh
                        if w*h<=100:continue
                        lines[key].append(f"{image['frame_id']},{track.track_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{track.score:.6f},{NATIVE8_TO_COCO6[int(track.cls)]},-1,-1\n")
                if (index+1)%100==0:print('CONTROLS_PROGRESS',name,index+1,len(images),len(specs),flush=True)
        saved={}
        for spec in specs:
            key=spec['name'];directory=args.out/key/name;directory.mkdir(parents=True,exist_ok=True)
            path=directory/'tracks.txt';path.write_text(''.join(lines[key]))
            if learners[key]:learners[key].save(directory/'learner.pt')
            report=dict(video=name,frames=len(images),causal_stats=trackers[key].causal_stats,
                tracker_stats=trackers[key].jepa_stats,updates=0 if learners[key] is None else learners[key].updates,
                output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),GT_read=False)
            (directory/'complete.json').write_text(json.dumps(report,indent=2));reports[key].append(report);saved[key]=report
        complete.parent.mkdir(parents=True,exist_ok=True);complete.write_text(json.dumps(saved,indent=2))
        print('CONTROLS_VIDEO_COMPLETE',name,time.perf_counter()-start,
              'shared_image_pair_forwards',shared.image_pair_forwards-initial_forwards,flush=True)
    for spec in specs:
        key=spec['name'];report=dict(status='COMPLETE',protocol=protocol,videos=reports[key],
            learner_stats=None if learners[key] is None else learners[key].stats,
            timing_scope='shared-comparison-execution-not-standalone-online-FPS')
        (args.out/key/'complete.json').write_text(json.dumps(report,indent=2))
    (args.out/'complete.json').write_text(json.dumps(dict(status='COMPLETE',controls=list(reports),
        videos=len(entries),shared_image_pair_forwards=shared.image_pair_forwards),indent=2))
    print('CONTROLS_COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
