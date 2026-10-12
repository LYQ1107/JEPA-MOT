"""Learn from one frozen causal teacher stream with exactly matched update counts.

The teacher decides physical associations; independently trained models are
evaluated later from their saved weights. No adaptation GT enters this runner.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import resize

from track_damot_domain import make_parser
from yolox.tracker.basetrack import BaseTrack
from .causal_tracker import CausalHistoryTracker
from .history import pack_history
from .online import ContinualLearner, OnlineConfig
from .run_stream import ROOT, COCO6_TO_NATIVE8, NATIVE8_TO_COCO6


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--teacher-calibration',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(17)
    entries=json.loads(args.inputs.read_text());specs=json.loads(args.controls.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    calibration=json.loads(args.teacher_calibration.read_text())
    if calibration['target_gt_used']:raise ValueError('Teacher calibration used target GT')
    learners={}
    for spec in specs:
        config=OnlineConfig(**spec['online_config'])
        if not config.enabled or config.reliability=='prediction_need':
            raise ValueError('Fixed-budget controls require updates and no hard need rejection')
        learner=ContinualLearner(spec['checkpoint'],config)
        if spec.get('resume_learner'):learner.resume(spec['resume_learner'])
        learners[spec['name']]=learner
    configs=[learner.config for learner in learners.values()]
    if len({(c.batch,c.interval,c.replay,c.seed) for c in configs})!=1:
        raise ValueError('Fixed teacher controls must share sample and update budgets')
    protocol=dict(inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),controls=specs,
        teacher='frozen-last with common native matcher, geometry, reliability and delay1',
        teacher_calibration_sha256=hashlib.sha256(args.teacher_calibration.read_bytes()).hexdigest(),
        GT_read=False,query_order='each learner predicts from teacher history before current rows are selected',
        adaptation_tracks='teacher outputs; learned deployment is evaluated independently')
    protocol['code_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ['anchor_controls.py','online.py','causal_tracker.py','history.py','models.py']}
    if any(learner.config.retention.endswith('_graph') for learner in learners.values()):
        protocol['code_sha256']['competition.py']=hashlib.sha256(Path(__file__).with_name('competition.py').read_bytes()).hexdigest()
    path=args.out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Anchor protocol changed')
    path.write_text(json.dumps(protocol,indent=2));reports={s['name']:[] for s in specs}
    for entry in entries:
        name=entry['video'];done=args.out/'video_completion'/name/'complete.json'
        if done.exists():
            saved=json.loads(done.read_text())
            for key,learner in learners.items():
                learner.resume(args.out/key/name/'learner.pt');reports[key].append(saved[key])
            continue
        annotation=json.loads((Path(entry['prepared'])/'annotations/val.json').read_text())
        if annotation.get('annotations') or annotation.get('gt_read'):raise ValueError('Teacher inputs contain GT')
        images=sorted(annotation['images'],key=lambda i:i['frame_id'])
        if len(images)!=entry['frames'] or not annotation['native_full_frames']:
            raise ValueError('Teacher requires complete native videos')
        with np.load(entry['features'],allow_pickle=False) as cache:
            features=cache['features'];offsets=cache['offsets'];image_ids=cache['image_ids']
        lookup={int(value):i for i,value in enumerate(image_ids)};dimension=features.shape[1]
        metadata=json.loads(Path(entry['features']+'.json').read_text())
        for learner in learners.values():
            if learner.model.config.feature_dim!=dimension:raise ValueError('Teacher feature dimension differs')
            for field in ['roi_mode','roi_context','checkpoint_sha256']:
                if learner.source_feature_metadata[field]!=metadata[field]:raise ValueError('Teacher readout differs')
            learner.reset_video()
        native=make_parser().parse_args(['--dataset',entry['dataset'],'--domain',entry['domain'],
            '--prepared-dir',entry['prepared'],'--output-dir',str(args.out/'teacher'/name),
            '--restore_ckpt',str(ROOT/'GeneralTrack-main/checkpoints/generaltrack_bdd.pth'),
            '--jepa-mode','all','--jepa-feature-dim',str(dimension),
            '--jepa-lambda','0.15','--jepa-history','4'])
        native.roialign_size=tuple(native.roialign_size);native.geometry_duplicates=True
        native.jepa_duplicate_iou=.2;native.source_calibration=calibration
        teacher=CausalHistoryTracker(native,None);BaseTrack._count=0
        lines=[];start=time.perf_counter()
        with torch.no_grad():
            for index,image in enumerate(images):
                current=teacher.frame_id+1
                existing=list(teacher.tracked_stracks)+list(teacher.lost_stracks)
                packed={}
                for track in existing:
                    rows=teacher._observations.get(int(track.track_id))
                    if rows:packed[int(track.track_id)]=pack_history(rows,current,4,dimension,image['width'],image['height'])
                queries={key:{} for key in learners}
                if packed:
                    ids=list(packed);batch=[np.stack([packed[tid][k] for tid in ids]) for k in range(3)]
                    for key,learner in learners.items():
                        queries[key]=dict(zip(ids,learner.predict(*batch)))
                # Current image/detection/feature rows are selected only after
                # every query is frozen; none is a predictor argument.
                raw=np.asarray(Image.open(image['file_name']).convert('RGB')).copy()
                tensor=resize(torch.from_numpy(raw).permute(2,0,1).float(),[720,1280],antialias=False).unsqueeze(0).cuda()
                detections=np.asarray(image['detection'],np.float32).reshape(-1,6)
                detections[:,5]=COCO6_TO_NATIVE8[detections[:,5].astype(int)]
                j=lookup[int(image['id'])];rows=np.asarray(features[offsets[j]:offsets[j+1]],np.float32)
                if len(rows)!=len(detections):raise ValueError('Teacher offsets changed')
                augmented=torch.from_numpy(np.concatenate([detections,rows],axis=1))
                tracks=teacher.update(augmented,[image['height'],image['width']],[image['height'],image['width']],tensor)
                for key,learner in learners.items():
                    accepted={tid:dict(sample,query=queries[key][tid].copy())
                              for tid,sample in teacher._accepted.items()}
                    with torch.enable_grad():learner.after_assignment(accepted,current)
                if len({learner.updates for learner in learners.values()})!=1:
                    raise ValueError('Fixed teacher controls diverged in actual update budget')
                for track in tracks:
                    x,y,w,h=track.tlwh
                    if w*h>100:
                        lines.append(f"{image['frame_id']},{track.track_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{track.score:.6f},{NATIVE8_TO_COCO6[int(track.cls)]},-1,-1\n")
                if (index+1)%100==0:print('ANCHOR_PROGRESS',name,index+1,len(images),flush=True)
        directory=args.out/'teacher'/name;directory.mkdir(parents=True,exist_ok=True)
        (directory/'tracks.txt').write_text(''.join(lines));saved={}
        for key,learner in learners.items():
            directory=args.out/key/name;directory.mkdir(parents=True,exist_ok=True)
            learner.save(directory/'learner.pt',video_complete=True)
            report=dict(video=name,frames=len(images),updates=learner.updates,GT_read=False,
                        teacher_causal_stats=teacher.causal_stats,learner_stats=learner.stats,
                        update_budget_matched_each_frame=True,adaptation_tracks_are_teacher=True)
            (directory/'complete.json').write_text(json.dumps(report,indent=2))
            saved[key]=report;reports[key].append(report)
        done.parent.mkdir(parents=True,exist_ok=True);done.write_text(json.dumps(saved,indent=2))
        print('ANCHOR_VIDEO_COMPLETE',name,time.perf_counter()-start,flush=True)
    for key in learners:
        (args.out/key/'complete.json').write_text(json.dumps(dict(protocol=protocol,videos=reports[key]),indent=2))
    (args.out/'complete.json').write_text(json.dumps(dict(status='COMPLETE',videos=len(entries),
        controls=list(learners),actual_update_counts_matched=True),indent=2))
    print('ANCHOR_CONTROLS_COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
