"""Measure real image -> detector -> frozen encoder -> causal tracking/update.

The detector worker uses its existing torch/MMCV environment. Its IPC and all
image reads are included in end-to-end timing; no cached feature feeds inference.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import resize

from track_damot_domain import make_parser
from yolox.tracker.basetrack import BaseTrack
from build_generaltrack_vjepa21_cache import (_letterbox_rgb, _normalize_frame,
    _temporal_maps, _pool_temporal_box_features, DEFAULT_MODEL)
from vjepa21_encoder import load_official_vjepa21_encoder, forward_tokens
from .causal_tracker import CausalHistoryTracker
from .online import ContinualLearner, OnlineConfig
from .shared_matcher import SharedMatcher
from .run_stream import ROOT, COCO6_TO_NATIVE8, NATIVE8_TO_COCO6
from .train_source import sha256


class TimedLearner(ContinualLearner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.query_times=[];self.learning_times=[];self.update_times=[]

    def predict(self,*args):
        torch.cuda.synchronize();start=time.perf_counter()
        output=super().predict(*args)
        torch.cuda.synchronize();self.query_times.append(1000*(time.perf_counter()-start))
        return output

    def after_assignment(self,*args):
        previous=self.updates;torch.cuda.synchronize();start=time.perf_counter()
        super().after_assignment(*args)
        torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
        self.learning_times.append(elapsed)
        if self.updates>previous:self.update_times.append(elapsed)


def summary(values):
    return dict(count=len(values),mean=float(np.mean(values)) if values else None,
                p50=float(np.percentile(values,50)) if values else None,
                p95=float(np.percentile(values,95)) if values else None)


def gpu_process_snapshot():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    uuid=subprocess.check_output(['nvidia-smi','-i',visible,'--query-gpu=uuid',
                                  '--format=csv,noheader'],text=True).strip()
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                  '--format=csv,noheader,nounits'],text=True).splitlines()
    return [dict(pid=int(parts[0]),used_memory_mib=int(parts[2])) for parts in
            [line.split(',') for line in rows] if parts[1].strip()==uuid]


def current_features(encoder,rgb,detections):
    height,width=rgb.shape[:2];canvas,scale,left,top=_letterbox_rgb(rgb,384)
    frame=_normalize_frame(canvas)
    clip=torch.stack([frame,frame],dim=1).unsqueeze(0).cuda()
    with torch.inference_mode():
        tokens=forward_tokens(encoder,clip);maps=_temporal_maps(tokens,2,384,16)
        features=_pool_temporal_box_features(maps,detections,temporal_pool='last',
            image_width=width,image_height=height,scale=scale,pad_left=left,pad_top=top,
            patch_size=16,layer_count=1,roi_pool='box_mean')
    features[detections[:,4]<.35]=0
    return np.asarray(features,np.float32)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stream',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--domain',required=True)
    parser.add_argument('--video',required=True)
    parser.add_argument('--detector-checkpoint',type=Path,required=True)
    parser.add_argument('--detector-config',type=Path,required=True)
    parser.add_argument('--method',choices=['native','frozen_last','trained','online'],required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--online-config',type=Path)
    parser.add_argument('--geometry-duplicates',action='store_true')
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--engineering-frames',type=int)
    parser.add_argument('--pixel-transport',choices=['file','rgb'],default='file')
    parser.add_argument('--reference-prepared',type=Path)
    parser.add_argument('--reference-features',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(17)
    data=json.loads(args.stream.read_text())
    if data.get('annotations') or data.get('gt_read'):raise ValueError('Live stream contains GT')
    images=sorted(data['images'],key=lambda i:i['frame_id'])
    total_frames=len(images)
    if args.engineering_frames:images=images[:args.engineering_frames]
    args.out.mkdir(parents=True,exist_ok=True)
    learner=None
    if args.method in ['trained','online']:
        config=OnlineConfig(**json.loads(args.online_config.read_text())) if args.online_config else OnlineConfig()
        if config.enabled!=(args.method=='online'):raise ValueError('Live learning mode differs from config')
        learner=TimedLearner(args.checkpoint,config)
        source=learner.source_feature_metadata
        if learner.model.config.feature_dim!=1024 or source['roi_mode']!='dense_current' or source['roi_context']!=1.:
            raise ValueError('Live reader requires the matched dense-current 1024-D source model')
        learner.reset_video()
    calibration=json.loads(args.calibration.read_text()) if args.calibration else None
    if calibration and calibration['target_gt_used']:raise ValueError('Live calibration used target GT')
    reference={};cache=None
    if args.reference_prepared:
        reference_data=json.loads((args.reference_prepared/'annotations/val.json').read_text())
        if reference_data.get('annotations') or reference_data.get('gt_read'):raise ValueError('Parity reference contains GT')
        reference={int(i['id']):i for i in reference_data['images']}
    if args.reference_features:
        cache=np.load(args.reference_features,allow_pickle=False)
        cache_lookup={int(i):j for j,i in enumerate(cache['image_ids'])}
    official=ROOT/'JEPA/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt'
    protocol=dict(method=args.method,stream_sha256=sha256(args.stream),
        detector_checkpoint_sha256=sha256(args.detector_checkpoint),
        encoder_checkpoint_sha256=sha256(official) if args.method!='native' else None,
        source_checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None,
        calibration=calibration,online_config=None if learner is None else vars(learner.config),
        native_class_mapping='coco6-to-bdd8',geometry_duplicates=args.geometry_duplicates,
        current_reader='dense_current, two identical current images, final 1024-D',
        cached_input_used=False,GT_read=False,engineering_smoke_only=bool(args.engineering_frames),
        timing_scope='wall time including image reads, detector IPC, perception, association and actual updates',
        startup_and_weight_loading_excluded=True,warmup_frames=args.warmup,
        pixel_transport=args.pixel_transport,
        shared_image_pair_optimization='same frozen matcher caching available to every method')
    if learner and source['checkpoint_sha256']!=protocol['encoder_checkpoint_sha256']:
        raise ValueError('Live official feature checkpoint differs from source')
    path=args.out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Live benchmark protocol differs')
    path.write_text(json.dumps(protocol,indent=2))
    if (args.out/'complete.json').exists():print('LIVE_ALREADY_COMPLETE',args.out,flush=True);return
    native=make_parser().parse_args(['--dataset',args.dataset,'--domain',args.domain,
        '--prepared-dir',str(args.out),'--output-dir',str(args.out),
        '--restore_ckpt',str(ROOT/'GeneralTrack-main/checkpoints/generaltrack_bdd.pth'),
        '--jepa-mode','none' if args.method=='native' else 'all',
        '--jepa-feature-dim','1024','--jepa-lambda','0.15','--jepa-history','4'])
    native.roialign_size=tuple(native.roialign_size);native.geometry_duplicates=args.geometry_duplicates
    native.jepa_duplicate_iou=.2
    if calibration:native.source_calibration=calibration
    BaseTrack._count=0;tracker=CausalHistoryTracker(native,learner)
    shared=SharedMatcher(tracker.model).eval();tracker.model=shared
    encoder=None
    if args.method!='native':
        encoder=load_official_vjepa21_encoder(official,model_name=DEFAULT_MODEL,img_size=384,
            num_frames=2,patch_size=16,tubelet_size=2,device=torch.device('cuda'))
    detector_command=[str(ROOT/'runtime/darth_env/bin/python'),'-u','-m','jepa_cmot.live_detector',
        '--checkpoint',str(args.detector_checkpoint),'--config',str(args.detector_config),'--dataset',args.dataset]
    if args.pixel_transport=='rgb':detector_command.append('--return-pixels')
    logs=(args.out/'detector_worker.log').open('a')
    worker=subprocess.Popen(detector_command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),
        stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=logs,text=True,bufsize=1)
    records=[];lines=[];last_reply={};detector_error=0.;feature_error=0.
    try:
        ready=json.loads(worker.stdout.readline())
        if ready['status']!='READY':raise RuntimeError('Detector worker did not initialize')
        gpu_processes_before=gpu_process_snapshot()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for index,image in enumerate(images):
                torch.cuda.synchronize();start=time.perf_counter()
                tracker.prepare_queries(image['width'],image['height'])
                query_ms=1000*(time.perf_counter()-start)
                step=time.perf_counter()
                worker.stdin.write(json.dumps(dict(file_name=image['file_name']))+'\n');worker.stdin.flush()
                response=worker.stdout.readline()
                if not response:raise RuntimeError('Live detector exited; see detector_worker.log')
                reply=json.loads(response);last_reply=reply
                detections=np.asarray(reply['rows'],np.float32).reshape(-1,6)
                decoded=None
                if args.pixel_transport=='rgb':
                    decoded=np.frombuffer(base64.b64decode(reply['rgb_uint8_base64']),dtype=np.uint8).reshape(reply['rgb_shape']).copy()
                detector_rpc_ms=1000*(time.perf_counter()-step)
                step=time.perf_counter()
                raw=decoded if decoded is not None else np.asarray(Image.open(image['file_name']).convert('RGB')).copy()
                tensor=resize(torch.from_numpy(raw).permute(2,0,1).float(),[720,1280],antialias=False).unsqueeze(0).cuda()
                rgb=(raw if decoded is not None else cv2.cvtColor(cv2.imread(image['file_name']),cv2.COLOR_BGR2RGB)) if encoder is not None else None
                torch.cuda.synchronize();decode_resize_ms=1000*(time.perf_counter()-step)
                step=time.perf_counter()
                features=current_features(encoder,rgb,detections) if encoder is not None else None
                torch.cuda.synchronize();encoder_ms=1000*(time.perf_counter()-step)
                original=detections.copy();detections[:,5]=COCO6_TO_NATIVE8[detections[:,5].astype(int)]
                if features is not None:detections=np.concatenate([detections,features],axis=1)
                shared.begin_frame();step=time.perf_counter()
                previous=0 if learner is None else learner.updates
                tracks=tracker.update(torch.from_numpy(detections),[image['height'],image['width']],
                                      [image['height'],image['width']],tensor)
                torch.cuda.synchronize();tracking_ms=1000*(time.perf_counter()-step)
                for track in tracks:
                    x,y,w,h=track.tlwh
                    if w*h>100:
                        lines.append(f"{image['frame_id']},{track.track_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{track.score:.6f},{NATIVE8_TO_COCO6[int(track.cls)]},-1,-1\n")
                total_ms=1000*(time.perf_counter()-start)
                records.append(dict(frame=int(image['frame_id']),end_to_end_ms=total_ms,
                    query_preparation_ms=query_ms,detector_rpc_ms=detector_rpc_ms,
                    detector_inference_ms=1000*reply['seconds'],image_decode_resize_ms=decode_resize_ms,
                    current_encoder_ms=encoder_ms,tracking_including_learning_ms=tracking_ms,
                    updated=bool(learner and learner.updates>previous)))
                # Reference arrays are used only after the real pipeline timed
                # this frame. They never supply inference or training inputs.
                if reference:
                    saved=np.asarray(reference[int(image['id'])]['detection'],np.float32).reshape(-1,6)
                    if original.shape!=saved.shape:raise ValueError('Live detection count differs from reference')
                    detector_error=max(detector_error,float(np.max(np.abs(original-saved))) if saved.size else 0.)
                if cache is not None and features is not None:
                    j=cache_lookup[int(image['id'])];saved=cache['features'][cache['offsets'][j]:cache['offsets'][j+1]]
                    if saved.shape!=features.shape:raise ValueError('Live feature offsets differ from reference')
                    feature_error=max(feature_error,float(np.max(np.abs(features-saved))) if saved.size else 0.)
                if (index+1)%100==0:print('LIVE_PROGRESS',args.video,index+1,len(images),flush=True)
    finally:
        if worker.poll() is None:
            worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
        worker.wait(timeout=30);logs.close()
    gpu_processes_after=gpu_process_snapshot()
    (args.out/'tracks.txt').write_text(''.join(lines))
    if learner:learner.save(args.out/'learner.pt',video_complete=True)
    steady=records[min(args.warmup,len(records)):]
    report=dict(dataset=args.dataset,domain=args.domain,video=args.video,frames=len(records),
        native_frames_available=total_frames,native_full_frames=len(records)==total_frames,
        engineering_smoke_only=bool(args.engineering_frames),protocol=protocol,
        adapter_parameters=0 if learner is None else sum(p.numel() for p in learner.model.parameters()),
        actual_updates=0 if learner is None else learner.updates,
        fps_all_native_processed=1000*len(records)/sum(x['end_to_end_ms'] for x in records),
        fps_after_warmup=None if not steady else 1000*len(steady)/sum(x['end_to_end_ms'] for x in steady),
        latency_ms={key:summary([x[key] for x in steady]) for key in records[0] if key.endswith('_ms')},
        predictor_query_call_ms=summary([] if learner is None else learner.query_times),
        online_update_call_ms=summary([] if learner is None else learner.update_times),
        online_update_timing_includes='pseudo-pair processing, batch/replay, gradients and optimizer step',
        parent_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        detector_peak_allocated_bytes=last_reply['peak_allocated_bytes'],
        combined_peak_reserved_bytes=torch.cuda.max_memory_reserved()+last_reply['peak_reserved_bytes'],
        memory_scope='torch process allocations/reservations; CUDA driver context and unrelated processes excluded',
        causal_stats=tracker.causal_stats,detector_max_reference_error=detector_error,
        feature_max_reference_error=feature_error,reference_arrays_never_used_as_inputs=True,
        shared_image_pair_forwards=shared.image_pair_forwards,GT_read=False)
    report['gpu_processes_before']=gpu_processes_before;report['gpu_processes_after']=gpu_processes_after
    report['other_gpu_jobs_present']=any(p['pid'] not in [os.getpid(),worker.pid]
        for p in gpu_processes_before+gpu_processes_after)
    report['timing_hardware']=dict(name=torch.cuda.get_device_name(),visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                   torch=torch.__version__)
    (args.out/'frame_timings.json').write_text(json.dumps(records,indent=2))
    (args.out/'complete.json').write_text(json.dumps(report,indent=2))
    print('LIVE_STREAM_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
