"""Recoverable cache queue for complete development/adaptation detector streams."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--detections',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--producer-pid',type=int,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    pending=[v for v in manifest['videos'] if v['dataset']==args.dataset
             and v['split'] in ['development','adaptation']]
    source='sunny' if args.dataset=='ua_detrac' else 'clear'
    while pending:
        progress=False
        for video in list(pending):
            directory=args.detections/video['domain']/video['split']/video['name']
            done=directory/'complete.json';cache=directory/'features_dense_current.npz'
            if cache.with_suffix('.npz.json').exists():pending.remove(video);progress=True;continue
            if not done.exists():continue
            report=json.loads(done.read_text())
            if report['frames']!=video['frames'] or not report['native_full_frames']:
                raise ValueError('Feature queue requires a complete native detector video')
            with (directory/'feature_extraction.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                annotation=str(directory/'prepared/annotations/val.json')
                # An extractor retained after restarting an old queue owns no
                # new lock; let its current video finish rather than duplicate it.
                active=False
                for process in Path('/proc').iterdir():
                    if not process.name.isdigit():continue
                    try:parts=(process/'cmdline').read_bytes().split(b'\0')
                    except OSError:continue
                    if b'jepa_cmot.dense_cache' in parts and annotation.encode() in parts:
                        active=True;break
                if active:continue
                if cache.with_suffix('.npz.json').exists():
                    pending.remove(video);progress=True;continue
                with (directory/'feature_extraction.log').open('a') as log:
                    subprocess.run([sys.executable,'-u','-m','jepa_cmot.dense_cache',
                        '--annotation',annotation,'--out',str(cache),'--dataset',args.dataset,
                        '--source-domain',source],stdout=log,stderr=subprocess.STDOUT,check=True)
            pending.remove(video);progress=True
            print('QUEUED_FEATURE_VIDEO_COMPLETE',args.dataset,video['domain'],video['name'],
                  'remaining',len(pending),flush=True)
        if not progress:
            missing=[v['name'] for v in pending if not
                (args.detections/v['domain']/v['split']/v['name']/'complete.json').exists()]
            if missing and not Path('/proc',str(args.producer_pid)).exists():
                raise RuntimeError('Detector producer exited with missing streams: '+','.join(missing))
            time.sleep(15)
    print('FEATURE_QUEUE_COMPLETE',args.dataset,flush=True)


if __name__=='__main__':main()
