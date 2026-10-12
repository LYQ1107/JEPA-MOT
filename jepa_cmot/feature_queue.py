"""Recoverable cache queue for complete development/adaptation detector streams."""
import argparse
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
            with (directory/'feature_extraction.log').open('a') as log:
                subprocess.run([sys.executable,'-u','-m','jepa_cmot.dense_cache',
                    '--annotation',str(directory/'prepared/annotations/val.json'),
                    '--out',str(cache),'--dataset',args.dataset,'--source-domain',source],
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            pending.remove(video);progress=True
            print('QUEUED_FEATURE_VIDEO_COMPLETE',args.dataset,video['domain'],video['name'],
                  'remaining',len(pending),flush=True)
        if not progress:
            if not Path('/proc',str(args.producer_pid)).exists():
                missing=[v['name'] for v in pending]
                raise RuntimeError('Detector producer exited with missing streams: '+','.join(missing))
            time.sleep(15)
    print('FEATURE_QUEUE_COMPLETE',args.dataset,flush=True)


if __name__=='__main__':main()
