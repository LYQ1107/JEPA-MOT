"""Deploy the scalar pair control only after real source training completes."""
import argparse
import json
from pathlib import Path
import time

from .continual_sequence import command


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training-pid',type=int,required=True)
    args=parser.parse_args();model=args.root/'models'/args.dataset/'seed17'
    while not (model/'metrics.json').exists():
        process=Path('/proc',str(args.training_pid))
        if not process.exists() or process.joinpath('stat').read_text().split()[2]=='Z':
            raise RuntimeError('Pair source training ended without a complete model')
        time.sleep(10)
    report=json.loads((model/'metrics.json').read_text())
    if report['status']!='complete' or report['target_gt_used']:raise ValueError('Invalid source pair experiment')
    command(args.root/(args.dataset+'_development.log'),
        ['jepa_cmot.development_controls','--dataset',args.dataset,
         '--manifest','runs/JEPA_CMOT/native_protocol_v3/manifest.json',
         '--detections','runs/JEPA_CMOT/formal_detection_streams_v1/'+args.dataset,
         '--controls','configs/JEPA_CMOT/'+args.dataset+'_pair_MLP_frozen.json',
         '--feature-producer-pid','20894','20960',
         '--out',str(args.root/'development'/(args.dataset+'_seed17'))])
    print('PAIR_FULL_DEVELOPMENT_COMPLETE',args.dataset,flush=True)


if __name__=='__main__':main()
