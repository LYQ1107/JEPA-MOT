"""Resume source objective comparisons once the registered cache is complete."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--cache-producer-pid',type=int,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    metadata=args.cache.with_suffix(args.cache.suffix+'.json')
    while not metadata.exists():
        if not Path('/proc',str(args.cache_producer_pid)).exists():
            raise RuntimeError('Cache producer exited before its completion metadata')
        time.sleep(15)
    cache_metadata=json.loads(metadata.read_text())
    if cache_metadata['status']!='complete' or not cache_metadata['gt_read']:
        raise ValueError('Registered source cache did not complete')
    domain='sunny' if args.dataset=='ua_detrac' else 'clear'
    width,height=(960,540) if args.dataset=='ua_detrac' else (1280,720)
    for method in ['discriminative','predictive','predictive_ranking']:
        command=[sys.executable,'-u','-m','jepa_cmot.train_source','--cache',str(args.cache),
            '--out',str(args.out/method),'--method',method,'--source-domain',domain,
            '--steps','2000','--batch','64','--prediction-gaps','1','2','4','8',
            '--width',str(width),'--height',str(height)]
        with (args.out/f'{method}.log').open('a') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('FORMAL_SOURCE_OBJECTIVE_COMPLETE',args.dataset,method,flush=True)
    report={m:json.loads((args.out/m/'metrics.json').read_text()) for m in
            ['discriminative','predictive','predictive_ranking']}
    (args.out/'complete.json').write_text(json.dumps(report,indent=2))
    print('PHASE_B_SOURCE_COMPLETE',args.dataset,flush=True)


if __name__=='__main__':main()
