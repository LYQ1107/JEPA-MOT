"""Sequential unlabeled adaptation, isolated snapshot evaluation, full matrices."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def wait_inputs(videos,detections,producer_pid,out):
    while True:
        entries=[];missing=[]
        for video in videos:
            root=detections/video['domain']/video['split']/video['name']
            if not (root/'features_dense_current.npz.json').exists():missing.append(video['name']);continue
            entries.append(dict(dataset=video['dataset'],domain=video['domain'],video=video['name'],
                frames=video['frames'],prepared=str((root/'prepared').resolve()),
                features=str((root/'features_dense_current.npz').resolve())))
        if not missing:
            out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(entries,indent=2));return entries
        if not Path('/proc',str(producer_pid)).exists():
            raise RuntimeError('Feature queue exited before required complete streams: '+','.join(missing))
        time.sleep(20)


def command(log,arguments):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable,'-u','-m']+arguments,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--detections',type=Path,required=True)
    parser.add_argument('--feature-producer-pid',type=int,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--online-config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reverse',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text())
    domains=['cloudy','rainy','night'] if args.dataset=='ua_detrac' else ['overcast','rainy','night']
    if args.reverse:domains.reverse()
    source='sunny' if args.dataset=='ua_detrac' else 'clear'
    protocol=dict(dataset=args.dataset,order=[source]+domains,
        checkpoint=str(args.checkpoint.resolve()),calibration=str(args.calibration.resolve()),
        online_config=json.loads(args.online_config.read_text()),gt_in_training=False,
        evaluation='isolated frozen snapshots after all adaptation; no evaluation feedback')
    protocol_path=args.out/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:
        raise ValueError('Existing continual experiment has a different protocol')
    protocol_path.write_text(json.dumps(protocol,indent=2))
    snapshots=[args.checkpoint];previous=None
    for stage,domain in enumerate(domains,1):
        selected=[v for v in manifest['videos'] if v['dataset']==args.dataset and
                  v['domain']==domain and v['split']=='adaptation']
        input_path=args.out/f'inputs/adaptation_{domain}.json'
        entries=wait_inputs(selected,args.detections,args.feature_producer_pid,input_path)
        output=args.out/f'adaptation/stage{stage}_{domain}'
        arguments=['jepa_cmot.run_stream','--inputs',str(input_path),'--out',str(output),
            '--method','online','--checkpoint',str(args.checkpoint),'--online-config',str(args.online_config),
            '--source-calibration',str(args.calibration),'--geometry-duplicates']
        if previous:arguments+=['--resume-learner',str(previous)]
        command(args.out/f'logs/adaptation_stage{stage}.log',arguments)
        previous=output/entries[-1]['video']/'learner.pt';snapshots.append(previous)
        print('CONTINUAL_ADAPTATION_STAGE_COMPLETE',args.dataset,stage,domain,flush=True)
    development=[v for v in manifest['videos'] if v['dataset']==args.dataset and v['split']=='development']
    eval_inputs=args.out/'inputs/development.json'
    wait_inputs(development,args.detections,args.feature_producer_pid,eval_inputs)
    matrix={}
    for stage,checkpoint in enumerate(snapshots):
        output=args.out/f'evaluation/stage{stage}'
        command(args.out/f'logs/evaluation_stage{stage}.log',['jepa_cmot.run_stream',
            '--inputs',str(eval_inputs),'--out',str(output),'--method','trained',
            '--checkpoint',str(checkpoint),'--source-calibration',str(args.calibration),'--geometry-duplicates'])
        matrix[str(stage)]={}
        for domain in [source]+domains:
            metric_path=output/(domain+'_metrics.json')
            command(args.out/f'logs/metrics_stage{stage}_{domain}.log',['jepa_cmot.evaluate_stream',
                '--manifest',str(args.manifest),'--tracks',str(output),'--dataset',args.dataset,
                '--domain',domain,'--split','development','--out',str(metric_path)])
            matrix[str(stage)][domain]=json.loads(metric_path.read_text())['metrics']
        (args.out/'continual_matrix_partial.json').write_text(json.dumps(matrix,indent=2))
        print('CONTINUAL_FROZEN_EVALUATION_COMPLETE',args.dataset,stage,flush=True)
    final=matrix[str(len(domains))];forgetting={}
    for domain in [source]+domains[:-1]:
        start=0 if domain==source else domains.index(domain)+1
        forgetting[domain]=max(matrix[str(stage)][domain]['HOTA'] for stage in range(start,len(domains)+1))-final[domain]['HOTA']
    (args.out/'complete.json').write_text(json.dumps(dict(protocol=protocol,matrix=matrix,
        forgetting_reference='historical-best-after-domain-learned',forgetting=forgetting,
        average_forgetting=sum(forgetting.values())/len(forgetting),sealed_test_used=False),indent=2))
    print('CONTINUAL_SEQUENCE_COMPLETE',args.dataset,flush=True)


if __name__=='__main__':main()
