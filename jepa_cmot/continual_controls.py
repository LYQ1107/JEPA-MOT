"""Matched continual controls share frozen perception and retain independent state."""
import argparse
import json
from pathlib import Path

from .continual_sequence import wait_inputs,command


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--detections',type=Path,required=True)
    parser.add_argument('--feature-producer-pid',type=int,required=True)
    parser.add_argument('--controls',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reverse',action='store_true')
    parser.add_argument('--teacher-calibration',type=Path)
    parser.add_argument('--compact-intermediate',action='store_true')
    parser.add_argument('--compact-native',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text());initial=json.loads(args.controls.read_text())
    source='sunny' if args.dataset=='ua_detrac' else 'clear'
    domains=['cloudy','rainy','night'] if args.dataset=='ua_detrac' else ['overcast','rainy','night']
    if args.reverse:domains.reverse()
    protocol=dict(dataset=args.dataset,order=[source]+domains,controls=initial,
        target_GT_in_training=False,sealed_test_used=False,
        evaluation='all saved stages, isolated frozen weights after training; no evaluation feedback')
    if args.teacher_calibration:
        protocol['fixed_teacher_calibration']=str(args.teacher_calibration.resolve())
        protocol['actual_update_counts_matched_each_frame']=True
    path=args.out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Continual controls protocol changed')
    path.write_text(json.dumps(protocol,indent=2))
    previous={s['name']:None for s in initial}
    snapshots=[{s['name']:s['checkpoint'] for s in initial}]
    if args.compact_native and not (args.out/'snapshots.json').exists():
        (args.out/'snapshots.json').write_text(json.dumps(snapshots,indent=2))
    for stage,domain in enumerate(domains,1):
        videos=[v for v in manifest['videos'] if v['dataset']==args.dataset and
                v['domain']==domain and v['split']=='adaptation']
        for video in videos:
            inputs=args.out/f'inputs/{domain}/{video["name"]}.json'
            wait_inputs([video],args.detections,args.feature_producer_pid,inputs)
            specs=[]
            for original in initial:
                spec=dict(original)
                if previous[spec['name']]:spec['resume_learner']=previous[spec['name']]
                specs.append(spec)
            config=args.out/f'controls/{domain}/{video["name"]}.json';config.parent.mkdir(parents=True,exist_ok=True)
            config.write_text(json.dumps(specs,indent=2))
            output=args.out/f'adaptation/stage{stage}_{domain}/{video["name"]}'
            runner='jepa_cmot.anchor_controls' if args.teacher_calibration else 'jepa_cmot.run_controls'
            arguments=[runner,'--inputs',str(inputs),'--controls',str(config),'--out',str(output)]
            if args.teacher_calibration:arguments+=['--teacher-calibration',str(args.teacher_calibration)]
            if (output/'complete.json').exists():
                # Completed videos need no replay of training commands. This
                # also permits retaining full state only at domain boundaries.
                for spec in initial:
                    report=json.loads((output/spec['name']/video['name']/'complete.json').read_text())
                    if report['frames']!=video['frames'] or report['GT_read']:
                        raise ValueError('Completed adaptation video is inconsistent')
            else:
                command(args.out/f'logs/adaptation/{domain}/{video["name"]}.log',arguments)
                if args.compact_native:
                    from .compact_checkpoints import compact
                    compact(args.out,completed_native=True)
            for spec in initial:
                previous[spec['name']]=str((output/spec['name']/video['name']/'learner.pt').resolve())
            print('CONTINUAL_CONTROL_VIDEO_COMPLETE',args.dataset,stage,domain,video['name'],flush=True)
        snapshots.append(dict(previous))
        (args.out/'snapshots.json').write_text(json.dumps(snapshots,indent=2))
        if args.compact_intermediate:
            from .compact_checkpoints import compact
            compact(args.out)
        print('CONTINUAL_CONTROL_STAGE_COMPLETE',args.dataset,stage,domain,flush=True)
    development=[v for v in manifest['videos'] if v['dataset']==args.dataset and v['split']=='development']
    inputs=args.out/'inputs/development.json'
    wait_inputs(development,args.detections,args.feature_producer_pid,inputs)
    matrix={s['name']:{} for s in initial}
    for stage,weights in enumerate(snapshots):
        specs=[dict(name=s['name'],method='trained',checkpoint=weights[s['name']],
                    calibration=s['calibration'],geometry_duplicates=True) for s in initial]
        config=args.out/f'controls/evaluation_stage{stage}.json';config.write_text(json.dumps(specs,indent=2))
        output=args.out/f'evaluation/stage{stage}'
        command(args.out/f'logs/evaluation_stage{stage}.log',['jepa_cmot.run_controls',
            '--inputs',str(inputs),'--controls',str(config),'--out',str(output)])
        for spec in specs:
            key=spec['name'];matrix[key][str(stage)]={}
            for domain in [source]+domains:
                metric_path=output/key/(domain+'_metrics.json')
                command(args.out/f'logs/metrics_stage{stage}_{key}_{domain}.log',
                    ['jepa_cmot.evaluate_stream','--manifest',str(args.manifest),'--tracks',str(output/key),
                    '--dataset',args.dataset,'--domain',domain,'--split','development','--out',str(metric_path)])
                matrix[key][str(stage)][domain]=json.loads(metric_path.read_text())['metrics']
        (args.out/'continual_matrix_partial.json').write_text(json.dumps(matrix,indent=2))
        print('CONTINUAL_CONTROL_FROZEN_EVALUATION_COMPLETE',args.dataset,stage,flush=True)
    forgetting={}
    for name,stages in matrix.items():
        values={}
        for domain in [source]+domains[:-1]:
            start=0 if domain==source else domains.index(domain)+1
            values[domain]=max(stages[str(stage)][domain]['HOTA'] for stage in range(start,4))-stages['3'][domain]['HOTA']
        forgetting[name]=dict(by_domain=values,average=sum(values.values())/len(values))
    (args.out/'complete.json').write_text(json.dumps(dict(protocol=protocol,matrix=matrix,
        forgetting=forgetting,reference='historical-best-after-domain-learned'),indent=2))
    print('CONTINUAL_CONTROLS_COMPLETE',args.dataset,flush=True)


if __name__=='__main__':main()
