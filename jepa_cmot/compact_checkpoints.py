"""Keep complete domain-boundary states and all intermediate model parameters."""
import argparse
import json
from pathlib import Path

import torch

from .train_source import sha256


def compact(root, completed_native=False):
    root=Path(root)
    snapshots=json.loads((root/'snapshots.json').read_text())
    kept={str(Path(path).resolve()) for stage in snapshots for path in stage.values()}
    if completed_native:
        # Keep resume states for the latest complete videos, including an
        # in-progress domain. Earlier complete native videos need only weights.
        for stage in (root/'adaptation').glob('*'):
            videos=sorted([v for v in stage.iterdir() if (v/'complete.json').is_file()],
                          key=lambda v:(v/'complete.json').stat().st_mtime)
            for video in videos[-2:]:
                kept.update(str(p.resolve()) for p in video.glob('*/*/learner.pt'))
    completed_stages=len(snapshots)-1;records=[]
    report_path=root/'checkpoint_compaction.json'
    if report_path.exists():records=json.loads(report_path.read_text())['intermediate_states']
    for checkpoint in root.glob('adaptation/*/*/*/*/learner.pt'):
        relative=checkpoint.relative_to(root)
        stage=int(relative.parts[1].split('_')[0][5:])
        if (stage>completed_stages and not completed_native) or str(checkpoint.resolve()) in kept:continue
        if not (checkpoint.parents[2]/'complete.json').exists():continue
        payload=torch.load(checkpoint,map_location='cpu')
        if completed_native and (payload.get('physical_video_complete') is False or
                                 (payload.get('physical_video_complete') is True and payload['pending'])):
            raise ValueError('Native intermediate is not a complete physical video state')
        lightweight=checkpoint.with_name('learner_weights.pt')
        export={key:payload[key] for key in ['state_dict','config','online_config',
                                           'source_feature_metadata','frames','updates','stats']}
        export['scope']='intermediate weights only; full optimizer/replay preserved at each domain boundary'
        torch.save(export,lightweight)
        saved=torch.load(lightweight,map_location='cpu')
        if not all(torch.equal(value,saved['state_dict'][key]) for key,value in payload['state_dict'].items()):
            raise ValueError('Intermediate model export differs')
        record=dict(original=str(checkpoint.resolve()),original_sha256=sha256(checkpoint),
                    original_bytes=checkpoint.stat().st_size,weights=str(lightweight.resolve()),
                    weights_sha256=sha256(lightweight),weights_bytes=lightweight.stat().st_size,
                    frames=payload['frames'],updates=payload['updates'],
                    completion_evidence=('payload physical_video_complete and native marker'
                        if payload.get('physical_video_complete') else 'native complete marker; legacy payload schema'))
        checkpoint.unlink();records.append(record)
    scope=('completed native videos; source, domain boundaries and latest two videos per stage retain full states'
           if completed_native else 'completed adaptation stages only; source and domain-boundary full states retained')
    result=dict(scope=scope,
                intermediate_states=records,domain_snapshots_retained=sorted(kept),
                bytes_released=sum(r['original_bytes']-r['weights_bytes'] for r in records))
    report_path.write_text(json.dumps(result,indent=2))
    print('CHECKPOINTS_COMPACTED',root,'bytes_released',result['bytes_released'],flush=True)
    return result


def compact_source(root):
    """Retain every model weight and first/middle/final complete source states."""
    root=Path(root);records=[];retained=[];models=0
    report_path=root/'source_checkpoint_compaction.json'
    if report_path.exists():records=json.loads(report_path.read_text())['intermediate_states']
    active=set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:argv=(process/'cmdline').read_bytes().decode().split('\0')
        except (OSError,UnicodeDecodeError):continue
        if 'jepa_cmot.train_source' in argv and '--out' in argv:
            active.add(str(Path(argv[argv.index('--out')+1]).resolve()))
    for metric_path in root.glob('**/metrics.json'):
        metric=json.loads(metric_path.read_text());directory=metric_path.parent
        if metric.get('status')!='complete' or metric.get('engineering_smoke_only') is not False:continue
        if str(directory.resolve()) in active:continue
        final=directory/'final.pt'
        if sha256(final)!=metric['checkpoint_sha256']:
            raise ValueError('Completed source checkpoint differs from its metrics')
        checkpoints=sorted(directory.glob('step_*.pt'))
        if not checkpoints:continue
        final_step=int(metric['steps']);models+=1
        keep={int(checkpoints[0].stem.split('_')[1]),final_step//2,final_step}
        for checkpoint in checkpoints:
            step=int(checkpoint.stem.split('_')[1])
            if step in keep:
                retained.append(str(checkpoint.resolve()));continue
            payload=torch.load(checkpoint,map_location='cpu')
            if payload['experiment_fingerprint']!=metric['experiment_fingerprint']:
                raise ValueError('Source intermediate belongs to a different experiment')
            export={key:payload[key] for key in ['state_dict','config','step','method','experiment_fingerprint','losses','seconds']}
            export['scope']='source intermediate model weights; first/middle/final optimizer states and final.pt retained'
            lightweight=checkpoint.with_name('weights_'+checkpoint.name[5:])
            torch.save(export,lightweight);saved=torch.load(lightweight,map_location='cpu')
            if not all(torch.equal(value,saved['state_dict'][key]) for key,value in payload['state_dict'].items()):
                raise ValueError('Source intermediate model export differs')
            record=dict(original=str(checkpoint.resolve()),original_sha256=sha256(checkpoint),
                        original_bytes=checkpoint.stat().st_size,weights=str(lightweight.resolve()),
                        weights_sha256=sha256(lightweight),weights_bytes=lightweight.stat().st_size,step=step)
            checkpoint.unlink();records.append(record)
        retained.append(str(final.resolve()))
    result=dict(scope='completed formal source training only; every intermediate model weight retained',
                completed_models=models,intermediate_states=records,full_checkpoints_retained=sorted(retained),
                bytes_released=sum(r['original_bytes']-r['weights_bytes'] for r in records))
    report_path.write_text(json.dumps(result,indent=2))
    print('SOURCE_CHECKPOINTS_COMPACTED',root,'bytes_released',result['bytes_released'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    targets=parser.add_mutually_exclusive_group(required=True)
    targets.add_argument('--root',type=Path)
    targets.add_argument('--source-root',type=Path)
    parser.add_argument('--completed-native',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    if args.root:compact(args.root, completed_native=args.completed_native)
    else:compact_source(args.source_root)


if __name__=='__main__':main()
