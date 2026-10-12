"""Keep complete domain-boundary states and all intermediate model parameters."""
import argparse
import json
from pathlib import Path

import torch

from .train_source import sha256


def compact(root):
    root=Path(root)
    snapshots=json.loads((root/'snapshots.json').read_text())
    kept={str(Path(path).resolve()) for stage in snapshots for path in stage.values()}
    completed_stages=len(snapshots)-1;records=[]
    report_path=root/'checkpoint_compaction.json'
    if report_path.exists():records=json.loads(report_path.read_text())['intermediate_states']
    for checkpoint in root.glob('adaptation/*/*/*/*/learner.pt'):
        relative=checkpoint.relative_to(root)
        stage=int(relative.parts[1].split('_')[0][5:])
        if stage>completed_stages or str(checkpoint.resolve()) in kept:continue
        if not (checkpoint.parents[2]/'complete.json').exists():continue
        payload=torch.load(checkpoint,map_location='cpu')
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
                    frames=payload['frames'],updates=payload['updates'])
        checkpoint.unlink();records.append(record)
    result=dict(scope='completed adaptation stages only; source and domain-boundary full states retained',
                intermediate_states=records,domain_snapshots_retained=sorted(kept),
                bytes_released=sum(r['original_bytes']-r['weights_bytes'] for r in records))
    report_path.write_text(json.dumps(result,indent=2))
    print('CHECKPOINTS_COMPACTED',root,'bytes_released',result['bytes_released'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();compact(args.root)


if __name__=='__main__':main()
