"""Combine completed, matched-input frozen readouts/heads; no partial split claims."""
import argparse
import json
from pathlib import Path

from .development_analysis import compare_domains


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--transformer',type=Path,required=True)
    parser.add_argument('--pair',type=Path,required=True)
    parser.add_argument('--prototypes',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();matrix={};reports={};protocols={}
    domains=['clear','overcast','rainy','night']
    for prefix,root in [('MLP',args.baseline),('Transformer',args.transformer),('Pair',args.pair),('Prototype',args.prototypes)]:
        complete=json.loads((root/'complete.json').read_text())
        if complete['protocol']['dataset']!='bdd100k' or complete['videos']!=19 or not complete['checkpoint_hashes_unchanged']:
            raise ValueError('Need actual complete frozen BDD development')
        protocols[prefix]=complete['protocol']
        for name,values in complete['matrix'].items():
            if set(values)!=set(domains):raise ValueError('A head is missing a full development domain')
            key=prefix+'/'+name;matrix[key]=values
            reports[key]={d:json.loads((root/'metrics'/name/(d+'.json')).read_text()) for d in domains}
    comparisons={}
    pairs=[('Transformer/predictive','Transformer/discriminative'),
           ('Transformer/predictive_ranking','Transformer/discriminative'),
           ('Transformer/predictive','MLP/predictive'),
           ('Transformer/discriminative','MLP/discriminative'),
           ('Pair/pair_MLP','MLP/discriminative'),
           ('Pair/pair_MLP','MLP/geometry_null'),
           ('Transformer/discriminative','MLP/geometry_null'),
           ('Transformer/predictive','MLP/geometry_null'),
           ('MLP/predictive','Pair/pair_MLP'),
           ('MLP/predictive','Prototype/frozen_ema')]
    for candidate,reference in pairs:
        comparisons[candidate+':'+reference]=compare_domains(reports[reference],reports[candidate])
    result=dict(status='COMPLETE_MATCHED_NATIVE_FROZEN_STRONG_HEAD_DEVELOPMENT',dataset='bdd100k',seed=17,
        videos=19,native_frames=3844,domain_metrics=matrix,
        equal_domain_average={name:{metric:sum(v[metric] for v in values.values())/4
            for metric in ['HOTA','AssA','IDF1']} for name,values in matrix.items()},
        actual_protocols=protocols,paired_comparisons=comparisons,
        limitation='fixed-seed development model exploration, no selection correction or cross-seed claim; frozen comparison is not continual performance',
        target_GT_in_training=False,sealed_used=False)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('STRONG_FROZEN_DEVELOPMENT_COMPARISONS_COMPLETE',flush=True)


if __name__=='__main__':main()
