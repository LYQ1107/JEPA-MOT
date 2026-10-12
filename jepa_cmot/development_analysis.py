"""Equal-domain development summaries with stratified paired video uncertainty."""
import argparse
import json
from pathlib import Path

import numpy as np

from .paired_bootstrap import aggregate,restore


def compare_domains(reference,candidate,repeats=2000,seed=17):
    domains=list(reference)
    if set(domains)!=set(candidate):raise ValueError('Comparison domains differ')
    arrays={};point=[];rng=np.random.RandomState(seed)
    for domain in domains:
        left=reference[domain];right=candidate[domain];names=sorted(left['videos'])
        if names!=sorted(right['videos']):raise ValueError('Paired videos differ')
        for name in names:
            if left['videos'][name]['scored_frames']!=right['videos'][name]['scored_frames']:
                raise ValueError('Paired annotation coverage differs')
        a=[restore(left['videos'][name]['raw']) for name in names]
        b=[restore(right['videos'][name]['raw']) for name in names]
        indices=np.arange(len(names));arrays[domain]=(a,b)
        point.append(aggregate(b,indices)-aggregate(a,indices))
    samples=[]
    for _ in range(repeats):
        differences=[]
        for a,b in arrays.values():
            selected=rng.randint(0,len(a),len(a))
            differences.append(aggregate(b,selected)-aggregate(a,selected))
        samples.append(np.mean(differences,axis=0))
    estimate=np.mean(point,axis=0);lower,upper=np.percentile(samples,[2.5,97.5],axis=0)
    return dict(metrics={name:dict(delta=float(estimate[i]),percentile_95=[float(lower[i]),float(upper[i])])
        for i,name in enumerate(['HOTA','AssA','IDF1'])},repeats=repeats,seed=seed,
        estimator='paired videos resampled within each domain; aggregate TrackEval then equal-domain mean',
        scope='fixed-seed development uncertainty, no cross-seed or selection-adjusted claim')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--candidate',default='predictive')
    parser.add_argument('--references',nargs='+',default=['geometry_null','frozen_last','discriminative',
                                                       'predictive_last_only','frozen_shuffle','frozen_random'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();complete=json.loads((args.root/'complete.json').read_text())
    matrix=complete['matrix'];domains=list(next(iter(matrix.values())))
    reports={name:{domain:json.loads((args.root/'metrics'/name/(domain+'.json')).read_text())
                   for domain in domains} for name in matrix}
    result=dict(dataset=complete['protocol']['dataset'],scope='complete frozen development split',
        domains=domains,methods=matrix,equal_domain_average={name:{metric:float(np.mean([d[metric] for d in values.values()]))
            for metric in ['HOTA','AssA','IDF1']} for name,values in matrix.items()},
        videos_by_domain={d:list(reports[args.candidate][d]['videos']) for d in domains},
        frames_by_domain={d:sum(v['native_frames'] for v in reports[args.candidate][d]['videos'].values()) for d in domains},
        candidate=args.candidate,paired_comparisons={name:compare_domains(reports[name],reports[args.candidate])
                                                   for name in args.references},
        source_checkpoint_hashes_unchanged=complete['checkpoint_hashes_unchanged'],sealed_used=False)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print('FULL_DEVELOPMENT_ANALYSIS_COMPLETE',result['equal_domain_average'],flush=True)


if __name__=='__main__':main()
