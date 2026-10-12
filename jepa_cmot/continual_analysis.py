"""Summarize complete native domain matrices with explicit transfer definitions."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(matrix,order):
    count=len(order)
    if set(matrix)!={str(i) for i in range(count)}:
        raise ValueError('A final continual summary requires every stage')
    for stage in matrix.values():
        if set(stage)!=set(order):raise ValueError('A continual stage is missing a domain')
    values={metric:np.asarray([[matrix[str(stage)][domain][metric] for domain in order]
            for stage in range(count)],float) for metric in ['HOTA','AssA','IDF1']}
    report={}
    for metric,table in values.items():
        forgetting={domain:max(0.,float(table[j:count-1,j].max()-table[-1,j]))
                    for j,domain in enumerate(order[:-1])}
        signed_forgetting={domain:float(table[j:count-1,j].max()-table[-1,j])
                           for j,domain in enumerate(order[:-1])}
        backward={domain:float(table[-1,j]-table[j,j]) for j,domain in enumerate(order[:-1])}
        forward={domain:float(table[j-1,j]-table[0,j]) for j,domain in enumerate(order[1:],1)}
        report[metric]=dict(final_by_domain=dict(zip(order,table[-1].tolist())),
            final_equal_domain_average=float(table[-1].mean()),
            average_forgetting=float(np.mean(list(forgetting.values()))),forgetting_by_domain=forgetting,
            signed_average_forgetting=float(np.mean(list(signed_forgetting.values()))),
            signed_forgetting_by_domain=signed_forgetting,
            backward_transfer_by_domain=backward,backward_transfer=float(np.mean(list(backward.values()))),
            forward_transfer_by_domain=forward,forward_transfer=float(np.mean(list(forward.values()))))
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--complete',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();experiment=json.loads(args.complete.read_text())
    order=experiment['protocol']['order'];matrices=dict(experiment['matrix'])
    frozen={}
    for name,matrix in experiment['matrix'].items():
        initial=matrix['0']
        # This reference is copied from an actual complete stage0 evaluation.
        # Frozen parameters/calibration and per-video physical resets make it
        # independent of adaptation stage; no scores are inferred for new inputs.
        frozen[name]={str(i):initial for i in range(len(order))}
    result=dict(dataset=experiment['protocol']['dataset'],order=order,
        metrics={name:summarize(matrix,order) for name,matrix in matrices.items()},
        frozen_source_checkpoint_references={name:summarize(matrix,order) for name,matrix in frozen.items()},
        frozen_reference_scope='actual stage0 weights/inputs; repeat the same immutable evaluation at each stage',
        definitions=dict(final_average='equal weight per registered domain; no reweighting by video count',
            forgetting='max(0, best learned-domain score before final stage minus final score)',
            signed_forgetting='best learned-domain score before final minus final, may be negative',
            backward_transfer='final minus score immediately after that domain was learned; exclude final new domain',
            forward_transfer='score before learning each target domain minus its actual stage0 score'))
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print('CONTINUAL_ANALYSIS_COMPLETE',args.out,flush=True)


if __name__=='__main__':main()
