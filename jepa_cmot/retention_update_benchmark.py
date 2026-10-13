"""Time the real update call on identical recorded pseudo batches and cloned state."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from .online import ContinualLearner, OnlineConfig
from .train_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    payload=torch.load(args.checkpoint,map_location='cpu')
    original=OnlineConfig(**payload['online_config'])
    if original.retention!='none_graph' or not payload['memory'] or not payload['recent']:
        raise ValueError('Need a completed replay graph checkpoint with real memory and recent pairs')
    rng=np.random.RandomState(17);batches=[]
    for _ in range(55):
        batch=[payload['recent'][int(i)] for i in rng.randint(0,len(payload['recent']),4)]
        batch += [payload['memory'][int(i)] for i in rng.randint(0,len(payload['memory']),4)]
        if any(len(e)!=9 for e in batch):raise ValueError('Need actual recorded physical candidate/query fields')
        batches.append(batch)
    timings={}
    for retention in ['none_graph','features_graph','relations_graph','competition_graph','distance_graph']:
        learner=ContinualLearner(args.source,original)
        learner.resume(args.checkpoint)
        learner.config=replace(original,retention=retention)
        # Fixed4 recent+4 memory examples are inserted into the fresh queue so
        # the unchanged real update function consumes exactly the same8 rows.
        # Reservoir sampling and pseudo generation are intentionally excluded.
        learner.memory=[];learner.pending={};values=[];columns=[]
        for index,batch in enumerate(batches):
            learner.recent=list(batch);learner.frames=(index+1)*original.interval-1
            previous=learner.updates
            torch.cuda.synchronize();start=time.perf_counter()
            learner.after_assignment({},index+1)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            if learner.updates!=previous+1:raise ValueError('Timed call did not execute an optimizer step')
            if index>=5:
                values.append(elapsed);columns.append(learner.stats['losses'][-1].get('competition_columns',0))
        timings[retention]=dict(measured_updates=len(values),warmup_updates=5,
            mean_ms=float(np.mean(values)),p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),
            active_competition_columns=sum(columns))
        del learner;torch.cuda.empty_cache()
    result=dict(scope='controlled actual update-call benchmark on saved pseudo examples; not a live video or learning-performance experiment',
        checkpoint_sha256=sha256(args.checkpoint),source_sha256=sha256(args.source),timings=timings,
        same_starting_model_optimizer_and_batches=True,examples='four actual remaining recent plus four actual reservoir samples, same seeded indices for every loss',
        timing_includes='original after_assignment update branch, transfer/collation, history forward, loss, backward, clipping, optimizer and stats',
        excludes='image/encoder/detector/matcher, pseudo-pair construction, reservoir draw; fixed selected rows inserted into recent queue',
        device=torch.cuda.get_device_name(),shared_server=True,target_GT_read=False,
        benchmark_states_adopted_for_research=False,code_sha256=sha256(Path(__file__).with_name('online.py')))
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('ACTUAL_RETENTION_UPDATE_BENCHMARK_COMPLETE',timings,flush=True)


if __name__=='__main__':main()
