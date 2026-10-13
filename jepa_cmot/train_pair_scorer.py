"""Source-supervised ordinary pair MLP and matched source-only score calibration."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from .pair_scorer import PairAssociationModel,PairConfig
from .train_source import SourcePairs,sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--reference-metrics',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17)
    metadata=json.loads(args.cache.with_suffix('.npz.json').read_text())
    reference=json.loads(args.reference_metrics.read_text())
    if not metadata['gt_read'] or metadata['target_domains_read'] or sha256(args.cache)!=reference['cache_sha256']:
        raise ValueError('Pair scorer needs the matching source-only cache')
    if (args.out/'metrics.json').exists():
        saved=json.loads((args.out/'metrics.json').read_text())
        if saved['cache_sha256']!=reference['cache_sha256'] or sha256(args.out/'final.pt')!=saved['checkpoint_sha256']:
            raise ValueError('Completed pair experiment differs')
        print('PAIR_SOURCE_ALREADY_COMPLETE',args.out,flush=True);return
    width,height=(960,540) if metadata['dataset']=='ua_detrac' else (1280,720)
    kwargs=dict(history=4,width=width,height=height,prediction_gaps=reference['prediction_gaps'])
    train=SourcePairs(args.cache,videos=reference['fit_videos'],**kwargs)
    valid=SourcePairs(args.cache,videos=reference['validation_videos'],**kwargs)
    args.out.mkdir(parents=True,exist_ok=True)
    model=PairAssociationModel().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    rng=np.random.RandomState(17);losses=[]
    for step in range(2000):
        batch=default_collate([train[int(i)] for i in rng.randint(0,len(train),64)])
        features,geometry,mask,candidates,available=[v.cuda() for v in batch]
        state=model(features,geometry,mask);logits=model.score(state,candidates).masked_fill(~available,-1e4)
        loss=F.cross_entropy(logits,torch.zeros(64,dtype=torch.long,device='cuda'))
        optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise FloatingPointError('Pair MLP update nonfinite')
        optimizer.step();losses.append(dict(step=step+1,ranking=float(loss.detach()),gradient_norm=float(norm)))
        if (step+1)%100==0:print('PAIR_SOURCE_PROGRESS',step+1,losses[-1],flush=True)
    model.eval();hits=0.;count=0;values=[];labels=[]
    with torch.no_grad():
        for batch in DataLoader(valid,batch_size=64):
            features,geometry,mask,candidates,available=[v.cuda() for v in batch]
            logits=model.score(model(features,geometry,mask),candidates).masked_fill(~available,-1e4)
            tied=torch.isclose(logits,logits.max(1).values[:,None],rtol=0,atol=1e-6)&available
            hits+=float((tied[:,0].float()/tied.sum(1)).sum());count+=len(features)
            positive=torch.zeros_like(available);positive[:,0]=True
            values.append(logits[available].cpu());labels.append(positive[available].float().cpu())
    checkpoint=args.out/'final.pt'
    torch.save(dict(state_dict=model.state_dict(),config=asdict(model.config),source_feature_metadata=metadata),checkpoint)
    scores=torch.cat(values);target=torch.cat(labels)
    slope=torch.nn.Parameter(torch.tensor(8.));intercept=torch.nn.Parameter((scores[target>0].mean()+scores[target==0].mean())/2)
    fit=torch.optim.Adam([slope,intercept],lr=.05);positive_weight=(target==0).sum()/(target>0).sum()
    for _ in range(200):
        calibration_loss=F.binary_cross_entropy_with_logits(F.softplus(slope)*(scores-intercept),target,pos_weight=positive_weight)
        fit.zero_grad();calibration_loss.backward();fit.step()
    calibration=dict(method='ordinary_pair_MLP',slope=float(F.softplus(slope).detach()),intercept=float(intercept.detach()),
        steps=200,trainable_parameters=2,fit_scope='source-validation-only',target_gt_used=False,
        score='sigmoid(slope*(pair_logit-intercept))',cache_sha256=reference['cache_sha256'],checkpoint_sha256=sha256(checkpoint))
    (args.out/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    report=dict(status='complete',architecture='history context then candidate-dependent scalar pair MLP',
        parameters=sum(p.numel() for p in model.parameters()),steps=2000,batch=64,seed=17,
        config=asdict(model.config),cache_sha256=reference['cache_sha256'],checkpoint_sha256=sha256(checkpoint),
        fit_videos=reference['fit_videos'],validation_videos=reference['validation_videos'],
        source_validation_rank1=hits/count,source_validation_prediction_error=None,
        prediction_error_note='scalar pair scorer has no latent forecast loss',target_gt_used=False,
        candidate_input_to_history_encoder=False,candidate_input_to_final_scorer=True,
        code_sha256={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('pair_scorer.py')]})
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.out/'losses.json').write_text(json.dumps(losses))
    print('PAIR_SCORER_SOURCE_COMPLETE',report['parameters'],report['source_validation_rank1'],flush=True)


if __name__=='__main__':main()
