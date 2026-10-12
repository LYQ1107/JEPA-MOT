"""Identical two-parameter source-only affinity calibration for every control."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from .models import HistoryPredictor,PredictorConfig
from .train_source import SourcePairs,sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--models',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--methods',nargs='+',default=['frozen_last','discriminative','predictive','predictive_ranking'])
    parser.add_argument('--reference-metrics',type=Path)
    args=parser.parse_args();torch.set_num_threads(4)
    metadata=json.loads(args.cache.with_suffix('.npz.json').read_text())
    if not metadata['gt_read'] or metadata['target_domains_read']:raise ValueError('Calibration is source-only')
    reference=json.loads((args.reference_metrics or args.models/'discriminative/metrics.json').read_text())
    width,height=(960,540) if args.dataset=='ua_detrac' else (1280,720)
    data=SourcePairs(args.cache,reference['config']['history'],reference['validation_videos'],width,height,
                     prediction_gaps=reference['prediction_gaps'])
    args.out.mkdir(parents=True,exist_ok=True)
    for method in args.methods:
        model=None;checkpoint=args.models/method/'final.pt'
        if method!='frozen_last':
            payload=torch.load(checkpoint,map_location='cpu');model=HistoryPredictor(PredictorConfig(**payload['config']))
            model.load_state_dict(payload['state_dict']);model.eval()
        values=[];labels=[]
        with torch.no_grad():
            for features,geometry,mask,candidates,valid in DataLoader(data,batch_size=128):
                query=features[:,-1] if model is None else model(features,geometry,mask)
                scores=torch.einsum('bd,bnd->bn',query,candidates)
                positive=torch.zeros_like(valid);positive[:,0]=True
                values.append(scores[valid]);labels.append(positive[valid].float())
        scores=torch.cat(values);target=torch.cat(labels)
        slope=torch.nn.Parameter(torch.tensor(8.))
        intercept=torch.nn.Parameter((scores[target>0].mean()+scores[target==0].mean())/2)
        optimizer=torch.optim.Adam([slope,intercept],lr=.05)
        positive_weight=(target==0).sum()/(target>0).sum()
        for step in range(200):
            logits=F.softplus(slope)*(scores-intercept)
            loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=positive_weight)
            optimizer.zero_grad();loss.backward();optimizer.step()
        report=dict(method=method,slope=float(F.softplus(slope).detach()),intercept=float(intercept.detach()),
            loss=float(loss.detach()),steps=200,trainable_parameters=2,fit_scope='source-validation-only',
            videos=reference['validation_videos'],cache_sha256=sha256(args.cache),
            checkpoint_sha256=None if model is None else sha256(checkpoint),
            target_gt_used=False,score='sigmoid(slope*(cosine-intercept))',
            source_feature_readout=metadata['roi_mode'],identity_predictor_parameters=0 if model is None else sum(p.numel() for p in model.parameters()))
        (args.out/(method+'.json')).write_text(json.dumps(report,indent=2));print('SOURCE_CALIBRATION_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
