"""Matched source budget and full native deployment for the temporal transformer."""
import argparse
import json
from pathlib import Path

from .compact_checkpoints import compact_source
from .continual_sequence import command


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args();dataset=args.dataset
    seed=str(args.seed)
    suffix='' if args.seed==17 else '_seed'+seed
    source,width,height=('sunny',960,540) if dataset=='ua_detrac' else ('clear',1280,720)
    cache=Path('runs/JEPA_CMOT/source_identity_cache_v1')/dataset/'features_dense_current.npz'
    models=args.root/'models'/dataset/('seed'+seed);calibration=args.root/'calibration'/dataset/('seed'+seed)
    for method in ['discriminative','predictive','predictive_ranking']:
        command(args.root/'logs'/(dataset+suffix+'_'+method+'.log'),
            ['jepa_cmot.train_source','--cache',str(cache),'--out',str(models/method),
             '--method',method,'--source-domain',source,'--architecture','transformer','--hidden','128',
             '--steps','2000','--batch','64','--seed',seed,'--prediction-gaps','1','2','4','8',
             '--width',str(width),'--height',str(height)])
    compact_source(models)
    command(args.root/'logs'/(dataset+suffix+'_calibration.log'),
        ['jepa_cmot.calibrate_source','--cache',str(cache),'--models',str(models),
         '--out',str(calibration),'--dataset',dataset])
    specs=[dict(name=method,method='trained',geometry_duplicates=True,
                checkpoint=str(models/method/'final.pt'),calibration=str(calibration/(method+'.json')))
           for method in ['discriminative','predictive','predictive_ranking']]
    config=args.root/'controls'/(dataset+suffix+'.json');config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(json.dumps(specs,indent=2)+'\n')
    command(args.root/'logs'/(dataset+suffix+'_development.log'),
        ['jepa_cmot.development_controls','--dataset',dataset,
         '--manifest','runs/JEPA_CMOT/native_protocol_v3/manifest.json',
         '--detections','runs/JEPA_CMOT/formal_detection_streams_v1/'+dataset,
         '--controls',str(config),'--feature-producer-pid','0',
         '--out',str(args.root/'development'/(dataset+'_seed'+seed))])
    print('TRANSFORMER_FULL_SOURCE_AND_DEVELOPMENT_COMPLETE',dataset,args.seed,flush=True)


if __name__=='__main__':main()
