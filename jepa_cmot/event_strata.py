"""Offline local correction strata with source-defined observed-motion bins."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .audit_counterfactual import audit
from .train_source import sha256


def speed(history, width, height):
    if len(history) < 2:return None
    a,b=history[-2:]
    center=lambda r:(np.asarray(r['box'][:2])+np.asarray(r['box'][2:]))/2
    return float(np.linalg.norm(center(b)-center(a))/np.hypot(width,height)/(b['frame']-a['frame']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--source-cache',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();protocol=json.loads((args.root/'protocol.json').read_text())
    with np.load(args.source_cache,allow_pickle=False) as data:
        video_ids=data['video_ids'];track_ids=data['track_ids'];frames=data['frame_ids'];boxes=data['boxes']
        groups=defaultdict(list)
        for i,(v,t) in enumerate(zip(video_ids,track_ids)):groups[int(v),int(t)].append(i)
        source_speeds=[]
        for indices in groups.values():
            ordered=sorted(indices,key=lambda i:frames[i])
            for a,b in zip(ordered,ordered[1:]):
                ca=(boxes[a,:2]+boxes[a,2:])/2;cb=(boxes[b,:2]+boxes[b,2:])/2
                source_speeds.append(float(np.linalg.norm(cb-ca)/np.hypot(1280,720)/(frames[b]-frames[a])))
    boundaries=np.percentile(source_speeds,[25,75]).tolist()
    counts={};excluded={}
    videos=[v for v in json.loads(args.manifest.read_text())['videos'] if v['dataset']=='bdd100k' and v['split']=='development']
    for spec in protocol['controls']:
        method=spec['name'];strata={'observation_gap':defaultdict(Counter),'observed_box_speed':defaultdict(Counter)};skipped=Counter()
        for video in videos:
            path=args.root/'videos'/video['name']/method/video['name']/'association_changes.jsonl'
            events=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            labels=json.loads(Path(video['labels']).read_text());report=audit(labels,events,'bdd100k')
            width,height=labels['images'][0]['width'],labels['images'][0]['height']
            for key in ['mixed_identity_history','unscored_history_or_current']:skipped[key]+=report['counts'].get(key,0)
            for case in report['cases']:
                gap=case['last_observation_gap'];gap_bin='1' if gap==1 else '2-4' if gap<=4 else '5-8' if gap<=8 else '>8'
                value=speed(case['history'],width,height)
                motion='unavailable' if value is None else 'low' if value<=boundaries[0] else 'middle' if value<=boundaries[1] else 'high'
                strata['observation_gap'][gap_bin][case['diagnosis']]+=1
                strata['observed_box_speed'][motion][case['diagnosis']]+=1
        counts[method]={name:{k:dict(v) for k,v in bins.items()} for name,bins in strata.items()};excluded[method]=dict(skipped)
    result=dict(scope='all recorded one-step assignment changes in complete BDD frozen development',
        source_motion_quantiles=boundaries,source_cache_sha256=sha256(args.source_cache),counts=counts,
        excluded_mixed_or_unscored=excluded,target_GT_used_only_for_offline_diagnosis=True,
        definition='past box-center displacement per native frame / image diagonal; source25/75 quantile bins',
        limitation='conditioned on changed decisions, not all opportunities or independent propagated baseline; gaps are observed-feature gaps, not measured occlusion; box speed mixes camera and object motion; no target-GT gate tuning',sealed_used=False)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('OFFLINE_EVENT_STRATA_COMPLETE',boundaries,flush=True)


if __name__=='__main__':main()
