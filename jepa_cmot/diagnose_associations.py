"""Offline GT diagnostics only; results never enter online learning/selection gates."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from eval_damot_generaltrack import box_iou_matrix


def transitions(labels,path,dataset):
    images={i['id']:i for i in labels['images']};gt=defaultdict(list);pred=defaultdict(list)
    for a in labels['annotations']:
        if not a.get('iscrowd'):gt[images[a['image_id']]['frame_id']].append(a)
    for line in Path(path).read_text().splitlines():
        v=line.split(',');pred[int(v[0])].append(dict(id=int(v[1]),box=list(map(float,v[2:6])),category=int(v[7])))
    last={};events={};counts=dict(matches=0,continuations=0,breaks=0)
    for frame in sorted(labels['evaluation_frame_ids']):
        g,p=gt[frame],pred[frame]
        scores=box_iou_matrix([a['bbox'] for a in g],[a['box'] for a in p])
        if dataset=='bdd100k' and scores.size:
            scores*=np.asarray([[a['category_id']==b['category'] for b in p] for a in g])
        if not scores.size:continue
        rows,columns=linear_sum_assignment(-scores)
        for i,j in zip(rows,columns):
            if scores[i,j]<.5:continue
            identity=int(g[i]['instance_id']);track=p[j]['id'];counts['matches']+=1
            if identity in last:
                previous_frame,previous_track=last[identity]
                correct=track==previous_track
                events[(frame,identity)]=dict(continuous=correct,gap=frame-previous_frame,
                    previous_track=previous_track,current_track=track)
                counts['continuations' if correct else 'breaks']+=1
            last[identity]=(frame,track)
    return events,counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--tracks',type=Path,required=True)
    parser.add_argument('--video',required=True)
    parser.add_argument('--baseline',default='geometry_null')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();labels=json.loads(args.labels.read_text())
    baseline,base_counts=transitions(labels,args.tracks/args.baseline/args.video/'tracks.txt',args.dataset)
    result=dict(scope='offline-GT-diagnostic-not-training-input',baseline=args.baseline,
                video=args.video,definition='continuity relative to last IoU05-matched observation of same GT identity',
                baseline_counts=base_counts,methods={})
    for directory in args.tracks.iterdir():
        path=directory/args.video/'tracks.txt'
        if not path.exists() or directory.name==args.baseline:continue
        events,counts=transitions(labels,path,args.dataset);keys=set(baseline)&set(events)
        corrected=[k for k in keys if not baseline[k]['continuous'] and events[k]['continuous']]
        harmful=[k for k in keys if baseline[k]['continuous'] and not events[k]['continuous']]
        result['methods'][directory.name]=dict(counts=counts,common_evaluable_transitions=len(keys),
            corrected_continuity=len(corrected),harmful_continuity=len(harmful),
            corrected_events=[dict(frame=f,GT_id=i,gap=events[(f,i)]['gap']) for f,i in sorted(corrected)],
            harmful_events=[dict(frame=f,GT_id=i,gap=events[(f,i)]['gap']) for f,i in sorted(harmful)])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print('ASSOCIATION_DIAGNOSTICS_COMPLETE',
          {k:{f:v[f] for f in ['corrected_continuity','harmful_continuity']} for k,v in result['methods'].items()},flush=True)


if __name__=='__main__':main()
