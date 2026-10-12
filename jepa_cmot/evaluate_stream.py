"""Offline class-aware TrackEval; native frames with absent XML remain unscored."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from eval_damot_generaltrack import HOTA,CLEAR,Identity,_build_sequence_data,box_iou_matrix


def ioa(boxes,regions):
    a=np.asarray(boxes,float).reshape(-1,4);b=np.asarray(regions,float).reshape(-1,4)
    if not len(a) or not len(b):return np.zeros((len(a),len(b)))
    lo=np.maximum(a[:,None,:2],b[None,:,:2])
    hi=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(-1)
    return intersection/np.maximum(a[:,2:].prod(-1)[:,None],1e-8)


def evaluate(labels,tracks,dataset):
    image_by_id={i['id']:i for i in labels['images']}
    frames=sorted(labels['evaluation_frame_ids']);gt=defaultdict(list);crowd=defaultdict(list)
    for a in labels['annotations']:
        frame=image_by_id[a['image_id']]['frame_id']
        target=crowd if a.get('iscrowd') else gt
        target[frame].append(dict(track_id=a['instance_id'],bbox=a['bbox'],category=a['category_id']))
    predictions=defaultdict(list)
    if Path(tracks).is_file():
        for line in Path(tracks).read_text().splitlines():
            v=line.split(',')
            predictions[int(v[0])].append(dict(track_id=int(v[1]),bbox=list(map(float,v[2:6])),category=int(v[7])))
    metrics=[HOTA(),CLEAR({'PRINT_CONFIG':False}),Identity()]
    categories=[1] if dataset=='ua_detrac' else list(range(1,7))
    classes={};ignored_predictions=0
    for category in categories:
        g,p=defaultdict(list),defaultdict(list)
        for frame in frames:
            current_gt=[a for a in gt[frame] if dataset=='ua_detrac' or a['category']==category]
            current_pred=[a for a in predictions[frame] if dataset=='ua_detrac' or a['category']==category]
            regions=list(labels.get('ignored_regions',[]))+[a['bbox'] for a in crowd[frame]
                                                          if dataset=='ua_detrac' or a['category']==category]
            ignored=set()
            if regions and current_pred:
                similarity=box_iou_matrix([a['bbox'] for a in current_gt],[a['bbox'] for a in current_pred])
                matched=set()
                if similarity.size:
                    rows,cols=linear_sum_assignment(-similarity)
                    matched={int(j) for i,j in zip(rows,cols) if similarity[i,j]>=.5}
                overlap=ioa([a['bbox'] for a in current_pred],regions).max(axis=1)
                ignored={i for i,value in enumerate(overlap) if value>=.5 and i not in matched}
            ignored_predictions+=len(ignored)
            g[(1,frame)]=current_gt;p[(1,frame)]=[a for i,a in enumerate(current_pred) if i not in ignored]
        data=_build_sequence_data(1,frames,g,p)
        classes[str(category)]={m.get_name():m.eval_sequence(data) for m in metrics}
    combined={m.get_name():m.combine_classes_det_averaged(
        {c:values[m.get_name()] for c,values in classes.items()}) for m in metrics}
    return dict(raw=combined,per_class=classes,native_frames=len(labels['images']),
                scored_frames=len(frames),unknown_annotation_frames=len(labels['images'])-len(frames),
                ignored_predictions=ignored_predictions)


def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    return value


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--tracks',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--domain',required=True)
    parser.add_argument('--split',default='development')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    selected=[v for v in manifest['videos'] if v['dataset']==args.dataset and v['domain']==args.domain
              and v['split']==args.split]
    reports={}
    for video in selected:
        directory=args.tracks/video['name']
        complete=json.loads((directory/'complete.json').read_text())
        if complete['frames']!=video['frames']:raise ValueError('Evaluation requires complete native video')
        reports[video['name']]=evaluate(json.loads(Path(video['labels']).read_text()),directory/'tracks.txt',args.dataset)
    combined={m.get_name():m.combine_sequences({v:r['raw'][m.get_name()] for v,r in reports.items()})
              for m in [HOTA(),CLEAR({'PRINT_CONFIG':False}),Identity()]}
    result=dict(dataset=args.dataset,domain=args.domain,split=args.split,videos=reports,raw=combined,
        metrics=dict(HOTA=100*float(np.mean(combined['HOTA']['HOTA'])),
                     AssA=100*float(np.mean(combined['HOTA']['AssA'])),
                     DetA=100*float(np.mean(combined['HOTA']['DetA'])),
                     IDF1=100*float(combined['Identity']['IDF1']),
                     MOTA=100*float(combined['CLEAR']['MOTA']),IDSW=int(combined['CLEAR']['IDSW'])),
        protocol='native-full-tracking-known-annotation-frames-class-aware-ignore-ioa05',
        gt_scope='offline-evaluation-only')
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(serializable(result),indent=2))
    print('EVALUATION_COMPLETE',result['metrics'],flush=True)


if __name__=='__main__':main()
