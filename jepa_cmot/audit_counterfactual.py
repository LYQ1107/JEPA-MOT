"""Offline native-group GT diagnosis of recorded one-step assignment changes."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from eval_damot_generaltrack import box_iou_matrix


def group(native_class):
    return 0 if native_class<2 else 1 if native_class<6 else 2


def audit(labels,events,dataset):
    images={i['id']:i for i in labels['images']};known=set(labels['evaluation_frame_ids'])
    by_frame=defaultdict(list);mapping=[0,2,3,4,6,7]
    for annotation in labels['annotations']:
        if not annotation.get('iscrowd'):
            by_frame[images[annotation['image_id']]['frame_id']].append(annotation)
    def identity(frame,xyxy,native_class):
        if frame not in known:return None
        candidates=[a for a in by_frame[frame] if dataset=='ua_detrac' or
                    group(mapping[a['category_id']-1])==group(native_class)]
        if not candidates:return None
        x1,y1,x2,y2=xyxy
        scores=box_iou_matrix([[x1,y1,x2-x1,y2-y1]],[a['bbox'] for a in candidates])[0]
        index=int(np.argmax(scores))
        return int(candidates[index]['instance_id']) if scores[index]>=.5 else None
    cases=[];counts=Counter();branches=defaultdict(Counter)
    for event in events:
        counts['recorded_changes']+=1;frame=event['frame'];category=event['native_class']
        previous=[identity(row['frame'],row['box'],category) for row in event['history']]
        matched=[value for value in previous if value is not None]
        if not matched or frame not in known:
            counts['unscored_history_or_current']+=1;continue
        expected,frequency=Counter(matched).most_common(1)[0];purity=frequency/len(matched)
        if purity<.75:
            counts['mixed_identity_history']+=1;continue
        a=event['native_choice'];b=event['jepa_choice']
        original=None if a is None else identity(frame,a['box'],a['native_class'])
        changed=None if b is None else identity(frame,b['box'],b['native_class'])
        if original!=expected and changed==expected:kind='corrected_local_identity'
        elif original==expected and changed!=expected:kind='harmful_local_identity'
        elif original==expected and changed==expected:kind='both_correct'
        else:kind='neither_correct'
        counts[kind]+=1;branches[event['branch']][kind]+=1
        cases.append(dict(event,diagnosis=kind,dominant_history_GT_id=expected,history_purity=purity,
                          native_current_GT_id=original,jepa_current_GT_id=changed,
                          last_observation_gap=frame-max(row['frame'] for row in event['history'])))
    return dict(scope='offline diagnosis only; no GT fed to prediction, training or gates',
        definition='dominant prior identity with >=.75 purity; nearest IoU05 native-group-compatible GT box',
        limitation='one-step decisions on one tracker state; not full independent baseline or official IDSW/AssA',
        counts=dict(counts),by_branch={key:dict(value) for key,value in branches.items()},cases=cases)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--events',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();events=[json.loads(line) for line in args.events.read_text().splitlines()]
    report=audit(json.loads(args.labels.read_text()),events,args.dataset)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2))
    print('COUNTERFACTUAL_DIAGNOSIS_COMPLETE',report['counts'],flush=True)


if __name__=='__main__':main()
