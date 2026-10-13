"""Reuse actual frozen source evaluations for identical stage0 policies only."""
import argparse
import hashlib
import json
from pathlib import Path

from .continual_sequence import wait_inputs
from .train_source import sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--sequence',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--detections',type=Path,required=True)
    args=parser.parse_args()
    reference=json.loads((args.reference/'complete.json').read_text())
    sequence=json.loads((args.sequence/'protocol.json').read_text())
    dataset=sequence['dataset']
    if reference['protocol']['dataset']!=dataset:raise ValueError('Reuse datasets differ')
    videos=[v for v in json.loads(args.manifest.read_text())['videos']
            if v['dataset']==dataset and v['split']=='development']
    inputs=args.sequence/'inputs/development.json'
    entries=wait_inputs(videos,args.detections,0,inputs)
    output=args.sequence/'evaluation/stage0'
    if list(output.glob('video_completion/*/complete.json')):
        raise ValueError('Do not replace an evaluation already in progress')
    specs=[dict(name=s['name'],method='trained',checkpoint=s['checkpoint'],
                calibration=s['calibration'],geometry_duplicates=True) for s in sequence['controls']]
    mappings={}
    for spec in specs:
        for old in reference['protocol']['controls']:
            if old['method']!='trained' or not old.get('geometry_duplicates'):continue
            if any(old.get(key) for key in ['feature_control','history_control','semantic_constant','affinity_control']):continue
            if sha256(old['checkpoint'])!=sha256(spec['checkpoint']):continue
            if json.loads(Path(old['calibration']).read_text())!=json.loads(Path(spec['calibration']).read_text()):continue
            mappings[spec['name']]=old['name'];break
        if spec['name'] not in mappings:raise ValueError('No identical actually evaluated source model: '+spec['name'])
    protocol=dict(inputs_sha256=hashlib.sha256(inputs.read_bytes()).hexdigest(),controls=specs,
        class_mapping='coco6-to-original-bdd8',reuse='frozen-fnet-correlation-compression-only',
        decision_and_learner_state='independent-per-control',GT_read=False)
    output.mkdir(parents=True,exist_ok=True);(output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    config=args.sequence/'controls/evaluation_stage0.json';config.parent.mkdir(parents=True,exist_ok=True)
    config.write_text(json.dumps(specs,indent=2));records=[]
    for video,entry in zip(videos,entries):
        saved={}
        for spec in specs:
            key=spec['name'];old_name=mappings[key]
            original=args.reference/'videos'/video['name']/old_name/video['name']
            report=json.loads((original/'complete.json').read_text())
            if report['frames']!=video['frames'] or report['updates'] or report['GT_read']:
                raise ValueError('Reuse requires an actual complete frozen native video')
            if sha256(original/'tracks.txt')!=report['output_sha256']:
                raise ValueError('Recorded frozen tracks changed')
            source_input=json.loads((args.reference/'inputs'/(video['name']+'.json')).read_text())
            if source_input!=[entry]:raise ValueError('Reuse detection/image/feature paths differ')
            target=output/key/video['name'];target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(original.resolve(),target_is_directory=True)
            saved[key]=report
            records.append(dict(control=key,reference_control=old_name,video=video['name'],
                reference=str(original.resolve()),tracks_sha256=report['output_sha256'],
                checkpoint_sha256=sha256(spec['checkpoint'])))
        marker=output/'video_completion'/video['name']/'complete.json'
        marker.parent.mkdir(parents=True,exist_ok=True);marker.write_text(json.dumps(saved,indent=2))
    (output/'actual_evaluation_reuse.json').write_text(json.dumps(dict(
        scope='identical static stage0 weights/calibration and native inputs; actual outputs reused',
        inputs_verified=True,checkpoints_and_calibrations_verified=True,records=records),indent=2))
    # The entire frozen evaluation is complete. Its canonical source learner
    # copies may have been removed after exact-weight cleanup; no replay or
    # optimizer resume is needed to reuse already verified immutable tracks.
    (output/'complete.json').write_text(json.dumps(dict(status='COMPLETE',
        controls=[s['name'] for s in specs],videos=len(videos),
        shared_image_pair_forwards=0,actual_frozen_evaluation_reused=True),indent=2))
    print('ACTUAL_FROZEN_STAGE0_REUSED',dataset,len(videos),len(specs),flush=True)


if __name__=='__main__':main()
