"""Source training and full frozen development deployment for paired readouts."""
import argparse
import json
from pathlib import Path
import time

from .continual_sequence import command
from .train_source import sha256


def cache_ready(directory, annotation):
    for readout in ['encoder', 'predictor']:
        path = directory/('features_'+readout+'_w4.npz.json')
        if not path.exists():
            return False
        metadata = json.loads(path.read_text())
        if metadata['engineering_smoke_only'] or metadata['window'] != 4:
            raise ValueError('Paired formal cache contains engineering or different-window inputs')
        if not metadata['precision'].startswith('FP32'):
            raise ValueError('Paired source/development readouts require the registered FP32 execution')
        if metadata['annotation_sha256'] != sha256(annotation):
            raise ValueError('Paired readout annotation changed')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--wait-pids', type=int, nargs='*', default=[])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    pending = list(args.wait_pids)
    while pending:
        for pid in pending[:]:
            process = Path('/proc', str(pid))
            if not process.exists() or process.joinpath('stat').read_text().split()[2] == 'Z':
                pending.remove(pid)
                continue
            argv = process.joinpath('cmdline').read_bytes().split(b'\0')
            if b'jepa_cmot.feature_queue' not in argv or b'ua_detrac' not in argv:
                raise ValueError('Waiting PID no longer belongs to the registered UA feature producer')
        if pending:
            time.sleep(20)
    manifest = json.loads(args.manifest.read_text())
    for dataset, source, width, height in [('bdd100k', 'clear', 1280, 720), ('ua_detrac', 'sunny', 960, 540)]:
        development = [v for v in manifest['videos'] if v['dataset'] == dataset and v['split'] == 'development']
        for video in development:
            original = Path('runs/JEPA_CMOT/formal_detection_streams_v1')/dataset/video['domain']/'development'/video['name']
            complete = json.loads((original/'complete.json').read_text())
            if complete['frames'] != video['frames'] or not complete['native_full_frames']:
                raise ValueError('Paired readout requires all original complete detector rows')
        annotation = Path('runs/JEPA_CMOT/source_clips_v1')/dataset/'source_annotation.json'
        source_cache = args.root/'source'/dataset
        if not cache_ready(source_cache, annotation):
            command(args.root/'logs'/(dataset+'_source_features.log'),
                ['jepa_cmot.paired_readout_cache', '--annotation', str(annotation), '--out', str(source_cache),
                 '--dataset', dataset, '--source-domain', source, '--source-gt', '--check-masked-target-invariance'])
        controls = {}
        for readout, hidden in [('encoder', 128), ('predictor', 79)]:
            cache = source_cache/('features_'+readout+'_w4.npz')
            models = args.root/'models'/dataset/readout/'seed17'
            calibration = args.root/'calibration'/dataset/readout/'seed17'
            for method in ['discriminative', 'predictive', 'predictive_ranking']:
                command(args.root/'logs'/(dataset+'_'+readout+'_'+method+'.log'),
                    ['jepa_cmot.train_source', '--cache', str(cache), '--out', str(models/method),
                     '--method', method, '--source-domain', source, '--seed', '17', '--hidden', str(hidden),
                     '--steps', '2000', '--batch', '64', '--prediction-gaps', '1', '2', '4', '8',
                     '--width', str(width), '--height', str(height)])
            from .compact_checkpoints import compact_source
            compact_source(models)
            command(args.root/'logs'/(dataset+'_'+readout+'_calibration.log'),
                ['jepa_cmot.calibrate_source', '--cache', str(cache), '--models', str(models),
                 '--out', str(calibration), '--dataset', dataset])
            specs = [dict(name=method, method='frozen_last' if method == 'frozen_last' else 'trained',
                          geometry_duplicates=True, calibration=str(calibration/(method+'.json')))
                     for method in ['frozen_last', 'discriminative', 'predictive', 'predictive_ranking']]
            for spec in specs:
                if spec['method'] == 'trained':
                    spec['checkpoint'] = str(models/spec['name']/'final.pt')
            config = args.root/'controls'/(dataset+'_'+readout+'.json')
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(json.dumps(specs, indent=2)+'\n')
            controls[readout] = config
        detections = args.root/'detections'/dataset
        for video in development:
            original = Path('runs/JEPA_CMOT/formal_detection_streams_v1')/dataset/video['domain']/'development'/video['name']
            directory = detections/video['domain']/'development'/video['name']
            directory.mkdir(parents=True, exist_ok=True)
            prepared = directory/'prepared'
            if not prepared.is_symlink():
                prepared.symlink_to((original/'prepared').resolve(), target_is_directory=True)
            annotation = original/'prepared/annotations/val.json'
            if not cache_ready(directory, annotation):
                command(args.root/'logs'/(dataset+'_'+video['name']+'_features.log'),
                    ['jepa_cmot.paired_readout_cache', '--annotation', str(annotation), '--out', str(directory),
                     '--dataset', dataset, '--source-domain', source, '--observed-domain', video['domain']])
        for readout in controls:
            output = args.root/'development'/(dataset+'_'+readout+'_seed17')
            if not (output/'complete.json').exists():
                command(args.root/'logs'/(dataset+'_'+readout+'_development.log'),
                    ['jepa_cmot.development_controls', '--dataset', dataset, '--manifest', str(args.manifest),
                     '--detections', str(detections), '--controls', str(controls[readout]),
                     '--feature-name', 'features_'+readout+'_w4.npz', '--feature-producer-pid', '0', '--out', str(output)])
            print('PAIRED_READOUT_FULL_DEVELOPMENT_COMPLETE', dataset, readout, flush=True)
    (args.root/'complete.json').write_text(json.dumps(dict(status='COMPLETE',
        datasets=['bdd100k', 'ua_detrac'], seed=17, sealed_test_used=False,
        protocol='paired w4 source clips, matching head capacity/budget, all frozen development videos'), indent=2)+'\n')


if __name__ == '__main__':
    main()
