"""Complete frozen controls as development caches arrive; never consume adaptation GT."""
import argparse
import json
import time
from pathlib import Path

from .continual_sequence import command
from .train_source import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--feature-producer-pid', type=int, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    specs = json.loads(args.controls.read_text())
    if any(s['method'] == 'online' or s.get('online_config', {}).get('enabled') for s in specs):
        raise ValueError('Development controls require frozen weights')
    videos = [v for v in json.loads(args.manifest.read_text())['videos']
              if v['dataset'] == args.dataset and v['split'] == 'development']
    args.out.mkdir(parents=True, exist_ok=True)
    weights = {s['name']: sha256(s['checkpoint']) for s in specs if s.get('checkpoint')}
    protocol = dict(dataset=args.dataset, split='development', controls=specs,
                    checkpoint_sha256=weights, sealed_test_used=False,
                    evaluation_GT_scope='offline metrics only', updates_enabled=False)
    path = args.out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Existing development protocol differs')
    path.write_text(json.dumps(protocol, indent=2))
    pending = list(videos)
    while pending:
        progress = False
        for video in list(pending):
            root = args.detections / video['domain'] / 'development' / video['name']
            features = root / 'features_dense_current.npz'
            if not features.with_suffix('.npz.json').exists():
                continue
            entry = dict(dataset=args.dataset, domain=video['domain'], video=video['name'],
                         frames=video['frames'], prepared=str((root / 'prepared').resolve()),
                         features=str(features.resolve()))
            inputs = args.out / 'inputs' / (video['name'] + '.json')
            inputs.parent.mkdir(parents=True, exist_ok=True)
            inputs.write_text(json.dumps([entry], indent=2))
            output = args.out / 'videos' / video['name']
            command(args.out / 'logs' / (video['name'] + '.log'),
                    ['jepa_cmot.run_controls', '--inputs', str(inputs),
                     '--controls', str(args.controls), '--out', str(output)])
            for spec in specs:
                directory = output / spec['name'] / video['name']
                report = json.loads((directory / 'complete.json').read_text())
                if report['updates'] != 0 or report['GT_read']:
                    raise ValueError('Frozen development evaluation changed learner state')
                link = args.out / 'tracks' / spec['name'] / video['name']
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.is_symlink():
                    link.symlink_to(directory.resolve(), target_is_directory=True)
            pending.remove(video)
            progress = True
            print('DEVELOPMENT_CONTROL_VIDEO_COMPLETE', args.dataset, video['domain'],
                  video['name'], 'remaining', len(pending), flush=True)
        if not progress:
            if not any(Path('/proc', str(pid)).exists() for pid in args.feature_producer_pid):
                raise RuntimeError('Feature producers exited with missing development caches: ' +
                                   ','.join(v['name'] for v in pending))
            time.sleep(20)
    matrix = {}
    for spec in specs:
        name = spec['name']
        matrix[name] = {}
        for domain in dict.fromkeys(v['domain'] for v in videos):
            output = args.out / 'metrics' / name / (domain + '.json')
            command(args.out / 'logs' / f'metrics_{name}_{domain}.log',
                    ['jepa_cmot.evaluate_stream', '--manifest', str(args.manifest),
                     '--tracks', str(args.out / 'tracks' / name), '--dataset', args.dataset,
                     '--domain', domain, '--split', 'development', '--out', str(output)])
            matrix[name][domain] = json.loads(output.read_text())['metrics']
    after = {s['name']: sha256(s['checkpoint']) for s in specs if s.get('checkpoint')}
    if after != weights:
        raise ValueError('Source checkpoints changed during frozen evaluation')
    (args.out / 'complete.json').write_text(json.dumps(dict(protocol=protocol, matrix=matrix,
        checkpoint_hashes_unchanged=True, videos=len(videos)), indent=2))
    print('DEVELOPMENT_CONTROLS_COMPLETE', args.dataset, flush=True)


if __name__ == '__main__':
    main()
