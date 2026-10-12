"""Run registered full sequences after existing experiment completion markers."""
import argparse
import json
import os
from pathlib import Path
import time

from .continual_sequence import command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for job in plan['jobs']:
        controls = json.loads(Path(job['controls']).read_text())
        for control in controls:
            for field in ['checkpoint', 'calibration']:
                if not Path(control[field]).is_file():
                    raise FileNotFoundError(control[field])
        if len({c['name'] for c in controls}) != len(controls):
            raise ValueError('Duplicate control names')
    if args.validate_only:
        print('QUEUE_INPUTS_VALIDATED', args.plan, flush=True)
        return
    pending = list(plan['dependencies'])
    while pending:
        for dependency in pending[:]:
            if Path(dependency['complete']).is_file():
                pending.remove(dependency)
                continue
            process = Path('/proc', str(dependency['pid']))
            if not process.exists() or process.joinpath('stat').read_text().split()[2] == 'Z':
                raise RuntimeError('Dependency ended without completion: ' + dependency['complete'])
            argv = process.joinpath('cmdline').read_bytes().split(b'\0')
            if dependency['entrypoint'].encode() not in argv:
                raise RuntimeError('Dependency PID no longer belongs to the registered experiment')
        if pending:
            time.sleep(20)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(plan['gpu'])
    for job in plan['jobs']:
        out = Path(job['out'])
        if (out / 'complete.json').is_file():
            continue
        dataset = job['dataset']
        arguments = ['jepa_cmot.continual_controls', '--dataset', dataset,
                     '--manifest', 'runs/JEPA_CMOT/native_protocol_v3/manifest.json',
                     '--detections', 'runs/JEPA_CMOT/formal_detection_streams_v1/' + dataset,
                     '--feature-producer-pid', '0', '--controls', job['controls'],
                     '--compact-intermediate', '--out', str(out)]
        if job.get('reverse'):
            arguments.append('--reverse')
        command(out.parent / (out.name + '.log'), arguments)
        command(out.parent / (out.name + '_analysis.log'),
                ['jepa_cmot.continual_analysis', '--complete', str(out / 'complete.json'),
                 '--out', str(out / 'analysis.json')])
        print('QUEUED_FULL_SEQUENCE_COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
