"""Move our controller after its active child finishes a complete native run."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def live(process):
    return process.exists() and process.joinpath('stat').read_text().split()[2] != 'Z'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    parent = Path('/proc', str(args.pid))
    argv = [s.decode() for s in parent.joinpath('cmdline').read_bytes().split(b'\0') if s]
    if parent.stat().st_uid != os.getuid() or 'jepa_cmot.continual_controls' not in argv:
        raise ValueError('Only our registered continual controller may move')
    output = Path(argv[argv.index('--out') + 1])
    record = dict(old_pid=args.pid, gpu=args.gpu, argv=argv, output=str(output), status='WAITING_CURRENT_CHILD')
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2)+'\n')
    os.kill(args.pid, signal.SIGSTOP)
    children = parent.joinpath('task', str(args.pid), 'children').read_text().split()
    child_outputs = []
    for pid in children:
        process = Path('/proc', pid)
        parts = [s.decode() for s in process.joinpath('cmdline').read_bytes().split(b'\0') if s]
        if not any(m in parts for m in ['jepa_cmot.run_controls', 'jepa_cmot.anchor_controls']):
            os.kill(args.pid, signal.SIGCONT)
            raise ValueError('Migration must wait at a native tracking child boundary')
        child_outputs.append(Path(parts[parts.index('--out') + 1]))
    while any(live(Path('/proc', pid)) for pid in children):
        time.sleep(15)
    if any(not (p/'complete.json').is_file() for p in child_outputs):
        os.kill(args.pid, signal.SIGCONT)
        raise RuntimeError('Current child ended without a complete native output; controller resumed')
    record.update(completed_child_outputs=[str(p) for p in child_outputs], status='WAITING_FREE_GPU')
    args.record.write_text(json.dumps(record, indent=2)+'\n')
    while True:
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
            '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        if used < 100:
            break
        time.sleep(20)
    with args.record.with_suffix('.controller.log').open('a') as log:
        replacement = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu)), start_new_session=True)
    record.update(new_pid=replacement.pid, status='REPLACEMENT_RUNNING_OLD_CONTROLLER_RETAINED')
    args.record.write_text(json.dumps(record, indent=2)+'\n')
    print('CONTROLLER_MOVED_AT_NATIVE_COMPLETION', args.pid, replacement.pid, args.gpu, flush=True)
    # Existing queued jobs still observe the original PID until the actual
    # full experiment marker arrives. The old controller dispatches no work.
    while replacement.poll() is None:
        time.sleep(20)
    if replacement.returncode != 0 or not (output/'complete.json').exists():
        record.update(status='REPLACEMENT_FAILED_OLD_CONTROLLER_RETAINED', returncode=replacement.returncode)
        args.record.write_text(json.dumps(record, indent=2)+'\n')
        raise RuntimeError('Replacement failed; inspect its log before deciding how to resume')
    os.kill(args.pid, signal.SIGTERM)
    os.kill(args.pid, signal.SIGCONT)
    record.update(status='COMPLETE_OLD_CONTROLLER_RELEASED')
    args.record.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
