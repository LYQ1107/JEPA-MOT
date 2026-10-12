#!/usr/bin/env python3
"""Stage project code and selected checkpoints; keep datasets on the NAS."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAS = Path('/data1/liuyeqiang/nas/A100')
ARCHIVE = NAS / 'data1/LWR/vranlee/SERVER_ONLY/avis'
EXCLUDES = [
    '.git', '.codex', '.agents', '__pycache__', 'outputs', '/data', '/datasets',
    'checkpoints', 'weights', 'work_dirs', 'wandb', 'logs', 'pre',
    'baseline_darth*', 'Graph-Domain-Adaptaion', 'SPA-main', 'TACT-main',
    '*.log', '*.pt', '*.pth*', '*.npz', '*.zip', '*.tar*', '*.mp4', '*.avi',
]


def sync_code(source, target):
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['rsync', '-a', '--no-owner', '--no-group'] +
        ['--exclude=' + item for item in EXCLUDES] + [str(source) + '/', str(target) + '/'],
        check=True,
    )
    print('CODE_READY', target, flush=True)


def copy_checkpoint(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    partial = target.with_suffix(target.suffix + '.partial')
    if source.name == 'vjepa2_1_vitl_dist_vitG_384.pt':
        sidecar = ARCHIVE / 'JEPA/outputs/official_vjepa21_cache_mapped_source_detector_thr02_score0/motsynth_dark_vjepa21_l384_w16_causalbox_predictor.npz.json'
        expected_hash = json.loads(sidecar.read_text())['checkpoint_sha256']
        remote = '/home/A100/' + str(source.relative_to(NAS))
        batch = ROOT / 'checkpoint_transfer.sftp'
        batch.write_text(f'get -af "{remote}" "{partial}"\n')
        subprocess.run([
            'sftp', '-q', '-P', '24557', '-R', '64', '-B', '262144',
            '-i', '/home/liuyeqiang/.ssh/id_ed25519_nas_cieelab_readonly',
            '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'HostKeyAlias=nas-cieelab-readonly',
            '-o', 'UserKnownHostsFile=/home/liuyeqiang/.ssh/known_hosts_nas_cieelab_readonly',
            '-b', str(batch), 'Valadmin@127.0.0.1',
        ], check=True)
        local_hash = hashlib.sha256()
        with partial.open('rb') as inp:
            for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
                local_hash.update(block)
        if partial.stat().st_size != before.st_size or local_hash.hexdigest() != expected_hash:
            raise RuntimeError('Checkpoint hash differs from archived provenance: ' + str(source))
        partial.replace(target)
        shutil.copystat(source, target)
        print('CHECKPOINT_READY', target, before.st_size, flush=True)
        return {'source': str(source), 'local': str(target), 'bytes': before.st_size,
                'sha256': expected_hash, 'local_hash_verified': True,
                'verification_source': str(sidecar)}
    source_hash = hashlib.sha256()
    total = 0
    last_update = time.monotonic()
    with source.open('rb') as inp, partial.open('wb') as out:
        while True:
            block = inp.read(8 * 1024 * 1024)
            if not block:
                break
            source_hash.update(block)
            out.write(block)
            total += len(block)
            if time.monotonic() - last_update >= 30:
                print(f'CHECKPOINT_COPY {source.name}: {total / before.st_size:.1%}', flush=True)
                last_update = time.monotonic()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source checkpoint changed while copying: ' + str(source))
    local_hash = hashlib.sha256()
    with partial.open('rb') as inp:
        for block in iter(lambda: inp.read(8 * 1024 * 1024), b''):
            local_hash.update(block)
    if total != before.st_size or local_hash.digest() != source_hash.digest():
        raise RuntimeError('Checkpoint transfer verification failed: ' + str(source))
    partial.replace(target)
    shutil.copystat(source, target)
    print('CHECKPOINT_READY', target, total, flush=True)
    return {'source': str(source), 'local': str(target), 'bytes': total,
            'sha256': source_hash.hexdigest(), 'local_hash_verified': True}


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.readlink() != source:
            raise RuntimeError('Existing link points elsewhere: ' + str(target))
    elif not target.exists():
        target.symlink_to(source, target_is_directory=source.is_dir())
    else:
        raise RuntimeError('Link target already exists as a regular path: ' + str(target))


def main():
    if '--skip-code' not in sys.argv:
        sync_code(ARCHIVE / 'GeneralTrack-main', ROOT / 'GeneralTrack-main')
        sync_code(ARCHIVE / 'DA-MOT/darth-main', ROOT / 'DA-MOT/darth-main')
    shutil.copy2(ARCHIVE / 'DA-MOT/AGENTS.md', ROOT / 'DA-MOT/AGENTS.md')
    weights = [
        ('JEPA/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt',
         'JEPA/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt'),
        ('GeneralTrack-main/checkpoints/generaltrack_bdd.pth',
         'GeneralTrack-main/checkpoints/generaltrack_bdd.pth'),
        ('JEPA/outputs/official_vjepa21_adapter/temporal_source_vjepa21_predictor/temporal_adapter.pt',
         'JEPA/checkpoints/source_temporal_vjepa21/temporal_adapter.pt'),
    ]
    manifest = {'nas_root': str(NAS), 'workspace': str(ROOT),
                'datasets': 'NAS only; use archive_paths.resolve_archive_path for old symlinks',
                'checkpoints': [], 'complete': False}
    path = ROOT / 'workspace_manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    for source, target in weights:
        manifest['checkpoints'].append(copy_checkpoint(ARCHIVE / source, ROOT / target))
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    links = {
        'datasets_archive': ARCHIVE / 'DA-MOT',
        'GeneralTrack-main/datasets/generaltrack_prepared_mapped': ARCHIVE / 'GeneralTrack-main/datasets/generaltrack_prepared_mapped',
        'GeneralTrack-main/datasets/generaltrack_prepared_thr02': ARCHIVE / 'GeneralTrack-main/datasets/generaltrack_prepared_thr02',
        'GeneralTrack-main/datasets/generaltrack_prepared': ARCHIVE / 'GeneralTrack-main/datasets/generaltrack_prepared',
        'JEPA/outputs/jepa_motsynth_mapped_splits': ARCHIVE / 'JEPA/outputs/jepa_motsynth_mapped_splits',
        'JEPA/outputs/official_vjepa21_cache_mapped_source_detector_thr02_score0': ARCHIVE / 'JEPA/outputs/official_vjepa21_cache_mapped_source_detector_thr02_score0',
        'GeneralTrack-main/outputs/reference_archive': ARCHIVE / 'GeneralTrack-main/outputs',
        'DA-MOT/darth-main/data': ARCHIVE / 'DA-MOT/darth-main/data',
    }
    for target, source in links.items():
        link(source, ROOT / target)
    manifest['links'] = {k: str(v) for k, v in links.items()}
    manifest['complete'] = True
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('STAGING_COMPLETE', path, flush=True)


if __name__ == '__main__':
    main()
