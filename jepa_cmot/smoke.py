"""Real GPU smoke of the archived image -> tracking -> TrackEval pipeline."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='5')
    parser.add_argument('--attempt', default='v1')
    parser.add_argument('--dataset', default='ua_detrac')
    parser.add_argument('--domain', default='cloudy')
    args = parser.parse_args()
    output = ROOT / 'runs/JEPA_CMOT/phase_a_smoke' / args.dataset / args.attempt
    output.mkdir(parents=True, exist_ok=True)
    original = ROOT / 'GeneralTrack-main/datasets/generaltrack_prepared_thr02' / args.dataset / args.domain
    data = json.loads((original / 'annotations/val.json').read_text())
    video = min(data['videos'], key=lambda item: item['id'])
    images = sorted((image for image in data['images'] if image['video_id'] == video['id']),
                    key=lambda item: (item['frame_id'], item['id']))[:6]
    ids = {image['id'] for image in images}
    subset = dict(data, videos=[video], images=images,
                  annotations=[ann for ann in data['annotations'] if ann['image_id'] in ids])
    metadata = json.loads((original / 'metadata.json').read_text())
    for name, annotation in [('inference', dict(subset, annotations=[])), ('evaluation', subset)]:
        prepared = output / name
        (prepared / 'annotations').mkdir(parents=True, exist_ok=True)
        (prepared / 'annotations/val.json').write_text(json.dumps(annotation))
        (prepared / 'metadata.json').write_text(json.dumps(metadata))
    del data
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    common = ['--dataset', args.dataset, '--domain', args.domain,
              '--prepared-dir', str(output / 'inference'),
              '--restore_ckpt', str(ROOT / 'GeneralTrack-main/checkpoints/generaltrack_bdd.pth'),
              '--resize', '720', '1280']

    def run(name, command):
        print('START', name, flush=True)
        with (output / (name + '.log')).open('w') as log:
            result = subprocess.run([sys.executable, '-u'] + command, env=env,
                                    cwd=ROOT / 'GeneralTrack-main', stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed with {result.returncode}; inspect {output / (name + ".log")}')
        print('DONE', name, flush=True)

    for method, extra in [('B0', []), ('B2', [
        '--jepa-cache', str(output / 'features.npz'), '--jepa-feature-dim', '1664',
        '--jepa-mode', 'high_fallback_duplicate_geom_jepa', '--jepa-high-min-sim', '0.90',
        '--jepa-high-iou', '0.05', '--jepa-relative-margin', '0',
        '--jepa-history-aggregation', 'last', '--jepa-duplicate-iou', '0.20',
    ])]:
        if method == 'B2':
            run('feature_extraction', [str(ROOT / 'JEPA/build_generaltrack_vjepa21_cache.py'),
                '--prepared-dir', str(output / 'inference'), '--output', str(output / 'features.npz'),
                '--checkpoint', str(ROOT / 'JEPA/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt'),
                '--roi-mode', 'causal_box_crop_predictor', '--roi-context', '1.5',
                '--window', '16', '--predictor-pool', 'mean', '--crop-batch-size', '1',
                '--feature-score-min', '0.85', '--log-every', '1'])
        run(method + '_track', [str(ROOT / 'GeneralTrack-main/tools/track_damot_domain.py')]
            + common + ['--output-dir', str(output / method)] + extra)
        run(method + '_eval', [str(ROOT / 'GeneralTrack-main/tools/eval_damot_generaltrack.py'),
            '--dataset', args.dataset, '--domain', args.domain,
            '--prepared-dir', str(output / 'evaluation'),
            '--track-results', str(output / method / 'track_results'),
            '--output-dir', str(output / method / 'eval'),
            '--summary-csv', str(output / 'smoke_summary.csv'), '--method', method,
            '--protocol', 'engineering-smoke-6-frames-not-research-result'])
    (output / 'complete.json').write_text(json.dumps({'status': 'pass', 'frames': len(images),
        'gt_in_inference': False, 'feature_score_min_smoke_only': 0.85, 'gpu': args.gpu}))
    print('SMOKE_PASS', output, flush=True)


if __name__ == '__main__':
    main()
