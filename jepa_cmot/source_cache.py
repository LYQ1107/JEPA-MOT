"""Prepare native-frame source GT crops for engineering or registered training."""
import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', default='5')
    parser.add_argument('--engineering-smoke', action='store_true')
    args = parser.parse_args()
    inventory = json.loads((ROOT / 'docs/JEPA_CMOT/DATA_INVENTORY.json').read_text())
    split_file = ROOT / 'docs/JEPA_CMOT' / ('PROPOSED_VIDEO_SPLITS.json' if args.engineering_smoke else 'VIDEO_SPLITS.json')
    splits = json.loads(split_file.read_text())
    if not args.engineering_smoke and splits['status'] != 'frozen':
        raise ValueError('Formal extraction requires frozen splits and checkpoint provenance')
    selected = splits['datasets']['ua_detrac']['sunny']['source_train']
    if args.engineering_smoke:selected = selected[:2]
    rows = {record['video']: record for record in inventory['ua_detrac']}
    videos, images, annotations = [], [], []
    for video_id, name in enumerate(selected, start=1):
        record = rows[name]
        xml = ET.parse(record['xml']).getroot()
        videos.append({'id': video_id, 'name': name, 'source_domain': 'sunny'})
        frames = xml.findall('frame')
        if args.engineering_smoke:frames = frames[:8]
        for frame in frames:
            frame_id = int(frame.attrib['num'])
            image_id = len(images) + 1
            images.append({'id': image_id, 'video_id': video_id, 'frame_id': frame_id,
                           'file_name': str(Path(record['image_root']) / f'img{frame_id:05d}.jpg'),
                           'width': 960, 'height': 540})
            for target in frame.findall('./target_list/target'):
                box = target.find('box').attrib
                annotations.append({'id': len(annotations) + 1, 'video_id': video_id,
                    'image_id': image_id, 'instance_id': int(target.attrib['id']),
                    'bbox': [float(box[k]) for k in ['left', 'top', 'width', 'height']],
                    'category_id': 1, 'iscrowd': 0})
    args.out.mkdir(parents=True, exist_ok=True)
    annotation = args.out / 'source_annotation.json'
    annotation.write_text(json.dumps({'videos': videos, 'images': images, 'annotations': annotations,
                                      'categories': [{'id': 1, 'name': 'vehicle'}]}))
    command = [sys.executable, '-u', str(ROOT / 'JEPA/build_uadetrac_source_gt_vjepa21_cache.py'),
        '--annotation', str(annotation), '--image-root', '/', '--out', str(args.out / 'features.npz'),
        '--checkpoint', str(ROOT / 'JEPA/checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt'),
        '--roi-mode', 'causal_box_crop_predictor', '--crop-batch-size', '1', '--log-every', '1']
    with (args.out / 'extraction.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata_path = args.out / 'features.npz.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(engineering_smoke_only=args.engineering_smoke, native_full_frames=not args.engineering_smoke,
                    source_video_names=selected, frame_sampling='consecutive-native-frames',
                    gt_supervision_scope='source-sunny-only')
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print('SOURCE_CACHE_COMPLETE', args.out, len(images), len(annotations), flush=True)


if __name__ == '__main__':
    main()
