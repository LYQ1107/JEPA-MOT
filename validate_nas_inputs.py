#!/usr/bin/env python3
"""Check one real archived image per dataset using the resumed path resolver."""
import json
from pathlib import Path

import cv2
from archive_paths import resolve_archive_path

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('/data1/liuyeqiang/nas/A100/data1/LWR/vranlee/SERVER_ONLY/avis')


def main():
    records = []
    for dataset, domain in [('ua_detrac', 'cloudy'), ('bdd100k', 'night'),
                            ('vkitti2', 'fog'), ('motsynth', 'dark')]:
        prepared = (
            ARCHIVE / 'GeneralTrack-main/datasets/generaltrack_prepared_mapped/source_detector_thr02' / dataset / domain
            if dataset == 'motsynth' else
            ARCHIVE / 'GeneralTrack-main/datasets/generaltrack_prepared_thr02' / dataset / domain
        )
        metadata = json.loads((prepared / 'metadata.json').read_text())
        annotation = json.loads((prepared / 'annotations/val.json').read_text())
        image = min(annotation['images'], key=lambda x: (x['video_id'], x['frame_id'], x['id']))
        original = Path(metadata['image_root']) / image['file_name']
        resolved = resolve_archive_path(str(original))
        pixels = cv2.imread(str(resolved))
        record = {'dataset': dataset, 'domain': domain, 'prepared': str(prepared),
                  'original_image': str(original), 'resolved_image': str(resolved),
                  'readable': pixels is not None,
                  'shape': list(pixels.shape) if pixels is not None else None}
        records.append(record)
        print(dataset, 'PASS' if record['readable'] else 'FAIL', str(resolved), flush=True)
        del annotation
    output = ROOT / 'results/nas_input_validation.json'
    output.write_text(json.dumps({'all_pass': all(r['readable'] for r in records),
                                  'records': records}, ensure_ascii=False, indent=2))
    if not all(r['readable'] for r in records):
        raise RuntimeError('Some archived dataset samples are not readable; see ' + str(output))


if __name__ == '__main__':
    main()
