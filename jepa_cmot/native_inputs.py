"""Export video-disjoint native streams; labels remain separate from inference.

This offline preparation process may read target annotations to produce an
evaluation store. Online adaptation opens only the label-free stream files.
No frame or video is selected using tracking performance or target identities.
"""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from archive_paths import resolve_archive_path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = [{'id': i + 1, 'name': n} for i, n in enumerate(
    ['pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'])]
UA_CLASSES = {'car': 2, 'van': 3, 'bus': 4, 'others': 2}


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(',', ':')))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ua_video(record, video_id):
    xml = ET.parse(record['xml']).getroot()
    images, annotations = [], []
    known_frames = [int(f.attrib['num']) for f in xml.findall('frame')]
    for path in sorted(Path(record['image_root']).glob('img*.jpg')):
        native_frame = int(path.stem[3:])
        images.append(dict(id=native_frame, video_id=video_id, frame_id=native_frame,
            native_frame_id=native_frame, width=960, height=540, file_name=str(path)))
    ignored = [[float(b.attrib[k]) for k in ['left', 'top', 'width', 'height']]
               for b in xml.findall('./ignored_region/box')]
    for frame in xml.findall('frame'):
        native_frame = int(frame.attrib['num'])
        for target in frame.findall('./target_list/target'):
            box = target.find('box').attrib
            attributes = target.find('attribute').attrib
            x, y, w, h = [float(box[k]) for k in ['left', 'top', 'width', 'height']]
            # Clip boxes exactly at image boundaries; keep every valid object.
            x2, y2 = min(960., x + w), min(540., y + h)
            x, y = max(0., x), max(0., y)
            if x2 <= x or y2 <= y:
                continue
            annotations.append(dict(id=len(annotations) + 1, image_id=native_frame,
                video_id=video_id, instance_id=int(target.attrib['id']),
                category_id=UA_CLASSES[attributes['vehicle_type']],
                bbox=[x, y, x2-x, y2-y], area=(x2-x)*(y2-y), iscrowd=0,
                truncation_ratio=float(attributes.get('truncation_ratio', 0)),
                occlusion_regions=[dict(r.attrib) for r in target.findall('./occlusion/region_overlap')]))
    return images, annotations, ignored, known_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads((ROOT/'docs/JEPA_CMOT/DATA_INVENTORY.json').read_text())
    proposal = json.loads((ROOT/'docs/JEPA_CMOT/PROPOSED_VIDEO_SPLITS.json').read_text())
    manifest = dict(status='native-inputs-prepared-checkpoint-audit-pending',
                    split_rule='sha256-video-order-fixed-before-any-new-development-results',
                    target_gt_scope='offline-evaluation-store-only', videos=[])
    ua = {r['video']: r for r in inventory['ua_detrac']}
    for dataset, domains in proposal['datasets'].items():
        for domain, partitions in domains.items():
            if dataset == 'bdd100k':
                ann = json.loads(Path(inventory['bdd100k'][domain]['all']['path']).read_text())
                names = {v['id']: v['name'] for v in ann['videos']}
                by_image = {i['id']: i for i in ann['images']}
                bdd_images, bdd_labels = {}, {}
                for image in ann['images']:
                    bdd_images.setdefault(names[image['video_id']], []).append(image)
                for label in ann['annotations']:
                    name = names[by_image[label['image_id']]['video_id']]
                    bdd_labels.setdefault(name, []).append(label)
            for split, videos in partitions.items():
                for video_id, name in enumerate(videos, 1):
                    if dataset == 'ua_detrac':
                        images, labels, ignored, known_frames = ua_video(ua[name], video_id)
                    else:
                        images = sorted(bdd_images[name], key=lambda i: i['frame_id'])
                        resolved_root = resolve_archive_path(
                            f'/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_{domain}/{name}')
                        images = [dict(i, video_id=video_id, native_frame_id=i['frame_id'],
                            frame_id=i['frame_id']+1,
                            file_name=str(resolved_root/Path(i['file_name']).name)) for i in images]
                        labels = [dict(a, video_id=video_id) for a in bdd_labels.get(name, [])]
                        ignored = []
                        known_frames = [i['frame_id'] for i in images]
                    if any(b['frame_id'] != a['frame_id']+1 for a,b in zip(images, images[1:])):
                        raise ValueError(f'Native sequence has missing frames: {name}')
                    base = dict(videos=[dict(id=video_id, name=name)], images=images, categories=CATEGORIES)
                    directory = args.out/dataset/domain/split/name
                    stream = directory/'stream.json'
                    stream_sha = write(stream, dict(base, annotations=[], gt_read=False,
                        native_full_frames=True, source_domain='sunny' if dataset=='ua_detrac' else 'clear'))
                    # Source fitting labels are accessible only in source_train.
                    label_path = directory/('source_labels.json' if split=='source_train' else 'evaluation_labels.json')
                    label_sha = write(label_path, dict(base, annotations=labels, ignored_regions=ignored,
                        evaluation_frame_ids=known_frames,
                        unknown_annotation_frames=sorted(set(i['frame_id'] for i in images)-set(known_frames)),
                        gt_scope='source-training' if split=='source_train' else 'offline-evaluation-only'))
                    manifest['videos'].append(dict(dataset=dataset,domain=domain,split=split,
                        name=name,frames=len(images),annotated_frames=len(known_frames),
                        stream=str(stream.resolve()),stream_sha256=stream_sha,
                        labels=str(label_path.resolve()),label_sha256=label_sha))
            print('NATIVE_DOMAIN_READY',dataset,domain,flush=True)
    write(args.out/'manifest.json',manifest)
    counts = {ds:sum(v['frames'] for v in manifest['videos'] if v['dataset']==ds)
              for ds in proposal['datasets']}
    print('NATIVE_INPUTS_COMPLETE',counts,flush=True)


if __name__ == '__main__':
    main()
