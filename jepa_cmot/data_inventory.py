"""Audit native video coverage and propose label-independent video partitions."""
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from archive_paths import resolve_archive_path

ROOT = Path(__file__).resolve().parents[1]
NAS = Path('/data1/liuyeqiang/nas/A100')
UA = NAS / 'testdata/vranlee/.MOTSynth.partial/UA-DETRAC'
OUTPUT = ROOT / 'docs/JEPA_CMOT'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(video):
    return str(video.get('name', video.get('file_name', video['id'])))


def partition(dataset, domain, videos, source=False):
    ordered = sorted(videos, key=lambda v: hashlib.sha256(
        f'JEPA_CMOT_20261013/{dataset}/{domain}/{v}'.encode()).hexdigest())
    count = len(ordered)
    first = int(count * (0.70 if source else 0.60))
    second = int(count * (0.15 if source else 0.20))
    return {('source_train' if source else 'adaptation'): ordered[:first],
            'development': ordered[first:first + second],
            'sealed': ordered[first + second:]}


def main():
    report = {'status': 'audit-in-progress-not-final-protocol', 'ua_detrac': [], 'bdd100k': {}}
    split = {'status': 'proposed-until-checkpoint-and-detector-audit', 'datasets': {}}
    historical = {}
    for domain in ['sunny', 'cloudy', 'rainy', 'night']:
        annotation = resolve_archive_path('/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/ua_detrac/annotations/box_track_' + domain + '_all_cocofmt.json')
        data = json.loads(annotation.read_text())
        names = {v['id']: name(v) for v in data['videos']}
        counts = Counter(names[image['video_id']] for image in data['images'])
        historical.update(counts)
    for native_split in ['Train', 'Test']:
        for path in sorted((UA / f'DETRAC-{native_split}-Annotations-XML').glob('*.xml')):
            xml = ET.parse(path).getroot()
            attributes = xml.find('sequence_attribute').attrib
            domain = attributes['sence_weather']
            video = xml.attrib['name']
            frames = xml.findall('frame')
            directory = UA / f'Insight-MVT_Annotation_{native_split}' / video
            images = [p.name for p in directory.iterdir() if p.suffix.lower() == '.jpg']
            record = {'video': video, 'domain': domain, 'native_split': native_split,
                      'xml': str(path), 'xml_sha256': sha(path), 'image_root': str(directory),
                      'native_frames': len(frames), 'native_images': len(images),
                      'historical_frames': historical.get(video, 0),
                      'first_frame': int(frames[0].attrib['num']),
                      'last_frame': int(frames[-1].attrib['num'])}
            report['ua_detrac'].append(record)
    split['datasets']['ua_detrac'] = {
        d: partition('ua_detrac', d, [r['video'] for r in report['ua_detrac'] if r['domain'] == d], d == 'sunny')
        for d in ['sunny', 'cloudy', 'rainy', 'night']}
    bdd = resolve_archive_path('/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/bdd_da/annotations/box_track_20')
    split['datasets']['bdd100k'] = {}
    for domain in ['clear', 'overcast', 'rainy', 'night']:
        records = {}
        for subset in ['all', 'train', 'val']:
            path = bdd / f'box_track_{domain}_{subset}_cocofmt.json'
            data = json.loads(path.read_text())
            records[subset] = {'path': str(path), 'sha256': sha(path),
                               'videos': sorted(name(v) for v in data['videos']),
                               'frames': len(data['images'])}
        records['train_val_video_overlap'] = sorted(set(records['train']['videos']) & set(records['val']['videos']))
        report['bdd100k'][domain] = records
        split['datasets']['bdd100k'][domain] = partition('bdd100k', domain, records['all']['videos'], domain == 'clear')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'DATA_INVENTORY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (OUTPUT / 'PROPOSED_VIDEO_SPLITS.json').write_text(json.dumps(split, ensure_ascii=False, indent=2))
    print('DATA_INVENTORY_READY', {d: sum(r['native_frames'] for r in report['ua_detrac'] if r['domain'] == d)
                                   for d in ['sunny', 'cloudy', 'rainy', 'night']}, flush=True)
    for domain, record in report['bdd100k'].items():
        print('BDD', domain, 'train/val overlap', len(record['train_val_video_overlap']), flush=True)


if __name__ == '__main__':
    main()
