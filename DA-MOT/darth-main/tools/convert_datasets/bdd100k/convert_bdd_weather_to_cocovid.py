"""Convert BDD100K weather-domain tracking annotations to CocoVID format.

Given a weather-domain image root directory (e.g. BDD100K_clear or BDD100K_night),
this script traverses all video folders and matches them to BDD100K tracking
annotations (per-video JSON files) to produce a unified CocoVID JSON for each split.
"""
import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert BDD100K weather-domain to CocoVID format.')
    parser.add_argument('--domain-name', type=str, required=True,
                        help='Domain name, e.g. clear, night, snowy')
    parser.add_argument('--image-root', type=str, required=True,
                        help='Root directory of the weather-domain images')
    parser.add_argument('--ann-train-dir', type=str, required=True,
                        help='Directory with per-video train annotation JSONs')
    parser.add_argument('--ann-val-dir', type=str, required=True,
                        help='Directory with per-video val annotation JSONs')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory for CocoVID JSONs')
    parser.add_argument('--classes', nargs='+', type=str,
                        default=['pedestrian', 'car', 'truck', 'bus',
                                 'motorcycle', 'bicycle'],
                        help='Categories to include')
    return parser.parse_args()


def load_bdd_annotation(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_image_size(img_path):
    with Image.open(img_path) as img:
        return img.width, img.height


def convert_domain(domain_name, image_root, ann_train_dir, ann_val_dir,
                   out_dir, classes):
    os.makedirs(out_dir, exist_ok=True)

    cat_name_to_id = {name: i + 1 for i, name in enumerate(classes)}
    categories = [{'id': i + 1, 'name': name} for i, name in enumerate(classes)]

    video_dirs = sorted([
        d for d in os.listdir(image_root)
        if osp.isdir(osp.join(image_root, d))
    ])

    split_data = {
        'train': {'videos': [], 'images': [], 'annotations': [],
                  'categories': categories},
        'val': {'videos': [], 'images': [], 'annotations': [],
                'categories': categories},
    }

    split_counters = {
        'train': {'video_id': 0, 'image_id': 0, 'ann_id': 0},
        'val': {'video_id': 0, 'image_id': 0, 'ann_id': 0},
    }

    global_instance_map = {'train': {}, 'val': {}}
    instance_counter = {'train': 0, 'val': 0}

    missing_annotations = []
    missing_images = []
    skipped_boxes = 0
    skipped_categories = defaultdict(int)
    fallback_filename_matches = []

    stats = {
        'domain_name': domain_name,
        'num_videos': {'train': 0, 'val': 0},
        'num_images': {'train': 0, 'val': 0},
        'num_annotations': {'train': 0, 'val': 0},
        'num_unique_instances': {'train': 0, 'val': 0},
        'category_histogram': {'train': defaultdict(int),
                               'val': defaultdict(int)},
        'missing_annotation_videos': [],
        'missing_image_frames': [],
        'skipped_invalid_boxes': 0,
        'skipped_categories': {},
    }

    print(f"Processing domain: {domain_name}")
    print(f"Image root: {image_root}")
    print(f"Found {len(video_dirs)} video directories")
    print(f"Categories: {classes}")
    print()

    for vid_idx, video_name in enumerate(video_dirs):
        if (vid_idx + 1) % 50 == 0:
            print(f"  Processing video {vid_idx + 1}/{len(video_dirs)}: "
                  f"{video_name}")

        train_json = osp.join(ann_train_dir, f'{video_name}.json')
        val_json = osp.join(ann_val_dir, f'{video_name}.json')

        if osp.exists(train_json):
            split = 'train'
            ann_path = train_json
        elif osp.exists(val_json):
            split = 'val'
            ann_path = val_json
        else:
            missing_annotations.append(video_name)
            continue

        frames = load_bdd_annotation(ann_path)
        video_img_dir = osp.join(image_root, video_name)
        actual_files = sorted([
            f for f in os.listdir(video_img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if len(actual_files) == 0:
            missing_images.append((video_name, 'no images found'))
            continue

        counters = split_counters[split]
        counters['video_id'] += 1
        video_id = counters['video_id']

        first_img_path = osp.join(video_img_dir, actual_files[0])
        width, height = get_image_size(first_img_path)

        split_data[split]['videos'].append({
            'id': video_id,
            'name': video_name,
            'fps': 5,
            'width': width,
            'height': height,
        })
        stats['num_videos'][split] += 1

        frame_name_to_file = {}
        for f in actual_files:
            frame_name_to_file[f] = f

        for frame in frames:
            frame_name = frame.get('name', '')
            frame_index = frame.get('frameIndex', None)

            if frame_name in frame_name_to_file:
                file_name = frame_name
            elif frame_index is not None and frame_index < len(actual_files):
                file_name = actual_files[frame_index]
                fallback_filename_matches.append(
                    (video_name, frame_name, file_name))
            else:
                missing_images.append((video_name, frame_name))
                continue

            img_path = osp.join(video_img_dir, file_name)
            if not osp.exists(img_path):
                missing_images.append((video_name, file_name))
                continue

            counters['image_id'] += 1
            image_id = counters['image_id']

            frame_id = frame_index if frame_index is not None else 0

            relative_file_name = f"{video_name}/{file_name}"

            split_data[split]['images'].append({
                'id': image_id,
                'file_name': relative_file_name,
                'height': height,
                'width': width,
                'frame_id': frame_id,
                'video_id': video_id,
            })
            stats['num_images'][split] += 1

            labels = frame.get('labels', [])
            if labels is None:
                labels = []

            for label in labels:
                category = label.get('category', '')
                if category not in cat_name_to_id:
                    skipped_categories[category] += 1
                    continue

                box2d = label.get('box2d', None)
                if box2d is None:
                    continue

                x1 = box2d['x1']
                y1 = box2d['y1']
                x2 = box2d['x2']
                y2 = box2d['y2']
                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    skipped_boxes += 1
                    continue

                raw_label_id = label.get('id', '')
                instance_key = (video_name, raw_label_id)
                if instance_key not in global_instance_map[split]:
                    instance_counter[split] += 1
                    global_instance_map[split][instance_key] = \
                        instance_counter[split]
                global_instance_id = global_instance_map[split][instance_key]

                counters['ann_id'] += 1
                ann_id = counters['ann_id']

                category_id = cat_name_to_id[category]

                split_data[split]['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'video_id': video_id,
                    'category_id': category_id,
                    'instance_id': global_instance_id,
                    'bbox': [round(x1, 2), round(y1, 2),
                             round(w, 2), round(h, 2)],
                    'area': round(w * h, 2),
                    'iscrowd': 0,
                })
                stats['num_annotations'][split] += 1
                stats['category_histogram'][split][category] += 1

    stats['num_unique_instances'] = {
        'train': instance_counter['train'],
        'val': instance_counter['val'],
    }
    stats['missing_annotation_videos'] = missing_annotations
    stats['missing_image_frames'] = missing_images[:100]
    stats['skipped_invalid_boxes'] = skipped_boxes
    stats['skipped_categories'] = dict(skipped_categories)
    stats['fallback_filename_matches_count'] = len(fallback_filename_matches)
    stats['category_histogram'] = {
        k: dict(v) for k, v in stats['category_histogram'].items()
    }

    for split in ['train', 'val']:
        out_file = osp.join(
            out_dir, f'box_track_{domain_name}_{split}_cocofmt.json')
        data = split_data[split]
        if len(data['images']) > 0:
            print(f"\nWriting {split}: {out_file}")
            print(f"  Videos: {len(data['videos'])}")
            print(f"  Images: {len(data['images'])}")
            print(f"  Annotations: {len(data['annotations'])}")
            with open(out_file, 'w') as f:
                json.dump(data, f)
        else:
            print(f"\nWARNING: {split} split is empty for domain {domain_name}")
            print(f"  Writing empty CocoVID to: {out_file}")
            with open(out_file, 'w') as f:
                json.dump(data, f)

    all_data = {
        'videos': [],
        'images': [],
        'annotations': [],
        'categories': categories,
    }
    next_video_id = 0
    next_image_id = 0
    next_ann_id = 0
    image_id_maps = {}
    video_id_maps = {}
    for split in ['train', 'val']:
        video_id_maps[split] = {}
        image_id_maps[split] = {}
        for video in split_data[split]['videos']:
            next_video_id += 1
            video_id_maps[split][video['id']] = next_video_id
            new_video = dict(video)
            new_video['id'] = next_video_id
            all_data['videos'].append(new_video)
        for image in split_data[split]['images']:
            next_image_id += 1
            image_id_maps[split][image['id']] = next_image_id
            new_image = dict(image)
            new_image['id'] = next_image_id
            new_image['video_id'] = video_id_maps[split][image['video_id']]
            all_data['images'].append(new_image)
        for ann in split_data[split]['annotations']:
            next_ann_id += 1
            new_ann = dict(ann)
            new_ann['id'] = next_ann_id
            new_ann['image_id'] = image_id_maps[split][ann['image_id']]
            new_ann['video_id'] = video_id_maps[split][ann['video_id']]
            all_data['annotations'].append(new_ann)
    all_file = osp.join(out_dir, f'box_track_{domain_name}_all_cocofmt.json')
    print(f"\nWriting all: {all_file}")
    print(f"  Videos: {len(all_data['videos'])}")
    print(f"  Images: {len(all_data['images'])}")
    print(f"  Annotations: {len(all_data['annotations'])}")
    with open(all_file, 'w') as f:
        json.dump(all_data, f)

    stats['num_videos']['all'] = len(all_data['videos'])
    stats['num_images']['all'] = len(all_data['images'])
    stats['num_annotations']['all'] = len(all_data['annotations'])

    stats_file = osp.join(out_dir, f'stats_{domain_name}.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nStats saved to: {stats_file}")

    if missing_annotations:
        missing_ann_file = osp.join(out_dir,
                                    f'missing_annotations_{domain_name}.txt')
        with open(missing_ann_file, 'w') as f:
            for v in missing_annotations:
                f.write(f"{v}\n")
        print(f"Missing annotations ({len(missing_annotations)} videos): "
              f"{missing_ann_file}")

    if missing_images:
        missing_img_file = osp.join(out_dir,
                                    f'missing_images_{domain_name}.txt')
        with open(missing_img_file, 'w') as f:
            for item in missing_images:
                f.write(f"{item}\n")
        print(f"Missing images ({len(missing_images)} entries): "
              f"{missing_img_file}")

    if fallback_filename_matches:
        fallback_file = osp.join(out_dir,
                                 f'fallback_matches_{domain_name}.txt')
        with open(fallback_file, 'w') as f:
            for video, orig, matched in fallback_filename_matches[:200]:
                f.write(f"{video}: {orig} -> {matched}\n")
        print(f"Fallback filename matches: "
              f"{len(fallback_filename_matches)} (logged first 200)")

    print(f"\n{'='*60}")
    print(f"Domain: {domain_name}")
    print(f"Total videos found: {len(video_dirs)}")
    print(f"  train videos: {stats['num_videos']['train']}")
    print(f"  val videos: {stats['num_videos']['val']}")
    print(f"  missing annotation: {len(missing_annotations)}")
    print(f"Train images: {stats['num_images']['train']}, "
          f"annotations: {stats['num_annotations']['train']}, "
          f"instances: {stats['num_unique_instances']['train']}")
    print(f"Val images: {stats['num_images']['val']}, "
          f"annotations: {stats['num_annotations']['val']}, "
          f"instances: {stats['num_unique_instances']['val']}")
    print(f"Skipped invalid boxes: {skipped_boxes}")
    print(f"Skipped categories: {dict(skipped_categories)}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    convert_domain(
        domain_name=args.domain_name,
        image_root=args.image_root,
        ann_train_dir=args.ann_train_dir,
        ann_val_dir=args.ann_val_dir,
        out_dir=args.out_dir,
        classes=args.classes,
    )


if __name__ == '__main__':
    main()
