"""Create a sequence/frame prefix for a bounded, reproducible smoke run.

This only selects images and their existing annotations for the evaluator. The
test pipeline does not load annotations, so the tracker still receives no
target GT. The output is a new file and never rewrites the source annotation.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-videos', type=int, default=3)
    parser.add_argument('--max-frames', type=int, default=30)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.output) and not args.overwrite:
        raise IOError('refusing to overwrite existing prefix: {}'.format(
            args.output))
    with open(args.source, 'r') as handle:
        source = json.load(handle)
    ordered_videos = []
    for image in sorted(source['images'],
                        key=lambda item: (item['video_id'], item['frame_id'])):
        if image['video_id'] not in ordered_videos:
            ordered_videos.append(image['video_id'])
    selected_videos = set(ordered_videos[:args.max_videos])
    selected_images = [
        image for image in source['images']
        if image['video_id'] in selected_videos and
        int(image['frame_id']) < args.max_frames]
    selected_image_ids = set(image['id'] for image in selected_images)
    selected_annotations = [
        ann for ann in source.get('annotations', [])
        if ann['image_id'] in selected_image_ids]
    result = dict(
        categories=source.get('categories', []),
        videos=[video for video in source.get('videos', [])
                if video['id'] in selected_videos],
        images=selected_images,
        annotations=selected_annotations)
    parent = os.path.dirname(os.path.abspath(args.output))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(args.output, 'w') as handle:
        json.dump(result, handle)
    print(json.dumps({
        'output': os.path.abspath(args.output),
        'videos': sorted(selected_videos),
        'images': len(selected_images),
        'annotations': len(selected_annotations),
        'target_gt_used_by_tracker': False,
        'target_gt_reserved_for_official_evaluator': True,
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
