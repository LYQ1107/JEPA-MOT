import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

from generaltrack_damot_utils import (
    DEFAULT_CONFIG,
    TRACK_FIELDS,
    append_or_replace_summary,
    box_iou_matrix,
    get_dataset_cfg,
    load_dataset_config,
    load_prepared_metadata,
    read_json,
    read_mot_result_file,
    repo_path,
    resolve_annotation,
    safe_video_name,
    write_csv,
    write_json,
)

sys.path.insert(0, str(repo_path("TrackEval")))
from trackeval.metrics.clear import CLEAR
from trackeval.metrics.hota import HOTA
from trackeval.metrics.identity import Identity


def make_parser():
    parser = argparse.ArgumentParser("Evaluate GeneralTrack DAMOT/DARTH domain results")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--prepared-dir", default=None)
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--track-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--det-source", default=os.environ.get("DET_SOURCE", "source_detector"))
    parser.add_argument("--summary-csv", default="outputs/generaltrack/generaltrack_multidataset_summary.csv")
    parser.add_argument("--method", default="GeneralTrack")
    parser.add_argument("--protocol", default="source-only-no-adaptation")
    return parser


def _metric_percent(metric, result, field):
    value = result[field]
    if isinstance(value, np.ndarray):
        return float(100.0 * np.mean(value))
    return float(100.0 * value)


def _metric_int(result, field):
    return int(result[field])


def _group_gt(annotation):
    image_by_id = {int(img["id"]): img for img in annotation["images"]}
    video_names = {
        int(v["id"]): safe_video_name(v.get("file_name", v.get("name", v["id"])))
        for v in annotation.get("videos", [])
    }
    frames_by_video = defaultdict(list)
    gt_by_video_frame = defaultdict(list)
    for image in annotation["images"]:
        frames_by_video[int(image["video_id"])].append(int(image["frame_id"]))
    for ann in annotation.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        image = image_by_id.get(int(ann.get("image_id", -1)))
        if image is None:
            continue
        track_id = int(ann.get("track_id", ann.get("instance_id", ann.get("mot_instance_id", ann.get("id", 0)))))
        gt_by_video_frame[(int(image["video_id"]), int(image["frame_id"]))].append(
            {"track_id": track_id, "bbox": [float(v) for v in ann["bbox"]]}
        )
    return video_names, frames_by_video, gt_by_video_frame


def _group_tracker(track_results_dir, video_names):
    tracker_by_video_frame = defaultdict(list)
    for video_id, video_name in video_names.items():
        result_file = Path(track_results_dir) / "{}.txt".format(video_name)
        rows = read_mot_result_file(result_file)
        for row in rows:
            tracker_by_video_frame[(video_id, int(row["frame_id"]))].append(row)
    return tracker_by_video_frame


def _remap_ids(items):
    ids = sorted({int(item["track_id"]) for item in items})
    return {tid: idx for idx, tid in enumerate(ids)}


def _build_sequence_data(video_id, frames, gt_by_video_frame, tracker_by_video_frame):
    all_gt = []
    all_tracker = []
    for frame in frames:
        all_gt.extend(gt_by_video_frame.get((video_id, frame), []))
        all_tracker.extend(tracker_by_video_frame.get((video_id, frame), []))
    gt_id_map = _remap_ids(all_gt)
    tracker_id_map = _remap_ids(all_tracker)

    data = {
        "num_timesteps": len(frames),
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(tracker_id_map),
        "num_gt_dets": len(all_gt),
        "num_tracker_dets": len(all_tracker),
    }
    for frame in frames:
        gt = gt_by_video_frame.get((video_id, frame), [])
        tracker = tracker_by_video_frame.get((video_id, frame), [])
        gt_ids = np.asarray([gt_id_map[int(item["track_id"])] for item in gt], dtype=int)
        tracker_ids = np.asarray(
            [tracker_id_map[int(item["track_id"])] for item in tracker], dtype=int
        )
        gt_boxes = [item["bbox"] for item in gt]
        tracker_boxes = [item["bbox"] for item in tracker]
        data["gt_ids"].append(gt_ids)
        data["tracker_ids"].append(tracker_ids)
        data["similarity_scores"].append(box_iou_matrix(gt_boxes, tracker_boxes))
    return data


def evaluate(annotation, track_results_dir):
    video_names, frames_by_video, gt_by_video_frame = _group_gt(annotation)
    tracker_by_video_frame = _group_tracker(track_results_dir, video_names)
    hota = HOTA()
    clear = CLEAR({"PRINT_CONFIG": False})
    identity = Identity({"PRINT_CONFIG": False})
    seq_hota = {}
    seq_clear = {}
    seq_identity = {}
    for video_id, frames in sorted(frames_by_video.items()):
        frames = sorted(set(frames))
        seq_name = video_names.get(video_id, str(video_id))
        seq_data = _build_sequence_data(video_id, frames, gt_by_video_frame, tracker_by_video_frame)
        seq_hota[seq_name] = hota.eval_sequence(seq_data)
        seq_clear[seq_name] = clear.eval_sequence(seq_data)
        seq_identity[seq_name] = identity.eval_sequence(seq_data)
    combined_hota = hota.combine_sequences(seq_hota)
    combined_clear = clear.combine_sequences(seq_clear)
    combined_identity = identity.combine_sequences(seq_identity)
    return {
        "sequence_count": len(frames_by_video),
        "frame_count": sum(len(set(v)) for v in frames_by_video.values()),
        "metrics": {
            "HOTA": _metric_percent(hota, combined_hota, "HOTA"),
            "DetA": _metric_percent(hota, combined_hota, "DetA"),
            "AssA": _metric_percent(hota, combined_hota, "AssA"),
            "MOTA": _metric_percent(clear, combined_clear, "MOTA"),
            "IDF1": _metric_percent(identity, combined_identity, "IDF1"),
            "IDSW": _metric_int(combined_clear, "IDSW"),
            "FP": _metric_int(combined_clear, "CLR_FP"),
            "FN": _metric_int(combined_clear, "CLR_FN"),
        },
        "per_sequence": {
            "HOTA": {k: {f: str(vv) for f, vv in v.items()} for k, v in seq_hota.items()},
            "CLEAR": {k: {f: str(vv) for f, vv in v.items()} for k, v in seq_clear.items()},
            "Identity": {k: {f: str(vv) for f, vv in v.items()} for k, v in seq_identity.items()},
        },
    }


def main():
    args = make_parser().parse_args()
    config = load_dataset_config(args.config)
    dataset_cfg = get_dataset_cfg(config, args.dataset)

    metadata = {}
    ann_file = None
    if args.prepared_dir:
        prepared_dir = repo_path(args.prepared_dir)
        metadata = load_prepared_metadata(prepared_dir)
        ann_file = prepared_dir / "annotations" / "val.json"
    elif args.ann_file:
        ann_file = repo_path(args.ann_file)
    else:
        ann_file = resolve_annotation(dataset_cfg, args.dataset, args.domain)
    if not ann_file or not Path(ann_file).exists():
        raise FileNotFoundError("Missing GT annotation for evaluation: {}".format(ann_file))

    track_results = repo_path(args.track_results)
    if not track_results.exists():
        raise FileNotFoundError("Missing track_results directory: {}".format(track_results))

    annotation = read_json(ann_file)
    results = evaluate(annotation, track_results)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_row = {
        "method": args.method,
        "protocol": args.protocol,
        "dataset": args.dataset,
        "source_domain": metadata.get("source_domain", dataset_cfg.get("source", "")),
        "target_domain": args.domain,
        "num_videos": results["sequence_count"],
        "num_frames": results["frame_count"],
        "det_source": args.det_source,
    }
    metrics_row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in results["metrics"].items()})

    write_json(output_dir / "metrics.json", {"summary": metrics_row, "details": results})
    write_csv(output_dir / "metrics.csv", [metrics_row], TRACK_FIELDS)
    append_or_replace_summary(repo_path(args.summary_csv), metrics_row)
    print("Wrote metrics: {}".format(output_dir / "metrics.json"))
    print("Updated summary: {}".format(repo_path(args.summary_csv)))
    print(json.dumps(metrics_row, indent=2))


if __name__ == "__main__":
    main()
