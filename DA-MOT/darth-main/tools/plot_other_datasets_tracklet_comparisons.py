#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from plot_vkitti2_tracklet_comparisons import (
    Candidate,
    REPO_ROOT,
    TrackObservation,
    center,
    gt_tracks,
    import_bytetrack,
    load_coco_tracks,
    load_json,
    match_track,
    plot_candidate,
    read_detection_rows,
    spatial_movement,
    track_stats,
    write_candidates,
)


CTTA_ROOT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/ContinualTTA_ObjectDetection-main"
)


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    ctta_key: str
    domains: tuple[str, ...]
    stages: dict[str, str]
    annotation_pattern: str
    ours_patterns: dict[str, str]
    darth_patterns: dict[str, str]
    ctta_match_thresh: float
    ctta_variant: str
    min_movement: float


CONFIGS = {
    "bdd": DatasetConfig(
        key="bdd",
        ctta_key="bdd_da",
        domains=("overcast", "rainy", "dawn_dusk", "night"),
        stages={
            "overcast": "stage_01_overcast",
            "rainy": "stage_02_rainy",
            "dawn_dusk": "stage_03_dawn_dusk",
            "night": "stage_04_night",
        },
        annotation_pattern=(
            "data/bdd_da/annotations/box_track_20/"
            "box_track_{domain}_all_cocofmt.json"
        ),
        ours_patterns={
            "overcast": (
                "work_dirs/limit_search_gpu23/full17_threshold/bdd/"
                "overcast/init03/**/track_coco.json"
            ),
            "rainy": (
                "work_dirs/limit_search_gpu23/full17_threshold/bdd/"
                "rainy/init03/**/track_coco.json"
            ),
            "dawn_dusk": (
                "work_dirs/limit_search_gpu23/full17_threshold/bdd/"
                "dawn_dusk/init03/**/track_coco.json"
            ),
            "night": (
                "work_dirs/limit_search_gpu23/phase1_threshold/bdd/"
                "night/init03/**/track_coco.json"
            ),
        },
        darth_patterns={
            domain: (
                "work_dirs/bdd_4domain/task1_single_target/eval/"
                f"tta_clear_to_{domain}/epoch_5/{domain}/**/track_coco.json"
            )
            for domain in ("overcast", "rainy", "dawn_dusk", "night")
        },
        ctta_match_thresh=0.5,
        ctta_variant="balanced_bdd_match050",
        min_movement=80.0,
    ),
    "motsynth": DatasetConfig(
        key="motsynth",
        ctta_key="motsynth",
        domains=("cloudy", "fog", "rain", "snow", "dark"),
        stages={
            "cloudy": "stage_01_cloudy",
            "dark": "stage_02_dark",
            "fog": "stage_03_fog",
            "rain": "stage_04_rain",
            "snow": "stage_05_snow",
        },
        annotation_pattern="data/motsynth/annotations/motsynth_{domain}_cocofmt.json",
        ours_patterns={
            domain: (
                "work_dirs/limit_search_gpu23/"
                f"{'phase1_threshold' if domain == 'rain' else 'full17_threshold'}"
                f"/motsynth/{domain}/obj05/**/track_coco.json"
            )
            for domain in ("cloudy", "fog", "rain", "snow", "dark")
        },
        darth_patterns={
            domain: (
                f"work_dirs/motsynth/test/baseline_normal_to_{domain}_5e/"
                "**/track_coco.json"
            )
            for domain in ("cloudy", "fog", "rain", "snow", "dark")
        },
        ctta_match_thresh=0.3,
        ctta_variant="balanced_motsynth_match030",
        min_movement=100.0,
    ),
    "ua_detrac": DatasetConfig(
        key="ua_detrac",
        ctta_key="ua_detrac",
        domains=("cloudy", "rainy", "night"),
        stages={
            "cloudy": "stage_01_cloudy",
            "night": "stage_02_night",
            "rainy": "stage_03_rainy",
        },
        annotation_pattern=(
            "data/ua_detrac/annotations/box_track_{domain}_all_cocofmt.json"
        ),
        ours_patterns={
            domain: (
                "work_dirs/limit_search_gpu23/"
                f"{'phase1_threshold' if domain == 'rainy' else 'full17_threshold'}"
                f"/ua_detrac/{domain}/init07/**/track_coco.json"
            )
            for domain in ("cloudy", "rainy", "night")
        },
        darth_patterns={
            domain: (
                f"work_dirs/ua_detrac/test/baseline_sunny_to_{domain}_5e/"
                "**/track_coco.json"
            )
            for domain in ("cloudy", "rainy", "night")
        },
        ctta_match_thresh=0.3,
        ctta_variant="balanced_ua_match030",
        min_movement=100.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(CONFIGS),
        default=tuple(CONFIGS),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "work_dirs/visualizations/other_datasets_tracklets",
    )
    parser.add_argument("--per-comparison", type=int, default=6)
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--match-iou", type=float, default=0.20)
    return parser.parse_args()


def find_single(pattern: str) -> Path:
    matches = sorted(REPO_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def ctta_detection_root(config: DatasetConfig, domain: str) -> Path:
    stage = config.stages[domain]
    return (
        CTTA_ROOT
        / "outputs/ctta_od"
        / config.ctta_key
        / stage
        / "detections"
        / domain
    )


def load_track_results(
    result_dir: Path,
    mapping: list[dict],
) -> dict[tuple[int, int], list[TrackObservation]]:
    image_id_by_video_frame = {
        (item["video_name"], int(item["frame_index_1based_in_video"])): int(
            item["image_id"]
        )
        for item in mapping
    }
    video_id_by_name = {
        item["video_name"]: int(item["video_id"]) for item in mapping
    }
    by_frame: dict[tuple[int, int], list[TrackObservation]] = defaultdict(list)
    for path in sorted(result_dir.glob("*.txt")):
        video_name = path.stem
        video_id = video_id_by_name[video_name]
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = [float(value) for value in line.split(",")]
            frame, track_id = int(row[0]), int(row[1])
            bbox = np.asarray(row[2:6], dtype=float)
            image_id = image_id_by_video_frame[(video_name, frame)]
            by_frame[(video_id, image_id)].append(
                TrackObservation(
                    frame=image_id,
                    track_id=track_id,
                    bbox=bbox,
                    center=center(bbox),
                )
            )
    return by_frame


def rebuild_ctta_tracks(
    config: DatasetConfig,
    domain: str,
    output_dir: Path,
) -> dict[tuple[int, int], list[TrackObservation]]:
    source = ctta_detection_root(config, domain)
    mapping = load_json(source / "image_mapping.json")
    frames_by_video: dict[str, list[dict]] = defaultdict(list)
    for item in mapping:
        frames_by_video[item["video_name"]].append(
            {
                "frame": int(item["frame_index_1based_in_video"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
            }
        )
    result_dir = (
        output_dir
        / "ctta_all5"
        / config.stages[domain]
        / domain
        / "track_results"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(result_dir.glob("*.txt"))
    if len(existing) == len(frames_by_video):
        return load_track_results(result_dir, mapping)

    BYTETracker, BaseTrack = import_bytetrack()
    tracker_args = SimpleNamespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=config.ctta_match_thresh,
        mot20=False,
    )
    for video_name, frames in sorted(frames_by_video.items()):
        detections = defaultdict(list)
        det_path = source / "bytetrack_dets" / video_name / "det/det.txt"
        for row in read_detection_rows(det_path, 0.99):
            detections[int(row[0])].append(row)
        BaseTrack._count = 0
        tracker = BYTETracker(tracker_args, frame_rate=30)
        output_rows = []
        for frame_info in sorted(frames, key=lambda item: item["frame"]):
            frame = frame_info["frame"]
            rows = detections.get(frame, [])
            boxes = np.asarray(
                [
                    [r[2], r[3], r[2] + r[4], r[3] + r[5], r[6]]
                    for r in rows
                ],
                dtype=np.float32,
            )
            if boxes.size == 0:
                boxes = np.zeros((0, 5), dtype=np.float32)
            targets = tracker.update(
                boxes,
                (frame_info["height"], frame_info["width"]),
                (frame_info["height"], frame_info["width"]),
            )
            for target in targets:
                x, y, width, height = [float(value) for value in target.tlwh]
                if width * height < 10:
                    continue
                output_rows.append(
                    [
                        frame,
                        int(target.track_id),
                        x,
                        y,
                        width,
                        height,
                        float(target.score),
                        -1,
                        -1,
                        -1,
                    ]
                )
        with (result_dir / f"{video_name}.txt").open("w") as handle:
            for row in output_rows:
                handle.write(
                    ",".join(
                        f"{value:.6f}" if isinstance(value, float) else str(value)
                        for value in row
                    )
                    + "\n"
                )
    return load_track_results(result_dir, mapping)


def build_candidates(
    comparison: str,
    domain: str,
    annotation_path: Path,
    ours: dict,
    comp: dict,
    min_movement: float,
    window_length: int,
    min_iou: float,
) -> list[Candidate]:
    annotation = load_json(annotation_path)
    tracks, video_names = gt_tracks(annotation)
    candidates = []
    for (video_id, gt_id), gt in tracks.items():
        if len(gt) < window_length:
            continue
        ours_match = match_track(gt, ours, video_id, min_iou)
        comp_match = match_track(gt, comp, video_id, min_iou)
        best = None
        for start in range(0, len(gt) - window_length + 1, 5):
            stop = start + window_length
            gt_window = gt[start:stop]
            ours_window = ours_match[start:stop]
            comp_window = comp_match[start:stop]
            movement = spatial_movement(gt_window)
            oc, os, oe = track_stats(gt_window, ours_window)
            cc, cs, ce = track_stats(gt_window, comp_window)
            error_gain = ce - oe
            if (
                movement < min_movement
                or oc < 0.80
                or cc < 0.45
                or oe > 0.08
                or error_gain < 0.02
            ):
                continue
            score = (
                18.0 * error_gain
                + 0.8 * (oc - cc)
                - 1.5 * oe
                + 0.002 * movement
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    gt_window,
                    ours_window,
                    comp_window,
                    oc,
                    os,
                    oe,
                    cc,
                    cs,
                    ce,
                    movement,
                )
        if best is None:
            continue
        (
            score,
            gt_window,
            ours_window,
            comp_window,
            oc,
            os,
            oe,
            cc,
            cs,
            ce,
            movement,
        ) = best
        candidates.append(
            Candidate(
                comparison=comparison,
                domain=domain,
                video_id=video_id,
                video_name=video_names[video_id],
                gt_id=gt_id,
                gt_length=len(gt_window),
                movement=movement,
                ours_coverage=oc,
                comp_coverage=cc,
                ours_switches=os,
                comp_switches=cs,
                ours_error=oe,
                comp_error=ce,
                score=score,
                gt=gt_window,
                ours=ours_window,
                comp=comp_window,
            )
        )
    return candidates


def select_unique_ids(candidates: list[Candidate], count: int) -> list[Candidate]:
    selected = []
    used_ids = set()
    for item in sorted(candidates, key=lambda value: value.score, reverse=True):
        if item.gt_id in used_ids:
            continue
        selected.append(item)
        used_ids.add(item.gt_id)
        if len(selected) == count:
            break
    return selected


def process_dataset(
    config: DatasetConfig,
    output_root: Path,
    per_comparison: int,
    window_length: int,
    min_iou: float,
) -> None:
    output_dir = output_root / config.key
    all_candidates = []
    inputs = {}
    for domain in config.domains:
        annotation = REPO_ROOT / config.annotation_pattern.format(domain=domain)
        ours_path = find_single(config.ours_patterns[domain])
        darth_path = find_single(config.darth_patterns[domain])
        ours = load_coco_tracks(ours_path)
        darth = load_coco_tracks(darth_path)
        ctaod = rebuild_ctta_tracks(config, domain, output_dir)
        inputs[domain] = {
            "annotation": str(annotation),
            "ours": str(ours_path),
            "darth": str(darth_path),
            "ctaod_variant": config.ctta_variant,
        }
        all_candidates.extend(
            build_candidates(
                "darth",
                domain,
                annotation,
                ours,
                darth,
                config.min_movement,
                window_length,
                min_iou,
            )
        )
        all_candidates.extend(
            build_candidates(
                "ctaod",
                domain,
                annotation,
                ours,
                ctaod,
                config.min_movement,
                window_length,
                min_iou,
            )
        )

    selected = []
    for comparison in ("darth", "ctaod"):
        selected.extend(
            select_unique_ids(
                [item for item in all_candidates if item.comparison == comparison],
                per_comparison,
            )
        )
    selected_keys = {
        (item.comparison, item.domain, item.video_name, item.gt_id)
        for item in selected
    }
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for old in figures.glob("tracklet_*"):
        old.unlink()
    for index, item in enumerate(selected, 1):
        plot_candidate(item, index, figures)
    write_candidates(output_dir / "candidate_scores.csv", all_candidates, selected_keys)
    manifest = {
        "dataset": config.key,
        "inputs": inputs,
        "selection": [
            {
                "tracklet": index,
                "comparison": item.comparison,
                "domain": item.domain,
                "video_name": item.video_name,
                "gt_id": item.gt_id,
                "frame_start": item.gt[0].frame,
                "frame_end": item.gt[-1].frame,
                "ours_error": item.ours_error,
                "comp_error": item.comp_error,
                "error_gap": item.comp_error - item.ours_error,
                "score": item.score,
            }
            for index, item in enumerate(selected, 1)
        ],
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest["selection"], indent=2))


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        process_dataset(
            CONFIGS[dataset],
            args.output_root,
            args.per_comparison,
            args.window_length,
            args.match_iou,
        )


if __name__ == "__main__":
    main()
