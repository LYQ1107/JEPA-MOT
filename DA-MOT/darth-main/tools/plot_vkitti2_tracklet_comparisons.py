#!/usr/bin/env python3
"""Select and plot VKITTI2 tracklets for qualitative comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CTTA_ROOT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/ContinualTTA_ObjectDetection-main"
)
BYTETRACK_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OC_SORT_yolox_old")

DOMAINS = ("overcast", "morning", "rain", "fog", "sunset")
OURS_STAGE = {
    "overcast": "stage01_after_overcast",
    "morning": "stage02_after_morning",
    "rain": "stage03_after_rain",
    "fog": "stage04_after_fog",
    "sunset": "stage05_after_sunset",
}
CTTA_STAGE = {
    "fog": "stage_01_fog",
    "morning": "stage_02_morning",
    "overcast": "stage_03_overcast",
    "rain": "stage_04_rain",
    "sunset": "stage_05_sunset",
}

COLORS = {
    "ours": "#D81B60",
    "darth": "#2878B5",
    "ctaod": "#2878B5",
    "gt": "#404040",
}


@dataclass
class TrackObservation:
    frame: int
    track_id: int
    bbox: np.ndarray
    center: np.ndarray


@dataclass
class Candidate:
    comparison: str
    domain: str
    video_id: int
    video_name: str
    gt_id: int
    gt_length: int
    movement: float
    ours_coverage: float
    comp_coverage: float
    ours_switches: int
    comp_switches: int
    ours_error: float
    comp_error: float
    score: float
    gt: list[TrackObservation]
    ours: list[TrackObservation | None]
    comp: list[TrackObservation | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "work_dirs/visualizations/vkitti2_tracklet_comparison",
    )
    parser.add_argument("--per-comparison", type=int, default=6)
    parser.add_argument("--min-track-length", type=int, default=30)
    parser.add_argument("--min-movement", type=float, default=150.0)
    parser.add_argument("--window-length", type=int, default=30)
    parser.add_argument("--match-iou", type=float, default=0.25)
    return parser.parse_args()


def load_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def center(bbox: np.ndarray) -> np.ndarray:
    return np.asarray(
        [bbox[0] + 0.5 * bbox[2], bbox[1] + 0.5 * bbox[3]], dtype=float
    )


def find_single(pattern: str) -> Path:
    matches = sorted(REPO_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def ours_track_path(domain: str) -> Path:
    if domain == "overcast":
        pattern = (
            "work_dirs/limit_search_gpu23/phase1_threshold/vkitti2/"
            "overcast/init03/**/track_coco.json"
        )
    else:
        pattern = (
            f"work_dirs/limit_search_gpu23/full17_threshold/vkitti2/"
            f"{domain}/init03/**/track_coco.json"
        )
    return find_single(pattern)


def darth_track_path(domain: str) -> Path:
    return find_single(
        f"work_dirs/vkitti2/test/darth_clone_to_{domain}_5e/**/track_coco.json"
    )


def annotation_path(domain: str) -> Path:
    return REPO_ROOT / f"data/vkitti2/annotations/vkitti2_{domain}_cocofmt.json"


def load_coco_tracks(path: Path) -> dict[tuple[int, int], list[TrackObservation]]:
    by_frame: dict[tuple[int, int], list[TrackObservation]] = defaultdict(list)
    for row in load_json(path):
        bbox = np.asarray(row["bbox"], dtype=float)
        obs = TrackObservation(
            frame=int(row["image_id"]),
            track_id=int(row["track_id"]),
            bbox=bbox,
            center=center(bbox),
        )
        by_frame[(int(row["video_id"]), int(row["image_id"]))].append(obs)
    return by_frame


def import_bytetrack():
    if str(BYTETRACK_ROOT) not in sys.path:
        sys.path.insert(0, str(BYTETRACK_ROOT))
    if "float" not in np.__dict__:
        np.float = float
    if "int" not in np.__dict__:
        np.int = int
    from trackers.byte_tracker.basetrack import BaseTrack
    from trackers.byte_tracker.byte_tracker import BYTETracker

    return BYTETracker, BaseTrack


def read_detection_rows(path: Path, score_threshold: float) -> list[list[float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = [float(value) for value in line.split(",")]
        if row[6] >= score_threshold:
            rows.append(row)
    return rows


def rebuild_ctta_tracks(
    domain: str, output_root: Path
) -> dict[tuple[int, int], list[TrackObservation]]:
    from types import SimpleNamespace

    stage = CTTA_STAGE[domain]
    source = CTTA_ROOT / f"outputs/ctta_od/vkitti2/{stage}/detections/{domain}"
    mapping = load_json(source / "image_mapping.json")
    frames_by_video: dict[str, list[dict]] = defaultdict(list)
    video_id_by_name = {}
    image_id_by_video_frame = {}
    for item in mapping:
        name = item["video_name"]
        frame = int(item["frame_index_1based_in_video"])
        frames_by_video[name].append(
            {
                "frame": frame,
                "width": int(item["width"]),
                "height": int(item["height"]),
            }
        )
        video_id_by_name[name] = int(item["video_id"])
        image_id_by_video_frame[(name, frame)] = int(item["image_id"])

    result_dir = output_root / "ctta_all5" / stage / "track_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[tuple[int, int], list[TrackObservation]] = defaultdict(list)
    existing = sorted(result_dir.glob("*.txt"))
    if len(existing) == len(frames_by_video):
        for video_name in sorted(frames_by_video):
            video_id = video_id_by_name[video_name]
            for line in (result_dir / f"{video_name}.txt").read_text().splitlines():
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

    BYTETracker, BaseTrack = import_bytetrack()
    tracker_args = SimpleNamespace(
        track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False
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
                if width * height < 2200:
                    continue
                tid = int(target.track_id)
                bbox = np.asarray([x, y, width, height], dtype=float)
                image_id = image_id_by_video_frame[(video_name, frame)]
                video_id = video_id_by_name[video_name]
                obs = TrackObservation(
                    frame=image_id, track_id=tid, bbox=bbox, center=center(bbox)
                )
                by_frame[(video_id, image_id)].append(obs)
                output_rows.append(
                    [frame, tid, x, y, width, height, float(target.score), -1, -1, -1]
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
    return by_frame


def gt_tracks(annotation: dict) -> tuple[dict, dict[int, str]]:
    image_by_id = {int(row["id"]): row for row in annotation["images"]}
    video_names = {int(row["id"]): row["name"] for row in annotation["videos"]}
    tracks: dict[tuple[int, int], list[TrackObservation]] = defaultdict(list)
    for row in annotation["annotations"]:
        if row.get("iscrowd", 0):
            continue
        video_id = int(row["video_id"])
        track_id = int(row.get("track_id", row.get("instance_id")))
        bbox = np.asarray(row["bbox"], dtype=float)
        image_id = int(row["image_id"])
        tracks[(video_id, track_id)].append(
            TrackObservation(
                frame=image_id, track_id=track_id, bbox=bbox, center=center(bbox)
            )
        )
    for observations in tracks.values():
        observations.sort(key=lambda item: image_by_id[item.frame]["frame_id"])
    return tracks, video_names


def match_track(
    gt: list[TrackObservation],
    predictions: dict[tuple[int, int], list[TrackObservation]],
    video_id: int,
    min_iou: float,
) -> list[TrackObservation | None]:
    matched = []
    for gt_obs in gt:
        candidates = predictions.get((video_id, gt_obs.frame), [])
        if not candidates:
            matched.append(None)
            continue
        ious = [bbox_iou(gt_obs.bbox, item.bbox) for item in candidates]
        best = int(np.argmax(ious))
        matched.append(candidates[best] if ious[best] >= min_iou else None)
    return matched


def track_stats(
    gt: list[TrackObservation], matched: list[TrackObservation | None]
) -> tuple[float, int, float]:
    valid = [(g, p) for g, p in zip(gt, matched) if p is not None]
    coverage = len(valid) / len(gt)
    ids = [p.track_id for _, p in valid]
    switches = sum(a != b for a, b in zip(ids, ids[1:]))
    errors = []
    for gt_obs, pred_obs in valid:
        scale = max(20.0, math.hypot(gt_obs.bbox[2], gt_obs.bbox[3]))
        errors.append(float(np.linalg.norm(gt_obs.center - pred_obs.center) / scale))
    return coverage, switches, float(np.mean(errors)) if errors else 10.0


def spatial_movement(track: list[TrackObservation]) -> float:
    points = np.asarray([item.center for item in track])
    return float(
        math.hypot(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    )


def build_candidates(
    comparison: str,
    domain: str,
    ours: dict,
    comp: dict,
    min_iou: float,
    min_length: int,
    min_movement: float,
    window_length: int,
) -> list[Candidate]:
    annotation = load_json(annotation_path(domain))
    tracks, video_names = gt_tracks(annotation)
    candidates = []
    for (video_id, gt_id), gt in tracks.items():
        movement = spatial_movement(gt)
        if len(gt) < min_length or movement < min_movement:
            continue
        ours_match = match_track(gt, ours, video_id, min_iou)
        comp_match = match_track(gt, comp, video_id, min_iou)
        best = None
        for start in range(0, len(gt) - window_length + 1, 5):
            stop = start + window_length
            gt_window = gt[start:stop]
            ours_window = ours_match[start:stop]
            comp_window = comp_match[start:stop]
            window_movement = spatial_movement(gt_window)
            oc, os, oe = track_stats(gt_window, ours_window)
            cc, cs, ce = track_stats(gt_window, comp_window)
            error_gain = ce - oe
            if (
                window_movement < min_movement
                or oc < 0.95
                or cc < 0.80
                or oe > 0.035
                or ce < 0.075
                or error_gain < 0.04
            ):
                continue
            score = 20.0 * error_gain + 0.5 * (oc - cc) - oe
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
                    window_movement,
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
            window_movement,
        ) = best
        candidates.append(
            Candidate(
                comparison=comparison,
                domain=domain,
                video_id=video_id,
                video_name=video_names[video_id],
                gt_id=gt_id,
                gt_length=len(gt_window),
                movement=window_movement,
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


def select_diverse(candidates: list[Candidate], count: int) -> list[Candidate]:
    selected = []
    used_gt_ids = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if candidate.gt_id in used_gt_ids:
            continue
        selected.append(candidate)
        used_gt_ids.add(candidate.gt_id)
        if len(selected) == count:
            break
    return selected


def segments(
    observations: list[TrackObservation | None],
) -> list[np.ndarray]:
    result = []
    current = []
    current_id = None
    previous_frame = None
    for obs in observations:
        if obs is None:
            if current:
                result.append(np.asarray(current))
            current, current_id, previous_frame = [], None, None
            continue
        if current and (
            obs.track_id != current_id
            or (previous_frame is not None and obs.frame != previous_frame + 1)
        ):
            result.append(np.asarray(current))
            current = []
        current.append(obs.center)
        current_id = obs.track_id
        previous_frame = obs.frame
    if current:
        result.append(np.asarray(current))
    return result


def plot_candidate(candidate: Candidate, index: int, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FCFCFC")

    gt_points = np.asarray([item.center for item in candidate.gt])
    ax.plot(
        gt_points[:, 0],
        gt_points[:, 1],
        color=COLORS["gt"],
        linewidth=1.0,
        linestyle="-",
        marker="x",
        markersize=5.2,
        markeredgewidth=1.0,
        alpha=0.82,
        label="Ground truth",
        zorder=1,
    )

    comp_label = "DARTH" if candidate.comparison == "darth" else "CTAOD"
    for segment_index, points in enumerate(segments(candidate.comp)):
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=COLORS[candidate.comparison],
            linewidth=1.15,
            marker="s",
            markersize=5.4,
            markeredgewidth=0,
            alpha=0.78,
            label=comp_label if segment_index == 0 else None,
            zorder=2,
        )
    for segment_index, points in enumerate(segments(candidate.ours)):
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=COLORS["ours"],
            linewidth=1.65,
            marker="o",
            markersize=7.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            alpha=0.96,
            label="Ours" if segment_index == 0 else None,
            zorder=4,
        )

    all_points = [gt_points]
    all_points.extend(segments(candidate.ours))
    all_points.extend(segments(candidate.comp))
    all_points = [item for item in all_points if len(item)]
    merged = np.concatenate(all_points, axis=0)
    x_margin = max(28.0, 0.08 * np.ptp(merged[:, 0]))
    y_margin = max(20.0, 0.10 * np.ptp(merged[:, 1]))
    ax.set_xlim(np.min(merged[:, 0]) - x_margin, np.max(merged[:, 0]) + x_margin)
    ax.set_ylim(np.max(merged[:, 1]) + y_margin, np.min(merged[:, 1]) - y_margin)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#353535")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)

    display_video_name = candidate.video_name.replace("_", " ")
    title = (
        f"{display_video_name}  |  GT ID {candidate.gt_id}  |  "
        f"Frames {candidate.gt[0].frame}-{candidate.gt[-1].frame}"
    )
    ax.set_title(title, fontsize=23.0, fontweight="semibold", pad=17)
    stats = (
        f"Ours: error {candidate.ours_error:.3f}, "
        f"coverage {100*candidate.ours_coverage:.1f}%     |     "
        f"{comp_label}: error {candidate.comp_error:.3f}, "
        f"coverage {100*candidate.comp_coverage:.1f}%"
    )
    ax.text(
        0.5,
        0.018,
        stats,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=15.5,
        color="#333333",
    )
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Ours"), labels.index(comp_label), labels.index("Ground truth")]
    legend = ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        frameon=False,
        ncol=3,
        fontsize=18.0,
        handlelength=2.7,
        columnspacing=1.7,
    )
    legend.get_texts()[0].set_fontweight("bold")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tracklet_{index:02d}_{candidate.comparison}_{candidate.domain}"
    fig.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_candidates(path: Path, candidates: list[Candidate], selected: set[tuple]) -> None:
    fields = [
        "selected",
        "comparison",
        "domain",
        "video_name",
        "gt_id",
        "frame_start",
        "frame_end",
        "gt_length",
        "movement",
        "ours_coverage",
        "comp_coverage",
        "ours_switches",
        "comp_switches",
        "ours_error",
        "comp_error",
        "error_gap",
        "score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in sorted(candidates, key=lambda value: value.score, reverse=True):
            key = (item.comparison, item.domain, item.video_name, item.gt_id)
            writer.writerow(
                {
                    "selected": int(key in selected),
                    "comparison": item.comparison,
                    "domain": item.domain,
                    "video_name": item.video_name,
                    "gt_id": item.gt_id,
                    "frame_start": item.gt[0].frame,
                    "frame_end": item.gt[-1].frame,
                    "gt_length": item.gt_length,
                    "movement": f"{item.movement:.4f}",
                    "ours_coverage": f"{item.ours_coverage:.6f}",
                    "comp_coverage": f"{item.comp_coverage:.6f}",
                    "ours_switches": item.ours_switches,
                    "comp_switches": item.comp_switches,
                    "ours_error": f"{item.ours_error:.6f}",
                    "comp_error": f"{item.comp_error:.6f}",
                    "error_gap": f"{item.comp_error - item.ours_error:.6f}",
                    "score": f"{item.score:.6f}",
                }
            )


def main() -> None:
    args = parse_args()
    all_candidates = []
    inputs = {}
    for domain in DOMAINS:
        ours_path = ours_track_path(domain)
        darth_path = darth_track_path(domain)
        ours = load_coco_tracks(ours_path)
        darth = load_coco_tracks(darth_path)
        ctaod = rebuild_ctta_tracks(domain, args.output_dir)
        inputs[domain] = {
            "ours": str(ours_path),
            "darth": str(darth_path),
            "annotation": str(annotation_path(domain)),
            "ctaod_variant": "balanced_vk_area2200",
        }
        all_candidates.extend(
            build_candidates(
                "darth",
                domain,
                ours,
                darth,
                args.match_iou,
                args.min_track_length,
                args.min_movement,
                args.window_length,
            )
        )
        all_candidates.extend(
            build_candidates(
                "ctaod",
                domain,
                ours,
                ctaod,
                args.match_iou,
                args.min_track_length,
                args.min_movement,
                args.window_length,
            )
        )

    selected = []
    for comparison in ("darth", "ctaod"):
        selected.extend(
            select_diverse(
                [item for item in all_candidates if item.comparison == comparison],
                args.per_comparison,
            )
        )
    selected_keys = {
        (item.comparison, item.domain, item.video_name, item.gt_id)
        for item in selected
    }
    output_images = args.output_dir / "figures"
    output_images.mkdir(parents=True, exist_ok=True)
    for old_figure in output_images.glob("tracklet_*"):
        old_figure.unlink()
    for index, item in enumerate(selected, 1):
        plot_candidate(item, index, output_images)
    write_candidates(args.output_dir / "candidate_scores.csv", all_candidates, selected_keys)
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(
            {
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
            },
            handle,
            indent=2,
        )
    print(json.dumps(load_json(args.output_dir / "manifest.json")["selection"], indent=2))


if __name__ == "__main__":
    main()
