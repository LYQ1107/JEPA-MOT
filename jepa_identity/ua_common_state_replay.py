"""Bounded CPU common-state replay for the existing UA-DETRAC tracks.

The registered H1 event tracer was run on BDD100K and requires a GPU-backed
closed-loop tracker for a fresh UA run.  This module deliberately has a
narrower scope: it replays assignment changes already present in the existing
UA ``predictive_ranking`` track files, reconstructs a geometry-ranked top-six
candidate set from the cached detector stream, and then evaluates the same
candidate-independent controls used by :mod:`common_state_audit`.

It is an evidence extension, not a replacement for the registered GPU
Wave-A.  No target label is used to construct a query or choose a candidate;
XML labels are read only after the replay state and every score are ready.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import torch

from .common_state_audit import (
    CONTROL_NAMES,
    EVALUABLE,
    VideoFeatureIndex,
    load_predictor,
    sha256_file,
    summarize,
)
from jepa_cmot.history import pack_history


COCO6_TO_NATIVE8 = (0, 2, 3, 4, 6, 7)
DEFAULT_VIDEO_ROOT = Path("runs/JEPA_CMOT/paired_readout_v1/development/ua_detrac_predictor_seed17")
DEFAULT_FORMAL_ROOT = Path("runs/JEPA_CMOT/formal_detection_streams_v1/ua_detrac")
DEFAULT_XML_ROOT = Path("/data1/liuyeqiang/nas/A100/testdata/vranlee/.MOTSynth.partial/UA-DETRAC")


def xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> list[float]:
    x, y, width, height = map(float, box[:4])
    return [x, y, x + width, y + height]


def xyxy_iou(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, left)
    bx1, by1, bx2, by2 = map(float, right)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def parse_xml_labels(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Return vehicle boxes keyed by native frame number."""

    root = ET.parse(path).getroot()
    labels: dict[int, list[dict[str, Any]]] = {}
    for frame in root.findall(".//frame"):
        frame_id = int(frame.attrib["num"])
        rows: list[dict[str, Any]] = []
        for target in frame.findall("./target_list/target"):
            box = target.find("./box")
            if box is None:
                continue
            rows.append({
                "gt_id": int(target.attrib["id"]),
                "box": xywh_to_xyxy([
                    float(box.attrib["left"]),
                    float(box.attrib["top"]),
                    float(box.attrib["width"]),
                    float(box.attrib["height"]),
                ]),
            })
        labels[frame_id] = rows
    return labels


def match_gt(labels: dict[int, list[dict[str, Any]]], frame: int,
             box: list[float] | np.ndarray, threshold: float = 0.50) -> tuple[int | None, float]:
    best_id: int | None = None
    best_iou = 0.0
    for row in labels.get(int(frame), []):
        overlap = xyxy_iou(box, row["box"])
        if overlap > best_iou:
            best_iou = overlap
            best_id = int(row["gt_id"])
    return (best_id, best_iou) if best_iou >= threshold else (None, best_iou)


def parse_tracks(path: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[int, int]]:
    """Read MOTChallenge-style output and return per-track rows and births."""

    tracks: dict[int, list[dict[str, Any]]] = {}
    for line in path.read_text().splitlines():
        fields = line.strip().split(",")
        if len(fields) < 8:
            continue
        frame, track_id = int(float(fields[0])), int(float(fields[1]))
        box = xywh_to_xyxy([float(value) for value in fields[2:6]])
        tracks.setdefault(track_id, []).append({
            "frame": frame,
            "box": box,
            "confidence": float(fields[6]),
            "native_class": int(float(fields[7])),
        })
    births: dict[int, int] = {}
    for track_id, rows in tracks.items():
        rows.sort(key=lambda row: (int(row["frame"]), int(track_id)))
        births[int(track_id)] = int(rows[0]["frame"])
    return tracks, births


def entry_for_video(video: str, input_path: Path, formal_root: Path) -> dict[str, Any]:
    records = json.loads(input_path.read_text())
    if not records:
        raise ValueError(f"empty input manifest: {input_path}")
    record = dict(records[0])
    prepared = Path(record["prepared"])
    if not prepared.exists():
        raise FileNotFoundError(prepared)
    video_dir = prepared.parent
    features = video_dir / "features_dense_current.npz"
    if not features.exists():
        features = formal_root / str(record["domain"]) / "development" / video / "features_dense_current.npz"
    if not features.exists():
        raise FileNotFoundError(features)
    return {
        "dataset": "ua_detrac",
        "domain": str(record["domain"]),
        "video": video,
        "frames": int(record["frames"]),
        "prepared": str(prepared),
        "features": str(features),
    }


def _candidate_row(index: VideoFeatureIndex, labels: dict[int, list[dict[str, Any]]],
                   frame: int, row: dict[str, Any], native_cost: float) -> dict[str, Any]:
    gt_id, gt_iou = match_gt(labels, frame, row["box"])
    return {
        "index": int(row["index"]),
        "box": np.asarray(row["box"], dtype=np.float32).tolist(),
        "native_class": int(row["native_class"]),
        "gt_id": gt_id,
        "iou": float(gt_iou),
        "native_cost": float(native_cost),
        "fused_cost": float(native_cost),
        "semantic": 0.0,
        "detection_score": float(row.get("score", 0.0)),
    }


def build_cases(video: str, entry: dict[str, Any], tracks_path: Path,
                xml_path: Path, *, max_cases: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build assignment-change cases without consulting target labels for selection."""

    index = VideoFeatureIndex(entry)
    labels = parse_xml_labels(xml_path)
    tracks, births = parse_tracks(tracks_path)
    cases: list[dict[str, Any]] = []
    counters = Counter()
    for track_id in sorted(tracks):
        rows = tracks[track_id]
        for position in range(3, len(rows)):
            current = rows[position]
            history_rows = rows[max(0, position - 4):position]
            frame = int(current["frame"])
            if frame not in index.images or not history_rows:
                counters["missing_detector_frame"] += 1
                continue
            # The replay uses only the old track output and detector geometry.
            # Labels are consulted below only to annotate the already selected
            # assignment change.
            detector_rows = index.frame_rows.get(frame, [])
            if not detector_rows:
                counters["empty_detector_frame"] += 1
                continue
            assigned = max(
                detector_rows,
                key=lambda row: (xyxy_iou(current["box"], row["box"]), -int(row["index"])),
            )
            assigned_iou = xyxy_iou(current["box"], assigned["box"])
            if assigned_iou < 0.90:
                counters["track_detection_join_failed"] += 1
                continue
            last_box = history_rows[-1]["box"]
            ordered = sorted(
                detector_rows,
                key=lambda row: (1.0 - xyxy_iou(last_box, row["box"]), int(row["index"])),
            )
            native_rows = ordered[:6]
            native_best = native_rows[0]
            if int(assigned["index"]) == int(native_best["index"]):
                counters["not_a_geometry_change"] += 1
                continue
            if all(int(row["index"]) != int(assigned["index"]) for row in native_rows):
                native_rows = native_rows[:-1] + [assigned]
                native_rows.sort(
                    key=lambda row: (1.0 - xyxy_iou(last_box, row["box"]), int(row["index"]))
                )
            expected_ids = []
            history = []
            for observation in history_rows:
                gt_id, gt_iou = match_gt(labels, int(observation["frame"]), observation["box"])
                if gt_id is not None:
                    expected_ids.append(int(gt_id))
                history.append({
                    "frame": int(observation["frame"]),
                    "box": list(observation["box"]),
                    "confidence": float(observation["confidence"]),
                })
            if not expected_ids:
                counters["unlabeled_history"] += 1
                continue
            counts = Counter(expected_ids)
            expected_gt_id, frequency = counts.most_common(1)[0]
            purity = frequency / len(expected_ids)
            if purity < 0.75:
                counters["mixed_history"] += 1
                continue
            candidate_gt_ids = []
            for row in native_rows:
                cost = 1.0 - xyxy_iou(last_box, row["box"])
                candidate_gt_ids.append(_candidate_row(index, labels, frame, row, cost))
            native_choice = next(row for row in candidate_gt_ids if int(row["index"]) == int(native_best["index"]))
            jepa_choice = next(row for row in candidate_gt_ids if int(row["index"]) == int(assigned["index"]))
            native_gt_id = native_choice.get("gt_id")
            jepa_gt_id = jepa_choice.get("gt_id")
            native_correct = native_gt_id == expected_gt_id
            jepa_correct = jepa_gt_id == expected_gt_id
            if native_correct and jepa_correct:
                diagnosis = "both_correct"
            elif not native_correct and not jepa_correct:
                diagnosis = "neither_correct"
            elif not native_correct and jepa_correct:
                diagnosis = "corrected_local_identity"
            else:
                diagnosis = "harmful_local_identity"
            cases.append({
                "method": "predictive_residual_audit",
                "replay_protocol": "ua_legacy_track_geometry_change_v1",
                "domain": str(entry["domain"]),
                "video": video,
                "frame": frame,
                "branch": "legacy_predictive_ranking_vs_geometry",
                "track_id": int(track_id),
                "diagnosis": diagnosis,
                "history_purity": float(purity),
                "history": history,
                "expected_gt_id": int(expected_gt_id),
                "native_gt_id": native_gt_id,
                "jepa_gt_id": jepa_gt_id,
                "native_iou": float(native_choice.get("iou", 0.0)),
                "jepa_iou": float(jepa_choice.get("iou", 0.0)),
                "candidate_gt_ids": candidate_gt_ids,
                "native_choice": native_choice,
                "jepa_choice": jepa_choice,
                "correct_candidate_present": any(row.get("gt_id") == expected_gt_id for row in candidate_gt_ids),
                "candidate_count": len(candidate_gt_ids),
                "last_observation_gap": frame - int(history_rows[-1]["frame"]),
                "native_class": int(current["native_class"]),
                "track_birth_frame": int(births[int(track_id)]),
            })
            counters["selected_cases"] += 1
            if max_cases is not None and len(cases) >= max_cases:
                break
        if max_cases is not None and len(cases) >= max_cases:
            break
    return cases, {
        "video": video,
        "xml": str(xml_path),
        "xml_sha256": sha256_file(xml_path),
        "tracks": str(tracks_path),
        "tracks_sha256": sha256_file(tracks_path),
        "track_count": len(tracks),
        "track_rows": sum(len(rows) for rows in tracks.values()),
        "counters": dict(counters),
        "selected_case_count": len(cases),
    }


def unit(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def _prepared_case(case: dict[str, Any], index: VideoFeatureIndex,
                   births: dict[int, int]) -> dict[str, Any] | None:
    history_rows = []
    for row in case.get("history", []):
        match = index.match(int(row["frame"]), row["box"], int(case.get("native_class", 0)))
        if match is None:
            return None
        history_rows.append((row, match))
    candidate_rows = []
    for candidate in case.get("candidate_gt_ids", []):
        match = index.match(int(case["frame"]), candidate["box"], int(candidate.get("native_class", 0)))
        if match is None:
            return None
        candidate_rows.append((candidate, match))
    if not history_rows or not candidate_rows:
        return None
    image = index.images[int(case["frame"])]
    width, height = int(image["width"]), int(image["height"])
    birth = int(case.get("track_birth_frame", births.get(int(case["track_id"]), history_rows[0][0]["frame"])))

    def observations(control: str) -> list[dict[str, Any]]:
        return [{
            "feature": index.transformed_feature(match, control),
            "box": np.asarray(row["box"], dtype=np.float32),
            "frame": int(row["frame"]),
            "birth": birth,
            "confidence": float(row.get("confidence", 1.0)),
        } for row, match in history_rows]

    original = observations("original")
    shuffled = observations("feature_shuffle")
    random = observations("feature_random")
    history_features, history_metadata, history_mask = pack_history(
        original, int(case["frame"]), 4, 1024, width, height
    )
    shuffle_features, shuffle_metadata, shuffle_mask = pack_history(
        shuffled, int(case["frame"]), 4, 1024, width, height
    )
    random_features, random_metadata, random_mask = pack_history(
        random, int(case["frame"]), 4, 1024, width, height
    )
    candidate_features = np.asarray([match["feature"] for _, match in candidate_rows], dtype=np.float32)
    shuffle_candidates = np.asarray([
        index.transformed_feature(match, "feature_shuffle") for _, match in candidate_rows
    ], dtype=np.float32)
    random_candidates = np.asarray([
        index.transformed_feature(match, "feature_random") for _, match in candidate_rows
    ], dtype=np.float32)
    last_box = np.asarray(case["history"][-1]["box"], dtype=np.float32)
    candidate_boxes = [np.asarray(candidate["box"], dtype=np.float32) for candidate, _ in candidate_rows]
    last = unit(original[-1]["feature"].reshape(1, -1))[0]
    mean4 = unit(np.sum([row["feature"] for row in original[-4:]], axis=0).reshape(1, -1))[0]
    ema_value = None
    for row in original:
        value = unit(row["feature"].reshape(1, -1))[0]
        ema_value = value if ema_value is None else unit((0.25 * value + 0.75 * ema_value).reshape(1, -1))[0]
    return {
        "case": case,
        "history_features": history_features,
        "history_metadata": history_metadata,
        "history_mask": history_mask,
        "shuffle_features": shuffle_features,
        "shuffle_metadata": shuffle_metadata,
        "shuffle_mask": shuffle_mask,
        "random_features": random_features,
        "random_metadata": random_metadata,
        "random_mask": random_mask,
        "candidate_features": candidate_features,
        "shuffle_candidates": shuffle_candidates,
        "random_candidates": random_candidates,
        "baseline_scores": {
            "geometry_iou": [1.0 - xyxy_iou(last_box, box) for box in candidate_boxes],
            "history_last": (1.0 - unit(candidate_features) @ last).tolist(),
            "history_mean4": (1.0 - unit(candidate_features) @ mean4).tolist(),
            "history_ema": (1.0 - unit(candidate_features) @ ema_value).tolist(),
            "native_recorded": [float(candidate.get("native_cost", 0.0)) for candidate, _ in candidate_rows],
            "feature_shuffle": (1.0 - unit(shuffle_candidates) @ unit(shuffled[-1]["feature"])).tolist(),
            "feature_random": (1.0 - unit(random_candidates) @ unit(random[-1]["feature"])).tolist(),
            "constant_affine": [0.0 for _ in candidate_rows],
        },
        "metadata": {
            "history_rows": len(history_rows),
            "candidate_rows": len(candidate_rows),
            "min_history_match_iou": min(float(match["match_iou"]) for _, match in history_rows),
            "min_candidate_match_iou": min(float(match["match_iou"]) for _, match in candidate_rows),
            "birth_source": "replayed_track_file",
        },
    }


def _batch_model_scores(model: torch.nn.Module, prepared: list[dict[str, Any]],
                        history_key: str, metadata_key: str, mask_key: str,
                        candidate_key: str, batch_size: int) -> list[list[float]]:
    output: list[list[float] | None] = [None] * len(prepared)
    # Candidate counts vary by frame.  Grouping by count keeps the model
    # forward batched without padding a candidate mask into the reported
    # score lists.
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(prepared):
        groups.setdefault(int(row[candidate_key].shape[0]), []).append(index)
    model.eval()
    with torch.no_grad():
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch = [prepared[index] for index in batch_indices]
                history_features = torch.from_numpy(np.stack([row[history_key] for row in batch]))
                history_metadata = torch.from_numpy(np.stack([row[metadata_key] for row in batch]))
                history_mask = torch.from_numpy(np.stack([row[mask_key] for row in batch]))
                queries = model(history_features, history_metadata, history_mask).numpy()
                queries = unit(queries)
                candidates = unit(np.stack([row[candidate_key] for row in batch]))
                costs = 1.0 - np.einsum("bd,bnd->bn", queries, candidates)
                for local, original in enumerate(batch_indices):
                    output[original] = costs[local].astype(np.float64).tolist()
    return [row if row is not None else [] for row in output]


def score_cases(cases: list[dict[str, Any]], entries: dict[str, dict[str, Any]],
                track_roots: dict[str, Path], model_paths: dict[str, Path],
                batch_size: int = 128) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    models = {}
    for name, path in model_paths.items():
        models[name], _ = load_predictor(path)
    indices: dict[str, VideoFeatureIndex] = {}
    births: dict[str, dict[int, int]] = {}
    prepared: list[dict[str, Any]] = []
    skips = Counter()
    skips_by_video: dict[str, int] = Counter()
    for case in cases:
        video = str(case["video"])
        if video not in indices:
            indices[video] = VideoFeatureIndex(entries[video])
            _, births[video] = parse_tracks(track_roots[video])
        try:
            item = _prepared_case(case, indices[video], births[video])
        except (KeyError, IndexError, ValueError, StopIteration, RuntimeError):
            item = None
        if item is None:
            skips["feature_box_mapping_failed"] += 1
            skips_by_video[video] += 1
            continue
        prepared.append(item)
    model_specs = {
        "mlp_ce": ("history_features", "history_metadata", "history_mask", "candidate_features"),
        "mlp_jepa_predictive": ("history_features", "history_metadata", "history_mask", "candidate_features"),
        "mlp_jepa_ranking": ("history_features", "history_metadata", "history_mask", "candidate_features"),
        "transformer_ce": ("history_features", "history_metadata", "history_mask", "candidate_features"),
        "transformer_jepa_predictive": ("history_features", "history_metadata", "history_mask", "candidate_features"),
        "transformer_jepa_ranking": ("history_features", "history_metadata", "history_mask", "candidate_features"),
    }
    model_scores = {
        name: _batch_model_scores(model, prepared, *spec, batch_size)
        for name, (model, spec) in ((name, (models[name], model_specs[name])) for name in model_specs)
    }
    scored: list[dict[str, Any]] = []
    for index, item in enumerate(prepared):
        enriched = dict(item["case"])
        enriched["scores"] = dict(item["baseline_scores"])
        for name, values in model_scores.items():
            enriched["scores"][name] = values[index]
        enriched["common_state_metadata"] = item["metadata"]
        scored.append(enriched)
    controls = {
        control: summarize(scored, control, 1700 + index * 101)
        for index, control in enumerate(CONTROL_NAMES)
    }
    by_domain = {}
    for domain in sorted({str(case["domain"]) for case in scored}):
        subset = [case for case in scored if str(case["domain"]) == domain]
        by_domain[domain] = {
            control: summarize(subset, control, 4700 + index * 101)
            for index, control in enumerate(CONTROL_NAMES)
        }
    return scored, {
        "scored_case_count": len(scored),
        "skipped_cases": dict(skips),
        "skipped_cases_by_video": dict(skips_by_video),
        "controls": controls,
        "by_domain": by_domain,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    video_root = Path(args.video_root)
    formal_root = Path(args.formal_root)
    xml_root = Path(args.xml_root)
    input_paths = sorted((video_root / "inputs").glob("*.json"))
    if not input_paths:
        raise FileNotFoundError(video_root / "inputs")
    cases: list[dict[str, Any]] = []
    entries: dict[str, dict[str, Any]] = {}
    track_paths: dict[str, Path] = {}
    video_reports = []
    per_video_limit = args.max_cases_per_video
    for input_path in input_paths:
        record = json.loads(input_path.read_text())[0]
        video = str(record["video"])
        entry = entry_for_video(video, input_path, formal_root)
        entries[video] = entry
        track_path = video_root / "videos" / video / "predictive_ranking" / video / "tracks.txt"
        if not track_path.exists():
            raise FileNotFoundError(track_path)
        track_paths[video] = track_path
        annotation_path = Path(entry["prepared"]) / "annotations" / "val.json"
        raw = json.loads(annotation_path.read_text())
        first_file = str(raw["images"][0]["file_name"])
        split = "Train" if "Annotation_Train" in first_file else "Test"
        xml_path = xml_root / f"DETRAC-{split}-Annotations-XML" / f"{video}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(xml_path)
        video_cases, report = build_cases(video, entry, track_path, xml_path, max_cases=per_video_limit)
        cases.extend(video_cases)
        report.update({
            "domain": entry["domain"],
            "prepared": entry["prepared"],
            "prepared_sha256": sha256_file(annotation_path),
            "features": entry["features"],
            "features_sha256": sha256_file(Path(entry["features"])),
        })
        video_reports.append(report)
    model_paths = {
        "mlp_ce": Path(args.mlp_ce),
        "mlp_jepa_predictive": Path(args.mlp_predictive),
        "mlp_jepa_ranking": Path(args.mlp_ranking),
        "transformer_ce": Path(args.transformer_ce),
        "transformer_jepa_predictive": Path(args.transformer_predictive),
        "transformer_jepa_ranking": Path(args.transformer_ranking),
    }
    for path in model_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    scored, summary = score_cases(cases, entries, track_paths, model_paths)
    args.out_cases.parent.mkdir(parents=True, exist_ok=True)
    with args.out_cases.open("w") as stream:
        for case in scored:
            stream.write(json.dumps(case, separators=(",", ":"), ensure_ascii=False) + "\n")
    report = {
        "status": "COMPLETE_UA_LEGACY_TRACK_CPU_COMMON_STATE_REPLAY_SEED17",
        "scope": "UA-DETRAC development videos represented by the existing predictive_ranking track files; only geometry-vs-track assignment changes are selected",
        "protocol": "ua_legacy_track_geometry_change_v1",
        "source_seed": 17,
        "input_video_count": len(input_paths),
        "source_case_count": len(cases),
        "scored_case_count": int(summary["scored_case_count"]),
        "skipped_cases": summary["skipped_cases"],
        "skipped_cases_by_video": summary["skipped_cases_by_video"],
        "videos": [item["video"] for item in video_reports],
        "domains": sorted({str(entry["domain"]) for entry in entries.values()}),
        "video_reports": video_reports,
        "controls": summary["controls"],
        "by_domain": summary["by_domain"],
        "model_checkpoints": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in model_paths.items()
        },
        "feature_source": "formal_detection_streams_v1 UA dense_current 1024-dimensional cache",
        "label_source": "raw UA-DETRAC Train/Test XML on the mounted NAS; labels read only after replay selection and scoring inputs are fixed",
        "cuda_available": bool(__import__("torch").cuda.is_available()),
        "target_GT_used_for_training_or_candidate_selection": False,
        "target_GT_used_for_evaluability_filter": True,
        "offline_GT_used_for_diagnosis": True,
        "sealed_used": False,
        "limitations": [
            "this is not a fresh GeneralTrack or JEPA closed-loop rollout; it reuses existing predictive_ranking tracks",
            "native candidates are reconstructed by last-history-box IoU and are not the original GeneralTrack native cost matrix",
            "candidate changes are selected from the old track output versus that geometry replay, so the result is not the registered BDD predictive-residual event audit",
            "the six controls use existing UA seed17 source checkpoints; no new training or extra seed was run",
            "candidate set is the reconstructed top-six subset and does not expose every post-matcher gate",
            "the result supplements H1 diagnosis and cannot establish the registered full-video Wave-A or an online HOTA/IDF1 claim",
        ],
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("UA_LEGACY_TRACK_REPLAY_COMPLETE", len(scored), flush=True)
    return report


def main() -> None:
    root = Path(".")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL_ROOT)
    parser.add_argument("--xml-root", type=Path, default=DEFAULT_XML_ROOT)
    parser.add_argument("--out-cases", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--max-cases-per-video", type=int, default=None)
    parser.add_argument("--mlp-ce", type=Path, default=root / "runs/JEPA_CMOT/source_history_v2/ua_dense_current/seed17/discriminative/final.pt")
    parser.add_argument("--mlp-predictive", type=Path, default=root / "runs/JEPA_CMOT/source_history_v2/ua_dense_current/seed17/predictive/final.pt")
    parser.add_argument("--mlp-ranking", type=Path, default=root / "runs/JEPA_CMOT/source_history_v2/ua_dense_current/seed17/predictive_ranking/final.pt")
    parser.add_argument("--transformer-ce", type=Path, default=root / "runs/JEPA_CMOT/transformer_v1/models/ua_detrac/seed17/discriminative/final.pt")
    parser.add_argument("--transformer-predictive", type=Path, default=root / "runs/JEPA_CMOT/transformer_v1/models/ua_detrac/seed17/predictive/final.pt")
    parser.add_argument("--transformer-ranking", type=Path, default=root / "runs/JEPA_CMOT/transformer_v1/models/ua_detrac/seed17/predictive_ranking/final.pt")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
