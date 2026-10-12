#!/usr/bin/env python3
"""Concatenate disjoint sunny-source V-JEPA 2.1 GT caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED = ("features", "offsets", "image_ids", "video_ids", "frame_ids",
            "track_ids", "boxes")


def normalize_image_index(features, offsets, image_ids):
    """Return one offset interval and one image id for every nonempty image.

    The source extractor retains offsets for empty GT frames, while its row
    metadata is intentionally detection-only.  Collapse the latter into the
    compact row-index used by the training cache without inventing any rows.
    """
    if offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0:
        raise ValueError("invalid offsets")
    if offsets[-1] != len(features):
        raise ValueError("offsets do not terminate at the feature row count")
    if len(offsets) == len(image_ids) + 1:
        compact_offsets = offsets
        compact_ids = image_ids
    elif len(image_ids) == len(features):
        starts = np.flatnonzero(
            np.r_[True, image_ids[1:] != image_ids[:-1]]
        ).astype(np.int64)
        compact_offsets = np.concatenate(
            [starts, np.asarray([len(features)], dtype=np.int64)]
        )
        compact_ids = image_ids[starts]
    else:
        raise ValueError(
            "image_ids must be row-aligned or offset-aligned: "
            f"features={len(features)} offsets={len(offsets)} "
            f"image_ids={len(image_ids)}"
        )
    if len(compact_offsets) != len(compact_ids) + 1:
        raise ValueError("failed to compact image index")
    if np.any(np.diff(compact_offsets) <= 0):
        raise ValueError("compact image intervals must be nonempty")
    return compact_offsets, compact_ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    features = []
    image_ids = []
    video_ids = []
    frame_ids = []
    track_ids = []
    boxes = []
    offsets = [0]
    seen_videos = set()
    part_meta = []
    for raw_path in args.inputs:
        path = raw_path.resolve()
        sidecar_path = path.with_suffix(path.suffix + ".json")
        if not path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"cache or sidecar missing: {path}")
        with np.load(path, allow_pickle=False) as cache:
            missing = [key for key in REQUIRED if key not in cache.files]
            if missing:
                raise ValueError(f"{path} missing fields: {missing}")
            f = np.asarray(cache["features"], dtype=np.float32)
            off = np.asarray(cache["offsets"], dtype=np.int64)
            ids = np.asarray(cache["image_ids"], dtype=np.int64)
            vids = np.asarray(cache["video_ids"], dtype=np.int64)
            frames = np.asarray(cache["frame_ids"], dtype=np.int64)
            tracks = np.asarray(cache["track_ids"], dtype=np.int64)
            b = np.asarray(cache["boxes"], dtype=np.float32)
        if f.ndim != 2:
            raise ValueError(f"invalid feature shape in {path}")
        off, ids = normalize_image_index(f, off, ids)
        if any(len(x) != len(f) for x in (vids, frames, tracks, b)):
            raise ValueError(f"row metadata mismatch in {path}")
        part_videos = set(int(x) for x in np.unique(vids).tolist())
        if seen_videos & part_videos:
            raise ValueError(f"source video overlap in {path}")
        seen_videos.update(part_videos)
        base = offsets[-1]
        features.append(f)
        image_ids.append(ids)
        video_ids.append(vids)
        frame_ids.append(frames)
        track_ids.append(tracks)
        boxes.append(b)
        offsets.extend((off[1:] + base).tolist())
        part_meta.append(json.loads(sidecar_path.read_text()))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    all_features = np.concatenate(features, axis=0)
    np.savez_compressed(
        output,
        features=all_features,
        offsets=np.asarray(offsets, dtype=np.int64),
        image_ids=np.concatenate(image_ids),
        video_ids=np.concatenate(video_ids),
        frame_ids=np.concatenate(frame_ids),
        track_ids=np.concatenate(track_ids),
        boxes=np.concatenate(boxes, axis=0),
    )
    meta = {
        "schema": "uadetrac-sunny-source-gt-official-vjepa21-cache-v1",
        "status": "complete",
        "inputs": [str(p.resolve()) for p in args.inputs],
        "input_sha256": [sha256_file(p.resolve()) for p in args.inputs],
        "feature_shape": [int(x) for x in all_features.shape],
        "feature_dtype": "float32",
        "num_source_videos": int(len(seen_videos)),
        "num_source_frames": int(sum(
            int(m.get("num_source_frames", 0)) for m in part_meta
        )),
        "num_source_images_with_gt": int(len(offsets) - 1),
        "num_gt_rows": int(len(all_features)),
        "window": int(part_meta[0].get("window", 16)),
        "roi_mode": part_meta[0].get("roi_mode"),
        "roi_context": part_meta[0].get("roi_context"),
        "predictor_pool": part_meta[0].get("predictor_pool"),
        "causal": all(bool(m.get("causal", False)) for m in part_meta),
        "future_frames_read": any(bool(m.get("future_frames_read", True)) for m in part_meta),
        "gt_read": all(bool(m.get("gt_read", False)) for m in part_meta),
        "source_domain": "sunny",
        "target_domains_read": [],
        "predictor_used": all(bool(m.get("predictor_used", False)) for m in part_meta),
        "parts": part_meta,
        "output": str(output),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
