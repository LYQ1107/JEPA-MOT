#!/usr/bin/env python3
"""Create deterministic per-domain video splits and matching JEPA caches.

The original prepared annotations and full caches are never modified.  A
periodic video split is used so that tuning metrics and holdout metrics are
computed on disjoint videos within every target domain.
"""

import copy
import gc
import json
import os
from pathlib import Path

import numpy as np


REPO = Path("/data1/liuyeqiang/JEPA_MOT_resume_20261012/GeneralTrack-main")
JEPA_ROOT = Path("/data1/liuyeqiang/JEPA_MOT_resume_20261012/JEPA")
PREPARED_ROOT = Path(os.environ.get(
    "JEPA_PREPARED_ROOT",
    str(REPO / "datasets/generaltrack_prepared_thr02"),
))
CACHE_ROOT = Path(os.environ.get(
    "JEPA_CACHE_ROOT",
    str(JEPA_ROOT / "outputs/official_vjepa21_cache"),
))
SPLIT_ROOT = Path(os.environ.get(
    "JEPA_SPLIT_ROOT",
    str(JEPA_ROOT / "outputs/jepa_tuning_splits"),
))

DEFAULT_DOMAINS = (
    ("bdd100k", "dawn_dusk"), ("bdd100k", "night"),
    ("bdd100k", "overcast"), ("bdd100k", "rainy"),
    ("vkitti2", "fog"), ("vkitti2", "morning"),
    ("vkitti2", "overcast"), ("vkitti2", "rain"),
    ("vkitti2", "sunset"),
    ("motsynth", "cloudy"), ("motsynth", "dark"),
    ("motsynth", "fog"), ("motsynth", "rain"),
    ("motsynth", "snow"),
)
_domain_text = os.environ.get("JEPA_DOMAINS")
DOMAINS = tuple(
    tuple(item.split("/", 1))
    for item in _domain_text.split()
) if _domain_text else DEFAULT_DOMAINS


def full_cache(dataset, domain):
    return CACHE_ROOT / f"{dataset}_{domain}_vjepa21_l384_w16_causalbox_predictor.npz"


def split_data(data, keep_video_ids):
    keep_video_ids = {int(x) for x in keep_video_ids}
    subset = copy.deepcopy(data)
    selected_videos = sorted(
        (v for v in data.get("videos", []) if int(v["id"]) in keep_video_ids),
        key=lambda v: int(v["id"]),
    )
    # GeneralTrack's BDD evaluator flushes the previous sequence with
    # ``video_id - 1``.  The source annotations use globally spaced video IDs
    # (e.g. 1, 6, 11, ...), so a strict subset must be renumbered locally while
    # retaining image IDs for the JEPA cache lookup.
    video_id_map = {int(video["id"]): index + 1
                    for index, video in enumerate(selected_videos)}
    subset["videos"] = []
    for video in selected_videos:
        video_copy = copy.deepcopy(video)
        video_copy["id"] = video_id_map[int(video["id"])]
        subset["videos"].append(video_copy)
    subset["images"] = []
    for image in data["images"]:
        old_video_id = int(image["video_id"])
        if old_video_id not in video_id_map:
            continue
        image_copy = copy.deepcopy(image)
        image_copy["video_id"] = video_id_map[old_video_id]
        subset["images"].append(image_copy)
    keep_image_ids = {int(i["id"]) for i in subset["images"]}
    subset["annotations"] = [
        copy.deepcopy(a) for a in data.get("annotations", [])
        if int(a.get("image_id", -1)) in keep_image_ids
    ]
    image_video_map = {
        int(image["id"]): int(image["video_id"])
        for image in subset["images"]
    }
    for annotation in subset["annotations"]:
        image_video_id = image_video_map.get(int(annotation.get("image_id", -1)))
        if image_video_id is not None:
            annotation["video_id"] = image_video_id
    return subset


def write_split(dataset, domain, split_name, subset, source_prepared, source_cache):
    split_dir = SPLIT_ROOT / dataset / domain / split_name
    annotation_dir = split_dir / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = annotation_dir / "val.json"
    annotation_path.write_text(json.dumps(subset, indent=2))

    metadata = json.loads((source_prepared / "metadata.json").read_text())
    metadata.update({
        "split": split_name,
        "split_source_prepared": str(source_prepared),
        "annotation_file": str(annotation_path),
        "num_videos": len(subset.get("videos", [])),
        "num_frames": len(subset.get("images", [])),
    })
    (split_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    with np.load(str(source_cache), allow_pickle=False) as cache:
        source_features = np.asarray(cache["features"], dtype=np.float32)
        source_offsets = np.asarray(cache["offsets"], dtype=np.int64)
        source_image_ids = np.asarray(cache["image_ids"], dtype=np.int64)
    source_index = {int(image_id): idx for idx, image_id in enumerate(source_image_ids.tolist())}

    feature_chunks = []
    offsets = [0]
    image_ids = []
    for image in subset["images"]:
        image_id = int(image["id"])
        if image_id not in source_index:
            raise RuntimeError(f"image id {image_id} missing from cache {source_cache}")
        idx = source_index[image_id]
        start, end = int(source_offsets[idx]), int(source_offsets[idx + 1])
        expected = len(image.get("detection", []))
        actual = end - start
        if expected != actual:
            raise RuntimeError(
                f"detection/cache row mismatch for image_id={image_id}: "
                f"annotation={expected} cache={actual} source={source_cache}"
            )
        feature_chunks.append(source_features[start:end])
        offsets.append(offsets[-1] + actual)
        image_ids.append(image_id)
    if feature_chunks:
        features = np.concatenate(feature_chunks, axis=0).astype(np.float32, copy=False)
    else:
        features = np.zeros((0, source_features.shape[1]), dtype=np.float32)
    split_cache = split_dir / "vjepa21_features.npz"
    np.savez_compressed(
        str(split_cache),
        features=features,
        offsets=np.asarray(offsets, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=np.int64),
    )
    source_sidecar = json.loads(Path(str(source_cache) + ".json").read_text())
    source_sidecar.update({
        "status": "complete",
        "split": split_name,
        "split_source_cache": str(source_cache),
        "prepared_dir": str(split_dir),
        "annotation": str(annotation_path),
        "feature_shape": [int(x) for x in features.shape],
        "image_count": len(image_ids),
        "video_count": len(subset.get("videos", [])),
    })
    Path(str(split_cache) + ".json").write_text(json.dumps(source_sidecar, indent=2))
    # The compressed source cache is materialized as a dense array above.
    # Release both the source and split arrays before the next domain so a
    # five-domain run does not retain several hundred MB per iteration.
    del source_features, source_offsets, source_image_ids, source_index
    del feature_chunks, features, offsets, image_ids, source_sidecar
    gc.collect()


def main():
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    for dataset, domain in DOMAINS:
        source_prepared = PREPARED_ROOT / dataset / domain
        source_cache = full_cache(dataset, domain)
        data = json.loads((source_prepared / "annotations/val.json").read_text())
        videos = sorted(data.get("videos", []), key=lambda v: int(v["id"]))
        if len(videos) < 2:
            raise RuntimeError(f"cannot split {dataset}/{domain}: only {len(videos)} videos")
        # Deterministic approximately 20% tuning videos, spread across order.
        tune_ids = {int(video["id"]) for idx, video in enumerate(videos) if idx % 5 == 0}
        if not tune_ids:
            tune_ids = {int(videos[0]["id"])}
        if len(tune_ids) == len(videos):
            tune_ids.remove(int(videos[-1]["id"]))
        holdout_ids = {int(video["id"]) for video in videos} - tune_ids
        tune = split_data(data, tune_ids)
        holdout = split_data(data, holdout_ids)
        write_split(dataset, domain, "tune", tune, source_prepared, source_cache)
        write_split(dataset, domain, "holdout", holdout, source_prepared, source_cache)
        print(f"{dataset}/{domain}: tune_videos={len(tune['videos'])} tune_frames={len(tune['images'])} "
              f"holdout_videos={len(holdout['videos'])} holdout_frames={len(holdout['images'])}")
        del data, tune, holdout
        gc.collect()


if __name__ == "__main__":
    main()
