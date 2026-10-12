#!/usr/bin/env python3
"""Filter an existing per-image V-JEPA cache to a prepared annotation set.

The original cache is preserved.  This is used for MOTSynth because the
mapped detector protocol keeps only detector-covered frames, which are a
strict subset of the older dense annotation preparation.
"""

import copy
import json
import os
from pathlib import Path

import numpy as np


JEPA_ROOT = Path("/data1/liuyeqiang/JEPA_MOT_resume_20261012/JEPA")
REPO = Path("/data1/liuyeqiang/JEPA_MOT_resume_20261012/GeneralTrack-main")
SOURCE_CACHE_ROOT = Path(os.environ.get(
    "JEPA_SOURCE_CACHE_ROOT", str(JEPA_ROOT / "outputs/official_vjepa21_cache")
))
TARGET_CACHE_ROOT = Path(os.environ.get(
    "JEPA_TARGET_CACHE_ROOT",
    str(JEPA_ROOT / "outputs/official_vjepa21_cache_mapped_source_detector_thr02"),
))
PREPARED_ROOT = Path(os.environ.get(
    "JEPA_MAPPED_PREPARED_ROOT",
    str(REPO / "datasets/generaltrack_prepared_mapped/source_detector_thr02"),
))
DOMAINS = ("cloudy", "dark", "fog", "rain", "snow")
CACHE_SUFFIX = "_vjepa21_l384_w16_causalbox_predictor.npz"


def load_ids(annotation_path):
    data = json.loads(annotation_path.read_text())
    return [int(image["id"]) for image in data["images"]]


def write_sidecar(source_sidecar, target_sidecar, prepared, features, image_ids,
                  source_cache, geometry_null=False):
    info = copy.deepcopy(source_sidecar)
    info.update({
        "status": "complete",
        "prepared_dir": str(prepared),
        "annotation": str(prepared / "annotations/val.json"),
        "source_cache": str(source_cache),
        "feature_shape": [int(x) for x in features.shape],
        "num_images": int(len(image_ids)),
        "image_ids_filtered": True,
        "mapped_detector_protocol": "source_detector_thr02",
        "feature_source": (
            "geometry_null_zero_features" if geometry_null
            else "official_vjepa21_encoder_causal_box_crop_predictor_filtered"
        ),
        "feature_rows_computed": 0 if geometry_null else int(features.shape[0]),
        "feature_rows_zero_filled": int(features.shape[0]) if geometry_null else 0,
    })
    if geometry_null:
        info["geometry_null_for"] = str(target_sidecar.with_name(
            target_sidecar.name.replace("_geometry_null.npz.json", ".npz")
        ))
    target_sidecar.write_text(json.dumps(info, indent=2))


def main():
    TARGET_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        name = f"motsynth_{domain}{CACHE_SUFFIX}"
        source = SOURCE_CACHE_ROOT / name
        target = TARGET_CACHE_ROOT / name
        null_target = TARGET_CACHE_ROOT / name.replace(
            ".npz", "_geometry_null.npz"
        )
        prepared = PREPARED_ROOT / "motsynth" / domain
        target_ids = load_ids(prepared / "annotations/val.json")

        with np.load(source, allow_pickle=False) as data:
            source_features = np.asarray(data["features"], dtype=np.float32)
            source_offsets = np.asarray(data["offsets"], dtype=np.int64)
            source_image_ids = np.asarray(data["image_ids"], dtype=np.int64)
            source_index = {
                int(image_id): index
                for index, image_id in enumerate(source_image_ids.tolist())
            }
            missing = [image_id for image_id in target_ids
                       if image_id not in source_index]
            if missing:
                raise RuntimeError(f"{domain}: missing cached image ids: {missing[:10]}")
            chunks = []
            offsets = [0]
            for image_id in target_ids:
                index = source_index[image_id]
                start = int(source_offsets[index])
                end = int(source_offsets[index + 1])
                chunks.append(source_features[start:end])
                offsets.append(offsets[-1] + end - start)
            features = (
                np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
                if chunks else np.zeros((0, source_features.shape[1]), dtype=np.float32)
            )

        image_ids = np.asarray(target_ids, dtype=np.int64)
        offsets = np.asarray(offsets, dtype=np.int64)
        np.savez_compressed(target, features=features, offsets=offsets,
                            image_ids=image_ids)
        source_sidecar = json.loads(Path(str(source) + ".json").read_text())
        write_sidecar(source_sidecar, Path(str(target) + ".json"), prepared,
                      features, image_ids, source, geometry_null=False)

        null_features = np.zeros_like(features, dtype=np.float32)
        np.savez_compressed(null_target, features=null_features, offsets=offsets,
                            image_ids=image_ids)
        write_sidecar(source_sidecar, Path(str(null_target) + ".json"), prepared,
                      null_features, image_ids, source, geometry_null=True)
        print(f"{domain}: images={len(image_ids)} feature_rows={len(features)} "
              f"target={target}", flush=True)


if __name__ == "__main__":
    main()
