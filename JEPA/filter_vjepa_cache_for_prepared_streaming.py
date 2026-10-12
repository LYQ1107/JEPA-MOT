#!/usr/bin/env python3
"""Low-memory filtering of V-JEPA NPZ caches to mapped MOTSynth frames."""

import copy
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format


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
DOMAINS = tuple(os.environ.get(
    "JEPA_DOMAINS", "cloudy dark fog rain snow"
).split())
CACHE_SUFFIX = "_vjepa21_l384_w16_causalbox_predictor.npz"


def write_npz_streaming(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as archive:
        for name, array in arrays:
            with archive.open(f"{name}.npy", "w", force_zip64=True) as handle:
                npy_format.write_array(handle, array, allow_pickle=False)


def sidecar(source_sidecar, path, prepared, source, features, image_ids,
            geometry_null):
    info = copy.deepcopy(source_sidecar)
    info.update({
        "status": "complete",
        "prepared_dir": str(prepared),
        "annotation": str(prepared / "annotations/val.json"),
        "source_cache": str(source),
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
        info["geometry_null_for"] = str(path.with_name(
            path.name.replace("_geometry_null.npz.json", ".npz")
        ))
    Path(str(path) + ".json").write_text(json.dumps(info, indent=2))


def filter_domain(domain):
    name = f"motsynth_{domain}{CACHE_SUFFIX}"
    source = SOURCE_CACHE_ROOT / name
    target = TARGET_CACHE_ROOT / name
    null_target = TARGET_CACHE_ROOT / name.replace(".npz", "_geometry_null.npz")
    prepared = PREPARED_ROOT / "motsynth" / domain
    target_ids = [int(image["id"]) for image in json.loads(
        (prepared / "annotations/val.json").read_text()
    )["images"]]
    if target.is_file() and null_target.is_file() \
            and Path(str(target) + ".json").is_file() \
            and Path(str(null_target) + ".json").is_file():
        print(f"{domain}: existing filtered caches retained", flush=True)
        return

    with np.load(source, allow_pickle=False) as source_npz:
        source_offsets = np.asarray(source_npz["offsets"], dtype=np.int64)
        source_image_ids = np.asarray(source_npz["image_ids"], dtype=np.int64)
    source_index = {
        int(image_id): index
        for index, image_id in enumerate(source_image_ids.tolist())
    }
    missing = [image_id for image_id in target_ids if image_id not in source_index]
    if missing:
        raise RuntimeError(f"{domain}: missing image ids {missing[:10]}")
    ranges = []
    offsets = [0]
    for image_id in target_ids:
        index = source_index[image_id]
        start, end = int(source_offsets[index]), int(source_offsets[index + 1])
        ranges.append((start, end))
        offsets.append(offsets[-1] + end - start)
    offsets = np.asarray(offsets, dtype=np.int64)
    image_ids = np.asarray(target_ids, dtype=np.int64)

    source_sidecar = json.loads(Path(str(source) + ".json").read_text())
    with tempfile.TemporaryDirectory(prefix=f"vjepa_filter_{domain}_") as temp:
        temp = Path(temp)
        source_features_npy = temp / "source_features.npy"
        with (
            zipfile.ZipFile(source) as archive,
            archive.open("features.npy") as src,
            source_features_npy.open("wb") as dst,
        ):
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        source_features = np.load(source_features_npy, mmap_mode="r",
                                  allow_pickle=False)
        filtered_npy = temp / "filtered_features.npy"
        filtered = npy_format.open_memmap(
            filtered_npy, mode="w+", dtype=source_features.dtype,
            shape=(int(offsets[-1]), int(source_features.shape[1])),
        )
        for out_index, (start, end) in enumerate(ranges):
            out_start, out_end = int(offsets[out_index]), int(offsets[out_index + 1])
            if end > start:
                filtered[out_start:out_end] = source_features[start:end]
        filtered.flush()
        del filtered
        del source_features

        filtered = np.load(filtered_npy, mmap_mode="r", allow_pickle=False)
        write_npz_streaming(target, (
            ("features", filtered), ("offsets", offsets), ("image_ids", image_ids)
        ))
        sidecar(source_sidecar, target, prepared, source, filtered, image_ids, False)

        null_npy = temp / "null_features.npy"
        feature_dim = int(filtered.shape[1])
        null = npy_format.open_memmap(
            null_npy, mode="w+", dtype=np.float32,
            shape=(int(offsets[-1]), feature_dim),
        )
        null[:] = 0
        null.flush()
        del null
        null = np.load(null_npy, mmap_mode="r", allow_pickle=False)
        write_npz_streaming(null_target, (
            ("features", null), ("offsets", offsets), ("image_ids", image_ids)
        ))
        sidecar(source_sidecar, null_target, prepared, source, null, image_ids, True)

    print(f"{domain}: images={len(image_ids)} feature_rows={int(offsets[-1])}",
          flush=True)


def main():
    TARGET_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        filter_domain(domain)


if __name__ == "__main__":
    main()
