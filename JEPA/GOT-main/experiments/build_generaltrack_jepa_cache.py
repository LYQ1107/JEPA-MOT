#!/usr/bin/env python3
"""Build causal-MOT-compatible frozen GOT-JEPA detection features.

The prepared GeneralTrack annotation contains the fixed detector rows used by
the baseline.  This script reads only those rows and the corresponding images,
then writes one 256-D vector per detection row in exactly the same image and
detection order.  It never opens GT annotations.  The vector is the spatially
pooled output of the official ``JEPA_predictor_cls`` applied to the frozen
DINOv2/ToMP head feature map.

The cache is deliberately separate from GeneralTrack's Python environment:
GeneralTrack can consume the small float16 vectors without loading the 1.6 GB
GOT-JEPA archive or importing its ToMP dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
PYTRACKING_ROOT = HERE.parent / "pytracking"
COTRACKER_ROOT = PYTRACKING_ROOT / "ltr" / "cotracker2" / "co_tracker"
for p in (COTRACKER_ROOT, PYTRACKING_ROOT, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mot_ocsort_jepa_official_v2 as official_api  # noqa: E402


SCHEMA = "generaltrack-got-jepa-detection-cache-v1"
FEATURE_DIM = 256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_json(path: Path) -> str:
    # Preserve the exact prepared annotation bytes as the provenance digest.
    return sha256(path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build frozen official GOT-JEPA cache for prepared GeneralTrack input")
    ap.add_argument("--prepared-dir", required=True,
                    help="GeneralTrack prepared domain directory containing annotations/val.json and metadata.json")
    ap.add_argument("--out", required=True, help="Output .npz cache path")
    ap.add_argument("--checkpoint", default="/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-videos", type=int, default=0,
                    help="Optional first-N videos for an end-to-end smoke; 0 means all videos")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Optional per-video frame prefix for smoke tests; 0 means all frames")
    ap.add_argument("--amp", action="store_true",
                    help="Use CUDA autocast for extraction; off by default for reproducibility")
    ap.add_argument("--pool", choices=["mean", "center3", "center5", "center7"], default="mean",
                    help="Spatial pooling of the frozen JEPA predictor map; center-K keeps the crop-center patch")
    ap.add_argument("--stage", choices=["predictor", "head"], default="predictor",
                    help="Pool the serialized JEPA predictor output (default) or the preceding ToMP head map")
    ap.add_argument("--context-factor", type=float, default=5.0,
                    help="Square crop side as factor * sqrt(box area); official default is 5")
    ap.add_argument("--maps-out", default=None,
                    help="Optional separate .npz path for the frozen 256x27x27 ToMP head maps")
    return ap.parse_args()


def load_prepared(prepared_dir: Path, max_frames: int, max_videos: int) -> Tuple[Path, List[dict]]:
    ann_path = prepared_dir / "annotations" / "val.json"
    meta_path = prepared_dir / "metadata.json"
    if not ann_path.exists():
        raise FileNotFoundError(ann_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    data = json.loads(ann_path.read_text())
    metadata = json.loads(meta_path.read_text())
    image_root = Path(metadata.get("image_root", ""))
    if not image_root.exists():
        raise FileNotFoundError("prepared metadata image_root does not exist: {}".format(image_root))
    images = sorted(
        data.get("images", []),
        key=lambda x: (int(x["video_id"]), int(x["frame_id"]), int(x["id"])),
    )
    if max_videos > 0:
        keep_videos = {int(v["id"]) for v in data.get("videos", [])[:max_videos]}
        if not keep_videos:
            keep_videos = sorted({int(x["video_id"]) for x in images})[:max_videos]
            keep_videos = set(keep_videos)
        images = [x for x in images if int(x["video_id"]) in keep_videos]
    if max_frames > 0:
        images = [x for x in images if int(x["frame_id"]) <= max_frames]
    return image_root, images


def count_detections(images: Sequence[dict]) -> int:
    return int(sum(len(x.get("detection", [])) for x in images))


def extract(args: argparse.Namespace) -> Dict[str, object]:
    prepared_dir = Path(args.prepared_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path = prepared_dir / "annotations" / "val.json"
    meta_path = prepared_dir / "metadata.json"
    image_root, images = load_prepared(prepared_dir, int(args.max_frames), int(args.max_videos))
    total_rows = count_detections(images)
    if total_rows <= 0:
        raise RuntimeError("prepared input contains no detector rows: {}".format(prepared_dir))
    if int(args.batch) <= 0:
        raise ValueError("--batch must be positive")

    import torch

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    t0 = time.perf_counter()
    print("[cache] prepared={} images={} detections={} device={} batch={}".format(
        prepared_dir, len(images), total_rows, args.device, args.batch), flush=True)
    encoder = official_api.OfficialLatent(checkpoint, str(args.device), int(args.batch),
                                           context_factor=float(args.context_factor))
    device = encoder.device

    features = np.empty((total_rows, FEATURE_DIM), dtype=np.float16)
    map_features = (np.empty((total_rows, 256, 27, 27), dtype=np.float16)
                    if args.maps_out else None)
    image_ids = np.empty((len(images),), dtype=np.int64)
    video_ids = np.empty((len(images),), dtype=np.int64)
    frame_ids = np.empty((len(images),), dtype=np.int64)
    offsets = np.zeros((len(images) + 1,), dtype=np.int64)

    crops: List[torch.Tensor] = []
    crop_offsets: List[int] = []
    row_cursor = 0
    next_report = 10000
    batches = 0

    def flush() -> None:
        nonlocal row_cursor, batches, crops, crop_offsets
        if not crops:
            return
        xb = torch.stack(crops, dim=0).to(device, non_blocking=True)
        with torch.inference_mode():
            if bool(args.amp) and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    z = encoder.net.extract_dino_features_spatial_intermediate_layers(xb, None)
                    z = encoder.net.head.extract_head_feat(z)
                    y = encoder.net.JEPA_predictor_cls(z)
            else:
                z = encoder.net.extract_dino_features_spatial_intermediate_layers(xb, None)
                z = encoder.net.head.extract_head_feat(z)
                y = encoder.net.JEPA_predictor_cls(z)
            if tuple(z.shape[1:]) != (256, 27, 27):
                raise RuntimeError("unexpected projected feature shape {}".format(tuple(z.shape)))
            if y.ndim != 4 or tuple(y.shape[1:]) != (256, 27, 27):
                raise RuntimeError("unexpected JEPA predictor shape {}".format(tuple(y.shape)))
            pool_input = y if args.stage == "predictor" else z
            if args.pool == "mean":
                vec = pool_input.mean(dim=(-1, -2))
            else:
                width = int(args.pool.replace("center", ""))
                start = (27 - width) // 2
                vec = pool_input[:, :, start:start + width, start:start + width].mean(dim=(-1, -2))
            vec = torch.nn.functional.normalize(vec.float(), dim=-1)
            if not bool(torch.isfinite(vec).all()):
                raise FloatingPointError("non-finite official JEPA vector")
        if map_features is not None:
            # The official semantic filter consumes the preceding ToMP head
            # map, not the pooled JEPA vector.  Keep this optional so the
            # ordinary compact vector cache remains unchanged.
            map_features[row_cursor:row_cursor + len(z)] = z.detach().cpu().numpy().astype(np.float16)
        vec_np = vec.detach().cpu().numpy().astype(np.float16)
        features[row_cursor:row_cursor + len(vec_np)] = vec_np
        row_cursor += len(vec_np)
        batches += 1
        crops = []
        crop_offsets = []

    for image_index, image_info in enumerate(images):
        image_ids[image_index] = int(image_info["id"])
        video_ids[image_index] = int(image_info["video_id"])
        frame_ids[image_index] = int(image_info["frame_id"])
        detections = image_info.get("detection", [])
        image_path = image_root / str(image_info["file_name"])
        img = cv2.imread(str(image_path)) if detections else None
        if detections and img is None:
            raise FileNotFoundError(image_path)
        for det_index, det in enumerate(detections):
            box = np.asarray(det[:4], dtype=np.float32)
            crops.append(encoder._tensor_crop(img, box))
            crop_offsets.append(row_cursor + len(crops) - 1)
            if len(crops) >= int(args.batch):
                flush()
        offsets[image_index + 1] = row_cursor + len(crops)
        # Keep the per-image offset exact even when the final batch is held.
        if image_index + 1 == len(images) or len(crops) >= int(args.batch):
            pass
        if row_cursor >= next_report:
            print("[cache] processed_rows={} / {} elapsed_sec={:.1f}".format(
                row_cursor, total_rows, time.perf_counter() - t0), flush=True)
            next_report += 10000

    flush()
    if row_cursor != total_rows:
        raise RuntimeError("feature row count {} != expected {}".format(row_cursor, total_rows))

    # The previous loop can hold a final partial batch across image boundaries.
    # Reconstruct offsets from the deterministic detection counts, independent
    # of the batching cursor.
    np.cumsum(np.asarray([len(x.get("detection", [])) for x in images], dtype=np.int64),
              out=offsets[1:])

    metadata = {
        "schema": SCHEMA,
        "status": "complete",
        "prepared_dir": str(prepared_dir),
        "annotation_path": str(ann_path),
        "annotation_sha256": hash_json(ann_path),
        "metadata_path": str(meta_path),
        "image_root": str(image_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": encoder.load_meta.get("checkpoint_sha256"),
        "feature_source": "official GOT_JEPA_378: DINOv2/ToMP head -> {} -> configured spatial pooling -> L2_normalize".format(
            "JEPA_predictor_cls" if args.stage == "predictor" else "ToMP head"),
        "pool": str(args.pool),
        "stage": str(args.stage),
        "feature_dim": FEATURE_DIM,
        "feature_dtype": "float16",
        "feature_map_shape": [256, 27, 27],
        "crop_size": 378,
        "context_factor": float(args.context_factor),
        "color_input": "BGR image read by cv2, converted to RGB inside OfficialLatent._tensor_crop",
        "box_format": "prepared detector x1,y1,x2,y2 in original image pixels",
        "image_order": "video_id, frame_id, image_id",
        "detection_order": "annotation image detection list order",
        "gt_read": False,
        "target_candidate_in_predictor": False,
        "max_frames": int(args.max_frames),
        "max_videos": int(args.max_videos),
        "num_images": len(images),
        "num_detections": total_rows,
        "num_batches": batches,
        "device": str(device),
        "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "amp": bool(args.amp),
        "elapsed_seconds": time.perf_counter() - t0,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }
    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    # np.savez appends .npz when given a string without that suffix.  Passing
    # an open file keeps the temporary name exact for the atomic replace.
    with temp_path.open("wb") as f:
        np.savez_compressed(f, features=features, image_ids=image_ids,
                            video_ids=video_ids, frame_ids=frame_ids, offsets=offsets)
    os.replace(temp_path, out_path)
    meta_out = out_path.with_suffix(out_path.suffix + ".json")
    meta_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    if map_features is not None:
        map_out = Path(args.maps_out).resolve()
        map_out.parent.mkdir(parents=True, exist_ok=True)
        map_temp = map_out.with_suffix(map_out.suffix + ".tmp")
        if map_temp.exists():
            map_temp.unlink()
        with map_temp.open("wb") as f:
            np.savez(f, features=map_features, image_ids=image_ids,
                     video_ids=video_ids, frame_ids=frame_ids, offsets=offsets)
        os.replace(map_temp, map_out)
        map_meta = dict(metadata)
        map_meta["schema"] = "generaltrack-got-jepa-map-cache-v1"
        map_meta["feature_source"] = "official GOT_JEPA_378: frozen ToMP head feature maps"
        map_meta["feature_dtype"] = "float16"
        map_meta["stored_tensor"] = "features[N,256,27,27]"
        map_meta["map_cache"] = str(map_out)
        map_meta_path = map_out.with_suffix(map_out.suffix + ".json")
        map_meta_path.write_text(json.dumps(map_meta, indent=2, ensure_ascii=False) + "\n")
        print("[cache] wrote {} and {}".format(map_out, map_meta_path), flush=True)
    print("[cache] wrote {} and {}".format(out_path, meta_out), flush=True)
    return metadata


def main() -> int:
    args = parse_args()
    try:
        payload = extract(args)
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print("[cache] FAILED {}: {}".format(type(exc).__name__, exc), file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
