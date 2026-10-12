#!/usr/bin/env python3
"""Extract source-only UA-DETRAC sunny GT crop features for adapter training.

This cache is intentionally separate from the target detector cache.  It may
read the sunny source annotations and track IDs for supervision, but it never
opens cloudy/night/rainy annotations.  The feature extractor remains the
official frozen GOT-JEPA checkpoint; only a later small adapter is trainable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PYTRACKING_ROOT = HERE.parent / "pytracking"
COTRACKER_ROOT = PYTRACKING_ROOT / "ltr" / "cotracker2" / "co_tracker"
for path in (COTRACKER_ROOT, PYTRACKING_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mot_ocsort_jepa_official_v2 as official_api  # noqa: E402


FEATURE_DIM = 256


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build sunny-source GT GOT-JEPA cache")
    ap.add_argument("--annotation", default="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/ua_detrac/annotations/box_track_sunny_all_cocofmt.json")
    ap.add_argument("--image-root", default="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/ua_detrac/images/sunny")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Per-video frame prefix; 0 means all source frames")
    ap.add_argument("--pool", choices=["mean", "center3", "center5", "center7"], default="center3")
    ap.add_argument("--stage", choices=["predictor", "head"], default="predictor",
                    help="Pool the serialized JEPA predictor output (default) or the preceding ToMP head map")
    ap.add_argument("--context-factor", type=float, default=5.0,
                    help="Square crop side as factor * sqrt(box area); official default is 5")
    return ap.parse_args()


def load_records(annotation: Path, image_root: Path, max_videos: int, max_frames: int):
    data = json.loads(annotation.read_text())
    videos = sorted(data.get("videos", []), key=lambda x: int(x["id"]))
    if max_videos > 0:
        keep = {int(v["id"]) for v in videos[:max_videos]}
    else:
        keep = {int(v["id"]) for v in videos}
    anns = {}
    for ann in data.get("annotations", []):
        vid = int(ann.get("video_id", -1))
        if vid not in keep:
            continue
        anns.setdefault(int(ann["image_id"]), []).append(ann)
    images = []
    for image in data.get("images", []):
        vid = int(image["video_id"])
        frame = int(image.get("frame_id", 0))
        if vid not in keep or (max_frames > 0 and frame >= max_frames):
            continue
        rows = anns.get(int(image["id"]), [])
        for ann in rows:
            bbox = np.asarray(ann["bbox"], dtype=np.float32)
            if bbox.shape != (4,) or bbox[2] <= 1 or bbox[3] <= 1:
                continue
            images.append((
                int(image["id"]), vid, frame, str(image["file_name"]),
                int(ann.get("track_id", ann.get("instance_id", ann["id"]))),
                np.asarray([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], dtype=np.float32),
            ))
    images.sort(key=lambda x: (x[1], x[2], x[0], x[4]))
    if not images:
        raise RuntimeError("no source GT rows after filtering")
    return images


def pool_predictor(y, pool: str):
    if pool == "mean":
        vec = y.mean(dim=(-1, -2))
    else:
        width = int(pool.replace("center", ""))
        start = (27 - width) // 2
        vec = y[:, :, start:start + width, start:start + width].mean(dim=(-1, -2))
    import torch
    return torch.nn.functional.normalize(vec.float(), dim=-1)


def main() -> int:
    args = parse_args()
    annotation = Path(args.annotation).resolve()
    image_root = Path(args.image_root).resolve()
    out = Path(args.out).resolve()
    if not annotation.exists():
        raise FileNotFoundError(annotation)
    if not image_root.exists():
        raise FileNotFoundError(image_root)
    if int(args.batch) <= 0:
        raise ValueError("--batch must be positive")
    records = load_records(annotation, image_root, int(args.max_videos), int(args.max_frames))
    import torch
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("[source-cache] rows={} videos={} device={} batch={} pool={}".format(
        len(records), len({r[1] for r in records}), args.device, args.batch, args.pool), flush=True)
    encoder = official_api.OfficialLatent(checkpoint, str(args.device), int(args.batch),
                                           context_factor=float(args.context_factor))
    crops = []
    features = np.empty((len(records), FEATURE_DIM), dtype=np.float16)
    row = 0
    batches = 0
    for image_id, _, _, file_name, _, box in records:
        img = cv2.imread(str(image_root / file_name))
        if img is None:
            raise FileNotFoundError(image_root / file_name)
        crops.append(encoder._tensor_crop(img, box))
        if len(crops) < int(args.batch) and row + len(crops) < len(records):
            continue
        xb = torch.stack(crops).to(encoder.device, non_blocking=True)
        with torch.inference_mode():
            z = encoder.net.extract_dino_features_spatial_intermediate_layers(xb, None)
            z = encoder.net.head.extract_head_feat(z)
            y = encoder.net.JEPA_predictor_cls(z)
            if tuple(z.shape[1:]) != (256, 27, 27) or tuple(y.shape[1:]) != (256, 27, 27):
                raise RuntimeError("unexpected feature shape z={} y={}".format(tuple(z.shape), tuple(y.shape)))
            vec = pool_predictor(y if args.stage == "predictor" else z, args.pool)
            if not bool(torch.isfinite(vec).all()):
                raise FloatingPointError("non-finite source feature")
        n = len(crops)
        features[row:row + n] = vec.detach().cpu().numpy().astype(np.float16)
        row += n
        batches += 1
        crops = []
        if row and row % 10000 < n:
            print("[source-cache] processed={} / {} elapsed_sec={:.1f}".format(
                row, len(records), time.perf_counter() - t0), flush=True)
    if row != len(records):
        raise RuntimeError("source feature rows {} != {}".format(row, len(records)))
    arrays = {
        "features": features,
        "image_ids": np.asarray([r[0] for r in records], dtype=np.int64),
        "video_ids": np.asarray([r[1] for r in records], dtype=np.int64),
        "frame_ids": np.asarray([r[2] for r in records], dtype=np.int64),
        "track_ids": np.asarray([r[4] for r in records], dtype=np.int64),
        "boxes": np.asarray([r[5] for r in records], dtype=np.float32),
    }
    meta = {
        "schema": "uadetrac-sunny-source-gt-got-jepa-cache-v1",
        "status": "complete", "annotation": str(annotation),
        "annotation_sha256": sha256(annotation), "image_root": str(image_root),
        "checkpoint": str(checkpoint), "checkpoint_sha256": encoder.checkpoint_sha256,
        "feature_source": "official GOT_JEPA_378: DINOv2/ToMP head -> {} -> configured spatial pooling -> L2_normalize".format(
            "JEPA_predictor_cls" if args.stage == "predictor" else "ToMP head"),
        "pool": str(args.pool), "feature_dim": FEATURE_DIM, "feature_dtype": "float16",
        "stage": str(args.stage),
        "crop_size": 378, "context_factor": float(args.context_factor),
        "box_format": "xyxy source GT boxes",
        "source_domain": "sunny", "supervision": "source GT track_id only",
        "gt_read": True, "target_domains_read": [], "num_rows": len(records),
        "num_videos": len({r[1] for r in records}), "num_batches": batches,
        "max_videos": int(args.max_videos), "max_frames": int(args.max_frames),
        "device": str(encoder.device),
        "gpu_name": torch.cuda.get_device_name(encoder.device) if encoder.device.type == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - t0, "command": " ".join(sys.argv), "cwd": os.getcwd(),
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, out)
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
