#!/usr/bin/env python3
"""Build a source-only GT-box cache with the official V-JEPA 2.1 encoder.

The cache is used only to train a small source association scorer.  It reads
the sunny source annotations and track IDs by design, but never opens a
cloudy/night/rainy annotation.  The feature readout is otherwise identical to
the target cache: dense letterboxed RGB frames, a causal 16-frame tubelet
window, and frozen encoder tokens pooled over the current box.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from archive_paths import resolve_archive_path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_generaltrack_vjepa21_cache import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PATCH,
    DEFAULT_RESOLUTION,
    DEFAULT_WINDOW,
    IMAGENET_MEAN,
    _letterbox_rgb,
    _normalize_frame,
    _extract_causal_box_crop_predictor_features,
    _pool_temporal_box_features,
    _temporal_maps,
    sha256_file,
)
from vjepa21_encoder import (  # noqa: E402
    encoder_metadata,
    forward_tokens,
    load_official_vjepa21_encoder,
    load_official_vjepa21_predictor,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build sunny-source GT V-JEPA 2.1 cache")
    ap.add_argument("--annotation", default="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/ua_detrac/annotations/box_track_sunny_all_cocofmt.json")
    ap.add_argument("--image-root", default="/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/ua_detrac/images/sunny")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-domain", default="sunny")
    ap.add_argument("--dataset", default="ua_detrac")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    ap.add_argument("--patch-size", type=int, default=DEFAULT_PATCH)
    ap.add_argument("--roi-pool", choices=["box_mean", "center", "center3"], default="box_mean")
    ap.add_argument("--roi-mode", choices=["dense", "causal_box_crop_predictor"], default="dense")
    ap.add_argument("--roi-context", type=float, default=1.5)
    ap.add_argument("--crop-batch-size", type=int, default=8)
    ap.add_argument("--predictor-pool", choices=["mean", "center", "center3"], default="mean")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--video-start", type=int, default=0,
                    help="Zero-based start index in sorted source videos.")
    ap.add_argument("--video-end", type=int, default=None,
                    help="Exclusive zero-based end index in sorted source videos.")
    ap.add_argument("--max-frames-per-video", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--no-autocast", action="store_true")
    return ap.parse_args()


def _load_source(annotation_path: Path, max_videos: int | None,
                  video_start: int = 0, video_end: int | None = None):
    data = json.loads(annotation_path.read_text())
    videos = sorted(data.get("videos", []), key=lambda x: int(x["id"]))
    if max_videos is not None:
        videos = videos[:int(max_videos)]
    start = max(0, int(video_start))
    end = len(videos) if video_end is None else max(start, int(video_end))
    videos = videos[start:end]
    selected = {int(v["id"]) for v in videos}
    images = [x for x in data.get("images", []) if int(x["video_id"]) in selected]
    images.sort(key=lambda x: (int(x["video_id"]), int(x.get("frame_id", 0)), int(x["id"])))
    anns = defaultdict(list)
    for ann in data.get("annotations", []):
        if int(ann.get("video_id", -1)) not in selected or ann.get("iscrowd", 0):
            continue
        box = np.asarray(ann.get("bbox", []), dtype=np.float32)
        if box.shape != (4,) or box[2] <= 1 or box[3] <= 1:
            continue
        track_id = int(ann.get("track_id", ann.get("instance_id", ann.get("mot_instance_id", ann["id"]))))
        xyxy = np.asarray([box[0], box[1], box[0] + box[2], box[1] + box[3]], dtype=np.float32)
        anns[int(ann["image_id"])].append((track_id, xyxy))
    for key in anns:
        anns[key].sort(key=lambda x: x[0])
    if not images:
        raise RuntimeError("no source images after video filtering")
    return videos, images, anns


def main() -> int:
    args = parse_args()
    annotation = Path(args.annotation).resolve()
    image_root = Path(args.image_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.out).resolve()
    if not annotation.is_file() or not image_root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("source annotation, image root, or checkpoint is missing")
    if args.window < 1 or (args.window > 1 and args.window % 2):
        raise ValueError("--window must be 1 or an even number")
    if args.resolution % args.patch_size:
        raise ValueError("resolution must be divisible by patch-size")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V-JEPA 2.1 extraction")

    videos, images, anns = _load_source(
        annotation, args.max_videos, args.video_start, args.video_end)
    encoder = load_official_vjepa21_encoder(
        checkpoint,
        model_name=args.model_name,
        img_size=args.resolution,
        num_frames=args.window,
        patch_size=args.patch_size,
        tubelet_size=2,
        device=device,
    )
    predictor = None
    if args.roi_mode == "causal_box_crop_predictor":
        predictor = load_official_vjepa21_predictor(
            checkpoint,
            embed_dim=int(encoder.embed_dim),
            img_size=args.resolution,
            num_frames=args.window,
            patch_size=args.patch_size,
            tubelet_size=2,
            device=device,
        )
    model_meta = encoder_metadata(
        encoder,
        model_name=args.model_name,
        checkpoint=checkpoint,
        num_frames=args.window,
        img_size=args.resolution,
        patch_size=args.patch_size,
        tubelet_size=2,
    )

    all_features = []
    all_image_ids = []
    all_video_ids = []
    all_frame_ids = []
    all_track_ids = []
    all_boxes = []
    offsets = [0]
    frame_buffer: deque[torch.Tensor] = deque(maxlen=args.window)
    rgb_buffer: deque[np.ndarray] = deque(maxlen=args.window)
    current_video = None
    frame_count_in_video = 0
    start_time = time.perf_counter()
    forwards = 0
    for index, image in enumerate(images):
        video_id = int(image["video_id"])
        frame_id = int(image.get("frame_id", 0))
        if video_id != current_video:
            current_video = video_id
            frame_buffer.clear()
            frame_count_in_video = 0
        frame_count_in_video += 1
        if args.max_frames_per_video is not None and frame_count_in_video > int(args.max_frames_per_video):
            continue
        image_path = resolve_archive_path(str(image_root / str(image["file_name"])))
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        if args.roi_mode == "causal_box_crop_predictor":
            rgb_buffer.append(rgb.copy())
            while len(rgb_buffer) < args.window:
                rgb_buffer.appendleft(rgb_buffer[0].copy())
        canvas, scale, pad_left, pad_top = _letterbox_rgb(rgb, args.resolution)
        frame = _normalize_frame(canvas)
        if not frame_buffer:
            frame_buffer.append(frame)
        while len(frame_buffer) < args.window:
            frame_buffer.appendleft(frame_buffer[0].clone())
        else:
            frame_buffer.append(frame)
        rows = anns.get(int(image["id"]), [])
        detections = np.asarray(
            [[*box.tolist(), 1.0, 1.0] for _, box in rows], dtype=np.float32
        ).reshape(-1, 6) if rows else np.zeros((0, 6), dtype=np.float32)
        if args.roi_mode == "causal_box_crop_predictor":
            pooled = _extract_causal_box_crop_predictor_features(
                encoder,
                predictor,
                list(rgb_buffer),
                detections,
                context=args.roi_context,
                resolution=args.resolution,
                device=device,
                batch_size=args.crop_batch_size,
                window=args.window,
                patch_size=args.patch_size,
                predictor_pool=args.predictor_pool,
            ) if len(rows) else np.zeros((0, 1664), dtype=np.float32)
        else:
            clip = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).to(device, non_blocking=True)
            with torch.inference_mode():
                tokens = forward_tokens(
                    encoder,
                    clip,
                    autocast_dtype=None if args.no_autocast else torch.bfloat16,
                )
                maps = _temporal_maps(tokens, args.window, args.resolution, args.patch_size)
            pooled = _pool_temporal_box_features(
                maps,
                detections,
                temporal_pool="last",
                image_width=width,
                image_height=height,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
                patch_size=args.patch_size,
                roi_pool=args.roi_pool,
            ) if len(rows) else np.zeros((0, int(encoder.embed_dim)), dtype=np.float32)
        if len(pooled) != len(rows):
            raise RuntimeError("source feature/GT row mismatch at image {}".format(image["id"]))
        if len(rows):
            all_features.append(pooled.astype(np.float32, copy=False))
            all_track_ids.extend(int(t) for t, _ in rows)
            all_boxes.extend(box for _, box in rows)
        all_image_ids.extend([int(image["id"])] * len(rows))
        all_video_ids.extend([video_id] * len(rows))
        all_frame_ids.extend([frame_id] * len(rows))
        offsets.append(offsets[-1] + len(rows))
        forwards += 1
        if args.roi_mode != "causal_box_crop_predictor":
            del clip, tokens, maps
        if (index + 1) % max(1, int(args.log_every)) == 0:
            print("[source-vjepa21] {}/{} rows={} elapsed_sec={:.1f}".format(
                index + 1, len(images), offsets[-1], time.perf_counter() - start_time), flush=True)

    features = (np.concatenate(all_features, axis=0).astype(np.float32, copy=False)
                if all_features else np.zeros((0, int(encoder.embed_dim)), dtype=np.float32))
    if len(features) != len(all_track_ids):
        raise RuntimeError("feature rows do not match source metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        features=features,
        offsets=np.asarray(offsets, dtype=np.int64),
        image_ids=np.asarray(all_image_ids, dtype=np.int64),
        video_ids=np.asarray(all_video_ids, dtype=np.int64),
        frame_ids=np.asarray(all_frame_ids, dtype=np.int64),
        track_ids=np.asarray(all_track_ids, dtype=np.int64),
        boxes=np.asarray(all_boxes, dtype=np.float32).reshape(-1, 4),
    )
    elapsed = time.perf_counter() - start_time
    meta = {
        "schema": "uadetrac-sunny-source-gt-official-vjepa21-cache-v1",
        "status": "complete",
        "annotation": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "image_root": str(image_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model": model_meta,
        "feature_shape": [int(x) for x in features.shape],
        "feature_dtype": "float32",
        "num_source_videos": len(videos),
        "num_source_frames": int(len(images)),
        "video_start": int(args.video_start),
        "video_end": None if args.video_end is None else int(args.video_end),
        "num_gt_rows": int(len(features)),
        "num_forwards": int(forwards),
        "window": int(args.window),
        "roi_pool": args.roi_pool,
        "roi_mode": args.roi_mode,
        "roi_context": float(args.roi_context),
        "predictor_pool": args.predictor_pool,
        "crop_batch_size": int(args.crop_batch_size),
        "causal": True,
        "future_frames_read": False,
        "gt_read": True,
        "source_domain": args.source_domain,
        "dataset": args.dataset,
        "target_domains_read": [],
        "feature_source": ("official_vjepa21_encoder_causal_box_crop_predictor_final"
                           if args.roi_mode == "causal_box_crop_predictor"
                           else "official_vjepa21_encoder_final_dense_tokens"),
        "predictor_used": args.roi_mode == "causal_box_crop_predictor",
        "rgb_input": True,
        "letterbox_padding": "imagenet_mean_pixel",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "elapsed_seconds": elapsed,
        "command": " ".join([os.environ.get("_", "python"), *sys.argv]),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
