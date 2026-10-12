"""Build a causal full-frame V-JEPA 2.1 predictor feature cache.

The target of the official predictor is the final tubelet of a clip.  All
patches in that tubelet are masked, so the representation for frame ``t`` is
forecast from the preceding frames only.  The resulting spatial prediction
map is pooled over the fixed detector boxes and stored in the same
offset/image-id format consumed by GeneralTrack.

This is deliberately separate from the detector-box crop cache: it tests
whether the predictor's scene-level temporal context is more useful for
association than a crop whose centre is fixed at the current detector box.
The cache builder reads candidate detections and pixels only; GT annotations
and track IDs are never opened.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from build_generaltrack_vjepa21_cache import (
    IMAGENET_MEAN,
    _letterbox_rgb,
    _load_detections,
    _normalize_frame,
    _pool_box_features,
    read_json,
    sha256_file,
)
from vjepa21_encoder import (
    encoder_metadata,
    forward_predictor_tokens,
    load_official_vjepa21_encoder,
    load_official_vjepa21_predictor,
)


DEFAULT_MODEL = "vjepa2_1_vit_large_384"


def _predictor_masks(window: int, resolution: int, patch_size: int,
                     device: torch.device):
    if window < 2 or window % 2:
        raise ValueError("--window must be an even video length for the predictor")
    spatial = resolution // patch_size
    temporal = window // 2
    final_start = (temporal - 1) * spatial * spatial
    masks_x = torch.arange(final_start, dtype=torch.long, device=device).view(1, -1)
    masks_y = torch.arange(final_start, temporal * spatial * spatial,
                           dtype=torch.long, device=device).view(1, -1)
    return masks_x, masks_y


def _extract_dense_predictor_features(encoder, predictor, frames: list[np.ndarray],
                                      detections: np.ndarray, *, resolution: int,
                                      patch_size: int, device: torch.device,
                                      image_width: int, image_height: int,
                                      batch_size: int = 1,
                                      roi_pool: str = "box_mean") -> np.ndarray:
    """Forecast the final tubelet and pool its predicted spatial map per box."""
    output_dim = 1664
    if len(detections) == 0:
        return np.zeros((0, output_dim), dtype=np.float32)
    if len(frames) < 2 or len(frames) % 2:
        raise ValueError(f"Expected an even causal frame buffer, got {len(frames)}")

    # Full-frame letterboxing is shared by the encoder and ROI coordinate map.
    canvases = []
    scale = pad_left = pad_top = None
    for rgb in frames:
        canvas, scale_i, pad_left_i, pad_top_i = _letterbox_rgb(rgb, resolution)
        canvases.append(_normalize_frame(canvas))
        if scale is None:
            scale, pad_left, pad_top = scale_i, pad_left_i, pad_top_i

    video = torch.stack(canvases, dim=1).unsqueeze(0).to(device, non_blocking=True)
    masks_x, masks_y = _predictor_masks(len(frames), resolution, patch_size, device)
    with torch.inference_mode():
        predicted = forward_predictor_tokens(
            encoder, predictor, video, masks_x, masks_y,
            autocast_dtype=torch.bfloat16,
        )
    expected = (resolution // patch_size) ** 2
    if predicted.ndim != 3 or predicted.shape[0] != 1 or predicted.shape[1] != expected:
        raise RuntimeError(
            f"Expected dense final predictor tokens [1,{expected},1664], got {tuple(predicted.shape)}"
        )
    spatial = resolution // patch_size
    feature_map = predicted[0].reshape(spatial, spatial, output_dim)
    features = _pool_box_features(
        feature_map,
        detections,
        image_width=image_width,
        image_height=image_height,
        scale=float(scale),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
        patch_size=patch_size,
        layer_count=1,
        roi_pool=roi_pool,
    )
    del video, masks_x, masks_y, predicted, feature_map
    return features.astype(np.float32, copy=False)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("Build causal dense V-JEPA 2.1 predictor cache")
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--resolution", type=int, default=384)
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--roi-pool", choices=["box_mean", "center", "center3"], default="box_mean")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-frames-per-video", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=25)
    return ap


def main() -> None:
    args = make_parser().parse_args()
    prepared = Path(args.prepared_dir).resolve()
    ann_path = prepared / "annotations" / "val.json"
    metadata_path = prepared / "metadata.json"
    annotation = read_json(ann_path)
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    if args.window < 2 or args.window % 2:
        raise ValueError("--window must be an even integer >= 2")
    if args.resolution % args.patch_size:
        raise ValueError("--resolution must be divisible by --patch-size")
    image_root = Path(args.image_root or metadata.get("image_root", "")).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {image_root}")
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    video_names = {
        int(v["id"]): str(v.get("file_name", v.get("name", v["id"])))
        for v in annotation.get("videos", [])
    }
    selected = sorted(video_names)
    if args.max_videos is not None:
        selected = selected[: int(args.max_videos)]
    selected_set = set(selected)
    images = [x for x in annotation.get("images", [])
              if int(x["video_id"]) in selected_set]
    images.sort(key=lambda x: (int(x["video_id"]), int(x["frame_id"]), int(x["id"])))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V-JEPA 2.1 extraction")
    encoder = load_official_vjepa21_encoder(
        checkpoint, model_name=args.model_name, img_size=args.resolution,
        num_frames=args.window, patch_size=args.patch_size, tubelet_size=2,
        n_output_distillation=1, device=device,
    )
    predictor = load_official_vjepa21_predictor(
        checkpoint, embed_dim=int(encoder.embed_dim), img_size=args.resolution,
        num_frames=args.window, patch_size=args.patch_size, tubelet_size=2,
        device=device,
    )

    features_all = []
    offsets = [0]
    image_ids = []
    rgb_buffer: deque[np.ndarray] = deque(maxlen=args.window)
    current_video = None
    frame_count = 0
    forward_count = 0
    started = time.perf_counter()
    for index, image_info in enumerate(images):
        video_id = int(image_info["video_id"])
        if video_id != current_video:
            current_video = video_id
            rgb_buffer.clear()
            frame_count = 0
        frame_count += 1
        if args.max_frames_per_video is not None and frame_count > args.max_frames_per_video:
            continue

        image_path = image_root / image_info["file_name"]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        rgb_buffer.append(rgb.copy())
        while len(rgb_buffer) < args.window:
            rgb_buffer.appendleft(rgb_buffer[0].copy())
        detections = _load_detections(image_info)
        current_features = _extract_dense_predictor_features(
            encoder, predictor, list(rgb_buffer), detections,
            resolution=args.resolution, patch_size=args.patch_size, device=device,
            image_width=width, image_height=height, roi_pool=args.roi_pool,
        )
        features_all.append(current_features)
        offsets.append(offsets[-1] + len(current_features))
        image_ids.append(int(image_info["id"]))
        forward_count += int(len(detections) > 0)
        if (index + 1) % max(1, args.log_every) == 0:
            print(
                f"[{index + 1}/{len(images)}] video={video_names.get(video_id, video_id)} "
                f"frame={image_info['frame_id']} detections={len(detections)} "
                f"forwards={forward_count} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    features = (np.concatenate(features_all, axis=0).astype(np.float32, copy=False)
                if features_all else np.zeros((0, 1664), dtype=np.float32))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output), features=features,
        offsets=np.asarray(offsets, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=np.int64),
    )
    meta = {
        "schema": "generaltrack-official-vjepa21-causal-dense-predictor-cache-v1",
        "status": "complete", "prepared_dir": str(prepared),
        "annotation": str(ann_path), "image_root": str(image_root),
        "model": encoder_metadata(
            encoder, model_name=args.model_name, checkpoint=checkpoint,
            num_frames=args.window, img_size=args.resolution,
            patch_size=args.patch_size, tubelet_size=2,
        ),
        "checkpoint_sha256": sha256_file(checkpoint),
        "annotation_sha256": sha256_file(ann_path),
        "feature_shape": [int(x) for x in features.shape],
        "num_images": len(image_ids), "num_detections": int(features.shape[0]),
        "video_ids": selected, "window": int(args.window),
        "target_mask": "all_patches_in_final_tubelet",
        "roi_pool": args.roi_pool, "feature_dim": 1664,
        "feature_source": "official_vjepa21_predictor_dense_final_tubelet",
        "predictor_used": True, "causal": True,
        "future_frames_read": False, "gt_read": False,
        "target_domains_read": [], "rgb_input": True,
        "letterbox_padding": "imagenet_mean_pixel", "device": str(device),
        "autocast": "bfloat16", "num_forwards": forward_count,
        "elapsed_seconds": float(time.perf_counter() - started),
        "command": " ".join([os.environ.get("_", "python"), *sys.argv]),
    }
    with output.with_suffix(output.suffix + ".json").open("w") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    del predictor, encoder
    gc.collect()


if __name__ == "__main__":
    main()
