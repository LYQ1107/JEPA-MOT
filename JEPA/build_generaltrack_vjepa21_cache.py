"""Build causal, dense official V-JEPA 2.1 features for GeneralTrack.

The cache is an execution optimization only.  Every feature for frame ``t``
is produced from a clip ending at ``t`` and from the fixed detector rows in
the prepared annotation.  No target annotations are opened by this script.

The current-frame representation is obtained from the frozen V-JEPA 2.1
encoder and pooled over the detector box on the dense patch grid.  By default
the last tubelet slice is used; an optional causal mean readout pools all
tubelet slices in the window ending at the current frame.  The predictor is
never constructed or called.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from archive_paths import rebase_archive_path, resolve_archive_path

from vjepa21_encoder import (
    encoder_metadata,
    forward_tokens,
    forward_predictor_tokens,
    load_official_vjepa21_encoder,
    load_official_vjepa21_predictor,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "vjepa2_1_vit_large_384"
DEFAULT_WINDOW = 16
DEFAULT_RESOLUTION = 384
DEFAULT_PATCH = 16
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def read_json(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _letterbox_rgb(rgb: np.ndarray, resolution: int):
    """Resize a wide frame without cropping and return mapping metadata."""

    height, width = rgb.shape[:2]
    scale = min(float(resolution) / float(width), float(resolution) / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    # Fill padding with the ImageNet mean in pixel space instead of black.
    canvas = np.empty((resolution, resolution, 3), dtype=np.uint8)
    fill = np.clip(np.round(IMAGENET_MEAN * 255.0), 0, 255).astype(np.uint8)
    canvas[...] = fill
    pad_left = (resolution - resized_width) // 2
    pad_top = (resolution - resized_height) // 2
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return canvas, float(scale), int(pad_left), int(pad_top)


def _normalize_frame(canvas: np.ndarray) -> torch.Tensor:
    frame = torch.from_numpy(canvas).permute(2, 0, 1).contiguous().float() / 255.0
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=frame.dtype).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=frame.dtype).view(3, 1, 1)
    return (frame - mean) / std


def _box_crop_rgb(rgb: np.ndarray, detection: np.ndarray, context: float, resolution: int) -> np.ndarray:
    """Make a square, mean-padded crop around one current-frame box."""

    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = [float(x) for x in detection[:4]]
    x1, x2 = sorted((max(0.0, min(float(width), x1)), max(0.0, min(float(width), x2))))
    y1, y2 = sorted((max(0.0, min(float(height), y1)), max(0.0, min(float(height), y2))))
    box_width = max(2.0, x2 - x1)
    box_height = max(2.0, y2 - y1)
    side = max(4, int(round(max(box_width, box_height) * float(context))))
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    left = int(round(center_x - 0.5 * side))
    top = int(round(center_y - 0.5 * side))
    right = left + side
    bottom = top + side

    fill = np.clip(np.round(IMAGENET_MEAN * 255.0), 0, 255).astype(np.uint8)
    canvas = np.empty((side, side, 3), dtype=np.uint8)
    canvas[...] = fill
    src_left, src_top = max(0, left), max(0, top)
    src_right, src_bottom = min(width, right), min(height, bottom)
    if src_right > src_left and src_bottom > src_top:
        dst_left, dst_top = src_left - left, src_top - top
        canvas[dst_top : dst_top + (src_bottom - src_top),
               dst_left : dst_left + (src_right - src_left)] = rgb[src_top:src_bottom, src_left:src_right]
    return cv2.resize(canvas, (resolution, resolution), interpolation=cv2.INTER_AREA)


def _pool_crop_tokens(
    tokens: torch.Tensor,
    *,
    feature_layers: str,
    layer_count: int,
    embed_dim: int,
) -> torch.Tensor:
    """Pool object-crop tokens and optionally combine official layer reads."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected crop tokens [B,N,D], got {tuple(tokens.shape)}")
    pooled = tokens.float().mean(dim=1)
    if layer_count == 1:
        return pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-8)
    if pooled.shape[1] != layer_count * embed_dim:
        raise ValueError(
            f"Expected {layer_count * embed_dim} hierarchical crop channels, got {pooled.shape[1]}"
        )
    layers = pooled.reshape(pooled.shape[0], layer_count, embed_dim)
    layers = layers / layers.norm(dim=2, keepdim=True).clamp_min(1e-8)
    if feature_layers == "hierarchical_concat":
        return layers.reshape(pooled.shape[0], layer_count * embed_dim)
    result = layers.mean(dim=1)
    return result / result.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _extract_box_crop_features(
    encoder,
    rgb: np.ndarray,
    detections: np.ndarray,
    *,
    context: float,
    resolution: int,
    device: torch.device,
    feature_layers: str,
    layer_count: int,
    batch_size: int,
    temporal_replicates: int = 2,
) -> np.ndarray:
    """Extract current-frame causal V-JEPA features for detection crops."""

    if len(detections) == 0:
        output_dim = int(encoder.embed_dim) * (layer_count if feature_layers == "hierarchical_concat" else 1)
        return np.zeros((0, output_dim), dtype=np.float32)
    crops = torch.stack(
        [_normalize_frame(_box_crop_rgb(rgb, detection, context, resolution))
         for detection in detections],
        dim=0,
    )
    outputs = []
    for start in range(0, len(crops), max(1, int(batch_size))):
        batch = crops[start : start + max(1, int(batch_size))].unsqueeze(2)
        batch = batch.repeat(1, 1, int(temporal_replicates), 1, 1).to(
            device, non_blocking=True
        )
        tokens = forward_tokens(
            encoder,
            batch,
            hierarchical=feature_layers != "final",
        )
        pooled = _pool_crop_tokens(
            tokens,
            feature_layers=feature_layers,
            layer_count=layer_count,
            embed_dim=int(encoder.embed_dim),
        )
        outputs.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        del batch, tokens, pooled
    return np.concatenate(outputs, axis=0)


def _pool_causal_crop_tokens(
    tokens: torch.Tensor,
    *,
    temporal_pool: str,
    window: int,
    feature_layers: str,
    layer_count: int,
    embed_dim: int,
) -> torch.Tensor:
    """Pool spatial tokens from a causal object crop video.

    The crop is centered using only the current detector box and is sampled
    at the same location from the current frame and its preceding frames.
    Each tubelet readout is normalized per official layer before the temporal
    readout, matching the dense-cache hierarchy treatment.
    """

    if tokens.ndim != 3:
        raise ValueError(f"Expected crop tokens [B,N,D], got {tuple(tokens.shape)}")
    if window < 2 or window % 2:
        raise ValueError(f"Expected an even causal crop window, got {window}")
    temporal = window // 2
    spatial_tokens = (tokens.shape[1] // temporal)
    if spatial_tokens * temporal != tokens.shape[1]:
        raise ValueError(
            f"Token count {tokens.shape[1]} is not divisible by temporal depth {temporal}"
        )
    if tokens.shape[2] != layer_count * embed_dim:
        raise ValueError(
            f"Expected crop channel dimension {layer_count * embed_dim}, got {tokens.shape[2]}"
        )

    # [B, temporal, spatial, layer, channel] -> spatially pooled reads.
    maps = tokens.float().reshape(
        tokens.shape[0], temporal, spatial_tokens, layer_count, embed_dim
    )
    per_time = maps.mean(dim=2)
    per_time = per_time / per_time.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if temporal_pool == "last":
        selected = per_time[:, -1:]
    else:
        count = {"mean": temporal, "recent2": 2, "recent4": 4}[temporal_pool]
        selected = per_time[:, -min(temporal, count):]
    layer_reads = selected.mean(dim=1)
    layer_reads = layer_reads / layer_reads.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if layer_count > 1 and feature_layers == "hierarchical_concat":
        return layer_reads.reshape(tokens.shape[0], layer_count * embed_dim)
    pooled = layer_reads.mean(dim=1)
    return pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _extract_causal_box_crop_features(
    encoder,
    rgb_frames: list[np.ndarray],
    detections: np.ndarray,
    *,
    context: float,
    resolution: int,
    device: torch.device,
    feature_layers: str,
    layer_count: int,
    batch_size: int,
    window: int,
    temporal_pool: str,
    patch_size: int,
) -> np.ndarray:
    """Extract video-encoder features from detector-centered causal crops."""

    output_dim = int(encoder.embed_dim) * (
        layer_count if feature_layers == "hierarchical_concat" else 1
    )
    if len(detections) == 0:
        return np.zeros((0, output_dim), dtype=np.float32)
    if len(rgb_frames) != window:
        raise ValueError(
            f"Expected {window} causal RGB frames, got {len(rgb_frames)}"
        )

    crop_rows = []
    for detection in detections:
        crop_rows.append(torch.stack([
            _normalize_frame(_box_crop_rgb(
                rgb, detection, context, resolution))
            for rgb in rgb_frames
        ], dim=0))
    crops = torch.stack(crop_rows, dim=0)
    outputs = []
    for start in range(0, len(crops), max(1, int(batch_size))):
        batch = crops[start : start + max(1, int(batch_size))]
        batch = batch.permute(0, 2, 1, 3, 4).contiguous().to(
            device, non_blocking=True
        )
        tokens = forward_tokens(
            encoder,
            batch,
            autocast_dtype=torch.bfloat16,
            hierarchical=feature_layers != "final",
        )
        pooled = _pool_causal_crop_tokens(
            tokens,
            temporal_pool=temporal_pool,
            window=window,
            feature_layers=feature_layers,
            layer_count=layer_count,
            embed_dim=int(encoder.embed_dim),
        )
        outputs.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        del batch, tokens, pooled
    return np.concatenate(outputs, axis=0)


def _predictor_masks(
    *,
    window: int,
    resolution: int,
    patch_size: int,
    device: torch.device,
    batch_size: int,
):
    """Mask a 3x3 center target in the final tubelet of a crop video."""

    spatial = resolution // patch_size
    temporal = window // 2
    if spatial < 3:
        raise ValueError(f"Predictor crop needs at least a 3x3 grid, got {spatial}")
    row = spatial // 2
    col = spatial // 2
    target = []
    for rr in range(row - 1, row + 2):
        for cc in range(col - 1, col + 2):
            target.append((temporal - 1) * spatial * spatial + rr * spatial + cc)
    target = np.asarray(sorted(target), dtype=np.int64)
    all_tokens = np.arange(temporal * spatial * spatial, dtype=np.int64)
    context = np.setdiff1d(all_tokens, target, assume_unique=True)
    masks_x = torch.from_numpy(context).to(device=device).view(1, -1).repeat(
        int(batch_size), 1
    )
    masks_y = torch.from_numpy(target).to(device=device).view(1, -1).repeat(
        int(batch_size), 1
    )
    return masks_x, masks_y


def _extract_causal_box_crop_predictor_features(
    encoder,
    predictor,
    rgb_frames: list[np.ndarray],
    detections: np.ndarray,
    *,
    context: float,
    resolution: int,
    device: torch.device,
    batch_size: int,
    window: int,
    patch_size: int,
    predictor_pool: str,
) -> np.ndarray:
    """Use the official predictor to forecast a crop's masked center tokens."""

    output_dim = 1664
    if len(detections) == 0:
        return np.zeros((0, output_dim), dtype=np.float32)
    if len(rgb_frames) != window:
        raise ValueError(
            f"Expected {window} causal RGB frames, got {len(rgb_frames)}"
        )

    crop_rows = []
    for detection in detections:
        crop_rows.append(torch.stack([
            _normalize_frame(_box_crop_rgb(
                rgb, detection, context, resolution))
            for rgb in rgb_frames
        ], dim=0))
    crops = torch.stack(crop_rows, dim=0)
    outputs = []
    for start in range(0, len(crops), max(1, int(batch_size))):
        batch = crops[start : start + max(1, int(batch_size))]
        batch = batch.permute(0, 2, 1, 3, 4).contiguous().to(
            device, non_blocking=True
        )
        masks_x, masks_y = _predictor_masks(
            window=window,
            resolution=resolution,
            patch_size=patch_size,
            device=device,
            batch_size=len(batch),
        )
        predicted = forward_predictor_tokens(
            encoder,
            predictor,
            batch,
            masks_x,
            masks_y,
            autocast_dtype=torch.bfloat16,
        )
        if predictor_pool == "center":
            pooled = predicted.float()[:, predicted.shape[1] // 2]
        elif predictor_pool == "center3":
            middle = predicted.shape[1] // 2
            pooled = predicted.float()[:, middle - 1:middle + 2].mean(dim=1)
        else:
            pooled = predicted.float().mean(dim=1)
        pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-8)
        outputs.append(pooled.cpu().numpy().astype(np.float32, copy=False))
        del batch, masks_x, masks_y, predicted, pooled
    return np.concatenate(outputs, axis=0)


def _extract_causal_box_crop_predictor_features_batched(
    encoder,
    predictor,
    items: list[tuple[list[np.ndarray], np.ndarray]],
    *,
    context: float,
    resolution: int,
    device: torch.device,
    batch_size: int,
    window: int,
    patch_size: int,
    predictor_pool: str,
    crop_backend: str,
) -> tuple[list[np.ndarray], int]:
    """Extract predictor features for several frames in shared crop batches.

    Each item still owns its own causal window ending at its current frame.
    Only the GPU batch dimension is changed: crops from adjacent frames are
    interleaved in one frozen JEPA forward and then split back to frame order.
    """

    output_dim = 1664
    outputs: list[list[np.ndarray]] = [[] for _ in items]
    crop_batch: list[tuple[int, np.ndarray]] = []
    forward_count = 0

    def flush() -> None:
        nonlocal forward_count
        if not crop_batch:
            return
        if crop_backend == "gpu_roi_align":
            batch = _make_gpu_causal_crop_batch(
                items,
                crop_batch,
                context=context,
                resolution=resolution,
                device=device,
            )
        else:
            batch = torch.stack([
                torch.stack([
                    _normalize_frame(_box_crop_rgb(
                        items[owner][0][frame], detection, context, resolution))
                    for frame in range(len(items[owner][0]))
                ], dim=0)
                for owner, detection in crop_batch
            ], dim=0).permute(0, 2, 1, 3, 4).contiguous()
            batch = batch.to(device, non_blocking=True)
        masks_x, masks_y = _predictor_masks(
            window=window,
            resolution=resolution,
            patch_size=patch_size,
            device=device,
            batch_size=len(batch),
        )
        predicted = forward_predictor_tokens(
            encoder,
            predictor,
            batch,
            masks_x,
            masks_y,
            autocast_dtype=torch.bfloat16,
        )
        if predictor_pool == "center":
            pooled = predicted.float()[:, predicted.shape[1] // 2]
        elif predictor_pool == "center3":
            middle = predicted.shape[1] // 2
            pooled = predicted.float()[:, middle - 1:middle + 2].mean(dim=1)
        else:
            pooled = predicted.float().mean(dim=1)
        pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = pooled.cpu().numpy().astype(np.float32, copy=False)
        forward_count += 1
        for row, (owner, _) in zip(pooled, crop_batch):
            outputs[owner].append(row)
        del batch, masks_x, masks_y, predicted, pooled
        crop_batch.clear()

    for item_index, (rgb_frames, detections) in enumerate(items):
        if len(detections) == 0:
            continue
        if len(rgb_frames) != window:
            raise ValueError(
                f"Expected {window} causal RGB frames, got {len(rgb_frames)}"
            )
        for detection in detections:
            crop_batch.append((item_index, np.asarray(detection, dtype=np.float32)))
            if len(crop_batch) >= max(1, int(batch_size)):
                flush()
    flush()

    result = []
    for item_output in outputs:
        if item_output:
            result.append(np.stack(item_output, axis=0).astype(np.float32, copy=False))
        else:
            result.append(np.zeros((0, output_dim), dtype=np.float32))
    return result, forward_count


def _make_gpu_causal_crop_batch(
    items: list[tuple[list[np.ndarray], np.ndarray]],
    crop_batch: list[tuple[int, np.ndarray]],
    *,
    context: float,
    resolution: int,
    device: torch.device,
) -> torch.Tensor:
    """Create normalized causal crops with batched GPU ROIAlign.

    The CPU implementation above is retained as the reference readout.  This
    path is numerically close to its mean-padded square crop, but avoids the
    repeated Python/OpenCV resize loop that dominates dense MOTSynth scenes.
    """

    grouped: dict[int, list[tuple[int, np.ndarray]]] = {}
    for batch_index, (owner, detection) in enumerate(crop_batch):
        grouped.setdefault(owner, []).append((batch_index, detection))
    result: list[torch.Tensor | None] = [None] * len(crop_batch)
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    for owner, records in grouped.items():
        rgb_frames = items[owner][0]
        height, width = rgb_frames[0].shape[:2]
        geometries = []
        for _, detection in records:
            x1, y1, x2, y2 = [float(x) for x in detection[:4]]
            x1, x2 = sorted((max(0.0, min(float(width), x1)),
                             max(0.0, min(float(width), x2))))
            y1, y2 = sorted((max(0.0, min(float(height), y1)),
                             max(0.0, min(float(height), y2))))
            box_width = max(2.0, x2 - x1)
            box_height = max(2.0, y2 - y1)
            side = max(4, int(round(max(box_width, box_height) * float(context))))
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            left = int(round(center_x - 0.5 * side))
            top = int(round(center_y - 0.5 * side))
            geometries.append((left, top, left + side, top + side))
        pad = max(
            1,
            max(-left for left, _, _, _ in geometries),
            max(-top for _, top, _, _ in geometries),
            max(right - width for _, _, right, _ in geometries),
            max(bottom - height for _, _, _, bottom in geometries),
        )
        frames = torch.from_numpy(np.stack(rgb_frames, axis=0)).permute(0, 3, 1, 2)
        frames = frames.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
        frames = (frames - mean) / std
        frames = F.pad(frames, (pad, pad, pad, pad), value=0.0)
        rois = []
        for frame_index in range(len(rgb_frames)):
            for left, top, right, bottom in geometries:
                rois.append((frame_index, left + pad, top + pad, right + pad, bottom + pad))
        rois = torch.as_tensor(rois, dtype=torch.float32, device=device)
        crops = roi_align(
            frames,
            rois,
            output_size=(resolution, resolution),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=False,
        )
        crops = crops.reshape(len(rgb_frames), len(records), 3, resolution, resolution)
        crops = crops.permute(1, 2, 0, 3, 4).contiguous()
        for record_index, (batch_index, _) in enumerate(records):
            result[batch_index] = crops[record_index]
        del frames, rois, crops

    if any(crop is None for crop in result):
        raise RuntimeError("GPU crop construction lost a detector row")
    return torch.stack([crop for crop in result if crop is not None], dim=0)


def _temporal_maps(tokens: torch.Tensor, window: int, resolution: int, patch_size: int) -> torch.Tensor:
    """Return video tokens as ``[temporal, height, width, channels]``."""

    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(f"Expected encoder output [1,N,D], got {tuple(tokens.shape)}")
    spatial = resolution // patch_size
    if window == 1:
        expected = spatial * spatial
        if tokens.shape[1] != expected:
            raise ValueError(f"Expected {expected} image tokens, got {tokens.shape[1]}")
        return tokens[0].reshape(1, spatial, spatial, tokens.shape[-1])
    if window % 2:
        raise ValueError("A video window must be even because the official tubelet size is 2")
    temporal = window // 2
    expected = temporal * spatial * spatial
    if tokens.shape[1] != expected:
        raise ValueError(f"Expected {expected} video tokens, got {tokens.shape[1]}")
    return tokens[0].reshape(temporal, spatial, spatial, tokens.shape[-1])


def _last_temporal_map(tokens: torch.Tensor, window: int, resolution: int, patch_size: int) -> torch.Tensor:
    """Backward-compatible last-tubelet readout."""

    return _temporal_maps(tokens, window, resolution, patch_size)[-1]


def _pool_box_features(
    feature_map: torch.Tensor,
    detections: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    scale: float,
    pad_left: int,
    pad_top: int,
    patch_size: int,
    layer_count: int = 1,
    roi_pool: str = "box_mean",
) -> np.ndarray:
    """Mean-pool dense patch tokens whose centers fall inside each box."""

    feature_map = feature_map.detach().float().cpu()
    grid_h, grid_w, feature_dim = feature_map.shape
    if layer_count < 1 or feature_dim % layer_count:
        raise ValueError(
            f"Feature dimension {feature_dim} is not divisible by layer count {layer_count}"
        )
    per_layer_dim = feature_dim // layer_count
    y_centers = (np.arange(grid_h, dtype=np.float32) + 0.5) * float(patch_size)
    x_centers = (np.arange(grid_w, dtype=np.float32) + 0.5) * float(patch_size)
    xx, yy = np.meshgrid(x_centers, y_centers)
    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    flat = feature_map.reshape(-1, feature_dim)

    if roi_pool not in {"box_mean", "center", "center3"}:
        raise ValueError(f"Unknown ROI pool: {roi_pool}")

    outputs = []
    for detection in detections:
        x1, y1, x2, y2 = [float(x) for x in detection[:4]]
        mapped = np.asarray(
            [
                x1 * scale + pad_left,
                y1 * scale + pad_top,
                x2 * scale + pad_left,
                y2 * scale + pad_top,
            ],
            dtype=np.float32,
        )
        # Clip to the actual letterboxed frame and use patch centers.
        mapped[[0, 2]] = np.clip(mapped[[0, 2]], 0, image_width * scale + pad_left)
        mapped[[1, 3]] = np.clip(mapped[[1, 3]], 0, image_height * scale + pad_top)
        inside = (
            (points[:, 0] >= mapped[0])
            & (points[:, 0] <= mapped[2])
            & (points[:, 1] >= mapped[1])
            & (points[:, 1] <= mapped[3])
        )
        if roi_pool in {"center", "center3"}:
            # A center readout avoids averaging background when a vehicle is
            # smaller than the dense ROI.  ``center3`` uses a fixed 3x3 patch
            # neighborhood around the box center and is still geometry-only.
            center_x = 0.5 * (mapped[0] + mapped[2])
            center_y = 0.5 * (mapped[1] + mapped[3])
            center_col = int(np.argmin(np.abs(x_centers - center_x)))
            center_row = int(np.argmin(np.abs(y_centers - center_y)))
            if roi_pool == "center":
                inside[:] = False
                inside[center_row * grid_w + center_col] = True
            else:
                radius = 1
                inside[:] = False
                row_lo = max(0, center_row - radius)
                row_hi = min(grid_h, center_row + radius + 1)
                col_lo = max(0, center_col - radius)
                col_hi = min(grid_w, center_col + radius + 1)
                inside.reshape(grid_h, grid_w)[row_lo:row_hi, col_lo:col_hi] = True
        if not np.any(inside):
            center = np.asarray([(mapped[0] + mapped[2]) / 2, (mapped[1] + mapped[3]) / 2])
            distances = np.sum((points - center[None, :]) ** 2, axis=1)
            inside[np.argmin(distances)] = True
        pooled = flat[torch.from_numpy(inside)].mean(dim=0)
        if layer_count == 1:
            pooled = pooled / pooled.norm().clamp_min(1e-8)
        else:
            # The official hierarchical output is concatenated as
            # [layer_1, layer_2, layer_3, layer_4].  Normalize each layer
            # before averaging so the result is not dominated by a layer's
            # raw norm; this remains a frozen, deterministic encoder readout.
            pooled = pooled.reshape(layer_count, per_layer_dim)
            pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-8)
            pooled = pooled.mean(dim=0)
            pooled = pooled / pooled.norm().clamp_min(1e-8)
        outputs.append(pooled.numpy().astype(np.float32))

    if not outputs:
        return np.zeros((0, feature_dim), dtype=np.float32)
    return np.stack(outputs, axis=0)


def _pool_temporal_box_features(
    temporal_maps: torch.Tensor,
    detections: np.ndarray,
    *,
    temporal_pool: str,
    image_width: int,
    image_height: int,
    scale: float,
    pad_left: int,
    pad_top: int,
    patch_size: int,
    layer_count: int = 1,
    roi_pool: str = "box_mean",
) -> np.ndarray:
    """Pool one current-frame box over a causal temporal token readout."""

    if temporal_maps.ndim != 4:
        raise ValueError(
            f"Expected temporal maps [T,H,W,D], got {tuple(temporal_maps.shape)}"
        )
    if temporal_pool not in {"last", "mean", "recent2", "recent4"}:
        raise ValueError(f"Unknown temporal pool: {temporal_pool}")
    if temporal_pool == "last" or temporal_maps.shape[0] == 1:
        return _pool_box_features(
            temporal_maps[-1],
            detections,
            image_width=image_width,
            image_height=image_height,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            patch_size=patch_size,
            layer_count=layer_count,
            roi_pool=roi_pool,
        )

    # Pool each tubelet independently before averaging.  This keeps the
    # detector geometry fixed while making the only temporal context change
    # the representation readout.  Re-normalize the average because each
    # individual readout is normalized by _pool_box_features.
    slice_count = {"mean": temporal_maps.shape[0], "recent2": 2, "recent4": 4}[temporal_pool]
    maps = temporal_maps[-min(temporal_maps.shape[0], slice_count):]
    per_slice = [
        _pool_box_features(
            feature_map,
            detections,
            image_width=image_width,
            image_height=image_height,
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            patch_size=patch_size,
            layer_count=layer_count,
            roi_pool=roi_pool,
        )
        for feature_map in maps
    ]
    pooled = np.mean(np.stack(per_slice, axis=0), axis=0, dtype=np.float32)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norms, 1e-8)


def _load_detections(image_info: dict) -> np.ndarray:
    detections = np.asarray(image_info.get("detection", []), dtype=np.float32)
    if detections.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    return detections.reshape(-1, 6)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Build causal official V-JEPA 2.1 GeneralTrack cache")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument(
        "--temporal-pool",
        choices=["last", "mean", "recent2", "recent4"],
        default="last",
        help="Causal token readout: current, all, or recent 2/4 tubelets",
    )
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH)
    parser.add_argument(
        "--roi-pool",
        choices=["box_mean", "center", "center3"],
        default="box_mean",
        help="Dense patch readout geometry: box mean or center patch neighborhood",
    )
    parser.add_argument(
        "--roi-mode",
        choices=["dense", "box_crop", "image_box_crop", "causal_box_crop", "causal_box_crop_predictor"],
        default="dense",
        help=("Readout source: dense frame tokens, one current-frame crop, "
              "a detector-centered causal crop video, or its official predictor"),
    )
    parser.add_argument("--roi-context", type=float, default=1.25)
    parser.add_argument("--crop-batch-size", type=int, default=16)
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=1,
        help=("For causal_box_crop_predictor, combine this many adjacent "
              "frames' detector crops into shared JEPA forwards"),
    )
    parser.add_argument(
        "--crop-backend",
        choices=["cpu", "gpu_roi_align"],
        default="cpu",
        help="Crop implementation for shared predictor batches",
    )
    parser.add_argument(
        "--feature-score-min",
        type=float,
        default=None,
        help=("Only extract JEPA features for detector rows with score above "
              "this value; lower-score rows remain zero-filled so cache "
              "offsets and detector alignment are unchanged"),
    )
    parser.add_argument(
        "--predictor-pool",
        choices=["mean", "center", "center3"],
        default="mean",
        help="Pooling over the official predictor masked target tokens",
    )
    parser.add_argument(
        "--feature-layers",
        choices=["final", "hierarchical_mean", "hierarchical_concat"],
        default="final",
        help="Official encoder readout: final layer or the four hierarchical layers",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-frames-per-video", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-autocast", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    prepared_dir = Path(args.prepared_dir).resolve()
    annotation_path = prepared_dir / "annotations" / "val.json"
    metadata_path = prepared_dir / "metadata.json"
    annotation = read_json(annotation_path)
    metadata = read_json(metadata_path) if metadata_path.is_file() else {}
    is_gt_detection_oracle = metadata.get("det_source") == "oracle_gt"

    if args.window < 1 or (args.window > 1 and args.window % 2):
        raise ValueError("--window must be 1 or an even number")
    if args.resolution % args.patch_size:
        raise ValueError("--resolution must be divisible by --patch-size")
    checkpoint = Path(args.checkpoint).resolve()
    image_root = rebase_archive_path(args.image_root or metadata.get("image_root", ""))
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image root: {image_root}")

    video_names = {
        int(v["id"]): str(v.get("file_name", v.get("name", v["id"])))
        for v in annotation.get("videos", [])
    }
    selected_video_ids = sorted(video_names)
    if args.max_videos is not None:
        selected_video_ids = selected_video_ids[: args.max_videos]
    selected_video_ids = set(selected_video_ids)
    images = [
        image
        for image in annotation["images"]
        if int(image["video_id"]) in selected_video_ids
    ]
    images.sort(key=lambda x: (int(x["video_id"]), int(x["frame_id"]), int(x["id"])))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V-JEPA 2.1 extraction")
    if args.roi_mode in {"box_crop", "image_box_crop"} and args.window != 1:
        raise ValueError(
            f"--roi-mode {args.roi_mode} requires --window 1 for current-frame crops"
        )
    if (args.roi_mode == "causal_box_crop_predictor" and
            args.feature_layers != "final"):
        raise ValueError(
            "--roi-mode causal_box_crop_predictor currently exposes the official final predictor readout"
        )
    # The public ViT-L checkpoint contains a pretrained video tubelet
    # projection, while its optional image-only projection is not present in
    # the checkpoint.  Two identical current crops therefore retain the
    # original video patch weights without reading a future frame.
    encoder_window = 2 if args.roi_mode in {"box_crop", "image_box_crop"} else args.window
    encoder = load_official_vjepa21_encoder(
        checkpoint,
        model_name=args.model_name,
        img_size=args.resolution,
        num_frames=encoder_window,
        patch_size=args.patch_size,
        tubelet_size=2,
        n_output_distillation=4 if args.feature_layers != "final" else None,
        device=device,
    )
    predictor = None
    if args.roi_mode == "causal_box_crop_predictor":
        predictor = load_official_vjepa21_predictor(
            checkpoint,
            embed_dim=int(encoder.embed_dim),
            img_size=args.resolution,
            num_frames=encoder_window,
            patch_size=args.patch_size,
            tubelet_size=2,
            device=device,
        )
    hierarchical = args.feature_layers != "final"
    layer_count = 4 if hierarchical else 1
    feature_output_dim = (
        1664 if args.roi_mode == "causal_box_crop_predictor"
        else int(encoder.embed_dim) * (
            layer_count if args.feature_layers == "hierarchical_concat" else 1
        )
    )
    model_meta = encoder_metadata(
        encoder,
        model_name=args.model_name,
        checkpoint=checkpoint,
        num_frames=encoder_window,
        img_size=args.resolution,
        patch_size=args.patch_size,
        tubelet_size=2,
    )

    all_features = []
    offsets = [0]
    image_ids = []
    current_video = None
    frame_buffer: deque[torch.Tensor] = deque(maxlen=args.window)
    rgb_buffer: deque[np.ndarray] = deque(maxlen=args.window)
    frame_count_in_video = 0
    forward_count = 0
    feature_rows_computed = 0
    feature_rows_zero_filled = 0
    start_time = time.perf_counter()

    def commit_item(item, features: np.ndarray) -> None:
        """Restore detector-row alignment and append one processed image."""

        detections = item["detections"]
        if args.feature_score_min is not None:
            expanded_features = np.zeros(
                (len(detections), feature_output_dim), dtype=np.float32
            )
            expanded_features[item["feature_mask"]] = features
            features = expanded_features
        if len(features) != len(detections):
            raise RuntimeError(
                f"Feature/detection mismatch at image_id={item['image_info']['id']}: "
                f"{len(features)} vs {len(detections)}"
            )
        all_features.append(features)
        offsets.append(offsets[-1] + len(features))
        image_ids.append(int(item["image_info"]["id"]))
        if (item["index"] + 1) % max(1, args.log_every) == 0:
            elapsed = time.perf_counter() - start_time
            print(
                f"[{item['index'] + 1}/{len(images)}] "
                f"video={video_names.get(item['video_id'], item['video_id'])} "
                f"frame={item['frame_id']} detections={len(detections)} "
                f"forwards={forward_count} elapsed={elapsed:.1f}s",
                flush=True,
            )

    pending_predictor = []

    def flush_predictor_pending() -> None:
        nonlocal forward_count
        if not pending_predictor:
            return
        predictor_items = [
            (item["rgb_frames"], item["feature_detections"])
            for item in pending_predictor
        ]
        batch_features, shared_forwards = _extract_causal_box_crop_predictor_features_batched(
            encoder,
            predictor,
            predictor_items,
            context=args.roi_context,
            resolution=args.resolution,
            device=device,
            batch_size=args.crop_batch_size,
            window=args.window,
            patch_size=args.patch_size,
            predictor_pool=args.predictor_pool,
            crop_backend=args.crop_backend,
        )
        forward_count += shared_forwards
        for item, features in zip(pending_predictor, batch_features):
            commit_item(item, features)
        pending_predictor.clear()

    for index, image_info in enumerate(images):
        video_id = int(image_info["video_id"])
        frame_id = int(image_info["frame_id"])
        if video_id != current_video:
            current_video = video_id
            frame_buffer.clear()
            rgb_buffer.clear()
            frame_count_in_video = 0

        frame_count_in_video += 1
        if args.max_frames_per_video is not None and frame_count_in_video > args.max_frames_per_video:
            # Smoke runs intentionally stop at a frame boundary.  The output
            # annotation must be restricted to the same frames by the caller.
            continue

        relative_path = image_info["file_name"]
        image_path = resolve_archive_path(str(image_root / relative_path))
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_height, original_width = rgb.shape[:2]
        detections = _load_detections(image_info)
        if args.feature_score_min is None:
            feature_mask = np.ones(len(detections), dtype=bool)
        else:
            feature_mask = detections[:, 4] > float(args.feature_score_min)
        feature_detections = detections[feature_mask]
        feature_rows_computed += int(len(feature_detections))
        feature_rows_zero_filled += int(len(detections) - len(feature_detections))

        # Advance the causal source window for every frame, including frames
        # with no detector rows.  Skipping empty frames would silently change
        # a 16-frame causal window into a variable-duration window.
        if args.roi_mode in {"causal_box_crop", "causal_box_crop_predictor"}:
            if not rgb_buffer:
                rgb_buffer.append(rgb.copy())
            else:
                rgb_buffer.append(rgb.copy())
            while len(rgb_buffer) < args.window:
                rgb_buffer.appendleft(rgb_buffer[0].copy())
        elif args.roi_mode == "dense":
            canvas, scale, pad_left, pad_top = _letterbox_rgb(rgb, args.resolution)
            frame = _normalize_frame(canvas)
            frame_buffer.append(frame)
            while len(frame_buffer) < args.window:
                frame_buffer.appendleft(frame_buffer[0].clone())

        if (args.roi_mode == "causal_box_crop_predictor" and
                args.frame_batch_size > 1):
            pending_predictor.append({
                "index": index,
                "image_info": image_info,
                "video_id": video_id,
                "frame_id": frame_id,
                "detections": detections,
                "feature_mask": feature_mask,
                "feature_detections": feature_detections,
                "rgb_frames": list(rgb_buffer),
            })
            if len(pending_predictor) >= args.frame_batch_size:
                flush_predictor_pending()
            continue

        if len(feature_detections):
            if args.roi_mode == "box_crop":
                features = _extract_box_crop_features(
                    encoder,
                    rgb,
                    feature_detections,
                    context=args.roi_context,
                    resolution=args.resolution,
                    device=device,
                    feature_layers=args.feature_layers,
                    layer_count=layer_count,
                    batch_size=args.crop_batch_size,
                    temporal_replicates=2,
                )
                forward_count += int((len(feature_detections) + args.crop_batch_size - 1) // args.crop_batch_size)
            elif args.roi_mode == "image_box_crop":
                features = _extract_box_crop_features(
                    encoder,
                    rgb,
                    feature_detections,
                    context=args.roi_context,
                    resolution=args.resolution,
                    device=device,
                    feature_layers=args.feature_layers,
                    layer_count=layer_count,
                    batch_size=args.crop_batch_size,
                    temporal_replicates=1,
                )
                forward_count += int((len(feature_detections) + args.crop_batch_size - 1) // args.crop_batch_size)
            elif args.roi_mode == "causal_box_crop":
                features = _extract_causal_box_crop_features(
                    encoder,
                    list(rgb_buffer),
                    feature_detections,
                    context=args.roi_context,
                    resolution=args.resolution,
                    device=device,
                    feature_layers=args.feature_layers,
                    layer_count=layer_count,
                    batch_size=args.crop_batch_size,
                    window=args.window,
                    temporal_pool=args.temporal_pool,
                    patch_size=args.patch_size,
                )
                forward_count += int((len(feature_detections) + args.crop_batch_size - 1) // args.crop_batch_size)
            elif args.roi_mode == "causal_box_crop_predictor":
                features = _extract_causal_box_crop_predictor_features(
                    encoder,
                    predictor,
                    list(rgb_buffer),
                    feature_detections,
                    context=args.roi_context,
                    resolution=args.resolution,
                    device=device,
                    batch_size=args.crop_batch_size,
                    window=args.window,
                    patch_size=args.patch_size,
                    predictor_pool=args.predictor_pool,
                )
                forward_count += int((len(feature_detections) + args.crop_batch_size - 1) // args.crop_batch_size)
            else:
                clip = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).to(device, non_blocking=True)
                tokens = forward_tokens(
                    encoder,
                    clip,
                    autocast_dtype=None if args.no_autocast else torch.bfloat16,
                    hierarchical=hierarchical,
                )
                temporal_maps = _temporal_maps(
                    tokens, args.window, args.resolution, args.patch_size
                )
                features = _pool_temporal_box_features(
                    temporal_maps,
                    feature_detections,
                    temporal_pool=args.temporal_pool,
                    image_width=original_width,
                    image_height=original_height,
                    scale=scale,
                    pad_left=pad_left,
                    pad_top=pad_top,
                    patch_size=args.patch_size,
                    layer_count=layer_count,
                    roi_pool=args.roi_pool,
                )
                if args.feature_layers == "hierarchical_concat":
                    token_layers = temporal_maps.reshape(
                        temporal_maps.shape[0],
                        args.resolution // args.patch_size,
                        args.resolution // args.patch_size,
                        layer_count,
                        int(encoder.embed_dim),
                    )
                    layer_features = []
                    for layer_idx in range(layer_count):
                        layer_features.append(_pool_temporal_box_features(
                            token_layers[..., layer_idx, :],
                            feature_detections,
                            temporal_pool=args.temporal_pool,
                            image_width=original_width,
                            image_height=original_height,
                            scale=scale,
                            pad_left=pad_left,
                            pad_top=pad_top,
                            patch_size=args.patch_size,
                            layer_count=1,
                            roi_pool=args.roi_pool,
                        ))
                    features = np.concatenate(layer_features, axis=1)
                forward_count += 1
                del clip, tokens, temporal_maps
        else:
            features = np.zeros((0, feature_output_dim), dtype=np.float32)

        commit_item({
            "index": index,
            "image_info": image_info,
            "video_id": video_id,
            "frame_id": frame_id,
            "detections": detections,
            "feature_mask": feature_mask,
        }, features)

    flush_predictor_pending()

    if all_features:
        features = np.concatenate(all_features, axis=0).astype(np.float32, copy=False)
    else:
        features = np.zeros((0, int(encoder.embed_dim)), dtype=np.float32)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        features=features,
        offsets=np.asarray(offsets, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=np.int64),
    )
    sidecar = {
        "schema": "generaltrack-official-vjepa21-causal-dense-cache-v1",
        "status": "complete",
        "prepared_dir": str(prepared_dir),
        "annotation": str(annotation_path),
        "image_root": str(image_root),
        "model": model_meta,
        "checkpoint_sha256": sha256_file(checkpoint),
        "annotation_sha256": sha256_file(annotation_path),
        "feature_shape": [int(x) for x in features.shape],
        "num_images": len(image_ids),
        "num_detections": int(features.shape[0]),
        "feature_score_min": (
            None if args.feature_score_min is None else float(args.feature_score_min)
        ),
        "feature_rows_computed": int(feature_rows_computed),
        "feature_rows_zero_filled": int(feature_rows_zero_filled),
        "video_ids": sorted(int(x) for x in selected_video_ids),
        "window": int(args.window),
        "temporal_pool": args.temporal_pool,
        "roi_mode": args.roi_mode,
        "roi_context": float(args.roi_context),
        "crop_batch_size": int(args.crop_batch_size),
        "frame_batch_size": int(args.frame_batch_size),
        "crop_backend": args.crop_backend,
        "crop_temporal_input": (
            "current_frame_repeated_twice"
            if args.roi_mode == "box_crop"
            else ("current_frame_image_mode"
                  if args.roi_mode == "image_box_crop" else
                  (("same_current_detector_box_cropped_from_causal_window"
                    if args.roi_mode == "causal_box_crop" else
                    "official_predictor_masked_center_from_causal_window")
                  if args.roi_mode == "causal_box_crop_predictor" else None)
            )
        ),
        "predictor_pool": (
            args.predictor_pool
            if args.roi_mode == "causal_box_crop_predictor" else None
        ),
        "causal": True,
        "future_frames_read": False,
        # The fair source-detector cache never consumes target annotations.
        # The isolated oracle cache is intentionally different: its
        # detector rows were constructed from GT boxes, and that fact must be
        # visible in provenance even though this extractor only reads the
        # prepared ``detection`` field.
        "gt_read": bool(is_gt_detection_oracle),
        "gt_detection_oracle": bool(is_gt_detection_oracle),
        "feature_source": (
            "official_vjepa21_encoder_final_dense_tokens"
            if args.roi_mode == "dense" and args.feature_layers == "final"
            else f"official_vjepa21_encoder_{args.roi_mode}_{args.feature_layers}"
        ),
        "feature_layers": args.feature_layers,
        "predictor_used": args.roi_mode == "causal_box_crop_predictor",
        "roi_pool": args.roi_pool,
        "rgb_input": True,
        "letterbox_padding": "imagenet_mean_pixel",
        "device": str(device),
        "autocast": None if args.no_autocast else "bfloat16",
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "command": " ".join([os.environ.get("_", "python"), *sys.argv]),
    }
    with output.with_suffix(output.suffix + ".json").open("w") as handle:
        json.dump(sidecar, handle, indent=2)
    print(json.dumps(sidecar, indent=2), flush=True)
    del encoder
    if predictor is not None:
        del predictor
    gc.collect()


if __name__ == "__main__":
    main()
