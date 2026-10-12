#!/usr/bin/env python3
"""Unsupervised, target-GT-free calibration of a frozen JEPA cache.

The transform is fit only on the candidate feature matrix in the supplied
cache.  It preserves the detector/image/frame ordering and never opens the
prepared annotations or any target labels.  This is intended to test whether
the domain shift is mostly a common background direction or a per-channel
scale problem in the frozen JEPA representation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser("Apply unlabeled target-only JEPA feature calibration")
    ap.add_argument("--input", required=True, help="Existing compact [N,256] JEPA cache")
    ap.add_argument("--out", required=True, help="Transformed cache path")
    ap.add_argument("--method", choices=["center", "diag", "pca", "frame_center", "frame_center_diag"], required=True)
    ap.add_argument("--strength", type=float, default=0.5,
                    help="PCA whitening strength; 0 leaves centered vectors unchanged")
    ap.add_argument("--eps", type=float, default=1e-4)
    return ap.parse_args()


def transform(x: np.ndarray, method: str, strength: float, eps: float):
    x = np.asarray(x, np.float32)
    mean = x.mean(axis=0, keepdims=True)
    y = x - mean
    payload = {"mean_norm": float(np.linalg.norm(mean)), "method": method}
    if method in ("frame_center", "frame_center_diag"):
        raise ValueError("frame-wise transform requires frame_ids")
    if method == "diag":
        scale = np.maximum(y.std(axis=0, keepdims=True), float(eps))
        y = y / scale
        payload["scale_min"] = float(scale.min())
        payload["scale_max"] = float(scale.max())
    elif method == "pca" and float(strength) > 0:
        cov = (y.T @ y) / max(1, len(y) - 1)
        eigval, eigvec = np.linalg.eigh(cov.astype(np.float64))
        eigval = np.maximum(eigval, float(eps))
        weights = eigval ** (-0.5 * float(strength))
        y = (y.astype(np.float64) @ eigvec @ np.diag(weights) @ eigvec.T).astype(np.float32)
        payload["eig_min"] = float(eigval.min())
        payload["eig_max"] = float(eigval.max())
        payload["strength"] = float(strength)
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    y = y / np.maximum(norms, 1e-8)
    return y.astype(np.float16), payload


def transform_framewise(x: np.ndarray, frame_ids: np.ndarray, method: str,
                        eps: float):
    """Remove each frame's unlabeled common direction before normalization."""
    x = np.asarray(x, np.float32)
    frame_ids = np.asarray(frame_ids, np.int64)
    if frame_ids.shape != (len(x),):
        raise ValueError("frame_ids must align with features")
    y = np.empty_like(x)
    frame_count = 0
    scale_min, scale_max = float("inf"), 0.0
    for frame in np.unique(frame_ids):
        idx = np.flatnonzero(frame_ids == frame)
        centered = x[idx] - x[idx].mean(axis=0, keepdims=True)
        if method == "frame_center_diag":
            scale = np.maximum(centered.std(axis=0, keepdims=True), float(eps))
            centered = centered / scale
            scale_min = min(scale_min, float(scale.min()))
            scale_max = max(scale_max, float(scale.max()))
        y[idx] = centered
        frame_count += 1
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-8)
    details = {"method": method, "frame_count": int(frame_count),
               "scale_min": None if method == "frame_center" else scale_min,
               "scale_max": None if method == "frame_center" else scale_max}
    return y.astype(np.float16), details


def main() -> int:
    args = parse_args()
    inp = Path(args.input).resolve()
    out = Path(args.out).resolve()
    sidecar = inp.with_suffix(inp.suffix + ".json")
    if not inp.exists() or not sidecar.exists():
        raise FileNotFoundError("cache or sidecar missing: {}".format(inp))
    with np.load(inp, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    features = np.asarray(arrays.get("features"), np.float32)
    if features.ndim != 2 or features.shape[1] != 256:
        raise ValueError("expected [N,256] features, got {}".format(features.shape))
    if not np.isfinite(features).all():
        raise ValueError("input features contain non-finite values")
    if args.method in ("frame_center", "frame_center_diag"):
        if "frame_ids" not in arrays or "offsets" not in arrays:
            raise ValueError("frame-wise transform requires frame_ids and offsets in cache")
        image_frames = np.asarray(arrays["frame_ids"], np.int64)
        offsets = np.asarray(arrays["offsets"], np.int64)
        if offsets.shape != (len(image_frames) + 1,) or int(offsets[-1]) != len(features):
            raise ValueError("cache frame_ids/offsets do not align with features")
        row_frames = np.repeat(image_frames, np.diff(offsets))
        transformed, details = transform_framewise(
            features, row_frames, args.method, args.eps)
    else:
        transformed, details = transform(features, args.method, args.strength, args.eps)
    arrays["features"] = transformed
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, out)
    source_meta = json.loads(sidecar.read_text())
    meta = {
        "schema": "generaltrack-got-jepa-cache-unsupervised-transform-v1",
        "status": "complete",
        "input_cache": str(inp),
        "input_cache_sha256": sha256(inp),
        "input_sidecar": str(sidecar),
        "input_meta": source_meta,
        "transform": details,
        "method": args.method,
        "feature_dim": 256,
        "feature_dtype": "float16",
        "fit_data": "candidate feature matrix in this cache only",
        "target_unlabeled_fit": True,
        "gt_read": False,
        "target_candidate_in_predictor": False,
        "num_rows": int(len(features)),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }
    out.with_suffix(out.suffix + ".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
