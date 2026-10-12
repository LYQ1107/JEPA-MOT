#!/usr/bin/env python3
"""Concatenate aligned frozen JEPA cache features for an ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load(path: Path):
    with np.load(str(path), allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float32)
        offsets = np.asarray(cache["offsets"], dtype=np.int64)
        image_ids = np.asarray(cache["image_ids"], dtype=np.int64)
    if features.ndim != 2 or offsets.shape != (len(image_ids) + 1,):
        raise ValueError(f"invalid cache shape: {path}")
    if offsets[0] != 0 or offsets[-1] != len(features):
        raise ValueError(f"invalid cache offsets: {path}")
    return features, offsets, image_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", required=True)
    parser.add_argument("--predictor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dense-scale", type=float, default=1.0)
    parser.add_argument("--predictor-scale", type=float, default=1.0)
    args = parser.parse_args()

    dense, dense_offsets, dense_ids = _load(Path(args.dense))
    predictor, predictor_offsets, predictor_ids = _load(Path(args.predictor))
    dense_index = {int(image_id): i for i, image_id in enumerate(dense_ids.tolist())}
    dense_rows = []
    for i, image_id in enumerate(predictor_ids.tolist()):
        dense_i = dense_index.get(int(image_id))
        if dense_i is None:
            raise ValueError(f"dense cache is missing image_id={image_id}")
        dense_start, dense_end = int(dense_offsets[dense_i]), int(dense_offsets[dense_i + 1])
        predictor_start, predictor_end = int(predictor_offsets[i]), int(predictor_offsets[i + 1])
        if dense_end - dense_start != predictor_end - predictor_start:
            raise ValueError(f"detection count mismatch for image_id={image_id}")
        dense_rows.append(dense[dense_start:dense_end])
    dense = np.concatenate(dense_rows, axis=0) if dense_rows else np.zeros((0, dense.shape[1]), np.float32)
    dense_offsets = predictor_offsets
    dense_ids = predictor_ids
    dense = dense / np.maximum(np.linalg.norm(dense, axis=1, keepdims=True), 1e-8)
    predictor = predictor / np.maximum(np.linalg.norm(predictor, axis=1, keepdims=True), 1e-8)
    features = np.concatenate(
        [float(args.dense_scale) * dense, float(args.predictor_scale) * predictor],
        axis=1,
    ).astype(np.float32, copy=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output), features=features, offsets=dense_offsets, image_ids=dense_ids
    )
    print({"output": str(output), "shape": list(features.shape),
           "dense_scale": args.dense_scale,
           "predictor_scale": args.predictor_scale})


if __name__ == "__main__":
    main()
