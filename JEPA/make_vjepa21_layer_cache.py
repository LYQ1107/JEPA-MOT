#!/usr/bin/env python3
"""Project an official hierarchical cache to one frozen V-JEPA layer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser("slice official V-JEPA hierarchical cache")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--layer", type=int, required=True, choices=[0, 1, 2, 3])
    args = ap.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    with np.load(source, allow_pickle=False) as z:
        features = np.asarray(z["features"], dtype=np.float32)
        offsets = np.asarray(z["offsets"], dtype=np.int64)
        image_ids = np.asarray(z["image_ids"], dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != 4096:
        raise ValueError("expected [N,4096] hierarchical concat cache, got {}".format(features.shape))
    start = int(args.layer) * 1024
    projected = features[:, start:start + 1024]
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-8)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), features=projected.astype(np.float32), offsets=offsets, image_ids=image_ids)
    source_meta_path = source.with_suffix(source.suffix + ".json")
    source_meta = json.loads(source_meta_path.read_text()) if source_meta_path.is_file() else {}
    meta = {
        "schema": "generaltrack-official-vjepa21-hierarchical-layer-cache-v1",
        "status": "complete",
        "source_cache": str(source),
        "source_cache_sha256": sha256(source),
        "source_cache_meta": source_meta,
        "feature_shape": [int(x) for x in projected.shape],
        "feature_layers": "hierarchical_single_layer",
        "selected_layer_index": int(args.layer),
        "layer_order": [5, 11, 17, 23],
        "feature_dim": 1024,
        "causal": bool(source_meta.get("causal", True)),
        "future_frames_read": bool(source_meta.get("future_frames_read", False)),
        "gt_read": False,
        "predictor_used": False,
        "command": " ".join(__import__("sys").argv),
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
