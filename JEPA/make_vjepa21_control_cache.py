#!/usr/bin/env python3
"""Create detector-preserving controls for a V-JEPA feature cache.

The controls keep image ids, offsets, and detector rows unchanged.  They are
only for attribution audits and never read annotations or ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("global_permutation", "zero"),
        default="global_permutation",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float32)
        offsets = np.asarray(cache["offsets"])
        image_ids = np.asarray(cache["image_ids"])
        extra = {k: np.asarray(cache[k]) for k in cache.files
                 if k not in {"features", "offsets", "image_ids"}}

    if features.ndim != 2 or offsets.ndim != 1 or image_ids.ndim != 1:
        raise ValueError("cache must contain features [N,D], offsets [M+1], image_ids [M]")
    if len(features) != int(offsets[-1]) or len(offsets) != len(image_ids) + 1:
        raise ValueError("cache offsets/image_ids do not describe features")

    out_features = np.zeros_like(features)
    if args.mode == "zero":
        # A zero vector is rejected by the tracker as missing, so this is a
        # strict no-feature control with exactly the same detector rows.
        pass
    else:
        rng = np.random.default_rng(args.seed)
        permutation = rng.permutation(len(features))
        out_features[:] = features[permutation]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=out_features,
        offsets=offsets,
        image_ids=image_ids,
        **extra,
    )
    sidecar = {
        "schema": "vjepa21-control-cache-v1",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "mode": args.mode,
        "seed": args.seed if args.mode != "zero" else None,
        "features_shape": list(out_features.shape),
        "detector_rows_unchanged": True,
        "gt_read": False,
        "future_frames_read": False,
        "purpose": "JEPA attribution control; not a primary result",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
