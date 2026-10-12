#!/usr/bin/env python3
"""Build zero-feature JEPA caches with the exact full-cache row alignment."""

import argparse
import gc
from pathlib import Path

import numpy as np


def build(source: Path, target: Path) -> None:
    with np.load(source, allow_pickle=False) as data:
        features = np.asarray(data["features"])
        offsets = np.asarray(data["offsets"], dtype=np.int64)
        image_ids = np.asarray(data["image_ids"], dtype=np.int64)
        null_features = np.zeros(features.shape, dtype=np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, features=null_features, offsets=offsets,
                        image_ids=image_ids)
    print(f"built {target} shape={null_features.shape} images={len(image_ids)}")
    del features, null_features, offsets, image_ids
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--name-template", default="motsynth_{domain}_vjepa21_l384_w16_causalbox_predictor")
    args = parser.parse_args()
    for domain in args.domains:
        stem = args.name_template.format(domain=domain)
        build(args.cache_root / f"{stem}.npz",
              args.cache_root / f"{stem}_geometry_null.npz")


if __name__ == "__main__":
    main()
