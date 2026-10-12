"""Create a video-prefix subset of a detector-aligned V-JEPA cache.

This utility preserves the feature rows and offsets exactly; it is used only
to make a cache agree with the temporary prefix annotation used by smoke
tests.  It does not inspect annotations or ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--num-images", required=True, type=int)
    args = ap.parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be positive")
    with np.load(args.input, allow_pickle=False) as source:
        features = np.asarray(source["features"], dtype=np.float32)
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        image_ids = np.asarray(source["image_ids"], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) != len(image_ids) + 1:
        raise ValueError("invalid cache offsets/image_ids")
    end_image = min(int(args.num_images), len(image_ids))
    end_row = int(offsets[end_image])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features[:end_row],
        offsets=offsets[:end_image + 1],
        image_ids=image_ids[:end_image],
    )
    meta = {
        "schema": "vjepa21-cache-prefix-subset-v1",
        "status": "complete",
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "num_images": int(end_image),
        "num_detections": int(end_row),
        "source_rows_unchanged": True,
        "gt_read": False,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
