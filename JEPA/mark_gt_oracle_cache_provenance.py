#!/usr/bin/env python3
"""Mark completed cache sidecars as using the isolated GT detector oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecars", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.sidecars:
        with path.open("r") as handle:
            metadata = json.load(handle)
        metadata["gt_read"] = True
        metadata["gt_detection_oracle"] = True
        metadata["oracle_provenance"] = (
            "Detector rows were generated from non-crowd UA-DETRAC annotations; "
            "this cache is an isolated upper bound and is excluded from fair tables."
        )
        with path.open("w") as handle:
            json.dump(metadata, handle, indent=2)
        print(path)


if __name__ == "__main__":
    main()
