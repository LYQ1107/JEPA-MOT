#!/usr/bin/env python3
"""Read-only inventory of the local MOT17 and DanceTrack assets.

The script intentionally writes only the requested JSON output.  It does not
touch the source datasets or any tracker repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List


MOT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17")
DANCE_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def read_ini(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("["):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def mot_inventory() -> Dict[str, object]:
    splits: Dict[str, List[Dict[str, object]]] = {}
    for split in ("train", "test"):
        rows: List[Dict[str, object]] = []
        root = MOT_ROOT / split
        for seq in sorted(p for p in root.iterdir() if p.is_dir()):
            ini = read_ini(seq / "seqinfo.ini")
            gt = seq / "gt" / "gt.txt"
            det = seq / "det" / "det.txt"
            rows.append(
                {
                    "name": seq.name,
                    "path": str(seq),
                    "frames_from_seqinfo": int(ini.get("seqLength", 0) or 0),
                    "frames_images": image_count(seq / "img1"),
                    "width": int(ini.get("imWidth", 0) or 0),
                    "height": int(ini.get("imHeight", 0) or 0),
                    "fps": float(ini.get("frameRate", 0) or 0),
                    "gt_file": str(gt) if gt.exists() else None,
                    "gt_rows": count_lines(gt),
                    "det_file": str(det) if det.exists() else None,
                    "det_rows": count_lines(det),
                    "det_sha256": sha256_file(det) if det.exists() else None,
                }
            )
        splits[split] = rows
    return {"root": str(MOT_ROOT), "splits": splits}


def dance_inventory() -> Dict[str, object]:
    splits: Dict[str, List[Dict[str, object]]] = {}
    for split in ("train", "val", "test"):
        rows: List[Dict[str, object]] = []
        root = DANCE_ROOT / split
        if root.exists():
            for seq in sorted(p for p in root.iterdir() if p.is_dir()):
                gt = seq / "gt" / "gt.txt"
                rows.append(
                    {
                        "name": seq.name,
                        "path": str(seq),
                        "frames_images": image_count(seq / "img1"),
                        "gt_file": str(gt) if gt.exists() else None,
                        "gt_rows": count_lines(gt),
                    }
                )
        splits[split] = rows
    ann = {}
    for name in ("train.json", "val.json", "test.json"):
        p = DANCE_ROOT / "annotations" / name
        ann[name] = {
            "path": str(p),
            "exists": p.exists(),
            "bytes": p.stat().st_size if p.exists() else 0,
            "sha256": sha256_file(p) if p.exists() else None,
        }
    return {"root": str(DANCE_ROOT), "splits": splits, "annotations": ann}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    payload = {
        "schema": "mot-dance-inventory-v1",
        "mot17": mot_inventory(),
        "dancetrack": dance_inventory(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "mot17_train_sequences": len(payload["mot17"]["splits"]["train"]),
        "mot17_test_sequences": len(payload["mot17"]["splits"]["test"]),
        "dance_train_sequences": len(payload["dancetrack"]["splits"]["train"]),
        "dance_val_sequences": len(payload["dancetrack"]["splits"]["val"]),
        "dance_test_sequences": len(payload["dancetrack"]["splits"]["test"]),
        "out": str(out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
