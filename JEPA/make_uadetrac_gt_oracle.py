#!/usr/bin/env python3
"""Create isolated UA-DETRAC GT-detection oracle inputs.

The generated detection files are deliberately kept outside the fair
source-detector inputs.  They contain one detector-style row for every
non-crowd annotation, with score 1.0 and the original category id.  The
annotation file used for evaluation is not modified.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_ANN_ROOT = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/ua_detrac/annotations"
)
DEFAULT_OUT_ROOT = Path(
    "/data1/liuyeqiang/JEPA_MOT_resume_20261012/GeneralTrack-main/outputs/oracle_gt/ua_detrac"
)
DOMAINS = ("cloudy", "night", "rainy")


def _read(path: Path):
    with path.open("r") as handle:
        return json.load(handle)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle)


def make_oracle_records(annotation):
    records_by_image = defaultdict(list)
    for ann in annotation.get("annotations", []):
        if int(ann.get("iscrowd", 0)):
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            raise ValueError("Annotation {} has no valid xywh bbox".format(ann.get("id")))
        records_by_image[int(ann["image_id"])].append(
            {
                "image_id": int(ann["image_id"]),
                "bbox": [float(x) for x in bbox[:4]],
                "score": 1.0,
                "category_id": int(ann.get("category_id", 1)),
            }
        )
    records = []
    for image in sorted(annotation.get("images", []), key=lambda item: int(item["id"])):
        records.extend(records_by_image.get(int(image["id"]), []))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    args = parser.parse_args()

    summary = []
    for domain in args.domains:
        ann_path = args.annotation_root / "box_track_{}_all_cocofmt.json".format(domain)
        annotation = _read(ann_path)
        records = make_oracle_records(annotation)
        out_dir = args.output_root / domain
        det_path = out_dir / "detections_coco.json"
        _write(det_path, records)
        metadata = {
            "schema": "ua-detrac-gt-detection-oracle-v1",
            "domain": domain,
            "annotation_source": str(ann_path),
            "detection_file": str(det_path),
            "num_images": len(annotation.get("images", [])),
            "num_annotations": len(records),
            "score": 1.0,
            "bbox_format": "xywh",
            "uses_crowd_annotations": False,
        }
        _write(out_dir / "metadata.json", metadata)
        summary.append(metadata)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
