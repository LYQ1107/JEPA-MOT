import argparse
import os
from pathlib import Path

from generaltrack_damot_utils import (
    DEFAULT_CONFIG,
    GeneralTrackInputError,
    attach_detections,
    category_names,
    get_dataset_cfg,
    keep_detection_mapped_frames,
    load_dataset_config,
    normalize_cocomot_annotations,
    read_json,
    repo_path,
    resolve_annotation,
    resolve_detection,
    resolve_image_root,
    write_json,
)


def make_parser():
    parser = argparse.ArgumentParser("Convert COCO-MOT annotations and detections to GeneralTrack input")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--det-file", default=None)
    parser.add_argument("--det-dir", default=None)
    parser.add_argument("--det-source", default=os.environ.get("DET_SOURCE", "source_detector"))
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--drop-missing-images",
        action="store_true",
        help="Explicitly remove frames whose image file is missing. The count is recorded in metadata.",
    )
    parser.add_argument(
        "--keep-det-mapped-frames-only",
        action="store_true",
        help="Keep only frames listed in detector image_mapping.json. Useful for sparse MOTSynth detections.",
    )
    parser.add_argument(
        "--allow-oracle-gt",
        action="store_true",
        help="Allow det_source=oracle_gt. Main experiments should not use this.",
    )
    return parser


def main():
    args = make_parser().parse_args()
    if args.det_source == "oracle_gt" and not args.allow_oracle_gt:
        raise GeneralTrackInputError(
            "oracle_gt detections are forbidden for the main baseline. Pass --allow-oracle-gt only for an oracle upper bound."
        )

    config = load_dataset_config(args.config)
    dataset_cfg = get_dataset_cfg(config, args.dataset)
    ann_file = repo_path(args.ann_file) if args.ann_file else resolve_annotation(dataset_cfg, args.dataset, args.domain)
    if not ann_file or not ann_file.exists():
        raise GeneralTrackInputError(
            "Missing annotation for {}:{}; checked config patterns and annotation_root.".format(
                args.dataset, args.domain
            )
        )
    if args.det_file and args.det_dir:
        raise GeneralTrackInputError("Use either --det-file or --det-dir, not both.")
    det_path = None
    if args.det_file:
        det_path = repo_path(args.det_file)
    elif args.det_dir:
        det_path = repo_path(args.det_dir)
    else:
        det_path = resolve_detection(dataset_cfg, args.dataset, args.domain, args.det_source)
    if not det_path or not det_path.exists():
        raise GeneralTrackInputError(
            "Missing detection result for {}:{} det_source={}. Generate source-detector detections first; GT boxes are not used.".format(
                args.dataset, args.domain, args.det_source
            )
        )

    image_root = repo_path(args.image_root) if args.image_root else resolve_image_root(dataset_cfg, args.dataset, args.domain)
    if not image_root.exists():
        raise GeneralTrackInputError("Missing image_root: {}".format(image_root))

    output_dir = repo_path(
        args.output_dir
        or Path("datasets") / "generaltrack_prepared" / args.dataset / args.domain
    )
    data = read_json(ann_file)
    prepared, norm_stats = normalize_cocomot_annotations(
        data,
        image_root=image_root,
        max_videos=args.max_videos,
        drop_missing_images=args.drop_missing_images,
    )
    det_stats = attach_detections(prepared, det_path)
    mapped_frame_stats = keep_detection_mapped_frames(
        prepared, det_stats.get("covered_file_names")
    ) if args.keep_det_mapped_frames_only else {
        "enabled": False,
        "dropped_unmapped_frames": [],
        "num_detections": det_stats["num_detections"],
        "missing_detections": det_stats["missing_detections"],
    }

    ann_out = output_dir / "annotations" / "val.json"
    metadata = {
        "dataset": args.dataset,
        "domain": args.domain,
        "source_domain": dataset_cfg.get("source", ""),
        "protocol": "source-only-no-adaptation",
        "det_source": args.det_source,
        "ann_file": str(ann_file),
        "det_path": str(det_path),
        "image_root": str(image_root),
        "annotation_file": str(ann_out),
        "num_videos": len(prepared["videos"]),
        "num_frames": len(prepared["images"]),
        "num_detections": mapped_frame_stats["num_detections"],
        "categories": category_names(prepared),
        "missing_images": norm_stats["missing_images"],
        "drop_missing_images": args.drop_missing_images,
        "missing_detections": mapped_frame_stats["missing_detections"],
        "det_image_mapping": {
            "num_covered_frames": det_stats.get("num_covered_frames"),
            "keep_mapped_frames_only": mapped_frame_stats["enabled"],
            "num_dropped_unmapped_frames": len(mapped_frame_stats["dropped_unmapped_frames"]),
            "dropped_unmapped_frames": mapped_frame_stats["dropped_unmapped_frames"],
        },
        "field_mappings": {
            "track_id_fields": norm_stats["track_id_fields"],
            "frame_id_offset": norm_stats["frame_id_offset"],
        },
    }

    write_json(ann_out, prepared)
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "conversion_stats.json", metadata)

    print("Prepared GeneralTrack input: {}".format(output_dir))
    print("  annotation: {}".format(ann_out))
    print("  videos: {}".format(metadata["num_videos"]))
    print("  frames: {}".format(metadata["num_frames"]))
    print("  detections: {}".format(metadata["num_detections"]))
    print("  categories: {}".format(", ".join(metadata["categories"])))
    print("  missing_images: {}".format(len(metadata["missing_images"])))
    print("  missing_detections: {}".format(len(metadata["missing_detections"])))
    print("  det_mapping_covered_frames: {}".format(metadata["det_image_mapping"]["num_covered_frames"]))
    print("  dropped_unmapped_frames: {}".format(metadata["det_image_mapping"]["num_dropped_unmapped_frames"]))


if __name__ == "__main__":
    main()
