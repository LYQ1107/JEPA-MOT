import argparse
import os

from generaltrack_damot_utils import (
    DEFAULT_CONFIG,
    GeneralTrackInputError,
    get_dataset_cfg,
    list_domains,
    load_dataset_config,
    repo_path,
    resolve_image_root,
    resolve_annotation,
    resolve_detection,
    write_csv,
)


def make_parser():
    parser = argparse.ArgumentParser("Inspect DAMOT/DARTH inputs for GeneralTrack")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--datasets", default="all", help="Space/comma separated list or all")
    parser.add_argument("--domains", default="all", help="Space/comma separated list or all")
    parser.add_argument("--det-source", default=os.environ.get("DET_SOURCE", "source_detector"))
    parser.add_argument("--output", default="outputs/generaltrack_input_check.csv")
    return parser


def split_arg(value):
    if value == "all":
        return "all"
    return [v for v in value.replace(",", " ").split() if v]


def main():
    args = make_parser().parse_args()
    config = load_dataset_config(args.config)
    datasets = config.get("datasets", {})
    selected_datasets = sorted(datasets) if args.datasets == "all" else split_arg(args.datasets)

    rows = []
    missing_messages = []
    for dataset in selected_datasets:
        try:
            dataset_cfg = get_dataset_cfg(config, dataset)
            domains = list_domains(dataset_cfg, args.domains if args.domains == "all" else split_arg(args.domains))
        except GeneralTrackInputError as exc:
            missing_messages.append("{}: {}".format(dataset, exc))
            continue

        source_ann = repo_path(dataset_cfg.get("source_ann", ""))
        for domain in domains:
            image_root = resolve_image_root(dataset_cfg, dataset, domain)
            ann = resolve_annotation(dataset_cfg, dataset, domain)
            det = resolve_detection(dataset_cfg, dataset, domain, args.det_source)
            row = {
                "dataset": dataset,
                "source_domain": dataset_cfg.get("source", ""),
                "target_domain": domain,
                "annotation": str(ann) if ann else "",
                "annotation_exists": bool(ann and ann.exists()),
                "source_ann": str(source_ann),
                "source_ann_exists": source_ann.exists(),
                "image_root": str(image_root),
                "image_root_exists": image_root.exists(),
                "det_source": args.det_source,
                "detection": str(det) if det else "",
                "detection_exists": bool(det and det.exists()),
            }
            rows.append(row)
            if not row["annotation_exists"]:
                missing_messages.append("{}:{} missing target annotation".format(dataset, domain))
            if not row["image_root_exists"]:
                missing_messages.append("{}:{} missing image_root {}".format(dataset, domain, image_root))
            if not row["detection_exists"]:
                missing_messages.append(
                    "{}:{} missing detection result for det_source={} under {}".format(
                        dataset,
                        domain,
                        args.det_source,
                        dataset_cfg.get("detection_roots", {}).get(args.det_source, ""),
                    )
                )

    fields = [
        "dataset",
        "source_domain",
        "target_domain",
        "annotation",
        "annotation_exists",
        "source_ann",
        "source_ann_exists",
        "image_root",
        "image_root_exists",
        "det_source",
        "detection",
        "detection_exists",
    ]
    write_csv(repo_path(args.output), rows, fields)
    print("Wrote input check: {}".format(repo_path(args.output)))
    if missing_messages:
        print("Missing GeneralTrack inputs:")
        for msg in missing_messages:
            print("  - {}".format(msg))
    else:
        print("All requested GeneralTrack inputs are present.")


if __name__ == "__main__":
    main()
