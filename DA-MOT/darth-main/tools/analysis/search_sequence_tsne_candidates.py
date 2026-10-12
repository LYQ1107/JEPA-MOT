#!/usr/bin/env python3
"""Rank single-sequence embedding comparisons and render fixed high-res t-SNE plots."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis import plot_bdd_night_method_vs_baseline_tsne as shared


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "extract", "plot"), default="all")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--ours-config", type=Path, required=True)
    parser.add_argument("--ours-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-name", default="Baseline Single-Target")
    parser.add_argument("--ours-name", default="Ours")
    parser.add_argument("--candidate-videos", default="")
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--num-tracks", type=int, default=8)
    parser.add_argument("--samples-per-track", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=25.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plot-python", default="/home/lwr/anaconda3/bin/python3")
    parser.add_argument("--reuse-selection-from", type=Path)
    parser.add_argument("--reuse-ours-features-from", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require_inputs(args):
    paths = [
        args.annotation,
        args.image_root,
        args.baseline_config,
        args.baseline_checkpoint,
        args.ours_config,
        args.ours_checkpoint,
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def make_selection(args):
    selection_path = args.output_dir / "selected_samples.json"
    if selection_path.exists() and not args.force:
        with selection_path.open() as handle:
            return json.load(handle)

    with args.annotation.open() as handle:
        data = json.load(handle)
    images = {int(item["id"]): item for item in data["images"]}
    category_ids = {
        int(item["id"]) for item in data["categories"] if item["name"] == args.category
    }
    if len(category_ids) != 1:
        raise RuntimeError("Expected one category named {!r}".format(args.category))

    tracks = defaultdict(list)
    for annotation in data["annotations"]:
        if int(annotation["category_id"]) not in category_ids:
            continue
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        image = images[int(annotation["image_id"])]
        video_id = annotation.get("video_id", image.get("video_id"))
        instance_id = annotation.get("instance_id", annotation.get("track_id"))
        if video_id is None or instance_id is None:
            raise RuntimeError("Annotation is missing video or identity information")
        item = dict(annotation)
        item["frame_id"] = int(image.get("frame_id", 0))
        tracks[(int(video_id), int(instance_id))].append(item)

    valid_per_video = defaultdict(list)
    for key, annotations in tracks.items():
        if len(annotations) >= args.samples_per_track:
            valid_per_video[key[0]].append((key, annotations))

    if args.candidate_videos:
        candidate_videos = [int(value) for value in args.candidate_videos.split(",")]
    else:
        counts = Counter({video: len(items) for video, items in valid_per_video.items()})
        candidate_videos = [video for video, _ in counts.most_common(args.num_candidates)]
    candidate_videos = candidate_videos[: args.num_candidates]

    video_names = {int(item["id"]): item.get("name", str(item["id"])) for item in data.get("videos", [])}
    candidates = []
    samples = []
    for video_id in candidate_videos:
        valid = valid_per_video[video_id]
        valid.sort(key=lambda item: (-len(item[1]), item[0][1]))
        selected = valid[: args.num_tracks]
        if len(selected) < args.num_tracks:
            raise RuntimeError("Video {} has only {} valid tracks".format(video_id, len(selected)))

        first_file = images[int(selected[0][1][0]["image_id"])]["file_name"]
        sequence_name = video_names.get(video_id) or Path(first_file).parts[-2]
        if sequence_name == str(video_id):
            sequence_name = Path(first_file).parts[-2]
        candidate = {
            "candidate_index": len(candidates),
            "video_id": video_id,
            "sequence_name": sequence_name,
            "sample_start": len(samples),
        }
        for identity_index, (key, annotations) in enumerate(selected):
            annotations.sort(key=lambda item: (item["frame_id"], item["image_id"]))
            indices = np.linspace(0, len(annotations) - 1, args.samples_per_track).round().astype(int)
            for sample_in_track, index in enumerate(indices.tolist()):
                annotation = annotations[index]
                image = images[int(annotation["image_id"])]
                image_path = args.image_root / image["file_name"]
                if not image_path.is_file():
                    raise FileNotFoundError(str(image_path))
                samples.append(
                    {
                        "sample_index": len(samples),
                        "candidate_index": candidate["candidate_index"],
                        "identity_index": identity_index,
                        "identity_label": "ID{}".format(key[1]),
                        "video_id": video_id,
                        "instance_id": key[1],
                        "sample_in_track": sample_in_track,
                        "image_id": int(annotation["image_id"]),
                        "frame_id": int(annotation["frame_id"]),
                        "file_name": image["file_name"],
                        "bbox_xywh": [float(value) for value in annotation["bbox"]],
                    }
                )
        candidate["sample_end"] = len(samples)
        candidates.append(candidate)

    payload = {
        "dataset": args.dataset,
        "domain": args.domain,
        "category": args.category,
        "seed": args.seed,
        "num_tracks": args.num_tracks,
        "samples_per_track": args.samples_per_track,
        "annotation": str(args.annotation.resolve()),
        "image_root": str(args.image_root.resolve()),
        "candidates": candidates,
        "samples": samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print("Saved {} candidates and {} samples to {}".format(len(candidates), len(samples), selection_path))
    return payload


def projection(features_a, features_b, seed, perplexity):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    all_features = np.vstack([features_a, features_b])
    dimensions = min(50, all_features.shape[1], all_features.shape[0] - 1)
    reduced = PCA(n_components=dimensions, random_state=seed).fit_transform(all_features)
    return TSNE(
        n_components=2,
        random_state=seed,
        perplexity=min(perplexity, len(reduced) - 1),
        init="pca",
        learning_rate="auto",
        max_iter=2500,
    ).fit_transform(reduced)


def render_candidate(args, selection, candidate, baseline_features, ours_features):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    start, end = candidate["sample_start"], candidate["sample_end"]
    samples = selection["samples"][start:end]
    labels = np.asarray([item["identity_index"] for item in samples], dtype=np.int64)
    baseline = baseline_features[start:end]
    ours = ours_features[start:end]
    baseline_metrics = shared.high_dimensional_metrics(baseline, labels)
    ours_metrics = shared.high_dimensional_metrics(ours, labels)
    points = projection(baseline, ours, args.seed, args.perplexity)
    count = len(baseline)
    baseline_points, ours_points = points[:count], points[count:]

    score = (
        2.0 * (ours_metrics["leave_one_out_1nn_identity_accuracy"] - baseline_metrics["leave_one_out_1nn_identity_accuracy"])
        + (ours_metrics["silhouette_cosine"] - baseline_metrics["silhouette_cosine"])
        + 0.25 * np.log(max(ours_metrics["inter_intra_ratio"], 1e-12) / max(baseline_metrics["inter_intra_ratio"], 1e-12))
    )

    colors = sns.color_palette("tab10", args.num_tracks)
    identity_names = {item["identity_index"]: item["identity_label"] for item in samples}
    padding = 3.0
    xmin, xmax = points[:, 0].min() - padding, points[:, 0].max() + padding
    ymin, ymax = points[:, 1].min() - padding, points[:, 1].max() + padding
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    panels = [
        (axes[0], baseline_points, args.baseline_name, baseline_metrics),
        (axes[1], ours_points, args.ours_name, ours_metrics),
    ]
    for axis, panel_points, title, metrics in panels:
        for identity in range(args.num_tracks):
            mask = labels == identity
            axis.scatter(
                panel_points[mask, 0], panel_points[mask, 1], s=105, alpha=0.86,
                color=colors[identity], edgecolors="black", linewidth=0.75,
                label=identity_names[identity], rasterized=False,
            )
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            "{}\nSilhouette={:.3f} | 1-NN={:.1%} | Inter/Intra={:.3f}".format(
                title, metrics["silhouette_cosine"],
                metrics["leave_one_out_1nn_identity_accuracy"], metrics["inter_intra_ratio"]
            ), fontsize=16, fontweight="bold",
        )
        axis.set_xlabel("Shared t-SNE Dimension 1", fontsize=13)
        axis.set_ylabel("Shared t-SNE Dimension 2", fontsize=13)
        axis.grid(True, alpha=0.22)
        axis.legend(loc="best", fontsize=9, framealpha=0.9, ncol=2)

    fig.suptitle(
        "{} / {} / {} - sequence {}\n{} identities x {} frames, same GT boxes and shared projection".format(
            args.dataset, args.domain, args.category, candidate["sequence_name"],
            args.num_tracks, args.samples_per_track,
        ), fontsize=18, fontweight="bold",
    )
    fig.tight_layout()
    stem = "candidate_{:02d}_video{}_{}".format(
        candidate["candidate_index"] + 1, candidate["video_id"],
        str(candidate["sequence_name"]).replace("/", "_").replace(" ", "_")
    )
    png_path = args.output_dir / (stem + "_tsne.png")
    pdf_path = args.output_dir / (stem + "_tsne.pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    points_path = args.output_dir / (stem + "_points.csv")
    with points_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "identity", "sample", "x", "y"])
        for method, panel_points in (("baseline", baseline_points), ("ours", ours_points)):
            for sample, point in zip(samples, panel_points):
                writer.writerow([method, sample["identity_label"], sample["sample_in_track"], float(point[0]), float(point[1])])

    return {
        "video_id": candidate["video_id"],
        "sequence_name": candidate["sequence_name"],
        "selection_score": float(score),
        "baseline": baseline_metrics,
        "ours": ours_metrics,
        "delta_1nn": ours_metrics["leave_one_out_1nn_identity_accuracy"] - baseline_metrics["leave_one_out_1nn_identity_accuracy"],
        "delta_silhouette": ours_metrics["silhouette_cosine"] - baseline_metrics["silhouette_cosine"],
        "ratio_gain": ours_metrics["inter_intra_ratio"] / max(baseline_metrics["inter_intra_ratio"], 1e-12) - 1.0,
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "points": str(points_path.resolve()),
    }


def plot_all(args):
    with (args.output_dir / "selected_samples.json").open() as handle:
        selection = json.load(handle)
    baseline = np.load(args.output_dir / "baseline_features.npz")["features"]
    ours = np.load(args.output_dir / "ours_features.npz")["features"]
    if baseline.shape != ours.shape or len(baseline) != len(selection["samples"]):
        raise RuntimeError("Feature caches do not match the selection")

    results = [render_candidate(args, selection, candidate, baseline, ours) for candidate in selection["candidates"]]
    results.sort(key=lambda item: item["selection_score"], reverse=True)
    payload = {
        "dataset": args.dataset,
        "domain": args.domain,
        "category": args.category,
        "selection_rule": "Ranked in original 256D space; t-SNE geometry is not used for selection",
        "baseline_checkpoint": str(args.baseline_checkpoint.resolve()),
        "ours_checkpoint": str(args.ours_checkpoint.resolve()),
        "results": results,
    }
    with (args.output_dir / "candidate_ranking.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    with (args.output_dir / "candidate_ranking.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "video_id", "sequence", "score", "baseline_1nn", "ours_1nn", "delta_1nn", "baseline_silhouette", "ours_silhouette", "delta_silhouette", "ratio_gain", "png"])
        for rank, result in enumerate(results, 1):
            writer.writerow([rank, result["video_id"], result["sequence_name"], result["selection_score"], result["baseline"]["leave_one_out_1nn_identity_accuracy"], result["ours"]["leave_one_out_1nn_identity_accuracy"], result["delta_1nn"], result["baseline"]["silhouette_cosine"], result["ours"]["silhouette_cosine"], result["delta_silhouette"], result["ratio_gain"], result["png"]])
    print(json.dumps(results, indent=2))


def invoke_plotter(args):
    command = [args.plot_python, str(Path(__file__).resolve()), "--stage", "plot"]
    for name in ("dataset", "domain", "category", "annotation", "image_root", "output_dir", "baseline_config", "baseline_checkpoint", "ours_config", "ours_checkpoint", "baseline_name", "ours_name", "num_candidates", "num_tracks", "samples_per_track", "seed", "perplexity", "dpi"):
        value = getattr(args, name)
        command.extend(["--" + name.replace("_", "-"), str(value)])
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    subprocess.check_call(command, cwd=str(ROOT), env=environment)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.stage in ("all", "extract"):
        require_inputs(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        selection_path = args.output_dir / "selected_samples.json"
        if args.reuse_selection_from and not selection_path.exists():
            shutil.copy2(args.reuse_selection_from, selection_path)
            print("Reused sample selection from {}".format(args.reuse_selection_from))
        ours_features_path = args.output_dir / "ours_features.npz"
        if args.reuse_ours_features_from and not ours_features_path.exists():
            shutil.copy2(args.reuse_ours_features_from, ours_features_path)
            print("Reused ours features from {}".format(args.reuse_ours_features_from))
        selection = make_selection(args)
        shared.extract_features("baseline", args.baseline_config, args.baseline_checkpoint, selection["samples"], args)
        shared.extract_features("ours", args.ours_config, args.ours_checkpoint, selection["samples"], args)
    if args.stage == "plot":
        plot_all(args)
    elif args.stage == "all":
        invoke_plotter(args)


if __name__ == "__main__":
    main()
