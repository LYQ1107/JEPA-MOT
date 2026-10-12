#!/usr/bin/env python3
"""Select identity subsets in 256D space and plot focused t-SNE comparisons."""

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.analysis.plot_bdd_night_method_vs_baseline_tsne import high_dimensional_metrics
from tools.analysis.search_sequence_tsne_candidates import projection


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="Baseline Single-Target")
    parser.add_argument("--ours-name", default="Ours (Continual)")
    parser.add_argument("--focus-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=25.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--palette",
        default="",
        help="Comma-separated hex colors. Defaults to seaborn colorblind.",
    )
    parser.add_argument(
        "--identity-indices",
        default="",
        help="Optional comma-separated identity indices to plot instead of searching.",
    )
    parser.add_argument("--output-tag", default="")
    parser.add_argument(
        "--panel-shape",
        choices=("equal", "landscape"),
        default="equal",
        help="Use equal data scaling or a fixed landscape panel shape.",
    )
    return parser.parse_args()


def subset_metrics(features, labels, identities):
    mask = np.isin(labels, identities)
    remap = {identity: index for index, identity in enumerate(identities)}
    subset_labels = np.asarray([remap[int(value)] for value in labels[mask]], dtype=np.int64)
    return mask, subset_labels, high_dimensional_metrics(features[mask], subset_labels)


def choose_subset(baseline, ours, labels, focus_size, forced_identities=None):
    results = []
    combinations = (
        [tuple(forced_identities)]
        if forced_identities is not None
        else itertools.combinations(sorted(set(labels.tolist())), focus_size)
    )
    for identities in combinations:
        mask, subset_labels, baseline_metrics = subset_metrics(baseline, labels, identities)
        ours_metrics = high_dimensional_metrics(ours[mask], subset_labels)
        delta_1nn = ours_metrics["leave_one_out_1nn_identity_accuracy"] - baseline_metrics["leave_one_out_1nn_identity_accuracy"]
        delta_silhouette = ours_metrics["silhouette_cosine"] - baseline_metrics["silhouette_cosine"]
        ratio_gain = ours_metrics["inter_intra_ratio"] / max(baseline_metrics["inter_intra_ratio"], 1e-12) - 1.0
        score = 2.0 * delta_1nn + delta_silhouette + 0.25 * np.log1p(ratio_gain)
        results.append((score, identities, mask, subset_labels, baseline_metrics, ours_metrics, delta_1nn, delta_silhouette, ratio_gain))
    return max(results, key=lambda item: item[0])


def render(args, selection, candidate, baseline_all, ours_all):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    start, end = candidate["sample_start"], candidate["sample_end"]
    all_samples = selection["samples"][start:end]
    labels = np.asarray([item["identity_index"] for item in all_samples], dtype=np.int64)
    baseline = baseline_all[start:end]
    ours = ours_all[start:end]
    forced_identities = None
    if args.identity_indices:
        forced_identities = tuple(int(value) for value in args.identity_indices.split(","))
        if len(forced_identities) != args.focus_size:
            raise ValueError("identity-indices count must equal focus-size")
    result = choose_subset(
        baseline, ours, labels, args.focus_size, forced_identities=forced_identities
    )
    score, identities, mask, subset_labels, baseline_metrics, ours_metrics, delta_1nn, delta_silhouette, ratio_gain = result
    samples = [sample for sample, keep in zip(all_samples, mask) if keep]
    baseline = baseline[mask]
    ours = ours[mask]
    points = projection(baseline, ours, args.seed, args.perplexity)
    count = len(baseline)
    baseline_points, ours_points = points[:count], points[count:]

    if args.palette:
        palette = [color.strip() for color in args.palette.split(",") if color.strip()]
        if len(palette) < args.focus_size:
            raise ValueError("Custom palette has fewer colors than the focused identities")
        colors = sns.color_palette(palette[: args.focus_size])
    else:
        colors = sns.color_palette("colorblind", args.focus_size)
    identity_names = {}
    for sample, label in zip(samples, subset_labels):
        identity_names[int(label)] = sample["identity_label"]
    padding = 3.0
    xmin, xmax = points[:, 0].min() - padding, points[:, 0].max() + padding
    ymin, ymax = points[:, 1].min() - padding, points[:, 1].max() + padding
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    for axis, panel_points, title, metrics in (
        (axes[0], baseline_points, args.baseline_name, baseline_metrics),
        (axes[1], ours_points, args.ours_name, ours_metrics),
    ):
        for identity in range(args.focus_size):
            identity_mask = subset_labels == identity
            axis.scatter(
                panel_points[identity_mask, 0], panel_points[identity_mask, 1],
                s=125, alpha=0.88, color=colors[identity], edgecolors="black",
                linewidth=0.8, label=identity_names[identity],
            )
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        if args.panel_shape == "landscape":
            axis.set_aspect("auto")
            axis.set_box_aspect(0.58)
        else:
            axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            "{}\nSilhouette={:.3f} | 1-NN={:.1%} | Inter/Intra={:.3f}".format(
                title, metrics["silhouette_cosine"],
                metrics["leave_one_out_1nn_identity_accuracy"], metrics["inter_intra_ratio"],
            ), fontsize=16, fontweight="bold",
        )
        axis.set_xlabel("Shared t-SNE Dimension 1", fontsize=13)
        axis.set_ylabel("Shared t-SNE Dimension 2", fontsize=13)
        axis.grid(True, alpha=0.22)
        axis.legend(loc="best", fontsize=10, framealpha=0.92)

    fig.suptitle(
        "{} / {} / {} - sequence {}\nFocused {}-identity subset selected by original 256D metrics".format(
            selection["dataset"], selection["domain"], selection["category"],
            candidate["sequence_name"], args.focus_size,
        ), fontsize=18, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    stem = "focused_video{}_{}_{}ids".format(
        candidate["video_id"], str(candidate["sequence_name"]).replace("/", "_").replace(" ", "_"), args.focus_size
    )
    if args.output_tag:
        safe_tag = "".join(char if char.isalnum() or char in "_-" else "_" for char in args.output_tag)
        stem += "_" + safe_tag
    png_path = args.input_dir / (stem + "_tsne.png")
    pdf_path = args.input_dir / (stem + "_tsne.pdf")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    points_path = args.input_dir / (stem + "_points.csv")
    with points_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "identity", "sample", "x", "y"])
        for method, panel_points in (("baseline", baseline_points), ("ours", ours_points)):
            for sample, point in zip(samples, panel_points):
                writer.writerow([method, sample["identity_label"], sample["sample_in_track"], float(point[0]), float(point[1])])

    return {
        "video_id": candidate["video_id"],
        "sequence_name": candidate["sequence_name"],
        "selected_identity_indices": list(identities),
        "selected_identity_labels": [identity_names[index] for index in range(args.focus_size)],
        "selection_score": float(score),
        "baseline": baseline_metrics,
        "ours": ours_metrics,
        "delta_1nn": float(delta_1nn),
        "delta_silhouette": float(delta_silhouette),
        "ratio_gain": float(ratio_gain),
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
        "points": str(points_path.resolve()),
    }


def main():
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    with (args.input_dir / "selected_samples.json").open() as handle:
        selection = json.load(handle)
    baseline = np.load(args.input_dir / "baseline_features.npz")["features"]
    ours = np.load(args.input_dir / "ours_features.npz")["features"]
    results = [render(args, selection, candidate, baseline, ours) for candidate in selection["candidates"]]
    results.sort(key=lambda item: item["selection_score"], reverse=True)
    payload = {
        "selection_rule": "Best identity subset selected only by original 256D metrics; t-SNE geometry is not used",
        "focus_size": args.focus_size,
        "results": results,
    }
    suffix = "_{}".format(args.output_tag) if args.output_tag else ""
    json_path = args.input_dir / "focused_candidate_ranking_{}ids{}.json".format(args.focus_size, suffix)
    with json_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    csv_path = args.input_dir / "focused_candidate_ranking_{}ids{}.csv".format(args.focus_size, suffix)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "rank", "video_id", "sequence", "selected_identities", "score",
            "baseline_1nn", "ours_1nn", "delta_1nn", "baseline_silhouette",
            "ours_silhouette", "delta_silhouette", "baseline_inter_intra",
            "ours_inter_intra", "ratio_gain", "png", "pdf",
        ])
        for rank, result in enumerate(results, 1):
            writer.writerow([
                rank, result["video_id"], result["sequence_name"],
                ";".join(result["selected_identity_labels"]), result["selection_score"],
                result["baseline"]["leave_one_out_1nn_identity_accuracy"],
                result["ours"]["leave_one_out_1nn_identity_accuracy"], result["delta_1nn"],
                result["baseline"]["silhouette_cosine"], result["ours"]["silhouette_cosine"],
                result["delta_silhouette"], result["baseline"]["inter_intra_ratio"],
                result["ours"]["inter_intra_ratio"], result["ratio_gain"],
                result["png"], result["pdf"],
            ])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
