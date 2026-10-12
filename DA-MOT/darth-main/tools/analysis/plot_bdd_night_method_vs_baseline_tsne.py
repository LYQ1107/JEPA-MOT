#!/usr/bin/env python3
"""Compare BDD100K-night car tracking embeddings with a shared t-SNE.

The extraction stage must run in the DARTH environment. The plotting stage
uses the base Python environment because scikit-learn is not installed in the
legacy DARTH environment. ``--stage all`` handles that handoff automatically.
"""

import argparse
import csv
import gc
import json
import os
import random
import subprocess
from collections import defaultdict
from pathlib import Path

# Avoid severe BLAS/OpenMP oversubscription on the shared experiment server.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANN = ROOT / "data/bdd_da/annotations/box_track_20/box_track_night_all_cocofmt.json"
DEFAULT_IMAGE_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/BDD100K_night")
DEFAULT_OUT = ROOT / "work_dirs/visualization/bdd_night_tsne"

DEFAULT_BASELINE_CONFIG = (
    ROOT
    / "work_dirs/bdd_4domain/task1_single_target/tta_clear_to_night"
    / "darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_night"
    / "darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_night.py"
)
DEFAULT_BASELINE_CKPT = DEFAULT_BASELINE_CONFIG.parent / "epoch_10.pth"

DEFAULT_OURS_CONFIG = (
    ROOT
    / "work_dirs/bdd_4domain/tri_module_pca_variants"
    / "spaP1_causalProtoShrink_roiV1Weak_4domain_5e"
    / "episode/adapt/stage04_after_night"
    / "darth_qdtrack_faster-rcnn_r50_fpn_12e_bdd_night.py"
)
DEFAULT_OURS_CKPT = DEFAULT_OURS_CONFIG.parent / "epoch_5.pth"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "extract", "plot"), default="all")
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANN)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--ours-config", type=Path, default=DEFAULT_OURS_CONFIG)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_OURS_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-tracks", type=int, default=12)
    parser.add_argument("--samples-per-track", type=int, default=15)
    parser.add_argument("--max-tracks-per-video", type=int, default=2)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--plot-python", default="/home/lwr/anaconda3/bin/python3")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_inputs(args):
    required = [
        args.annotation,
        args.image_root,
        args.baseline_config,
        args.baseline_checkpoint,
        args.ours_config,
        args.ours_checkpoint,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def select_samples(args):
    selection_path = args.output_dir / "selected_samples.json"
    if selection_path.exists() and not args.force:
        with selection_path.open() as handle:
            payload = json.load(handle)
        expected = (args.num_tracks, args.samples_per_track, args.seed)
        actual = (payload["num_tracks"], payload["samples_per_track"], payload["seed"])
        if actual != expected:
            raise RuntimeError(
                "Existing selected_samples.json uses different sampling settings; "
                "pass --force to replace it."
            )
        return payload["samples"]

    with args.annotation.open() as handle:
        data = json.load(handle)
    image_map = {int(item["id"]): item for item in data["images"]}
    car_ids = {int(item["id"]) for item in data["categories"] if item["name"] == "car"}
    if len(car_ids) != 1:
        raise RuntimeError("Expected exactly one car category, found {}".format(sorted(car_ids)))

    tracks = defaultdict(list)
    for ann in data["annotations"]:
        if int(ann["category_id"]) not in car_ids or int(ann.get("iscrowd", 0)) != 0:
            continue
        instance_id = ann.get("instance_id", ann.get("track_id"))
        if instance_id is None:
            raise RuntimeError("Car annotation has neither instance_id nor track_id")
        key = (int(ann["video_id"]), int(instance_id))
        item = dict(ann)
        item["frame_id"] = int(image_map[int(ann["image_id"])]["frame_id"])
        tracks[key].append(item)

    valid = [(key, anns) for key, anns in tracks.items() if len(anns) >= args.samples_per_track]
    valid.sort(key=lambda item: (-len(item[1]), item[0][0], item[0][1]))
    selected = []
    video_counts = defaultdict(int)
    for key, anns in valid:
        if video_counts[key[0]] >= args.max_tracks_per_video:
            continue
        selected.append((key, anns))
        video_counts[key[0]] += 1
        if len(selected) == args.num_tracks:
            break
    if len(selected) < args.num_tracks:
        raise RuntimeError(
            "Only {} tracks satisfy the balanced selection rules; need {}".format(
                len(selected), args.num_tracks
            )
        )

    samples = []
    for identity_index, (key, anns) in enumerate(selected):
        anns = sorted(anns, key=lambda item: (item["frame_id"], item["image_id"]))
        indices = np.linspace(0, len(anns) - 1, args.samples_per_track).round().astype(int)
        for sample_in_track, index in enumerate(indices.tolist()):
            ann = anns[index]
            image = image_map[int(ann["image_id"])]
            image_path = args.image_root / image["file_name"]
            if not image_path.is_file():
                raise FileNotFoundError(str(image_path))
            samples.append(
                {
                    "sample_index": len(samples),
                    "identity_index": identity_index,
                    "identity_label": "V{}_ID{}".format(key[0], key[1]),
                    "video_id": key[0],
                    "instance_id": key[1],
                    "sample_in_track": sample_in_track,
                    "image_id": int(ann["image_id"]),
                    "frame_id": int(ann["frame_id"]),
                    "file_name": image["file_name"],
                    "bbox_xywh": [float(value) for value in ann["bbox"]],
                }
            )

    payload = {
        "dataset": "BDD100K",
        "domain": "night",
        "category": "car",
        "seed": args.seed,
        "num_tracks": args.num_tracks,
        "samples_per_track": args.samples_per_track,
        "max_tracks_per_video": args.max_tracks_per_video,
        "annotation": str(args.annotation.resolve()),
        "image_root": str(args.image_root.resolve()),
        "samples": samples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with selection_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print("Saved deterministic sample selection to {}".format(selection_path))
    return samples


def _unwrap_meta(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, dict):
        raise TypeError("Unexpected img_metas structure: {}".format(type(value)))
    return value


def extract_features(label, config_path, checkpoint_path, samples, args):
    output_path = args.output_dir / "{}_features.npz".format(label)
    if output_path.exists() and not args.force:
        data = np.load(str(output_path))
        if data["features"].shape[0] != len(samples):
            raise RuntimeError("Cached feature count does not match selected samples")
        print("Using cached {} features from {}".format(label, output_path))
        return output_path

    import logging

    import torch
    import torch.nn.functional as F
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    from mmcv.runner import load_checkpoint
    from mmdet.datasets.pipelines import Compose

    import darth.datasets  # noqa: F401 - register custom pipelines
    import darth.models  # noqa: F401 - register custom models and heads
    from darth.models import build_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = Config.fromfile(str(config_path))
    model = build_model(cfg.model)
    quiet_logger = logging.getLogger("tsne_checkpoint_loader")
    quiet_logger.setLevel(logging.ERROR)
    load_checkpoint(model, str(checkpoint_path), map_location="cpu", logger=quiet_logger)
    model.to(args.device).eval()
    pipeline = Compose(cfg.data.test.pipeline)

    grouped = defaultdict(list)
    for sample in samples:
        grouped[int(sample["image_id"])].append(sample)
    features = np.empty((len(samples), 256), dtype=np.float32)

    for image_number, image_samples in enumerate(grouped.values(), 1):
        first = image_samples[0]
        image_path = args.image_root / first["file_name"]
        data = pipeline(
            dict(
                img_info=dict(filename=str(image_path), frame_id=int(first["frame_id"])),
                img_prefix=None,
            )
        )
        data = collate([data], samples_per_gpu=1)
        data = scatter(data, [torch.device(args.device)])[0]
        image_tensor = data["img"][0]
        meta = _unwrap_meta(data["img_metas"][0])

        boxes = []
        for sample in image_samples:
            x, y, width, height = sample["bbox_xywh"]
            boxes.append([x, y, x + width, y + height])
        boxes = np.asarray(boxes, dtype=np.float32)
        scale_factor = np.asarray(meta["scale_factor"], dtype=np.float32)
        boxes *= scale_factor.reshape(1, 4)
        image_height, image_width = meta["img_shape"][:2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, image_width - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, image_height - 1)
        box_tensor = torch.from_numpy(boxes).to(args.device)

        with torch.no_grad():
            backbone_features = model.detector.extract_feat(image_tensor)
            embeddings = model.track_head.extract_bbox_feats(backbone_features, [box_tensor])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings = embeddings.detach().cpu().numpy().astype(np.float32)
        if embeddings.shape != (len(image_samples), 256):
            raise RuntimeError("Unexpected embedding shape: {}".format(embeddings.shape))
        for sample, embedding in zip(image_samples, embeddings):
            features[int(sample["sample_index"])] = embedding

        if image_number % 25 == 0 or image_number == len(grouped):
            print(
                "{}: processed {}/{} unique frames".format(
                    label, image_number, len(grouped)
                ),
                flush=True,
            )

    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite values found in {} features".format(label))
    norms = np.linalg.norm(features, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise RuntimeError("{} features are not L2-normalized".format(label))

    labels = np.asarray([sample["identity_index"] for sample in samples], dtype=np.int64)
    np.savez_compressed(
        str(output_path),
        features=features,
        labels=labels,
        sample_indices=np.arange(len(samples), dtype=np.int64),
        config=np.asarray(str(config_path.resolve())),
        checkpoint=np.asarray(str(checkpoint_path.resolve())),
    )
    print("Saved {} features to {}".format(label, output_path))

    del model, features, data, image_tensor, backbone_features, embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path


def high_dimensional_metrics(features, labels):
    similarities = np.clip(features @ features.T, -1.0, 1.0)
    distances = 1.0 - similarities
    same = labels[:, None] == labels[None, :]
    diagonal = np.eye(len(labels), dtype=bool)
    intra = float(distances[same & ~diagonal].mean())
    inter = float(distances[~same].mean())

    nearest_distances = distances.copy()
    nearest_distances[diagonal] = np.inf
    nearest = nearest_distances.argmin(axis=1)
    knn_accuracy = float(np.mean(labels[nearest] == labels))

    from sklearn.metrics import silhouette_score

    silhouette = float(silhouette_score(features, labels, metric="cosine"))
    return {
        "mean_intra_identity_cosine_distance": intra,
        "mean_inter_identity_cosine_distance": inter,
        "inter_intra_ratio": inter / max(intra, 1e-12),
        "silhouette_cosine": silhouette,
        "leave_one_out_1nn_identity_accuracy": knn_accuracy,
    }


def tsne_intra_variance(points, labels):
    values = []
    for label in sorted(set(labels.tolist())):
        cluster = points[labels == label]
        center = cluster.mean(axis=0)
        values.append(float(np.mean(np.sum((cluster - center) ** 2, axis=1))))
    return float(np.mean(values))


def plot_results(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_path = args.output_dir / "selected_samples.json"
    if not selection_path.exists():
        raise FileNotFoundError(str(selection_path))
    with selection_path.open() as handle:
        selection = json.load(handle)
    samples = selection["samples"]

    baseline_data = np.load(str(args.output_dir / "baseline_features.npz"))
    ours_data = np.load(str(args.output_dir / "ours_features.npz"))
    baseline_features = baseline_data["features"]
    ours_features = ours_data["features"]
    baseline_labels = baseline_data["labels"]
    ours_labels = ours_data["labels"]
    if baseline_features.shape != ours_features.shape:
        raise RuntimeError("Feature arrays are not balanced")
    if not np.array_equal(baseline_labels, ours_labels):
        raise RuntimeError("Identity labels differ between methods")

    all_features = np.vstack([baseline_features, ours_features])
    pca_dims = min(50, all_features.shape[1], all_features.shape[0] - 1)
    pca = PCA(n_components=pca_dims, random_state=args.seed)
    reduced = pca.fit_transform(all_features)
    perplexity = min(args.perplexity, float(len(reduced) - 1))
    embedding = TSNE(
        n_components=2,
        random_state=args.seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=2000,
    ).fit_transform(reduced)
    count = len(baseline_features)
    baseline_points = embedding[:count]
    ours_points = embedding[count:]

    baseline_metrics = high_dimensional_metrics(baseline_features, baseline_labels)
    ours_metrics = high_dimensional_metrics(ours_features, ours_labels)
    baseline_metrics["tsne_intra_variance"] = tsne_intra_variance(
        baseline_points, baseline_labels
    )
    ours_metrics["tsne_intra_variance"] = tsne_intra_variance(ours_points, ours_labels)

    padding = 3.0
    xmin, xmax = embedding[:, 0].min() - padding, embedding[:, 0].max() + padding
    ymin, ymax = embedding[:, 1].min() - padding, embedding[:, 1].max() + padding
    unique_labels = sorted(set(baseline_labels.tolist()))
    colors = sns.color_palette("husl", len(unique_labels))
    identity_names = {
        int(sample["identity_index"]): sample["identity_label"] for sample in samples
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True, sharey=True)
    panels = [
        (
            axes[0],
            baseline_points,
            baseline_labels,
            "Baseline Single-Target (10 epochs)",
            baseline_metrics,
        ),
        (
            axes[1],
            ours_points,
            ours_labels,
            "Ours (Continual Adaptation)",
            ours_metrics,
        ),
    ]
    for axis, points, labels, title, metrics in panels:
        for color_index, identity in enumerate(unique_labels):
            mask = labels == identity
            axis.scatter(
                points[mask, 0],
                points[mask, 1],
                s=85,
                alpha=0.82,
                c=[colors[color_index]],
                edgecolors="black",
                linewidth=0.9,
                label=identity_names[identity],
            )
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            "{}\nSilhouette={:.3f} | 1-NN={:.1%} | High-D ratio={:.3f}".format(
                title,
                metrics["silhouette_cosine"],
                metrics["leave_one_out_1nn_identity_accuracy"],
                metrics["inter_intra_ratio"],
            ),
            fontsize=14,
            fontweight="bold",
        )
        axis.set_xlabel("Shared t-SNE Dimension 1", fontsize=12)
        axis.set_ylabel("Shared t-SNE Dimension 2", fontsize=12)
        axis.grid(True, alpha=0.3)
        axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)

    variance_ratio = baseline_metrics["tsne_intra_variance"] / max(
        ours_metrics["tsne_intra_variance"], 1e-12
    )
    fig.suptitle(
        "BDD100K Night Car Tracking Embeddings: Baseline vs Ours\n"
        "12 identities x 15 frames, shared PCA + t-SNE | "
        "1-NN: {:.1%} -> {:.1%}".format(
            baseline_metrics["leave_one_out_1nn_identity_accuracy"],
            ours_metrics["leave_one_out_1nn_identity_accuracy"],
        ),
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    png_path = args.output_dir / "bdd_night_car_tsne_comparison.png"
    pdf_path = args.output_dir / "bdd_night_car_tsne_comparison.pdf"
    fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)

    metrics_payload = {
        "dataset": "BDD100K",
        "domain": "night",
        "category": "car",
        "feature": "L2-normalized 256D track_head.embed_head output on GT boxes",
        "projection": {
            "seed": args.seed,
            "pca_dimensions": pca_dims,
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "tsne_perplexity": perplexity,
            "tsne_iterations": 2000,
            "joint_projection": True,
        },
        "sampling": {
            "num_tracks": selection["num_tracks"],
            "samples_per_track": selection["samples_per_track"],
            "total_per_method": count,
        },
        "baseline": {
            "name": "baseline_single_target_epoch10",
            "config": str(args.baseline_config.resolve()),
            "checkpoint": str(args.baseline_checkpoint.resolve()),
            "metrics": baseline_metrics,
        },
        "ours": {
            "name": "spaP1_causalProtoShrink_roiV1Weak_stage04_after_night_epoch5",
            "config": str(args.ours_config.resolve()),
            "checkpoint": str(args.ours_checkpoint.resolve()),
            "metrics": ours_metrics,
        },
        "ratios": {
            "baseline_over_ours_tsne_intra_variance": variance_ratio,
            "ours_over_baseline_high_dim_inter_intra_ratio": (
                ours_metrics["inter_intra_ratio"]
                / max(baseline_metrics["inter_intra_ratio"], 1e-12)
            ),
        },
    }
    metrics_path = args.output_dir / "tsne_metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(metrics_payload, handle, indent=2)

    points_path = args.output_dir / "tsne_points.csv"
    with points_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "sample_index", "identity_index", "identity_label", "x", "y"]
        )
        for method, points in (("baseline_epoch10", baseline_points), ("ours", ours_points)):
            for sample, point in zip(samples, points):
                writer.writerow(
                    [
                        method,
                        sample["sample_index"],
                        sample["identity_index"],
                        sample["identity_label"],
                        float(point[0]),
                        float(point[1]),
                    ]
                )

    print("Saved figure to {}".format(png_path))
    print("Saved PDF to {}".format(pdf_path))
    print("Saved metrics to {}".format(metrics_path))
    print(json.dumps(metrics_payload["baseline"]["metrics"], indent=2))
    print(json.dumps(metrics_payload["ours"]["metrics"], indent=2))


def invoke_plot_python(args):
    command = [
        args.plot_python,
        str(Path(__file__).resolve()),
        "--stage",
        "plot",
        "--annotation",
        str(args.annotation),
        "--image-root",
        str(args.image_root),
        "--output-dir",
        str(args.output_dir),
        "--baseline-config",
        str(args.baseline_config),
        "--baseline-checkpoint",
        str(args.baseline_checkpoint),
        "--ours-config",
        str(args.ours_config),
        "--ours-checkpoint",
        str(args.ours_checkpoint),
        "--seed",
        str(args.seed),
        "--num-tracks",
        str(args.num_tracks),
        "--samples-per-track",
        str(args.samples_per_track),
        "--max-tracks-per-video",
        str(args.max_tracks_per_video),
        "--perplexity",
        str(args.perplexity),
    ]
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    subprocess.check_call(command, cwd=str(ROOT), env=environment)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.stage in ("all", "extract"):
        ensure_inputs(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        samples = select_samples(args)
        extract_features(
            "baseline", args.baseline_config, args.baseline_checkpoint, samples, args
        )
        extract_features("ours", args.ours_config, args.ours_checkpoint, samples, args)
    if args.stage == "plot":
        plot_results(args)
    elif args.stage == "all":
        invoke_plot_python(args)


if __name__ == "__main__":
    main()
