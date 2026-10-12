#!/usr/bin/env python3
"""Train a target-unlabeled JEPA/geometry pair adapter.

The adapter is a transductive diagnostic: it reads only ``videos``,
``images`` and ``images[].detection`` from a target prepared stream plus its
detector-aligned frozen V-JEPA cache.  Adjacent detector boxes supply
geometric pseudo-positive pairs; all operating-point selection is made on a
video-disjoint held-out subset of those pseudo labels.  The prepared
``annotations`` list is never accessed.

The checkpoint uses the same eight-feature interface as the source-only
pair adapter, so it can be evaluated without changing GeneralTrack's
detector, native association threshold, or lifecycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


IMAGE_W = 960.0
IMAGE_H = 540.0
GAP_SCALE = 8.0
FEATURE_DIM = 8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iou(a, b):
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(aa + bb - inter, 1e-6)


def center_norm(a, b):
    ac = np.asarray([(float(a[0]) + float(a[2])) * 0.5,
                     (float(a[1]) + float(a[3])) * 0.5])
    bc = np.asarray([(float(b[0]) + float(b[2])) * 0.5,
                     (float(b[1]) + float(b[3])) * 0.5])
    scale = max(20.0, float(np.sqrt(max(1.0,
        (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))))))
    return float(np.linalg.norm(ac - bc) / scale)


def pair_features(anchor_box, candidate_box, anchor_feat, candidate_feat,
                  gap: int) -> np.ndarray:
    af = np.asarray(anchor_feat, np.float32)
    cf = np.asarray(candidate_feat, np.float32)
    sim = float(np.dot(af, cf) /
                max(float(np.linalg.norm(af) * np.linalg.norm(cf)), 1e-8))
    acx = (float(anchor_box[0]) + float(anchor_box[2])) * 0.5
    acy = (float(anchor_box[1]) + float(anchor_box[3])) * 0.5
    ccx = (float(candidate_box[0]) + float(candidate_box[2])) * 0.5
    ccy = (float(candidate_box[1]) + float(candidate_box[3])) * 0.5
    aw = max(float(anchor_box[2] - anchor_box[0]), 1e-3)
    ah = max(float(anchor_box[3] - anchor_box[1]), 1e-3)
    cw = max(float(candidate_box[2] - candidate_box[0]), 1e-3)
    ch = max(float(candidate_box[3] - candidate_box[1]), 1e-3)
    return np.asarray([
        sim, iou(anchor_box, candidate_box), abs(ccx - acx) / IMAGE_W,
        abs(ccy - acy) / IMAGE_H, np.log((cw * ch) / (aw * ah)),
        np.log(cw / aw), np.log(ch / ah), float(gap) / GAP_SCALE,
    ], dtype=np.float32)


def load_rows(prepared: Path, cache: Path, min_score: float):
    """Return candidate-only rows and per-video/per-frame indexes."""
    data = json.loads((prepared / "annotations" / "val.json").read_text())
    with np.load(cache, allow_pickle=False) as source:
        features = np.asarray(source["features"], np.float32)
        offsets = np.asarray(source["offsets"], np.int64)
        image_ids = np.asarray(source["image_ids"], np.int64)
    if offsets.shape != (len(image_ids) + 1,):
        raise ValueError("invalid cache offsets/image_ids")
    if int(offsets[-1]) != len(features):
        raise ValueError("cache offsets do not describe feature rows")
    image_index = {int(image_id): i for i, image_id in enumerate(image_ids.tolist())}
    images = {int(image["id"]): image for image in data.get("images", [])}
    if set(images) != set(image_index):
        raise ValueError("prepared/cache image-id sets differ")
    row_features = features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    by_video = defaultdict(list)
    by_frame = defaultdict(list)
    rows = []
    for image in sorted(data.get("images", []),
                        key=lambda x: (int(x["video_id"]), int(x["frame_id"]), int(x["id"]))):
        image_id = int(image["id"])
        cache_pos = image_index[image_id]
        start, end = int(offsets[cache_pos]), int(offsets[cache_pos + 1])
        detections = np.asarray(image.get("detection", []), np.float32).reshape(-1, 6)
        if end - start != len(detections):
            raise ValueError(f"cache/detection row mismatch at image_id={image_id}")
        for local, detection in enumerate(detections):
            if float(detection[4]) < float(min_score):
                continue
            row = {
                "row": start + local,
                "video": int(image["video_id"]),
                "frame": int(image["frame_id"]),
                "box": detection[:4].astype(np.float32),
                "score": float(detection[4]),
                "feature": row_features[start + local],
            }
            rows.append(row)
            by_frame[(row["video"], row["frame"])].append(row)
    for key in by_frame:
        by_frame[key].sort(key=lambda x: x["row"])
    return rows, by_frame, sorted({int(v["id"]) for v in data.get("videos", [])})


def make_samples(by_frame, videos, fit_videos, *, min_iou, max_center_norm,
                 max_gap, neg_per_positive, seed):
    rng = np.random.RandomState(int(seed))
    positives = []
    metadata = []
    for video in sorted(fit_videos):
        frame_ids = sorted(frame for (vid, frame) in by_frame if vid == video)
        for position, frame in enumerate(frame_ids):
            anchors = by_frame[(video, frame)]
            selected = None
            # UA-DETRAC's prepared detector stream is temporally sampled;
            # consecutive image records can be several source frames apart.
            # Search the next sampled record while retaining its real frame
            # gap in the pair interface.
            for next_frame in frame_ids[position + 1:]:
                gap = int(next_frame) - int(frame)
                if gap > int(max_gap):
                    break
                candidates = by_frame.get((video, next_frame), [])
                if not candidates:
                    continue
                geometric = []
                for ai, anchor in enumerate(anchors):
                    for bi, candidate in enumerate(candidates):
                        overlap = iou(anchor["box"], candidate["box"])
                        displacement = center_norm(anchor["box"], candidate["box"])
                        if overlap >= float(min_iou) and displacement <= float(max_center_norm):
                            geometric.append((-overlap, displacement, ai, bi))
                if geometric:
                    geometric.sort()
                    selected = (gap, candidates, geometric)
                    break
            if selected is None:
                continue
            gap, candidates, geometric = selected
            used_a, used_b = set(), set()
            for _, _, ai, bi in geometric:
                if ai in used_a or bi in used_b:
                    continue
                used_a.add(ai); used_b.add(bi)
                anchor, positive = anchors[ai], candidates[bi]
                positives.append((anchor, positive, gap))
                metadata.append({"video": video, "frame": frame, "gap": gap,
                                 "anchor_row": anchor["row"], "positive_row": positive["row"]})
    if not positives:
        raise RuntimeError("no target geometric pseudo-positive pairs")

    x_rows, y_rows = [], []
    for (anchor, positive, gap), meta in zip(positives, metadata):
        x_rows.append(pair_features(anchor["box"], positive["box"],
                                     anchor["feature"], positive["feature"], gap))
        y_rows.append(1.0)
        candidates = by_frame[(int(anchor["video"]), int(anchor["frame"] + gap))]
        negatives = [candidate for candidate in candidates
                     if candidate["row"] != positive["row"]]
        if len(negatives) > int(neg_per_positive):
            scores = np.asarray([
                float(np.dot(anchor["feature"], candidate["feature"]))
                for candidate in negatives], np.float32)
            order = np.argsort(scores, kind="mergesort")[::-1]
            hard = max(1, int(neg_per_positive) // 2)
            chosen = [negatives[int(index)] for index in order[:hard]]
            rest = [negatives[int(index)] for index in order[hard:]]
            rng.shuffle(rest)
            negatives = chosen + rest[:max(0, int(neg_per_positive) - len(chosen))]
        for negative in negatives:
            x_rows.append(pair_features(anchor["box"], negative["box"],
                                         anchor["feature"], negative["feature"], gap))
            y_rows.append(0.0)
    return (np.asarray(x_rows, np.float32), np.asarray(y_rows, np.float32),
            {"positive_count": len(positives),
             "negative_count": int(np.sum(np.asarray(y_rows) == 0.0)),
             "pair_metadata_count": len(metadata)})


def parse_args():
    ap = argparse.ArgumentParser("Train target-unlabeled JEPA pair adapter")
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--prepared-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--min-iou", type=float, default=0.45)
    ap.add_argument("--max-center-norm", type=float, default=1.25)
    ap.add_argument("--max-gap", type=int, default=8)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--neg-per-positive", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(int(args.seed)); np.random.seed(int(args.seed))
    import torch
    import torch.nn as nn

    cache = args.cache.resolve(); prepared = args.prepared_dir.resolve()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    rows, by_frame, videos = load_rows(prepared, cache, args.min_score)
    split = max(1, min(len(videos) - 1, int(round(len(videos) * 0.70))))
    fit_videos = set(videos[:split]); held_videos = set(videos[split:])
    if not held_videos:
        fit_videos = set(videos); held_videos = set(videos)
    x_fit, y_fit, fit_stats = make_samples(
        by_frame, videos, fit_videos, min_iou=args.min_iou,
        max_center_norm=args.max_center_norm, max_gap=args.max_gap,
        neg_per_positive=args.neg_per_positive, seed=args.seed)
    x_held, y_held, held_stats = make_samples(
        by_frame, videos, held_videos, min_iou=args.min_iou,
        max_center_norm=args.max_center_norm, max_gap=args.max_gap,
        neg_per_positive=args.neg_per_positive, seed=args.seed + 1)

    mean = x_fit.mean(axis=0).astype(np.float32)
    std = np.maximum(x_fit.std(axis=0), 1e-4).astype(np.float32)
    x_fit_n = (x_fit - mean) / std
    x_held_n = (x_held - mean) / std
    torch.manual_seed(int(args.seed)); torch.set_num_threads(4)
    device = torch.device(args.device)
    model = nn.Sequential(
        nn.Linear(FEATURE_DIM, 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1),
    ).to(device)
    pos = max(float(np.sum(y_fit == 1.0)), 1.0)
    neg = max(float(np.sum(y_fit == 0.0)), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                            weight_decay=float(args.weight_decay))
    tx = torch.from_numpy(x_fit_n).to(device)
    ty = torch.from_numpy(y_fit).view(-1, 1).to(device)
    rng = np.random.RandomState(int(args.seed))
    losses = []
    started = time.perf_counter()
    for _ in range(int(args.epochs)):
        order = rng.permutation(len(y_fit)); total = 0.0; count = 0
        for start in range(0, len(order), int(args.batch)):
            index = order[start:start + int(args.batch)]
            logits = model(tx[index]); loss = loss_fn(logits, ty[index])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += float(loss.detach()) * len(index); count += len(index)
        losses.append(total / max(count, 1))
    with torch.inference_mode():
        held_p = torch.sigmoid(model(torch.from_numpy(x_held_n).to(device))).cpu().numpy().reshape(-1)
    threshold_rows = []
    for threshold in np.arange(0.20, 0.951, 0.025):
        pred = held_p >= float(threshold)
        tp = int(np.sum(pred & (y_held == 1.0)))
        fp = int(np.sum(pred & (y_held == 0.0)))
        fn = int(np.sum((~pred) & (y_held == 1.0)))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        threshold_rows.append({"threshold": float(threshold), "tp": tp, "fp": fp,
                               "fn": fn, "precision": precision,
                               "recall": recall, "f1": f1})
    best = max(threshold_rows, key=lambda x: (x["f1"], x["precision"], -x["threshold"]))
    payload = {
        "schema": "uadetrac-target-unlabeled-jepa-pair-adapter-v1",
        "status": "complete", "cache": str(cache),
        "cache_sha256": sha256(cache), "prepared_dir": str(prepared),
        "prepared_candidate_fields_only": True, "feature_dim": FEATURE_DIM,
        "feature_interface": ["cosine_jepa", "iou", "abs_center_dx_over_960",
                              "abs_center_dy_over_540", "log_area_ratio",
                              "log_width_ratio", "log_height_ratio", "gap_over_8"],
        "videos": videos, "fit_videos": sorted(fit_videos),
        "heldout_videos": sorted(held_videos), "candidate_rows": len(rows),
        "fit_samples": int(len(y_fit)), "heldout_samples": int(len(y_held)),
        "fit_stats": fit_stats, "heldout_stats": held_stats,
        "epochs": int(args.epochs), "batch": int(args.batch),
        "min_iou": float(args.min_iou), "max_center_norm": float(args.max_center_norm),
        "max_gap": int(args.max_gap), "min_score": float(args.min_score),
        "neg_per_positive": int(args.neg_per_positive), "seed": int(args.seed),
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "loss_first": float(losses[0]), "loss_last": float(losses[-1]),
        "losses": [float(x) for x in losses],
        "heldout_threshold_sweep": threshold_rows,
        "deploy_threshold": float(best["threshold"]),
        "selected_heldout_quality": best,
        "training_supervision": "adjacent target detector-box IoU/centre pseudo-pairs only",
        "target_unlabeled_fit": True, "gt_read": False,
        "target_domains_read": videos,
        "elapsed_seconds": float(time.perf_counter() - started),
        "command": " ".join(str(x) for x in __import__("sys").argv),
    }
    checkpoint = out / "pair_adapter.pt"
    torch.save({"model": model.state_dict(), "normalization_mean": mean,
                "normalization_std": std, "deploy_threshold": float(best["threshold"]),
                "metadata": payload}, checkpoint)
    payload["checkpoint"] = str(checkpoint); payload["checkpoint_sha256"] = sha256(checkpoint)
    (out / "train_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
