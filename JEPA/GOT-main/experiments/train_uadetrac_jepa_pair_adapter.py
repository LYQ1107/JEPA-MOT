#!/usr/bin/env python3
"""Train a source-only JEPA/geometry pair scorer for GeneralTrack.

The frozen center-pooled GOT-JEPA vector is useful only when the candidate
crop is a plausible continuation of a track.  This trainer learns that
decision boundary from sunny UA-DETRAC track IDs and boxes, using a
video-disjoint source split.  It never opens a target-domain annotation or
target-domain cache.  The small MLP is intentionally an association adapter,
not a detector or a target-label tuner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


FEATURE_DIM = 8
IMAGE_W = 960.0
IMAGE_H = 540.0
GAP_SCALE = 8.0


def parse_args():
    ap = argparse.ArgumentParser("Train source-only UA-DETRAC JEPA pair adapter")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-gap", type=int, default=8)
    ap.add_argument("--neg-per-positive", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    ab = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(aa + ab - inter, 1e-6)


def pair_features(anchor_box: np.ndarray, candidate_box: np.ndarray,
                  anchor_feat: np.ndarray, candidate_feat: np.ndarray,
                  gap: int, gap_scale: float = GAP_SCALE) -> np.ndarray:
    """Build the fixed source-trained pair interface."""
    af = np.asarray(anchor_feat, np.float32)
    cf = np.asarray(candidate_feat, np.float32)
    sim = float(np.dot(af, cf) / max(float(np.linalg.norm(af) * np.linalg.norm(cf)), 1e-8))
    acx = (float(anchor_box[0]) + float(anchor_box[2])) * 0.5
    acy = (float(anchor_box[1]) + float(anchor_box[3])) * 0.5
    ccx = (float(candidate_box[0]) + float(candidate_box[2])) * 0.5
    ccy = (float(candidate_box[1]) + float(candidate_box[3])) * 0.5
    aw = max(float(anchor_box[2] - anchor_box[0]), 1e-3)
    ah = max(float(anchor_box[3] - anchor_box[1]), 1e-3)
    cw = max(float(candidate_box[2] - candidate_box[0]), 1e-3)
    ch = max(float(candidate_box[3] - candidate_box[1]), 1e-3)
    return np.asarray([
        sim,
        iou_xyxy(anchor_box, candidate_box),
        abs(ccx - acx) / IMAGE_W,
        abs(ccy - acy) / IMAGE_H,
        float(np.log((cw * ch) / (aw * ah))),
        float(np.log(cw / aw)),
        float(np.log(ch / ah)),
        float(gap) / float(gap_scale),
    ], np.float32)


def make_samples(features, video_ids, frame_ids, track_ids, boxes, fit_videos,
                 max_gap: int, neg_per_positive: int, seed: int,
                 gap_scale: float = GAP_SCALE):
    by_frame = defaultdict(list)
    by_track = defaultdict(list)
    for i, (v, f, t) in enumerate(zip(video_ids, frame_ids, track_ids)):
        if int(v) in fit_videos:
            by_frame[(int(v), int(f))].append(int(i))
            by_track[(int(v), int(t))].append(int(i))
    for key in by_frame:
        by_frame[key].sort()
    for key in by_track:
        by_track[key].sort(key=lambda i: (int(frame_ids[i]), i))

    rng = np.random.RandomState(int(seed))
    x_rows, y_rows = [], []
    positive_count = 0
    negative_count = 0
    for (v, t), anchors in sorted(by_track.items()):
        for ai in anchors:
            selected = None
            for gap in range(1, int(max_gap) + 1):
                candidate_ids = by_frame.get((v, int(frame_ids[ai]) + gap), [])
                same = [j for j in candidate_ids if int(track_ids[j]) == int(track_ids[ai])]
                if same:
                    selected = (gap, candidate_ids, same[0])
                    break
            if selected is None:
                continue
            gap, candidates, positive = selected
            x_rows.append(pair_features(boxes[ai], boxes[positive], features[ai], features[positive], gap, gap_scale))
            y_rows.append(1.0)
            positive_count += 1
            negatives = [j for j in candidates if int(track_ids[j]) != int(track_ids[ai])]
            if len(negatives) > int(neg_per_positive):
                # Include the most appearance-similar negatives as hard
                # negatives, plus a deterministic random remainder.
                sims = [float(np.dot(features[ai], features[j])) for j in negatives]
                order = np.argsort(np.asarray(sims), kind="mergesort")[::-1]
                hard_n = max(1, int(neg_per_positive) // 2)
                chosen = [negatives[int(k)] for k in order[:hard_n]]
                rest = [negatives[int(k)] for k in order[hard_n:]]
                rng.shuffle(rest)
                chosen.extend(rest[:max(0, int(neg_per_positive) - len(chosen))])
                negatives = chosen
            for negative in negatives:
                x_rows.append(pair_features(boxes[ai], boxes[negative], features[ai], features[negative], gap, gap_scale))
                y_rows.append(0.0)
                negative_count += 1
    if not x_rows:
        raise RuntimeError("no source pair samples")
    x = np.asarray(x_rows, np.float32)
    y = np.asarray(y_rows, np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], {"positive_count": positive_count, "negative_count": negative_count}


def quality(model, x, y, groups):
    import torch
    with torch.inference_mode():
        p = torch.sigmoid(model(torch.from_numpy(x))).cpu().numpy()
    rows = {}
    for key, idxs in groups.items():
        idxs = np.asarray(idxs, np.int64)
        if len(idxs) < 2:
            continue
        order = np.argsort(p[idxs], kind="mergesort")[::-1]
        rows[key] = (float(y[idxs[order[0]]]) == 1.0, float(p[idxs[order[0]]]))
    rank1 = float(np.mean([v[0] for v in rows.values()])) if rows else None
    return p, {"evaluated_groups": len(rows), "rank1": rank1}


def main():
    args = parse_args()
    random.seed(int(args.seed)); np.random.seed(int(args.seed))
    import torch
    import torch.nn as nn

    cache = Path(args.cache).resolve()
    sidecar = cache.with_suffix(cache.suffix + ".json")
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or not sidecar.exists():
        raise FileNotFoundError("cache or sidecar missing: {}".format(cache))
    cache_meta = json.loads(sidecar.read_text())
    with np.load(cache, allow_pickle=False) as z:
        features = np.asarray(z["features"], np.float32)
        video_ids = np.asarray(z["video_ids"], np.int64)
        frame_ids = np.asarray(z["frame_ids"], np.int64)
        track_ids = np.asarray(z["track_ids"], np.int64)
        boxes = np.asarray(z["boxes"], np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    videos = sorted(np.unique(video_ids).tolist())
    split = max(1, min(len(videos) - 1, int(round(len(videos) * 0.70))))
    fit_videos = set(int(v) for v in videos[:split])
    held_videos = set(int(v) for v in videos[split:])

    x_fit, y_fit, fit_stats = make_samples(
        features, video_ids, frame_ids, track_ids, boxes, fit_videos,
        int(args.max_gap), int(args.neg_per_positive), int(args.seed))
    x_held, y_held, held_stats = make_samples(
        features, video_ids, frame_ids, track_ids, boxes, held_videos,
        int(args.max_gap), int(args.neg_per_positive), int(args.seed) + 1)
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
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.RandomState(int(args.seed))
    losses = []
    t0 = time.perf_counter()
    tx = torch.from_numpy(x_fit_n).to(device)
    ty = torch.from_numpy(y_fit).view(-1, 1).to(device)
    for epoch in range(int(args.epochs)):
        order = rng.permutation(len(y_fit))
        total = 0.0; count = 0
        for start in range(0, len(order), int(args.batch)):
            idx = order[start:start + int(args.batch)]
            logits = model(tx[idx])
            loss = loss_fn(logits, ty[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += float(loss.detach()) * len(idx); count += len(idx)
        losses.append(total / max(count, 1))

    # Threshold is selected on held-out sunny source only.  The deployment
    # path uses the pair score as a positive-only reward, so favor the F1
    # operating point rather than any target-domain metric.
    with torch.inference_mode():
        held_p = torch.sigmoid(model(torch.from_numpy(x_held_n).to(device))).cpu().numpy().reshape(-1)
    threshold_rows = []
    for threshold in np.arange(0.35, 0.951, 0.025):
        pred = held_p >= float(threshold)
        tp = int(np.sum(pred & (y_held == 1.0)))
        fp = int(np.sum(pred & (y_held == 0.0)))
        fn = int(np.sum((~pred) & (y_held == 1.0)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        threshold_rows.append({"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn,
                               "precision": precision, "recall": recall, "f1": f1})
    best = max(threshold_rows, key=lambda r: (r["f1"], r["precision"], -r["threshold"]))

    payload = {
        "schema": "uadetrac-sunny-source-jepa-pair-adapter-v1", "status": "complete",
        "cache": str(cache), "cache_sha256": sha256(cache),
        "source_cache_sidecar": str(sidecar), "source_cache_meta": cache_meta,
        "feature_dim": FEATURE_DIM, "feature_interface": [
            "cosine_jepa", "iou", "abs_center_dx_over_960", "abs_center_dy_over_540",
            "log_area_ratio", "log_width_ratio", "log_height_ratio", "gap_over_8"],
        "fit_videos": sorted(fit_videos), "heldout_videos": sorted(held_videos),
        "fit_samples": int(len(y_fit)), "heldout_samples": int(len(y_held)),
        "fit_stats": fit_stats, "heldout_stats": held_stats,
        "epochs": int(args.epochs), "batch": int(args.batch), "max_gap": int(args.max_gap),
        "gap_scale": float(GAP_SCALE),
        "neg_per_positive": int(args.neg_per_positive), "seed": int(args.seed),
        "normalization_mean": mean.tolist(), "normalization_std": std.tolist(),
        "loss_first": float(losses[0]), "loss_last": float(losses[-1]),
        "losses": [float(x) for x in losses], "heldout_threshold_sweep": threshold_rows,
        "deploy_threshold": float(best["threshold"]), "selected_heldout_quality": best,
        "training_supervision": "sunny source track_id temporal pairs with geometry negatives",
        "gt_read": True, "target_domains_read": [],
        "elapsed_seconds": time.perf_counter() - t0, "command": " ".join(sys.argv),
    }
    ckpt = out / "pair_adapter.pt"
    torch.save({"model": model.state_dict(), "normalization_mean": mean,
                "normalization_std": std, "deploy_threshold": float(best["threshold"]),
                "metadata": payload}, ckpt)
    payload["checkpoint"] = str(ckpt); payload["checkpoint_sha256"] = sha256(ckpt)
    (out / "train_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
