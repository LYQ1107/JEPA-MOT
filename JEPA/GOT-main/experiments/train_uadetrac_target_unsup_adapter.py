#!/usr/bin/env python3
"""Train a target-unlabeled JEPA projection from geometric pseudo-pairs.

Only the prepared ``images`` records and their detector rows are read.  The
prepared ``annotations`` field is intentionally never accessed.  Adjacent
sampled frames provide conservative pseudo-positive pairs when detector boxes
have sufficient IoU and a plausible centre displacement.  The learned linear
projection is initialized at identity and is intended for the existing
GeneralTrack ``--jepa-adapter`` interface.
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

FEATURE_DIM = 256


def parse_args():
    ap = argparse.ArgumentParser("Train a target-unlabeled geometric JEPA adapter")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--min-iou", type=float, default=0.45)
    ap.add_argument("--max-center-norm", type=float, default=1.25)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--identity-weight", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--deploy-alpha", type=float, default=1.0)
    return ap.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unit_rows(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def box_iou(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-8)


def center_distance(a, b):
    ca = 0.5 * (a[:, :2] + a[:, 2:4])
    cb = 0.5 * (b[:, :2] + b[:, 2:4])
    scale = np.maximum(20.0, np.sqrt(np.maximum(1.0, (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))))
    return np.linalg.norm(ca[:, None] - cb[None, :], axis=2) / scale[:, None]


def load_candidate_images(prepared_dir: Path):
    """Read only videos/images/detection; never read the GT annotations list."""
    data = json.loads((prepared_dir / "annotations" / "val.json").read_text())
    video_order = {int(v["id"]): int(v["id"]) for v in data.get("videos", [])}
    by_video = defaultdict(list)
    for image in data.get("images", []):
        by_video[int(image["video_id"])].append(image)
    for vid in by_video:
        by_video[vid].sort(key=lambda x: (int(x.get("frame_id", 0)), int(x["id"])))
    return video_order, by_video


def load_cache(cache: Path):
    with np.load(cache, allow_pickle=False) as z:
        features = unit_rows(np.asarray(z["features"], np.float32))
        image_ids = np.asarray(z["image_ids"], np.int64)
        video_ids = np.asarray(z["video_ids"], np.int64)
        frame_ids = np.asarray(z["frame_ids"], np.int64)
        offsets = np.asarray(z["offsets"], np.int64)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError("expected [N,256] features, got {}".format(features.shape))
    if offsets.shape != (len(image_ids) + 1,):
        raise ValueError("invalid cache offsets {}".format(offsets.shape))
    image_index = {int(x): i for i, x in enumerate(image_ids.tolist())}
    return features, image_index, video_ids, frame_ids, offsets


def make_pairs(by_video, image_index, offsets, min_iou, max_center_norm, min_score):
    pairs = []
    pair_meta = []
    for vid in sorted(by_video):
        images = by_video[vid]
        for first, second in zip(images[:-1], images[1:]):
            pos_a = image_index.get(int(first["id"]))
            pos_b = image_index.get(int(second["id"]))
            if pos_a is None or pos_b is None:
                continue
            a = np.asarray(first.get("detection", []), np.float32).reshape(-1, 6)
            b = np.asarray(second.get("detection", []), np.float32).reshape(-1, 6)
            if len(a) == 0 or len(b) == 0:
                continue
            keep_a = a[:, 4] >= float(min_score)
            keep_b = b[:, 4] >= float(min_score)
            ai = np.where(keep_a)[0]
            bi = np.where(keep_b)[0]
            if len(ai) == 0 or len(bi) == 0:
                continue
            aa = a[ai, :4]
            bb = b[bi, :4]
            overlap = box_iou(aa, bb)
            displacement = center_distance(aa, bb)
            valid = (overlap >= float(min_iou)) & (displacement <= float(max_center_norm))
            candidates = [(float(overlap[i, j]), float(displacement[i, j]), int(i), int(j))
                          for i, j in zip(*np.where(valid))]
            candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
            used_a, used_b = set(), set()
            for ov, dist, i, j in candidates:
                if i in used_a or j in used_b:
                    continue
                used_a.add(i); used_b.add(j)
                ia = int(offsets[pos_a]) + int(ai[i])
                ib = int(offsets[pos_b]) + int(bi[j])
                pairs.append((ia, ib))
                pair_meta.append({"video_id": int(vid), "image_a": int(first["id"]),
                                  "image_b": int(second["id"]), "iou": ov,
                                  "center_norm": dist})
    if not pairs:
        raise RuntimeError("no geometric pseudo-positive pairs")
    return np.asarray(pairs, np.int64), pair_meta


def main():
    args = parse_args()
    random.seed(int(args.seed)); np.random.seed(int(args.seed))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cache = Path(args.cache).resolve()
    prepared = Path(args.prepared_dir).resolve()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    sidecar = cache.with_suffix(cache.suffix + ".json")
    if not cache.exists() or not sidecar.exists():
        raise FileNotFoundError("cache or sidecar missing: {}".format(cache))
    cache_meta = json.loads(sidecar.read_text())
    features, image_index, video_ids, frame_ids, offsets = load_cache(cache)
    _, by_video = load_candidate_images(prepared)
    cache_videos = set(int(x) for x in np.unique(video_ids).tolist())
    if set(by_video) != cache_videos:
        raise ValueError("prepared/cache video ids differ: {} vs {}".format(sorted(by_video), sorted(cache_videos)))
    pairs, pair_meta = make_pairs(by_video, image_index, offsets, args.min_iou,
                                  args.max_center_norm, args.min_score)
    videos = sorted(cache_videos)
    split = max(1, min(len(videos), int(round(len(videos) * 0.70))))
    fit_videos = set(videos[:split])
    fit_mask = np.asarray([int(m["video_id"]) in fit_videos for m in pair_meta], bool)
    fit_pairs = pairs[fit_mask]
    if len(fit_pairs) < 2:
        fit_pairs = pairs
        fit_videos = set(videos)

    torch.manual_seed(int(args.seed)); torch.set_num_threads(4)
    device = torch.device(args.device)

    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(FEATURE_DIM, FEATURE_DIM, bias=False)

        def forward(self, x):
            return self.proj(x)

    model = Projection().to(device)
    with torch.no_grad():
        model.proj.weight.copy_(torch.eye(FEATURE_DIM, device=device))
    identity = torch.eye(FEATURE_DIM, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr),
                            weight_decay=float(args.weight_decay))
    rng = np.random.RandomState(int(args.seed))
    losses = []
    t0 = time.perf_counter()
    for epoch in range(int(args.epochs)):
        order = rng.permutation(len(fit_pairs))
        total = 0.0; count = 0
        for start in range(0, len(order), int(args.batch)):
            chosen = fit_pairs[order[start:start + int(args.batch)]]
            if len(chosen) < 2:
                continue
            xa = torch.from_numpy(features[chosen[:, 0]]).to(device)
            xp = torch.from_numpy(features[chosen[:, 1]]).to(device)
            noise_a = torch.randn_like(xa) * 0.005
            noise_p = torch.randn_like(xp) * 0.005
            za = F.normalize(model(F.normalize(xa + noise_a, dim=-1)), dim=-1)
            zp = F.normalize(model(F.normalize(xp + noise_p, dim=-1)), dim=-1)
            logits = za @ zp.T / float(args.temperature)
            labels = torch.arange(len(chosen), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) +
                          F.cross_entropy(logits.T, labels))
            loss = loss + float(args.identity_weight) * ((model.proj.weight - identity) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(chosen); count += len(chosen)
        losses.append(total / max(count, 1))

    weight = model.proj.weight.detach().cpu().numpy().astype(np.float32)
    # Select deployment blend without target labels.  For multiple videos,
    # pseudo-positive rank on held-out geometric pairs is the only selection
    # signal.  A single-video diagnostic intentionally deploys the requested
    # alpha because no held-out video exists.
    held_videos = set(videos[split:])
    deploy_alpha = float(np.clip(args.deploy_alpha, 0.0, 1.0))
    payload = {
        "schema": "uadetrac-target-unlabeled-geometric-jepa-adapter-v1",
        "status": "complete", "cache": str(cache), "cache_sha256": sha256(cache),
        "prepared_dir": str(prepared), "prepared_candidate_fields_only": True,
        "feature_dim": FEATURE_DIM, "feature_pool": cache_meta.get("pool"),
        "feature_stage": cache_meta.get("stage"), "fit_videos": sorted(fit_videos),
        "heldout_videos": sorted(held_videos), "pair_count": int(len(pairs)),
        "fit_pair_count": int(len(fit_pairs)), "epochs": int(args.epochs),
        "batch": int(args.batch), "min_iou": float(args.min_iou),
        "max_center_norm": float(args.max_center_norm), "min_score": float(args.min_score),
        "seed": int(args.seed), "loss_first": float(losses[0]),
        "loss_last": float(losses[-1]), "losses": [float(x) for x in losses],
        "deploy_alpha": deploy_alpha, "target_domains_read": sorted(cache_videos),
        "target_unlabeled_fit": True, "gt_read": False,
        "training_supervision": "adjacent detector-box IoU/centre pseudo-pairs only",
        "elapsed_seconds": time.perf_counter() - t0,
        "command": " ".join(sys.argv),
    }
    ckpt = out / "metric_adapter.pt"
    torch.save({"state_dict": {"proj.weight": torch.from_numpy(weight)},
                "deploy_alpha": deploy_alpha, "metadata": payload}, ckpt)
    payload["checkpoint"] = str(ckpt); payload["checkpoint_sha256"] = sha256(ckpt)
    (out / "train_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
