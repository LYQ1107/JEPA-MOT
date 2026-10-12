#!/usr/bin/env python3
"""Train a causal source-only temporal predictor for UA-DETRAC JEPA vectors.

The cache contains sunny source GT crops only.  A video-disjoint source split
is used for fitting and held-out alpha selection; target domains are never
opened.  At deployment the predictor receives only accepted observations from
the past and emits a candidate-independent vector for the next frame.
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
FEATURE_DIM = 256


def parse_args():
    ap = argparse.ArgumentParser("Train source-only causal UA-DETRAC JEPA temporal predictor")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--feature-dim", type=int, default=None,
                    help="Optional check; otherwise infer from the cache.")
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalize_rows(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def pack_history(features, frame_ids, indices, target_frame, history):
    seq = np.zeros((history, FEATURE_DIM), np.float32)
    mask = np.zeros((history,), np.float32)
    gaps = np.zeros((history,), np.float32)
    vals = list(indices)[-history:]
    if vals:
        off = history - len(vals)
        seq[off:] = features[np.asarray(vals, np.int64)]
        mask[off:] = 1.0
        gaps[off:] = np.asarray(
            [max(1, int(target_frame) - int(frame_ids[i])) for i in vals], np.float32
        ) / 8.0
    return np.concatenate([seq.reshape(-1), mask, gaps]).astype(np.float32)


def build_samples(features, video_ids, frame_ids, track_ids, videos, history):
    groups = defaultdict(list)
    by_frame = defaultdict(list)
    for i, (v, f, t) in enumerate(zip(video_ids, frame_ids, track_ids)):
        if int(v) in videos:
            groups[(int(v), int(t))].append(int(i))
            by_frame[(int(v), int(f))].append(int(i))
    samples = []
    for key, vals in sorted(groups.items()):
        vals.sort(key=lambda i: (int(frame_ids[i]), i))
        # The target at position p is the next accepted observation after the
        # history.  No current candidate is passed to the predictor.
        for p in range(1, len(vals)):
            target = int(vals[p])
            hist = vals[max(0, p - history):p]
            if not hist:
                continue
            frame = int(frame_ids[target])
            candidates = by_frame.get((int(video_ids[target]), frame), [])
            if len(candidates) < 2:
                continue
            samples.append({
                "x": pack_history(features, frame_ids, hist, frame, history),
                "y": np.asarray(features[target], np.float32),
                "video": int(video_ids[target]), "track": int(track_ids[target]), "frame": frame,
                "target": target, "candidates": np.asarray(candidates, np.int64),
                "history": np.asarray(hist, np.int64),
            })
    if not samples:
        raise RuntimeError("no temporal source samples")
    return samples


def make_net(torch, nn, input_dim):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU())
            self.pred = nn.Linear(512, FEATURE_DIM)

        def forward(self, x):
            return self.pred(self.trunk(x))
    return Net()


def quality(samples, features, model, torch, alpha):
    model.eval()
    hits, margins = [], []
    with torch.no_grad():
        for s in samples:
            pred = model(torch.from_numpy(s["x"]).unsqueeze(0)).cpu().numpy()[0]
            base = normalize_rows(np.mean(features[s["history"]], axis=0, keepdims=True))[0]
            query = normalize_rows(((1.0 - alpha) * base + alpha * pred)[None])[0]
            cand = normalize_rows(features[s["candidates"]])
            sims = cand @ query
            pos = int(np.where(s["candidates"] == int(s["target"]))[0][0])
            neg = np.delete(sims, pos)
            hits.append(float(sims[pos] >= np.max(neg)))
            margins.append(float(sims[pos] - np.max(neg)))
    return {
        "samples": len(samples),
        "rank1": float(np.mean(hits)) if hits else None,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "p10_margin": float(np.percentile(margins, 10)) if margins else None,
    }


def main():
    global FEATURE_DIM
    args = parse_args()
    random.seed(int(args.seed)); np.random.seed(int(args.seed))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cache = Path(args.cache).resolve(); out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or not cache.with_suffix(cache.suffix + ".json").exists():
        raise FileNotFoundError("cache or sidecar missing: {}".format(cache))
    cache_meta = json.loads(cache.with_suffix(cache.suffix + ".json").read_text())
    with np.load(cache, allow_pickle=False) as z:
        features = normalize_rows(np.asarray(z["features"], np.float32))
        video_ids = np.asarray(z["video_ids"], np.int64)
        frame_ids = np.asarray(z["frame_ids"], np.int64)
        track_ids = np.asarray(z["track_ids"], np.int64)
    if features.ndim != 2:
        raise ValueError("unexpected feature shape {}".format(features.shape))
    FEATURE_DIM = int(features.shape[1])
    if args.feature_dim is not None and int(args.feature_dim) != FEATURE_DIM:
        raise ValueError("cache feature dim {} != requested {}".format(
            FEATURE_DIM, args.feature_dim))
    videos = sorted(np.unique(video_ids).tolist())
    split = max(1, min(len(videos) - 1, int(round(len(videos) * 0.70))))
    fit_videos = set(int(x) for x in videos[:split])
    held_videos = set(int(x) for x in videos[split:])
    fit = build_samples(features, video_ids, frame_ids, track_ids, fit_videos, int(args.history))
    held = build_samples(features, video_ids, frame_ids, track_ids, held_videos, int(args.history))

    torch.manual_seed(int(args.seed)); torch.set_num_threads(4)
    device = torch.device(args.device)
    input_dim = int(args.history) * FEATURE_DIM + 2 * int(args.history)
    model = make_net(torch, nn, input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.RandomState(int(args.seed))
    # Avoid treating two temporal samples from the same identity as
    # InfoNCE negatives in one batch.  Each batch draws at most one sample
    # from a source video/track key.
    track_buckets = defaultdict(list)
    for i, sample in enumerate(fit):
        track_buckets[(int(sample["video"]), int(sample["track"]))].append(int(i))
    track_keys = sorted(track_buckets)
    losses = []; t0 = time.perf_counter()
    for epoch in range(int(args.epochs)):
        rng.shuffle(track_keys); total = 0.0; count = 0
        for start in range(0, len(track_keys), int(args.batch)):
            batch_keys = track_keys[start:start + int(args.batch)]
            idx = np.asarray([track_buckets[k][int(rng.randint(len(track_buckets[k])))] for k in batch_keys], dtype=np.int64)
            if len(idx) < 2:
                continue
            xb = torch.from_numpy(np.stack([fit[int(i)]["x"] for i in idx])).to(device)
            yb = torch.from_numpy(np.stack([fit[int(i)]["y"] for i in idx])).to(device)
            pred = F.normalize(model(xb), dim=-1); target = F.normalize(yb, dim=-1)
            logits = pred @ target.T / float(args.temperature)
            labels = torch.arange(len(idx), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            loss = loss + 0.25 * (1.0 - (pred * target).sum(-1)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += float(loss.detach()) * len(idx); count += len(idx)
        losses.append(total / max(count, 1))

    rows = []
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        rows.append({"alpha": alpha, "quality": quality(held, features, model, torch, alpha)})
    best = max(rows, key=lambda r: ((r["quality"]["rank1"] if r["quality"]["rank1"] is not None else -1.0),
                                    (r["quality"]["mean_margin"] if r["quality"]["mean_margin"] is not None else -1.0),
                                    -float(r["alpha"])))
    payload = {
        "schema": "uadetrac-sunny-source-jepa-temporal-v1", "status": "complete",
        "cache": str(cache), "cache_sha256": sha256(cache), "feature_dim": FEATURE_DIM,
        "feature_pool": cache_meta.get("pool"), "feature_stage": cache_meta.get("stage"),
        "history": int(args.history),
        "fit_videos": sorted(fit_videos), "heldout_videos": sorted(held_videos),
        "fit_samples": len(fit), "heldout_samples": len(held),
        "epochs": int(args.epochs), "batch": int(args.batch), "temperature": float(args.temperature),
        "lr": float(args.lr), "weight_decay": float(args.weight_decay), "seed": int(args.seed),
        "loss_first": float(losses[0]), "loss_last": float(losses[-1]), "losses": [float(x) for x in losses],
        "heldout_alpha_sweep": rows, "deploy_alpha": float(best["alpha"]),
        "selected_heldout_quality": best["quality"], "target_domains_read": [],
        "gt_read": True, "training_supervision": "sunny source track_id temporal next-observation pairs",
        "elapsed_seconds": time.perf_counter() - t0, "command": " ".join(sys.argv),
    }
    ckpt = out / "temporal_adapter.pt"
    torch.save({"state_dict": model.state_dict(), "history": int(args.history),
                "deploy_alpha": float(best["alpha"]), "metadata": payload}, ckpt)
    payload["checkpoint"] = str(ckpt); payload["checkpoint_sha256"] = sha256(ckpt)
    (out / "train_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
