#!/usr/bin/env python3
"""Train a source-only metric adapter on frozen sunny UA-DETRAC features.

The target domains are not opened.  Video-disjoint sunny source videos are
split deterministically into fit and held-out groups.  The trainable module is
initialized to identity and learns an L2-normalized projection from temporal
same-track pairs; the held-out source split selects only the deployment blend
alpha.  The resulting 256-D interface is directly consumable by GeneralTrack.
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
from uadetrac_jepa_metric_adapter import apply_numpy_adapter, normalize_rows  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser("Train source-only UA-DETRAC JEPA metric adapter")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-gap", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    return ap.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_pairs(video_ids, frame_ids, track_ids, fit_videos, max_gap):
    groups = defaultdict(list)
    for i, (v, f, t) in enumerate(zip(video_ids, frame_ids, track_ids)):
        if int(v) in fit_videos:
            groups[(int(v), int(t))].append(int(i))
    pairs = []
    for key, values in sorted(groups.items()):
        values.sort(key=lambda i: (int(frame_ids[i]), i))
        for a, i in enumerate(values[:-1]):
            # Use the nearest later observation, with a deterministic bounded
            # temporal gap.  The pair is source supervision, never a target
            # evaluation signal.
            for j in values[a + 1:]:
                gap = int(frame_ids[j]) - int(frame_ids[i])
                if gap <= 0:
                    continue
                if gap <= int(max_gap):
                    pairs.append((int(i), int(j), key))
                break
    if not pairs:
        raise RuntimeError("no source temporal pairs")
    return pairs


def rank_quality(features, video_ids, frame_ids, track_ids, videos, weight=None, alpha=1.0, max_gap=8):
    rows = np.arange(len(features), dtype=np.int64)
    if weight is not None:
        transformed = apply_numpy_adapter(features, weight, alpha)
    else:
        transformed = normalize_rows(features)
    by_frame = defaultdict(list)
    by_track = defaultdict(list)
    for i in rows:
        if int(video_ids[i]) not in videos:
            continue
        by_frame[(int(video_ids[i]), int(frame_ids[i]))].append(int(i))
        by_track[(int(video_ids[i]), int(track_ids[i]))].append(int(i))
    hits = []
    margins = []
    evaluated = 0
    for (v, t), ids in sorted(by_track.items()):
        ids.sort(key=lambda i: (int(frame_ids[i]), i))
        for i in ids:
            later = [j for j in ids if 0 < int(frame_ids[j]) - int(frame_ids[i]) <= int(max_gap)]
            if not later:
                continue
            j = later[0]
            candidates = by_frame.get((v, int(frame_ids[j])), [])
            if len(candidates) < 2:
                continue
            sims = transformed[candidates] @ transformed[i]
            pos = int(candidates.index(j))
            neg = np.delete(sims, pos)
            hits.append(float(sims[pos] >= np.max(neg)))
            margins.append(float(sims[pos] - np.max(neg)))
            evaluated += 1
    return {
        "videos": sorted(int(x) for x in videos), "evaluated": int(evaluated),
        "rank1": float(np.mean(hits)) if hits else None,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "p10_margin": float(np.percentile(margins, 10)) if margins else None,
    }


def main():
    args = parse_args()
    random.seed(int(args.seed)); np.random.seed(int(args.seed))
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cache = Path(args.cache).resolve(); out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or not cache.with_suffix(cache.suffix + ".json").exists():
        raise FileNotFoundError("cache or sidecar missing: {}".format(cache))
    cache_meta = json.loads(cache.with_suffix(cache.suffix + ".json").read_text())
    with np.load(cache, allow_pickle=False) as z:
        features = np.asarray(z["features"], np.float32)
        video_ids = np.asarray(z["video_ids"], np.int64)
        frame_ids = np.asarray(z["frame_ids"], np.int64)
        track_ids = np.asarray(z["track_ids"], np.int64)
    if features.ndim != 2 or features.shape[0] != len(video_ids) or features.shape[1] < 2:
        raise ValueError("unexpected source cache shapes {} {}".format(features.shape, video_ids.shape))
    feature_dim = int(features.shape[1])
    features = normalize_rows(features)
    videos = sorted(np.unique(video_ids).tolist())
    split = max(1, min(len(videos) - 1, int(round(len(videos) * 0.70))))
    fit_videos = set(int(x) for x in videos[:split]); held_videos = set(int(x) for x in videos[split:])
    pairs = make_pairs(video_ids, frame_ids, track_ids, fit_videos, int(args.max_gap))
    # Build one deterministic pair per track per mini-batch when possible.
    by_key = defaultdict(list)
    for i, j, key in pairs:
        by_key[key].append((i, j))
    keys = sorted(by_key)
    torch.manual_seed(int(args.seed)); torch.set_num_threads(4)
    device = torch.device(args.device)
    class Projection(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(feature_dim, feature_dim, bias=False)

        def forward(self, x):
            return self.proj(x)

    model = Projection().to(device)
    with torch.no_grad():
        model.proj.weight.copy_(torch.eye(feature_dim))
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    identity = torch.eye(feature_dim, device=device)
    rng = np.random.RandomState(int(args.seed))
    losses = []
    t0 = time.perf_counter()
    for epoch in range(int(args.epochs)):
        rng.shuffle(keys)
        total = 0.0; count = 0
        for start in range(0, len(keys), int(args.batch)):
            batch_keys = keys[start:start + int(args.batch)]
            if len(batch_keys) < 2:
                continue
            chosen = [by_key[k][int(rng.randint(len(by_key[k])))] for k in batch_keys]
            ai = np.asarray([x[0] for x in chosen], np.int64); pi = np.asarray([x[1] for x in chosen], np.int64)
            xa = torch.from_numpy(features[ai]).to(device)
            xp = torch.from_numpy(features[pi]).to(device)
            noise_a = torch.randn_like(xa) * 0.01; noise_p = torch.randn_like(xp) * 0.01
            za = F.normalize(model(F.normalize(xa + noise_a, dim=-1)), dim=-1)
            zp = F.normalize(model(F.normalize(xp + noise_p, dim=-1)), dim=-1)
            logits = za @ zp.T / float(args.temperature)
            labels = torch.arange(len(chosen), device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            # Keep the learned representation close to the identity unless
            # source-heldout evidence supports a change.
            loss = loss + 0.01 * ((model.proj.weight - identity) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += float(loss.detach()) * len(chosen); count += len(chosen)
        losses.append(total / max(count, 1))
    # The projection is nested so the checkpoint has the same `proj.weight`
    # interface consumed by the deployment loader.
    weight = model.proj.weight.detach().cpu().numpy().astype(np.float32)
    base_quality = rank_quality(features, video_ids, frame_ids, track_ids, held_videos, None, 0.0, int(args.max_gap))
    alpha_rows = []
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        alpha_rows.append({"alpha": alpha, "quality": rank_quality(features, video_ids, frame_ids, track_ids, held_videos, weight, alpha, int(args.max_gap))})
    best = max(alpha_rows, key=lambda r: ((r["quality"]["rank1"] if r["quality"]["rank1"] is not None else -1.0),
                                          (r["quality"]["mean_margin"] if r["quality"]["mean_margin"] is not None else -1.0),
                                          -float(r["alpha"])))
    payload = {
        "schema": "uadetrac-sunny-source-jepa-metric-adapter-v1", "status": "complete",
        "cache": str(cache), "cache_sha256": sha256(cache), "source_cache_sidecar": str(cache.with_suffix(cache.suffix + ".json")),
        "feature_dim": feature_dim, "feature_pool": cache_meta.get("pool"),
        "feature_stage": cache_meta.get("stage"), "fit_videos": sorted(fit_videos),
        "heldout_videos": sorted(held_videos), "pair_count": len(pairs), "track_count": len(keys),
        "epochs": int(args.epochs), "batch": int(args.batch), "max_gap": int(args.max_gap),
        "seed": int(args.seed), "loss_first": float(losses[0]), "loss_last": float(losses[-1]),
        "losses": [float(x) for x in losses], "heldout_base": base_quality,
        "heldout_alpha_sweep": alpha_rows, "deploy_alpha": float(best["alpha"]),
        "selected_heldout_quality": best["quality"], "target_domains_read": [],
        "gt_read": True, "training_supervision": "sunny source track_id temporal pairs only",
        "elapsed_seconds": time.perf_counter() - t0, "command": " ".join(sys.argv),
    }
    ckpt = out / "metric_adapter.pt"
    torch.save({"state_dict": model.state_dict(), "deploy_alpha": float(best["alpha"]), "metadata": payload}, ckpt)
    payload["checkpoint"] = str(ckpt); payload["checkpoint_sha256"] = sha256(ckpt)
    (out / "train_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
