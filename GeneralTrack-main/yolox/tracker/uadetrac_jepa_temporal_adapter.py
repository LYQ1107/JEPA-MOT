"""Inference-only loader for the source-trained UA-DETRAC temporal adapter."""
from __future__ import annotations

import numpy as np

FEATURE_DIM = 256


def _normalize(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _pack(features, frames, current_frame, history, feature_dim=None):
    feature_dim = int(FEATURE_DIM if feature_dim is None else feature_dim)
    seq = np.zeros((history, feature_dim), np.float32)
    mask = np.zeros((history,), np.float32)
    gaps = np.zeros((history,), np.float32)
    vals = list(zip(frames, features))[-history:]
    if vals:
        off = history - len(vals)
        seq[off:] = _normalize(np.stack([v for _, v in vals]))
        mask[off:] = 1.0
        gaps[off:] = np.asarray([max(1, int(current_frame) - int(f)) for f, _ in vals], np.float32) / 8.0
    return np.concatenate([seq.reshape(-1), mask, gaps]).astype(np.float32)


def load_temporal_adapter(path):
    import torch
    import torch.nn as nn

    payload = torch.load(str(path), map_location="cpu")
    state = payload.get("state_dict", payload)
    history = int(payload.get("history", payload.get("metadata", {}).get("history", 4)))
    feature_dim = int(payload.get("feature_dim", payload.get("metadata", {}).get("feature_dim", FEATURE_DIM)))
    input_dim = history * feature_dim + 2 * history

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.GELU())
            self.pred = nn.Linear(512, feature_dim)

        def forward(self, x):
            return self.pred(self.trunk(x))

    net = Net()
    net.load_state_dict(state, strict=True)
    net.eval()
    alpha = float(payload.get("deploy_alpha", 1.0))

    def predict(history_features, history_frames, current_frame):
        if len(history_features) < 2:
            return _normalize(np.mean(np.stack(history_features), axis=0, keepdims=True))[0] if history_features else None
        x = _pack(history_features, history_frames, current_frame, history, feature_dim)
        with torch.no_grad():
            pred = net(torch.from_numpy(x).unsqueeze(0)).numpy()[0]
        base = _normalize(np.mean(np.stack(list(history_features)[-history:]), axis=0, keepdims=True))[0]
        return _normalize(((1.0 - alpha) * base + alpha * pred)[None])[0]

    return {"predict": predict, "history": history, "deploy_alpha": alpha,
            "metadata": payload.get("metadata", {})}
