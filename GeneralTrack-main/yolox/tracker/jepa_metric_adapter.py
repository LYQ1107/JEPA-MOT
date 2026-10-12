"""NumPy inference helper for the source-trained frozen-feature adapter."""
from __future__ import annotations

import numpy as np


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load_adapter(path):
    import torch

    payload = torch.load(str(path), map_location="cpu")
    state = payload.get("state_dict", payload)
    # The UA-DETRAC source trainer uses a bare nn.Linear (``weight``), while
    # wrapped versions may serialize it as ``proj.weight``.  Both represent
    # the same 256x256 frozen-feature projection.
    weight = state.get("proj.weight", state.get("weight"))
    if weight is None:
        raise ValueError("adapter checkpoint is missing proj.weight")
    weight = weight.detach().cpu().numpy().astype(np.float32)
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("expected a square [D,D] adapter weight, got {}".format(weight.shape))
    return weight, float(payload.get("deploy_alpha", 1.0)), payload


def apply_adapter(features, weight, alpha):
    base = normalize_rows(features)
    projected = normalize_rows(base @ np.asarray(weight, dtype=np.float32).T)
    a = float(np.clip(alpha, 0.0, 1.0))
    return normalize_rows((1.0 - a) * base + a * projected).astype(np.float32)
