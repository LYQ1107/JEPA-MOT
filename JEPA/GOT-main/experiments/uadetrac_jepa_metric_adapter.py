"""Small source-trained metric adapter for frozen GOT-JEPA vectors."""
from __future__ import annotations

from pathlib import Path

import numpy as np


FEATURE_DIM = 256


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def load_numpy_adapter(path: str | Path):
    """Load a checkpoint emitted by train_uadetrac_jepa_metric_adapter."""
    import torch

    payload = torch.load(str(Path(path).resolve()), map_location="cpu")
    state = payload.get("state_dict", payload)
    # The source trainer currently uses a bare ``nn.Linear`` (state key
    # ``weight``); accept the namespaced form as well so the checkpoint
    # interface remains stable if the module is wrapped in ``proj`` later.
    weight = state.get("proj.weight", state.get("weight"))
    if weight is None:
        raise ValueError("adapter checkpoint is missing proj.weight")
    weight = weight.detach().cpu().numpy().astype(np.float32)
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("expected square adapter weight [D,D], got {}".format(weight.shape))
    alpha = float(payload.get("deploy_alpha", 1.0))
    return weight, alpha, payload


def apply_numpy_adapter(features: np.ndarray, weight: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    features = normalize_rows(features)
    projected = normalize_rows(features @ np.asarray(weight, dtype=np.float32).T)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return normalize_rows((1.0 - alpha) * features + alpha * projected).astype(np.float32)
