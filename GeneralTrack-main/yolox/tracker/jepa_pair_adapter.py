"""CPU inference for the source-trained JEPA/geometry pair adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np


FEATURE_DIM = 8
IMAGE_W = 960.0
IMAGE_H = 540.0
GAP_SCALE = 8.0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    ab = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(aa + ab - inter, 1e-6)


def pair_feature(anchor_box, candidate_box, anchor_feat, candidate_feat, gap, gap_scale=GAP_SCALE):
    a = np.asarray(anchor_box, np.float32); b = np.asarray(candidate_box, np.float32)
    af = np.asarray(anchor_feat, np.float32); cf = np.asarray(candidate_feat, np.float32)
    sim = float(np.dot(af, cf) / max(float(np.linalg.norm(af) * np.linalg.norm(cf)), 1e-8))
    acx = (float(a[0]) + float(a[2])) * 0.5; acy = (float(a[1]) + float(a[3])) * 0.5
    ccx = (float(b[0]) + float(b[2])) * 0.5; ccy = (float(b[1]) + float(b[3])) * 0.5
    aw = max(float(a[2] - a[0]), 1e-3); ah = max(float(a[3] - a[1]), 1e-3)
    cw = max(float(b[2] - b[0]), 1e-3); ch = max(float(b[3] - b[1]), 1e-3)
    return np.asarray([
        sim, _iou(a, b), abs(ccx - acx) / IMAGE_W, abs(ccy - acy) / IMAGE_H,
        np.log((cw * ch) / (aw * ah)), np.log(cw / aw), np.log(ch / ah),
        float(gap) / float(gap_scale),
    ], np.float32)


class PairAdapter:
    def __init__(self, checkpoint: str | Path):
        import torch
        import torch.nn as nn

        payload = torch.load(str(Path(checkpoint).resolve()), map_location="cpu")
        state = payload.get("model", payload.get("state_dict", payload))
        self.model = nn.Sequential(
            nn.Linear(FEATURE_DIM, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1),
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.mean = np.asarray(payload["normalization_mean"], np.float32).reshape(1, FEATURE_DIM)
        self.std = np.maximum(np.asarray(payload["normalization_std"], np.float32).reshape(1, FEATURE_DIM), 1e-4)
        self.threshold = float(payload.get("deploy_threshold", 0.5))
        self.gap_scale = float(payload.get("gap_scale", GAP_SCALE))

    def score_matrix(self, tracks, detections, feature_getter, frame_id: int) -> np.ndarray:
        """Return P(same identity) for every track/detection pair."""
        rows = []
        refs = []
        for i, track in enumerate(tracks):
            tf = feature_getter(track)
            if tf is None:
                continue
            gap = max(1, int(frame_id) - int(getattr(track, "frame_id", frame_id)))
            for j, det in enumerate(detections):
                df = getattr(det, "jepa_feature", None)
                if df is None:
                    continue
                rows.append(pair_feature(track.tlbr, det.tlbr, tf, df, gap, self.gap_scale))
                refs.append((i, j))
        out = np.zeros((len(tracks), len(detections)), np.float32)
        if not rows:
            return out
        import torch
        x = (np.asarray(rows, np.float32) - self.mean) / self.std
        with torch.inference_mode():
            p = torch.sigmoid(self.model(torch.from_numpy(x))).numpy().reshape(-1)
        for (i, j), value in zip(refs, p):
            out[int(i), int(j)] = float(value)
        return out


def load_pair_adapter(path: str | Path | None):
    return PairAdapter(path) if path else None
