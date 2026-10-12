#!/usr/bin/env python3
"""Bounded feature/API smoke for the public GOT-JEPA ToMP checkpoint.

The experiment is deliberately not a tracker benchmark.  It loads the public
ToMPnet_PT checkpoint, extracts DINOv2-ViT-L + bkMlp feature maps for all public
MOT17 detections in a short frame prefix, and applies the serialized JEPA heads
to a causal history average.  No GT, track IDs, or production tracker files
are read or modified.  This gives the MOT adapter a measured feature API,
latency, and memory baseline before a full OC-SORT integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "pytracking"
MOT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def install_shims(repo: Path) -> None:
    """Import ToMP modules without optional pytracking GUI dependencies."""
    import importlib.util

    p_root = repo / "pytracking"
    pkg = types.ModuleType("pytracking")
    pkg.__path__ = [str(p_root)]
    pkg.TensorDict = type("TensorDict", (dict,), {})
    pkg.TensorList = list
    sys.modules["pytracking"] = pkg
    utils = types.ModuleType("pytracking.utils")
    utils.__path__ = [str(p_root / "utils")]
    sys.modules["pytracking.utils"] = utils
    spec = importlib.util.spec_from_file_location(
        "pytracking.utils.plotting", str(p_root / "utils" / "plotting.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pytracking.utils.plotting"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    sys.modules.setdefault("jpeg4py", types.ModuleType("jpeg4py"))
    local = types.ModuleType("ltr.admin.local")
    local.EnvironmentSettings = type("EnvironmentSettings", (), {})
    sys.modules["ltr.admin.local"] = local


def parse_detections(path: Path, max_frames: int):
    frames = {}
    with path.open("r", errors="replace") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            try:
                fr = int(float(p[0]))
                if fr < 1 or fr > max_frames:
                    continue
                x, y, w, h = [float(v) for v in p[2:6]]
                s = float(p[6]) if len(p) > 6 else 1.0
            except Exception:
                continue
            frames.setdefault(fr, []).append([x, y, x + max(0.0, w), y + max(0.0, h), s])
    return {k: np.asarray(v, dtype=np.float32) for k, v in frames.items()}


def crop_image(img: np.ndarray, box: np.ndarray, size: int = 378) -> np.ndarray:
    """Make a padded square crop in RGB float-compatible uint8 format."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    # ToMP uses a search area factor of five; retaining context is important
    # for a fair feature API smoke.
    side = max(4.0, (bw * bh) ** 0.5 * 5.0)
    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    xa, ya = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
    xb, yb = xa + int(round(side)), ya + int(round(side))
    pad_l, pad_t = max(0, -xa), max(0, -ya)
    pad_r, pad_b = max(0, xb - w), max(0, yb - h)
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        xa, xb, ya, yb = xa + pad_l, xb + pad_l, ya + pad_t, yb + pad_t
    crop = img[ya:yb, xa:xb]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def build_net(checkpoint: dict, tompnet_pt, torch):
    """Construct exact serialized architecture while avoiding external files."""
    candidates = [
        Path("/home/lwr/.cache/torch/hub/facebookresearch_dinov2_main"),
        Path("/home/user/.cache/torch/hub/facebookresearch_dinov2_main"),
    ]
    dino_repo = next((p for p in candidates if p.exists()), None)
    if dino_repo is None:
        raise FileNotFoundError("local DINOv2 hub source not found")

    def dino_architecture(model_name="dinov2_vitl14"):
        return torch.hub.load(str(dino_repo), model_name, source="local", pretrained=False)

    tompnet_pt.backbones.dinov2 = dino_architecture
    from cotracker.models.core.cotracker.cotracker import CoTracker2
    from ltr.cotracker2.co_tracker.cotracker.predictor import CoTrackerPredictor

    def cotracker_architecture(checkpoint_path=None, device="cuda"):
        pred = CoTrackerPredictor.__new__(CoTrackerPredictor)
        torch.nn.Module.__init__(pred)
        pred.support_grid_size = 6
        pred.model = CoTracker2(stride=4, window_len=8, add_space_attn=True)
        pred.interp_shape = pred.model.model_resolution
        sub = {k[len("cotracker_model.model.") :]: v for k, v in checkpoint["net"].items()
               if k.startswith("cotracker_model.model.")}
        miss, unexp = pred.model.load_state_dict(sub, strict=False)
        if miss or unexp:
            raise RuntimeError(f"CoTracker subset mismatch missing={miss} unexpected={unexp}")
        pred.model.eval()
        return pred

    tompnet_pt.backbones.cotracker_predictor = cotracker_architecture
    net = checkpoint["constructor"].get()
    miss, unexp = net.load_state_dict(checkpoint["net"], strict=False)
    if miss or unexp:
        raise RuntimeError(f"ToMP checkpoint mismatch missing={miss} unexpected={unexp}")
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sequence", default="MOT17-04-FRCNN")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint).resolve()
    payload = {
        "schema": "got-jepa-tomp-feature-adapter-smoke-v1",
        "status": "started",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "sequence": args.sequence,
        "max_frames": args.max_frames,
        "batch": args.batch,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "eval_gt_read": False,
        "track_ids_read": False,
        "production_tracker_modified": False,
    }
    t_all = time.perf_counter()
    try:
        import torch

        payload["python"] = sys.version
        payload["torch"] = torch.__version__
        payload["torch_cuda"] = torch.version.cuda
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(args.device)
        install_shims(REPO)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        import ltr.models.tracking.tompnet_PT as tompnet_pt

        t_build = time.perf_counter()
        net = build_net(checkpoint, tompnet_pt, torch)
        net = net.to(device).eval()
        payload["build_seconds"] = time.perf_counter() - t_build
        payload["parameters"] = sum(p.numel() for p in net.parameters())

        seq_dir = MOT_ROOT / "train" / args.sequence
        det_path = seq_dir / "det" / "det.txt"
        dets = parse_detections(det_path, args.max_frames)
        crops, refs = [], []
        for fr in range(1, args.max_frames + 1):
            d = dets.get(fr, np.empty((0, 5), dtype=np.float32))
            if len(d) == 0:
                continue
            img = cv2.imread(str(seq_dir / "img1" / (f"{fr:06d}.jpg")))
            if img is None:
                raise FileNotFoundError(f"missing image for frame {fr}")
            for j, row in enumerate(d):
                # DINO expects RGB ImageNet-normalized float tensors.
                rgb = cv2.cvtColor(crop_image(img, row[:4]), cv2.COLOR_BGR2RGB)
                x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
                x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                crops.append(x)
                refs.append((fr, j))
        payload["det_rows"] = len(crops)
        payload["det_sha256"] = sha256(det_path)
        if not crops:
            raise RuntimeError("no detections in requested prefix")

        # Extract in bounded batches.  Keep only pooled feature maps for the
        # history probe, so peak memory is independent of total rows.
        feature_maps = []
        backbone_shapes = set()
        backbone_finite = True
        projection_seconds = 0.0
        t_feat = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, len(crops), args.batch):
                xb = torch.stack(crops[i : i + args.batch]).to(device, non_blocking=True)
                z = net.extract_dino_features_spatial_intermediate_layers(xb, None)
                backbone_shapes.add(tuple(z.shape))
                # ``extract_dino_features_spatial_intermediate_layers`` returns
                # the 1024-channel DINO+bkMlp map.  ToMP's head projection is
                # the subsequent 1024->256 stage used by the JEPA predictors.
                if z.ndim != 4 or z.shape[1:] != (1024, 27, 27):
                    raise RuntimeError(f"unexpected feature shape {tuple(z.shape)}")
                finite_z = bool(torch.isfinite(z).all())
                backbone_finite = backbone_finite and finite_z
                if not finite_z:
                    raise FloatingPointError("non-finite DINO+bkMlp feature")
                t_proj = time.perf_counter()
                z = net.head.extract_head_feat(z)
                projection_seconds += time.perf_counter() - t_proj
                if z.shape[1:] != (256, 27, 27):
                    raise RuntimeError(f"unexpected projected feature shape {tuple(z.shape)}")
                feature_maps.append(z.detach().cpu())
        feature_maps = torch.cat(feature_maps, dim=0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            payload["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            payload["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        payload["feature_seconds"] = time.perf_counter() - t_feat
        payload["backbone_feature_shapes"] = [list(s) for s in sorted(backbone_shapes)]
        payload["backbone_feature_finite"] = backbone_finite
        payload["head_projection_seconds"] = projection_seconds
        payload["head_projection_rows_per_second"] = float(len(crops) / max(projection_seconds, 1e-9))
        payload["feature_rows"] = int(feature_maps.shape[0])
        payload["feature_shape"] = list(feature_maps.shape)
        payload["feature_rows_per_second"] = float(len(crops) / max(payload["feature_seconds"], 1e-9))

        # Causal history->predictor API: for each detection, average at most
        # four *earlier* feature maps from the same frame-ordered stream.  No
        # current/future candidate is used to form the predictor input.
        out_cls, out_breg, cos_cls, cos_breg = [], [], [], []
        t_pred = time.perf_counter()
        with torch.inference_mode():
            for i in range(len(crops)):
                h0 = max(0, i - 4)
                hist = feature_maps[h0:i]
                if len(hist) == 0:
                    hist = feature_maps[i : i + 1]
                hist = hist.mean(dim=0, keepdim=True).to(device)
                y1 = net.JEPA_predictor_cls(hist)
                y2 = net.JEPA_predictor_breg(hist)
                if not torch.isfinite(y1).all() or not torch.isfinite(y2).all():
                    raise FloatingPointError("non-finite JEPA predictor output")
                p1 = torch.nn.functional.normalize(y1.mean(dim=(-1, -2)), dim=-1)
                p2 = torch.nn.functional.normalize(y2.mean(dim=(-1, -2)), dim=-1)
                out_cls.append(p1.cpu())
                out_breg.append(p2.cpu())
                if i > 0:
                    prev1 = out_cls[-2]
                    prev2 = out_breg[-2]
                    cos_cls.append(float((p1.cpu() * prev1).sum()))
                    cos_breg.append(float((p2.cpu() * prev2).sum()))
        predictor_seconds = time.perf_counter() - t_pred
        payload["predictor_cls_shape"] = [len(out_cls), 256]
        payload["predictor_breg_shape"] = [len(out_breg), 256]
        payload["predictor_outputs_finite"] = True
        payload["predictor_seconds"] = predictor_seconds
        payload["predictor_calls"] = len(out_cls)
        payload["predictor_calls_per_second"] = float(len(out_cls) / max(predictor_seconds, 1e-9))
        payload["history_k"] = 4
        payload["pooled_cosine_cls_mean"] = float(np.mean(cos_cls)) if cos_cls else None
        payload["pooled_cosine_cls_p10"] = float(np.percentile(cos_cls, 10)) if cos_cls else None
        payload["pooled_cosine_breg_mean"] = float(np.mean(cos_breg)) if cos_breg else None
        payload["pooled_cosine_breg_p10"] = float(np.percentile(cos_breg, 10)) if cos_breg else None
        payload["status"] = "complete"
    except Exception as exc:
        payload["status"] = "failure"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    payload["elapsed_seconds"] = time.perf_counter() - t_all
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
