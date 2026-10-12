#!/usr/bin/env python3
"""One-crop FastReID compatibility/inference smoke for the Deep-OC-SORT env.

This is deliberately independent of the MOT-JEPA harness: it records whether
the locally vendored FastReID checkpoints/configs can produce finite embeddings
for one MOT17 and one DanceTrack image.  It does not alter detector streams or
tracker results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run_case(name: str, config: Path, weights: Path, image_path: Path, device: str) -> dict:
    out = {
        "name": name, "config": str(config), "weights": str(weights),
        "image": str(image_path), "config_exists": config.exists(),
        "weights_exists": weights.exists(), "image_exists": image_path.exists(),
    }
    if weights.exists():
        out["weights_sha256"] = file_sha256(weights)
    t0 = time.perf_counter()
    try:
        # The vendored FastReID release imports torch._six, removed by torch
        # 2.x.  This compatibility alias is local to the smoke process.
        try:
            import torch
            if "torch._six" not in sys.modules:
                sys.modules["torch._six"] = types.SimpleNamespace(string_classes=(str,))
        except Exception:
            torch = None
        from fast_reid_interfece import FastReIDInterface

        model = FastReIDInterface(str(config), str(weights), device, batch_size=1)
        out["init_seconds"] = float(time.perf_counter() - t0)
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError("cv2.imread returned None")
        out["image_shape"] = list(image.shape)
        h, w = image.shape[:2]
        # A deterministic central crop is sufficient to test preprocessing,
        # model loading, output shape, normalization, and finite values.
        det = np.asarray([[max(0, w // 2 - 40), max(0, h // 2 - 100),
                           min(w - 1, w // 2 + 40), min(h - 1, h // 2 + 100), .9]], dtype=np.float32)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        feat = np.asarray(model.inference(image, det))
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        out["inference_seconds"] = float(time.perf_counter() - t1)
        out["output_shape"] = list(feat.shape)
        out["output_dtype"] = str(feat.dtype)
        out["output_norm"] = float(np.linalg.norm(feat[0])) if len(feat) else None
        out["finite"] = bool(np.isfinite(feat).all())
        out["input_size"] = list(model.cfg.INPUT.SIZE_TEST)
        out["device"] = device
    except Exception as exc:  # retain a machine-readable failure artifact
        out["error"] = type(exc).__name__ + ": " + str(exc)
        out["elapsed_seconds"] = float(time.perf_counter() - t0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/fastreid_smoke_deepocsort_v1.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mot-config", default="/data1/LWR/vranlee/M4FTMoveOut4Doing/HybridSORT/fast_reid/configs/MOT17/sbs_S50.yml")
    ap.add_argument("--dance-config", default="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT/fast_reid/configs/DanceTrack/sbs_S50.yml")
    ap.add_argument("--mot-weights", default="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT/pretrained/mot17_sbs_S50.pth")
    ap.add_argument("--dance-weights", default="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT/pretrained/dancetrack_sbs_S50.pth")
    ap.add_argument("--mot-image", default="/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17/train/MOT17-05-FRCNN/img1/000001.jpg")
    ap.add_argument("--dance-image", default="/data1/LWR/vranlee/DATASETS/JDE/dancetrack/val/dancetrack0004/img1/00000001.jpg")
    args = ap.parse_args()
    cases = [
        run_case("mot17", Path(args.mot_config), Path(args.mot_weights), Path(args.mot_image), args.device),
        run_case("dance", Path(args.dance_config), Path(args.dance_weights), Path(args.dance_image), args.device),
    ]
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        torch_version = None; cuda_available = False
    try:
        import fastreid
        fastreid_version = getattr(fastreid, "__version__", None)
        fastreid_file = str(Path(fastreid.__file__).resolve())
    except Exception as exc:
        fastreid_version = None; fastreid_file = None
    payload = {
        "schema": "fastreid-deepocsort-smoke-v1", "python": sys.version,
        "torch": torch_version, "cuda_available": cuda_available,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fastreid_version": fastreid_version, "fastreid_file": fastreid_file,
        "compatibility_shim": "torch._six.string_classes=(str,)",
        "cases": cases,
        "note": "One deterministic crop per dataset; no tracker/detector state is modified.",
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
