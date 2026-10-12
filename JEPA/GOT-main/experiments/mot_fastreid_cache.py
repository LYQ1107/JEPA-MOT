#!/usr/bin/env python3
"""Extract/cache FastReID SBS-S50 embeddings for fixed MOT17 detections.

The cache is an input artifact, not a tracker output.  It stores one array per
frame in the same order as ``det.txt`` rows parsed by the JEPA runner.  The
script is intentionally independent of the production HybridSORT code and
records configuration/weight hashes for auditability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

MOT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17")
DEFAULT_CFG = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TrackTrack/2FastReID/configs/MOT17/sbs_S50.yml")
DEFAULT_WEIGHT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT/pretrained/mot17_sbs_S50.pth")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_det(path: Path, max_frames: int):
    out = {}
    with path.open("r", errors="replace") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            try:
                fr = int(float(p[0]))
                if fr < 1 or (max_frames and fr > max_frames):
                    continue
                x, y, w, h = [float(v) for v in p[2:6]]
                score = float(p[6]) if len(p) > 6 else 1.0
            except Exception:
                continue
            out.setdefault(fr, []).append([x, y, x + max(0., w), y + max(0., h), score])
    return {fr: np.asarray(rows, dtype=np.float32) for fr, rows in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", default="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-09-FRCNN,MOT17-05-FRCNN,MOT17-11-FRCNN")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--out-dir", default="experiments/results/mot_fastreid_cache_mot120")
    ap.add_argument("--mot-root", default=str(MOT_ROOT))
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--weight", default=str(DEFAULT_WEIGHT))
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    # Import only after argument parsing so a MOT-only metadata check does not
    # require FastReID in the parent interpreter.
    try:
        from fastreid.emb_computer import EmbeddingComputer
    except Exception as exc:
        raise RuntimeError("run with PYTHONPATH=TrackTrack/2FastReID in the ovtr/HybridSORT env") from exc
    root = Path(args.mot_root).resolve()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    seqs = [s for s in args.sequences.split(",") if s]
    embedder = EmbeddingComputer(config_path=args.config, weight_path=args.weight, max_batch=args.batch)
    rows_meta = []
    all_t0 = time.perf_counter()
    for seq in seqs:
        seq_dir = root / "train" / seq
        det = parse_det(seq_dir / "det/det.txt", args.max_frames)
        frame_features = {}
        t0 = time.perf_counter(); count = 0
        for fr in range(1, args.max_frames + 1):
            d = det.get(fr, np.empty((0, 5), dtype=np.float32))
            if len(d) == 0:
                frame_features[fr] = np.empty((0, 2048), dtype=np.float32)
                continue
            img = cv2.imread(str(seq_dir / "img1" / ("{:06d}.jpg".format(fr))))
            if img is None:
                raise FileNotFoundError("image missing: {}".format(seq_dir / "img1" / ("{:06d}.jpg".format(fr))))
            z = embedder.compute_embedding(img, d[:, :4]).astype(np.float32)
            if z.shape != (len(d), 2048) or not np.isfinite(z).all():
                raise FloatingPointError("invalid FastReID output {} {} {}".format(seq, fr, z.shape))
            frame_features[fr] = z
            count += len(d)
        dst = out / (seq + ".npz")
        # Object array keeps variable frame lengths while retaining exact row
        # order.  ``allow_pickle=True`` is explicit at read time.
        arr = np.empty((args.max_frames + 1,), dtype=object)
        for fr in range(1, args.max_frames + 1):
            arr[fr] = frame_features[fr]
        np.savez_compressed(dst, features=arr)
        rows_meta.append({"sequence": seq, "frames": args.max_frames, "det_rows": int(sum(len(x) for x in det.values())),
                          "feature_rows": count, "feature_dim": 2048, "file": str(dst),
                          "file_sha256": sha256(dst), "det_source": str(seq_dir / "det/det.txt"),
                          "det_sha256": sha256(seq_dir / "det/det.txt"),
                          "seconds": time.perf_counter() - t0})
        print(seq, count, "rows", "%.2fs" % (time.perf_counter() - t0), flush=True)
    payload = {"schema": "mot-fastreid-cache-v1", "status": "complete", "dataset": "MOT17",
               "sequences": seqs, "max_frames": args.max_frames, "rows": rows_meta,
               "config": str(Path(args.config).resolve()), "config_sha256": sha256(Path(args.config)),
               "weight": str(Path(args.weight).resolve()), "weight_sha256": sha256(Path(args.weight)),
               "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
               "platform": platform.platform(), "seconds_total": time.perf_counter() - all_t0,
               "warning": "dataset-specific FastReID checkpoint; use as explicit upper-bound/association feature artifact."}
    (out / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
