#!/usr/bin/env python3
"""Build checkpoint-bound 300-frame GOT-JEPA pooled caches for DanceTrack.

Only train annotations are read, and only to obtain clean teacher crop boxes.
Evaluation uses images plus the ID-free D-CTRL detection cache; validation GT
is intentionally never parsed by this cache stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mot_dance_jepa_got as got  # noqa: E402
import mot_dance_jepa_real as dance  # noqa: E402


TRAIN_DEFAULT = "dancetrack0001,dancetrack0002,dancetrack0006"
EVAL_DEFAULT = "dancetrack0004,dancetrack0007,dancetrack0014"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def names(value: str):
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=got.CHECKPOINT_DEFAULT)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--dance-train", default=TRAIN_DEFAULT)
    ap.add_argument("--dance-eval", default=EVAL_DEFAULT)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="cache manifest JSON")
    args = ap.parse_args()
    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir).resolve(); cache.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    base = {"schema": "dance-got-jepa-cache300-v1", "status": "started",
            "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
            "platform": platform.platform(), "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint), "cache_dir": str(cache),
            "train_sequences": names(args.dance_train), "eval_sequences": names(args.dance_eval),
            "max_frames": int(args.max_frames), "batch": int(args.batch), "device": args.device,
            "eval_gt_read": False, "teacher_gt_scope": "train sequences only"}
    t0 = time.perf_counter()
    try:
        if args.max_frames != 300:
            raise ValueError("cache builder is fixed to max_frames=300")
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        got.seed_all(17)
        encoder = got.official_api.OfficialLatent(checkpoint, args.device, args.batch)
        base["official_load"] = encoder.load_meta
        train_names, eval_names = names(args.dance_train), names(args.dance_eval)
        train_gt = dance.limit_frame_dicts(
            dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/train.json", "train", train_names),
            args.max_frames)
        records = {}
        # Detector and clean teacher caches for train sequences.
        for seq in train_names:
            seq_dir = dance.DANCE_ROOT / "train" / seq
            det = dance.parse_frame_det_cache(dance.DANCE_CTRL_CACHE / seq, args.max_frames)
            dpath = cache / f"{seq}_detector.npz"
            _, dm = got.extract_pooled_records(encoder, seq_dir, got.detector_records(det), dpath,
                                               "detector", args.max_frames, args.batch)
            tpath = cache / f"{seq}_teacher.npz"
            _, tm = got.extract_pooled_records(encoder, seq_dir, got.gt_records(train_gt[seq]), tpath,
                                               "clean_train_teacher", args.max_frames, args.batch)
            records[seq] = {"detector": dm, "teacher": tm}
        # Evaluation detector features only: no val annotation access.
        for seq in eval_names:
            seq_dir = dance.DANCE_ROOT / "val" / seq
            det = dance.parse_frame_det_cache(dance.DANCE_CTRL_CACHE / seq, args.max_frames)
            dpath = cache / f"{seq}_detector.npz"
            _, dm = got.extract_pooled_records(encoder, seq_dir, got.detector_records(det), dpath,
                                               "detector", args.max_frames, args.batch)
            records[seq] = {"detector": dm, "teacher": None}
        base.update({"status": "complete", "records": records,
                     "cache_files": sorted(str(p) for p in cache.glob("*.npz")),
                     "elapsed_seconds": time.perf_counter() - t0,
                     "protocol": {"checkpoint_bound": True, "source_feature_shape": [256, 27, 27],
                                  "pooled_dim": 256, "detector_has_ids": False,
                                  "eval_gt_read": False, "teacher_gt_only_train": True}})
        out.write_text(json.dumps(base, indent=2, sort_keys=True, default=float) + "\n")
        print(json.dumps({"status": "complete", "out": str(out), "cache_dir": str(cache),
                          "sequences": sorted(records), "elapsed_seconds": base["elapsed_seconds"]},
                         indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        base.update({"status": "failure", "error_type": type(exc).__name__, "error": str(exc),
                     "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - t0})
        out.write_text(json.dumps(base, indent=2, sort_keys=True, default=float) + "\n")
        print(json.dumps({"status": "failure", "out": str(out), "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
