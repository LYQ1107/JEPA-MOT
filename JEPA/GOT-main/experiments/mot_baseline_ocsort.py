#!/usr/bin/env python3
"""Run the local, mature OC-SORT public-detection tracker on MOT17.

This small adapter intentionally keeps the detector stream untouched.  It is
used to establish a reproducible baseline before adding the independent JEPA
association module.  The tracker implementation is imported from the local
HybridSORT checkout; no production file is modified.
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
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
HYBRID_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT")
MOT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17")
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))

from trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402
from trackers.byte_tracker.byte_tracker import BYTETracker  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_det(path: Path, max_frames: int) -> Dict[int, np.ndarray]:
    out: Dict[int, List[List[float]]] = {}
    with path.open("r", errors="replace") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            try:
                fr = int(float(p[0]))
                if fr < 1 or (max_frames and fr > max_frames):
                    continue
                x, y, w, h = map(float, p[2:6])
                score = float(p[6]) if len(p) > 6 else 1.0
            except (TypeError, ValueError):
                continue
            out.setdefault(fr, []).append([x, y, x + max(0.0, w), y + max(0.0, h), score])
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def write_track(path: Path, rows: List[List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for fr, tid, x1, y1, x2, y2, score in rows:
            f.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},-1,-1,-1\n".format(
                int(fr), int(tid), x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1), score))


def run_sequence(seq: str, max_frames: int, out: Path, track_thresh: float,
                 iou_threshold: float, max_age: int, min_hits: int,
                 delta_t: int, inertia: float, use_byte: bool,
                 tracker_name: str = "ocsort") -> Dict[str, object]:
    seq_dir = MOT_ROOT / "train" / seq
    det_path = seq_dir / "det" / "det.txt"
    dets = parse_det(det_path, max_frames)
    info = {}
    for line in (seq_dir / "seqinfo.ini").read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1); info[k.strip()] = v.strip()
    iw, ih = float(info.get("imWidth", 1)), float(info.get("imHeight", 1))
    # Both adapters consume the exact same public detector rows.  The
    # ByteTrack branch uses the canonical two-stage high/low score association;
    # OC-SORT uses observation-centric motion association.
    if tracker_name == "ocsort":
        tracker = OCSort(det_thresh=float(track_thresh), max_age=int(max_age),
                         min_hits=int(min_hits), iou_threshold=float(iou_threshold),
                         delta_t=int(delta_t), asso_func="iou", inertia=float(inertia),
                         use_byte=bool(use_byte))
    elif tracker_name == "bytetrack":
        from types import SimpleNamespace
        bargs = SimpleNamespace(track_thresh=float(track_thresh), track_buffer=int(max_age),
                                match_thresh=float(iou_threshold), asso="iou", mot20=False)
        tracker = BYTETracker(bargs, frame_rate=30)
    else:
        raise ValueError("unknown tracker {}".format(tracker_name))
    rows: List[List[float]] = []
    t0 = time.perf_counter()
    emitted = 0
    for fr in range(1, int(max_frames) + 1):
        d = dets.get(fr, np.empty((0, 5), dtype=np.float32))
        boxes = d[:, :4] if len(d) else np.empty((0, 4), dtype=np.float32)
        scores = d[:, 4] if len(d) else np.empty((0,), dtype=np.float32)
        cates = np.zeros((len(d),), dtype=np.float32)
        if tracker_name == "ocsort":
            # update_public does not need image dimensions, unlike update().
            ret = tracker.update_public(boxes, cates, scores)
        else:
            ret = tracker.update(d, (ih, iw), (ih, iw))
        for r in ret:
            if tracker_name == "ocsort":
                if len(r) < 7:
                    continue
                x1, y1, x2, y2, tid, _, lag = [float(x) for x in r[:7]]
                out_fr = int(fr + int(round(lag)))
            else:
                # STrack objects are returned by canonical BYTETracker.
                x1, y1, x2, y2 = [float(x) for x in r.tlbr]
                tid, lag = float(r.track_id), 0.0
                out_fr = fr
            if out_fr < 1 or out_fr > max_frames:
                continue
            score = float(getattr(r, "score", 1.0)) if tracker_name != "ocsort" else 1.0
            rows.append([out_fr, int(round(tid)), x1, y1, x2, y2, score])
            emitted += 1
    elapsed = max(time.perf_counter() - t0, 1e-9)
    # Multiple head-padding rows can target the same frame/id.  TrackEval
    # rejects duplicate detections; preserve the first occurrence.
    uniq = {}
    for row in rows:
        uniq.setdefault((int(row[0]), int(row[1])), row)
    rows = sorted(uniq.values(), key=lambda z: (z[0], z[1]))
    dst = out / "mot17" / "tracks" / tracker_name / (seq + ".txt")
    write_track(dst, rows)
    return {
        "sequence": seq, "frames": int(max_frames), "det_rows": int(sum(len(v) for v in dets.values())),
        "track_rows": len(rows), "emitted_before_dedup": emitted,
        "fps": float(max_frames / elapsed), "det_source": str(det_path),
        "det_sha256": sha256(det_path), "track_sha256": sha256(dst),
        "tracker": tracker_name,
        "params": {"track_thresh": track_thresh, "iou_threshold": iou_threshold,
                   "max_age": max_age, "min_hits": min_hits, "delta_t": delta_t,
                   "inertia": inertia, "use_byte": use_byte},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="experiments/results/baseline_ocsort_mot20")
    ap.add_argument("--tracker", choices=["ocsort", "bytetrack"], default="ocsort")
    ap.add_argument("--sequences", default="MOT17-04-FRCNN")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--track-thresh", type=float, default=0.20)
    ap.add_argument("--iou-threshold", type=float, default=0.30)
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--min-hits", type=int, default=1)
    ap.add_argument("--delta-t", type=int, default=3)
    ap.add_argument("--inertia", type=float, default=0.2)
    ap.add_argument("--use-byte", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    seqs = [s for s in args.sequences.split(",") if s]
    rows = [run_sequence(s, args.max_frames, out, args.track_thresh,
                         args.iou_threshold, args.max_age, args.min_hits,
                         args.delta_t, args.inertia, args.use_byte, args.tracker) for s in seqs]
    payload = {
        "schema": "mot-baseline-ocsort-v1", "status": "complete",
        "dataset": "mot17", "eval_sequences": seqs, "max_frames": args.max_frames,
        "methods": [args.tracker], "rows": rows,
        "tracker_source": str(HYBRID_ROOT / "trackers/ocsort_tracker/ocsort.py"),
        "tracker_source_sha256": sha256(HYBRID_ROOT / "trackers/ocsort_tracker/ocsort.py"),
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
        "platform": platform.platform(), "detector_only": True,
        "eval_gt_read": False,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
