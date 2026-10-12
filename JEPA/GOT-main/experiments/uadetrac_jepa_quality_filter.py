#!/usr/bin/env python3
"""Target-GT-free JEPA quality gate for GeneralTrack tracklets.

This is an offline output-quality diagnostic.  It keeps long native tracklets
and selectively retains short tracklets only when their detector confidence,
causal JEPA feature continuity, and local motion are plausible.  The script
reads ``images[].detection`` and a frozen feature cache; it never accesses the
prepared ``annotations`` list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser("GT-free JEPA tracklet quality filter")
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--track-results", required=True)
    ap.add_argument("--jepa-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-long", type=int, default=5)
    ap.add_argument("--short-min-score", type=float, default=0.55)
    ap.add_argument("--short-min-sim", type=float, default=0.95)
    ap.add_argument("--single-min-score", type=float, default=0.85)
    ap.add_argument("--max-center-norm", type=float, default=2.5)
    ap.add_argument("--min-iou", type=float, default=0.20)
    return ap.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iou(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(aa + bb - inter, 1e-8)


def center_norm(a, b):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    ca = 0.5 * (a[:2] + a[2:]); cb = 0.5 * (b[:2] + b[2:])
    scale = max(20.0, float(np.sqrt(max(1.0, (a[2] - a[0]) * (a[3] - a[1])))))
    return float(np.linalg.norm(ca - cb) / scale)


def load_candidate_index(prepared: Path):
    data = json.loads((prepared / "annotations" / "val.json").read_text())
    videos = {int(v["id"]): str(v.get("name", v.get("file_name", v["id"])))
              for v in data.get("videos", [])}
    images = sorted(data.get("images", []),
                    key=lambda x: (int(x["video_id"]), int(x["frame_id"]), int(x["id"])))
    by_name_frame = {}
    for image in images:
        vid = int(image["video_id"]); frame = int(image["frame_id"])
        by_name_frame[(videos.get(vid, str(vid)), frame)] = image
    return videos, by_name_frame


def load_cache(path: Path):
    with np.load(path, allow_pickle=False) as z:
        features = np.asarray(z["features"], np.float32)
        image_ids = np.asarray(z["image_ids"], np.int64)
        offsets = np.asarray(z["offsets"], np.int64)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    return features, {int(x): i for i, x in enumerate(image_ids.tolist())}, offsets


def parse_track_file(path: Path):
    rows = []
    with path.open("r", errors="replace") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            try:
                frame = int(float(p[0])); tid = int(float(p[1]))
                x, y, w, h = [float(v) for v in p[2:6]]
                score = float(p[6])
            except (TypeError, ValueError):
                continue
            rows.append({"frame": frame, "tid": tid,
                         "box": np.asarray([x, y, x + max(0.0, w), y + max(0.0, h)], np.float32),
                         "score": score, "raw": line.rstrip("\n")})
    return rows


def attach(rows, image, features, image_index, offsets, min_iou):
    detections = np.asarray(image.get("detection", []), np.float32).reshape(-1, 6)
    cache_idx = image_index.get(int(image["id"]))
    if cache_idx is None:
        return
    start, end = int(offsets[cache_idx]), int(offsets[cache_idx + 1])
    image_features = features[start:end]
    for row in rows:
        if len(detections) == 0 or len(image_features) != len(detections):
            row["feature"] = None; row["det_iou"] = 0.0
            continue
        j = max(range(len(detections)), key=lambda q: iou(row["box"], detections[q, :4]))
        q = iou(row["box"], detections[j, :4])
        row["feature"] = image_features[j] if q >= float(min_iou) else None
        row["det_iou"] = float(q)


def keep_track(rows, args):
    rows = sorted(rows, key=lambda r: (r["frame"], r["tid"]))
    if len(rows) >= int(args.min_long):
        return True, {"reason": "long", "length": len(rows)}
    scores = np.asarray([r["score"] for r in rows], np.float32)
    if len(rows) == 1:
        ok = bool(scores[0] >= float(args.single_min_score))
        return ok, {"reason": "single_score" if ok else "single_reject",
                    "length": 1, "mean_score": float(scores[0])}
    sims = []
    motion = []
    for prev, cur in zip(rows[:-1], rows[1:]):
        if prev["feature"] is not None and cur["feature"] is not None:
            sims.append(float(np.dot(prev["feature"], cur["feature"])))
        motion.append(center_norm(prev["box"], cur["box"]))
    ok = bool(len(sims) == len(rows) - 1 and
              float(np.mean(scores)) >= float(args.short_min_score) and
              float(np.min(sims)) >= float(args.short_min_sim) and
              float(np.max(motion)) <= float(args.max_center_norm))
    return ok, {"reason": "short_keep" if ok else "short_reject",
                "length": len(rows), "mean_score": float(np.mean(scores)),
                "min_adjacent_sim": float(np.min(sims)) if sims else None,
                "max_center_norm": float(np.max(motion)) if motion else None}


def main():
    args = parse_args()
    prepared = Path(args.prepared_dir).resolve()
    track_root = Path(args.track_results).resolve()
    cache = Path(args.jepa_cache).resolve()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    videos, by_name_frame = load_candidate_index(prepared)
    features, image_index, offsets = load_cache(cache)
    stats = {"files": 0, "input_rows": 0, "output_rows": 0,
             "input_tracklets": 0, "kept_tracklets": 0,
             "short_kept": 0, "short_rejected": 0, "single_kept": 0,
             "per_file": {}}
    details = []
    for path in sorted(track_root.glob("*.txt")):
        name = path.stem
        rows = parse_track_file(path)
        for row in rows:
            image = by_name_frame.get((name, int(row["frame"])))
            if image is not None:
                attach([row], image, features, image_index, offsets, args.min_iou)
            else:
                row["feature"] = None; row["det_iou"] = 0.0
        groups = defaultdict(list)
        for row in rows:
            groups[int(row["tid"])].append(row)
        kept = []
        file_stats = {"input_rows": len(rows), "input_tracklets": len(groups),
                      "kept_tracklets": 0, "output_rows": 0}
        for tid, group in sorted(groups.items()):
            ok, info = keep_track(group, args)
            details.append({"video": name, "track_id": int(tid), "keep": bool(ok), **info})
            if ok:
                kept.extend(group); file_stats["kept_tracklets"] += 1
                if info["reason"] == "short_keep": stats["short_kept"] += 1
                if info["reason"] == "single_score": stats["single_kept"] += 1
            elif info["reason"] == "short_reject" or info["reason"] == "single_reject":
                stats["short_rejected"] += 1
        kept.sort(key=lambda r: (r["frame"], r["tid"]))
        with (out / path.name).open("w") as f:
            for row in kept:
                f.write(row["raw"] + "\n")
        file_stats["output_rows"] = len(kept)
        stats["files"] += 1; stats["input_rows"] += len(rows)
        stats["output_rows"] += len(kept); stats["input_tracklets"] += len(groups)
        stats["kept_tracklets"] += file_stats["kept_tracklets"]
        stats["per_file"][name] = file_stats
    payload = {
        "schema": "uadetrac-jepa-quality-filter-v1", "status": "complete",
        "prepared_dir": str(prepared), "track_results": str(track_root),
        "jepa_cache": str(cache), "jepa_cache_sha256": sha256(cache),
        "gt_read": False, "target_candidate_in_predictor": False,
        "parameters": vars(args), "stats": stats, "details": details,
        "output_dir": str(out),
    }
    (out / "filter_diagnostics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
    printable = dict(payload)
    printable["details_count"] = len(details)
    printable.pop("details", None)
    print(json.dumps(printable, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
