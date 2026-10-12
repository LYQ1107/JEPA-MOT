#!/usr/bin/env python3
"""Causal-style JEPA tracklet stitching for fixed GeneralTrack outputs.

This is an intentionally conservative post-association experiment.  It does
not alter detector boxes or scores: it only merges two non-overlapping native
track IDs when a source-calibrated JEPA prototype and a constant-velocity
spatial check agree.  The script reads prepared detector rows and a frozen
feature cache, never ``annotations``/GT, and writes the original rows with
canonicalized IDs.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser("JEPA tracklet relinking for GeneralTrack outputs")
    ap.add_argument("--prepared-dir", required=True)
    ap.add_argument("--track-results", required=True)
    ap.add_argument("--jepa-cache", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-gap", type=int, default=50)
    ap.add_argument("--history", type=int, default=4)
    ap.add_argument("--min-sim", type=float, default=0.97)
    ap.add_argument("--max-center-norm", type=float, default=4.0)
    return ap.parse_args()


def unit(x):
    x = np.asarray(x, np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-8)


def box_xyxy(row):
    x, y, w, h = [float(v) for v in row[:4]]
    return np.asarray([x, y, x + w, y + h], np.float32)


def iou(a, b):
    xa = max(float(a[0]), float(b[0])); ya = max(float(a[1]), float(b[1]))
    xb = min(float(a[2]), float(b[2])); yb = min(float(a[3]), float(b[3]))
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    ab = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(aa + ab - inter, 1e-8)


def center(box):
    return np.asarray([(float(box[0]) + float(box[2])) * 0.5,
                       (float(box[1]) + float(box[3])) * 0.5], np.float32)


def safe_name(value):
    value = str(value)
    return value.replace("/", "_").replace("\\", "_")


def load_prepared(path: Path):
    # Deliberately keep this reader restricted to videos/images/detection.
    # The GT annotations field is never accessed.
    data = json.loads((path / "annotations" / "val.json").read_text())
    by_video = defaultdict(list)
    video_names = {}
    for video in data.get("videos", []):
        vid = int(video["id"])
        video_names[safe_name(video.get("file_name", video.get("name", vid)))] = vid
    for image in data.get("images", []):
        by_video[int(image["video_id"])].append(image)
    for vid in by_video:
        by_video[vid].sort(key=lambda x: (int(x.get("frame_id", 0)), int(x["id"])))
    return by_video, video_names


def load_cache(path: Path):
    with np.load(path, allow_pickle=False) as z:
        features = np.asarray(z["features"], np.float32)
        image_ids = np.asarray(z["image_ids"], np.int64)
        offsets = np.asarray(z["offsets"], np.int64)
    if features.ndim != 2 or features.shape[1] != 256:
        raise ValueError("expected [N,256] cache, got {}".format(features.shape))
    if offsets.shape != (len(image_ids) + 1,):
        raise ValueError("invalid cache offsets {}".format(offsets.shape))
    return features, {int(x): i for i, x in enumerate(image_ids.tolist())}, offsets


def parse_track_file(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        p = [x.strip() for x in line.split(",")]
        if len(p) < 7:
            continue
        rows.append({"parts": p, "frame": int(float(p[0])), "tid": int(float(p[1])),
                     "box": box_xyxy([float(x) for x in p[2:6]]),
                     "score": float(p[6])})
    return rows


def attach_features(rows, images, features, image_index, offsets):
    # Prepared UA-DETRAC keeps the original sparse frame_id (e.g. 1, 7, 14)
    # in the runtime MOT rows.  Do not replace it with the list position.
    by_frame = defaultdict(list)
    for image in images:
        by_frame[int(image.get("frame_id", 0))].append(image)
    for row in rows:
        image_list = by_frame.get(int(row["frame"]), [])
        if not image_list:
            row["feature"] = None
            continue
        image = image_list[0]
        pos = image_index.get(int(image["id"]))
        if pos is None:
            row["feature"] = None
            continue
        start, end = int(offsets[pos]), int(offsets[pos + 1])
        detections = image.get("detection", [])
        best = -1.0; best_idx = -1
        for j, det in enumerate(detections):
            if isinstance(det, dict):
                db = np.asarray(det.get("bbox", det.get("box", [])), np.float32)
                bbox_format = det.get("bbox_format")
            else:
                # Prepared GeneralTrack rows are [x1,y1,x2,y2,score,class].
                db = np.asarray(det[:4], np.float32)
                bbox_format = "xyxy"
            if db.shape != (4,):
                continue
            # GeneralTrack's prepared detector rows are xyxy.  Keep support
            # for dict/xywh records used by older preparation utilities.
            if bbox_format == "xyxy" or not isinstance(det, dict):
                cand = db
            else:
                cand = np.asarray([db[0], db[1], db[0] + db[2], db[1] + db[3]], np.float32)
            score = iou(row["box"], cand)
            if score > best:
                best, best_idx = score, j
        if best_idx < 0 or start + best_idx >= end:
            row["feature"] = None
        else:
            row["feature"] = unit(features[start + best_idx])


def tracklets(rows, history):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["tid"])].append(row)
    out = {}
    for tid, obs in grouped.items():
        obs.sort(key=lambda x: x["frame"])
        out[tid] = obs
    return out


def prototype(obs, reverse, history):
    vals = [x["feature"] for x in (obs[-history:] if reverse else obs[:history])
            if x.get("feature") is not None]
    if not vals:
        return None
    return unit(np.mean(np.asarray(vals, np.float32), axis=0))


def predicted_center(obs, target_frame):
    tail = obs[-5:]
    if len(tail) < 2:
        return center(obs[-1]["box"])
    a, b = tail[0], tail[-1]
    dt = max(1, int(b["frame"]) - int(a["frame"]))
    velocity = (center(b["box"]) - center(a["box"])) / float(dt)
    return center(b["box"]) + velocity * float(int(target_frame) - int(b["frame"]))


def relink(rows, max_gap, history, min_sim, max_center_norm):
    groups = tracklets(rows, history)
    candidates = []
    tids = sorted(groups)
    for a in tids:
        oa = groups[a]
        pa = prototype(oa, True, history)
        if pa is None:
            continue
        for b in tids:
            if a == b:
                continue
            ob = groups[b]
            if int(oa[-1]["frame"]) >= int(ob[0]["frame"]):
                continue
            gap = int(ob[0]["frame"]) - int(oa[-1]["frame"]) - 1
            if gap < 0 or gap > int(max_gap):
                continue
            pb = prototype(ob, False, history)
            if pb is None:
                continue
            sim = float(np.dot(pa, pb))
            pred = predicted_center(oa, ob[0]["frame"])
            scale = max(20.0, math.sqrt(max(1.0, float((oa[-1]["box"][2] - oa[-1]["box"][0]) *
                                                        (oa[-1]["box"][3] - oa[-1]["box"][1])))))
            spatial = float(np.linalg.norm(pred - center(ob[0]["box"])) / scale)
            if sim >= float(min_sim) and spatial <= float(max_center_norm):
                candidates.append({"a": a, "b": b, "sim": sim, "spatial_norm": spatial, "gap": gap})
    candidates.sort(key=lambda x: (-x["sim"], x["spatial_norm"], x["gap"], x["a"], x["b"]))
    successor = set(); predecessor = set(); parent = {tid: tid for tid in tids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    accepted = []
    for edge in candidates:
        a, b = edge["a"], edge["b"]
        if a in successor or b in predecessor or find(a) == find(b):
            continue
        # Keep the earlier native ID as the canonical identity.
        parent[find(b)] = find(a)
        successor.add(a); predecessor.add(b); accepted.append(edge)
    mapping = {tid: find(tid) for tid in tids}
    for row in rows:
        row["new_tid"] = int(mapping[int(row["tid"])])
    return rows, {"tracklet_count": len(tids), "candidate_count": len(candidates),
                  "merge_count": len(accepted), "accepted": accepted,
                  "min_sim": float(min_sim), "max_gap": int(max_gap),
                  "max_center_norm": float(max_center_norm), "gt_read": False}


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: (x["frame"], x["tid"]))
    with path.open("w") as f:
        for row in rows:
            p = list(row["parts"])
            p[1] = str(int(row["new_tid"]))
            f.write(",".join(p) + "\n")


def main():
    args = parse_args()
    prepared = Path(args.prepared_dir).resolve()
    track_root = Path(args.track_results).resolve()
    out_root = Path(args.output_dir).resolve()
    cache = Path(args.jepa_cache).resolve()
    by_video, video_names = load_prepared(prepared)
    features, image_index, offsets = load_cache(cache)
    output_meta = {"schema": "uadetrac-jepa-tracklet-relink-v1", "status": "complete",
                   "prepared_dir": str(prepared), "track_results": str(track_root),
                   "jepa_cache": str(cache), "gt_read": False,
                   "target_candidate_in_predictor": False, "videos": [],
                   "command": " ".join(__import__("sys").argv)}
    total_merges = 0
    for track_path in sorted(track_root.glob("*.txt")):
        name = safe_name(track_path.stem)
        vid = video_names.get(name)
        if vid is None:
            continue
        rows = parse_track_file(track_path)
        attach_features(rows, by_video[vid], features, image_index, offsets)
        rows, meta = relink(rows, args.max_gap, args.history, args.min_sim, args.max_center_norm)
        write_rows(out_root / track_path.name, rows)
        meta.update({"video": name, "rows": len(rows)})
        output_meta["videos"].append(meta); total_merges += int(meta["merge_count"])
    output_meta["video_count"] = len(output_meta["videos"])
    output_meta["total_merges"] = total_merges
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "relink_diagnostics.json").write_text(json.dumps(output_meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output_meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
