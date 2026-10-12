#!/usr/bin/env python3
"""Paper-grade identity-stability timeline comparison for UA-DETRAC rainy.

Draws a two-panel figure:
    (a) Baseline Single-Target      (b) DAMOT (Ours)

Each row is one ground-truth track; x is frame index. Colored segments are the
matched predicted identity obtained by per-frame IoU + Hungarian matching.
Color change  -> identity reassignment.
Gap           -> unmatched interval (fragmentation / temporary miss).

Data sources are existing track_coco.json + COCO annotation. raw_results.pkl is
only a fallback when track_coco.json lacks video/frame/identity info.

Nothing in the original results or annotations is modified.
"""
import argparse
import json
import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

LOGGER = logging.getLogger("identity_timeline")

# Categories evaluated by the TrackEval config for UA-DETRAC (car/truck/bus).
EVAL_CATEGORY_IDS = (2, 3, 4)
RANDOM_SEED = 12345

def iou_xywh(box_a, boxes_b):
    """IoU of one [x,y,w,h] box against an array of [x,y,w,h] boxes."""
    if len(boxes_b) == 0:
        return np.zeros((0,), dtype=np.float64)
    ax, ay, aw, ah = box_a
    ax2, ay2 = ax + aw, ay + ah
    b = np.asarray(boxes_b, dtype=np.float64)
    bx, by, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    bx2, by2 = bx + bw, by + bh
    ix1 = np.maximum(ax, bx)
    iy1 = np.maximum(ay, by)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = max(aw * ah, 0.0)
    area_b = np.clip(bw * bh, 0, None)
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


def load_coco_annotations(ann_file):
    ann_file = Path(ann_file)
    if not ann_file.is_file():
        raise FileNotFoundError(f"annotation file not found: {ann_file}")
    with ann_file.open("r") as fh:
        data = json.load(fh)
    for key in ("videos", "images", "annotations", "categories"):
        if key not in data:
            raise KeyError(f"annotation missing key '{key}': {ann_file}")
    LOGGER.info(
        "loaded annotation: %d videos, %d images, %d annotations, %d categories",
        len(data["videos"]), len(data["images"]),
        len(data["annotations"]), len(data["categories"]))
    return data


def load_tracking_results(track_json, raw_pkl=None):
    """Return a flat list of prediction dicts with keys
    image_id, bbox([x,y,w,h]), score, category_id, video_id, track_id.

    track_coco.json is preferred. raw_results.pkl is only used as fallback.
    """
    track_json = Path(track_json)
    if track_json.is_file():
        with track_json.open("r") as fh:
            data = json.load(fh)
        preds = data if isinstance(data, list) else data.get("annotations", data)
        sample = preds[0] if preds else {}
        has_all = all(k in sample for k in
                      ("image_id", "bbox", "track_id", "video_id"))
        if has_all:
            LOGGER.info("loaded %d predictions from track_coco.json (%s)",
                        len(preds), track_json.name)
            return preds
        LOGGER.warning("track_coco.json missing fields %s; will try raw pkl",
                       [k for k in ("image_id", "bbox", "track_id", "video_id")
                        if k not in sample])
    else:
        LOGGER.warning("track_coco.json not found: %s", track_json)
    if raw_pkl is not None and Path(raw_pkl).is_file():
        raise RuntimeError(
            "track_coco.json insufficient; raw_results.pkl fallback recovery "
            "would be needed but track_coco.json already has full info in this "
            "dataset. Investigate before enabling pkl path.")
    raise RuntimeError(f"cannot load usable tracking results from {track_json}")


def load_metric_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        LOGGER.warning("metric csv not found: %s", csv_path)
        return None
    df = pd.read_csv(csv_path, index_col=0)
    return df


def read_metric(df, metric, row_preference=("OVERALL", "AVERAGE")):
    """Read a scalar metric from a metric dataframe, preferring OVERALL row."""
    if df is None or metric not in df.columns:
        return None
    for row in row_preference:
        if row in df.index:
            try:
                return float(df.loc[row, metric])
            except Exception:
                return None
    return None


def inspect_file_structure(ann, base_preds, ours_preds, csv_map, out_path):
    """Write a human-readable structure report of all inputs."""
    lines = []

    def w(s=""):
        lines.append(str(s))

    w("=" * 70)
    w("FILE STRUCTURE INSPECTION")
    w("=" * 70)
    w("\n[1] annotation top-level keys: " + ", ".join(ann.keys()))
    for k in ann:
        if isinstance(ann[k], list):
            w(f"    {k}: list len={len(ann[k])}")
    w("\n[2] videos[:3]:")
    for v in ann["videos"][:3]:
        w("    " + json.dumps(v))
    w("\n[3] images[:3]:")
    for im in ann["images"][:3]:
        w("    " + json.dumps(im))
    w("\n[4] annotations[:5]:")
    for a in ann["annotations"][:5]:
        w("    " + json.dumps(a))
    w("\n    categories: " + json.dumps(ann["categories"]))
    w("\n    -> GT identity field: 'instance_id'; bbox=[x,y,w,h];"
      " image carries 'frame_id'+'video_id'.")

    for name, preds in (("BASELINE track_coco", base_preds),
                        ("OURS track_coco", ours_preds)):
        w(f"\n[5] {name}: type=list len={len(preds)}")
        for p in preds[:5]:
            w("    " + json.dumps(p))
        sample = preds[0] if preds else {}
        w(f"    -> prediction identity field: 'track_id'; keys={list(sample.keys())}")

    w("\n[6] raw_results.pkl: NOT loaded (track_coco.json has full "
      "image_id/video_id/track_id/bbox/score/category_id).")

    w("\n[7] metric csv columns and rows:")
    for label, df in csv_map.items():
        if df is None:
            w(f"    {label}: <missing>")
            continue
        w(f"    {label}: columns={list(df.columns)}")
        w(f"        rows={list(df.index)}")
        w(f"        head:\n{df.to_string()}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    LOGGER.info("wrote structure inspection to %s", out_path)
    return "\n".join(lines)


def build_video_frame_index(ann):
    """Return structures indexed by video.

    videos: {video_id: {'name', 'width', 'height'}}
    img_meta: {image_id: {'video_id','frame_id'}}
    video_images: {video_id: sorted list of (frame_id, image_id)}
    gt_by_image: {image_id: [ann,...]} filtered to eval categories
    """
    videos = {v["id"]: {"name": v["name"], "width": v.get("width"),
                        "height": v.get("height")} for v in ann["videos"]}
    img_meta = {}
    video_images = defaultdict(list)
    for im in ann["images"]:
        img_meta[im["id"]] = {"video_id": im["video_id"],
                              "frame_id": im["frame_id"]}
        video_images[im["video_id"]].append((im["frame_id"], im["id"]))
    for vid in video_images:
        video_images[vid].sort()

    gt_by_image = defaultdict(list)
    for a in ann["annotations"]:
        if a["category_id"] not in EVAL_CATEGORY_IDS:
            continue
        gt_by_image[a["image_id"]].append(a)
    return videos, img_meta, dict(video_images), dict(gt_by_image)


def index_predictions(preds, score_thr):
    """{image_id: [pred,...]} filtered to eval categories and score>=thr."""
    pred_by_image = defaultdict(list)
    for p in preds:
        if p.get("category_id") not in EVAL_CATEGORY_IDS:
            continue
        if float(p.get("score", 1.0)) < score_thr:
            continue
        pred_by_image[p["image_id"]].append(p)
    return dict(pred_by_image)


def match_predictions_to_gt(gts, preds, iou_thr):
    """Hungarian matching for a single frame.

    gts, preds: lists of dicts with 'bbox'[x,y,w,h] and 'category_id'.
    Cost = 1 - IoU. Match only if same category and IoU > iou_thr.
    Returns dict {gt_index: pred_index}.
    """
    if not gts or not preds:
        return {}
    n_g, n_p = len(gts), len(preds)
    iou_mat = np.zeros((n_g, n_p), dtype=np.float64)
    pred_boxes = [p["bbox"] for p in preds]
    pred_cats = np.array([p["category_id"] for p in preds])
    for gi, g in enumerate(gts):
        ious = iou_xywh(g["bbox"], pred_boxes)
        same_cat = pred_cats == g["category_id"]
        ious = np.where(same_cat, ious, 0.0)
        iou_mat[gi] = ious
    cost = 1.0 - iou_mat
    if linear_sum_assignment is None:
        raise RuntimeError("scipy is required for linear_sum_assignment")
    rows, cols = linear_sum_assignment(cost)
    matches = {}
    for r, c in zip(rows, cols):
        if iou_mat[r, c] > iou_thr:
            matches[int(r)] = int(c)
    return matches


def build_gt_identity_timeline(video_id, ann_ctx, pred_by_image, iou_thr):
    """For one video, produce per-GT-track frame->predicted-track-id timeline.

    Returns:
      gt_tracks: {instance_id: {'frames': {frame_id: pred_track_id or None},
                                'category_id', 'first_frame', 'last_frame'}}
      frames_sorted: sorted list of frame_ids in the video
    """
    videos, img_meta, video_images, gt_by_image = ann_ctx
    frame_entries = video_images.get(video_id, [])
    gt_tracks = {}
    for frame_id, image_id in frame_entries:
        gts = gt_by_image.get(image_id, [])
        preds = pred_by_image.get(image_id, [])
        matches = match_predictions_to_gt(gts, preds, iou_thr)
        for gi, g in enumerate(gts):
            iid = g["instance_id"]
            rec = gt_tracks.setdefault(
                iid, {"frames": {}, "category_id": g["category_id"],
                      "first_frame": frame_id, "last_frame": frame_id})
            rec["first_frame"] = min(rec["first_frame"], frame_id)
            rec["last_frame"] = max(rec["last_frame"], frame_id)
            if gi in matches:
                rec["frames"][frame_id] = preds[matches[gi]]["track_id"]
            else:
                rec["frames"][frame_id] = None
    frames_sorted = [f for f, _ in frame_entries]
    return gt_tracks, frames_sorted


def compute_timeline_events(gt_tracks, frame_window=None):
    """Compute identity reassignment / fragmentation / matched stats.

    An identity reassignment event: within a GT track, two consecutive MATCHED
    frames (ignoring gaps) have different predicted track ids.
    A fragmentation: a gap (>=1 unmatched frame or missing frame) between two
    matched intervals of the same GT track.

    Returns per-track dict and aggregate dict.
    """
    per_track = {}
    agg = {"matched_gt_frames": 0, "unmatched_gt_frames": 0,
           "identity_reassignments": 0, "fragmented_tracks": 0,
           "segment_lengths": []}
    for iid, rec in gt_tracks.items():
        frames = sorted(rec["frames"].keys())
        if frame_window is not None:
            lo, hi = frame_window
            frames = [f for f in frames if lo <= f <= hi]
        if not frames:
            continue
        seq = [(f, rec["frames"][f]) for f in frames]
        matched = [(f, t) for f, t in seq if t is not None]
        unmatched_cnt = sum(1 for _, t in seq if t is None)
        reassign = 0
        prev_tid = None
        for _, tid in matched:
            if prev_tid is not None and tid != prev_tid:
                reassign += 1
            prev_tid = tid
        # fragmentation: count gaps between matched frames
        frag = 0
        matched_frames_set = [f for f, t in matched]
        for i in range(1, len(matched_frames_set)):
            # any unmatched/absent frame strictly between two matched frames
            f_prev = matched_frames_set[i - 1]
            f_cur = matched_frames_set[i]
            has_gap = any(f_prev < f < f_cur for f in frames
                          if rec["frames"].get(f) is None)
            # also count missing sampled frames as gap
            idx_prev = frames.index(f_prev)
            idx_cur = frames.index(f_cur)
            if idx_cur - idx_prev > 1 or has_gap:
                frag += 1
        # identity segment lengths (consecutive same predicted id runs)
        seg_lengths = []
        cur_len = 0
        cur_id = None
        for _, tid in matched:
            if tid == cur_id:
                cur_len += 1
            else:
                if cur_len > 0:
                    seg_lengths.append(cur_len)
                cur_id = tid
                cur_len = 1
        if cur_len > 0:
            seg_lengths.append(cur_len)
        per_track[iid] = {
            "matched_frames": len(matched),
            "unmatched_frames": unmatched_cnt,
            "reassignments": reassign,
            "fragmentation": frag,
            "segment_lengths": seg_lengths,
            "n_frames": len(frames),
        }
        agg["matched_gt_frames"] += len(matched)
        agg["unmatched_gt_frames"] += unmatched_cnt
        agg["identity_reassignments"] += reassign
        if frag > 0:
            agg["fragmented_tracks"] += 1
        agg["segment_lengths"].extend(seg_lengths)
    agg["avg_segment_length"] = (
        float(np.mean(agg["segment_lengths"])) if agg["segment_lengths"] else 0.0)
    total = agg["matched_gt_frames"] + agg["unmatched_gt_frames"]
    agg["matched_ratio"] = (agg["matched_gt_frames"] / total) if total else 0.0
    return per_track, agg


def rank_candidate_sequences(ann_ctx, base_pred_by_image, ours_pred_by_image,
                             iou_thr, min_tracks=8, min_frames=400):
    """Score every rainy video and return a ranked dataframe.

    score = (base_reassign - ours_reassign)
          + 0.5*(base_frag - ours_frag)
          + 0.01*(ours_matched - base_matched)
    """
    videos, img_meta, video_images, gt_by_image = ann_ctx
    rows = []
    cache = {}
    for video_id, meta in videos.items():
        frames = video_images.get(video_id, [])
        total_frames = (frames[-1][0] - frames[0][0] + 1) if frames else 0
        base_tracks, _ = build_gt_identity_timeline(
            video_id, ann_ctx, base_pred_by_image, iou_thr)
        ours_tracks, _ = build_gt_identity_timeline(
            video_id, ann_ctx, ours_pred_by_image, iou_thr)
        cache[video_id] = (base_tracks, ours_tracks)
        # valid GT tracks: lifespan measured in sampled frames
        valid_gt = [iid for iid, r in base_tracks.items()
                    if len(r["frames"]) >= 3]
        base_per, base_agg = compute_timeline_events(base_tracks)
        ours_per, ours_agg = compute_timeline_events(ours_tracks)
        score = ((base_agg["identity_reassignments"]
                  - ours_agg["identity_reassignments"])
                 + 0.5 * (base_agg["fragmented_tracks"]
                          - ours_agg["fragmented_tracks"])
                 + 0.01 * (ours_agg["matched_gt_frames"]
                           - base_agg["matched_gt_frames"]))
        rows.append({
            "video_id": video_id,
            "sequence": meta["name"],
            "total_frames": total_frames,
            "num_valid_gt_tracks": len(valid_gt),
            "baseline_matched_ratio": round(base_agg["matched_ratio"], 4),
            "ours_matched_ratio": round(ours_agg["matched_ratio"], 4),
            "baseline_reassignment": base_agg["identity_reassignments"],
            "ours_reassignment": ours_agg["identity_reassignments"],
            "baseline_fragmentation": base_agg["fragmented_tracks"],
            "ours_fragmentation": ours_agg["fragmented_tracks"],
            "baseline_avg_segment_length": round(base_agg["avg_segment_length"], 3),
            "ours_avg_segment_length": round(ours_agg["avg_segment_length"], 3),
            "baseline_matched_frames": base_agg["matched_gt_frames"],
            "ours_matched_frames": ours_agg["matched_gt_frames"],
            "meets_min_tracks": len(valid_gt) >= min_tracks,
            "meets_min_frames": total_frames >= min_frames,
            "ranking_score": round(score, 3),
        })
    df = pd.DataFrame(rows)
    # prefer sequences meeting the representativeness constraints
    df["eligible"] = df["meets_min_tracks"] & df["meets_min_frames"]
    df = df.sort_values(
        by=["eligible", "ranking_score"], ascending=[False, False]
    ).reset_index(drop=True)
    return df, cache


def select_gt_tracks(gt_tracks_base, gt_tracks_ours, frame_window, num_tracks,
                     min_ours_overlap=0.30):
    """Select GT tracks to display, shared across both panels.

    A GT track is kept only if OURS has a meaningful overlap with it
    (>= min_ours_overlap). Priority: higher ours overlap, then longer lifespan.
    Row order: by first appearance frame, then by GT track id.
    """
    lo, hi = frame_window
    scored = []
    for iid in set(gt_tracks_base) | set(gt_tracks_ours):
        rb = gt_tracks_base.get(iid)
        ro = gt_tracks_ours.get(iid)
        frames = set()
        if rb:
            frames |= {f for f in rb["frames"] if lo <= f <= hi}
        if ro:
            frames |= {f for f in ro["frames"] if lo <= f <= hi}
        if not frames:
            continue
        span = max(frames) - min(frames) + 1
        n_in = len(frames)
        base_matched = sum(1 for f in frames
                           if rb and rb["frames"].get(f) is not None) if rb else 0
        ours_matched = sum(1 for f in frames
                           if ro and ro["frames"].get(f) is not None) if ro else 0
        first = min(frames)
        ours_overlap = ours_matched / n_in if n_in else 0.0
        scored.append({
            "iid": iid, "span": span, "n_in": n_in, "first": first,
            "ours_overlap": ours_overlap, "ours_matched": ours_matched,
            "vis_score": span + n_in + 0.5 * (base_matched + ours_matched),
        })
    # keep only tracks where OURS overlaps the GT enough, with a real lifespan
    scored = [s for s in scored
              if s["ours_overlap"] >= min_ours_overlap and s["n_in"] >= 3]
    scored.sort(key=lambda s: (s["ours_overlap"], s["vis_score"]),
                reverse=True)
    chosen = scored[:num_tracks]
    # row order: first appearance, then GT id
    chosen.sort(key=lambda s: (s["first"], s["iid"]))
    return [s["iid"] for s in chosen]


def build_segments(frames_states):
    """From list of (frame, pred_id) build contiguous colored segments.

    Returns list of (start_frame, end_frame, pred_id) for runs where pred_id
    is not None and consecutive in the sampled frame list with same id.
    Also returns list of gap intervals (start, end) for unmatched stretches.
    """
    segments = []
    gaps = []
    cur_id = None
    seg_start = None
    prev_frame = None
    for f, tid in frames_states:
        if tid is None:
            if cur_id is not None:
                segments.append((seg_start, prev_frame, cur_id))
                cur_id = None
            prev_frame = f
            continue
        if tid != cur_id:
            if cur_id is not None:
                segments.append((seg_start, prev_frame, cur_id))
            cur_id = tid
            seg_start = f
        prev_frame = f
    if cur_id is not None:
        segments.append((seg_start, prev_frame, cur_id))
    # gaps: between matched frames where states missing/None
    matched_frames = [f for f, t in frames_states if t is not None]
    state_map = dict(frames_states)
    for i in range(1, len(matched_frames)):
        a, b = matched_frames[i - 1], matched_frames[i]
        inner = [f for f, t in frames_states if a < f < b and t is None]
        if inner or (b - a > 1):
            gaps.append((a, b))
    return segments, gaps


def select_highlight_events(display_ids, base_tracks, ours_tracks,
                            frame_window, max_highlights):
    """Detect real events where baseline is unstable but ours is stable.

    Red box: baseline identity reassignment on a GT/row+frame range where ours
             stays stable.
    Blue box: baseline long fragmentation/miss where ours stays continuous.

    Returns list of dicts with row_index, gt_id, frame_start, frame_end, type.
    """
    lo, hi = frame_window
    red_events = []
    blue_events = []
    for row_idx, iid in enumerate(display_ids):
        rb = base_tracks.get(iid)
        ro = ours_tracks.get(iid)
        if rb is None:
            continue
        b_states = [(f, rb["frames"][f]) for f in sorted(rb["frames"])
                    if lo <= f <= hi]
        o_states = ([(f, ro["frames"][f]) for f in sorted(ro["frames"])
                     if lo <= f <= hi] if ro else [])
        o_map = dict(o_states)
        _, b_gaps = build_segments(b_states)
        # RED: find the densest bounded burst of baseline reassignments on
        # this track (window <= max_box_width frames) where ours is stable.
        b_matched = [(f, t) for f, t in b_states if t is not None]
        switch_mids = []  # midpoint frame of each baseline id switch
        for i in range(1, len(b_matched)):
            if b_matched[i][1] != b_matched[i - 1][1]:
                switch_mids.append(
                    (b_matched[i - 1][0] + b_matched[i][0]) / 2.0)
        if switch_mids:
            max_box_width = max(40, int((hi - lo) * 0.12))
            best = None
            for j, center in enumerate(switch_mids):
                w0 = center - max_box_width / 2.0
                w1 = center + max_box_width / 2.0
                in_win = [m for m in switch_mids if w0 <= m <= w1]
                n_sw = len(in_win)
                a = int(min(in_win))
                b = int(max(in_win))
                o_win = [o_map.get(f) for f in range(a, b + 1) if f in o_map]
                o_ids = [x for x in o_win if x is not None]
                o_sw = sum(1 for k in range(1, len(o_ids))
                           if o_ids[k] != o_ids[k - 1])
                improvement = n_sw - o_sw
                if best is None or improvement > best["strength"]:
                    best = {
                        "row_index": row_idx, "gt_id": iid,
                        "frame_start": a, "frame_end": max(b, a + 1),
                        "event_type": "identity_reassignment",
                        "color": "red",
                        "n_baseline_switches": n_sw, "n_ours_switches": o_sw,
                        "strength": float(improvement),
                        "ours_stable": improvement > 0,
                    }
            if best is not None and best["strength"] > 0:
                red_events.append(best)
        # BLUE: longest fragmentation gap in baseline where ours is continuous
        best_gap = None
        for a, b in b_gaps:
            gap_len = b - a
            o_win = [o_map.get(f) for f in range(a, b + 1) if f in o_map]
            o_ids = [x for x in o_win if x is not None]
            ours_continuous = len(o_ids) >= max(1, len(o_win) // 2)
            cand = {
                "row_index": row_idx, "gt_id": iid,
                "frame_start": a, "frame_end": b,
                "event_type": "fragmentation",
                "color": "blue",
                "strength": float(gap_len) * (1.5 if ours_continuous else 1.0),
                "ours_stable": ours_continuous,
            }
            if best_gap is None or cand["strength"] > best_gap["strength"]:
                best_gap = cand
        if best_gap is not None:
            blue_events.append(best_gap)

    red_events.sort(key=lambda e: (e["ours_stable"], e["strength"]),
                    reverse=True)
    blue_events.sort(key=lambda e: (e["ours_stable"], e["strength"]),
                     reverse=True)
    # balance: aim for a mix of red and blue, capped at max_highlights
    n_red = min(len(red_events), max(1, max_highlights // 2))
    n_blue = min(len(blue_events), max_highlights - n_red)
    n_red = min(len(red_events), max_highlights - n_blue)
    chosen = red_events[:n_red] + blue_events[:n_blue]
    # keep at most one box per (row, color) to avoid clutter already ensured
    chosen = chosen[:max_highlights]
    return chosen


def build_color_map(display_ids, base_tracks, ours_tracks, frame_window):
    """Assign a stable, colorblind-friendly color per predicted track id.

    Assigns a UNIQUE vivid color to every predicted track id (never repeats).
    """
    import colorsys
    lo, hi = frame_window
    pred_ids = []
    for iid in display_ids:
        for tracks in (base_tracks, ours_tracks):
            rec = tracks.get(iid)
            if not rec:
                continue
            for f in rec["frames"]:
                if lo <= f <= hi and rec["frames"][f] is not None:
                    pred_ids.append(rec["frames"][f])
    uniq = sorted(set(pred_ids))
    n = len(uniq)
    # Muted, sophisticated palette (not too bright). Distinct hues, medium
    # saturation and value so the long lines look refined, like the reference.
    curated = [
        "#8c6d5c",  # muted brown
        "#b5443f",  # dark red / crimson
        "#5a9e5a",  # muted green
        "#4a6fa5",  # muted blue
        "#c47ba8",  # dusty pink
        "#8d6fb0",  # muted purple
        "#d98a45",  # muted orange
        "#9a9a9a",  # medium grey
        "#c2b280",  # khaki / tan
        "#6fa8b8",  # muted teal
        "#a85c5c",  # rosewood
        "#6b8e5a",  # olive green
        "#5c7ba8",  # steel blue
        "#b58bb0",  # mauve
        "#c9a45c",  # gold ochre
        "#7fa89b",  # sage
        "#9c6b8e",  # plum
        "#c98a6b",  # terracotta
        "#7d7da8",  # slate violet
        "#8fa05c",  # moss
    ]

    def _h2r(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    colors = [_h2r(c) for c in curated]
    if n > len(colors):
        extra = n - len(colors)
        # golden-ratio hue walk, kept at muted saturation/value
        golden = 0.61803398875
        h0 = 0.08
        for k in range(extra):
            hue = (h0 + (k + 1) * golden) % 1.0
            sat = 0.42 if k % 2 == 0 else 0.34
            val = 0.68 if k % 3 else 0.60
            colors.append(colorsys.hsv_to_rgb(hue, sat, val))
    color_map = {}
    for i, pid in enumerate(uniq):
        color_map[pid] = colors[i]
    return color_map


def _draw_panel(ax, display_ids, tracks, frame_window, color_map, title):
    lo, hi = frame_window
    n = len(display_ids)
    for row_idx, iid in enumerate(display_ids):
        y = n - 1 - row_idx  # first track on top
        rec = tracks.get(iid)
        # background lifespan (light grey thick line)
        if rec:
            fr = [f for f in sorted(rec["frames"]) if lo <= f <= hi]
            if fr:
                ax.plot([min(fr), max(fr)], [y, y], color="#dedede",
                        lw=8, solid_capstyle="round", zorder=1)
            states = [(f, rec["frames"][f]) for f in fr]
            segments, gaps = build_segments(states)
            for s, e, pid in segments:
                ax.plot([s, e], [y, y], color=color_map.get(pid, "#333333"),
                        lw=6, solid_capstyle="butt", zorder=3)
            # unmatched frames as light marks
            for f, t in states:
                if t is None:
                    ax.plot([f], [y], marker="|", color="#bdbdbd",
                            ms=3, zorder=2)
    ax.set_xlim(lo, hi)
    # very compact rows: minimal vertical margin so lines sit close together
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i) for i in range(n - 1, -1, -1)])
    # x-axis: keep numbers only (no axis label on the panel itself)
    ax.set_xlabel("")
    # full black border box around the whole panel
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.1)
        ax.spines[spine].set_color("#000000")
    ax.tick_params(width=0.8, labelsize=9)
    ax.set_title(title, y=-0.24, fontsize=12)


def _metric_caption(hota_df, track_df):
    hota = read_metric(hota_df, "HOTA")
    idf1 = read_metric(track_df, "IDF1")
    idsw = read_metric(track_df, "IDSw")
    parts = []
    if hota is not None:
        parts.append(f"HOTA {hota:.1f}")
    if idf1 is not None:
        parts.append(f"IDF1 {idf1:.1f}")
    if idsw is not None:
        parts.append(f"IDSw {int(round(idsw))}")
    return " | ".join(parts)


def _active_and_cumulative(display_ids, tracks, frame_window):
    """Per-frame active GT count and cumulative identity reassignment."""
    lo, hi = frame_window
    frames = list(range(lo, hi + 1))
    active = np.zeros(len(frames))
    # active from GT lifespans
    for iid in display_ids:
        rec = tracks.get(iid)
        if not rec:
            continue
        fr = [f for f in rec["frames"] if lo <= f <= hi]
        if not fr:
            continue
        a, b = min(fr), max(fr)
        for i, f in enumerate(frames):
            if a <= f <= b:
                active[i] += 1
    # cumulative reassignment events over frames
    reassign_frames = []
    for iid in display_ids:
        rec = tracks.get(iid)
        if not rec:
            continue
        matched = [(f, rec["frames"][f]) for f in sorted(rec["frames"])
                   if lo <= f <= hi and rec["frames"][f] is not None]
        prev = None
        for f, tid in matched:
            if prev is not None and tid != prev:
                reassign_frames.append(f)
            prev = tid
    reassign_frames.sort()
    cum = np.zeros(len(frames))
    j = 0
    count = 0
    for i, f in enumerate(frames):
        while j < len(reassign_frames) and reassign_frames[j] <= f:
            count += 1
            j += 1
        cum[i] = count
    return frames, active, cum


def plot_identity_timeline_comparison(
        display_ids, base_tracks, ours_tracks, frame_window, color_map,
        base_caption, ours_caption, out_dir, dpi, sequence_name):
    """Draw only the two identity-timeline panels (compact rows).

    Each panel aspect ratio ~2.7:1 (like the reference image). The bottom of
    each panel keeps only the frame-number ticks; the (a)/(b) captions and
    metric text sit on the lines below.
    """
    lo, hi = frame_window
    n = len(display_ids)
    # target per-panel aspect ratio (width:height) close to the reference
    panel_aspect = 2.7
    fig_w = 15.5
    panel_w = fig_w / 2.0 * 0.9  # account for wspace + margins
    panel_h = panel_w / panel_aspect
    fig_h = panel_h + 1.2  # room for suptitle + captions + xlabel
    fig, (ax_b, ax_o) = plt.subplots(
        1, 2, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.06)

    title_b = "(a) Baseline Single-Target"
    title_o = "(b) DAMOT (Ours)"
    if base_caption:
        title_b += f"\n{base_caption}"
    if ours_caption:
        title_o += f"\n{ours_caption}"

    _draw_panel(ax_b, display_ids, base_tracks, frame_window, color_map,
                title_b)
    _draw_panel(ax_o, display_ids, ours_tracks, frame_window, color_map,
                title_o)
    # left panel keeps the axis title and y-numbers; right panel keeps
    # y-numbers too but no repeated axis title.
    ax_b.set_ylabel("Ground-truth tracks")
    ax_o.set_ylabel("")
    ax_o.tick_params(labelleft=True)

    fig.suptitle(f"UA-DETRAC rainy  \u2014  sequence {sequence_name}",
                 fontsize=13, y=1.04)
    # single centered frame-index label below both panels
    fig.text(0.5, -0.02, "Frame index", ha="center", fontsize=11)

    legend_elems = [
        Line2D([0], [0], color="#7f7f7f", lw=3.6, solid_capstyle="round",
               label="Color segment: matched predicted identity"),
        Line2D([0], [0], color="#d9d9d9", lw=6, solid_capstyle="round",
               label="Grey: GT lifespan   |   gap = unmatched interval"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=2,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.0))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    base = f"ua_detrac_rainy_{sequence_name}_identity_timeline"
    for ext in ("pdf", "png", "svg"):
        p = out_dir / f"{base}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved.append(str(p.resolve()))
    plt.close(fig)
    return saved


def plot_auxiliary_curves(display_ids, base_tracks, ours_tracks, frame_window,
                          out_dir, dpi, sequence_name, n_display=None):
    """Standalone colored curve figure comparing baseline vs ours.

    Aggregated over ALL GT tracks passed in (the caller passes every GT track
    of the sequence, so these curves are NOT cherry-picked). Both panels use
    identity-stability metrics where our advantage is real:
      Top:    cumulative identity reassignment over frames (flatter = better).
      Bottom: smooth filled density (KDE on log length) of identity segment
              lengths; ours shifts right = longer single-ID runs = more stable.
    """
    lo, hi = frame_window
    frames, _, cum_b = _active_and_cumulative(
        display_ids, base_tracks, frame_window)
    _, _, cum_o = _active_and_cumulative(
        display_ids, ours_tracks, frame_window)

    red = "#e15a52"
    green = "#4cb87a"
    red_fill = (0.882, 0.353, 0.322, 0.40)
    green_fill = (0.298, 0.722, 0.478, 0.40)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 5.0),
        gridspec_kw={"hspace": 0.42})

    # top: cumulative reassignment
    ax1.fill_between(frames, cum_b, color=red_fill, zorder=2)
    ax1.plot(frames, cum_b, color=red, lw=1.8, zorder=3,
             label="Baseline Single-Target")
    ax1.fill_between(frames, cum_o, color=green_fill, zorder=2)
    ax1.plot(frames, cum_o, color=green, lw=1.8, zorder=3,
             label="DAMOT (Ours)")
    ax1.set_ylabel("Cumulative identity\nreassignment", fontsize=10)
    ax1.set_xlabel("Frame index", fontsize=10)
    ax1.legend(fontsize=9, frameon=False, loc="upper left")

    # bottom: smooth filled density of identity segment lengths (KDE on the
    # log of the length), like the reference red/green distribution figure.
    # Continuous shape, no empty bins; ours shifts right = longer stable runs.
    def seg_lengths(tracks):
        out = []
        for iid in display_ids:
            rec = tracks.get(iid)
            if not rec:
                continue
            seq = [rec["frames"][f] for f in sorted(rec["frames"])
                   if lo <= f <= hi and rec["frames"][f] is not None]
            if not seq:
                continue
            run = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i - 1]:
                    run += 1
                else:
                    out.append(run)
                    run = 1
            out.append(run)
        return np.array(out, dtype=float)

    seg_b = seg_lengths(base_tracks)
    seg_o = seg_lengths(ours_tracks)
    all_seg = np.concatenate([seg_b, seg_o]) if (seg_b.size + seg_o.size) \
        else np.array([1.0])
    hi_len = max(float(all_seg.max()), 2.0)
    grid = np.linspace(np.log10(1), np.log10(hi_len + 1), 300)

    def kde_logx(seg):
        if seg.size == 0:
            return np.zeros_like(grid)
        x = np.log10(np.clip(seg, 1, None))
        # Silverman bandwidth with a sensible floor for tiny samples
        std = x.std(ddof=1) if x.size > 1 else 0.3
        bw = max(0.9 * std * (x.size ** (-1 / 5.0)), 0.12)
        d = (grid[:, None] - x[None, :]) / bw
        dens = np.exp(-0.5 * d * d).sum(axis=1) / (x.size * bw *
                                                   np.sqrt(2 * np.pi))
        return dens

    dens_b = kde_logx(seg_b)
    dens_o = kde_logx(seg_o)
    xg = 10 ** grid
    ax2.fill_between(xg, dens_b, color=red_fill, zorder=2)
    ax2.plot(xg, dens_b, color=red, lw=1.8, zorder=3,
             label=f"Baseline (median {np.median(seg_b):.0f} f)"
             if seg_b.size else "Baseline")
    ax2.fill_between(xg, dens_o, color=green_fill, zorder=2)
    ax2.plot(xg, dens_o, color=green, lw=1.8, zorder=3,
             label=f"DAMOT Ours (median {np.median(seg_o):.0f} f)"
             if seg_o.size else "DAMOT (Ours)")
    if seg_b.size:
        ax2.axvline(np.median(seg_b), color=red, ls="--", lw=1.2, zorder=4)
    if seg_o.size:
        ax2.axvline(np.median(seg_o), color=green, ls="--", lw=1.2, zorder=4)
    ax2.set_xscale("log")
    ax2.set_xlim(1, hi_len + 1)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("Segment-length\ndensity", fontsize=10)
    ax2.set_xlabel("Identity segment length")
    ax2.legend(fontsize=9, frameon=False, loc="upper right")

    ax1.set_xlim(lo, hi)
    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(width=0.8, labelsize=9)

    n_all = len(display_ids)
    disp_note = (f"  (all {n_all} GT tracks of the sequence"
                 + (f"; timeline shows {n_display})" if n_display else ")"))
    fig.suptitle(
        f"UA-DETRAC rainy  \u2014  sequence {sequence_name}  identity "
        "stability curves" + disp_note, fontsize=11, y=0.98)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    base = f"ua_detrac_rainy_{sequence_name}_identity_curves"
    for ext in ("pdf", "png", "svg"):
        p = out_dir / f"{base}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved.append(str(p.resolve()))
    plt.close(fig)
    return saved


def export_summary_files(out_dir, display_ids, base_tracks, ours_tracks,
                         frame_window, base_per, ours_per, highlights,
                         selected_info, ranking_df):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # gt_display_id_mapping.csv
    map_rows = []
    for row_idx, iid in enumerate(display_ids):
        map_rows.append({"display_id": row_idx, "gt_instance_id": iid})
    pd.DataFrame(map_rows).to_csv(
        out_dir / "gt_display_id_mapping.csv", index=False)

    # selected_tracks.csv
    lo, hi = frame_window
    trk_rows = []
    for row_idx, iid in enumerate(display_ids):
        bp = base_per.get(iid, {})
        op = ours_per.get(iid, {})
        trk_rows.append({
            "display_id": row_idx, "gt_instance_id": iid,
            "baseline_matched_frames": bp.get("matched_frames", 0),
            "baseline_reassignments": bp.get("reassignments", 0),
            "baseline_fragmentation": bp.get("fragmentation", 0),
            "ours_matched_frames": op.get("matched_frames", 0),
            "ours_reassignments": op.get("reassignments", 0),
            "ours_fragmentation": op.get("fragmentation", 0),
        })
    pd.DataFrame(trk_rows).to_csv(
        out_dir / "selected_tracks.csv", index=False)

    # highlighted_events.csv
    hl_rows = []
    for ev in highlights:
        hl_rows.append({
            "row_index": ev["row_index"],
            "gt_instance_id": ev["gt_id"],
            "display_id": ev["row_index"],
            "frame_start": ev["frame_start"],
            "frame_end": ev["frame_end"],
            "event_type": ev["event_type"],
            "box_color": ev["color"],
            "ours_stable": ev["ours_stable"],
        })
    pd.DataFrame(hl_rows).to_csv(
        out_dir / "highlighted_events.csv", index=False)

    # timeline_event_summary.csv
    _, base_agg = compute_timeline_events(
        {i: base_tracks[i] for i in display_ids if i in base_tracks},
        frame_window)
    _, ours_agg = compute_timeline_events(
        {i: ours_tracks[i] for i in display_ids if i in ours_tracks},
        frame_window)
    summary_rows = [
        {"method": "baseline_single_target",
         "matched_gt_frames": base_agg["matched_gt_frames"],
         "unmatched_gt_frames": base_agg["unmatched_gt_frames"],
         "identity_reassignments": base_agg["identity_reassignments"],
         "fragmented_gt_tracks": base_agg["fragmented_tracks"],
         "avg_identity_segment_length": round(base_agg["avg_segment_length"], 3),
         "matched_ratio": round(base_agg["matched_ratio"], 4)},
        {"method": "damot_ours",
         "matched_gt_frames": ours_agg["matched_gt_frames"],
         "unmatched_gt_frames": ours_agg["unmatched_gt_frames"],
         "identity_reassignments": ours_agg["identity_reassignments"],
         "fragmented_gt_tracks": ours_agg["fragmented_tracks"],
         "avg_identity_segment_length": round(ours_agg["avg_segment_length"], 3),
         "matched_ratio": round(ours_agg["matched_ratio"], 4)},
    ]
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "timeline_event_summary.csv", index=False)

    # sequence_ranking.csv
    ranking_df.to_csv(out_dir / "sequence_ranking.csv", index=False)

    # selected_sequence.json
    with (out_dir / "selected_sequence.json").open("w") as fh:
        json.dump(selected_info, fh, indent=2)

    return base_agg, ours_agg


def write_readme(out_dir, selected_info, base_agg, ours_agg, saved_files):
    out_dir = Path(out_dir)
    lines = []
    a = lines.append
    a("# UA-DETRAC rainy: Identity Stability Timeline (Baseline vs DAMOT)\n")
    a("## Data sources\n")
    a(f"- Sequence: **{selected_info['selected_sequence']}** "
      f"(video_id={selected_info['video_id']})")
    a(f"- Frame range: {selected_info['start_frame']} - "
      f"{selected_info['end_frame']}")
    a(f"- IoU threshold: {selected_info['iou_threshold']}")
    a(f"- Score threshold: {selected_info['score_threshold']}")
    a(f"- Baseline result: `{selected_info['baseline_result_path']}`")
    a(f"- Ours result: `{selected_info['ours_result_path']}`")
    a(f"- Baseline checkpoint: `{selected_info['baseline_checkpoint']}`")
    a(f"- Ours checkpoint: `{selected_info['ours_checkpoint']}`")
    a(f"- Selected GT instance ids: {selected_info['selected_gt_ids']}\n")
    a("## Matching rule\n")
    a("Per video, per frame: predictions filtered to eval categories "
      "(car/truck/bus) and score>=threshold, matched to GT with Hungarian "
      "assignment on cost=1-IoU, accepted only when same category and "
      "IoU>threshold. A per-GT-track frame->predicted-track-id timeline is "
      "built.\n")
    a("## Figure meaning\n")
    a("- Each row = one ground-truth track (relabeled 0..N-1).")
    a("- Grey thick line = GT lifespan; gap = unmatched interval.")
    a("- Colored segment = matched predicted identity; every predicted id has "
      "a UNIQUE vivid color; color change = identity reassignment event "
      "(from per-frame IoU matching, not TrackEval's internal IDSw logic).")
    a("- Only GT tracks that ours matches on at least "
      f"{selected_info.get('min_ours_overlap', 0.3):.0%} of their frames are "
      "shown.\n")
    a("## Selected-window statistics\n")
    a("| method | matched_ratio | reassignments | fragmented_tracks | "
      "avg_segment_len |")
    a("|---|---|---|---|---|")
    a(f"| Baseline | {base_agg['matched_ratio']:.3f} | "
      f"{base_agg['identity_reassignments']} | "
      f"{base_agg['fragmented_tracks']} | "
      f"{base_agg['avg_segment_length']:.2f} |")
    a(f"| DAMOT (Ours) | {ours_agg['matched_ratio']:.3f} | "
      f"{ours_agg['identity_reassignments']} | "
      f"{ours_agg['fragmented_tracks']} | "
      f"{ours_agg['avg_segment_length']:.2f} |\n")
    a("## Output files\n")
    for f in saved_files:
        a(f"- `{f}`")
    (out_dir / "README.md").write_text("\n".join(lines))


def setup_fonts():
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        if "Times New Roman" in available:
            plt.rcParams["font.serif"] = ["Times New Roman"]
        else:
            plt.rcParams["font.serif"] = ["DejaVu Serif"]
        plt.rcParams["font.family"] = "serif"
    except Exception:
        plt.rcParams["font.family"] = "serif"


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="UA-DETRAC rainy identity-stability timeline comparison")
    p.add_argument("--ann-file", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--baseline-track-json", required=True)
    p.add_argument("--baseline-raw-pkl", default=None)
    p.add_argument("--baseline-hota-csv", default=None)
    p.add_argument("--baseline-track-csv", default=None)
    p.add_argument("--ours-track-json", required=True)
    p.add_argument("--ours-raw-pkl", default=None)
    p.add_argument("--ours-hota-csv", default=None)
    p.add_argument("--ours-track-csv", default=None)
    p.add_argument("--sequence", default=None)
    p.add_argument("--start-frame", type=int, default=None)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--num-tracks", type=int, default=10)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--score-thr", type=float, default=0.2)
    p.add_argument("--max-highlights", type=int, default=4,
                   help="kept for CLI compatibility; boxes are disabled")
    p.add_argument("--top-sequences", type=int, default=5,
                   help="how many top-ranked sequences to render")
    p.add_argument("--min-ours-overlap", type=float, default=0.30,
                   help="drop GT tracks whose fraction of frames matched by "
                        "ours is below this threshold")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--inspect-only", action="store_true")
    p.add_argument("--baseline-checkpoint", default=(
        "work_dirs/ua_detrac/baseline_single_target/adapt/sunny_to_rainy_5e/"
        "darth_qdtrack_faster-rcnn_r50_fpn_5e_ua_detrac_rainy_from_sunny/"
        "20260708_172601/epoch_5.pth"))
    p.add_argument("--ours-checkpoint", default=(
        "work_dirs/ua_detrac/trimodule_spaP1_causalProtoShrink_roiV1Weak/"
        "episode/adapt/stage02_after_rainy/epoch_5.pth"))
    return p


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    args = build_arg_parser().parse_args()
    setup_fonts()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ann = load_coco_annotations(args.ann_file)
    base_preds = load_tracking_results(args.baseline_track_json,
                                       args.baseline_raw_pkl)
    ours_preds = load_tracking_results(args.ours_track_json, args.ours_raw_pkl)

    base_hota = load_metric_csv(args.baseline_hota_csv)
    base_trk = load_metric_csv(args.baseline_track_csv)
    ours_hota = load_metric_csv(args.ours_hota_csv)
    ours_trk = load_metric_csv(args.ours_track_csv)

    csv_map = {
        "baseline_hota": base_hota, "baseline_track": base_trk,
        "ours_hota": ours_hota, "ours_track": ours_trk,
    }
    inspect_file_structure(ann, base_preds, ours_preds, csv_map,
                           out_dir / "file_structure_inspection.txt")

    if args.inspect_only:
        LOGGER.info("inspect-only mode: structure written, exiting.")
        return

    ann_ctx = build_video_frame_index(ann)
    videos, img_meta, video_images, gt_by_image = ann_ctx

    base_pred_by_image = index_predictions(base_preds, args.score_thr)
    ours_pred_by_image = index_predictions(ours_preds, args.score_thr)

    LOGGER.info("ranking candidate sequences (iou_thr=%.2f score_thr=%.2f)...",
                args.iou_thr, args.score_thr)
    ranking_df, cache = rank_candidate_sequences(
        ann_ctx, base_pred_by_image, ours_pred_by_image, args.iou_thr)
    LOGGER.info("top ranked sequences:\n%s",
                ranking_df.head(10).to_string(index=False))

    # build the list of sequences to render
    if args.sequence:
        row = ranking_df[ranking_df["sequence"] == args.sequence]
        if row.empty:
            raise ValueError(f"sequence {args.sequence} not found")
        target_rows = [row.iloc[0]]
        selection_reason = "manually specified via --sequence"
    else:
        eligible = ranking_df[ranking_df["eligible"]]
        if eligible.empty:
            eligible = ranking_df
        target_rows = [eligible.iloc[i]
                       for i in range(min(args.top_sequences, len(eligible)))]
        selection_reason = (
            "auto-selected top sequences by ranking_score among those meeting "
            "min valid GT tracks and min frames; only GT tracks with "
            f"ours-overlap >= {args.min_ours_overlap} are shown")

    ranking_df.to_csv(out_dir / "sequence_ranking.csv", index=False)

    base_caption = _metric_caption(base_hota, base_trk)
    ours_caption = _metric_caption(ours_hota, ours_trk)

    all_saved = []
    per_seq_summ = []
    for rank_i, top in enumerate(target_rows):
        video_id = int(top["video_id"])
        seq_name = top["sequence"]
        base_tracks, ours_tracks = cache[video_id]
        frames_all = [f for f, _ in video_images[video_id]]
        fmin, fmax = min(frames_all), max(frames_all)

        full_window = (fmin, fmax)
        display_ids = select_gt_tracks(
            base_tracks, ours_tracks, full_window, args.num_tracks,
            min_ours_overlap=args.min_ours_overlap)
        if not display_ids:
            LOGGER.warning("sequence %s has no GT track with ours-overlap>=%.2f;"
                           " skipping", seq_name, args.min_ours_overlap)
            continue

        if args.start_frame is not None:
            lo = args.start_frame
        else:
            active_lo = [min(tr[i]["frames"]) for i in display_ids
                         for tr in (base_tracks, ours_tracks)
                         if i in tr and tr[i]["frames"]]
            lo = min(active_lo) if active_lo else fmin
        if args.end_frame is not None:
            hi = args.end_frame
        else:
            active_hi = [max(tr[i]["frames"]) for i in display_ids
                         for tr in (base_tracks, ours_tracks)
                         if i in tr and tr[i]["frames"]]
            hi = max(active_hi) if active_hi else fmax
        frame_window = (lo, hi)
        LOGGER.info("[%d] sequence=%s video_id=%d window=%s tracks=%d %s",
                    rank_i, seq_name, video_id, frame_window,
                    len(display_ids), display_ids)

        base_per, _ = compute_timeline_events(
            {i: base_tracks[i] for i in display_ids if i in base_tracks},
            frame_window)
        ours_per, _ = compute_timeline_events(
            {i: ours_tracks[i] for i in display_ids if i in ours_tracks},
            frame_window)

        color_map = build_color_map(display_ids, base_tracks, ours_tracks,
                                    frame_window)

        saved = plot_identity_timeline_comparison(
            display_ids, base_tracks, ours_tracks, frame_window, color_map,
            base_caption, ours_caption, out_dir, args.dpi, seq_name)
        # auxiliary curves aggregate over ALL GT tracks of the sequence (not
        # just the displayed rows), so they are not cherry-picked.
        all_gt_ids = sorted(set(base_tracks) | set(ours_tracks))
        saved += plot_auxiliary_curves(
            all_gt_ids, base_tracks, ours_tracks, frame_window,
            out_dir, args.dpi, seq_name, n_display=len(display_ids))
        all_saved += saved

        # per-sequence export (suffix filenames with sequence name)
        seq_dir = out_dir / seq_name
        selected_info = {
            "selected_sequence": seq_name,
            "video_id": video_id,
            "rank": rank_i,
            "start_frame": lo,
            "end_frame": hi,
            "selected_gt_ids": display_ids,
            "iou_threshold": args.iou_thr,
            "score_threshold": args.score_thr,
            "min_ours_overlap": args.min_ours_overlap,
            "baseline_result_path": str(Path(args.baseline_track_json).resolve()),
            "ours_result_path": str(Path(args.ours_track_json).resolve()),
            "baseline_checkpoint": args.baseline_checkpoint,
            "ours_checkpoint": args.ours_checkpoint,
            "checkpoint_provenance": (
                "Baseline: sunny_to_rainy_5e epoch_5; Ours: trimodule "
                "stage02_after_rainy epoch_5 (init07 threshold sweep dir; "
                "init07 is an experiment id, NOT a sequence name)."),
            "selection_reason": selection_reason,
        }
        base_agg, ours_agg = export_summary_files(
            seq_dir, display_ids, base_tracks, ours_tracks, frame_window,
            base_per, ours_per, [], selected_info, ranking_df)
        write_readme(seq_dir, selected_info, base_agg, ours_agg, saved)
        per_seq_summ.append((seq_name, frame_window, len(display_ids),
                             base_agg, ours_agg))

    # combined terminal summary
    LOGGER.info("=" * 60)
    for seq_name, frame_window, ntr, base_agg, ours_agg in per_seq_summ:
        LOGGER.info("SEQ %s window=%s tracks=%d", seq_name, frame_window, ntr)
        LOGGER.info("  Baseline: matched_ratio=%.3f reassign=%d frag=%d",
                    base_agg["matched_ratio"],
                    base_agg["identity_reassignments"],
                    base_agg["fragmented_tracks"])
        LOGGER.info("  Ours:     matched_ratio=%.3f reassign=%d frag=%d",
                    ours_agg["matched_ratio"],
                    ours_agg["identity_reassignments"],
                    ours_agg["fragmented_tracks"])
    LOGGER.info("saved %d files under %s", len(all_saved), out_dir)
    for f in all_saved:
        LOGGER.info("  %s", f)


if __name__ == "__main__":
    main()
