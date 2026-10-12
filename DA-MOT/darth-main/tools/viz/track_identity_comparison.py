#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Track identity continuity comparison figure (Baseline vs Ours/DAMOT).

Renders a two-panel figure sharing X (frame index) and Y (ground-truth track
id) axes. Each GT track is a thick colored horizontal band; matched prediction
identities are overlaid as thin dark segments; unmatched intervals are shown as
gaps; ID switches / fragmentations are boxed in red; long stable associations
are boxed in blue. A thin bottom strip shows per-frame matched-GT count.

------------------------------------------------------------------------------
INPUT FORMAT ASSUMPTIONS (do NOT silently change your data):

This project stores results in COCO JSON, not MOTChallenge txt, so the loader
auto-detects the format from the file content:

1) COCO GROUND-TRUTH dict with keys: videos, images, annotations, categories.
   - images[i]  = {id, file_name, frame_id, video_id, ...}
   - annots[i]  = {image_id, bbox:[x,y,w,h], category_id, instance_id, ...}
   - GT track id  = 'instance_id' (fallback: 'track_id').
   - frame index  = images[image_id].frame_id ; video   = images[image_id].video_id

2) COCO PREDICTION list of dicts (mmtracking track results):
   - {image_id, bbox:[x,y,w,h], score, category_id, track_id, video_id}
   - pred track id = 'track_id'. frame/video resolved via the GT images table.

3) MOTChallenge txt fallback (comma OR space separated):
   frame_id, track_id, x, y, w, h, conf, class_id, visibility
   Missing conf/class/visibility columns are tolerated.

All bboxes are treated as [x, y, w, h]. Because the COCO files hold many
videos, a single shared VIDEO is selected (see VIDEO_NAME / auto-pick) so both
panels use exactly the same sequence, GT tracks, colors and frame range.
------------------------------------------------------------------------------
"""

import os
import sys
import json
import warnings
from collections import defaultdict, OrderedDict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - fallback greedy matcher
    _HAS_SCIPY = False

# ============================================================================
# USER PARAMETERS (edit these)
# ============================================================================
GT_FILE = ("/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/data/"
           "ua_detrac/annotations/box_track_rainy_all_cocofmt.json")
BASELINE_RESULT = ("/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/"
                   "work_dirs/ua_detrac/test/baseline_sunny_to_rainy_5e/"
                   "qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_rainy_eval/"
                   "20260708_175830/track_coco.json")
OURS_RESULT = ("/data1/LWR/vranlee/SERVER_ONLY/avis/DA-MOT/darth-main/"
               "work_dirs/limit_search_gpu23/phase1_threshold/ua_detrac/rainy/"
               "init07/qdtrack_faster-rcnn_r50_fpn_20e_ua_detrac_rainy_eval/"
               "20260713_211223/track_coco.json")

OUTPUT_PDF = "track_identity_comparison.pdf"
OUTPUT_PNG = "track_identity_comparison.png"

# Frame window (inclusive-exclusive on the x axis).
FRAME_START = 0
FRAME_END = 700
MAX_TRACKS = 10
IOU_THRESHOLD = 0.5

# Which shared video sequence to visualize. COCO files contain many videos and
# both panels MUST use the same one. Set an explicit name (e.g. "MVI_40863")
# or leave None to auto-pick the video with the most GT activity in-window.
VIDEO_NAME = None

# Only keep this GT category id when matching (UA-DETRAC: 2 == car).
# Set to None to keep all categories.
GT_CATEGORY_ID = None

# Continuity thresholds.
STABLE_MIN_LEN = 80          # min contiguous same-pred length for a blue box
FRAG_GAP_MIN = 5             # min unmatched run length counted as fragmentation
MAX_RED_BOXES = 4            # per panel
MAX_BLUE_BOXES = 3           # per panel

SHOW_SUMMARY_STRIP = True

# Max real-frame gap treated as "still contiguous". GT/preds may be temporally
# subsampled (e.g. every ~5 frames), so adjacency is defined in real-frame
# units rather than +1. Auto-estimated in main() from the GT sampling step;
# this is only the fallback default.
_MAX_GAP = 1

# ============================================================================
# COLOR / STYLE
# ============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Soft qualitative palette (colorblind-friendly-ish, muted) for GT lifespans.
GT_PALETTE = [
    "#8c6d5c", "#c0504d", "#4f9d69", "#4472c4", "#7f7f7f",
    "#c879b8", "#e07b39", "#7a5aa8", "#4aa8b0", "#b0b0b0",
    "#5b8a72", "#a05d5d",
]
PRED_COLOR = "#1a1a1a"          # matched prediction segment
STABLE_EDGE = "#2b7bba"         # blue box
ERROR_EDGE = "#c23b3b"          # muted red box
UNMATCHED_COLOR = "#ffffff"     # gap fill

# ============================================================================
# DATA LOADING
# ============================================================================
# Internal detection record: dict with keys frame, track, video, bbox(xywh),
#   score, category. We normalize every input source to a list of these.

def _check_file(path, label):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("%s not found: %s" % (label, path))


def _bbox_xywh(b):
    """Return a clean [x, y, w, h] float list."""
    x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return [x, y, w, h]


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _build_image_index(gt_dict):
    """image_id -> (frame_id, video_id, video_name) from a COCO GT dict."""
    vid_name = {}
    for v in gt_dict.get("videos", []):
        vid_name[v["id"]] = v.get("name", str(v["id"]))
    idx = {}
    for im in gt_dict["images"]:
        # frame_id may be 0- or 1-based; kept as-is here, normalized later.
        fid = im.get("frame_id")
        if fid is None:
            fid = im["id"]
        vid = im.get("video_id", 0)
        idx[im["id"]] = (int(fid), int(vid), vid_name.get(vid, str(vid)))
    return idx


def load_coco_gt(path):
    """Load COCO GT. Returns (detections, image_index, meta)."""
    d = _load_json(path)
    img_idx = _build_image_index(d)
    dets = []
    dup = set()
    seen = set()
    for a in d["annotations"]:
        iid = a["image_id"]
        if iid not in img_idx:
            continue
        frame, vid, vname = img_idx[iid]
        tid = a.get("instance_id", a.get("track_id"))
        if tid is None:
            continue
        cat = a.get("category_id")
        if GT_CATEGORY_ID is not None and cat != GT_CATEGORY_ID:
            continue
        key = (vid, frame, int(tid))
        if key in seen:
            dup.add(key)
        seen.add(key)
        dets.append(dict(frame=frame, track=int(tid), video=vid,
                         video_name=vname, bbox=_bbox_xywh(a["bbox"]),
                         score=1.0, category=cat))
    if dup:
        warnings.warn("GT has %d duplicated (video,frame,track) records" % len(dup))
    return dets, img_idx


def load_coco_pred(path, img_idx):
    """Load a COCO prediction list, resolving frame/video via the GT images."""
    d = _load_json(path)
    if isinstance(d, dict):  # some dumps wrap predictions
        d = d.get("annotations", d.get("results", []))
    dets = []
    dup = set()
    seen = set()
    for a in d:
        iid = a.get("image_id")
        if iid is None or iid not in img_idx:
            continue
        frame, vid, vname = img_idx[iid]
        tid = a.get("track_id", a.get("instance_id"))
        if tid is None:
            continue
        cat = a.get("category_id")
        if GT_CATEGORY_ID is not None and cat != GT_CATEGORY_ID:
            continue
        key = (vid, frame, int(tid))
        if key in seen:
            dup.add(key)
        seen.add(key)
        dets.append(dict(frame=frame, track=int(tid), video=vid,
                         video_name=vname, bbox=_bbox_xywh(a["bbox"]),
                         score=float(a.get("score", 1.0)), category=cat))
    if dup:
        warnings.warn("Pred %s has %d duplicated records" %
                      (os.path.basename(path), len(dup)))
    return dets


def load_mot_txt(path):
    """MOTChallenge fallback loader (comma or whitespace separated)."""
    dets = []
    seen = set()
    dup = set()
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split(",") if "," in ln else ln.split()
            if len(parts) < 6:
                continue  # need at least frame,id,x,y,w,h
            vals = [p for p in parts if p != ""]
            frame = int(float(vals[0]))
            tid = int(float(vals[1]))
            bbox = _bbox_xywh(vals[2:6])
            score = float(vals[6]) if len(vals) > 6 else 1.0
            cat = int(float(vals[7])) if len(vals) > 7 else -1
            key = (0, frame, tid)
            if key in seen:
                dup.add(key)
            seen.add(key)
            dets.append(dict(frame=frame, track=tid, video=0,
                             video_name="seq", bbox=bbox, score=score,
                             category=cat))
    if dup:
        warnings.warn("MOT file %s has %d duplicated records" %
                      (os.path.basename(path), len(dup)))
    return dets


def load_mot_file(path, img_idx=None, is_gt=False):
    """Dispatch loader by content. Returns (detections, img_idx_or_None)."""
    _check_file(path, "input")
    if path.endswith(".json"):
        d = _load_json(path)
        if is_gt or (isinstance(d, dict) and "annotations" in d and "images" in d):
            return load_coco_gt(path)
        return load_coco_pred(path, img_idx), img_idx
    return load_mot_txt(path), None

# ============================================================================
# MATCHING
# ============================================================================

def compute_iou_matrix(gt_boxes, pred_boxes):
    """IoU matrix between two arrays of [x,y,w,h] boxes. Shape (G, P)."""
    G, P = len(gt_boxes), len(pred_boxes)
    if G == 0 or P == 0:
        return np.zeros((G, P), dtype=float)
    g = np.asarray(gt_boxes, dtype=float)
    p = np.asarray(pred_boxes, dtype=float)
    gx1, gy1 = g[:, 0], g[:, 1]
    gx2, gy2 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    px1, py1 = p[:, 0], p[:, 1]
    px2, py2 = p[:, 0] + p[:, 2], p[:, 1] + p[:, 3]
    garea = np.maximum(g[:, 2], 0) * np.maximum(g[:, 3], 0)
    parea = np.maximum(p[:, 2], 0) * np.maximum(p[:, 3], 0)

    ix1 = np.maximum(gx1[:, None], px1[None, :])
    iy1 = np.maximum(gy1[:, None], py1[None, :])
    ix2 = np.minimum(gx2[:, None], px2[None, :])
    iy2 = np.minimum(gy2[:, None], py2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = garea[:, None] + parea[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def match_frame(gt_frame, pred_frame, iou_threshold):
    """Hungarian match one frame.

    gt_frame / pred_frame: lists of detection dicts for a single frame.
    Returns dict gt_track_id -> pred_track_id for pairs with IoU >= threshold.
    """
    if not gt_frame or not pred_frame:
        return {}
    iou = compute_iou_matrix([g["bbox"] for g in gt_frame],
                             [p["bbox"] for p in pred_frame])
    if iou.size == 0:
        return {}
    if _HAS_SCIPY:
        rows, cols = linear_sum_assignment(-iou)
        pairs = zip(rows, cols)
    else:  # greedy fallback
        pairs = []
        flat = [(iou[i, j], i, j) for i in range(iou.shape[0])
                for j in range(iou.shape[1])]
        flat.sort(reverse=True)
        ur, uc = set(), set()
        for _, i, j in flat:
            if i in ur or j in uc:
                continue
            ur.add(i); uc.add(j)
            pairs.append((i, j))
    out = {}
    for i, j in pairs:
        if iou[i, j] >= iou_threshold:
            out[gt_frame[i]["track"]] = pred_frame[j]["track"]
    return out


def _index_by_frame(dets, video):
    """Group detections of one video by frame index. Returns dict frame->list."""
    by_frame = defaultdict(list)
    for d in dets:
        if d["video"] != video:
            continue
        by_frame[d["frame"]].append(d)
    return by_frame


def build_gt_identity_timelines(gt, prediction, video, frame_lo, frame_hi):
    """For each GT track, per-frame matched pred id within [frame_lo, frame_hi).

    Returns:
      timelines: dict gt_id -> dict frame -> pred_id (matched frames only)
      gt_present: dict gt_id -> set of frames where GT exists
      matched_per_frame: dict frame -> count of matched GT
    """
    gt_by_frame = _index_by_frame(gt, video)
    pred_by_frame = _index_by_frame(prediction, video)

    timelines = defaultdict(dict)
    gt_present = defaultdict(set)
    matched_per_frame = defaultdict(int)

    frames = range(frame_lo, frame_hi)
    for fr in frames:
        gframe = gt_by_frame.get(fr, [])
        for g in gframe:
            gt_present[g["track"]].add(fr)
        pframe = pred_by_frame.get(fr, [])
        matches = match_frame(gframe, pframe, IOU_THRESHOLD)
        for gid, pid in matches.items():
            timelines[gid][fr] = pid
            matched_per_frame[fr] += 1
    return timelines, gt_present, matched_per_frame

# ============================================================================
# SEGMENTS & STATISTICS
# ============================================================================

def split_contiguous_segments(frames, values, max_gap=None):
    """Split parallel (frames, values) into runs of consecutive frames sharing
    the same value.

    frames: sorted list of int frame ids.
    values: list of the same length (e.g. pred ids).
    max_gap: two observations are still "contiguous" if their real-frame
             difference is <= max_gap. GT/preds are often temporally subsampled
             (e.g. every ~5 frames), so adjacency is defined in real-frame units
             rather than +1. Defaults to the auto-estimated module value.
    Returns list of (start_frame, end_frame, value) with end inclusive.
    """
    if max_gap is None:
        max_gap = _MAX_GAP
    segs = []
    if not frames:
        return segs
    s = frames[0]
    prev = frames[0]
    cur_val = values[0]
    for fr, val in zip(frames[1:], values[1:]):
        if (fr - prev) <= max_gap and val == cur_val:
            prev = fr
            continue
        segs.append((s, prev, cur_val))
        s = fr
        prev = fr
        cur_val = val
    segs.append((s, prev, cur_val))
    return segs


def _matched_segments(timeline):
    """Contiguous same-pred segments from a gt timeline (frame->pred)."""
    if not timeline:
        return []
    frames = sorted(timeline.keys())
    values = [timeline[f] for f in frames]
    return split_contiguous_segments(frames, values)


def count_id_switches(timeline):
    """Number of pred-id changes between temporally adjacent matched frames.

    Only counts a switch when the two successive matched observations are
    contiguous in real-frame units (gap <= _MAX_GAP); a pred-id change across a
    large unmatched gap is accounted as a fragmentation instead of a switch."""
    if not timeline:
        return 0
    frames = sorted(timeline.keys())
    switches = 0
    for a, b in zip(frames[:-1], frames[1:]):
        if (b - a) <= _MAX_GAP and timeline[b] != timeline[a]:
            switches += 1
    return switches


def count_fragmentations(timeline, gt_present):
    """Number of gaps: GT present but unmatched for >= FRAG_GAP_MIN real frames
    between two matched observations (a track that is lost then re-acquired)."""
    if not timeline:
        return 0
    matched = sorted(timeline.keys())
    if len(matched) < 2:
        return 0
    frag = 0
    present = gt_present
    for a, b in zip(matched[:-1], matched[1:]):
        if (b - a) <= _MAX_GAP:
            continue
        gap_present = [f for f in present if a < f < b]
        if len(gap_present) >= FRAG_GAP_MIN:
            frag += 1
    return frag


def timeline_stats(timelines, gt_present, selected):
    """Aggregate statistics over the selected GT tracks."""
    matched_frames = 0
    unmatched_frames = 0
    switches = 0
    frags = 0
    seg_lengths = []
    for gid in selected:
        tl = timelines.get(gid, {})
        present = gt_present.get(gid, set())
        matched_frames += len(tl)
        unmatched_frames += len(present - set(tl.keys()))
        switches += count_id_switches(tl)
        frags += count_fragmentations(tl, present)
        for s, e, _ in _matched_segments(tl):
            seg_lengths.append(e - s + 1)
    avg_seg = float(np.mean(seg_lengths)) if seg_lengths else 0.0
    return dict(n_tracks=len(selected), matched=matched_frames,
                unmatched=unmatched_frames, switches=switches,
                frags=frags, avg_seg=avg_seg)

# ============================================================================
# TRACK SELECTION
# ============================================================================

def _track_lifespan(gt_present, gid):
    fs = gt_present.get(gid, set())
    if not fs:
        return 0, 0, 0
    return min(fs), max(fs), len(fs)


def select_representative_tracks(gt_present, tl_base, tl_ours, max_tracks):
    """Choose shared GT tracks shown in BOTH panels, same order.

    Priority (per spec):
      1. long in-window duration (span);
      2. at least one method has an ID switch or fragmentation;
      3. many visible GT frames;
      4. cap at max_tracks.
    """
    candidates = []
    for gid, frames in gt_present.items():
        if not frames:
            continue
        lo, hi, vis = _track_lifespan(gt_present, gid)
        span = hi - lo + 1
        sw = count_id_switches(tl_base.get(gid, {})) + \
            count_id_switches(tl_ours.get(gid, {}))
        fg = count_fragmentations(tl_base.get(gid, {}), frames) + \
            count_fragmentations(tl_ours.get(gid, {}), frames)
        has_err = (sw + fg) > 0
        candidates.append(dict(gid=gid, span=span, vis=vis,
                               has_err=has_err, err=sw + fg))
    # Rank: errors first (to show interesting behavior), then span, then vis.
    candidates.sort(key=lambda c: (c["has_err"], c["span"], c["vis"]),
                    reverse=True)
    selected = [c["gid"] for c in candidates[:max_tracks]]
    # Stable, readable order along y: by first appearance then id.
    selected.sort(key=lambda g: (min(gt_present[g]), g))
    return selected

# ============================================================================
# HIGHLIGHT INTERVALS
# ============================================================================

def find_stable_intervals(timeline, min_len):
    """Contiguous same-pred runs of length >= min_len (candidate blue boxes).
    Returns list of (start, end, pred_id, length)."""
    out = []
    for s, e, val in _matched_segments(timeline):
        L = e - s + 1
        if L >= min_len:
            out.append((s, e, val, L))
    out.sort(key=lambda t: t[3], reverse=True)
    return out


def find_error_intervals(timeline, gt_present):
    """Local windows around ID switches / fragmentations (candidate red boxes).
    Returns list of (start, end, kind, severity)."""
    out = []
    if not timeline:
        return out
    frames = sorted(timeline.keys())
    # ID switches: window around the switch frame (only across adjacent obs).
    for a, b in zip(frames[:-1], frames[1:]):
        if (b - a) <= _MAX_GAP and timeline[b] != timeline[a]:
            pad = max(6, 3 * _MAX_GAP)
            out.append((a - pad // 2, b + pad // 2, "switch", 2))
    # Fragmentations: the unmatched gap where GT is still present.
    for a, b in zip(frames[:-1], frames[1:]):
        if (b - a) <= _MAX_GAP:
            continue
        gap_present = [f for f in gt_present if a < f < b]
        if len(gap_present) >= FRAG_GAP_MIN:
            out.append((a, b, "frag", len(gap_present)))
    out.sort(key=lambda t: t[3], reverse=True)
    return out


def _dedupe_intervals(intervals, max_n, min_gap=15):
    """Keep up to max_n intervals that don't overlap too closely."""
    kept = []
    for iv in intervals:
        s, e = iv[0], iv[1]
        c = 0.5 * (s + e)
        if all(abs(c - 0.5 * (k[0] + k[1])) > min_gap for k in kept):
            kept.append(iv)
        if len(kept) >= max_n:
            break
    return kept

# ============================================================================
# DRAWING
# ============================================================================

def _present_segments(present_set, frame_lo, frame_hi):
    """Contiguous runs of frames where GT exists (for lifespan bands)."""
    frames = sorted(f for f in present_set if frame_lo <= f < frame_hi)
    if not frames:
        return []
    return [(s, e) for s, e, _ in
            split_contiguous_segments(frames, [0] * len(frames))]


def draw_highlight_boxes(ax, y, intervals, edge_color, frame_lo, frame_hi,
                         half_height=0.34):
    """Draw rounded highlight boxes centered on row y in DATA coordinates."""
    for iv in intervals:
        s = max(iv[0], frame_lo)
        e = min(iv[1], frame_hi - 1)
        if e <= s:
            e = s + 1
        box = FancyBboxPatch(
            (s, y - half_height), (e - s), 2 * half_height,
            boxstyle="round,pad=0.0,rounding_size=2.5",
            linewidth=1.8, edgecolor=edge_color, facecolor="none",
            mutation_aspect=0.02, zorder=6, clip_on=True)
        ax.add_patch(box)


def draw_single_method_panel(ax, timelines, gt_present, selected, id_map,
                             frame_lo, frame_hi, gt_colors,
                             annotate_pred=True):
    """Render one method panel onto ax."""
    for gid in selected:
        y = id_map[gid]
        color = gt_colors[gid]
        present = gt_present.get(gid, set())
        tl = timelines.get(gid, {})

        # (1) GT lifespan: thick soft band, per contiguous presence segment.
        for s, e in _present_segments(present, frame_lo, frame_hi):
            ax.plot([s, e], [y, y], color=color, linewidth=6.0,
                    alpha=0.65, solid_capstyle="butt", zorder=2)

        # (3) unmatched intervals: light gray dashed line over GT band.
        matched = set(tl.keys())
        unmatched = sorted(f for f in present
                           if f not in matched and frame_lo <= f < frame_hi)
        for s, e in [(a, b) for a, b, _ in
                     split_contiguous_segments(unmatched, [0] * len(unmatched))] \
                if unmatched else []:
            ax.plot([s, e], [y, y], color="#9a9a9a", linewidth=1.4,
                    alpha=0.9, linestyle=(0, (2, 2)), zorder=3)

        # (2) matched prediction identity segments: thin dark line.
        segs = _matched_segments(tl)
        for (s, e, pid) in segs:
            ax.plot([s, e], [y, y], color=PRED_COLOR, linewidth=2.4,
                    alpha=0.95, solid_capstyle="butt", zorder=4)

        # pred-id labels only where an ID switch happens (avoid clutter).
        if annotate_pred and len(segs) > 1:
            for (s, e, pid) in segs:
                if (e - s) < 12:
                    continue
                ax.text(0.5 * (s + e), y - 0.42, "P%d" % pid,
                        ha="center", va="bottom", fontsize=6.0,
                        color=PRED_COLOR, alpha=0.8, zorder=7)

        # (5) blue stable-association boxes.
        stable = _dedupe_intervals(
            find_stable_intervals(tl, STABLE_MIN_LEN), MAX_BLUE_BOXES)
        draw_highlight_boxes(ax, y, stable, STABLE_EDGE, frame_lo, frame_hi)

        # (4) red error boxes (switch / fragmentation).
        errs = _dedupe_intervals(
            find_error_intervals(tl, present), MAX_RED_BOXES)
        draw_highlight_boxes(ax, y, errs, ERROR_EDGE, frame_lo, frame_hi)

    ax.set_xlim(frame_lo, frame_hi)
    ax.set_ylim(-0.6, len(selected) - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(range(len(selected)))
    ax.set_yticklabels([str(id_map[g]) for g in selected])
    # very light horizontal guides only
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def draw_summary_strip(ax, matched_per_frame, frame_lo, frame_hi):
    """Thin bottom strip: per-frame matched-GT count (Plan A)."""
    xs = list(range(frame_lo, frame_hi))
    ys = [matched_per_frame.get(f, 0) for f in xs]
    ax.plot(xs, ys, color="#9a9a9a", linewidth=0.9)
    ax.set_xlim(frame_lo, frame_hi)
    ymax = max(ys) if ys else 1
    ax.set_ylim(0, max(1, ymax))
    ax.set_yticks([0, ymax])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

# ============================================================================
# VIDEO SELECTION & MAIN
# ============================================================================

def _resolve_frame_base(gt):
    """Return the minimum frame id seen in GT (to normalize 0/1-based)."""
    return min((d["frame"] for d in gt), default=0)


def estimate_frame_step(dets):
    """Estimate the temporal sampling step of a sequence.

    GT/predictions may be annotated every k-th frame (e.g. UA-DETRAC keeps
    ~every 5th frame). We take, per GT track, the most common positive gap
    between consecutive present frames and return the global mode. Contiguity
    is then defined as (frame_gap <= step) so a normally-sampled track reads as
    one segment instead of many length-1 dots. Returns >= 1."""
    per_track = defaultdict(list)
    for d in dets:
        per_track[d["track"]].append(d["frame"])
    gaps = defaultdict(int)
    for frames in per_track.values():
        fs = sorted(set(frames))
        for a, b in zip(fs[:-1], fs[1:]):
            g = b - a
            if g > 0:
                gaps[g] += 1
    if not gaps:
        return 1
    # Most frequent gap is the sampling step.
    step = max(gaps, key=gaps.get)
    return max(1, int(step))


def pick_video(gt, img_idx):
    """Pick the shared video id. Honors VIDEO_NAME if set, else the video with
    the most GT annotations inside [FRAME_START, FRAME_END)."""
    name_to_vid = {}
    for _, (fr, vid, vname) in img_idx.items():
        name_to_vid[vname] = vid
    if VIDEO_NAME is not None:
        if VIDEO_NAME not in name_to_vid:
            raise ValueError("VIDEO_NAME %r not found. Available: %s" %
                             (VIDEO_NAME, sorted(name_to_vid.keys())))
        return name_to_vid[VIDEO_NAME], VIDEO_NAME
    counts = defaultdict(int)
    names = {}
    for d in gt:
        if FRAME_START <= d["frame"] < FRAME_END:
            counts[d["video"]] += 1
            names[d["video"]] = d["video_name"]
    if not counts:
        raise ValueError("No GT annotations in frame window; adjust FRAME_*.")
    vid = max(counts, key=counts.get)
    return vid, names[vid]


def _print_stats(name, st):
    print("\n[%s]" % name)
    print("  selected GT tracks     : %d" % st["n_tracks"])
    print("  matched frames         : %d" % st["matched"])
    print("  unmatched frames       : %d" % st["unmatched"])
    print("  ID switches            : %d" % st["switches"])
    print("  fragmentations         : %d" % st["frags"])
    print("  avg identity seg length: %.1f" % st["avg_seg"])


def main():
    # ---- load ----
    print("Loading GT   :", GT_FILE)
    gt, img_idx = load_mot_file(GT_FILE, is_gt=True)
    print("Loading base :", BASELINE_RESULT)
    base, _ = load_mot_file(BASELINE_RESULT, img_idx=img_idx)
    print("Loading ours :", OURS_RESULT)
    ours, _ = load_mot_file(OURS_RESULT, img_idx=img_idx)

    # ---- shared video ----
    video, vname = pick_video(gt, img_idx)
    print("Shared video : %s (id=%s)" % (vname, video))

    # ---- normalize frame base so the axis starts near 0 ----
    base_off = _resolve_frame_base([d for d in gt if d["video"] == video])
    if base_off != 0:
        print("Normalizing frame base by -%d" % base_off)
        for coll in (gt, base, ours):
            for d in coll:
                d["frame"] -= base_off

    frame_lo, frame_hi = FRAME_START, FRAME_END

    # ---- estimate the GT temporal sampling step -> real-frame adjacency ----
    global _MAX_GAP
    _MAX_GAP = estimate_frame_step([d for d in gt if d["video"] == video])
    print("Estimated GT frame step / max_gap: %d" % _MAX_GAP)

    # ---- build timelines ----
    tl_base, gt_present_b, mpf_base = build_gt_identity_timelines(
        gt, base, video, frame_lo, frame_hi)
    tl_ours, gt_present_o, mpf_ours = build_gt_identity_timelines(
        gt, ours, video, frame_lo, frame_hi)
    # GT presence is method-independent; merge for robustness.
    gt_present = defaultdict(set)
    for src in (gt_present_b, gt_present_o):
        for gid, fs in src.items():
            gt_present[gid] |= fs

    # ---- select shared tracks ----
    selected = select_representative_tracks(
        gt_present, tl_base, tl_ours, MAX_TRACKS)
    if not selected:
        raise RuntimeError("No GT tracks selected in the given window/video.")

    id_map = OrderedDict((gid, i) for i, gid in enumerate(selected))
    gt_colors = {gid: GT_PALETTE[i % len(GT_PALETTE)]
                 for i, gid in enumerate(selected)}

    print("\nSelected GT track IDs (y -> real GT id):")
    for gid in selected:
        print("  %2d (GT %d)" % (id_map[gid], gid))

    st_base = timeline_stats(tl_base, gt_present, selected)
    st_ours = timeline_stats(tl_ours, gt_present, selected)
    _print_stats("Baseline", st_base)
    _print_stats("Ours / DAMOT", st_ours)

    # ---- figure ----
    if SHOW_SUMMARY_STRIP:
        fig, axes = plt.subplots(
            2, 2, figsize=(15, 4.5), sharex="col", sharey="row",
            gridspec_kw=dict(height_ratios=[10, 1], hspace=0.08, wspace=0.06))
        (ax_b, ax_o) = axes[0]
        (sb, so) = axes[1]
    else:
        fig, axes = plt.subplots(
            1, 2, figsize=(15, 4.5), sharex=True, sharey=True,
            gridspec_kw=dict(wspace=0.06))
        ax_b, ax_o = axes
        sb = so = None

    draw_single_method_panel(ax_b, tl_base, gt_present, selected, id_map,
                             frame_lo, frame_hi, gt_colors)
    draw_single_method_panel(ax_o, tl_ours, gt_present, selected, id_map,
                             frame_lo, frame_hi, gt_colors)

    ax_b.set_ylabel("ground truth tracks")
    xticks = list(range(frame_lo, frame_hi + 1, 100))

    if SHOW_SUMMARY_STRIP:
        draw_summary_strip(sb, mpf_base, frame_lo, frame_hi)
        draw_summary_strip(so, mpf_ours, frame_lo, frame_hi)
        for a in (ax_b, ax_o):
            a.tick_params(labelbottom=False)
        for a in (sb, so):
            a.set_xticks(xticks)
        sb.set_xlabel("frame index")
        so.set_xlabel("frame index")
        # panel labels below the strips
        sb.text(0.5, -1.15, "(a) Baseline", transform=sb.transAxes,
                ha="center", va="top", fontsize=12)
        so.text(0.5, -1.15, "(b) DAMOT", transform=so.transAxes,
                ha="center", va="top", fontsize=12)
    else:
        for a in (ax_b, ax_o):
            a.set_xticks(xticks)
            a.set_xlabel("frame index")
        ax_b.text(0.5, -0.16, "(a) Baseline", transform=ax_b.transAxes,
                  ha="center", va="top", fontsize=12)
        ax_o.text(0.5, -0.16, "(b) DAMOT", transform=ax_o.transAxes,
                  ha="center", va="top", fontsize=12)

    # ---- single shared legend on top ----
    legend_handles = [
        Line2D([0], [0], color=GT_PALETTE[0], lw=6, alpha=0.65,
               label="Ground-truth lifespan"),
        Line2D([0], [0], color=PRED_COLOR, lw=2.4, label="Matched prediction"),
        Line2D([0], [0], color="#9a9a9a", lw=1.4, linestyle=(0, (2, 2)),
               label="Unmatched interval"),
        Line2D([0], [0], color=STABLE_EDGE, lw=1.8, label="Stable association"),
        Line2D([0], [0], color=ERROR_EDGE, lw=1.8,
               label="ID switch / fragmentation"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=5,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.06))

    fig.savefig(OUTPUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    print("\nSaved:\n  %s\n  %s" % (OUTPUT_PDF, OUTPUT_PNG))


if __name__ == "__main__":
    main()
