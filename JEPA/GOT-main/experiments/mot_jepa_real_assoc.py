#!/usr/bin/env python3
"""Real-image MOT association smoke for a GOT-JEPA style track token.

This experiment intentionally lives outside the production LocateMOT/MASA
trees.  It consumes read-only MOTChallenge data and a fixed detector stream,
extracts a small frozen crop descriptor, and compares the same online
motion/lifecycle/Hungarian tracker with four appearance variants:

* ``baseline``: raw appearance EMA;
* ``gated_ema``: raw EMA with the low-quality update gate;
* ``jepa_inv``: history -> clean-current-crop teacher target predictor;
* ``jepa_cov``: the same predictor capacity with a covariance regularizer;
* ``no_teacher``: shuffled teacher targets (capacity/control sanity check).

The tracker never reads GT IDs.  IDs are attached to detections only in this
driver, after the online step, for post-hoc metrics and for fitting the
teacher predictor on *training videos*.  The teacher crop is a clean GT crop
and is never available to the evaluator.  Metrics are explicitly
``*_proxy`` metrics: they use IoU=0.5 frame matching and a transparent
association calculation, not a claim of official TrackEval output.  The
script writes all provenance and per-sequence rows to JSON/CSV under the
``experiments`` tree.

The default command is deliberately bounded (120 frames, one train and one
validation video per domain).  Increase ``--max-frames``/sequence lists only
after auditing this smoke.
"""
from __future__ import print_function

import argparse
import configparser
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Re-use only the small, dependency-light predictor/tracker implementation.
# This import does not touch LocateMOT/MASA production code.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from mot_jepa_synthetic import (  # noqa: E402
    Detection,
    LinearJEPA,
    MOTTracker,
    SequenceData,
    build_context,
    normalize,
)

try:  # scipy is present in both project environments, but keep a fallback.
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - exercised only on very small installs
    linear_sum_assignment = None


DEFAULT_MOT_ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17"
DEFAULT_DANCE_ROOT = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"
DEFAULT_DANCE_DET_ROOT = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/detections_ctrl"


@dataclass
class RealSequence:
    """A sequence converted to the common normalized representation."""

    name: str
    dataset: str
    split: str
    seq_dir: str
    image_dir: str
    gt_path: str
    det_path: str
    width: int
    height: int
    frame_ids: List[int]
    gt_boxes_px: np.ndarray       # [T, N, 4], NaN for absent GT
    detector_boxes_px: List[np.ndarray]
    detector_scores: List[np.ndarray]
    data: SequenceData
    raw_gt_ids: List[str]
    selected_gt_count: int
    detector_digest: str
    image_digest: str
    load_notes: Dict[str, object]


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def short_digest(path: str) -> str:
    try:
        return sha256_file(path)[:16]
    except Exception:
        return "missing"


def parse_seqinfo(seq_dir: str) -> Dict[str, str]:
    cfg = configparser.ConfigParser()
    p = os.path.join(seq_dir, "seqinfo.ini")
    if not cfg.read(p) or "Sequence" not in cfg:
        raise FileNotFoundError("missing/invalid seqinfo.ini: {}".format(p))
    return {k: v for k, v in cfg["Sequence"].items()}


def numeric_image_names(image_dir: str, max_frames: int) -> List[Tuple[int, str]]:
    rows = []
    for name in os.listdir(image_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        try:
            fid = int(stem)
        except ValueError:
            continue
        if fid <= max_frames:
            rows.append((fid, os.path.join(image_dir, name)))
    rows.sort()
    return rows


def _parse_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def load_gt_rows(path: str, dataset: str, max_frame: int) -> Dict[int, List[Tuple[str, np.ndarray, float]]]:
    """Read MOTChallenge GT, retaining marked person boxes only."""
    by_frame: Dict[int, List[Tuple[str, np.ndarray, float]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            z = [x.strip() for x in line.strip().split(",")]
            if len(z) < 6:
                continue
            try:
                frame = int(float(z[0]))
            except Exception:
                continue
            if frame < 1 or frame > max_frame:
                continue
            raw_id = str(z[1])
            x, y, w, h = (_parse_float(v) for v in z[2:6])
            if w <= 1 or h <= 1:
                continue
            # MOT17: mark, class, visibility are columns 7-9.  DanceTrack
            # uses the same layout and class=1/person; some local exports can
            # omit the trailing columns, in which case retain the row.
            mark = _parse_float(z[6], 1.0) if len(z) > 6 else 1.0
            cls = _parse_float(z[7], 1.0) if len(z) > 7 else 1.0
            vis = _parse_float(z[8], 1.0) if len(z) > 8 else 1.0
            if mark <= 0.0 or (dataset == "mot17" and abs(cls - 1.0) > 1e-6):
                continue
            # DanceTrack is a person-only set.  Keep class 1 and rows with an
            # omitted class field; do not use visibility as an online cue.
            if dataset == "dancetrack" and abs(cls - 1.0) > 1e-6:
                continue
            by_frame[frame].append((raw_id, np.asarray([x, y, x + w, y + h], dtype=np.float64), vis))
    return by_frame


def load_detector_rows(
    dataset: str,
    det_path: str,
    frame_ids: Sequence[int],
    score_threshold: float,
    max_detections: int,
) -> Dict[int, List[Tuple[np.ndarray, float]]]:
    """Read fixed detector rows in either per-frame or one-file MOT format."""
    wanted = set(int(x) for x in frame_ids)
    by_frame: Dict[int, List[Tuple[np.ndarray, float]]] = defaultdict(list)
    files: List[str]
    if os.path.isdir(det_path):
        files = []
        for fid in frame_ids:
            p = os.path.join(det_path, "{:06d}.txt".format(int(fid)))
            if not os.path.exists(p):
                # A few exports use non-padded names.
                p2 = os.path.join(det_path, "{}.txt".format(int(fid)))
                p = p2 if os.path.exists(p2) else p
            if os.path.exists(p):
                files.append(p)
        for p in files:
            try:
                default_frame = int(os.path.splitext(os.path.basename(p))[0])
            except Exception:
                default_frame = -1
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    z = [x.strip() for x in line.strip().split(",")]
                    if len(z) < 5:
                        continue
                    # D-CTRL cache: frame,x,y,w,h,score (frame may repeat).
                    off = 0
                    try:
                        frame = int(float(z[0]))
                        x, y, w, h = (_parse_float(v) for v in z[1:5])
                        score = _parse_float(z[5], 1.0) if len(z) > 5 else 1.0
                    except Exception:
                        frame = default_frame
                        off = 0
                        x, y, w, h = (_parse_float(v) for v in z[off:off + 4])
                        score = _parse_float(z[off + 4], 1.0) if len(z) > off + 4 else 1.0
                    if frame not in wanted or w <= 1 or h <= 1 or score < score_threshold:
                        continue
                    by_frame[frame].append((np.asarray([x, y, x + w, y + h], dtype=np.float64), float(score)))
    else:
        files = [det_path]
        with open(det_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                z = [x.strip() for x in line.strip().split(",")]
                if len(z) < 6:
                    continue
                try:
                    frame = int(float(z[0]))
                    if frame not in wanted:
                        continue
                    # MOT det.txt: frame,-1,x,y,w,h,score,...
                    x, y, w, h = (_parse_float(v) for v in z[2:6])
                    score = _parse_float(z[6], 1.0) if len(z) > 6 else 1.0
                except Exception:
                    continue
                if w <= 1 or h <= 1 or score < score_threshold:
                    continue
                by_frame[frame].append((np.asarray([x, y, x + w, y + h], dtype=np.float64), float(score)))

    # Keep the detector stream fixed and deterministic for every method.  The
    # top-k rule is applied before any GT-derived labeling.
    for frame in list(by_frame.keys()):
        vals = sorted(by_frame[frame], key=lambda q: (-q[1], q[0][0], q[0][1]))
        by_frame[frame] = vals[:max_detections]
    return by_frame


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_assignment(matrix: np.ndarray, maximize: bool = True) -> List[Tuple[int, int]]:
    """Small deterministic assignment fallback when scipy is unavailable."""
    if matrix.size == 0:
        return []
    n, m = matrix.shape
    pairs = [(float(matrix[r, c]), r, c) for r in range(n) for c in range(m)]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]) if maximize else (x[0], x[1], x[2]))
    used_r, used_c, out = set(), set(), []
    for _, r, c in pairs:
        if r not in used_r and c not in used_c:
            used_r.add(r)
            used_c.add(c)
            out.append((r, c))
    return out


def assign_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.3) -> List[Tuple[int, int, float]]:
    mat = iou_matrix(a, b)
    if mat.size == 0:
        return []
    if linear_sum_assignment is not None:
        rr, cc = linear_sum_assignment(-mat)
        pairs = list(zip(rr.tolist(), cc.tolist()))
    else:
        pairs = greedy_assignment(mat, maximize=True)
    return [(int(r), int(c), float(mat[r, c])) for r, c in pairs if float(mat[r, c]) >= threshold]


def crop_descriptor(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """A frozen, deterministic 64-D RGB/spatial/gradient crop descriptor."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(w - 1, int(math.floor(x1))))
    y1 = max(0, min(h - 1, int(math.floor(y1))))
    x2 = max(x1 + 1, min(w, int(math.ceil(x2))))
    y2 = max(y1 + 1, min(h, int(math.ceil(y2))))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(64, dtype=np.float64)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
    vals: List[float] = []
    full = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    vals.extend(full.mean(axis=(0, 1)).tolist())
    vals.extend(full.std(axis=(0, 1)).tolist())
    vals.extend(rgb.transpose(2, 0, 1).reshape(-1).tolist())  # 48 spatial values
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    vals.extend([float(mag.mean()), float(mag.std())])
    hist, _ = np.histogram(np.clip(mag, 0.0, 4.0), bins=8, range=(0.0, 4.0), density=False)
    vals.extend((hist.astype(np.float64) / max(float(hist.sum()), 1.0)).tolist())
    out = np.asarray(vals, dtype=np.float64)
    if out.size != 64:
        out = np.resize(out, 64)
    return normalize(out)


def _best_detection(frame: Sequence[Detection], obj: int) -> Optional[Detection]:
    vals = [d for d in frame if int(d.gt_id) == int(obj)]
    return max(vals, key=lambda d: d.score) if vals else None


def collect_real_pairs(
    sequences: Sequence[RealSequence],
    history_len: int,
    dropout: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Build context/clean-target pairs from train videos only."""
    rng = np.random.RandomState(int(seed))
    contexts: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    per_seq: Dict[str, int] = {}
    for seq in sequences:
        n = seq.data.positions.shape[1]
        count = 0
        for t in range(1, len(seq.data.detections)):
            for obj in range(n):
                target = seq.data.clean_embeddings[t, obj]
                if not seq.data.visible[t, obj] or np.linalg.norm(target) < 1e-6:
                    continue
                hist: List[Optional[Detection]] = []
                for tt in range(max(0, t - history_len), t):
                    d = _best_detection(seq.data.detections[tt], obj)
                    # Student-side temporal masking is independent of the
                    # clean teacher crop.  It is deterministic per seed.
                    if d is not None and dropout > 0.0 and rng.rand() < dropout:
                        d = None
                    hist.append(d)
                observed = [d for d in hist if d is not None and d.score >= 0.55]
                if observed:
                    last = observed[-1]
                    center = last.center
                    velocity = (observed[-1].center - observed[-2].center) if len(observed) >= 2 else np.zeros(2)
                    last_idx = max(k for k, item in enumerate(hist) if item is last)
                    age = t - (t - len(hist) + last_idx + 1)
                else:
                    center = seq.data.positions[max(0, t - 1), obj]
                    velocity = np.zeros(2, dtype=np.float64)
                    age = history_len
                contexts.append(build_context(hist, center, velocity, history_len, target.size, age))
                targets.append(target)
                count += 1
        per_seq[seq.name] = count
    if not contexts:
        raise RuntimeError("no usable context/teacher pairs from train sequences")
    return np.stack(contexts), np.stack(targets), per_seq


def _global_idtp(pairs: Sequence[Tuple[str, int]]) -> int:
    if not pairs:
        return 0
    gs = sorted(set(g for g, _ in pairs))
    ts = sorted(set(t for _, t in pairs))
    mat = np.zeros((len(gs), len(ts)), dtype=np.int64)
    gi, ti = {g: i for i, g in enumerate(gs)}, {t: i for i, t in enumerate(ts)}
    for g, t in pairs:
        mat[gi[g], ti[t]] += 1
    if linear_sum_assignment is not None:
        rr, cc = linear_sum_assignment(-mat)
        return int(sum(mat[r, c] for r, c in zip(rr, cc)))
    return int(sum(mat[r, c] for r, c in greedy_assignment(mat, maximize=True)))


def evaluate_real_sequence(
    seq: RealSequence,
    tracker: MOTTracker,
    iou_threshold: float,
) -> Dict[str, object]:
    """Evaluate tracker outputs with frame IoU matching and proxy HOTA terms."""
    gt_counts: Dict[str, int] = defaultdict(int)
    tr_counts: Dict[int, int] = defaultdict(int)
    pair_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    observations: List[Tuple[str, int]] = []
    last_match: Dict[str, Tuple[int, int]] = {}
    idsw = 0
    tp = fp = fn = 0
    iou_sum = 0.0
    detector_tp = detector_gt = 0
    frame_rows: List[Dict[str, object]] = []
    occ_tp = occ_total = 0
    t0 = time.perf_counter()

    for t, (frame_id, detections) in enumerate(zip(seq.frame_ids, seq.data.detections)):
        # Detector recall is measured independently of tracker output.
        gt_boxes = seq.gt_boxes_px[t]
        valid_gt = np.where(np.isfinite(gt_boxes[:, 0]))[0]
        detector_gt += int(len(valid_gt))
        det_px = seq.detector_boxes_px[t]
        dmatch = assign_iou(det_px, gt_boxes[valid_gt], threshold=iou_threshold) if len(valid_gt) else []
        detector_tp += len(dmatch)

        output = tracker.step(detections)
        pred_rows: List[Tuple[int, np.ndarray]] = []
        for tid, di in output.items():
            if di < 0 or di >= len(det_px):
                continue
            pred_rows.append((int(tid), det_px[di]))
        pred_boxes = np.stack([x[1] for x in pred_rows], axis=0) if pred_rows else np.zeros((0, 4))
        gt_valid_boxes = gt_boxes[valid_gt] if len(valid_gt) else np.zeros((0, 4))
        matches = assign_iou(pred_boxes, gt_valid_boxes, threshold=iou_threshold)
        matched_pred = set()
        matched_gt = set()
        frame_tp = 0
        for pr, gr, ov in matches:
            tid = pred_rows[pr][0]
            obj = int(valid_gt[gr])
            raw_id = seq.raw_gt_ids[obj]
            matched_pred.add(pr)
            matched_gt.add(gr)
            frame_tp += 1
            iou_sum += ov
            gt_counts[raw_id] += 1
            tr_counts[tid] += 1
            pair_counts[(raw_id, tid)] += 1
            observations.append((raw_id, tid))
            if raw_id in last_match and last_match[raw_id][1] != tid and last_match[raw_id][0] == frame_id - 1:
                idsw += 1
            last_match[raw_id] = (frame_id, tid)
            # ``occluded`` here means detector-visible GT lacked a reliable
            # high-IoU candidate in this frame; it is an analysis mask only.
            if seq.data.occluded[t, obj]:
                occ_total += 1
                occ_tp += 1
        frame_fp = len(pred_rows) - len(matched_pred)
        frame_fn = len(valid_gt) - len(matched_gt)
        tp += frame_tp
        fp += frame_fp
        fn += frame_fn
        frame_rows.append({
            "frame": int(frame_id), "gt": int(len(valid_gt)), "pred": int(len(pred_rows)),
            "tp": int(frame_tp), "fp": int(frame_fp), "fn": int(frame_fn),
        })

    elapsed = max(time.perf_counter() - t0, 1e-9)
    idtp = _global_idtp(observations)
    idfp = max(tp - idtp, 0)
    idfn = max(detector_gt - idtp, 0)
    idf1 = 2.0 * idtp / max(2.0 * idtp + idfp + idfn, 1.0)
    deta = tp / max(tp + fp + fn, 1)
    assa_terms = []
    for g, tid in observations:
        co = pair_counts[(g, tid)]
        den = gt_counts[g] + tr_counts[tid] - co
        assa_terms.append(co / max(float(den), 1.0))
    assa = float(np.mean(assa_terms)) if assa_terms else 0.0
    hota = math.sqrt(max(deta, 0.0) * max(assa, 0.0))
    mota = 1.0 - (fn + fp + idsw) / max(float(detector_gt), 1.0)
    return {
        "sequence": seq.name,
        "frames": int(len(seq.frame_ids)),
        "gt_instances": int(detector_gt),
        "pred_instances": int(tp + fp),
        "TP_proxy": int(tp), "FP_proxy": int(fp), "FN_proxy": int(fn),
        "IDTP_proxy": int(idtp), "IDFP_proxy": int(idfp), "IDFN_proxy": int(idfn),
        "IDF1_proxy": float(idf1),
        "HOTA_proxy": float(hota), "AssA_proxy": float(assa), "DetA_proxy": float(deta),
        "MOTA_proxy": float(mota), "IDSW": int(idsw),
        "IoU_mean_proxy": float(iou_sum / max(tp, 1)),
        "detector_recall_proxy": float(detector_tp / max(detector_gt, 1)),
        "occlusion_association_proxy": float(occ_tp / max(occ_total, 1)),
        "FPS_wall": float(len(seq.frame_ids) / elapsed),
        "elapsed_sec": float(elapsed),
        "metric_definition": "IoU>=0.5 frame matching; IDF1 via global identity assignment; AssA is pair Jaccard mean; HOTA=sqrt(AssA*DetA); MOTA uses proxy FP/FN/IDSW. Not official TrackEval.",
        "frame_rows": frame_rows,
    }


def make_real_sequence(
    dataset: str,
    split: str,
    name: str,
    root: str,
    dance_det_root: str,
    max_frames: int,
    max_objects: int,
    max_detections: int,
    score_threshold: float,
    assign_threshold: float,
) -> RealSequence:
    if dataset == "mot17":
        # The detector variant is part of the sequence name (FRCNN).
        seq_dir = os.path.join(root, split, name)
        image_dir = os.path.join(seq_dir, "img1")
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        det_path = os.path.join(seq_dir, "det", "det.txt")
    else:
        seq_dir = os.path.join(root, split, name)
        image_dir = os.path.join(seq_dir, "img1")
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")
        det_path = os.path.join(dance_det_root, name)
    info = parse_seqinfo(seq_dir)
    width, height = int(info["imwidth"]), int(info["imheight"])
    image_rows = numeric_image_names(image_dir, max_frames)
    if not image_rows:
        raise RuntimeError("no images under {}".format(image_dir))
    frame_ids = [fid for fid, _ in image_rows]
    max_frame = max(frame_ids)
    gt_by_frame = load_gt_rows(gt_path, dataset, max_frame)
    counts: Dict[str, int] = defaultdict(int)
    for vals in gt_by_frame.values():
        for rid, _, _ in vals:
            counts[rid] += 1
    ordered_ids = sorted(counts, key=lambda rid: (-counts[rid], rid))
    if max_objects > 0:
        ordered_ids = ordered_ids[:max_objects]
    raw_ids = list(ordered_ids)
    id_to_local = {rid: i for i, rid in enumerate(raw_ids)}
    n, T = len(raw_ids), len(frame_ids)
    gt_px = np.full((T, n, 4), np.nan, dtype=np.float64)
    visible = np.zeros((T, n), dtype=bool)
    positions = np.zeros((T, n, 2), dtype=np.float64)
    sizes = np.zeros((T, n, 2), dtype=np.float64)
    clean = np.zeros((T, n, 64), dtype=np.float64)
    detector_rows = load_detector_rows(dataset, det_path, frame_ids, score_threshold, max_detections)
    detector_boxes_px: List[np.ndarray] = []
    detector_scores: List[np.ndarray] = []
    detections: List[List[Detection]] = []
    image_digest_h = hashlib.sha256()
    missing_images = 0
    unmatched_gt = 0
    for t, (fid, image_path) in enumerate(image_rows):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            missing_images += 1
            image = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Hash the exact bytes used for this bounded run, not the whole
            # multi-gigabyte dataset.
            try:
                image_digest_h.update(open(image_path, "rb").read())
            except Exception:
                pass
        rows = gt_by_frame.get(fid, [])
        for rid, box, _vis in rows:
            if rid not in id_to_local:
                continue
            j = id_to_local[rid]
            box = box.copy()
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            gt_px[t, j] = box
            visible[t, j] = True
            center = np.asarray([(box[0] + box[2]) * 0.5 / width, (box[1] + box[3]) * 0.5 / height])
            size = np.asarray([(box[2] - box[0]) / width, (box[3] - box[1]) / height])
            positions[t, j] = center
            sizes[t, j] = size
            clean[t, j] = crop_descriptor(image, box)

        raw_det = detector_rows.get(fid, [])
        boxes = []
        scores = []
        for box, score in raw_det:
            b = box.copy()
            b[[0, 2]] = np.clip(b[[0, 2]], 0, width)
            b[[1, 3]] = np.clip(b[[1, 3]], 0, height)
            if b[2] <= b[0] or b[3] <= b[1]:
                continue
            boxes.append(b)
            scores.append(float(score))
        box_arr = np.stack(boxes, axis=0) if boxes else np.zeros((0, 4), dtype=np.float64)
        score_arr = np.asarray(scores, dtype=np.float64)
        detector_boxes_px.append(box_arr)
        detector_scores.append(score_arr)
        # Post-hoc one-to-one labels for fitting/evaluation only.  The online
        # tracker sees only center/size/embedding/score.
        valid_gt = np.where(visible[t])[0]
        labels = [-1] * len(boxes)
        if len(valid_gt) and len(boxes):
            pairs = assign_iou(box_arr, gt_px[t, valid_gt], threshold=assign_threshold)
            for di, gi, _ in pairs:
                labels[di] = int(valid_gt[gi])
            unmatched_gt += max(len(valid_gt) - len(pairs), 0)
        frame_dets: List[Detection] = []
        for di, (b, score) in enumerate(zip(boxes, scores)):
            center = np.asarray([(b[0] + b[2]) * 0.5 / width, (b[1] + b[3]) * 0.5 / height])
            size = np.asarray([(b[2] - b[0]) / width, (b[3] - b[1]) / height])
            frame_dets.append(Detection(center=center, size=size, embedding=crop_descriptor(image, b), score=score, gt_id=labels[di]))
        detections.append(frame_dets)

    # Fill missing positions/sizes by carrying the last valid state, solely
    # for context construction; missingness remains explicit in ``visible``.
    for j in range(n):
        for t in range(T):
            if t > 0 and not visible[t, j]:
                positions[t, j] = positions[t - 1, j]
                sizes[t, j] = sizes[t - 1, j]
            elif not visible[t, j] and t == 0:
                positions[t, j] = 0.5
                sizes[t, j] = np.asarray([0.05, 0.1])
    # Detector-missing/low-score mask is an analysis-only visibility proxy.
    occluded = np.zeros((T, n), dtype=bool)
    for t in range(T):
        for j in range(n):
            obs = _best_detection(detections[t], j)
            occluded[t, j] = bool(visible[t, j] and (obs is None or obs.score < 0.55))
    det_h = hashlib.sha256()
    for t, fid in enumerate(frame_ids):
        det_h.update(str(fid).encode("ascii"))
        for b, s in zip(detector_boxes_px[t], detector_scores[t]):
            det_h.update(np.asarray(b, dtype=np.float64).tobytes())
            det_h.update(np.asarray([s], dtype=np.float64).tobytes())
    return RealSequence(
        name=name, dataset=dataset, split=split, seq_dir=seq_dir, image_dir=image_dir,
        gt_path=gt_path, det_path=det_path, width=width, height=height,
        frame_ids=frame_ids, gt_boxes_px=gt_px, detector_boxes_px=detector_boxes_px,
        detector_scores=detector_scores,
        data=SequenceData(positions=positions, sizes=sizes, visible=visible,
                          clean_embeddings=clean, detections=detections, occluded=occluded),
        raw_gt_ids=raw_ids, selected_gt_count=n,
        detector_digest=det_h.hexdigest()[:16], image_digest=image_digest_h.hexdigest()[:16],
        load_notes={"missing_images": missing_images, "unmatched_gt_labels": unmatched_gt,
                    "gt_counts_selected": {k: int(v) for k, v in counts.items() if k in id_to_local}},
    )


def aggregate_rows(rows: Sequence[Dict[str, object]], methods: Sequence[str]) -> Dict[str, object]:
    keys = ["IDF1_proxy", "HOTA_proxy", "AssA_proxy", "DetA_proxy", "MOTA_proxy", "IDSW", "FPS_wall", "detector_recall_proxy"]
    out: Dict[str, object] = {}
    for method in methods:
        vals = [r for r in rows if r["method"] == method]
        rec: Dict[str, object] = {"n_sequences": len(vals)}
        for key in keys:
            a = np.asarray([float(v[key]) for v in vals], dtype=np.float64)
            rec[key + "_mean"] = float(a.mean()) if len(a) else 0.0
            rec[key + "_std"] = float(a.std()) if len(a) else 0.0
        out[method] = rec
    return out


def run(args: argparse.Namespace) -> Dict[str, object]:
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if not methods:
        raise ValueError("--methods cannot be empty")
    # Explicit video-disjoint defaults.  MOT17 names include detector variant;
    # DanceTrack val is the held-out split from train.
    mot_train_names = [x.strip() for x in args.mot_train.split(",") if x.strip()]
    mot_eval_names = [x.strip() for x in args.mot_eval.split(",") if x.strip()]
    dance_train_names = [x.strip() for x in args.dance_train.split(",") if x.strip()]
    dance_eval_names = [x.strip() for x in args.dance_eval.split(",") if x.strip()]
    failures: List[Dict[str, object]] = []
    train: List[RealSequence] = []
    evals: List[RealSequence] = []
    for dataset, split, names, dest in (
        ("mot17", "train", mot_train_names, train),
        ("mot17", "train", mot_eval_names, evals),
        ("dancetrack", "train", dance_train_names, train),
        ("dancetrack", "val", dance_eval_names, evals),
    ):
        for name in names:
            try:
                seq = make_real_sequence(dataset, split, name, args.mot_root if dataset == "mot17" else args.dance_root,
                                         args.dance_det_root, args.max_frames, args.max_objects,
                                         args.max_detections, args.score_threshold, args.assign_threshold)
                dest.append(seq)
                print("[load] {} {} frames={} selected_gt={} detections={} det_digest={}".format(
                    dataset, name, len(seq.frame_ids), seq.selected_gt_count,
                    sum(len(x) for x in seq.data.detections), seq.detector_digest), flush=True)
            except Exception as exc:
                failures.append({"stage": "load", "dataset": dataset, "split": split, "sequence": name,
                                 "error": repr(exc), "traceback": traceback.format_exc()})
                print("[FAIL load] {} {}: {}".format(dataset, name, exc), flush=True)
    if not train or not evals:
        raise RuntimeError("need at least one train and one eval sequence; failures={}".format(failures))

    contexts, targets, pair_counts = collect_real_pairs(train, args.history_len, args.context_dropout, args.seed)
    predictor_inv = LinearJEPA(covariance_weight=0.0, ridge=args.ridge)
    fit_inv = predictor_inv.fit(contexts, targets)
    predictor_cov = LinearJEPA(covariance_weight=args.covariance_weight, ridge=args.ridge,
                               iterations=args.cov_iterations)
    fit_cov = predictor_cov.fit(contexts, targets)
    shuffled = np.random.RandomState(args.seed + 99173).permutation(len(targets))
    predictor_no = LinearJEPA(covariance_weight=0.0, ridge=args.ridge)
    fit_no = predictor_no.fit(contexts, targets[shuffled])
    print("[fit] pairs={} per_sequence={} inv_mse={:.6f} cov_mse={:.6f} no_teacher_mse={:.6f}".format(
        len(targets), pair_counts, fit_inv["invariance_mse"], fit_cov["invariance_mse"], fit_no["invariance_mse"]), flush=True)

    rows: List[Dict[str, object]] = []
    for seq in evals:
        for method in methods:
            if method == "baseline" or method == "gated_ema":
                pred = None
            elif method == "jepa_inv":
                pred = predictor_inv
            elif method == "jepa_cov":
                pred = predictor_cov
            elif method == "no_teacher":
                pred = predictor_no
            else:
                raise ValueError("unknown method {}".format(method))
            try:
                tracker = MOTTracker(method=method, embedding_dim=64, predictor=pred,
                                     history_len=args.history_len, max_age=args.max_age,
                                     match_cost=args.match_cost, spawn_score=args.spawn_score)
                metrics = evaluate_real_sequence(seq, tracker, args.eval_iou_threshold)
                metrics.update({"dataset": seq.dataset, "split": seq.split, "method": method,
                                "detector_digest": seq.detector_digest, "image_digest": seq.image_digest})
                rows.append(metrics)
                print("[eval] {:12s} {:18s} IDF1*={:.4f} HOTA*={:.4f} AssA*={:.4f} DetA*={:.4f} MOTA*={:.4f} IDSW={} FPS={:.1f}".format(
                    method, seq.name, metrics["IDF1_proxy"], metrics["HOTA_proxy"], metrics["AssA_proxy"],
                    metrics["DetA_proxy"], metrics["MOTA_proxy"], metrics["IDSW"], metrics["FPS_wall"]), flush=True)
            except Exception as exc:
                failures.append({"stage": "eval", "sequence": seq.name, "method": method,
                                 "error": repr(exc), "traceback": traceback.format_exc()})
                print("[FAIL eval] {} {}: {}".format(seq.name, method, exc), flush=True)

    source = []
    for seq in train + evals:
        source.append({"dataset": seq.dataset, "split": seq.split, "sequence": seq.name,
                       "seq_dir": seq.seq_dir, "image_dir": seq.image_dir, "gt_path": seq.gt_path,
                       "det_path": seq.det_path, "width": seq.width, "height": seq.height,
                       "frames": len(seq.frame_ids), "frame_first_last": [seq.frame_ids[0], seq.frame_ids[-1]],
                       "selected_gt_ids": seq.raw_gt_ids, "selected_gt_count": seq.selected_gt_count,
                       "detector_digest_sha256_16": seq.detector_digest,
                       "image_digest_sha256_16": seq.image_digest, "load_notes": seq.load_notes})
    summary: Dict[str, object] = {
        "experiment": "real_mot_jepa_association_smoke",
        "status": "complete" if rows else "failed",
        "config": {
            "methods": methods, "history_len": args.history_len, "max_frames": args.max_frames,
            "max_objects": args.max_objects, "max_detections": args.max_detections,
            "score_threshold": args.score_threshold, "assign_threshold": args.assign_threshold,
            "eval_iou_threshold": args.eval_iou_threshold, "context_dropout": args.context_dropout,
            "covariance_weight": args.covariance_weight, "cov_iterations": args.cov_iterations,
            "ridge": args.ridge, "max_age": args.max_age, "match_cost": args.match_cost,
            "spawn_score": args.spawn_score, "seed": args.seed,
            "mot_train": mot_train_names, "mot_eval": mot_eval_names,
            "dance_train": dance_train_names, "dance_eval": dance_eval_names,
            "train_eval_video_disjoint": True,
            "detector_protocol": "MOT17 FRCNN det.txt; DanceTrack official YOLOX-X/ByteTrack cache; same rows for all methods",
            "teacher_protocol": "clean current GT crop on train videos only, stop-gradient concept; student history uses detector crops with deterministic dropout",
            "online_uses_gt": False,
            "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "opencv": cv2.__version__, "scipy_assignment": linear_sum_assignment is not None,
            "script_sha256": short_digest(os.path.abspath(__file__)),
        },
        "data_roots": {"mot17": os.path.abspath(args.mot_root), "dancetrack": os.path.abspath(args.dance_root),
                       "dancetrack_detector": os.path.abspath(args.dance_det_root)},
        "sources": source,
        "fit": {"pairs": int(len(targets)), "per_sequence": pair_counts,
                "jepa_inv": fit_inv, "jepa_cov": fit_cov, "no_teacher": fit_no},
        "results": rows,
        "aggregate": aggregate_rows(rows, methods),
        "failures": failures,
        "metric_warning": "All starred metrics are transparent IoU/frame-matching proxies; official TrackEval was not invoked by this smoke. Do not report them as official MOTChallenge scores.",
    }
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    csv_out = os.path.splitext(out)[0] + ".csv"
    fields = ["dataset", "split", "sequence", "method", "frames", "gt_instances", "pred_instances",
              "TP_proxy", "FP_proxy", "FN_proxy", "IDF1_proxy", "HOTA_proxy", "AssA_proxy",
              "DetA_proxy", "MOTA_proxy", "IDSW", "IoU_mean_proxy", "detector_recall_proxy",
              "FPS_wall", "detector_digest", "image_digest"]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fields})
    print("[write] {}\n[write] {}".format(out, csv_out), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mot-root", default=DEFAULT_MOT_ROOT)
    ap.add_argument("--dance-root", default=DEFAULT_DANCE_ROOT)
    ap.add_argument("--dance-det-root", default=DEFAULT_DANCE_DET_ROOT)
    ap.add_argument("--mot-train", default="MOT17-02-FRCNN")
    ap.add_argument("--mot-eval", default="MOT17-04-FRCNN")
    ap.add_argument("--dance-train", default="dancetrack0001")
    ap.add_argument("--dance-eval", default="dancetrack0007")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--max-objects", type=int, default=24)
    ap.add_argument("--max-detections", type=int, default=40)
    ap.add_argument("--score-threshold", type=float, default=0.10)
    ap.add_argument("--assign-threshold", type=float, default=0.30)
    ap.add_argument("--eval-iou-threshold", type=float, default=0.50)
    ap.add_argument("--history-len", type=int, default=4)
    ap.add_argument("--context-dropout", type=float, default=0.15)
    ap.add_argument("--covariance-weight", type=float, default=0.05)
    ap.add_argument("--cov-iterations", type=int, default=30)
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--max-age", type=int, default=7)
    ap.add_argument("--match-cost", type=float, default=1.18)
    ap.add_argument("--spawn-score", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="baseline,gated_ema,jepa_inv,jepa_cov,no_teacher")
    ap.add_argument("--output", default=os.path.join(HERE, "results", "real_mot_jepa_smoke.json"))
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        print("[fatal] {}".format(exc), file=sys.stderr)
        traceback.print_exc()
        # Keep a machine-readable failure record when the output path is
        # available; this is useful for auditing blocked real-data runs.
        try:
            out = os.path.abspath(args.output)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(os.path.splitext(out)[0] + ".failure.json", "w", encoding="utf-8") as f:
                json.dump({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                           "config": vars(args), "script_sha256": short_digest(os.path.abspath(__file__))}, f, indent=2)
        except Exception:
            pass
        raise
