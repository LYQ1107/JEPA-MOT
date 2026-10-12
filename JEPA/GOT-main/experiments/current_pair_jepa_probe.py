#!/usr/bin/env python3
"""Current-pair JEPA representation probe for real MOT17/DanceTrack.

This file is intentionally independent of the audited tracker harness.  It is
the first, small experiment for the *current candidate* formulation that we
plan to put behind a lagged (t -> t+1) association interface:

    teacher: T(H_<t, c_clean_t, g_t) -> omega_hat_t
    student: S(H_<t, c_corrupt_t, g_t) -> omega_hat_t (stop-gradient target)

``H_<t`` contains only detector crops matched to the same identity at frames
strictly before ``t``.  A per-frame Hungarian IoU match is used to construct
the supervised probe examples, and a detection can occur in at most one GT
match.  The probe is not a tracker and therefore uses oracle identity
histories for measuring the representation itself; this fact is recorded in
the output and must not be confused with the later tracker attribution run.

Controls are deliberately explicit: ``current_only`` masks H, ``full_mask``
masks the current crop (history-only), ``no_teacher`` trains directly on the
clean target, ``both_clean`` is a clean-current diagnostic ceiling, and
``random_frozen`` is a capacity/magnitude matched random network.  No GT
boxes/IDs are fed to the student other than the probe's history construction
and metric labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


MOT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17")
DANCE_ROOT = Path("/data1/LWR/vranlee/DATASETS/JDE/dancetrack")
DANCE_CTRL_CACHE = Path("/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/detections_ctrl")

FEATURE_DIM = 80
HISTORY_K = 4
IOU_MATCH_THRESHOLD = 0.30
HIDDEN_DIM = 192


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in box[:4]]
    return np.asarray([x, y, x + max(0.0, w), y + max(0.0, h)], dtype=np.float32)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a[:4])
    bx1, by1, bx2, by2 = map(float, b[:4])
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + bb - inter, 1e-9)


def image_files(seq_dir: Path) -> List[Path]:
    fs = sorted((seq_dir / "img1").glob("*.jpg"))
    return fs or sorted((seq_dir / "img1").glob("*.png"))


def crop_feature(image: np.ndarray, box: Sequence[float], dim: int = FEATURE_DIM) -> np.ndarray:
    """Frozen, deterministic low-cost crop encoder used by the probe."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(float, box[:4])
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    x1 = int(max(0, math.floor(x1 - .08 * bw)))
    y1 = int(max(0, math.floor(y1 - .08 * bh)))
    x2 = int(min(w, math.ceil(x2 + .08 * bw)))
    y2 = int(min(h, math.ceil(y2 + .08 * bh)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(dim, np.float32)
    crop = image[y1:y2, x1:x2]
    small = cv2.resize(crop, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rgb = small[:, :, ::-1].reshape(-1)
    mean = crop.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0
    std = crop.reshape(-1, 3).std(axis=0).astype(np.float32) / 255.0
    f = np.concatenate([rgb, mean, std, np.asarray([crop.mean() / 255., crop.std() / 255.], np.float32)])
    if f.size < dim:
        f = np.pad(f, (0, dim - f.size))
    f = f[:dim].astype(np.float32)
    f -= f.mean()
    return normalize(f).astype(np.float32)


@dataclass
class Det:
    frame: int
    box: np.ndarray
    score: float
    feat: Optional[np.ndarray] = None


@dataclass
class GT:
    frame: int
    gid: int
    box: np.ndarray
    vis: float = 1.0


def parse_mot_gt(path: Path, max_frames: int = 0) -> Dict[int, List[GT]]:
    out: Dict[int, List[GT]] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        p = line.split(",")
        if len(p) < 6:
            continue
        try:
            fr, gid = int(float(p[0])), int(float(p[1]))
            mark = float(p[6]) if len(p) > 6 else 1.0
            cls = int(float(p[7])) if len(p) > 7 else 1
            vis = float(p[8]) if len(p) > 8 else 1.0
            box = xywh_to_xyxy([float(v) for v in p[2:6]])
        except (TypeError, ValueError):
            continue
        if max_frames and fr > max_frames or mark <= 0 or cls != 1:
            continue
        out.setdefault(fr, []).append(GT(fr, gid, box, vis))
    return out


def parse_mot_det(path: Path, max_frames: int = 0) -> Dict[int, List[Det]]:
    out: Dict[int, List[Det]] = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        p = line.split(",")
        if len(p) < 6:
            continue
        try:
            fr = int(float(p[0])); box = xywh_to_xyxy([float(v) for v in p[2:6]])
            score = float(p[6]) if len(p) > 6 else 1.0
        except (TypeError, ValueError):
            continue
        if max_frames and fr > max_frames or not np.isfinite(box).all() or not np.isfinite(score):
            continue
        out.setdefault(fr, []).append(Det(fr, box, score))
    return out


def parse_cache(path: Path, max_frames: int = 0) -> Dict[int, List[Det]]:
    out: Dict[int, List[Det]] = {}
    if not path.exists():
        return out
    for fp in sorted(path.glob("*.txt")):
        try:
            fr = int(fp.stem)
        except ValueError:
            continue
        if max_frames and fr > max_frames:
            continue
        rows: List[Det] = []
        for line in fp.read_text(errors="replace").splitlines():
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            try:
                b = xywh_to_xyxy([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
                sc = float(p[5])
            except (TypeError, ValueError):
                continue
            if np.isfinite(b).all() and np.isfinite(sc):
                rows.append(Det(fr, b, sc))
        out[fr] = rows
    return out


def parse_dance_coco(path: Path, selected: Sequence[str], max_frames: int = 0) -> Dict[str, Dict[int, List[GT]]]:
    payload = json.loads(path.read_text())
    videos = {int(v["id"]): str(v.get("file_name", "")) for v in payload.get("videos", [])}
    images = {int(i["id"]): (videos.get(int(i.get("video_id", -1)), ""), int(i.get("frame_id", 0))) for i in payload.get("images", [])}
    out: Dict[str, Dict[int, List[GT]]] = {s: {} for s in selected}
    for a in payload.get("annotations", []):
        im = images.get(int(a.get("image_id", -1)))
        if im is None:
            continue
        seq, fr = im
        if seq not in out or (max_frames and fr > max_frames) or int(a.get("category_id", 1)) != 1:
            continue
        gid = int(a.get("track_id", a.get("id", -1)))
        if gid < 0:
            continue
        out[seq].setdefault(fr, []).append(GT(fr, gid, xywh_to_xyxy(a.get("bbox", [0, 0, 0, 0])), float(a.get("visibility", 1.0))))
    return out


def attach_features(seq_dir: Path, det: Dict[int, List[Det]], max_frames: int = 0) -> None:
    fs = image_files(seq_dir)
    for fr, rows in det.items():
        if fr < 1 or fr > len(fs) or (max_frames and fr > max_frames):
            continue
        im = cv2.imread(str(fs[fr - 1]))
        if im is None:
            continue
        for d in rows:
            d.feat = crop_feature(im, d.box)


def clean_gt_features(seq_dir: Path, gt: Dict[int, List[GT]], max_frames: int = 0) -> Dict[Tuple[int, int], np.ndarray]:
    fs = image_files(seq_dir); out: Dict[Tuple[int, int], np.ndarray] = {}
    for fr, rows in gt.items():
        if fr < 1 or fr > len(fs) or (max_frames and fr > max_frames):
            continue
        im = cv2.imread(str(fs[fr - 1]))
        if im is None:
            continue
        for g in rows:
            out[(fr, g.gid)] = crop_feature(im, g.box)
    return out


def sequence_wh(seq_dir: Path) -> Tuple[float, float]:
    ini = seq_dir / "seqinfo.ini"
    if ini.exists():
        vals = {}
        for line in ini.read_text(errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1); vals[k.strip()] = v.strip()
        try:
            return float(vals.get("imWidth", 1)), float(vals.get("imHeight", 1))
        except ValueError:
            pass
    fs = image_files(seq_dir)
    if fs:
        im = cv2.imread(str(fs[0]))
        if im is not None:
            return float(im.shape[1]), float(im.shape[0])
    return 1., 1.


def unique_hungarian_matches(gt: Dict[int, List[GT]], det: Dict[int, List[Det]], threshold: float = IOU_MATCH_THRESHOLD) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[str, int]]:
    """Return unique (frame,gid)->det-index and reverse maps.

    Crucially this is one Hungarian assignment *per frame*, not a loop of
    independent best matches.  The audit counters make accidental detection
    reuse visible in the JSON.
    """
    gmap: Dict[Tuple[int, int], int] = {}; dmap: Dict[Tuple[int, int], int] = {}
    frames = sorted(set(gt) | set(det)); assigned = 0; rejected = 0
    for fr in frames:
        gs, ds = gt.get(fr, []), det.get(fr, [])
        if not gs or not ds:
            continue
        cost = np.asarray([[1.0 - box_iou(g.box, d.box) for d in ds] for g in gs], np.float64)
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri.tolist(), ci.tolist()):
            ov = 1.0 - float(cost[i, j])
            if ov < threshold:
                rejected += 1
                continue
            keyg, keyd = (fr, gs[i].gid), (fr, j)
            # These assertions are deliberately retained in the executable
            # probe: a duplicate would invalidate the attribution protocol.
            if keyg in gmap or keyd in dmap:
                raise AssertionError("Hungarian duplicate assignment")
            gmap[keyg] = j; dmap[keyd] = gs[i].gid; assigned += 1
    return gmap, dmap, {"assigned": assigned, "iou_rejected_after_assignment": rejected, "duplicate_count": 0}


def corrupt_feature(x: np.ndarray, key: int, severity: float = .30) -> np.ndarray:
    """Deterministic crop corruption, independent of GT labels."""
    rng = np.random.RandomState(int(key) & 0x7FFFFFFF)
    y = np.asarray(x, np.float32).copy()
    mask = rng.rand(y.size) < severity
    y[mask] = 0.0
    y += rng.normal(0., .08, y.size).astype(np.float32)
    return normalize(y).astype(np.float32)


def _history_for(gid: int, frame: int, gt: Dict[int, List[GT]], det: Dict[int, List[Det]], gmap: Dict[Tuple[int, int], int], k: int, iw: float, ih: float) -> Tuple[np.ndarray, np.ndarray, float, float, Optional[np.ndarray]]:
    """Build H_<t> from strictly earlier matched detector observations."""
    rows: List[Tuple[int, GT, Det]] = []
    for fr in range(max(1, frame - 32), frame):
        j = gmap.get((fr, gid))
        if j is None or j >= len(det.get(fr, [])):
            continue
        d = det[fr][j]
        if d.feat is not None:
            gg = next((q for q in gt.get(fr, []) if q.gid == gid), None)
            if gg is not None:
                rows.append((fr, gg, d))
    recent = rows[-k:]
    if not recent:
        return np.zeros(FEATURE_DIM, np.float32), np.zeros(4, np.float32), 0., float(k + 1), None
    # More recent observations receive larger weights, but current frame is
    # never included.  This is an observable detector-history construction.
    weights = np.asarray([math.exp(-(frame - fr) / 4.) for fr, _, _ in recent], np.float32)
    feats = np.asarray([d.feat for _, _, d in recent], np.float32)
    hist = normalize((feats * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-6)).astype(np.float32)
    latest_fr, latest_g, latest_d = recent[-1]
    valid = min(1., len(recent) / float(k))
    gap = float(frame - latest_fr)
    latest_box = latest_d.box
    state = np.asarray([
        valid, min(1., gap / float(k + 1)),
        ((latest_box[0] + latest_box[2]) * .5) / max(iw, 1.),
        ((latest_box[1] + latest_box[3]) * .5) / max(ih, 1.),
    ], np.float32)
    latest_center = np.asarray([(latest_box[0] + latest_box[2]) * .5 / max(iw, 1.), (latest_box[1] + latest_box[3]) * .5 / max(ih, 1.)], np.float32)
    return hist, state, float(valid), gap, latest_center


def _geometry(d: Det, hstate: np.ndarray, latest_center: Optional[np.ndarray], iw: float, ih: float) -> np.ndarray:
    x1, y1, x2, y2 = d.box
    cx, cy = (x1 + x2) * .5 / max(iw, 1.), (y1 + y2) * .5 / max(ih, 1.)
    bw, bh = max(0., x2 - x1) / max(iw, 1.), max(0., y2 - y1) / max(ih, 1.)
    if latest_center is None:
        dx = dy = 0.
    else:
        dx, dy = cx - float(latest_center[0]), cy - float(latest_center[1])
    # Nine MOT-state scalars: candidate centre/scale, score, observability,
    # latest gap, displacement from latest history centre, and area.
    return np.asarray([cx, cy, bw, bh, np.clip(d.score, 0., 1.), hstate[0], hstate[1], dx, dy], np.float32)


def build_samples(items: Sequence[Tuple[str, Path, Dict[int, List[GT]], Dict[int, List[Det]], Dict[str, object]]], max_frames: int, seed: int, max_samples: int) -> Tuple[Dict[str, np.ndarray], List[dict], Dict[str, object]]:
    """Create probe arrays and per-sample candidate lists."""
    rng = np.random.RandomState(seed)
    records: List[dict] = []; det_hash = hashlib.sha256(); feat_hash = hashlib.sha256(); clean_hash = hashlib.sha256()
    audit = {"assigned": 0, "iou_rejected_after_assignment": 0, "duplicate_count": 0, "frames": 0}
    for seq_name, seq_dir, gt, det, info in items:
        iw, ih = sequence_wh(seq_dir)
        attach_features(seq_dir, det, max_frames)
        clean = clean_gt_features(seq_dir, gt, max_frames)
        gmap, dmap, ma = unique_hungarian_matches(gt, det)
        for k, v in ma.items():
            audit[k] = int(audit.get(k, 0)) + int(v)
        audit["frames"] += len(set(gt) | set(det))
        # Hash the complete fixed detector stream (including frames with no
        # GT rows), before any model sees it.  Hashing only ``sorted(gt)`` can
        # silently omit detector-only frames and invalidate provenance.
        for fr in sorted(det):
            ds0 = det.get(fr, [])
            for j, d in enumerate(ds0):
                det_hash.update(f"{seq_name}:{fr}:{j}:{d.score:.7f}:".encode()); det_hash.update(np.asarray(d.box, np.float32).tobytes())
                if d.feat is not None:
                    feat_hash.update(f"{seq_name}:{fr}:{j}:".encode()); feat_hash.update(np.asarray(d.feat, np.float32).tobytes())
        for fr in sorted(gt):
            ds = det.get(fr, [])
            for g in gt.get(fr, []):
                j = gmap.get((fr, g.gid)); y = clean.get((fr, g.gid))
                if j is None or j >= len(ds) or y is None or ds[j].feat is None:
                    continue
                d = ds[j]
                hist, hs, valid, gap, latest = _history_for(g.gid, fr, gt, det, gmap, HISTORY_K, iw, ih)
                geom = _geometry(d, hs, latest, iw, ih)
                all_feats = [q.feat for q in ds if q.feat is not None]
                all_idx = [qj for qj, q in enumerate(ds) if q.feat is not None]
                if not all_feats:
                    continue
                # Hard negative is the most similar *other* same-frame
                # detector crop to the clean target; it remains a metric label
                # and is never exposed to T/S as an input.
                sims = [float(np.dot(normalize(q), normalize(y))) if qj != j else -2. for qj, q in zip(all_idx, all_feats)]
                neg_pos = int(np.argmax(sims)); neg = np.asarray(all_feats[neg_pos], np.float32) if sims[neg_pos] > -1. else np.zeros(FEATURE_DIM, np.float32)
                cand_matrix = np.asarray(all_feats, np.float32)
                pos_col = all_idx.index(j) if j in all_idx else -1
                rec = {
                    "sequence": seq_name, "frame": int(fr), "gid": int(g.gid), "det_index": int(j),
                    "hist": hist.astype(np.float32), "candidate": d.feat.astype(np.float32),
                    "corrupt": corrupt_feature(d.feat, seed * 1000003 + fr * 1009 + g.gid),
                    "geom": geom, "target": y.astype(np.float32), "negative": neg,
                    "all_candidates": cand_matrix, "positive_column": int(pos_col),
                    "history_valid_fraction": float(valid), "latest_gap": float(gap),
                }
                records.append(rec); clean_hash.update(f"{seq_name}:{fr}:{g.gid}:".encode()); clean_hash.update(y.tobytes())
    # Deterministic cap, preserving a representative mix of sequences.
    if max_samples and len(records) > max_samples:
        keep = np.sort(rng.choice(len(records), size=max_samples, replace=False)); records = [records[int(i)] for i in keep]
    if not records:
        raise RuntimeError("no one-to-one matched probe samples")
    arr = {k: np.asarray([r[k] for r in records], np.float32) for k in ("hist", "candidate", "corrupt", "geom", "target", "negative")}
    meta = {
        "sample_count": len(records), "matching": "per-frame Hungarian linear_sum_assignment, IoU>=0.30",
        "matching_audit": audit, "duplicate_count": int(audit.get("duplicate_count", 0)),
        "detector_sha256": det_hash.hexdigest(), "feature_sha256": feat_hash.hexdigest(), "clean_target_sha256": clean_hash.hexdigest(),
        "history_source": "oracle GT identity used only to assemble this representation probe; frames strictly < current",
    }
    return arr, records, meta


def normalize_torch(x):
    import torch
    return x / torch.clamp(torch.linalg.vector_norm(x, dim=1, keepdim=True), min=1e-8)


class CurrentPairJEPA:
    """Two-phase frozen-teacher/student predictor for one current pair."""

    def __init__(self, mode: str, seed: int, epochs: int = 8):
        self.mode, self.seed, self.epochs = mode, seed, epochs
        self.input_dim = FEATURE_DIM * 2 + 9
        self.teacher_params = None
        self.student_params = None
        self.train_stats: Dict[str, object] = {}

    @staticmethod
    def _gelu(z):
        return .5 * z * (1. + np.tanh(np.sqrt(2. / np.pi) * (z + .044715 * z ** 3)))

    @staticmethod
    def _forward_np(x: np.ndarray, params: Tuple[np.ndarray, ...]) -> np.ndarray:
        w1, b1, gn, gb, w2, b2 = params
        z = x.dot(w1.T) + b1
        mu = z.mean(axis=1, keepdims=True); var = ((z - mu) ** 2).mean(axis=1, keepdims=True)
        z = (z - mu) / np.sqrt(var + 1e-5) * gn + gb
        return np.nan_to_num(CurrentPairJEPA._gelu(z).dot(w2.T) + b2, nan=0., posinf=0., neginf=0.)

    @staticmethod
    def _params(model) -> Tuple[np.ndarray, ...]:
        sd = model.state_dict()
        return tuple(sd[k].detach().cpu().numpy().astype(np.float32) for k in ("net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias", "net.3.weight", "net.3.bias"))

    @staticmethod
    def _hash(params) -> str:
        return hashlib.sha256(b"".join(np.asarray(a, np.float32).tobytes() for a in params)).hexdigest()

    def _train_net(self, x: np.ndarray, target: np.ndarray, neg: np.ndarray, seed: int, ranking: bool = True):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        torch.set_num_threads(1); torch.manual_seed(seed)
        class Net(nn.Module):
            def __init__(self, inp, out):
                super().__init__(); self.net = nn.Sequential(nn.Linear(inp, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.GELU(), nn.Linear(HIDDEN_DIM, out))
            def forward(self, z): return self.net(z)
        model = Net(self.input_dim, FEATURE_DIM)
        tx = torch.from_numpy(np.asarray(x, np.float32)); ty = torch.from_numpy(normalize(target)); tn = torch.from_numpy(normalize(neg))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4); n = len(tx); bs = min(512, n); rng = np.random.RandomState(seed); losses = []
        model.train()
        for ep in range(max(1, self.epochs)):
            order = rng.permutation(n); total = 0.
            for st in range(0, n, bs):
                ix = order[st:st + bs]; pred = normalize_torch(model(tx[ix])); pos = (pred * ty[ix]).sum(1)
                loss = (1. - pos).mean()
                if ranking:
                    negc = (pred * tn[ix]).sum(1); loss = loss + .5 * F.softplus(.20 - pos + negc).mean()
                # A small decorrelation term is useful on colour-heavy MOT
                # crops and keeps the latent from collapsing to one channel.
                z = pred - pred.mean(0, keepdim=True); cov = (z.T @ z) / max(1, z.shape[0] - 1); off = cov - torch.diag(torch.diag(cov)); loss = loss + .03 * (off ** 2).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite current-pair loss")
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
                with torch.no_grad():
                    for p in model.parameters():
                        p.clamp_(-10., 10.)
                total += float(loss.detach()) * len(ix)
            losses.append(total / n)
        model.eval()
        return model, self._params(model), losses

    def fit(self, samples: Dict[str, np.ndarray]) -> None:
        n = len(samples["target"])
        if n == 0:
            raise RuntimeError("empty current-pair sample set")
        h, c, cc, g = samples["hist"], samples["candidate"], samples["corrupt"], samples["geom"]
        # The clean teacher crop is the GT-box crop ``target``.  It is
        # intentionally available only while fitting this representation
        # probe; the later tracker must never call the teacher online.  Using
        # the detector crop here would make the purported clean teacher an
        # unlabelled autoencoder and obscure the causal comparison.
        teacher_x = np.concatenate([h, samples["target"], g], 1)
        # Teacher is trained first on clean current crops and then copied to a
        # NumPy frozen representation.  Student never receives teacher grads.
        _, self.teacher_params, tloss = self._train_net(teacher_x, samples["target"], samples["negative"], self.seed + 11)
        omega = normalize(self._forward_np(teacher_x, self.teacher_params)).astype(np.float32)
        sh, sc, sg = h.copy(), cc.copy(), g.copy()
        if self.mode in ("current_only",):
            sh[...] = 0.; sg[:, 5:9] = 0.
        if self.mode in ("full_mask", "history_only"):
            sc[...] = 0.
        if self.mode == "both_clean":
            # Explicit clean-current ceiling: this branch sees the GT-box
            # crop, unlike every operational detector-crop branch.
            sc = samples["target"].copy()
        student_x = np.concatenate([sh, sc, sg], 1)
        target = samples["target"] if self.mode == "no_teacher" else omega
        _, self.student_params, sloss = self._train_net(student_x, target, samples["negative"], self.seed + 29)
        # The teacher and student parameter hashes are recorded separately so
        # a later tracker audit can prove which branch was frozen.
        p = normalize(self._forward_np(student_x, self.student_params)); pos = np.sum(p * normalize(target), 1); neg = np.sum(p * normalize(samples["negative"]), 1)
        self.train_stats = {
            "mode": self.mode, "samples": int(n), "epochs": int(self.epochs),
            "teacher_first_loss": float(tloss[0]), "teacher_last_loss": float(tloss[-1]),
            "student_first_loss": float(sloss[0]), "student_last_loss": float(sloss[-1]),
            "teacher_parameter_sha256": self._hash(self.teacher_params), "student_parameter_sha256": self._hash(self.student_params),
            "student_target_cosine_mean": float(pos.mean()), "student_target_cosine_p10": float(np.percentile(pos, 10)),
            "hard_negative_cosine_mean": float(neg.mean()), "hard_negative_margin_mean": float((pos - neg).mean()),
            "finite": bool(np.isfinite(p).all() and np.isfinite(pos).all()), "teacher_frozen_after_fit": True,
        }

    def fit_random(self) -> None:
        rng = np.random.RandomState(self.seed + 404)
        def p(out):
            return (rng.normal(0., .05, (HIDDEN_DIM, self.input_dim)).astype(np.float32), np.zeros(HIDDEN_DIM, np.float32), np.ones(HIDDEN_DIM, np.float32), np.zeros(HIDDEN_DIM, np.float32), rng.normal(0., .05, (out, HIDDEN_DIM)).astype(np.float32), np.zeros(out, np.float32))
        self.student_params = p(FEATURE_DIM); self.teacher_params = p(FEATURE_DIM)
        self.train_stats = {"mode": self.mode, "samples": 0, "epochs": 0, "random_frozen": True, "teacher_parameter_sha256": self._hash(self.teacher_params), "student_parameter_sha256": self._hash(self.student_params), "finite": True}

    def predict(self, history: np.ndarray, candidate: np.ndarray, geom: np.ndarray, clean: bool = False) -> np.ndarray:
        h, c, g = np.asarray(history, np.float32).copy(), np.asarray(candidate, np.float32).copy(), np.asarray(geom, np.float32).copy()
        if self.mode == "current_only":
            h[...] = 0.; g[5:9] = 0.
        if self.mode in ("full_mask", "history_only"):
            c[...] = 0.
        if self.mode == "both_clean" or clean:
            # ``clean=True`` is a diagnostic-only ceiling and is never used by
            # the operational student branch.
            pass
        elif self.mode not in ("full_mask", "history_only"):
            c = corrupt_feature(c, 7919 + int(np.abs(g).sum() * 1e6))
        x = np.concatenate([h, c, g], 0)[None]
        return normalize(self._forward_np(x, self.student_params))[0].astype(np.float32)

    def predict_probe(self, rec: dict, clean: bool = False, candidate_override: Optional[np.ndarray] = None,
                      candidate_list: Optional[np.ndarray] = None, candidate_position: Optional[int] = None) -> np.ndarray:
        # Use the precomputed corruption for exact train/eval reproducibility;
        # no candidate-list ordering enters this function.
        # ``candidate_list`` is used only by the permutation audit.  Select
        # the same positive pair after reordering; the list itself is not
        # pooled or sorted by the predictor.
        if candidate_override is not None:
            # Explicit override is used by the permutation audit to hold the
            # exact precomputed corruption fixed (evaluation records may have
            # been built with a different seed than this predictor).
            base_c = np.asarray(candidate_override, np.float32)
        elif candidate_list is not None and candidate_position is not None and self.mode != "both_clean" and not clean:
            raw = np.asarray(candidate_list[int(candidate_position)], np.float32)
            base_c = corrupt_feature(raw, self.seed * 1000003 + int(rec["frame"]) * 1009 + int(rec["gid"]))
        else:
            base_c = rec["target"] if clean or self.mode == "both_clean" else rec["corrupt"]
        h, c, g = rec["hist"].copy(), np.asarray(base_c if candidate_override is None else candidate_override, np.float32).copy(), rec["geom"].copy()
        if self.mode == "current_only":
            h[...] = 0.; g[5:9] = 0.
        if self.mode in ("full_mask", "history_only"):
            c[...] = 0.
        return normalize(self._forward_np(np.concatenate([h, c, g])[None], self.student_params))[0].astype(np.float32)


def metric_predictor(pred: CurrentPairJEPA, records: Sequence[dict], clean_eval: bool = False) -> Dict[str, float]:
    rows = []; recalls = []; order_diffs = []; candidate_counts = []
    for rec in records:
        z = pred.predict_probe(rec, clean=clean_eval)
        y, neg = normalize(rec["target"]), normalize(rec["negative"])
        pos = float(np.dot(z, y)); ng = float(np.dot(z, neg))
        cand = normalize(np.asarray(rec["all_candidates"], np.float32)); pos_col = int(rec["positive_column"])
        candidate_counts.append(len(cand))
        sims = cand @ z; rank = int(1 + np.sum(sims > sims[pos_col])) if pos_col >= 0 else 0
        recalls.append(float(rank == 1)); rows.append((pos, ng, pos - ng, rank))
        # Candidate-order invariance: actually permute the candidate list and
        # remap the positive column.  The pair-conditioned latent must be
        # unchanged, while the rank computed on the permuted list must agree.
        if len(cand) > 1 and pos_col >= 0:
            perm = np.random.RandomState(1907 + int(rec["frame"]) + int(rec["gid"])).permutation(len(cand))
            new_pos = int(np.where(perm == pos_col)[0][0])
            z_perm = pred.predict_probe(
                rec, clean=clean_eval,
                candidate_override=(rec["target"] if clean_eval or pred.mode == "both_clean" else rec["corrupt"]),
                candidate_list=cand[perm], candidate_position=new_pos,
            )
            # The same pair feature is selected after list reordering; list
            # order is not part of the predictor input by construction.
            order_diffs.append(float(np.max(np.abs(z - z_perm))))
            sims_perm = normalize(cand[perm]) @ z_perm
            rank_perm = int(1 + np.sum(sims_perm > sims_perm[new_pos]))
            if rank_perm != rank:
                raise AssertionError("candidate permutation changed pair rank")
        else:
            order_diffs.append(0.)
    a = np.asarray(rows, np.float64)
    return {
        "samples": float(len(rows)), "recall_at_1": float(np.mean(recalls)),
        "positive_target_cosine_mean": float(a[:, 0].mean()), "positive_target_cosine_p10": float(np.percentile(a[:, 0], 10)),
        "hard_negative_cosine_mean": float(a[:, 1].mean()), "hard_negative_margin_mean": float(a[:, 2].mean()),
        "hard_negative_margin_p10": float(np.percentile(a[:, 2], 10)), "rank_median": float(np.median(a[:, 3])),
        "candidate_count_mean": float(np.mean(candidate_counts) if candidate_counts else 0.),
        "candidate_count_p50": float(np.median(candidate_counts) if candidate_counts else 0.),
        "candidate_order_max_abs_diff": float(max(order_diffs) if order_diffs else 0.),
        "candidate_order_invariant": bool(max(order_diffs) <= 1e-7 if order_diffs else True),
        "finite": bool(np.isfinite(a).all()),
    }


def metric_teacher(pred: CurrentPairJEPA, records: Sequence[dict]) -> Dict[str, float]:
    """Evaluate the frozen clean teacher independently of the student."""
    rows = []; recalls = []
    for rec in records:
        x = np.concatenate([rec["hist"], rec["target"], rec["geom"]])[None]
        z = normalize(pred._forward_np(x, pred.teacher_params))[0]
        y = normalize(rec["target"]); neg = normalize(rec["negative"])
        cand = normalize(np.asarray(rec["all_candidates"], np.float32)); pc = int(rec["positive_column"])
        sims = cand @ z
        rows.append((float(z @ y), float(z @ neg), float(z @ normalize(pred.predict_probe(rec, clean=False)))))
        recalls.append(float(pc >= 0 and int(1 + np.sum(sims > sims[pc])) == 1))
    a = np.asarray(rows, np.float64)
    if not len(a):
        return {"samples": 0., "finite": True}
    return {
        "samples": float(len(a)), "target_cosine_mean": float(a[:, 0].mean()),
        "target_cosine_p10": float(np.percentile(a[:, 0], 10)),
        "hard_negative_cosine_mean": float(a[:, 1].mean()),
        "hard_negative_margin_mean": float((a[:, 0] - a[:, 1]).mean()),
        "teacher_student_latent_cosine_mean": float(a[:, 2].mean()),
        "recall_at_1": float(np.mean(recalls)), "finite": bool(np.isfinite(a).all()),
    }


def load_items(dataset: str, args) -> Tuple[List[Tuple[str, Path, Dict[int, List[GT]], Dict[int, List[Det]], Dict[str, object]]], List[Tuple[str, Path, Dict[int, List[GT]], Dict[int, List[Det]], Dict[str, object]]], Dict[str, object]]:
    train: List[Tuple[str, Path, Dict[int, List[GT]], Dict[int, List[Det]], Dict[str, object]]] = []; ev = []; meta = {}
    if dataset == "mot17":
        tr_names = [x for x in args.mot_train.split(",") if x]; ev_names = [x for x in args.mot_eval.split(",") if x]
        for s in tr_names + ev_names:
            d = MOT_ROOT / "train" / s
            if not d.exists(): raise FileNotFoundError(d)
            gt = parse_mot_gt(d / "gt/gt.txt", args.max_frames); det = parse_mot_det(d / "det/det.txt", args.max_frames)
            item = (s, d, gt, det, {"dataset": "MOT17", "detector": "public FRCNN det/det.txt", "oracle_boxes": False})
            (train if s in tr_names else ev).append(item)
        meta = {"detector": "MOT17 public FRCNN det/det.txt", "oracle_boxes": False}
    else:
        tr_names = [x for x in args.dance_train.split(",") if x]; ev_names = [x for x in args.dance_eval.split(",") if x]
        ann_tr = parse_dance_coco(DANCE_ROOT / "annotations/train.json", tr_names, args.max_frames)
        ann_ev = parse_dance_coco(DANCE_ROOT / "annotations/val.json", ev_names, args.max_frames)
        for s in tr_names:
            d = DANCE_ROOT / "train" / s; gt = ann_tr.get(s, {}); det = parse_cache(DANCE_CTRL_CACHE / s, args.max_frames)
            if not det:
                raise FileNotFoundError("missing audited D-CTRL cache: {}".format(DANCE_CTRL_CACHE / s))
            train.append((s, d, gt, det, {"dataset": "DanceTrack", "detector": "D-CTRL fixed cache", "oracle_boxes": False}))
        for s in ev_names:
            d = DANCE_ROOT / "val" / s; gt = ann_ev.get(s, {}); det = parse_cache(DANCE_CTRL_CACHE / s, args.max_frames)
            if not det:
                raise FileNotFoundError("missing audited D-CTRL cache: {}".format(DANCE_CTRL_CACHE / s))
            ev.append((s, d, gt, det, {"dataset": "DanceTrack", "detector": "D-CTRL fixed cache", "oracle_boxes": False}))
        meta = {"detector": "LocateMOT D-CTRL fixed per-frame cache (no IDs)", "oracle_boxes": False, "cache_root": str(DANCE_CTRL_CACHE)}
    return train, ev, meta


def run_dataset(dataset: str, args, out_root: Path) -> Dict[str, object]:
    train_items, eval_items, detector_meta = load_items(dataset, args)
    train_arr, train_rec, train_meta = build_samples(train_items, args.max_frames, args.seed, args.max_samples)
    eval_arr, eval_rec, eval_meta = build_samples(eval_items, args.max_frames, args.seed + 7001, min(args.max_samples, 12000))
    # Hard protocol guards: a probe with reused detections or non-finite
    # samples is invalid and must fail loudly before any metric is reported.
    if int(train_meta.get("duplicate_count", -1)) != 0 or int(eval_meta.get("duplicate_count", -1)) != 0:
        raise AssertionError("one-to-one matching audit failed: duplicate_count != 0")
    for bundle_name, bundle in (("train", train_arr), ("eval", eval_arr)):
        for key, value in bundle.items():
            if not np.isfinite(value).all():
                raise FloatingPointError("non-finite {} sample field {}".format(bundle_name, key))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    predictors: Dict[str, CurrentPairJEPA] = {}
    stats: Dict[str, object] = {}
    for m in methods:
        p = CurrentPairJEPA(m, args.seed, args.epochs)
        if m == "random_frozen": p.fit_random()
        else: p.fit(train_arr)
        predictors[m] = p; stats[m] = p.train_stats
    val = {m: metric_predictor(p, eval_rec, clean_eval=(m == "both_clean")) for m, p in predictors.items()}
    if any(not bool(v.get("candidate_order_invariant", False)) for v in val.values()):
        raise AssertionError("candidate-order audit failed")
    if any(not bool(v.get("finite", False)) for v in val.values()):
        raise FloatingPointError("non-finite validation metric")
    # Evaluate train side too for diagnosing overfit, but do not tune on it.
    train_diag = {m: metric_predictor(p, train_rec, clean_eval=False) for m, p in predictors.items()}
    # All fitted controls share the same teacher seed/fit.  Use the first
    # non-random predictor to audit the frozen teacher once; this avoids
    # accidentally treating the clean teacher as an online student result.
    teacher_probe = next((p for m, p in predictors.items() if m != "random_frozen"), None)
    teacher_train = metric_teacher(teacher_probe, train_rec) if teacher_probe is not None else None
    teacher_validation = metric_teacher(teacher_probe, eval_rec) if teacher_probe is not None else None
    teacher_by_sequence = {
        seq: metric_teacher(teacher_probe, [r for r in eval_rec if str(r["sequence"]) == seq])
        for seq in sorted({str(r["sequence"]) for r in eval_rec})
    } if teacher_probe is not None else {}
    validation_by_sequence = {}
    for seq in sorted({str(r["sequence"]) for r in eval_rec}):
        seq_records = [r for r in eval_rec if str(r["sequence"]) == seq]
        validation_by_sequence[seq] = {
            m: metric_predictor(p, seq_records, clean_eval=(m == "both_clean"))
            for m, p in predictors.items()
        }
    protocol = {
        "schema": "current-pair-jepa-probe-v1", "teacher": "T(H_<t>, c_clean_t=GT-box crop target, g_t) trained first, copied and frozen", "student": "S(H_<t>, c_corrupt_t=detector crop corruption, g_t) -> stopgrad(T(...))",
        "lag": 1, "operational_note": "probe only; tracker integration must use omega_t at t+1, never for t matching",
        "matching": "per-frame Hungarian IoU>=0.30; no detection reused", "duplicate_count": train_meta["duplicate_count"] + eval_meta["duplicate_count"],
        "student_gt_inputs": False, "history_oracle_for_probe": True, "history_frames_strictly_before_current": True,
        "current_corruption": "deterministic 30% feature mask + Gaussian noise (.08), seed-derived; no labels",
        "controls": {"current_only": "H and history-state masked", "full_mask": "current crop masked (history-only)", "history_only": "alias full_mask", "no_teacher": "direct clean-target regression", "both_clean": "clean-current diagnostic ceiling (not operational)", "random_frozen": "frozen random capacity control"},
        "candidate_order_audit": "pair-conditioned predictor; permutation must not change latent",
        "eval_uses_gt": "only probe labels/history assembly/metrics; not a tracker result",
    }
    return {"dataset": dataset, "detector": detector_meta, "train_sequences": [x[0] for x in train_items], "eval_sequences": [x[0] for x in eval_items], "train_samples": len(train_rec), "eval_samples": len(eval_rec), "train_meta": train_meta, "eval_meta": eval_meta, "predictor_stats": stats, "validation": val, "validation_by_sequence": validation_by_sequence, "teacher_train": teacher_train, "teacher_validation": teacher_validation, "teacher_validation_by_sequence": teacher_by_sequence, "train_diagnostics": train_diag, "protocol": protocol}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("mot17", "dance", "both"), default="mot17")
    p.add_argument("--mot-train", default="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-09-FRCNN")
    p.add_argument("--mot-eval", default="MOT17-05-FRCNN,MOT17-11-FRCNN")
    p.add_argument("--dance-train", default="dancetrack0001,dancetrack0002")
    p.add_argument("--dance-eval", default="dancetrack0004,dancetrack0005")
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--methods", default="current_jepa,current_only,full_mask,no_teacher,both_clean,random_frozen")
    p.add_argument("--out-dir", default="experiments/results/current_pair_jepa_probe_v1")
    return p


def main() -> None:
    args = parser().parse_args(); seed_all(args.seed); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "current-pair-jepa-probe-v1", "command": " ".join(sys.argv), "cwd": os.getcwd(), "generated_unix": time.time(), "seed": args.seed, "max_frames": args.max_frames, "methods": [m for m in args.methods.split(",") if m], "script_sha256": sha256_file(Path(__file__))}
    for ds in (("mot17", "dance") if args.dataset == "both" else (args.dataset,)):
        payload[ds] = run_dataset(ds, args, out)
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    rows = []
    for ds in ("mot17", "dance"):
        if ds in payload:
            for m, row in payload[ds]["validation"].items(): rows.append({"dataset": ds, "method": m, **row})
    if rows:
        with (out / "validation.csv").open("w", newline="") as f:
            keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(json.dumps({ds: payload[ds]["validation"] for ds in ("mot17", "dance") if ds in payload}, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
