#!/usr/bin/env python3
"""Direct MOT-prototype JEPA adapter on a frozen GOT-JEPA encoder.

This is an isolated, bounded experiment.  The GOT-JEPA/DINO head map is
frozen and pooled to a 256-D detector appearance vector.  A train-only clean
teacher target is the next-frame embedding of the same GT-matched identity;
the student sees only native OC-SORT history, the current detector crop and
observable track geometry (with MOT-style masking/noise during training).
At inference the predicted prototype is consumed strictly one frame later
and only opens an OC-SORT native continuation when its IoU/appearance margin
is ambiguous or a short gap is present.  Detector boxes and row counts are
never changed.  Evaluation GT is not loaded by the tracker; it is reserved
for the separate TrackEval audit after the quality gate passes.

The direct prototype target is intentional: the preceding ToMP-filter smoke
showed that the serialized single-object filter interface is poorly matched
to a multi-object candidate set, while the frozen GOT-JEPA DINO appearance
space itself is highly discriminative.  This module therefore tests the
MOT-specific JEPA placement (per-track association prototype prediction)
without changing the production tracker or the official checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse only read-only parsing/OC-SORT and the frozen feature loader.  The
# v2 module is not used for its ToMP-filter target; importing its utilities
# keeps detector-row matching exactly identical to the audited baseline.
import mot_jepa_adapter_v2_clean_teacher as v2lib  # noqa: E402
import mot_ocsort_jepa_official as official_api  # noqa: E402

base = v2lib.base

CHECKPOINT_DEFAULT = "/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar"
MOT_ROOT = base.MOT_ROOT
FIT_SEQUENCES = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN")
EVAL_SEQUENCES = ("MOT17-05-FRCNN", "MOT17-11-FRCNN")
MAX_FRAMES_DEFAULT = 20
HISTORY_K = 4
FEATURE_DIM = 256
GEOM_DIM = 10
TRAIN_SEED = 37


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def norm(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, np.float32)
    return a / max(float(np.linalg.norm(a)), 1e-8)


def pooled_map(m: np.ndarray) -> np.ndarray:
    return norm(np.asarray(m, np.float32).mean(axis=(-1, -2)))


def feature_vec(v: np.ndarray) -> np.ndarray:
    """Convert either a frozen DINO map or an already pooled vector to 256-D.

    Native OC-SORT history stores the full ``[C,H,W]`` map, whereas the
    predictor input is deliberately a compact per-track prototype.  Keeping
    this conversion at the boundary prevents accidental map-shaped tensors
    from leaking into the MLP (and makes cached/serialized variants robust).
    """
    a = np.asarray(v, np.float32)
    if a.ndim >= 3:
        return pooled_map(a)
    a = a.reshape(-1)
    if a.size != FEATURE_DIM:
        raise ValueError(f"expected {FEATURE_DIM}-D feature, got shape {np.asarray(v).shape}")
    return norm(a)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(base.iou(a, b))


def geometry(box: np.ndarray, width: float, height: float, velocity: np.ndarray,
             age: int, gap: int, score: float, missing: float) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(1., x2 - x1), max(1., y2 - y1)
    return np.asarray([(x1 + x2) / (2 * max(width, 1.)),
                       (y1 + y2) / (2 * max(height, 1.)),
                       bw / max(width, 1.), bh / max(height, 1.),
                       float(velocity[0]) / max(width, 1.),
                       float(velocity[1]) / max(height, 1.),
                       age / 30., gap / 30., score, missing], np.float32)


def feature_input(history: Sequence[Tuple[int, np.ndarray, np.ndarray, float]],
                  current_vec: np.ndarray, box: np.ndarray, width: float,
                  height: float, frame: int, score: float,
                  missing: float = 0.) -> np.ndarray:
    """Observable K-history + current prototype + MOT state token."""
    hist = list(history)[-HISTORY_K:]
    h = np.zeros((HISTORY_K, FEATURE_DIM), np.float32)
    if hist:
        h[-len(hist):] = np.stack([feature_vec(x[1]) for x in hist])
    if hist:
        old = hist[-1][2]
        old_c = np.asarray([(old[0] + old[2]) / 2., (old[1] + old[3]) / 2.])
        new_c = np.asarray([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
        velocity = new_c - old_c
        age = len(hist)
        gap = max(1, int(frame) - int(hist[-1][0]))
    else:
        velocity = np.zeros(2, np.float32); age = 0; gap = 1
    g = geometry(box, width, height, velocity, age, gap, score, missing)
    return np.concatenate([h.reshape(-1), feature_vec(current_vec), g]).astype(np.float32)


@dataclass
class DirectSample:
    seq: str
    frame: int
    gid: int
    x: np.ndarray
    teacher_context: np.ndarray
    current_vec: np.ndarray
    target_vec: np.ndarray
    history_vec: np.ndarray
    candidate_vecs: np.ndarray
    positive_index: int
    native_iou_margin: float
    native_id_changed: bool
    history_gap: int
    history_empty: bool
    hard: bool
    hard_reason: Tuple[str, ...]
    visible_next: float = 1.0


def _native_margin(row: dict, cand_rows: Sequence[dict]) -> Tuple[float, bool]:
    """Margin of the native-ID candidate over the best alternate by IoU."""
    same = [j for j, r in enumerate(cand_rows)
            if int(r.get("native_id", -1)) == int(row.get("native_id", -2))]
    if not same:
        return -1.0, False
    j = max(same, key=lambda q: box_iou(row["box"], cand_rows[q]["box"]))
    q = box_iou(row["box"], cand_rows[j]["box"])
    alt = [box_iou(row["box"], r["box"]) for k, r in enumerate(cand_rows)
           if k != j]
    return float(q - (max(alt) if alt else 0.0)), True


def build_direct_samples(prepared: Sequence[v2lib.PreparedSequence],
                         max_candidate_maps: bool = True) -> Tuple[List[DirectSample], Dict[str, object]]:
    """Build causal train/quality windows from fit videos only.

    GT is used solely to select a positive identity and to construct the
    clean teacher history/next-frame target.  ``x`` is built from native
    OC-SORT memory and detector geometry, so no GT field reaches the student.
    """
    samples: List[DirectSample] = []
    stats: Dict[str, object] = {"sequences": {}, "count": 0,
                                "history_future_violations": 0,
                                "teacher_history_future_violations": 0,
                                "duplicate_raw_row_labels": 0,
                                "empty_native_history": 0,
                                "empty_teacher_history": 0,
                                "hard_count": 0, "hard_reasons": {}}
    for p in prepared:
        raw_map, raw_stats = v2lib.assign_raw_rows_one_to_one(p.raw, p.det)
        stats["duplicate_raw_row_labels"] += int(raw_stats.get("duplicate_raw_assignments", 0))
        native_histories: Dict[int, List[Tuple[int, np.ndarray, np.ndarray, float]]] = {}
        gt_histories: Dict[int, List[Tuple[int, np.ndarray, np.ndarray, float]]] = {}
        seq_count = 0
        nframes = max(v2lib.max_frames_for(p), max(p.det) if p.det else 0)
        for fr in range(1, nframes + 1):
            current_pairs = p.gt_by_det.get(fr, {})
            next_pairs = p.gt_by_det.get(fr + 1, {})
            rows = p.raw.get(fr, [])
            for dj, gid in sorted(current_pairs.items()):
                if int(gid) not in p.gt_by_gid or fr + 1 not in p.gt_by_gid[int(gid)]:
                    continue
                nj = p.gt_by_gid[int(gid)][fr + 1]
                row = raw_map.get(fr, {}).get(int(dj))
                next_row = raw_map.get(fr + 1, {}).get(int(nj))
                if row is None or next_row is None:
                    continue
                cand_rows = p.raw.get(fr + 1, [])
                if not cand_rows:
                    continue
                pos_idx = max(range(len(cand_rows)), key=lambda q: box_iou(cand_rows[q]["box"], next_row["box"]))
                if box_iou(cand_rows[pos_idx]["box"], next_row["box"]) < .30:
                    continue
                native = int(row.get("native_id", -1))
                nhist = native_histories.get(native, [])
                ghist = gt_histories.get(int(gid), [])
                if nhist and max(x[0] for x in nhist) >= fr:
                    stats["history_future_violations"] += 1
                if ghist and max(x[0] for x in ghist) >= fr:
                    stats["teacher_history_future_violations"] += 1
                if not nhist:
                    stats["empty_native_history"] += 1
                if not ghist:
                    stats["empty_teacher_history"] += 1
                current_vec = pooled_map(row["feat_map"])
                target_vec = pooled_map(next_row["feat_map"])
                history_vec = pooled_map(nhist[-1][1]) if nhist else current_vec.copy()
                teacher_context = (feature_vec(np.mean(np.stack([x[1] for x in ghist[-HISTORY_K:]]), axis=0))
                                   if ghist else np.zeros(FEATURE_DIM, np.float32))
                x = feature_input(nhist, current_vec, row["box"], p.width, p.height,
                                  fr, float(row.get("score", 1.)), 0.)
                margin, has_native = _native_margin(row, cand_rows)
                next_native = int(next_row.get("native_id", -2))
                changed = native != next_native
                gap = max(1, fr - int(nhist[-1][0])) if nhist else 999
                reasons: List[str] = []
                if margin < .10: reasons.append("native_iou_margin_low")
                if gap > 1: reasons.append("gap_gt1")
                if not nhist: reasons.append("history_empty")
                if changed or not has_native: reasons.append("native_id_changed_or_missing")
                hard = bool(reasons)
                cand = np.stack([pooled_map(r["feat_map"]) for r in cand_rows]).astype(np.float32)
                samples.append(DirectSample(
                    p.name, fr, int(gid), x, teacher_context, current_vec, target_vec,
                    history_vec, cand, int(pos_idx), float(margin), bool(changed),
                    int(gap), not bool(nhist), hard, tuple(reasons), 1.0))
                seq_count += 1
                if hard:
                    stats["hard_count"] += 1
                    for reason in reasons:
                        stats["hard_reasons"][reason] = int(stats["hard_reasons"].get(reason, 0)) + 1
            # Update both memories only after all frame-t samples are created.
            for dj, gid in sorted(current_pairs.items()):
                rr = raw_map.get(fr, {}).get(int(dj))
                if rr is None:
                    continue
                gt_histories.setdefault(int(gid), []).append(
                    (fr, np.asarray(rr["feat_map"], np.float32), rr["box"].copy(), float(rr.get("score", 1.))))
                gt_histories[int(gid)] = gt_histories[int(gid)][-HISTORY_K:]
            for rr in rows:
                native_id = int(rr.get("native_id", -1))
                native_histories.setdefault(native_id, []).append(
                    (fr, np.asarray(rr["feat_map"], np.float32), rr["box"].copy(), float(rr.get("score", 1.))))
                native_histories[native_id] = native_histories[native_id][-HISTORY_K:]
        stats["sequences"][p.name] = {"frames": int(nframes), "samples": int(seq_count)}
    stats["count"] = len(samples)
    return samples, stats


def make_masked_training_samples(samples: Sequence[DirectSample], seed: int) -> Tuple[List[DirectSample], Dict[str, int]]:
    """Add deterministic occlusion views for the visibility head.

    These copies retain the same stop-gradient next-frame target but set the
    supervision visibility to zero and mask current/history features.  They
    are synthetic training augmentations, never quality/eval rows.
    """
    rng = np.random.RandomState(seed + 1907)
    out = list(samples); added = 0
    for i, s in enumerate(samples):
        if i % 4 != 0:
            continue
        xm = s.x.copy()
        for k in range(HISTORY_K):
            if k % 2 == 0:
                xm[k * FEATURE_DIM:(k + 1) * FEATURE_DIM] = 0.
        xm[HISTORY_K * FEATURE_DIM:HISTORY_K * FEATURE_DIM + FEATURE_DIM] = 0.
        xm[-1] = 1.0
        # Copy all labels/diagnostics, changing only the observable view and
        # visibility target.  A tiny deterministic geometry jitter models a
        # detector box perturbation without consulting GT.
        xm[-GEOM_DIM:-GEOM_DIM + 4] += rng.normal(0., .005, 4).astype(np.float32)
        out.append(DirectSample(
            s.seq, s.frame, s.gid, xm, s.teacher_context.copy(), s.current_vec.copy(),
            s.target_vec.copy(), s.history_vec.copy(), s.candidate_vecs.copy(),
            s.positive_index, s.native_iou_margin, s.native_id_changed,
            s.history_gap, s.history_empty, s.hard, tuple(list(s.hard_reason) + ["synthetic_mask"]), 0.0))
        added += 1
    return out, {"base": len(samples), "synthetic_masked": added, "total": len(out)}


class PrototypeNet:
    """State-conditioned residual prototype predictor with quality heads."""
    def __init__(self, input_dim: int, seed: int, alpha: float = .50):
        import torch
        import torch.nn as nn
        torch.manual_seed(int(seed))
        self.torch = torch; self.input_dim = int(input_dim); self.seed = int(seed); self.alpha = float(alpha)
        class Net(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.trunk = nn.Sequential(nn.Linear(d, 512), nn.LayerNorm(512), nn.GELU())
                self.delta = nn.Linear(512, FEATURE_DIM)
                self.visibility = nn.Linear(512, 1)
                self.uncertainty = nn.Linear(512, 1)
            def forward(self, x):
                z = self.trunk(x)
                return self.delta(z), self.visibility(z).squeeze(-1), self.uncertainty(z).squeeze(-1)
        self.net = Net(self.input_dim)

    def state_dict(self): return self.net.state_dict()

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch
        self.net.eval()
        with torch.no_grad():
            raw, vis, unc = self.net(torch.from_numpy(np.asarray(x, np.float32)))
            cur = torch.from_numpy(np.asarray(x, np.float32))[:, HISTORY_K * FEATURE_DIM:HISTORY_K * FEATURE_DIM + FEATURE_DIM]
            pred = torch.nn.functional.normalize(cur + self.alpha * torch.tanh(raw), dim=-1)
        return pred.cpu().numpy().astype(np.float32), vis.cpu().numpy(), unc.cpu().numpy()


def corrupt_batch(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    z = np.asarray(x, np.float32).copy(); hdim = HISTORY_K * FEATURE_DIM
    for i in range(len(z)):
        for k in range(HISTORY_K):
            if rng.rand() < .22:
                z[i, k * FEATURE_DIM:(k + 1) * FEATURE_DIM] = 0.
        if rng.rand() < .15:
            z[i, hdim:hdim + FEATURE_DIM] = 0.
            z[i, -1] = 1.
        z[i, :hdim + FEATURE_DIM] += rng.normal(0., .018, hdim + FEATURE_DIM).astype(np.float32)
        if rng.rand() < .20:
            z[i, hdim + FEATURE_DIM:hdim + FEATURE_DIM + 4] += rng.normal(0., .004, 4).astype(np.float32)
    return z


def train_model(samples: Sequence[DirectSample], target_mode: str, epochs: int, seed: int) -> Tuple[PrototypeNet, Dict[str, object]]:
    import torch
    torch.set_num_threads(1)
    if not samples:
        raise RuntimeError("empty prototype training samples")
    x = np.stack([s.x for s in samples]).astype(np.float32)
    y = np.stack([s.target_vec for s in samples]).astype(np.float32)
    if target_mode == "shuffle":
        y = y[np.random.RandomState(seed + 999).permutation(len(y))]
    elif target_mode != "teacher":
        raise ValueError(target_mode)
    pos = np.stack([s.target_vec for s in samples]).astype(np.float32)
    maxn = max([len(s.candidate_vecs) - 1 for s in samples] + [1])
    neg = np.zeros((len(samples), maxn, FEATURE_DIM), np.float32)
    nmask = np.zeros((len(samples), maxn), np.float32)
    for i, s in enumerate(samples):
        others = np.delete(s.candidate_vecs, s.positive_index, axis=0)
        if len(others):
            neg[i, :len(others)] = others; nmask[i, :len(others)] = 1.
    vis = np.asarray([s.visible_next for s in samples], np.float32)
    tx, ty, tp, tn, tm, tv = map(torch.from_numpy, (x, y, pos, neg, nmask, vis))
    model = PrototypeNet(x.shape[1], seed).net.train()
    # Keep a wrapper-like object for prediction/checkpoint handling.
    wrapper = PrototypeNet.__new__(PrototypeNet)
    wrapper.torch = torch; wrapper.input_dim = x.shape[1]; wrapper.seed = seed; wrapper.alpha = .50; wrapper.net = model
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    rng = np.random.RandomState(seed); losses = []; bs = min(128, len(x))
    for _ep in range(int(epochs)):
        order = rng.permutation(len(x)); total = 0.
        for st in range(0, len(x), bs):
            ix = order[st:st + bs]
            xb = torch.from_numpy(corrupt_batch(x[ix], rng))
            raw, vlog, ulog = model(xb)
            cur = xb[:, HISTORY_K * FEATURE_DIM:HISTORY_K * FEATURE_DIM + FEATURE_DIM]
            pred = torch.nn.functional.normalize(cur + wrapper.alpha * torch.tanh(raw), dim=-1)
            yt = torch.nn.functional.normalize(ty[ix], dim=-1)
            pos_sim = (pred * torch.nn.functional.normalize(tp[ix], dim=-1)).sum(1)
            ng = torch.einsum("bd,bmd->bm", pred, torch.nn.functional.normalize(tn[ix], dim=-1))
            ng = torch.where(tm[ix] > 0, ng, torch.full_like(ng, -1.))
            hard = ng.max(1).values
            cos = (pred * yt).sum(1)
            rank = torch.nn.functional.softplus(.12 - pos_sim + hard)
            err = (1. - cos).clamp_min(0.)
            unc = torch.exp(-ulog.clamp(-5., 5.)) * err + ulog
            vis_loss = torch.nn.functional.binary_cross_entropy_with_logits(vlog, tv[ix])
            loss = (1. - cos).mean() + .65 * rank.mean() + .10 * unc.mean() + .10 * vis_loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
            total += float(loss.detach()) * len(ix)
        losses.append(total / len(x))
    wrapper.net.eval()
    pred, vlog, ulog = wrapper.predict(x)
    cos = np.sum(pred * norm(y), axis=1)
    return wrapper, {
        "target": target_mode, "samples": int(len(x)), "epochs": int(epochs),
        "first_loss": float(losses[0]), "last_loss": float(losses[-1]),
        "train_target_cosine_mean": float(np.mean(cos)),
        "train_target_cosine_p10": float(np.percentile(cos, 10)),
        "train_visibility_mean": float(np.mean(1. / (1. + np.exp(-vlog)))),
        "train_visibility_target_mean": float(np.mean(vis)),
        "train_uncertainty_mean": float(np.mean(ulog)),
        "finite": bool(np.isfinite(pred).all() and np.isfinite(vlog).all() and np.isfinite(ulog).all()),
    }


def save_model(model: PrototypeNet, path: Path, metadata: Dict[str, object]) -> str:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.net.state_dict(), "input_dim": model.input_dim,
                "seed": model.seed, "alpha": model.alpha, "metadata": metadata,
                "schema": "mot-jepa-direct-prototype-v1"}, str(path))
    return sha256(path)


def _candidate_metrics(proto: np.ndarray, s: DirectSample) -> Tuple[float, float, float]:
    scores = np.asarray(s.candidate_vecs, np.float32).dot(norm(proto))
    j = int(s.positive_index)
    if j < 0 or j >= len(scores):
        return 0., 0., 1. / max(1, len(scores))
    neg = np.delete(scores, j)
    mx = float(np.max(neg)) if len(neg) else -np.inf
    return float(float(scores[j]) >= mx), float(scores[j] - mx), 1. / max(1, len(scores))


def evaluate_quality(samples: Sequence[DirectSample], student: PrototypeNet,
                     no_teacher: PrototypeNet, random_model: PrototypeNet,
                     max_per_seq: int = 40, stress: bool = False) -> Dict[str, object]:
    groups: Dict[str, List[DirectSample]] = {}
    for s in samples: groups.setdefault(s.seq, []).append(s)
    chosen: List[DirectSample] = []
    for seq in sorted(groups):
        vals = groups[seq]
        if max_per_seq > 0 and len(vals) > max_per_seq:
            idx = np.linspace(0, len(vals) - 1, max_per_seq, dtype=int)
            vals = [vals[int(i)] for i in idx]
        chosen.extend(vals)

    def score_subset(vals: Sequence[DirectSample], mode: str) -> Dict[str, object]:
        hit = []; margin = []; chance = []; tc = []; vis = []; unc = []
        rng = np.random.RandomState(777)
        for s in vals:
            x = s.x.copy()
            if stress:
                x = corrupt_batch(x[None], rng)[0]
            if mode == "teacher": proto = s.target_vec
            elif mode == "raw_current": proto = s.current_vec
            elif mode == "history": proto = s.history_vec
            else:
                mod = {"student": student, "no_teacher": no_teacher, "random_frozen": random_model}[mode]
                pp, vv, uu = mod.predict(x[None]); proto = pp[0]
                vis.append(float(1. / (1. + np.exp(-vv[0])))); unc.append(float(uu[0])); tc.append(float(np.dot(norm(proto), s.target_vec)))
            h, m, c = _candidate_metrics(proto, s); hit.append(h); margin.append(m); chance.append(c)
        return {"n": int(len(hit)), "recall_at1": float(np.mean(hit)) if hit else None,
                "mean_margin": float(np.mean(margin)) if margin else None,
                "p10_margin": float(np.percentile(margin, 10)) if margin else None,
                "p25_margin": float(np.percentile(margin, 25)) if margin else None,
                "chance_mean": float(np.mean(chance)) if chance else None,
                "delta_chance": float(np.mean(hit) - np.mean(chance)) if hit else None,
                "positive_margin_fraction": float(np.mean(np.asarray(margin) > 0)) if margin else None,
                "teacher_cosine_mean": float(np.mean(tc)) if tc else None,
                "visibility_mean": float(np.mean(vis)) if vis else None,
                "uncertainty_mean": float(np.mean(unc)) if unc else None}

    methods = ("teacher", "raw_current", "history", "student", "no_teacher", "random_frozen")
    rec: Dict[str, object] = {"samples": int(len(chosen)), "stress": bool(stress), "label_scope": "fit-sequence temporal suffix only"}
    for mode in methods:
        rec[mode] = score_subset(chosen, mode)
    hard = [s for s in chosen if s.hard]
    rec["hard_definition"] = "native_iou_margin<0.10 OR gap>1 OR native-id change/missing OR empty history (observable rule)"
    rec["hard_samples"] = int(len(hard))
    for mode in methods:
        rec["hard_" + mode] = score_subset(hard, mode)
    t = rec["teacher"]; st = rec["student"]; hst = rec["hard_student"]; hnt = rec["hard_no_teacher"]
    gate = {
        "pre_registered": {
            "teacher_recall_at1_ge": .80, "teacher_margin_gt": 0.,
            "student_recall_delta_chance_ge": .10, "student_margin_gt": 0.,
            "student_teacher_cosine_ge": .50, "hard_student_delta_chance_ge": .05,
            "hard_student_margin_gt": 0., "hard_n_min": 10,
        },
        "teacher_status": "pass" if t["recall_at1"] is not None and t["recall_at1"] >= .80 and t["mean_margin"] > 0 else "fail",
        "student_status": "pass" if st["recall_at1"] is not None and st["delta_chance"] >= .10 and st["mean_margin"] > 0 and (st.get("teacher_cosine_mean") or -1) >= .50 else "fail",
        "hard_status": "pass" if len(hard) >= 10 and hst["delta_chance"] >= .05 and hst["mean_margin"] > 0 else "fail",
    }
    gate["status"] = "pass" if all(gate[k] == "pass" for k in ("teacher_status", "student_status", "hard_status")) else "fail"
    gate["teacher_quality_fail"] = gate["teacher_status"] != "pass"
    gate["student_quality_fail"] = gate["student_status"] != "pass"
    gate["hard_quality_fail"] = gate["hard_status"] != "pass"
    gate["integration_decision"] = "GO_CANDIDATE" if gate["status"] == "pass" else "NO-GO"
    # A no-teacher control is retained for attribution.  It is not used to
    # relax the pre-registered gate, but its hard delta is surfaced explicitly.
    gate["hard_student_minus_no_teacher_margin"] = (hst["mean_margin"] - hnt["mean_margin"]
                                                     if hst["mean_margin"] is not None and hnt["mean_margin"] is not None else None)
    rec["gate"] = gate
    return rec


@dataclass
class OnlineState:
    cid: int
    native_id: int
    last_frame: int
    box: np.ndarray
    score: float
    history: List[Tuple[int, np.ndarray, np.ndarray, float]] = field(default_factory=list)
    prototype: Optional[np.ndarray] = None
    visibility: float = 1.0
    uncertainty: float = 0.0


def _margin(values: np.ndarray, j: int) -> float:
    if len(values) <= 1: return float("inf")
    return float(values[j] - np.max(np.delete(values, j)))


def online_track(p: v2lib.PreparedSequence, models: Dict[str, Optional[PrototypeNet]],
                 thresholds: Dict[str, float], max_age: int = 30) -> Tuple[Dict[str, Dict[int, List[Tuple[np.ndarray, int, float]]]], Dict[str, dict]]:
    """Causal native-OC-SORT continuation with lagged prototype repair."""
    outputs: Dict[str, Dict[int, List[Tuple[np.ndarray, int, float]]]] = {}
    diags: Dict[str, dict] = {}
    nframes = v2lib.max_frames_for(p)
    for mode, model in models.items():
        if mode == "raw":
            outputs[mode] = {fr: [(r["box"], int(r["native_id"]), float(r["score"])) for r in rs]
                             for fr, rs in p.raw.items()}
            diags[mode] = {"mode": "raw", "eval_gt_read_by_tracker": False, "lag": "t->t+1"}
            continue
        states: List[OnlineState] = []; next_fresh = 1_000_000; out: Dict[int, List[Tuple[np.ndarray, int, float]]] = {}
        d = {"mode": mode, "lag": "t->t+1", "eval_gt_read_by_tracker": False,
             "native_continuations": 0, "ambiguous_opened": 0, "gap_candidates": 0,
             "accepted_repairs": 0, "accepted_ambiguous": 0, "gate_rejects": 0,
             "filter_updates": 0, "candidate_pairs": 0}
        for fr in range(1, nframes + 1):
            cur = p.raw.get(fr, []); active = [s for s in states if fr - s.last_frame <= max_age]
            used_s: set = set(); used_r: set = set(); matches: List[Tuple[int, int, str]] = []
            cvec = np.stack([pooled_map(r["feat_map"]) for r in cur]).astype(np.float32) if cur else np.empty((0, FEATURE_DIM), np.float32)
            iou_cache = {i: np.asarray([box_iou(s.box, r["box"]) for r in cur], np.float32) for i, s in enumerate(active)}
            app_cache: Dict[int, np.ndarray] = {}
            for i, s in enumerate(active):
                if s.prototype is None or not len(cur):
                    app_cache[i] = np.zeros(len(cur), np.float32)
                else:
                    app_cache[i] = cvec.dot(norm(s.prototype))

            ambiguous: set = set(); native_opts: List[Tuple[float, int, int]] = []
            amb_thr = float(thresholds.get("ambiguous_fused_margin", .10))
            for i, s in enumerate(active):
                same = [j for j, r in enumerate(cur)
                        if int(r["native_id"]) == int(s.native_id) and iou_cache[i][j] >= .20]
                if not same: continue
                j = max(same, key=lambda q: float(iou_cache[i][q]))
                fused = .45 * iou_cache[i] + .55 * ((app_cache[i] + 1.) / 2.)
                mar = _margin(fused, j)
                gap = fr - s.last_frame
                if mar < amb_thr and s.prototype is not None and len(cur) > 1:
                    if float(s.visibility) >= float(thresholds.get("visibility_min", .25)) and abs(float(s.uncertainty)) <= float(thresholds.get("uncertainty_max", 1.0)):
                        ambiguous.add(i); d["ambiguous_opened"] += 1
                    else:
                        d["gate_rejects"] += 1
                if i not in ambiguous and gap <= 1:
                    native_opts.append((float(iou_cache[i][j]), i, j))
                elif gap > 1:
                    d["gap_candidates"] += 1
            for _q, i, j in sorted(native_opts, reverse=True):
                if i in used_s or j in used_r: continue
                matches.append((i, j, "native")); used_s.add(i); used_r.add(j); d["native_continuations"] += 1

            rem_s: List[int] = []
            for i, s in enumerate(active):
                if i in used_s or s.prototype is None: continue
                gap = fr - s.last_frame
                if i in ambiguous or 2 <= gap <= 8:
                    rem_s.append(i)
            rem_r = [j for j in range(len(cur)) if j not in used_r]
            d["candidate_pairs"] += len(rem_s) * len(rem_r)
            if rem_s and rem_r:
                fused = np.full((len(rem_s), len(rem_r)), -np.inf, np.float32)
                for a, i in enumerate(rem_s):
                    for b, j in enumerate(rem_r):
                        fused[a, b] = .45 * float(iou_cache[i][j]) + .55 * float((app_cache[i][j] + 1.) / 2.)
                rr, cc = linear_sum_assignment(-fused)
                for a, b in zip(rr, cc):
                    a, b = int(a), int(b); i, j = rem_s[a], rem_r[b]
                    q = float(iou_cache[i][j]); ap = float(app_cache[i][j]); row_mar = _margin(fused[a], b); col_mar = _margin(fused[:, b], a)
                    is_amb = i in ambiguous
                    if is_amb:
                        ok = q >= float(thresholds.get("ambiguous_iou_min", .05)) and ap >= float(thresholds.get("ambiguous_app_min", .20)) and row_mar >= 0. and col_mar >= 0.
                    else:
                        ok = (q >= .02 or ap >= .25) and row_mar >= 0. and col_mar >= 0.
                    if ok:
                        matches.append((i, j, "ambiguous" if is_amb else "gap")); used_s.add(i); used_r.add(j); d["accepted_repairs"] += 1; d["accepted_ambiguous"] += int(is_amb)
                    else:
                        d["gate_rejects"] += 1

            updated: List[Tuple[OnlineState, np.ndarray]] = []
            for i, j, _kind in matches:
                s, r = active[i], cur[j]; old = s.box.copy(); box = np.asarray(r["box"], np.float32)
                vel = np.asarray([(box[0] + box[2] - old[0] - old[2]) / 2., (box[1] + box[3] - old[1] - old[3]) / 2.], np.float32)
                vec = pooled_map(r["feat_map"]); s.history = (s.history + [(fr, np.asarray(r["feat_map"], np.float32), box.copy(), float(r["score"]))])[-HISTORY_K:]
                s.box = box.copy(); s.last_frame = fr; s.native_id = int(r["native_id"]); s.score = float(r["score"])
                x = feature_input(s.history[:-1], vec, box, p.width, p.height, fr, s.score, 0.)
                updated.append((s, x)); r["cid_v3"] = s.cid
            active_cids = {int(s.cid) for s in states if fr - s.last_frame <= max_age}
            for j, r in enumerate(cur):
                if j in used_r: continue
                native = int(r["native_id"]); cid = native if native > 0 and native not in active_cids else next_fresh
                while cid in active_cids: next_fresh += 1; cid = next_fresh
                next_fresh = max(next_fresh, cid + 1); vec = pooled_map(r["feat_map"])
                s = OnlineState(cid, native, fr, np.asarray(r["box"], np.float32).copy(), float(r["score"]),
                                [(fr, np.asarray(r["feat_map"], np.float32), np.asarray(r["box"], np.float32).copy(), float(r["score"]))])
                states.append(s); active_cids.add(cid); r["cid_v3"] = cid
                x = feature_input([], vec, r["box"], p.width, p.height, fr, float(r["score"]), 0.); updated.append((s, x))
            for s, x in updated:
                if mode == "history":
                    s.prototype = pooled_map(s.history[-1][1]); s.visibility = 1.; s.uncertainty = 0.
                else:
                    pred, vlog, ulog = model.predict(np.asarray(x, np.float32)[None]); s.prototype = pred[0]; s.visibility = float(1. / (1. + np.exp(-vlog[0]))); s.uncertainty = float(ulog[0])
                d["filter_updates"] += 1
            out[fr] = [(r["box"], int(r.get("cid_v3", r["native_id"])), float(r["score"])) for r in cur]
        outputs[mode] = out; diags[mode] = d
    raw_ids = [int(r["native_id"]) for fr in range(1, nframes + 1) for r in p.raw.get(fr, [])]
    for mode in outputs:
        if mode != "raw":
            ids = [int(t[1]) for fr in range(1, nframes + 1) for t in outputs[mode].get(fr, [])]
            diags[mode]["id_differences_vs_raw"] = int(sum(a != b for a, b in zip(raw_ids, ids)))
    return outputs, diags


def write_tracks(path: Path, rows: Dict[int, List[Tuple[np.ndarray, int, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for fr in sorted(rows):
            for box, tid, score in rows[fr]:
                x1, y1, x2, y2 = map(float, box)
                f.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},-1,-1,-1\n".format(
                    int(fr), int(tid), x1, y1, max(0., x2 - x1), max(0., y2 - y1), float(score)))


def _prepared_manifest(prepared: Sequence[v2lib.PreparedSequence]) -> Dict[str, object]:
    return {p.name: {"frames": int(v2lib.max_frames_for(p)), "det_rows": int(sum(len(v) for v in p.det.values())),
                     "metadata": p.metadata} for p in prepared}


def _copy_sources(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in ("mot_jepa_adapter_v3_direct.py", "mot_jepa_adapter_v2_clean_teacher.py",
                 "mot_ocsort_jepa_official.py", "mot_ocsort_jepa.py", "audit_trackeval_real.py"):
        src = HERE / name
        if src.exists(): shutil.copy2(src, out / name)


def _torch_version() -> Optional[str]:
    try:
        import torch
        return str(torch.__version__)
    except Exception:
        return None


def run_eval(eval_seqs: Sequence[str], extractor: official_api.OfficialLatent,
             cache: Path, models: Dict[str, Optional[PrototypeNet]], thresholds: Dict[str, float],
             out: Path, max_frames: int) -> Dict[str, object]:
    """Run no-GT 20-frame eval only after a passing train quality gate."""
    prepared = [v2lib.prepare_sequence(seq, max_frames, extractor, cache, read_gt=False) for seq in eval_seqs]
    rows_meta = []; eval_block = {"eval_sequences": list(eval_seqs), "tracks": {}, "rows": []}
    for p in prepared:
        outputs, diags = online_track(p, models, thresholds)
        for mode, tracks in outputs.items():
            path = out / "mot17" / "tracks" / mode / (p.name + ".txt"); write_tracks(path, tracks)
            item = {"sequence": p.name, "method": mode, "frames": int(max_frames),
                    "det_rows": int(sum(len(v) for v in p.det.values())),
                    "track_rows": int(sum(len(v) for v in tracks.values())),
                    "track_sha256": sha256(path), "diagnostics": diags.get(mode, {}),
                    "eval_gt_read_by_tracker": False}
            rows_meta.append(item); eval_block["rows"].append(item)
            eval_block["tracks"].setdefault(mode, {})[p.name] = str(path)
    eval_block["prepared_manifest"] = _prepared_manifest(prepared)
    eval_block["eval_gt_read"] = False; eval_block["thresholds"] = thresholds
    return {"eval": eval_block, "rows": rows_meta}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Direct MOT-prototype JEPA adapter")
    ap.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--fit-sequences", default=",".join(FIT_SEQUENCES))
    ap.add_argument("--eval-sequences", default=",".join(EVAL_SEQUENCES))
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES_DEFAULT)
    ap.add_argument("--fit-end-frame", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--teacher-batch", type=int, default=8)
    ap.add_argument("--quality-max-per-seq", type=int, default=40)
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    fit_seqs = tuple(x for x in args.fit_sequences.split(",") if x)
    eval_seqs = tuple(x for x in args.eval_sequences.split(",") if x)
    fit_end = int(args.fit_end_frame)
    cache = Path(args.cache_dir).resolve() if args.cache_dir else out / "cache"
    payload: Dict[str, object] = {
        "schema": "mot-jepa-adapter-v3-direct-prototype-train-v1", "status": "started",
        "dataset": "MOT17", "fit_sequences": list(fit_seqs), "eval_sequences_reserved": list(eval_seqs),
        "quality_sequences": list(fit_seqs), "max_frames": int(args.max_frames), "fit_end_frame": fit_end,
        "fit_sample_t_lt": fit_end, "quality_sample_t_ge": fit_end, "quality_target_t_ge": fit_end + 1,
        "history_k": HISTORY_K, "feature_dim": FEATURE_DIM, "geometry_dim": GEOM_DIM,
        "teacher": "stop-gradient clean GT-gid matched next-frame pooled GOT-JEPA DINO embedding",
        "student": "corrupted native OC-SORT history/current/geometry -> next-frame prototype + visibility/uncertainty",
        "candidate_in_predictor": False, "association_lag": "t->t+1", "eval_gt_read_by_tracker": False,
        "production_tracker_modified": False, "checkpoint": str(Path(args.checkpoint).resolve()),
        "cache_dir": str(cache), "device": args.device, "seed": int(args.seed), "epochs": int(args.epochs),
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
        "torch": _torch_version(), "source_sha256_started": sha256(Path(__file__).resolve()),
    }
    t0 = time.perf_counter()
    try:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        extractor = official_api.OfficialLatent(Path(args.checkpoint), args.device, args.teacher_batch)
        payload["official_load"] = extractor.load_meta
        fit_prepared = [v2lib.prepare_sequence(seq, args.max_frames, extractor, cache, read_gt=True) for seq in fit_seqs]
        all_samples, all_stats = build_direct_samples(fit_prepared)
        if int(all_stats.get("history_future_violations", 0)) or int(all_stats.get("teacher_history_future_violations", 0)):
            raise AssertionError("future history in direct samples")
        fit_samples = [s for s in all_samples if int(s.frame) < fit_end]
        held_samples = [s for s in all_samples if int(s.frame) >= fit_end]
        payload["fit_manifest"] = _prepared_manifest(fit_prepared)
        payload["sample_stats_all_before_split"] = all_stats
        payload["fit_sample_count"] = len(fit_samples); payload["quality_sample_count_before_cap"] = len(held_samples)
        train_aug, aug_stats = make_masked_training_samples(fit_samples, args.seed)
        payload["training_augmentation"] = aug_stats
        student, student_stats = train_model(train_aug, "teacher", args.epochs, args.seed)
        no_teacher, no_teacher_stats = train_model(train_aug, "shuffle", args.epochs, args.seed + 1)
        # A deterministic frozen random network is a control, never a release
        # candidate.  Fit its parameters through the same constructor but do
        # not let it affect any threshold or gate.
        random_model = PrototypeNet(student.input_dim, args.seed + 404)
        payload["fit_student"] = student_stats; payload["fit_no_teacher"] = no_teacher_stats
        ckdir = out / "checkpoints"; ckdir.mkdir(parents=True, exist_ok=True)
        meta = {"schema": "mot-jepa-direct-prototype-v1", "checkpoint_sha256": extractor.checkpoint_sha256,
                "fit_sequences": list(fit_seqs), "fit_end_frame": fit_end, "history_k": HISTORY_K,
                "student_x_gt_fields": 0, "teacher_gt_history_train_only": True,
                "target": "next-frame pooled DINO head embedding", "source_sha256": sha256(Path(__file__).resolve())}
        student_sha = save_model(student, ckdir / "trained_direct_prototype.pt", {**meta, "target_mode": "teacher"})
        no_sha = save_model(no_teacher, ckdir / "no_teacher_direct_prototype.pt", {**meta, "target_mode": "shuffle"})
        payload["checkpoints"] = {"student": str(ckdir / "trained_direct_prototype.pt"), "student_sha256": student_sha,
                                   "no_teacher": str(ckdir / "no_teacher_direct_prototype.pt"), "no_teacher_sha256": no_sha}
        held = v2lib.select_quality_samples(held_samples, int(args.quality_max_per_seq))
        quality = evaluate_quality(held, student, no_teacher, random_model, max_per_seq=0, stress=False)
        quality_stress = evaluate_quality(held, student, no_teacher, random_model, max_per_seq=0, stress=True)
        quality["stress_view"] = quality_stress
        payload["quality"] = quality
        gate = quality.get("gate", {})
        data_checks = {"all_finite": bool(student_stats.get("finite") and no_teacher_stats.get("finite")),
                       "duplicate_raw_row_labels": int(all_stats.get("duplicate_raw_row_labels", 0)),
                       "history_future_violations": int(all_stats.get("history_future_violations", 0)),
                       "teacher_history_future_violations": int(all_stats.get("teacher_history_future_violations", 0)),
                       "student_x_gt_fields": 0, "fit_gt_used_only_for_teacher_labels": True,
                       "eval_gt_used": False, "fit_sequences_only": list(fit_seqs),
                       "eval_sequences_not_loaded_before_gate": True}
        gate = dict(gate); gate["data_checks"] = data_checks
        gate["status"] = "pass" if gate.get("status") == "pass" and data_checks["all_finite"] and data_checks["duplicate_raw_row_labels"] == 0 and data_checks["history_future_violations"] == 0 and data_checks["teacher_history_future_violations"] == 0 else "fail"
        gate["teacher_quality_fail"] = gate.get("teacher_status") != "pass"
        gate["student_quality_fail"] = gate.get("student_status") != "pass"
        gate["hard_quality_fail"] = gate.get("hard_status") != "pass"
        gate["integration_decision"] = "GO_CANDIDATE" if gate.get("status") == "pass" else "NO-GO"
        quality["gate"] = gate; payload["quality_gate"] = gate
        _write_json(out / "quality_gate.json", quality)
        # No eval files/GT are touched unless every pre-registered quality and
        # data check passes.  This branch is the only place that can load the
        # reserved 05/11 detector stream.
        if gate.get("status") == "pass":
            thresholds = {"ambiguous_fused_margin": .10, "ambiguous_iou_min": .05,
                          "ambiguous_app_min": .20, "visibility_min": .25,
                          "uncertainty_max": 1.0, "source": "fixed before eval; quality gate only"}
            eval_payload = run_eval(eval_seqs, extractor, cache,
                                    {"raw": None, "history": None, "trained_adapter": student,
                                     "no_teacher": no_teacher, "random_frozen": random_model},
                                    thresholds, out, args.max_frames)
            payload.update(eval_payload); payload["eval_started_after_gate"] = True
        else:
            payload["eval_started_after_gate"] = False; payload["eval_skipped_reason"] = "quality_gate_fail"
        payload["status"] = "complete"; payload["elapsed_seconds"] = time.perf_counter() - t0
        _write_json(out / "train_results.json", payload)
        _write_json(out / "audit.json", {
            "schema": "mot-jepa-adapter-v3-direct-audit-v1", "protocol": "native_ocsort_lagged_direct_prototype",
            "source_sha256": sha256(Path(__file__).resolve()), "official_checkpoint_sha256": extractor.checkpoint_sha256,
            "fit_sequences": list(fit_seqs), "fit_sample_t_lt": fit_end,
            "quality_sequences": list(fit_seqs), "quality_sample_t_ge": fit_end,
            "eval_sequences_reserved": list(eval_seqs), "teacher_gt_history": True,
            "student_gt_history": False, "student_x_gt_fields": 0,
            "target": "clean GT-gid matched next-frame pooled DINO head embedding (stop-gradient)",
            "online_trigger": "native-vs-second fused margin low or gap 2..8; strict t->t+1",
            "quality_gate": gate, "data_checks": data_checks,
            "eval_started_after_gate": bool(payload.get("eval_started_after_gate")),
            "checkpoint_sha256": {"student": student_sha, "no_teacher": no_sha},
        })
        _copy_sources(out)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        payload.update({"status": "failure", "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - t0})
        _write_json(out / "failure.json", payload); _write_json(out / "train_results.json", payload); _copy_sources(out)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 1


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=float) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
