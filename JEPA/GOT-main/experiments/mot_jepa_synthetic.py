#!/usr/bin/env python3
"""A small, reproducible MOT experiment inspired by GOT-JEPA.

This file deliberately has *no* dependency on torch, torchvision, scipy, or
the large GOT checkpoints.  It can therefore be run with the system Python or
with the project's older AVI environment.  The experiment is not a MOT
benchmark implementation; it is a mechanism smoke test that keeps the
detections fixed while comparing a raw-EMA association baseline with a
context-to-target (JEPA-style) predictor.

The analogy to GOT-JEPA is explicit:

* teacher target: a clean current-object appearance latent (available only
  while fitting the synthetic predictor),
* student context: masked/noisy historical track observations,
* predictor: a linear model that predicts the teacher latent from context,
* occlusion handling: do not update the appearance memory with low-quality
  observations and use the predicted latent while a track is missing.

The linear predictor is intentionally small.  It is a stand-in for a learned
MLP/transformer predictor, not a claim that a linear model is the final MOT
architecture.  See ``experiments/README_MOT_JEPA.md`` for the proposed
integration points in the actual ToMP/GOT-JEPA code.
"""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


def normalize(x: Array, axis: int = -1, eps: float = 1e-8) -> Array:
    """L2-normalize an array without producing NaNs for zero vectors."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def cosine_distance(a: Array, b: Array) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(1.0 - np.dot(a, b) / den)


@dataclass
class Detection:
    """Synthetic detector output.  ``gt_id == -1`` denotes a false positive."""

    center: Array
    size: Array
    embedding: Array
    score: float
    gt_id: int


@dataclass
class SequenceData:
    positions: Array                 # [T, N, 2]
    sizes: Array                     # [T, N, 2]
    visible: Array                   # [T, N] bool (ground-truth visibility)
    clean_embeddings: Array          # [T, N, D], teacher-only branch
    detections: List[List[Detection]]
    occluded: Array                  # [T, N] bool, generated occlusion mask


def _scenario_config(name: str) -> Dict[str, float]:
    """Return detector/occlusion knobs for a named synthetic condition."""
    if name == "mild":
        return {
            "random_occ": 0.12,
            "occ_det_prob": 0.70,
            "occ_mix": 0.50,
            "visible_noise": 0.08,
            "occ_noise": 0.28,
            "false_positive_rate": 0.25,
            "score_visible": 0.92,
            "score_occ": 0.34,
        }
    if name == "hard":
        return {
            "random_occ": 0.30,
            "occ_det_prob": 0.62,
            "occ_mix": 0.92,
            "visible_noise": 0.14,
            "occ_noise": 0.48,
            "false_positive_rate": 0.70,
            "score_visible": 0.86,
            "score_occ": 0.29,
        }
    raise ValueError("unknown scenario {!r}; use mild or hard".format(name))


def _reflect01(x: float, velocity: float) -> Tuple[float, float]:
    """Reflect a point at the unit-square boundary."""
    if x < 0.03:
        return 0.03 + (0.03 - x), abs(velocity)
    if x > 0.97:
        return 0.97 - (x - 0.97), -abs(velocity)
    return x, velocity


def generate_sequence(
    seed: int,
    n_frames: int = 72,
    n_objects: int = 6,
    embedding_dim: int = 12,
    scenario: str = "hard",
) -> SequenceData:
    """Generate trajectories, clean teacher latents, and detector outputs.

    The first object in every crossing pair is occasionally hidden and the
    detector can emit a low-score, appearance-contaminated box during that
    interval.  This creates the failure mode where updating an EMA template
    with the current box causes an identity switch.
    """
    cfg = _scenario_config(scenario)
    rng = np.random.RandomState(int(seed))

    # Pairwise crossing trajectories.  The small y offsets avoid a permanent
    # collision while still creating short ambiguous windows near the centre.
    n_pairs = int(math.ceil(n_objects / 2.0))
    positions = np.zeros((n_frames, n_objects, 2), dtype=np.float64)
    velocities = np.zeros((n_objects, 2), dtype=np.float64)
    starts = np.zeros((n_objects, 2), dtype=np.float64)
    for i in range(n_objects):
        pair = i // 2
        side = i % 2
        y = 0.18 + 0.64 * (pair / max(n_pairs - 1, 1))
        y += rng.uniform(-0.035, 0.035)
        starts[i] = [0.16 if side == 0 else 0.84, np.clip(y, 0.07, 0.93)]
        velocities[i] = [0.0105 if side == 0 else -0.0105,
                         rng.uniform(-0.0018, 0.0018)]

    # Identity appearance prototypes and slowly changing style directions.
    base = normalize(rng.normal(size=(n_objects, embedding_dim)))
    style = normalize(rng.normal(size=(n_objects, embedding_dim)))
    phase = rng.uniform(0.0, 2.0 * math.pi, size=n_objects)
    clean = np.zeros((n_frames, n_objects, embedding_dim), dtype=np.float64)

    # Random contiguous occlusion bursts.  Keep an explicit mask so the
    # evaluator can report an occlusion-only association metric.
    occluded = np.zeros((n_frames, n_objects), dtype=bool)
    for i in range(n_objects):
        n_bursts = 2 if scenario == "hard" else 1
        for _ in range(n_bursts):
            lo = max(5, n_frames // 7)
            hi = max(lo + 1, n_frames - 7)
            start = int(rng.randint(lo, hi))
            max_len = 9 if scenario == "hard" else 6
            length = int(rng.randint(3, max_len + 1))
            occluded[start:min(n_frames, start + length), i] = True

    for t in range(n_frames):
        for i in range(n_objects):
            if t == 0:
                p = starts[i].copy()
            else:
                # Mild non-linearity gives the context predictor something to
                # model while preserving a useful constant-velocity baseline.
                p = positions[t - 1, i] + velocities[i]
                p += np.array([
                    0.0015 * math.sin(0.09 * t + phase[i]),
                    0.0012 * math.cos(0.07 * t + phase[i]),
                ])
            p[0], velocities[i, 0] = _reflect01(float(p[0]), float(velocities[i, 0]))
            p[1], velocities[i, 1] = _reflect01(float(p[1]), float(velocities[i, 1]))
            positions[t, i] = p

            # The teacher sees a clean current crop.  Detector noise below is
            # never fed to this branch during predictor fitting.
            drift = 0.16 * math.sin(0.055 * t + phase[i])
            clean[t, i] = normalize(base[i] + drift * style[i] + 0.015 * rng.normal(size=embedding_dim))

    # Geometric overlap adds object-aware occlusions on top of random bursts.
    for t in range(n_frames):
        for i in range(n_objects):
            for j in range(i):
                if np.linalg.norm(positions[t, i] - positions[t, j]) < 0.072:
                    # Deterministic front/back choice; both are not hidden at
                    # once, which keeps the sequence evaluable.
                    occluded[t, i if ((t + i + j) % 3) else j] = True

    visible = ~occluded
    sizes = np.zeros((n_frames, n_objects, 2), dtype=np.float64)
    for t in range(n_frames):
        sizes[t, :, 0] = 0.075 + 0.006 * np.sin(0.03 * t + np.arange(n_objects))
        sizes[t, :, 1] = 0.105 + 0.006 * np.cos(0.025 * t + np.arange(n_objects))

    detections: List[List[Detection]] = []
    for t in range(n_frames):
        frame: List[Detection] = []
        for i in range(n_objects):
            is_occ = bool(occluded[t, i])
            p_det = cfg["occ_det_prob"] if is_occ else 0.94
            if rng.rand() > p_det:
                continue

            # Find a nearby identity to imitate during overlap/occlusion.
            nearest = None
            if n_objects > 1:
                distances = np.linalg.norm(positions[t] - positions[t, i], axis=1)
                distances[i] = 999.0
                near_idx = int(np.argmin(distances))
                if distances[near_idx] < 0.16:
                    nearest = near_idx

            # A detector often needs one or two frames to recover after a
            # full occlusion.  Those boxes are deliberately shifted toward a
            # nearby identity and have a contaminated crop.  This is the
            # adverse observation on which a prediction-based template should
            # differ from a raw EMA update.
            recent_occ = any(
                bool(occluded[tt, i])
                for tt in range(max(0, t - 2), t)
            )
            if is_occ:
                noise = cfg["occ_noise"]
                mix = cfg["occ_mix"] if nearest is not None else 0.0
            else:
                noise = cfg["visible_noise"]
                # A short crossing window can still produce a contaminated
                # detector crop, even though the object is technically visible.
                mix = 0.22 if nearest is not None and rng.rand() < 0.35 else 0.0
                if recent_occ and nearest is not None:
                    mix = max(mix, 0.86)
                    noise = max(noise, 0.30)

            emb = clean[t, i].copy()
            if nearest is not None and mix > 0.0:
                emb = (1.0 - mix) * emb + mix * clean[t, nearest]
            emb = normalize(emb + noise * rng.normal(size=embedding_dim))
            pos_noise = (0.022 if is_occ else 0.008) * rng.normal(size=2)
            if recent_occ and nearest is not None and not is_occ:
                # Move part-way into the neighbour's predicted location while
                # retaining enough signal for a valid detector box.
                pos_noise += 0.52 * (positions[t, nearest] - positions[t, i])
            score = cfg["score_occ"] + 0.08 * rng.randn() if is_occ else cfg["score_visible"] + 0.05 * rng.randn()
            if recent_occ and not is_occ:
                score = min(score, 0.48)
            score = float(np.clip(score, 0.05, 0.99))
            frame.append(Detection(
                center=np.clip(positions[t, i] + pos_noise, 0.0, 1.0),
                size=sizes[t, i].copy(),
                embedding=emb,
                score=score,
                gt_id=i,
            ))

        # False positives are concentrated near the centre/crossings.  Their
        # low scores make spawning conservative but still test the gate.
        n_fp = int(rng.poisson(cfg["false_positive_rate"]))
        for _ in range(n_fp):
            anchor = int(rng.randint(n_objects))
            fp_center = positions[t, anchor] + 0.07 * rng.normal(size=2)
            fp_center = np.clip(fp_center, 0.0, 1.0)
            donor = int(rng.randint(n_objects))
            fp_emb = normalize(clean[t, donor] + 0.48 * rng.normal(size=embedding_dim))
            frame.append(Detection(
                center=fp_center,
                size=np.array([0.07, 0.10], dtype=np.float64),
                embedding=fp_emb,
                score=float(np.clip(0.25 + 0.08 * rng.randn(), 0.05, 0.55)),
                gt_id=-1,
            ))
        detections.append(frame)

    return SequenceData(
        positions=positions,
        sizes=sizes,
        visible=visible,
        clean_embeddings=clean,
        detections=detections,
        occluded=occluded,
    )


def _find_detection(frame: Sequence[Detection], gt_id: int) -> Optional[Detection]:
    """Get the highest-score observation for one object in a frame."""
    candidates = [d for d in frame if d.gt_id == gt_id]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.score)


def build_context(
    history: Sequence[Optional[Detection]],
    current_center: Array,
    current_velocity: Array,
    history_len: int,
    embedding_dim: int,
    frame_age: int,
    mask_score_below: float = 0.55,
) -> Array:
    """Encode masked track context for the student predictor.

    Each lag contributes (embedding, center, confidence, mask).  Missing
    observations are represented by zeros and an explicit mask, mirroring a
    masked temporal context token in a JEPA implementation.
    """
    chunks: List[float] = []
    # ``history`` is ordered oldest -> newest and may contain fewer than K
    # entries for a newly spawned track.
    for lag in range(1, history_len + 1):
        idx = len(history) - lag
        det = history[idx] if idx >= 0 else None
        # Low-score crops are treated as masked tokens for the student.  This
        # is important: feeding a known contaminated appearance back into the
        # context would defeat the visibility-gated adaptation we are trying
        # to test.  The baseline still keeps and updates its raw EMA.
        if det is None or float(det.score) < mask_score_below:
            chunks.extend([0.0] * embedding_dim)
            chunks.extend([0.0, 0.0, 0.0, 0.0])
        else:
            chunks.extend(np.asarray(det.embedding, dtype=np.float64).tolist())
            chunks.extend([
                float(det.center[0]), float(det.center[1]),
                float(det.score), 1.0,
            ])
    chunks.extend(np.asarray(current_center, dtype=np.float64).tolist())
    chunks.extend(np.asarray(current_velocity, dtype=np.float64).tolist())
    chunks.append(float(min(frame_age, 30)) / 30.0)
    return np.asarray(chunks, dtype=np.float64)


class LinearJEPA:
    """Low-cost context-to-target predictor with an optional VICReg term.

    ``fit`` is the NumPy analogue of fitting a student predictor against a
    stop-gradient teacher target.  The ``covariance_weight`` branch adds the
    off-diagonal covariance penalty used by the repository's VICReg-style
    actor.  No target/clean tensor is retained by the tracker at inference.
    """

    def __init__(
        self,
        covariance_weight: float = 0.0,
        ridge: float = 1e-2,
        iterations: int = 80,
        learning_rate: float = 0.08,
    ) -> None:
        self.covariance_weight = float(covariance_weight)
        self.ridge = float(ridge)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.mean: Optional[Array] = None
        self.scale: Optional[Array] = None
        self.weight: Optional[Array] = None
        self.loss_history: List[Dict[str, float]] = []

    @staticmethod
    def _offdiag(cov: Array) -> Array:
        return cov - np.diag(np.diag(cov))

    def fit(self, contexts: Array, targets: Array) -> Dict[str, float]:
        x = np.asarray(contexts, dtype=np.float64)
        y = normalize(np.asarray(targets, dtype=np.float64))
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("contexts/targets must be two-dimensional with equal rows")
        if x.shape[0] < 4:
            raise ValueError("at least four context-target pairs are required")

        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-5] = 1.0
        xn = (x - self.mean) / self.scale
        xa = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float64)], axis=1)

        # Ridge initialization is stable even when the synthetic context has
        # repeated zero/mask columns.
        eye = np.eye(xa.shape[1], dtype=np.float64)
        eye[-1, -1] = 0.0
        self.weight = np.linalg.solve(
            xa.T.dot(xa) + self.ridge * eye,
            xa.T.dot(y),
        )

        # A few first-order steps add the VICReg covariance regularizer.  The
        # invariance-only control uses zero steps/weight and is therefore the
        # same predictor capacity without the covariance term.
        if self.covariance_weight > 0.0:
            n = float(xa.shape[0])
            d = float(y.shape[1])
            for it in range(self.iterations):
                pred = xa.dot(self.weight)
                err = pred - y
                grad = xa.T.dot(err) / n + self.ridge * self.weight
                centered = pred - pred.mean(axis=0, keepdims=True)
                cov = centered.T.dot(centered) / max(n - 1.0, 1.0)
                off = self._offdiag(cov)
                # d ||offdiag(C)||_F^2 / d pred = 4 Xc off / ((n-1)d)
                cov_grad_pred = 4.0 * centered.dot(off) / max((n - 1.0) * d, 1.0)
                grad += self.covariance_weight * xa.T.dot(cov_grad_pred) / n
                step = self.learning_rate / math.sqrt(1.0 + 0.03 * it)
                self.weight -= step * grad
                if it in (0, self.iterations - 1):
                    self.loss_history.append({
                        "iter": float(it),
                        "invariance": float(np.mean(err * err)),
                        "covariance": float(np.mean(off * off)),
                    })

        pred = self.predict(x)
        centered = pred - pred.mean(axis=0, keepdims=True)
        cov = centered.T.dot(centered) / max(pred.shape[0] - 1, 1)
        stats = {
            "pairs": float(x.shape[0]),
            "invariance_mse": float(np.mean((pred - y) ** 2)),
            "covariance_offdiag_mse": float(np.mean(self._offdiag(cov) ** 2)),
            "covariance_weight": float(self.covariance_weight),
        }
        return stats

    def predict(self, contexts: Array) -> Array:
        if self.weight is None or self.mean is None or self.scale is None:
            raise RuntimeError("LinearJEPA.predict called before fit")
        x = np.asarray(contexts, dtype=np.float64)
        one = x.ndim == 1
        if one:
            x = x[None, :]
        xn = (x - self.mean) / self.scale
        xa = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float64)], axis=1)
        out = normalize(xa.dot(self.weight))
        return out[0] if one else out


class TrackState:
    def __init__(self, track_id: int, det: Detection, frame: int, history_len: int) -> None:
        self.track_id = int(track_id)
        self.center = det.center.astype(np.float64).copy()
        self.velocity = np.zeros(2, dtype=np.float64)
        self.size = det.size.astype(np.float64).copy()
        self.appearance = normalize(det.embedding.astype(np.float64).copy())
        self.missed = 0
        self.last_score = float(det.score)
        self.history_len = int(history_len)
        self.history: List[Optional[Detection]] = [det]
        self.last_frame = int(frame)

    def append(self, det: Optional[Detection]) -> None:
        self.history.append(det)
        if len(self.history) > self.history_len:
            self.history = self.history[-self.history_len:]


class MOTTracker:
    """A minimal online association tracker used for all ablations."""

    def __init__(
        self,
        method: str,
        embedding_dim: int,
        predictor: Optional[LinearJEPA] = None,
        history_len: int = 4,
        max_age: int = 7,
        match_cost: float = 1.18,
        spawn_score: float = 0.58,
    ) -> None:
        self.method = method
        self.embedding_dim = int(embedding_dim)
        self.predictor = predictor
        self.history_len = int(history_len)
        self.max_age = int(max_age)
        self.match_cost = float(match_cost)
        self.spawn_score = float(spawn_score)
        self.tracks: Dict[int, TrackState] = {}
        self.next_id = 1
        self.frame = -1

    def _predicted_embedding(self, tr: TrackState) -> Array:
        if self.method not in ("jepa_inv", "jepa_cov", "no_teacher") or self.predictor is None:
            return tr.appearance.copy()
        ctx = build_context(
            tr.history,
            tr.center,
            tr.velocity,
            self.history_len,
            self.embedding_dim,
            tr.missed,
        )
        try:
            return self.predictor.predict(ctx)
        except RuntimeError:
            return tr.appearance.copy()

    def _cost(self, tr: TrackState, det: Detection, pred_center: Array, pred_emb: Array) -> float:
        # Motion is normalized so that a 0.08 displacement is roughly a unit
        # cost.  Appearance gets more weight after a missed/low-confidence
        # observation, where the JEPA prediction is most useful.
        motion = min(float(np.linalg.norm(det.center - pred_center) / 0.08), 3.0)
        raw_app = cosine_distance(tr.appearance, det.embedding)
        pred_app = cosine_distance(pred_emb, det.embedding)
        if self.method in ("jepa_inv", "jepa_cov", "no_teacher"):
            # Trust the predicted latent when the current crop is unreliable
            # or when the track has been absent for several frames.
            pred_weight = 0.72 if tr.missed > 0 or det.score < 0.55 else 0.42
            app = (1.0 - pred_weight) * raw_app + pred_weight * pred_app
        else:
            app = raw_app
        # Detector confidence only softens the appearance term; it is not a
        # privileged ground-truth visibility flag.
        confidence_penalty = 0.10 * (1.0 - float(det.score))
        return 0.53 * motion + 0.47 * app + confidence_penalty

    def step(self, detections: Sequence[Detection]) -> Dict[int, int]:
        self.frame += 1
        tracks = list(self.tracks.values())
        pred_centers = [tr.center + tr.velocity for tr in tracks]
        pred_embeddings = [self._predicted_embedding(tr) for tr in tracks]

        assignments: List[Tuple[int, int]] = []
        if tracks and detections:
            cost = np.zeros((len(tracks), len(detections)), dtype=np.float64)
            for r, tr in enumerate(tracks):
                for c, det in enumerate(detections):
                    cost[r, c] = self._cost(tr, det, pred_centers[r], pred_embeddings[r])
            assignments = _assign_with_unmatched(cost, self.match_cost)

        matched_tracks = set()
        matched_dets = set()
        output: Dict[int, int] = {}
        for r, c in assignments:
            tr = tracks[r]
            det = detections[c]
            matched_tracks.add(r)
            matched_dets.add(c)
            residual = det.center - pred_centers[r]
            tr.velocity = 0.70 * tr.velocity + 0.30 * residual
            tr.center = det.center.astype(np.float64).copy()
            tr.size = det.size.astype(np.float64).copy()
            tr.last_score = float(det.score)
            tr.missed = 0
            tr.last_frame = self.frame
            # The JEPA branch only adapts its raw memory with a reliable crop;
            # this is the analogue of GOT-JEPA's visibility-gated update.
            if self.method in ("jepa_inv", "jepa_cov", "no_teacher", "gated_ema"):
                if det.score >= 0.55:
                    tr.appearance = normalize(0.82 * tr.appearance + 0.18 * det.embedding)
            else:
                alpha = 0.20 + 0.30 * float(det.score)
                tr.appearance = normalize((1.0 - alpha) * tr.appearance + alpha * det.embedding)
            tr.append(det)
            output[tr.track_id] = c

        # Missing tracks keep a predicted state and an explicit mask token.
        for r, tr in enumerate(tracks):
            if r in matched_tracks:
                continue
            tr.center = pred_centers[r]
            tr.missed += 1
            tr.last_score *= 0.96
            tr.append(None)

        # Retire stale tracks after the update.  Keep IDs monotonic so IDSW
        # accounting remains unambiguous.
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].missed > self.max_age:
                del self.tracks[tid]

        # Spawn only reasonably confident unmatched detections.  The same
        # policy is used for baseline and JEPA to keep the input/control fair.
        for c, det in enumerate(detections):
            if c in matched_dets or det.score < self.spawn_score:
                continue
            # Avoid duplicate tracks on a nearby unmatched box.
            if any(np.linalg.norm(tr.center - det.center) < 0.045 for tr in self.tracks.values()):
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackState(tid, det, self.frame, self.history_len)
            output[tid] = c

        return output


def _assign_with_unmatched(cost: Array, unmatched_cost: float) -> List[Tuple[int, int]]:
    """Exact small rectangular assignment with an unmatched option.

    The benchmark uses six objects and a handful of false positives, so a
    bitmask dynamic program is both clearer and faster than adding a scipy
    dependency.  A greedy fallback keeps the helper usable for unusually
    large detector lists.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    if m > 16:
        pairs = []
        used_r = set()
        used_c = set()
        for flat in np.argsort(cost, axis=None):
            r, c = np.unravel_index(int(flat), cost.shape)
            if r in used_r or c in used_c or cost[r, c] >= unmatched_cost:
                continue
            used_r.add(int(r))
            used_c.add(int(c))
            pairs.append((int(r), int(c)))
        return pairs

    @lru_cache(maxsize=None)
    def solve(row: int, mask: int) -> Tuple[float, Tuple[int, ...]]:
        if row == n:
            return 0.0, ()
        best_cost, best_tail = solve(row + 1, mask)
        best_cost += unmatched_cost
        best_choice = (-1,) + best_tail
        for col in range(m):
            if mask & (1 << col):
                continue
            c = float(cost[row, col])
            if c >= unmatched_cost:
                continue
            tail_cost, tail = solve(row + 1, mask | (1 << col))
            total = c + tail_cost
            if total < best_cost:
                best_cost = total
                best_choice = (col,) + tail
        return best_cost, best_choice

    _, choices = solve(0, 0)
    return [(r, int(c)) for r, c in enumerate(choices) if c >= 0]


def collect_training_pairs(
    sequences: Sequence[SequenceData],
    history_len: int,
    embedding_dim: int,
) -> Tuple[Array, Array]:
    contexts: List[Array] = []
    targets: List[Array] = []
    for seq in sequences:
        for t in range(1, len(seq.detections)):
            for obj in range(seq.positions.shape[1]):
                # Build a clean, object-indexed context for predictor fitting.
                hist: List[Optional[Detection]] = []
                for tt in range(max(0, t - history_len), t):
                    hist.append(_find_detection(seq.detections[tt], obj))
                # At training time use the last observed state, never the clean
                # current target position.  This prevents target leakage.
                observed = [d for d in hist if d is not None and d.score >= 0.55]
                if observed:
                    last = observed[-1]
                    if len(observed) >= 2:
                        velocity = observed[-1].center - observed[-2].center
                    else:
                        velocity = np.zeros(2, dtype=np.float64)
                    center = last.center
                    # Dataclass equality is not safe for numpy-array fields;
                    # locate the last object by identity instead.
                    last_idx = max(k for k, item in enumerate(hist) if item is last)
                    age = t - (t - len(hist) + last_idx + 1)
                else:
                    center = seq.positions[max(0, t - 1), obj]
                    velocity = np.zeros(2, dtype=np.float64)
                    age = history_len
                contexts.append(build_context(hist, center, velocity, history_len, embedding_dim, age))
                # Stop-gradient teacher target: clean current latent only.
                targets.append(seq.clean_embeddings[t, obj])
    return np.stack(contexts), np.stack(targets)


def _detection_digest(sequences: Sequence[SequenceData]) -> str:
    h = hashlib.sha256()
    for seq in sequences:
        for frame in seq.detections:
            for d in frame:
                h.update(np.asarray(d.center, dtype=np.float64).tobytes())
                h.update(np.asarray(d.embedding, dtype=np.float64).tobytes())
                h.update(np.asarray([d.score, d.gt_id], dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _global_idtp(pairs: Sequence[Tuple[int, int]]) -> int:
    """Maximum bipartite count for (gt_id, track_id) observations."""
    if not pairs:
        return 0
    g_ids = sorted(set(g for g, _ in pairs))
    t_ids = sorted(set(t for _, t in pairs))
    counts = np.zeros((len(g_ids), len(t_ids)), dtype=np.int64)
    gi = {g: i for i, g in enumerate(g_ids)}
    ti = {t: i for i, t in enumerate(t_ids)}
    for g, t in pairs:
        counts[gi[g], ti[t]] += 1
    # Maximize counts using the same exact assignment helper on negative cost.
    rows = min(counts.shape[0], counts.shape[1])
    if rows == 0:
        return 0
    rows, cols = _max_assignment(counts)
    return int(sum(counts[r, c] for r, c in zip(rows, cols) if c >= 0))


def _max_assignment(matrix: Array) -> Tuple[List[int], List[int]]:
    """Small maximum-weight assignment, allowing unmatched rows."""
    n, m = matrix.shape
    if m > 16:
        # Greedy is only used for pathological numbers of tracks.
        used = set()
        rs, cs = [], []
        for r, c in sorted(((r, c) for r in range(n) for c in range(m)),
                           key=lambda rc: -matrix[rc[0], rc[1]]):
            if c not in used:
                rs.append(r)
                cs.append(c)
                used.add(c)
        return rs, cs

    @lru_cache(maxsize=None)
    def solve(row: int, mask: int) -> Tuple[int, Tuple[int, ...]]:
        if row == n:
            return 0, ()
        best, tail = solve(row + 1, mask)
        best_choice = (-1,) + tail
        for col in range(m):
            if mask & (1 << col):
                continue
            val, t = solve(row + 1, mask | (1 << col))
            val += int(matrix[row, col])
            if val > best:
                best, best_choice = val, (col,) + t
        return best, best_choice
    _, choice = solve(0, 0)
    rs = [r for r, c in enumerate(choice) if c >= 0]
    cs = [int(c) for c in choice if c >= 0]
    return rs, cs


def evaluate_sequence(seq: SequenceData, tracker: MOTTracker) -> Dict[str, object]:
    """Run one tracker and return MOT-like identity/occlusion statistics."""
    pair_observations: List[Tuple[int, int]] = []
    switches = 0
    previous_for_gt: Dict[int, int] = {}
    first_for_gt: Dict[int, int] = {}
    matched_valid = 0
    matched_total = 0
    association_correct = 0
    gt_visible_total = int(np.sum(seq.visible))
    gt_detected_total = 0
    occ_valid = 0
    occ_correct = 0
    post_occ_valid = 0
    post_occ_correct = 0
    # Track the end of each contiguous occlusion for recovery accounting.
    was_occ = np.zeros(seq.positions.shape[1], dtype=bool)

    for t, detections in enumerate(seq.detections):
        gt_detected_total += sum(1 for d in detections if d.gt_id >= 0)
        output = tracker.step(detections)
        for tid, di in output.items():
            det = detections[di]
            matched_total += 1
            if det.gt_id < 0:
                continue
            matched_valid += 1
            first_tid = first_for_gt.setdefault(det.gt_id, tid)
            correct = (tid == first_tid)
            pair_observations.append((det.gt_id, tid))
            if det.gt_id in previous_for_gt and previous_for_gt[det.gt_id] != tid:
                switches += 1
            previous_for_gt[det.gt_id] = tid
            association_correct += int(correct)
            if seq.occluded[t, det.gt_id]:
                occ_valid += 1
                occ_correct += int(correct)
            if was_occ[det.gt_id] and not seq.occluded[t, det.gt_id]:
                post_occ_valid += 1
                post_occ_correct += int(correct)
        was_occ = seq.occluded[t].copy()

    idtp = _global_idtp(pair_observations)
    # This smoke test evaluates *association given a detector stream*.  Use
    # the same observed GT set in IDF1's denominator; hidden/missed GT boxes
    # are reported separately through recall.  Calling this ``IDF1`` is kept
    # for convenience, and the JSON also exposes ``IDF1_definition``.
    idfp = max(matched_total - idtp, 0)
    idfn = max(gt_detected_total - idtp, 0)
    idf1 = (2.0 * idtp) / max(2.0 * idtp + idfp + idfn, 1.0)
    # Association accuracy is identity correctness *conditioned on a valid
    # matched detection*.  Detection recall is reported separately below.
    identity_consistency = association_correct / max(matched_valid, 1)
    # Keep the short name as a compatibility alias, but expose the precise
    # definition below so it is not mistaken for official MOT AssA.
    assoc_acc = identity_consistency
    occ_acc = occ_correct / max(occ_valid, 1)
    post_occ_acc = post_occ_correct / max(post_occ_valid, 1)
    det_recall = matched_valid / max(gt_detected_total, 1)
    # A transparent HOTA-like proxy: geometric mean of detection and identity
    # association accuracy.  It is not the official HOTA implementation.
    hota_proxy = math.sqrt(max(det_recall, 0.0) * max(assoc_acc, 0.0))
    return {
        "IDF1": float(idf1),
        "IDF1_observed": float(idf1),
        "IDF1_proxy": float(idf1),
        "IDSW": float(switches),
        "association_accuracy": float(assoc_acc),
        "identity_consistency": float(identity_consistency),
        "matched_valid": float(matched_valid),
        "matched_total": float(matched_total),
        "gt_detected": float(gt_detected_total),
        "gt_visible": float(gt_visible_total),
        "IDF1_definition": "global IDTP matching over detector-observed GT boxes; no IoU/frame matching, so this is an IDF1 proxy rather than official MOTChallenge IDF1",
        "association_accuracy_definition": "fraction of valid matched observations whose tracker ID equals that GT's first assigned tracker ID (identity consistency; initial assignment is not judged)",
        "det_recall": float(det_recall),
        "occlusion_association_accuracy": float(occ_acc),
        "occlusion_identity_consistency": float(occ_acc),
        "post_occlusion_recovery_accuracy": float(post_occ_acc),
        "HOTA_proxy": float(hota_proxy),
    }


def _aggregate(rows: Sequence[Dict[str, object]], metric_keys: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in metric_keys:
        # Per-sequence rows expose ``key`` while per-seed rows expose
        # ``key_mean``/``key_std``.  Accept both forms to make the two-level
        # aggregation explicit and avoid duplicating this helper.
        values = np.asarray([
            float(r[key] if key in r else r[key + "_mean"])
            for r in rows
        ], dtype=np.float64)
        out[key + "_mean"] = float(values.mean()) if values.size else 0.0
        out[key + "_std"] = float(values.std(ddof=0)) if values.size else 0.0
    return out


def run_experiment(args: argparse.Namespace) -> Dict[str, object]:
    scenario_names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    method_names = [s.strip() for s in args.methods.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")

    metric_keys = [
        "IDF1", "IDSW", "association_accuracy", "det_recall",
        "occlusion_association_accuracy", "post_occlusion_recovery_accuracy",
        "HOTA_proxy",
    ]
    all_rows: List[Dict[str, object]] = []
    summary: Dict[str, object] = {
        "config": {
            "train_sequences": args.train_sequences,
            "eval_sequences": args.eval_sequences,
            "frames": args.frames,
            "objects": args.objects,
            "embedding_dim": args.embedding_dim,
            "history_len": args.history_len,
            "scenarios": scenario_names,
            "methods": method_names,
            "seeds": seeds,
            "covariance_weight": args.covariance_weight,
            "numpy_version": np.__version__,
            "python": sys.version.split()[0],
        },
        "results": {},
    }

    for scenario in scenario_names:
        scenario_rows: Dict[str, List[Dict[str, object]]] = {m: [] for m in method_names}
        for seed in seeds:
            # Strict train/eval separation: train seeds and evaluation seeds
            # are disjoint deterministic transforms of the requested seed.
            train_sequences = [generate_sequence(
                seed=100000 + seed * 1000 + k,
                n_frames=args.frames,
                n_objects=args.objects,
                embedding_dim=args.embedding_dim,
                scenario=scenario,
            ) for k in range(args.train_sequences)]
            eval_sequences = [generate_sequence(
                seed=200000 + seed * 1000 + k,
                n_frames=args.frames,
                n_objects=args.objects,
                embedding_dim=args.embedding_dim,
                scenario=scenario,
            ) for k in range(args.eval_sequences)]
            contexts, targets = collect_training_pairs(train_sequences, args.history_len, args.embedding_dim)
            predictor_inv = LinearJEPA(covariance_weight=0.0, ridge=args.ridge)
            fit_inv = predictor_inv.fit(contexts, targets)
            predictor_cov = LinearJEPA(
                covariance_weight=args.covariance_weight,
                ridge=args.ridge,
                iterations=args.cov_iterations,
            )
            fit_cov = predictor_cov.fit(contexts, targets)
            fit_no_teacher = None

            # Identical detector stream for every method in this seed/scenario.
            digest = _detection_digest(eval_sequences)
            for method in method_names:
                if method == "baseline":
                    predictor = None
                elif method == "gated_ema":
                    # Gate-only control: exactly the JEPA branch's quality
                    # gated EMA, but without a context predictor.
                    predictor = None
                elif method == "jepa_inv":
                    predictor = predictor_inv
                elif method == "jepa_cov":
                    predictor = predictor_cov
                elif method == "no_teacher":
                    # Same predictor capacity, but target rows are shuffled:
                    # this is a no-teacher-loss control, not a tuned baseline.
                    shuffled = np.random.RandomState(900000 + seed).permutation(len(targets))
                    predictor = LinearJEPA(covariance_weight=0.0, ridge=args.ridge)
                    fit_no_teacher = predictor.fit(contexts, targets[shuffled])
                else:
                    raise ValueError("unknown method {!r}".format(method))

                method_metrics: List[Dict[str, float]] = []
                for seq in eval_sequences:
                    tracker = MOTTracker(
                        method=method,
                        embedding_dim=args.embedding_dim,
                        predictor=predictor,
                        history_len=args.history_len,
                        max_age=args.max_age,
                        match_cost=args.match_cost,
                        spawn_score=args.spawn_score,
                    )
                    # evaluate_sequence only reads detections; each method gets
                    # a fresh tracker and the same objects/digest.
                    method_metrics.append(evaluate_sequence(seq, tracker))
                agg = _aggregate(method_metrics, metric_keys)
                row: Dict[str, object] = {
                    "scenario": scenario,
                    "seed": seed,
                    "method": method,
                    "detections_sha256_16": digest,
                    "fit_invariance": fit_inv if method == "jepa_inv" else (fit_no_teacher if method == "no_teacher" else None),
                    "fit_covariance": fit_cov if method == "jepa_cov" else None,
                }
                row.update(agg)
                scenario_rows[method].append(row)
                all_rows.append(row)

        summary["results"][scenario] = {}
        for method, rows in scenario_rows.items():
            summary["results"][scenario][method] = {
                "n_runs": len(rows),
                # ``rows`` already contains per-seed means/stds.  Aggregate
                # the raw metric names here so the public JSON keys remain
                # ``IDF1_mean``/``IDF1_std`` (rather than a confusing nested
                # ``IDF1_mean_mean`` spelling).
                "metrics": _aggregate(rows, metric_keys),
                "runs": rows,
            }

    # Human-readable deltas relative to the raw EMA baseline.
    deltas: Dict[str, object] = {}
    for scenario in scenario_names:
        base = summary["results"][scenario].get("baseline")
        if not base:
            continue
        base_metrics = base["metrics"]
        deltas[scenario] = {}
        for method in method_names:
            if method == "baseline" or method not in summary["results"][scenario]:
                continue
            mm = summary["results"][scenario][method]["metrics"]
            deltas[scenario][method] = {
                "delta_IDF1_mean": mm["IDF1_mean"] - base_metrics["IDF1_mean"],
                "delta_IDSW_mean": mm["IDSW_mean"] - base_metrics["IDSW_mean"],
                "delta_occlusion_accuracy_mean": mm["occlusion_association_accuracy_mean"] - base_metrics["occlusion_association_accuracy_mean"],
            }
    summary["deltas_vs_baseline"] = deltas

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    csv_path = os.path.splitext(out_path)[0] + ".csv"
    csv_fields = ["scenario", "seed", "method", "detections_sha256_16"]
    csv_fields += [k + suffix for k in (metric_keys) for suffix in ("_mean", "_std")]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key, "") for key in csv_fields})

    _print_report(summary, out_path, csv_path)
    return summary


def _print_report(summary: Dict[str, object], out_path: str, csv_path: str) -> None:
    cfg = summary["config"]
    print("MOT JEPA synthetic mechanism smoke")
    print("config: {}".format(json.dumps(cfg, sort_keys=True)))
    for scenario, methods in summary["results"].items():
        print("scenario={}".format(scenario))
        for method, payload in methods.items():
            m = payload["metrics"]
            print(
                "  {:10s} IDF1 {:.4f}±{:.4f} | IDSW {:.2f}±{:.2f} | "
                "assoc {:.4f} | occ {:.4f} | post-occ {:.4f} | HOTA* {:.4f}".format(
                    method,
                    m["IDF1_mean"], m["IDF1_std"],
                    m["IDSW_mean"], m["IDSW_std"],
                    m["association_accuracy_mean"],
                    m["occlusion_association_accuracy_mean"],
                    m["post_occlusion_recovery_accuracy_mean"],
                    m["HOTA_proxy_mean"],
                )
            )
    print("results_json: {}".format(out_path))
    print("results_csv: {}".format(csv_path))
    print("* HOTA is a transparent detection/association proxy, not official HOTA.")


def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-sequences", type=int, default=12)
    p.add_argument("--eval-sequences", type=int, default=12)
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--objects", type=int, default=6)
    p.add_argument("--embedding-dim", type=int, default=12)
    p.add_argument("--history-len", type=int, default=4)
    p.add_argument("--scenarios", default="mild,hard")
    p.add_argument("--methods", default="baseline,gated_ema,jepa_inv,jepa_cov,no_teacher")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--covariance-weight", type=float, default=0.08)
    p.add_argument("--cov-iterations", type=int, default=60)
    p.add_argument("--ridge", type=float, default=0.03)
    p.add_argument("--max-age", type=int, default=7)
    p.add_argument("--match-cost", type=float, default=1.18)
    p.add_argument("--spawn-score", type=float, default=0.58)
    p.add_argument("--output", default="experiments/results/mot_jepa_synthetic.json")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = make_argparser().parse_args(argv)
    if args.train_sequences < 1 or args.eval_sequences < 1 or args.frames < 8:
        raise SystemExit("train/eval sequences must be >=1 and frames >=8")
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
