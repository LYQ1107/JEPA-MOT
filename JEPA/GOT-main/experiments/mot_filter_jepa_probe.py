#!/usr/bin/env python3
"""Lagged MOT association-filter JEPA probe (handcrafted 80-D encoder).

At frame t a clean teacher computes a closed-form association filter w_T from
strictly earlier detector history, the clean GT crop at t, and same-frame hard
negatives.  A student sees only the causal history and a corrupted detector
crop at t and predicts w_T.  Neither vector is used to score frame t: scoring
is delayed to the fixed detector candidate set at t+1.  This leave-current-out
replay is a representation diagnostic (oracle identity histories are used to
assemble examples), not an online tracker or leaderboard result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import current_pair_jepa_probe as base  # noqa: E402

DIM = 80
K = 4
RIDGE_LAMBDA = 0.25


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _prior_features(gid: int, frame: int, gt, det, gmap, k: int = K):
    rows = []
    for fr in range(max(1, frame - 32), frame):
        j = gmap.get((fr, gid))
        if j is None or j >= len(det.get(fr, [])):
            continue
        d = det[fr][j]
        if d.feat is not None:
            rows.append((fr, d))
    return rows[-k:]


def ridge_filter(history: Sequence[np.ndarray], clean: np.ndarray, negatives: Sequence[np.ndarray], lam: float = RIDGE_LAMBDA, return_diag: bool = False):
    """Dual ridge association filter, with +1 positive and -1 negatives."""
    rows = [np.asarray(x, np.float32) for x in history if x is not None]
    rows.append(np.asarray(clean, np.float32))
    rows.extend(np.asarray(x, np.float32) for x in negatives if x is not None)
    if not rows:
        out = np.zeros(DIM, np.float32); diag = {"condition_number": float("nan"), "residual_l2": float("nan"), "invalid": True}
        return (out, diag) if return_diag else out
    # Use float64 for the dual solve.  MOT crop rows can be highly correlated
    # (especially on adjacent frames), so float32 condition estimates can hide
    # an unstable filter and make the claimed teacher quality irreproducible.
    X0 = np.asarray(rows, np.float64)
    X = X0 / np.maximum(np.linalg.norm(X0, axis=1, keepdims=True), 1e-12)
    y = np.concatenate([np.ones(len(history) + 1, np.float64), -np.ones(len(negatives), np.float64)])
    gram = X @ X.T + float(lam) * np.eye(len(X), dtype=np.float64)
    cond = float(np.linalg.cond(gram))
    invalid = (not np.isfinite(cond)) or cond > 1e12
    try:
        alpha = np.linalg.solve(gram, y)
        residual = float(np.linalg.norm(gram @ alpha - y))
        w = base.normalize((X.T @ alpha).astype(np.float32)).astype(np.float32)
        invalid = invalid or (not np.isfinite(residual)) or (not np.isfinite(w).all())
    except np.linalg.LinAlgError:
        alpha = np.zeros(len(X), np.float64); residual = float("inf"); w = np.zeros(DIM, np.float32); invalid = True
    diag = {"condition_number": cond, "residual_l2": residual, "invalid": bool(invalid), "rows": int(len(X)), "dtype": "float64"}
    return (w, diag) if return_diag else w


def _arrays(records: Sequence[dict]) -> Dict[str, np.ndarray]:
    keys = ("hist", "candidate", "corrupt", "geom", "target", "teacher_w", "next_negative")
    return {k: np.asarray([r[k] for r in records], np.float32) for k in keys}


def build_filter_records(items, max_frames: int, seed: int, max_samples: int):
    rng = np.random.RandomState(seed)
    records: List[dict] = []
    det_h = hashlib.sha256(); feat_h = hashlib.sha256(); clean_h = hashlib.sha256()
    audit = {"assigned": 0, "iou_rejected_after_assignment": 0, "duplicate_count": 0, "frames": 0}
    for seq, seq_dir, gt, det, info in items:
        iw, ih = base.sequence_wh(seq_dir)
        base.attach_features(seq_dir, det, max_frames)
        clean = base.clean_gt_features(seq_dir, gt, max_frames)
        gmap, dmap, ma = base.unique_hungarian_matches(gt, det, base.IOU_MATCH_THRESHOLD if hasattr(base, "IOU_MATCH_THRESHOLD") else .30)
        for k0, v in ma.items(): audit[k0] = int(audit.get(k0, 0)) + int(v)
        audit["frames"] += len(set(gt) | set(det))
        # Hash all detector-only frames, not just GT frames.
        for fr in sorted(det):
            for j, d in enumerate(det[fr]):
                det_h.update(f"{seq}:{fr}:{j}:{d.score:.7f}:".encode()); det_h.update(np.asarray(d.box, np.float32).tobytes())
                if d.feat is not None:
                    feat_h.update(f"{seq}:{fr}:{j}:".encode()); feat_h.update(np.asarray(d.feat, np.float32).tobytes())
        for fr in sorted(gt):
            # A lagged example needs a positive match at t+1.
            if fr + 1 not in det or fr + 1 not in gt:
                continue
            ds, next_ds = det.get(fr, []), det.get(fr + 1, [])
            for g in gt.get(fr, []):
                j = gmap.get((fr, g.gid)); nj = gmap.get((fr + 1, g.gid)); y = clean.get((fr, g.gid))
                if j is None or nj is None or j >= len(ds) or nj >= len(next_ds) or y is None or ds[j].feat is None or next_ds[nj].feat is None:
                    continue
                prior = _prior_features(g.gid, fr, gt, det, gmap)
                hist = base._history_for(g.gid, fr, gt, det, gmap, K, iw, ih)
                hvec, hstate, valid, gap, latest = hist
                geom = base._geometry(ds[j], hstate, latest, iw, ih)
                # Current-frame hard negatives for the teacher; labels never
                # enter the student input.
                cur_feats = [d.feat for q, d in enumerate(ds) if q != j and d.feat is not None]
                sims = sorted(cur_feats, key=lambda z: float(np.dot(base.normalize(z), base.normalize(y))), reverse=True)
                negs = sims[:8]
                tw, ridge_diag = ridge_filter([d.feat for _, d in prior], y, negs, return_diag=True)
                next_feats = [d.feat for d in next_ds if d.feat is not None]
                next_idx = [q for q, d in enumerate(next_ds) if d.feat is not None]
                if nj not in next_idx:
                    continue
                next_cands = np.asarray(next_feats, np.float32); pos_col = next_idx.index(nj)
                next_sims = [float(np.dot(base.normalize(z), tw)) if q != pos_col else -2. for q, z in enumerate(next_feats)]
                next_neg = next_feats[int(np.argmax(next_sims))] if next_feats and max(next_sims) > -1 else np.zeros(DIM, np.float32)
                rec = {
                    "sequence": seq, "frame": int(fr), "gid": int(g.gid), "det_index": int(j),
                    "hist": hvec.astype(np.float32), "candidate": ds[j].feat.astype(np.float32),
                    "corrupt": base.corrupt_feature(ds[j].feat, seed * 1000003 + fr * 1009 + g.gid),
                    "geom": geom.astype(np.float32), "target": y.astype(np.float32), "teacher_w": tw.astype(np.float32),
                    "next_candidates": next_cands, "next_positive_column": int(pos_col), "next_negative": np.asarray(next_neg, np.float32),
                    "history_valid_fraction": float(valid), "latest_gap": float(gap), "ridge_diag": ridge_diag,
                }
                records.append(rec); clean_h.update(f"{seq}:{fr}:{g.gid}:".encode()); clean_h.update(y.tobytes())
    if max_samples and len(records) > max_samples:
        keep = np.sort(rng.choice(len(records), max_samples, replace=False)); records = [records[int(i)] for i in keep]
    if not records:
        raise RuntimeError("no lagged one-to-one records with positive t+1 match")
    conds = np.asarray([r["ridge_diag"]["condition_number"] for r in records], np.float64)
    residuals = np.asarray([r["ridge_diag"]["residual_l2"] for r in records], np.float64)
    invalid_count = int(sum(bool(r["ridge_diag"].get("invalid", True)) for r in records))
    return _arrays(records), records, {
        "sample_count": len(records), "matching": "per-frame Hungarian linear_sum_assignment, IoU>=0.30",
        "matching_audit": audit, "duplicate_count": int(audit["duplicate_count"]),
        "detector_sha256": det_h.hexdigest(), "feature_sha256": feat_h.hexdigest(), "clean_target_sha256": clean_h.hexdigest(),
        "lag": 1, "leave_current_out": True, "history_source": "oracle identity only for replay assembly; frames < t",
        "ridge_dtype": "float64", "ridge_condition_number_median": float(np.nanmedian(conds)), "ridge_condition_number_max": float(np.nanmax(conds)),
        "ridge_residual_l2_median": float(np.nanmedian(residuals)), "ridge_residual_l2_max": float(np.nanmax(residuals)), "ridge_invalid_count": invalid_count,
    }


def normalize_torch(x):
    import torch
    return x / torch.clamp(torch.linalg.vector_norm(x, dim=1, keepdim=True), min=1e-8)


class FilterStudent:
    def __init__(self, mode: str, seed: int, epochs: int = 8):
        self.mode, self.seed, self.epochs = mode, seed, epochs
        self.input_dim = DIM * 2 + 9
        self.params = None; self.stats = {}

    @staticmethod
    def _gelu(z): return .5 * z * (1. + np.tanh(np.sqrt(2. / np.pi) * (z + .044715 * z ** 3)))
    @staticmethod
    def _forward(x, p):
        w1, b1, gn, gb, w2, b2 = p; z = x.dot(w1.T) + b1; mu = z.mean(1, keepdims=True); var = ((z - mu) ** 2).mean(1, keepdims=True); z = (z - mu) / np.sqrt(var + 1e-5) * gn + gb; return np.nan_to_num(FilterStudent._gelu(z).dot(w2.T) + b2, nan=0., posinf=0., neginf=0.)
    @staticmethod
    def _copy(model):
        sd = model.state_dict(); return tuple(sd[k].detach().cpu().numpy().astype(np.float32) for k in ("net.0.weight", "net.0.bias", "net.1.weight", "net.1.bias", "net.3.weight", "net.3.bias"))
    @staticmethod
    def _hash(p): return hashlib.sha256(b"".join(np.asarray(x, np.float32).tobytes() for x in p)).hexdigest()

    def fit(self, arr: Dict[str, np.ndarray]):
        import torch, torch.nn as nn, torch.nn.functional as F
        torch.set_num_threads(1); torch.manual_seed(self.seed)
        class Net(nn.Module):
            def __init__(self, inp, out): super().__init__(); self.net = nn.Sequential(nn.Linear(inp, 192), nn.LayerNorm(192), nn.GELU(), nn.Linear(192, out))
            def forward(self, x): return self.net(x)
        h, c, g = arr["hist"].copy(), arr["corrupt"].copy(), arr["geom"].copy()
        if self.mode == "current_only": h[...] = 0.; g[:, 5:9] = 0.
        if self.mode in ("full_mask", "history_only"): c[...] = 0.
        if self.mode == "both_clean": c = arr["target"].copy()
        x = np.concatenate([h, c, g], 1)
        y = arr["target"] if self.mode == "no_teacher" else arr["teacher_w"]
        n = len(x); model = Net(self.input_dim, DIM); tx = torch.from_numpy(x); ty = torch.from_numpy(base.normalize(y)); tn = torch.from_numpy(base.normalize(arr["next_negative"]))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4); bs = min(512, n); rng = np.random.RandomState(self.seed); losses = []
        model.train()
        for ep in range(max(1, self.epochs)):
            order = rng.permutation(n); total = 0.
            for st in range(0, n, bs):
                ix = order[st:st + bs]; p = normalize_torch(model(tx[ix])); pos = (p * ty[ix]).sum(1); neg = (p * tn[ix]).sum(1); loss = (1. - pos).mean() + .5 * F.softplus(.2 - pos + neg).mean(); z = p - p.mean(0, keepdim=True); cov = (z.T @ z) / max(1, z.shape[0] - 1); off = cov - torch.diag(torch.diag(cov)); loss = loss + .03 * (off ** 2).mean()
                if not torch.isfinite(loss): raise FloatingPointError("non-finite filter student loss")
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
                with torch.no_grad():
                    for q in model.parameters(): q.clamp_(-10., 10.)
                total += float(loss.detach()) * len(ix)
            losses.append(total / n)
        model.eval(); self.params = self._copy(model); pp = base.normalize(self._forward(x, self.params)); cc = np.sum(pp * base.normalize(y), 1)
        self.stats = {"mode": self.mode, "samples": int(n), "epochs": int(self.epochs), "first_loss": float(losses[0]), "last_loss": float(losses[-1]), "target_cosine_mean": float(cc.mean()), "target_cosine_p10": float(np.percentile(cc, 10)), "parameter_sha256": self._hash(self.params), "finite": bool(np.isfinite(pp).all())}

    def fit_random(self):
        rng = np.random.RandomState(self.seed + 404); self.params = (rng.normal(0., .05, (192, self.input_dim)).astype(np.float32), np.zeros(192, np.float32), np.ones(192, np.float32), np.zeros(192, np.float32), rng.normal(0., .05, (DIM, 192)).astype(np.float32), np.zeros(DIM, np.float32)); self.stats = {"mode": self.mode, "random_frozen": True, "samples": 0, "parameter_sha256": self._hash(self.params), "finite": True}

    def predict(self, rec: dict, clean: bool = False) -> np.ndarray:
        h, c, g = rec["hist"].copy(), (rec["target"] if clean or self.mode == "both_clean" else rec["corrupt"]).copy(), rec["geom"].copy()
        if self.mode == "current_only": h[...] = 0.; g[5:9] = 0.
        if self.mode in ("full_mask", "history_only"): c[...] = 0.
        return base.normalize(self._forward(np.concatenate([h, c, g])[None], self.params))[0].astype(np.float32)


def score(proto: np.ndarray, rec: dict) -> Tuple[float, float, int]:
    c = base.normalize(np.asarray(rec["next_candidates"], np.float32)); sims = c @ base.normalize(proto); p = int(rec["next_positive_column"]); pos = float(sims[p]); other = np.delete(sims, p); hard = float(np.max(other)) if len(other) else -1.; rank = int(1 + np.sum(sims > pos)); return pos, pos - hard, rank


def metrics(proto_fn, records: Sequence[dict], clean: bool = False) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    vals = []; ranks = []; counts = []; order_diffs = []; rank_mismatches = 0
    for r in records:
        z = proto_fn(r, clean); p, m, k = score(z, r); vals.append((p, m)); ranks.append(float(k == 1)); counts.append(len(r["next_candidates"]))
        # Explicitly permute the *next-frame* candidate list and remap the
        # positive column.  The pair-conditioned prototype must not depend on
        # list order, and the rank must remain identical.
        cand = np.asarray(r["next_candidates"], np.float32); pc = int(r["next_positive_column"])
        if len(cand) > 1 and pc >= 0:
            perm = np.random.RandomState(1301 + int(r["frame"]) + int(r["gid"])).permutation(len(cand))
            r_perm = dict(r)
            r_perm["next_candidates"] = cand[perm]
            new_pc = int(np.where(perm == pc)[0][0])
            r_perm["next_positive_column"] = new_pc
            z_perm = proto_fn(r_perm, clean)
            order_diffs.append(float(np.max(np.abs(z - z_perm))))
            new_pc = int(r_perm["next_positive_column"]); sims = base.normalize(cand[perm], axis=1) @ base.normalize(z_perm); rk = int(1 + np.sum(sims > sims[new_pc]))
            if rk != k: rank_mismatches += 1
        else:
            order_diffs.append(0.)
    a = np.asarray(vals, np.float64)
    if len(a) == 0:
        empty = {"samples": 0., "recall_at_1": float("nan"), "positive_next_cosine_mean": float("nan"), "margin_mean": float("nan"), "margin_p10": float("nan"), "candidate_count_mean": float("nan"), "candidate_order_max_abs_diff": 0., "candidate_order_invariant": True, "permutation_rank_mismatch_count": 0, "finite": True}
        return empty, {"r1": np.zeros(0), "margin": np.zeros(0), "cos": np.zeros(0)}
    return {"samples": float(len(a)), "recall_at_1": float(np.mean(ranks)), "positive_next_cosine_mean": float(a[:, 0].mean()), "margin_mean": float(a[:, 1].mean()), "margin_p10": float(np.percentile(a[:, 1], 10)), "candidate_count_mean": float(np.mean(counts)), "candidate_order_max_abs_diff": float(max(order_diffs) if order_diffs else 0.), "candidate_order_invariant": bool((max(order_diffs) if order_diffs else 0.) <= 1e-7 and rank_mismatches == 0), "permutation_rank_mismatch_count": int(rank_mismatches), "finite": bool(np.isfinite(a).all())}, {"r1": np.asarray(ranks), "margin": a[:, 1], "cos": a[:, 0]}


def paired(a, b, seed=17, n_boot=2000):
    d = np.asarray(a) - np.asarray(b); rng = np.random.RandomState(seed); ix = rng.randint(0, len(d), (n_boot, len(d))); q = d[ix].mean(1); return {"n": int(len(d)), "mean_diff": float(d.mean()), "ci95_low": float(np.percentile(q, 2.5)), "ci95_high": float(np.percentile(q, 97.5)), "n_boot": int(n_boot), "seed": int(seed)}


def split_fit_holdout(records: Sequence[dict], fraction: float = .80) -> Tuple[List[dict], List[dict], dict]:
    """Deterministic train-fit/within-train-heldout split by frame.

    The split is performed independently per sequence, so no held-out frame
    contributes to student fitting or to the preregistered quality gate.
    """
    fit, hold = [], []; detail = {}
    for seq in sorted({str(r["sequence"]) for r in records}):
        rr = [r for r in records if str(r["sequence"]) == seq]
        frames = sorted({int(r["frame"]) for r in rr})
        cut = int(np.floor(len(frames) * fraction)); cut = min(max(cut, 1), max(1, len(frames) - 1))
        fit_frames, hold_frames = set(frames[:cut]), set(frames[cut:])
        # Tiny sequences still get a non-empty holdout when possible.
        if not hold_frames and len(frames) > 1:
            hold_frames = {frames[-1]}; fit_frames = set(frames[:-1])
        for r in rr:
            (fit if int(r["frame"]) in fit_frames else hold).append(r)
        detail[seq] = {"all_frames": frames, "fit_frames": sorted(fit_frames), "holdout_frames": sorted(hold_frames), "fit_samples": sum(int(r["frame"]) in fit_frames for r in rr), "holdout_samples": sum(int(r["frame"]) in hold_frames for r in rr)}
    return fit, hold, detail


def run(args):
    train_items, eval_items, detector_meta = base.load_items("mot17", args)
    ta, tr, tm = build_filter_records(train_items, args.max_frames, args.seed, args.max_samples)
    ea, er, em = build_filter_records(eval_items, args.max_frames, args.seed + 7001, min(args.max_samples, 12000))
    if tm["duplicate_count"] or em["duplicate_count"]: raise AssertionError("duplicate one-to-one match")
    train_fit, train_hold, split_detail = split_fit_holdout(tr)
    if not train_fit or not train_hold:
        raise RuntimeError("train-fit/heldout split is empty")
    ta_fit = _arrays(train_fit)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    models = {}; stats = {}
    for m in methods:
        if m in ("raw", "history"):
            continue
        s = FilterStudent(m, args.seed, args.epochs)
        if m == "random_frozen": s.fit_random()
        else: s.fit(ta_fit)
        models[m] = s; stats[m] = s.stats
    def fn(m, rec, clean=False):
        if m == "raw": return rec["candidate"]
        if m == "history": return rec["hist"]
        return models[m].predict(rec, clean)
    validation = {}; score_arrays = {}
    for m in methods: validation[m], score_arrays[m] = metrics(lambda r, c, mm=m: fn(mm, r, c), er, clean=(m == "both_clean"))
    if any(not bool(v.get("candidate_order_invariant", False)) for v in validation.values()):
        raise AssertionError("candidate permutation audit failed")
    fit_validation = {m: metrics(lambda r, c, mm=m: fn(mm, r, c), train_fit, clean=(m == "both_clean"))[0] for m in methods}
    hold_validation = {}; hold_arrays = {}
    for m in methods:
        hold_validation[m], hold_arrays[m] = metrics(lambda r, c, mm=m: fn(mm, r, c), train_hold, clean=(m == "both_clean"))
    if any(not bool(v.get("candidate_order_invariant", False)) for v in hold_validation.values()):
        raise AssertionError("heldout candidate permutation audit failed")
    byseq = {}
    for seq in sorted({r["sequence"] for r in er}):
        rr = [r for r in er if r["sequence"] == seq]; byseq[seq] = {m: metrics(lambda r, c, mm=m: fn(mm, r, c), rr, clean=(m == "both_clean"))[0] for m in methods}
    teacher_scores = metrics(lambda r, c: r["teacher_w"], er)[0]
    teacher_holdout = metrics(lambda r, c: r["teacher_w"], train_hold)[0]
    paired_out = {}
    if "current_jepa" in score_arrays:
        for ctrl in ("raw", "history", "no_teacher", "random_frozen", "both_clean"):
            if ctrl in score_arrays:
                paired_out["current_jepa_minus_" + ctrl] = {k: paired(score_arrays["current_jepa"][k], score_arrays[ctrl][k], args.seed + i) for i, k in enumerate(("r1", "margin", "cos"))}
    # A preregistered, non-eval quality gate.  Eval metrics remain descriptive
    # and cannot decide whether an online tracker is attempted.
    gate = {
        "source": "within-train heldout frames (last 20% per sequence)", "teacher_holdout_r1": float(teacher_holdout.get("recall_at_1", float("nan"))),
        "teacher_holdout_margin": float(teacher_holdout.get("margin_mean", float("nan"))), "teacher_threshold_r1": .80,
        "teacher_useful": bool(teacher_holdout.get("recall_at_1", 0.) >= .80 and teacher_holdout.get("margin_mean", -1.) > 0.),
        "current_beats_no_teacher": bool("current_jepa" in hold_validation and "no_teacher" in hold_validation and hold_validation["current_jepa"].get("recall_at_1", -1.) > hold_validation["no_teacher"].get("recall_at_1", -1.)),
        "current_beats_random": bool("current_jepa" in hold_validation and "random_frozen" in hold_validation and hold_validation["current_jepa"].get("recall_at_1", -1.) > hold_validation["random_frozen"].get("recall_at_1", -1.)),
        "current_beats_history": bool("current_jepa" in hold_validation and "history" in hold_validation and hold_validation["current_jepa"].get("recall_at_1", -1.) > hold_validation["history"].get("recall_at_1", -1.) and hold_validation["current_jepa"].get("margin_mean", -1.) > hold_validation["history"].get("margin_mean", -1.)),
        "meaningful_margin_required": .005,
    }
    # Require a positive, practically meaningful margin against both direct
    # and history controls and an actual win over the history baseline.  A
    # one-sample R@1 fluctuation on a 20-frame smoke must not authorize tracker
    # integration.
    current_margin = hold_validation.get("current_jepa", {}).get("margin_mean", -1.)
    nt_margin = hold_validation.get("no_teacher", {}).get("margin_mean", -1.)
    gate["current_margin"] = float(current_margin); gate["no_teacher_margin"] = float(nt_margin)
    gate["current_margin_delta_vs_no_teacher"] = float(current_margin - nt_margin)
    gate["proceed_lagged_tracker"] = bool(gate["teacher_useful"] and gate["current_beats_no_teacher"] and gate["current_beats_random"] and gate["current_beats_history"] and (current_margin - nt_margin) >= gate["meaningful_margin_required"])
    protocol = {"schema": "mot-filter-jepa-probe-v2", "teacher": "closed-form dual ridge w_T at t from history + clean GT current + same-frame hard negatives (float64 solve)", "student": "MLP predicts w_T from causal history + corrupted detector current", "lag": 1, "score_frame": "t+1 only", "leave_current_out": True, "matching": "per-frame Hungarian IoU>=0.30", "duplicate_count": 0, "history_oracle_for_probe": True, "student_gt_inputs": False, "controls": methods, "deployment_corruption_default": "off", "eval_uses_gt": "probe history assembly and labels only", "quality_gate_source": "train heldout, not eval"}
    return {"dataset": "mot17", "detector": detector_meta, "train_sequences": [x[0] for x in train_items], "eval_sequences": [x[0] for x in eval_items], "train_samples": len(tr), "train_fit_samples": len(train_fit), "train_holdout_samples": len(train_hold), "train_split": split_detail, "eval_samples": len(er), "train_meta": tm, "eval_meta": em, "predictor_stats": stats, "fit_validation": fit_validation, "train_holdout_validation": hold_validation, "validation": validation, "validation_by_sequence": byseq, "teacher_t1_validation": teacher_scores, "teacher_train_holdout_t1_validation": teacher_holdout, "paired_bootstrap": paired_out, "quality_gate": gate, "protocol": protocol}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--mot-train", default="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-09-FRCNN"); p.add_argument("--mot-eval", default="MOT17-05-FRCNN,MOT17-11-FRCNN"); p.add_argument("--max-frames", type=int, default=20); p.add_argument("--max-samples", type=int, default=30000); p.add_argument("--epochs", type=int, default=8); p.add_argument("--seed", type=int, default=17); p.add_argument("--methods", default="raw,history,current_jepa,current_only,full_mask,no_teacher,random_frozen,both_clean"); p.add_argument("--out-dir", default="experiments/results/mot_filter_jepa_probe_mot20"); args = p.parse_args(); base.seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True); payload = {"schema": "mot-filter-jepa-probe-v1", "command": " ".join(sys.argv), "cwd": os.getcwd(), "generated_unix": time.time(), "script_sha256": sha256_file(Path(__file__)), "base_script_sha256": sha256_file(HERE / "current_pair_jepa_probe.py"), "seed": args.seed, "max_frames": args.max_frames, "methods": args.methods.split(",")}
    try:
        payload["mot17"] = run(args)
    except Exception as exc:
        failure = {"status": "failed", "error": type(exc).__name__ + ": " + str(exc), "script_sha256": payload["script_sha256"], "command": payload["command"]}; (out / "failure.json").write_text(json.dumps(failure, indent=2) + "\n"); raise
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n"); print(json.dumps(payload["mot17"]["validation"], indent=2, sort_keys=True, default=float))


if __name__ == "__main__": main()
