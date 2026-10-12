#!/usr/bin/env python3
"""MOT-specific JEPA adapter (train/evaluate implementation).

This is an isolated experiment; it does not edit HybridSORT/OC-SORT.  A
frozen public GOT-JEPA ToMP checkpoint supplies DINO/head crop maps and the
serialized ``filter_predictor -> JEPA_cls`` teacher.  A small student is
trained only on MOT17 train sequences 02/04/09, with detector-history
corruption, next-frame contrastive supervision, and visibility/uncertainty
heads.  Online evaluation uses only OC-SORT detector rows and a strict
one-frame lag.
"""
from __future__ import annotations

import argparse
import csv
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
# The serialized GOT-JEPA archive refers to ``ltr.*`` classes during
# unpickling.  Add the vendored pytracking root before torch.load; this is a
# read-only import-path fix and does not alter the production tracker.
PYTRACKING_ROOT = HERE.parent / "pytracking"
if str(PYTRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTRACKING_ROOT))
# CoTracker is vendored as a nested package in the public GOT tree rather
# than installed site-wide; expose its package root for checkpoint
# reconstruction.
COTRACKER_ROOT = PYTRACKING_ROOT / "ltr" / "cotracker2" / "co_tracker"
if str(COTRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_ROOT))
import mot_ocsort_jepa as base  # noqa: E402
import mot_ocsort_jepa_official as official_api  # noqa: E402

CHECKPOINT_DEFAULT = "/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar"
FIT_SEQUENCES = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN")
HELDOUT_SEQUENCES = ("MOT17-05-FRCNN", "MOT17-11-FRCNN")
MAX_FRAMES_DEFAULT = 120
HISTORY_K = 4
FEATURE_DIM = 256
TRAIN_SEED = 37
TEACHER_NEG_BETA = 0.25
REPAIR_GAP_MIN, REPAIR_GAP_MAX = 2, 8
REPAIR_IOU_MIN, REPAIR_MARGIN_MIN, REPAIR_SCORE_MIN = 0.05, 0.03, 0.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_json(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def norm_rows(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


def pooled_map(m: np.ndarray) -> np.ndarray:
    return norm(np.asarray(m, np.float32).mean(axis=(-1, -2)))


def seq_dims(seq: str) -> Tuple[float, float]:
    info = {}
    for line in (base.MOT_ROOT / "train" / seq / "seqinfo.ini").read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1); info[k.strip().lower()] = v.strip()
    return float(info.get("imwidth", 1920)), float(info.get("imheight", 1080))


@dataclass
class PreparedSequence:
    name: str
    det: Dict[int, List[base.Det]]
    raw: Dict[int, List[dict]]
    maps: Dict[int, np.ndarray]
    gt: Dict[int, List[base.GT]]
    gt_by_gid: Dict[int, Dict[int, int]]
    gt_by_det: Dict[int, Dict[int, int]]
    width: float
    height: float
    metadata: Dict[str, object]


def match_gt_one_to_one(det: Dict[int, List[base.Det]], gt: Dict[int, List[base.GT]],
                        iou_threshold: float = 0.30):
    """Frame-local Hungarian detector↔GT pairing used only offline for labels."""
    by_gid: Dict[int, Dict[int, int]] = {}; by_det: Dict[int, Dict[int, int]] = {}
    stats = {"frames": 0, "gt_rows": 0, "det_rows": 0, "matched": 0,
             "unmatched_gt": 0, "duplicate_assignment_count": 0, "iou_values": []}
    for fr in sorted(gt):
        gs, ds = gt.get(fr, []), det.get(fr, []); stats["frames"] += 1
        stats["gt_rows"] += len(gs); stats["det_rows"] += len(ds)
        if not gs or not ds:
            stats["unmatched_gt"] += len(gs); continue
        mat = np.zeros((len(gs), len(ds)), np.float32)
        for i, g in enumerate(gs):
            for j, d in enumerate(ds): mat[i, j] = base.iou(g.box, d.box)
        rr, cc = linear_sum_assignment(-mat)
        used_d = set()
        for i, j in zip(rr, cc):
            q = float(mat[i, j])
            if q < iou_threshold: continue
            if int(j) in used_d:
                stats["duplicate_assignment_count"] += 1; continue
            used_d.add(int(j)); gid = int(gs[i].gid)
            by_gid.setdefault(gid, {})[int(fr)] = int(j)
            by_det.setdefault(int(fr), {})[int(j)] = gid
            stats["matched"] += 1; stats["iou_values"].append(q)
        stats["unmatched_gt"] += len(gs) - sum(1 for i, j in zip(rr, cc) if mat[i, j] >= iou_threshold)
    if stats["iou_values"]:
        stats["iou_mean"] = float(np.mean(stats["iou_values"])); stats["iou_p10"] = float(np.percentile(stats["iou_values"], 10))
    else:
        stats["iou_mean"] = None; stats["iou_p10"] = None
    stats.pop("iou_values", None)
    return by_gid, by_det, stats


def prepare_sequence(seq: str, max_frames: int, extractor: official_api.OfficialLatent,
                     cache_dir: Path, read_gt: bool = True) -> PreparedSequence:
    sd = base.MOT_ROOT / "train" / seq
    det = base.parse_det(sd / "det" / "det.txt", max_frames)
    maps, cache_meta = extractor.extract_sequence(seq, det, max_frames, cache_dir)
    params = {"track_thresh": .2, "iou_threshold": .3, "max_age": 30,
              "min_hits": 1, "delta_t": 3, "inertia": .2}
    raw, ocsort_seconds = base.run_ocsort_rows(seq, max_frames, det, params)
    official_api.attach_maps(raw, maps, det)
    gt = base.parse_gt(sd / "gt" / "gt.txt", max_frames) if read_gt else {}
    gid_map, det_map, pair_stats = match_gt_one_to_one(det, gt) if read_gt else ({}, {}, {"gt_read": False})
    w, h = seq_dims(seq)
    meta = {"cache": cache_meta, "ocsort_seconds": ocsort_seconds,
            "det_sha256": sha256(sd / "det" / "det.txt"), "gt_sha256": (sha256(sd / "gt" / "gt.txt") if read_gt else None),
            "gt_read": bool(read_gt),
            "pair_stats": pair_stats, "det_rows": int(sum(len(x) for x in det.values()))}
    return PreparedSequence(seq, det, raw, maps, gt, gid_map, det_map, w, h, meta)


def row_for_det(rows: List[dict], dets: List[base.Det], j: int) -> Optional[dict]:
    if not rows or j < 0 or j >= len(dets): return None
    return max(rows, key=lambda r: base.iou(r["box"], dets[j].box))


def assign_raw_rows_one_to_one(raw: Dict[int, List[dict]], det: Dict[int, List[base.Det]],
                               iou_threshold: float = .20) -> Tuple[Dict[int, Dict[int, dict]], Dict[str, int]]:
    """Map detector indices to OC-SORT rows once per frame via Hungarian.

    ``attach_maps`` uses a nearest detector lookup for every emitted OC-SORT
    row, so a naïve reverse lookup can assign the same row to several GT
    labels.  A frame-local one-to-one map removes that bookkeeping artifact;
    the map is reused by all temporal windows, while GT still only selects
    detector indices offline.
    """
    out: Dict[int, Dict[int, dict]] = {}
    stats = {"frames": 0, "det_rows": 0, "raw_rows": 0, "matched": 0,
             "unmatched_detector": 0, "duplicate_raw_assignments": 0}
    for fr in sorted(det):
        ds, rs = det.get(fr, []), raw.get(fr, []); stats["frames"] += 1
        stats["det_rows"] += len(ds); stats["raw_rows"] += len(rs)
        if not ds or not rs:
            stats["unmatched_detector"] += len(ds); continue
        mat = np.zeros((len(ds), len(rs)), np.float32)
        for i, d in enumerate(ds):
            for j, r in enumerate(rs):
                mat[i, j] = base.iou(d.box, r["box"])
        rr, cc = linear_sum_assignment(-mat); used = set()
        for i, j in zip(rr, cc):
            if float(mat[i, j]) < iou_threshold:
                continue
            if int(j) in used:
                stats["duplicate_raw_assignments"] += 1; continue
            used.add(int(j)); out.setdefault(fr, {})[int(i)] = rs[int(j)]; stats["matched"] += 1
        stats["unmatched_detector"] += len(ds) - len(out.get(fr, {}))
    return out, stats


def geometry(box: np.ndarray, width: float, height: float, velocity: np.ndarray,
             age: int, gap: int, score: float, missing: float) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box); bw, bh = max(1., x2-x1), max(1., y2-y1)
    return np.asarray([(x1+x2)/(2*max(width,1.)), (y1+y2)/(2*max(height,1.)),
                       bw/max(width,1.), bh/max(height,1.), velocity[0]/max(width,1.),
                       velocity[1]/max(height,1.), age/30., gap/30., score, missing], np.float32)


def feature_input(history: Sequence[Tuple[int, np.ndarray, np.ndarray, float]], current_map: np.ndarray,
                  box: np.ndarray, width: float, height: float, frame: int,
                  score: float, missing: float = 0.) -> np.ndarray:
    """Observable state token: K pooled history maps + current map + MOT state."""
    hist = list(history)[-HISTORY_K:]; hv = [pooled_map(x[1]) for x in hist]
    h = np.zeros((HISTORY_K, FEATURE_DIM), np.float32)
    if hv: h[-len(hv):] = np.stack(hv)
    cur = pooled_map(current_map)
    if hist:
        old_box = hist[-1][2]; old_c = np.asarray([(old_box[0]+old_box[2])/2, (old_box[1]+old_box[3])/2])
        new_c = np.asarray([(box[0]+box[2])/2, (box[1]+box[3])/2]); vel = new_c-old_c
        age = len(hist); gap = max(1, frame-int(hist[-1][0]))
    else:
        vel = np.zeros(2, np.float32); age = 0; gap = 1
    g = geometry(box, width, height, vel, age, gap, score, missing)
    return np.concatenate([h.reshape(-1), cur, g]).astype(np.float32)


@dataclass
class Sample:
    seq: str
    frame: int
    gid: int
    x: np.ndarray
    context_map: np.ndarray
    current_map: np.ndarray
    positive_map: np.ndarray
    negative_vecs: np.ndarray
    candidate_maps: List[np.ndarray]
    positive_index: int
    history_max_frame: int
    visible_next: float = 1.0
    teacher: Optional[np.ndarray] = None


def build_samples(prepared: Sequence[PreparedSequence], store_candidate_maps: bool = True) -> Tuple[List[Sample], Dict[str, object]]:
    """Build labelled windows while history itself comes only from OC-SORT memory."""
    samples: List[Sample] = []; stats = {"sequences": {}, "history_future_violations": 0,
                                           "empty_history": 0, "invalid_windows": 0,
                                           "duplicate_raw_row_labels": 0}
    for p in prepared:
        histories: Dict[int, List[Tuple[int,np.ndarray,np.ndarray,float]]] = {}
        raw_map, raw_map_stats = assign_raw_rows_one_to_one(p.raw, p.det)
        stats.setdefault("raw_assignment", {})[p.name] = raw_map_stats
        stats["duplicate_raw_row_labels"] += int(raw_map_stats.get("duplicate_raw_assignments", 0))
        seq_count = 0
        for fr in range(1, max(int(max(p.det) if p.det else 0), 0)+1):
            rows, dets = p.raw.get(fr, []), p.det.get(fr, [])
            # Label construction is frame-local and one-to-one; no GT enters
            # the observable history or student input.
            current_pairs = p.gt_by_det.get(fr, {})
            next_pairs = p.gt_by_det.get(fr+1, {})
            for dj, gid in current_pairs.items():
                if gid not in p.gt_by_gid or fr+1 not in p.gt_by_gid[gid]: continue
                nj = p.gt_by_gid[gid][fr+1]
                # The detector↔raw map is fixed once per frame and reused by
                # every temporal window.  This permits the same observation
                # to supervise both t-1→t and t→t+1 windows, while retaining
                # one-to-one rows among different GT identities in a frame.
                row = raw_map.get(fr, {}).get(int(dj)); next_row = raw_map.get(fr+1, {}).get(int(nj))
                if row is None or next_row is None: continue
                native = int(row["native_id"]); hist = histories.get(native, [])
                if hist and max(x[0] for x in hist) >= fr: stats["history_future_violations"] += 1
                if not hist: stats["empty_history"] += 1
                context = np.mean(np.stack([x[1] for x in hist[-HISTORY_K:]]), axis=0).astype(np.float32) if hist else np.asarray(row["feat_map"], np.float32)
                # All public detector rows at t+1 are candidates.  The GT gid
                # is used only to mark the positive index in this offline set.
                cand_rows = p.raw.get(fr+1, []); cand_maps = [np.asarray(r["feat_map"], np.float32) for r in cand_rows] if store_candidate_maps else []
                pos_idx = max(range(len(cand_rows)), key=lambda q: base.iou(cand_rows[q]["box"], next_row["box"])) if cand_rows else -1
                if pos_idx < 0 or base.iou(cand_rows[pos_idx]["box"], next_row["box"]) < .30: continue
                # Negative vectors are retained even when full candidate maps
                # are omitted for the fit split; this keeps the teacher target
                # MOT-specific while avoiding one 256x27x27 copy per negative.
                neg = [pooled_map(r["feat_map"]) for q,r in enumerate(cand_rows) if q != pos_idx]
                history_max = max([x[0] for x in hist], default=-1)
                x = feature_input(hist, row["feat_map"], row["box"], p.width, p.height, fr, row["score"], 0.)
                samples.append(Sample(p.name, fr, int(gid), x, context, np.asarray(row["feat_map"], np.float32),
                                      np.asarray(next_row["feat_map"], np.float32), np.asarray(neg, np.float32).reshape(-1,FEATURE_DIM),
                                      cand_maps, pos_idx, history_max))
                seq_count += 1
                # A deterministic train-only occlusion view gives the
                # visibility head a non-constant target.  The clean teacher
                # target remains unchanged; only the student input is masked.
                if seq_count % 5 == 0:
                    xm = x.copy(); xm[HISTORY_K*FEATURE_DIM:HISTORY_K*FEATURE_DIM+FEATURE_DIM] = 0.0
                    xm[-1] = 1.0
                    samples.append(Sample(p.name, fr, int(gid), xm, context,
                                          np.asarray(row["feat_map"], np.float32), np.asarray(next_row["feat_map"], np.float32),
                                          np.asarray(neg, np.float32).reshape(-1,FEATURE_DIM), cand_maps, pos_idx,
                                          history_max, visible_next=0.0))
                    seq_count += 1
            # Accepted detector observations are the only memory update.
            for r in rows:
                histories.setdefault(int(r["native_id"]), []).append((fr, np.asarray(r["feat_map"], np.float32), r["box"].copy(), float(r["score"])))
                histories[int(r["native_id"])] = histories[int(r["native_id"] )][-HISTORY_K:]
        stats["sequences"][p.name] = {"samples": seq_count, "frames": len(p.det)}
    stats["count"] = len(samples)
    return samples, stats


def make_teacher_targets(samples: List[Sample], extractor: official_api.OfficialLatent,
                         batch: int = 8) -> Dict[str, object]:
    """Frozen official filter teacher with stop-gradient and hard-negative correction."""
    norms = []; t0 = time.perf_counter()
    for st in range(0, len(samples), batch):
        chunk = samples[st:st+batch]
        raw = extractor.predict_filter_batch([s.context_map for s in chunk], [s.current_map for s in chunk])
        for s, omega in zip(chunk, raw):
            o = norm(omega)
            neg = np.mean(s.negative_vecs, axis=0) if len(s.negative_vecs) else np.zeros(FEATURE_DIM, np.float32)
            # MOT-specific association target: suppress same-frame hard
            # negative direction while retaining the official filter scale.
            # ``o`` is already a unit direction.  Keep the hard-negative
            # correction as a scalar beta times a unit negative direction;
            # multiplying two vectors here would silently apply an
            # element-wise gate and make the target definition needlessly
            # dimension-dependent.
            s.teacher = norm(o - TEACHER_NEG_BETA * norm(neg))
            norms.append(float(np.linalg.norm(omega)))
    if not samples: raise RuntimeError("no teacher samples")
    return {"samples": len(samples), "batch": batch, "negative_beta": TEACHER_NEG_BETA,
            "teacher_filter_norm_mean": float(np.mean(norms)), "teacher_filter_norm_p10": float(np.percentile(norms,10)),
            "seconds": time.perf_counter()-t0, "finite": bool(all(np.isfinite(s.teacher).all() for s in samples))}


def _sample_vec(value: np.ndarray) -> np.ndarray:
    """Return a unit 256-D vector from either a crop map or a compact vector."""
    a = np.asarray(value, np.float32)
    if a.ndim == 1:
        return norm(a)
    return pooled_map(a)


def compact_samples(samples: List[Sample], keep_candidate_maps: bool) -> None:
    """Drop large spatial maps after teacher construction when not needed.

    A 256x27x27 float32 map is ~0.75 MB.  Keeping one copy for every fit
    sample makes a 120-frame MOT prefix unnecessarily memory hungry.  The
    student only needs pooled current/positive vectors after the frozen
    teacher target has been materialized; held-out ranking retains candidate
    maps explicitly for its diagnostic score.
    """
    for s in samples:
        s.context_map = _sample_vec(s.context_map)
        s.current_map = _sample_vec(s.current_map)
        s.positive_map = _sample_vec(s.positive_map)
        if not keep_candidate_maps:
            s.candidate_maps = []


def select_quality_samples(samples: Sequence[Sample], max_per_seq: int) -> List[Sample]:
    """Deterministically bound held-out quality memory without label peeking."""
    groups: Dict[str, List[Sample]] = {}
    for s in samples:
        groups.setdefault(s.seq, []).append(s)
    out: List[Sample] = []
    for seq in sorted(groups):
        vals = groups[seq]
        if max_per_seq > 0 and len(vals) > max_per_seq:
            # Uniform temporal subsampling is fixed before quality numbers are
            # inspected; it is not an eval-driven model/threshold choice.
            idx = np.linspace(0, len(vals)-1, max_per_seq, dtype=int)
            vals = [vals[int(i)] for i in idx]
        out.extend(vals)
    return out


class StudentNet:
    """Tiny state-conditioned filter predictor with visibility/uncertainty heads."""
    def __init__(self, input_dim: int, seed: int = TRAIN_SEED):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        class Net(nn.Module):
            def __init__(self, d):
                super().__init__(); self.trunk = nn.Sequential(nn.Linear(d,512), nn.LayerNorm(512), nn.GELU())
                self.filter_head = nn.Linear(512, FEATURE_DIM); self.visibility_head = nn.Linear(512,1); self.uncertainty_head = nn.Linear(512,1)
            def forward(self, x):
                z=self.trunk(x); return self.filter_head(z), self.visibility_head(z).squeeze(-1), self.uncertainty_head(z).squeeze(-1)
        self.net = Net(input_dim)
        self.input_dim = input_dim; self.seed = seed

    def state_dict(self): return self.net.state_dict()
    def train(self): self.net.train(); return self
    def eval(self): self.net.eval(); return self
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray,float,float]:
        import torch
        self.net.eval()
        with torch.no_grad():
            y,v,u=self.net(torch.from_numpy(np.asarray(x,np.float32)))
            y=torch.nn.functional.normalize(y,dim=-1)
        return y.cpu().numpy(), v.cpu().numpy(), u.cpu().numpy()


def corrupt_batch(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    z = np.asarray(x, np.float32).copy(); kdim=HISTORY_K*FEATURE_DIM
    for i in range(len(z)):
        # History dropout/masking models occlusion; current corruption is
        # student-only and never used to generate the teacher target.
        for k in range(HISTORY_K):
            if rng.rand() < .25: z[i,k*FEATURE_DIM:(k+1)*FEATURE_DIM] = 0.
        if rng.rand() < .20: z[i,kdim:kdim+FEATURE_DIM] = 0.
        z[i,:kdim+FEATURE_DIM] += rng.normal(0., .025, kdim+FEATURE_DIM).astype(np.float32)
        if rng.rand() < .20: z[i,-1] = 1.0  # missing/visibility scalar
    return z


def train_student(samples: List[Sample], target: str, epochs: int = 8, seed: int = TRAIN_SEED) -> Tuple[StudentNet, Dict[str, object]]:
    import torch
    torch.set_num_threads(1); rng=np.random.RandomState(seed)
    if target == "teacher":
        y=np.stack([s.teacher for s in samples]).astype(np.float32)
    elif target == "direct":
        y=np.stack([_sample_vec(s.current_map) for s in samples]).astype(np.float32)
    else: raise ValueError(target)
    x=np.stack([s.x for s in samples]).astype(np.float32); pos=np.stack([_sample_vec(s.positive_map) for s in samples]).astype(np.float32)
    maxn=max([len(s.negative_vecs) for s in samples] + [1]); neg=np.zeros((len(samples),maxn,FEATURE_DIM),np.float32); nmask=np.zeros((len(samples),maxn),np.float32)
    for i,s in enumerate(samples):
        if len(s.negative_vecs): neg[i,:len(s.negative_vecs)]=s.negative_vecs; nmask[i,:len(s.negative_vecs)]=1.
    model=StudentNet(x.shape[1],seed).train(); net=model.net; opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4)
    vis_target=np.asarray([s.visible_next for s in samples],np.float32)
    tx,ty,tp,tn,tm,tv=map(torch.from_numpy,(x,y,pos,neg,nmask,vis_target)); losses=[]; bs=min(128,len(x))
    for ep in range(epochs):
        order=rng.permutation(len(x)); total=0.
        for st in range(0,len(x),bs):
            ix=order[st:st+bs]; xb=torch.from_numpy(corrupt_batch(x[ix],rng)); raw,vlog,ulog=net(xb); pred=torch.nn.functional.normalize(raw,dim=-1); yt=ty[ix]; pp=torch.nn.functional.normalize(tp[ix],dim=-1)
            cos=(pred*yt).sum(1); pos_sim=(pred*pp).sum(1); ng=torch.einsum('bd,bmd->bm',pred,tn[ix]); ng=torch.where(tm[ix]>0,ng,torch.full_like(ng,-1.)); hard=ng.max(1).values
            contrast=torch.nn.functional.softplus(.20-pos_sim+hard); unc=torch.exp(-ulog)*(1-cos)+ulog; vis=torch.nn.functional.binary_cross_entropy_with_logits(vlog,tv[ix])
            loss=(1-cos).mean()+.15*((pred-yt)**2).mean()+.50*contrast.mean()+.10*unc.mean()+.10*vis
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step(); total+=float(loss.detach())*len(ix)
        losses.append(total/len(x))
    model.eval();
    with torch.no_grad(): pred,vl,ul=net(torch.from_numpy(x)); pred=torch.nn.functional.normalize(pred,dim=-1); c=(pred*ty).sum(1)
    stats={"target":target,"samples":len(samples),"epochs":epochs,"first_loss":losses[0],"last_loss":losses[-1],"train_teacher_cos_mean":float(c.mean()),"train_teacher_cos_p10":float(torch.quantile(c,.10)),"train_visibility_mean":float(torch.sigmoid(vl).mean()),"train_visibility_target_mean":float(np.mean(vis_target)),"train_uncertainty_mean":float(ul.mean()),"finite":bool(torch.isfinite(pred).all())}
    return model,stats


def save_student(model: StudentNet, path: Path, metadata: Dict[str, object]) -> str:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"state_dict": model.state_dict(), "input_dim": model.input_dim,
           "seed": model.seed, "metadata": metadata, "schema": "mot-jepa-student-v1"}
    torch.save(obj, str(path))
    return sha256(path)


def load_student(path: Path) -> StudentNet:
    import torch
    obj = torch.load(str(path), map_location="cpu")
    model = StudentNet(int(obj["input_dim"]), int(obj.get("seed", TRAIN_SEED)))
    model.net.load_state_dict(obj["state_dict"], strict=True); model.eval()
    return model


def score_quality(samples: Sequence[Sample], model: StudentNet, no_teacher_model: StudentNet,
                  extractor: official_api.OfficialLatent,
                  max_per_seq: int = 200) -> Dict[str, object]:
    """Held-out t→t+1 ranking gate; labels are used only for this report."""
    groups: Dict[str, List[Sample]] = {}
    for s in samples: groups.setdefault(s.seq, []).append(s)
    chosen=[]
    for seq, vals in sorted(groups.items()):
        if max_per_seq and max_per_seq > 0 and len(vals) > max_per_seq:
            step=max(1, len(vals)//max_per_seq); vals=vals[::step][:max_per_seq]
        chosen.extend(vals)
    rec={"samples":len(chosen),"by_sequence":{},"label_scope":"heldout t+1 only; not used for fitting/online thresholds"}
    all_stats=[]
    for target_name in ("teacher","student","no_teacher"):
        hits=[]; margins=[]; chances=[]; cos=[]; cosine_hits=[]; cosine_margins=[]
        for s in chosen:
            if s.teacher is None or s.positive_index < 0 or not s.candidate_maps: continue
            if target_name=="teacher": filt=s.teacher
            elif target_name=="student": filt=model.predict(s.x[None])[0][0]; cos.append(float(np.dot(norm(filt),s.teacher)))
            else: filt=no_teacher_model.predict(s.x[None])[0][0]
            scores=extractor.classifier_scores(norm(filt),s.candidate_maps)
            if s.positive_index >= len(scores): continue
            pos=float(scores[s.positive_index]); neg=np.delete(scores,s.positive_index)
            margin=pos-float(np.max(neg)) if len(neg) else float("inf")
            hits.append(float(pos >= (float(np.max(neg)) if len(neg) else -np.inf))); margins.append(margin); chances.append(1./len(scores))
            cvec=np.stack([pooled_map(m) for m in s.candidate_maps]); cs=cvec.dot(norm(filt)); cp=float(cs[s.positive_index]); cn=np.delete(cs,s.positive_index)
            cosine_hits.append(float(cp >= (float(np.max(cn)) if len(cn) else -np.inf))); cosine_margins.append(cp-float(np.max(cn)) if len(cn) else float("inf"))
            all_stats.append((s.seq,target_name,float(margin),float(len(scores)),float(pos)))
        rec[target_name]={"n":len(hits),"recall_at1":float(np.mean(hits)) if hits else None,
                          "mean_margin":float(np.mean(margins)) if margins else None,
                          "p10_margin":float(np.percentile(margins,10)) if margins else None,
                          "p25_margin":float(np.percentile(margins,25)) if margins else None,
                          "positive_margin_fraction":float(np.mean(np.asarray(margins)>0)) if margins else None,
                          "chance_mean":float(np.mean(chances)) if chances else None,
                          "delta_chance":(float(np.mean(hits)-np.mean(chances)) if hits else None),
                          "cosine_recall_at1":float(np.mean(cosine_hits)) if cosine_hits else None,
                          "cosine_mean_margin":float(np.mean(cosine_margins)) if cosine_margins else None,
                          "cosine_mean_margin_p25":float(np.percentile(cosine_margins,25)) if cosine_margins else None,
                          "cosine_mean_margin_p10":float(np.percentile(cosine_margins,10)) if cosine_margins else None}
        if target_name=="student": rec[target_name]["student_teacher_cosine_mean"]=float(np.mean(cos)) if cos else None
    for seq in sorted(groups):
        vals=[x for x in all_stats if x[0]==seq]
        rec["by_sequence"][seq]={"n":len(vals),"teacher_margin_mean":float(np.mean([x[2] for x in vals if x[1]=="teacher"])) if any(x[1]=="teacher" for x in vals) else None,
                                 "student_margin_mean":float(np.mean([x[2] for x in vals if x[1]=="student"])) if any(x[1]=="student" for x in vals) else None}
    t=rec.get("teacher",{}); st=rec.get("student",{}); nt=rec.get("no_teacher",{})
    # Pre-registered gate; no eval TrackEval values are consulted here.
    rec["gate"]={"pre_registered":{"teacher_margin_gt":0.,"teacher_delta_chance_gt":.10,"student_margin_gt":0.,"student_delta_chance_gt":.10,"student_teacher_cosine_ge":.50},
                 "status":"pass" if (t.get("mean_margin") is not None and st.get("mean_margin") is not None and t.get("mean_margin",-1)>0 and t.get("delta_chance",-1)>.10 and st.get("mean_margin",-1)>0 and st.get("delta_chance",-1)>.10 and st.get("student_teacher_cosine_mean",-1)>=.50) else "fail",
                 "teacher_status":"pass" if t.get("mean_margin",-1)>0 and t.get("delta_chance",-1)>.10 else "fail",
                 "student_status":"pass" if st.get("mean_margin",-1)>0 and st.get("delta_chance",-1)>.10 and st.get("student_teacher_cosine_mean",-1)>=.50 else "fail",
                 "no_teacher_control":nt}
    return rec


def derive_online_thresholds(quality: Dict[str, object]) -> Dict[str, object]:
    """Derive fixed ambiguity gates from train-heldout quality only.

    The rule is deliberately simple and recorded in the checkpoint metadata:
    use the teacher's held-out cosine-margin 25th percentile, floored at .03
    and capped at .20.  The student never sees eval labels when this value is
    selected.  Visibility/uncertainty gates are fixed safety bounds, not tuned
    on TrackEval.
    """
    t = quality.get("teacher", {}) if isinstance(quality, dict) else {}
    p25 = t.get("cosine_mean_margin_p25")
    # Older quality payloads may not carry the percentile; use the declared
    # fixed floor in that case rather than inspecting eval output.
    try:
        p25f = float(p25)
    except Exception:
        p25f = REPAIR_MARGIN_MIN
    fused_margin = float(min(.20, max(REPAIR_MARGIN_MIN, p25f)))
    return {
        "ambiguity_rule": "native-vs-second fused margin < train-heldout teacher cosine-margin p25; floor=.03 cap=.20",
        "ambiguous_fused_margin": fused_margin,
        "ambiguous_iou_min": REPAIR_IOU_MIN,
        "visibility_min": .25,
        "uncertainty_max": 2.0,
        "appearance_weight_ambiguous": .55,
        "iou_weight_ambiguous": .45,
        "source": "train-heldout only",
    }


@dataclass
class OnlineState:
    cid: int
    native_id: int
    last_frame: int
    box: np.ndarray
    score: float
    history: List[Tuple[int,np.ndarray,np.ndarray,float]] = field(default_factory=list)
    omega: Optional[np.ndarray] = None
    visibility: float = 1.0
    uncertainty: float = 0.0


def _margin(v: np.ndarray, j: int) -> float:
    if len(v)<=1: return float("inf")
    return float(v[j]-np.max(np.delete(v,j)))


def online_track(p: PreparedSequence, extractor: official_api.OfficialLatent,
                 models: Dict[str, Optional[StudentNet]], max_age: int = 30,
                 thresholds: Optional[Dict[str, object]] = None) -> Tuple[Dict[str, Dict[int,List[Tuple[np.ndarray,int,float]]]], Dict[str,dict]]:
    """Run causal OC-SORT identity passes using detector observations only.

    The default ``history``/``trained_adapter`` modes are strict: a native
    OC-SORT continuation is kept whenever its IoU gate passes, and JEPA can
    only repair a short post-gap reappearance.  Modes ending in
    ``_ambiguous`` additionally open a *pre-registered* native continuation
    when the native-vs-second-candidate fused margin is below a threshold
    derived from train-heldout quality.  This is the MOT-specific part: an
    ID switch can occur while both detections are present, so a reappearance-
    only hook would never see the hard association event.

    All filter predictions are generated after frame-t assignment and are
    consumed only for frame ``t+1``.  No GT fields are read here.
    """
    thresholds = dict(thresholds or {})
    amb_margin_thr = float(thresholds.get("ambiguous_fused_margin", REPAIR_MARGIN_MIN))
    amb_iou_min = float(thresholds.get("ambiguous_iou_min", REPAIR_IOU_MIN))
    vis_min = float(thresholds.get("visibility_min", .25))
    unc_max = float(thresholds.get("uncertainty_max", 2.0))
    amb_app_w = float(thresholds.get("appearance_weight_ambiguous", .55))
    amb_iou_w = float(thresholds.get("iou_weight_ambiguous", .45))
    outputs: Dict[str, Dict[int, List[Tuple[np.ndarray, int, float]]]] = {}
    diagnostics: Dict[str, dict] = {}
    nframes = max_frames_for(p)
    for mode, model in models.items():
        if mode == "raw":
            outputs[mode] = {fr:[(r["box"], int(r["native_id"]), float(r["score"])) for r in rs]
                             for fr, rs in p.raw.items()}
            diagnostics[mode] = {"mode":"raw", "eval_gt_read_by_tracker":False,
                                 "association_lag":"detector_rows_only"}
            continue

        ambiguous_branch = mode.endswith("_ambiguous")
        core_mode = mode[:-10] if ambiguous_branch else mode
        states: List[OnlineState] = []
        fresh = 1_000_000
        out: Dict[int,List[Tuple[np.ndarray,int,float]]] = {}
        d = {"mode": mode, "core_mode": core_mode,
             "ambiguous_branch": ambiguous_branch,
             "ambiguous_fused_margin_threshold": amb_margin_thr,
             "visibility_min": vis_min, "uncertainty_max": unc_max,
             "native_continuations": 0, "ambiguous_native_candidates": 0,
             "ambiguous_native_opened": 0, "ambiguous_gate_rejects": 0,
             "eligible_states": 0, "candidate_pairs": 0,
             "accepted_repairs": 0, "accepted_ambiguous": 0,
             "classifier_calls": 0, "association_score_space":"pooled_cosine_plus_official_classifier_diagnostic",
             "filter_updates": 0, "lag":"t->t+1",
             "eval_gt_read_by_tracker": False}

        for fr in range(1, nframes + 1):
            cur = p.raw.get(fr, [])
            # Rows are shared by diagnostic modes; clear only our local output
            # marker so a preceding mode cannot leak an identity.
            for row in cur:
                row.pop("cid_v1", None)
            active = [s for s in states if fr - s.last_frame <= max_age]
            used_s: set = set(); used_r: set = set(); matches: List[Tuple[int,int,str]] = []
            cur_maps = [np.asarray(row["feat_map"], np.float32) for row in cur]

            # Precompute appearance/fused scores for each state.  Cosine on
            # pooled JEPA maps is the stable association score; the serialized
            # classifier is also called and retained as a diagnostic channel,
            # but its absolute scale is not silently mixed into the filter.
            score_cache: Dict[int, np.ndarray] = {}
            iou_cache: Dict[int, np.ndarray] = {}
            for i, s in enumerate(active):
                ious = np.asarray([base.iou(s.box, row["box"]) for row in cur], np.float32)
                iou_cache[i] = ious
                if s.omega is None or not cur:
                    score_cache[i] = np.zeros(len(cur), np.float32)
                    continue
                cvec = np.stack([pooled_map(m) for m in cur_maps])
                app_cos = cvec.dot(norm(s.omega)).astype(np.float32)
                app = app_cos.copy()
                if core_mode not in ("history", "raw"):
                    # This verifies the official filter/classifier path but
                    # avoids its uncalibrated magnitude dominating MOT IoU.
                    cls = extractor.classifier_scores(s.omega, cur_maps)
                    d["classifier_calls"] += 1
                    app = (.65 * app_cos + .35 * np.tanh(cls)).astype(np.float32)
                score_cache[i] = app

            # Native continuation pass.  In the ambiguous branch we inspect
            # the strongest alternative before deciding whether to force it.
            ambiguous_states: set = set()
            native_options: List[Tuple[float,int,int]] = []
            for i, s in enumerate(active):
                choices = [j for j, row in enumerate(cur)
                           if int(row["native_id"]) == int(s.native_id)
                           and iou_cache[i][j] >= .20]
                if not choices:
                    continue
                j = max(choices, key=lambda q: float(iou_cache[i][q]))
                if ambiguous_branch and s.omega is not None and len(cur) > 1 and fr - s.last_frame == 1:
                    vals = score_cache[i]
                    fused = .45 * iou_cache[i] + .55 * ((vals + 1.) / 2.)
                    second = float(np.max(np.delete(fused, j))) if len(fused) > 1 else -np.inf
                    margin = float(fused[j] - second)
                    d["ambiguous_native_candidates"] += 1
                    gate = (float(s.visibility) >= vis_min and abs(float(s.uncertainty)) <= unc_max)
                    if margin < amb_margin_thr and gate:
                        ambiguous_states.add(i); d["ambiguous_native_opened"] += 1
                        continue
                    if margin < amb_margin_thr and not gate:
                        d["ambiguous_gate_rejects"] += 1
                native_options.append((float(iou_cache[i][j]), i, j))
            # Resolve duplicate native labels deterministically by IoU.
            for _, i, j in sorted(native_options, reverse=True):
                if i in used_s or j in used_r:
                    continue
                matches.append((i, j, "native")); used_s.add(i); used_r.add(j)
                d["native_continuations"] += 1

            # Strict mode only opens post-gap states.  Ambiguous mode also
            # includes the explicitly opened one-frame native states.
            rem_s: List[int] = []
            for i, s in enumerate(active):
                if i in used_s or s.omega is None:
                    continue
                gap = fr - s.last_frame
                if i in ambiguous_states or REPAIR_GAP_MIN <= gap <= REPAIR_GAP_MAX:
                    rem_s.append(i)
            rem_r = [j for j in range(len(cur)) if j not in used_r]
            d["eligible_states"] += len(rem_s); d["candidate_pairs"] += len(rem_s) * len(rem_r)
            if rem_s and rem_r:
                fused = np.full((len(rem_s), len(rem_r)), -np.inf, np.float32)
                ious = np.zeros_like(fused); apps = np.zeros_like(fused)
                for a, si in enumerate(rem_s):
                    for b, j in enumerate(rem_r):
                        q = float(iou_cache[si][j]); apv = float(score_cache[si][j])
                        ious[a, b] = q; apps[a, b] = apv
                        if si in ambiguous_states:
                            fused[a, b] = amb_iou_w * q + amb_app_w * ((apv + 1.) / 2.)
                        else:
                            fused[a, b] = .70 * q + .30 * ((apv + 1.) / 2.)
                rr, cc = linear_sum_assignment(-fused)
                for a, b in zip(rr, cc):
                    a, b = int(a), int(b); si, j = rem_s[a], rem_r[b]
                    q, apv = float(ious[a,b]), float(apps[a,b]); row_margin = _margin(fused[a], b); col_margin = _margin(fused[:,b], a)
                    if si in ambiguous_states:
                        # Require a plausible geometric overlap and a
                        # non-adversarial appearance score.  The native-vs-
                        # second ambiguity threshold is the train-heldout
                        # gate; no eval labels enter this decision.
                        ok = q >= amb_iou_min and apv >= -.05 and row_margin >= 0.0 and col_margin >= 0.0
                    else:
                        ok = q >= REPAIR_IOU_MIN and apv >= REPAIR_SCORE_MIN and row_margin >= REPAIR_MARGIN_MIN and col_margin >= REPAIR_MARGIN_MIN
                    if ok:
                        matches.append((si, j, "ambiguous" if si in ambiguous_states else "adapter")); used_s.add(si); used_r.add(j)
                        d["accepted_repairs"] += 1
                        if si in ambiguous_states:
                            d["accepted_ambiguous"] += 1

            # Assignment is complete.  Only now append the accepted detector
            # to state history and build the filter used at frame t+1.
            updated: List[Tuple[OnlineState,np.ndarray,dict]] = []
            for si, rj, _kind in matches:
                s, row = active[si], cur[rj]
                x = feature_input(s.history, row["feat_map"], row["box"], p.width, p.height, fr, row["score"], 0.)
                fm = np.asarray(row["feat_map"], np.float32)
                s.last_frame = fr; s.box = row["box"].copy(); s.native_id = int(row["native_id"]); s.score = float(row["score"])
                s.history = (s.history + [(fr, fm, row["box"].copy(), float(row["score"]))])[-HISTORY_K:]
                row["cid_v1"] = s.cid; updated.append((s, x, row))

            active_cids = {int(s.cid) for s in active if fr - s.last_frame <= max_age}
            for j, row in enumerate(cur):
                if j in used_r:
                    continue
                cid = int(row["native_id"])
                if cid <= 0 or cid in active_cids:
                    while fresh in active_cids:
                        fresh += 1
                    cid = fresh; fresh += 1
                fm = np.asarray(row["feat_map"], np.float32)
                s = OnlineState(cid, int(row["native_id"]), fr, row["box"].copy(), float(row["score"]),
                                [(fr, fm, row["box"].copy(), float(row["score"]))])
                states.append(s); active_cids.add(cid); row["cid_v1"] = cid
                x = feature_input([], row["feat_map"], row["box"], p.width, p.height, fr, row["score"], 0.)
                updated.append((s, x, row))

            if core_mode == "history":
                for s, _, _ in updated:
                    s.omega = pooled_map(s.history[-1][1]); s.visibility = 1.; s.uncertainty = 0.
                    d["filter_updates"] += 1
            elif model is not None and updated:
                pred, vis, unc = model.predict(np.stack([x for _, x, _ in updated]))
                for s, y, v, u in zip([q[0] for q in updated], pred, vis, unc):
                    s.omega = norm(y); s.visibility = float(1. / (1. + np.exp(-float(v)))); s.uncertainty = float(u)
                    d["filter_updates"] += 1
            out[fr] = [(row["box"], int(row.get("cid_v1", row["native_id"])), float(row["score"])) for row in cur]
        outputs[mode] = out; diagnostics[mode] = d

    # Compare every non-raw emission ID to raw by ordered detector row.  The
    # row count/boxes remain fixed, so this isolates identity changes.
    raw_ids = [int(r["native_id"]) for fr in range(1, nframes + 1) for r in p.raw.get(fr, [])]
    for mode in outputs:
        if mode == "raw":
            continue
        ids = [int(t[1]) for fr in range(1, nframes + 1) for t in outputs[mode].get(fr, [])]
        diagnostics[mode]["id_differences_vs_raw"] = int(sum(a != b for a, b in zip(raw_ids, ids)))
    return outputs, diagnostics


def max_frames_for(p: PreparedSequence) -> int:
    return max(p.det.keys()) if p.det else 0


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=float) + "\n")


def write_tracks(path: Path, rows: Dict[int, List[Tuple[np.ndarray, int, float]]]) -> None:
    """Write detector-backed rows in MOTChallenge comma-separated format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for fr in sorted(rows):
            for box, tid, score in rows[fr]:
                x1, y1, x2, y2 = map(float, box)
                f.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},-1,-1,-1\n".format(
                    int(fr), int(tid), x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1), float(score)))


def _copy_run_sources(out: Path) -> None:
    """Place immutable source snapshots beside every experiment artifact."""
    out.mkdir(parents=True, exist_ok=True)
    for name in ("mot_jepa_adapter_v1.py", "train_adapter.py", "run_adapter.py"):
        src = HERE / name
        if src.exists():
            shutil.copy2(src, out / name)


def _prepared_manifest(prepared: Sequence[PreparedSequence]) -> Dict[str, object]:
    return {p.name: {"frames": int(max_frames_for(p)), "det_rows": int(sum(len(v) for v in p.det.values())),
                     "metadata": p.metadata} for p in prepared}


def _safe_torch_version() -> Optional[str]:
    try:
        import torch
        return str(torch.__version__)
    except Exception:
        return None


def train_adapter_main(argv: Optional[Sequence[str]] = None) -> int:
    """Fit the MOT-specific student and emit a train-heldout release gate."""
    ap = argparse.ArgumentParser(description="Train the isolated MOT-JEPA adapter")
    ap.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--fit-sequences", default=",".join(FIT_SEQUENCES))
    ap.add_argument("--heldout-sequences", default=",".join(HELDOUT_SEQUENCES))
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES_DEFAULT)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--teacher-batch", type=int, default=8)
    ap.add_argument("--quality-max-per-seq", type=int, default=120)
    ap.add_argument("--fit-end-frame", type=int, default=0,
                    help="train-fit observation prefix end (inclusive); quality uses the following frames")
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    fit_seqs = tuple(x for x in args.fit_sequences.split(",") if x)
    held_seqs = tuple(x for x in args.heldout_sequences.split(",") if x)
    # Quality must be train-heldout, never the final MOT17-05/11 eval videos.
    # A prefix/suffix split is fixed before any quality values are inspected.
    fit_end = int(args.fit_end_frame) if int(args.fit_end_frame) > 0 else max(2, int(round(args.max_frames * .80)))
    cache = Path(args.cache_dir).resolve() if args.cache_dir else out / "cache"
    payload: Dict[str, object] = {
        "schema": "mot-jepa-adapter-train-v1", "status": "started", "dataset": "MOT17",
        "fit_sequences": list(fit_seqs), "heldout_sequences": list(held_seqs),
        "quality_sequences": list(fit_seqs), "fit_observation_frames": [1, fit_end],
        "quality_observation_frames": [fit_end + 1, int(args.max_frames)],
        "max_frames": int(args.max_frames), "history_k": HISTORY_K, "feature_dim": FEATURE_DIM,
        "seed": int(args.seed), "epochs": int(args.epochs), "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": args.device, "cache_dir": str(cache), "eval_gt_read_by_tracker": False,
        "teacher_filter_output_normalization": "L2-unit direction for target; official classifier raw scale and cosine diagnostics both recorded",
        "student_filter_output_normalization": "L2-unit direction (positive scale invariant for cosine; no eval calibration)",
        "candidate_in_predictor": False, "association_lag": "t->t+1", "gt_history": False,
        "production_tracker_modified": False, "torch": _safe_torch_version(),
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
        "source_sha256_started": sha256(Path(__file__).resolve()),
    }
    t0 = time.perf_counter()
    try:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        extractor = official_api.OfficialLatent(Path(args.checkpoint), args.device, args.teacher_batch)
        payload["official_load"] = extractor.load_meta

        fit_prepared = [prepare_sequence(seq, args.max_frames, extractor, cache, read_gt=True) for seq in fit_seqs]
        # Build once with candidate maps so the same causal histories can be
        # partitioned into fit and train-heldout suffixes.  No eval sequence
        # (MOT17-05/11) is touched in this quality stage.
        all_train_samples, all_train_stats = build_samples(fit_prepared, store_candidate_maps=True)
        fit_samples = [s for s in all_train_samples if int(s.frame) < fit_end]
        held = [s for s in all_train_samples if int(s.frame) >= fit_end]
        fit_stats = dict(all_train_stats); fit_stats["all_count_before_temporal_split"] = len(all_train_samples)
        fit_stats["count"] = len(fit_samples); fit_stats["temporal_split"] = {"fit_frame_t_lt": fit_end, "quality_frame_t_ge": fit_end}
        held_stats = dict(all_train_stats); held_stats["count"] = len(held); held_stats["temporal_split"] = {"fit_frame_t_lt": fit_end, "quality_frame_t_ge": fit_end}
        payload["fit_manifest"] = _prepared_manifest(fit_prepared)
        payload["fit_sample_stats"] = fit_stats
        payload["train_heldout_sample_stats_before_cap"] = held_stats
        if not fit_samples:
            raise RuntimeError("no fit samples")
        teacher_stats = make_teacher_targets(fit_samples, extractor, args.teacher_batch)
        # The fit path no longer needs spatial maps once the frozen teacher
        # target is materialized; this is deliberately before student fitting.
        compact_samples(fit_samples, keep_candidate_maps=False)
        student, student_stats = train_student(fit_samples, "teacher", args.epochs, args.seed)
        direct, direct_stats = train_student(fit_samples, "direct", args.epochs, args.seed + 1)
        ckpt_dir = out / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_meta = {
            "schema": "mot-jepa-adapter-student-v1", "official_checkpoint_sha256": extractor.checkpoint_sha256,
            "fit_sequences": list(fit_seqs), "max_frames": int(args.max_frames), "history_k": HISTORY_K,
            "feature_dim": FEATURE_DIM, "seed": int(args.seed), "source_sha256": sha256(Path(__file__).resolve()),
            "filter_normalization": "L2-unit", "teacher_negative_beta": TEACHER_NEG_BETA,
        }
        student_path = ckpt_dir / "trained_adapter.pt"; direct_path = ckpt_dir / "direct_no_teacher.pt"
        student_sha = save_student(student, student_path, {**model_meta, "target": "frozen_official_filter"})
        direct_sha = save_student(direct, direct_path, {**model_meta, "target": "direct_current_pool"})

        # Held-out quality is independent of eval sequences.  Build labels only
        # for the declared train-heldout split and retain candidates for ranking.
        held = select_quality_samples(held, int(args.quality_max_per_seq))
        held_teacher_stats = make_teacher_targets(held, extractor, args.teacher_batch)
        compact_samples(held, keep_candidate_maps=True)
        quality = score_quality(held, student, direct, extractor, max_per_seq=0)
        thresholds = derive_online_thresholds(quality)
        # Explicitly expose the chosen cosine percentile for auditability.
        quality.setdefault("threshold_derivation", thresholds)
        quality["teacher_target_stats"] = held_teacher_stats
        quality["fit_teacher_target_stats"] = teacher_stats

        data_checks = {
            "history_future_violations": int(fit_stats.get("history_future_violations", 0)),
            "duplicate_raw_row_labels": int(fit_stats.get("duplicate_raw_row_labels", 0)),
            "all_finite": bool(teacher_stats.get("finite") and student_stats.get("finite") and direct_stats.get("finite")),
            "fit_gt_used_only_for_labels": True, "heldout_gt_used_only_for_quality": True,
            "eval_gt_used": False,
        }
        gate = dict(quality.get("gate", {})); gate["data_checks"] = data_checks
        gate["status"] = "pass" if gate.get("status") == "pass" and data_checks["history_future_violations"] == 0 and data_checks["duplicate_raw_row_labels"] == 0 and data_checks["all_finite"] else "fail"
        quality["gate"] = gate
        payload.update({
            "status": "complete", "fit_teacher": teacher_stats, "student": student_stats,
            "direct_no_teacher": direct_stats, "checkpoints": {"trained_adapter": str(student_path), "trained_adapter_sha256": student_sha,
                                                                   "direct_no_teacher": str(direct_path), "direct_no_teacher_sha256": direct_sha},
            "train_heldout_manifest": _prepared_manifest(fit_prepared), "heldout_sample_stats": held_stats,
            "heldout_quality": quality, "online_thresholds": thresholds, "data_checks": data_checks,
            "quality_gate": gate, "teacher_frozen": True, "heldout_teacher_refit": False,
            "fit_split_digest": hash_json({"seqs": fit_seqs, "frames": args.max_frames, "det": [p.metadata.get("det_sha256") for p in fit_prepared]}),
            "heldout_split_digest": hash_json({"seqs": fit_seqs, "frames": [fit_end, args.max_frames], "kind": "train-heldout-suffix"}),
            "eval_sequences_reserved": list(held_seqs),
        })
        _write_json(out / "quality_gate.json", quality)
        _write_json(out / "train_results.json", payload)
        _write_json(out / "audit.json", {
            "schema": "mot-jepa-adapter-audit-v1", "protocol": "frozen_official_filter_student_lagged_v1",
            "source_sha256": sha256(Path(__file__).resolve()), "official_checkpoint_sha256": extractor.checkpoint_sha256,
            "train_fit_split": list(fit_seqs), "train_heldout_split": list(held_seqs), "eval_split": list(held_seqs),
            "lag_frames": 1, "leave_current_out": False, "candidate_in_predictor": False,
            "gt_history": False, "oracle_semantics": None, "quality_gate": gate,
            "filter_output_normalization": payload["teacher_filter_output_normalization"],
            "online_thresholds": thresholds, "checkpoint_sha256": {"trained_adapter": student_sha, "direct_no_teacher": direct_sha},
        })
        payload["elapsed_seconds"] = time.perf_counter() - t0
        _write_json(out / "train_results.json", payload)
        _copy_run_sources(out)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        payload.update({"status": "failure", "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - t0})
        _write_json(out / "failure.json", payload); _write_json(out / "train_results.json", payload); _copy_run_sources(out)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 1


def _load_training_payload(model_dir: Path) -> Dict[str, object]:
    for name in ("train_results.json", "results.json"):
        p = model_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return {}


def run_adapter_main(argv: Optional[Sequence[str]] = None) -> int:
    """Run strict and ambiguity-opened causal branches without reading GT."""
    ap = argparse.ArgumentParser(description="Run the isolated MOT-JEPA adapter")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--sequences", default=",".join(HELDOUT_SEQUENCES))
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES_DEFAULT)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-age", type=int, default=30)
    args = ap.parse_args(argv)
    model_dir = Path(args.model_dir).resolve(); out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    seqs = tuple(x for x in args.sequences.split(",") if x)
    train_payload = _load_training_payload(model_dir)
    ck = train_payload.get("checkpoints", {}) if isinstance(train_payload, dict) else {}
    student_path = Path(ck.get("trained_adapter", model_dir / "checkpoints" / "trained_adapter.pt"))
    direct_path = Path(ck.get("direct_no_teacher", model_dir / "checkpoints" / "direct_no_teacher.pt"))
    thresholds = train_payload.get("online_thresholds", {}) if isinstance(train_payload, dict) else {}
    payload: Dict[str, object] = {
        "schema": "mot-jepa-adapter-run-v1", "status": "started", "dataset": "MOT17",
        "sequences": list(seqs), "max_frames": int(args.max_frames), "model_dir": str(model_dir),
        "checkpoint": str(Path(args.checkpoint).resolve()), "eval_gt_read_by_tracker": False,
        "candidate_in_predictor": False, "association_lag": "t->t+1", "gt_history": False,
        "online_thresholds": thresholds, "methods": ["raw", "history", "no_teacher", "trained_adapter", "trained_adapter_ambiguous", "no_teacher_ambiguous"],
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
        "source_sha256_started": sha256(Path(__file__).resolve()),
    }
    t0 = time.perf_counter()
    try:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        extractor = official_api.OfficialLatent(Path(args.checkpoint), args.device, args.batch)
        student = load_student(student_path); direct = load_student(direct_path)
        payload["official_load"] = extractor.load_meta
        payload["model_sha256"] = {"trained_adapter": sha256(student_path), "direct_no_teacher": sha256(direct_path)}
        cache = Path(args.cache_dir).resolve() if args.cache_dir else out / "cache"
        all_rows: List[Dict[str, object]] = []; all_diag: Dict[str, object] = {}; manifests: Dict[str, object] = {}
        methods = {"raw": None, "history": None, "no_teacher": direct, "trained_adapter": student,
                   "trained_adapter_ambiguous": student, "no_teacher_ambiguous": direct}
        for seq in seqs:
            p = prepare_sequence(seq, args.max_frames, extractor, cache, read_gt=False)
            # Hard assertion: eval preparation must not contain labels.  This
            # catches accidental GT reads before any tracker loop runs.
            if p.metadata.get("gt_read") is not False or p.gt or p.gt_by_gid or p.gt_by_det:
                raise AssertionError("eval sequence contains GT state")
            manifests[seq] = p.metadata
            outputs, diagnostics = online_track(p, extractor, methods, max_age=args.max_age, thresholds=thresholds)
            all_diag[seq] = diagnostics
            for method, rows in outputs.items():
                path = out / "mot17" / "tracks" / method / (seq + ".txt")
                write_tracks(path, rows)
                all_rows.append({"sequence": seq, "method": method, "frames": int(args.max_frames),
                                 "track_rows": int(sum(len(v) for v in rows.values())), "track_sha256": sha256(path),
                                 "id_differences_vs_raw": diagnostics.get(method, {}).get("id_differences_vs_raw", 0)})
        payload.update({"status": "complete", "feature_cache": str(cache), "manifests": manifests,
                        "rows": all_rows, "diagnostics": all_diag, "elapsed_seconds": time.perf_counter()-t0,
                        "source_sha256": sha256(Path(__file__).resolve()), "production_tracker_modified": False})
        _write_json(out / "results.json", payload)
        _write_json(out / "reflection.json", {
            "schema": "mot-jepa-adapter-reflection-v1", "status": "complete", "result_path": str(out / "results.json"),
            "finding": "Strict and train-heldout-thresholded ambiguous native-continuation branches emitted; run official TrackEval before claims.",
            "eval_gt_read_by_tracker": False, "candidate_in_predictor": False,
            "next_step": "Audit all methods with vendored official TrackEval and compare HOTA/IDF1/AssA/DetA/MOTA/IDSW against raw.",
            "threshold_source": "train-heldout only", "model_sha256": payload.get("model_sha256"),
        })
        _write_json(out / "audit.json", {
            "schema": "mot-jepa-adapter-audit-v1", "protocol": "frozen_official_filter_student_lagged_v1",
            "source_sha256": payload["source_sha256"], "official_checkpoint_sha256": extractor.checkpoint_sha256,
            "eval_split": list(seqs), "eval_gt_read": False, "gt_history": False, "lag_frames": 1,
            "quality_thresholds": thresholds, "model_sha256": payload["model_sha256"],
            "all_eval_prepared_without_gt": True, "diagnostics": all_diag,
        })
        _copy_run_sources(out)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        payload.update({"status": "failure", "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter()-t0})
        _write_json(out / "failure.json", payload); _write_json(out / "results.json", payload); _copy_run_sources(out)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["train", "run"])
    # Preserve a useful direct entry point while wrappers expose explicit CLIs.
    args, rest = ap.parse_known_args()
    return train_adapter_main(rest) if args.phase == "train" else run_adapter_main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
