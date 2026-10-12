#!/usr/bin/env python3
"""OC-SORT + a causal MOT-specific JEPA association adapter.

This file is deliberately self contained and does not modify HybridSORT.  A
fixed public detector stream is first processed by the local OC-SORT tracker.
The JEPA adapter then performs an online identity association pass over those
active rows.  At time ``t`` it predicts a per-track association filter
``omega_t`` from observations strictly before/at ``t``; the filter is stored
and scored only against rows at ``t+1``.  Thus the predictor never sees the
next-frame candidate set or evaluation GT.

The initial implementation uses a compact, generic crop descriptor (spatial
RGB/moments) so the strict branch has no MOT-trained ReID dependency.  A
FastReID cache can be supplied for an explicitly labelled upper-bound run.
The teacher target is a positive-vs-hard-negative association filter rather
than a clean embedding: omega = positive future descriptor minus the
same-frame hard-negative mean.  Student training uses train-sequence GT only,
with current/history corruption, and keeps raw/history/no-teacher/random
controls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
HYBRID_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT")
MOT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17")
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))
from trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a[:4]); bx1, by1, bx2, by2 = map(float, b[:4])
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0., ix2 - ix1), max(0., iy2 - iy1)
    inter = iw * ih
    aa = max(0., ax2-ax1) * max(0., ay2-ay1)
    bb = max(0., bx2-bx1) * max(0., by2-by1)
    return inter / max(aa + bb - inter, 1e-8)


def crop_feature(image: np.ndarray, box: Sequence[float], dim: int = 80) -> np.ndarray:
    """Generic frozen descriptor; no dataset labels or GT are consulted."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(float, box[:4]); bw, bh = max(2., x2-x1), max(2., y2-y1)
    x1 = int(max(0, math.floor(x1 - .08*bw))); y1 = int(max(0, math.floor(y1 - .08*bh)))
    x2 = int(min(w, math.ceil(x2 + .08*bw))); y2 = int(min(h, math.ceil(y2 + .08*bh)))
    if x2 <= x1 or y2 <= y1: return np.zeros(dim, np.float32)
    crop = image[y1:y2, x1:x2]
    small = cv2.resize(crop, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    rgb = small[:, :, ::-1].reshape(-1)
    mean = crop.reshape(-1, 3).mean(0).astype(np.float32) / 255.
    std = crop.reshape(-1, 3).std(0).astype(np.float32) / 255.
    f = np.concatenate([rgb, mean, std, np.asarray([crop.mean()/255., crop.std()/255.], np.float32)])
    if f.size < dim: f = np.pad(f, (0, dim-f.size))
    return norm(f[:dim])


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


def parse_det(path: Path, max_frames: int) -> Dict[int, List[Det]]:
    out: Dict[int, List[Det]] = {}
    with path.open("r", errors="replace") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6: continue
            try:
                fr = int(float(p[0]));
                if fr < 1 or (max_frames and fr > max_frames): continue
                x,y,w,h = [float(v) for v in p[2:6]]; s = float(p[6]) if len(p)>6 else 1.
            except Exception: continue
            out.setdefault(fr, []).append(Det(fr, np.asarray([x,y,x+max(0.,w),y+max(0.,h)], np.float32), s))
    return out


def parse_gt(path: Path, max_frames: int) -> Dict[int, List[GT]]:
    out: Dict[int, List[GT]] = {}
    if not path.exists(): return out
    with path.open("r", errors="replace") as f:
        for line in f:
            p=line.strip().split(",")
            if len(p)<6: continue
            try:
                fr=int(float(p[0])); gid=int(float(p[1]));
                if fr<1 or (max_frames and fr>max_frames): continue
                if len(p)>6 and float(p[6])<=0: continue
                if len(p)>7 and int(float(p[7])) not in (1,): continue
                x,y,w,h=[float(v) for v in p[2:6]]
            except Exception: continue
            out.setdefault(fr, []).append(GT(fr,gid,np.asarray([x,y,x+max(0.,w),y+max(0.,h)],np.float32)))
    return out


def load_features(seq: str, det: Dict[int,List[Det]], source: str, cache_dir: Optional[Path]) -> Dict[int,np.ndarray]:
    if source == "fastreid":
        if cache_dir is None: raise ValueError("--feature-cache required for fastreid")
        p = cache_dir / (seq + ".npz")
        z = np.load(p, allow_pickle=True)["features"]
        out = {fr: np.asarray(z[fr], np.float32) for fr in range(1, len(z))}
        for fr, rows in det.items():
            if out.get(fr, np.empty((0,))).shape[0] != len(rows):
                raise ValueError("feature/detection row mismatch {} frame {}".format(seq,fr))
            for d, f in zip(rows, out[fr]): d.feat = norm(f)
        return out
    # Strict generic descriptor, image-only.
    seq_dir = MOT_ROOT / "train" / seq
    out = {}
    for fr, rows in det.items():
        im = cv2.imread(str(seq_dir / "img1" / ("{:06d}.jpg".format(fr))))
        if im is None: raise FileNotFoundError(seq_dir / "img1" / ("{:06d}.jpg".format(fr)))
        fs = np.stack([crop_feature(im,d.box) for d in rows], axis=0) if rows else np.empty((0,80),np.float32)
        out[fr] = fs
        for d, f in zip(rows, fs): d.feat = f
    return out


def best_det(rows: List[Det], box: np.ndarray, thr: float=.30) -> Optional[Det]:
    best=None; bi=0.
    for d in rows:
        q=iou(d.box,box)
        if q>bi: bi,best=q,d
    return best if bi>=thr else None


def project_features(feats: Dict[int,np.ndarray], dim: int, seed: int) -> Tuple[Dict[int,np.ndarray], np.ndarray]:
    src_dim = next((x.shape[1] for x in feats.values() if x.ndim==2 and x.shape[0]), 80)
    rng=np.random.RandomState(seed+7001)
    # Fixed normalized random projection keeps the strict branch generic and
    # avoids fitting any representation on evaluation identities.
    R=rng.normal(size=(src_dim,dim)).astype(np.float32)
    R/=np.maximum(np.linalg.norm(R,axis=0,keepdims=True),1e-8)
    out={fr: np.asarray([norm(x.dot(R)) for x in arr],np.float32) if len(arr) else np.empty((0,dim),np.float32)
         for fr,arr in feats.items()}
    return out,R


def geom(box: np.ndarray, iw: float, ih: float, age: float, dt: float, vel: np.ndarray, score: float, missing: float) -> np.ndarray:
    x1,y1,x2,y2=map(float,box); w=max(1.,x2-x1); h=max(1.,y2-y1)
    return np.asarray([(x1+x2)/2/max(iw,1.),(y1+y2)/2/max(ih,1.),w/max(iw,1.),h/max(ih,1.),
                       float(vel[0])/max(iw,1.),float(vel[1])/max(ih,1.),age/30.,dt/30.,score,missing],np.float32)


class FilterJEPA:
    """Residual student for a per-track association filter omega."""
    def __init__(self, dim: int, k: int, seed: int, epochs: int=8, mode: str="jepa", device: str="cpu"):
        self.dim,self.k,self.seed,self.epochs,self.mode,self.device_name=dim,k,seed,epochs,mode,device
        self.params=None; self.stats={}; self.alpha=.35
    @property
    def input_dim(self): return self.k*self.dim + self.dim + 10
    def _forward_np(self,x):
        w1,b1,g,bg,w2,b2=self.params
        z=x.dot(w1.T)+b1; mu=z.mean(1,keepdims=True); var=((z-mu)**2).mean(1,keepdims=True)
        z=(z-mu)/np.sqrt(var+1e-5)*g+bg
        z=.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
        return z.dot(w2.T)+b2
    def fit(self, samples: Dict[str,np.ndarray]):
        import torch, torch.nn as nn
        torch.set_num_threads(1); torch.manual_seed(self.seed)
        x=np.asarray(samples["x"],np.float32); y=np.asarray(samples["y"],np.float32); neg=np.asarray(samples["neg"],np.float32)
        if len(x)==0: raise RuntimeError("no JEPA samples")
        class Net(nn.Module):
            def __init__(self, inp, out):
                super().__init__(); self.net=nn.Sequential(nn.Linear(inp,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,out))
            def forward(self,z): return self.net(z)
        net=Net(self.input_dim,self.dim)
        tx=torch.from_numpy(x); ty=torch.from_numpy(y); tn=torch.from_numpy(neg)
        if self.mode=="no_teacher":
            g=torch.Generator().manual_seed(self.seed+9901); ty=ty[torch.randperm(len(ty),generator=g)]
        opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=1e-4); rng=np.random.RandomState(self.seed)
        losses=[]; net.train(); bs=min(512,len(x))
        for ep in range(self.epochs):
            order=rng.permutation(len(x)); total=0.
            for st in range(0,len(x),bs):
                ix=order[st:st+bs]; xb,yb,nb=tx[ix],ty[ix],tn[ix]
                raw=net(xb); cur=xb[:,self.k*self.dim:self.k*self.dim+self.dim]
                pred=nn.functional.normalize(cur+self.alpha*torch.tanh(raw),dim=1)
                yt=nn.functional.normalize(yb,dim=1); nt=nn.functional.normalize(nb,dim=1)
                pos=(pred*yt).sum(1); ng=(pred*nt).sum(1)
                loss=(1-pos).mean()+.7*nn.functional.softplus(.10-pos+ng).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
                total += float(loss.detach())*len(ix)
            losses.append(total/len(x))
        net.eval(); sd=net.state_dict(); self.params=tuple(sd[k].detach().numpy().astype(np.float32) for k in ("net.0.weight","net.0.bias","net.1.weight","net.1.bias","net.3.weight","net.3.bias"))
        p=self.predict_batch(x); pos=np.sum(p*norm_rows(y),1); ng=np.sum(p*norm_rows(neg),1)
        self.stats={"samples":len(x),"epochs":self.epochs,"first_loss":losses[0],"last_loss":losses[-1],
                    "target_cosine_mean":float(pos.mean()),"margin_mean":float((pos-ng).mean()),
                    "margin_p10":float(np.percentile(pos-ng,10)),"finite":bool(np.isfinite(p).all())}
    def fit_random(self):
        rng=np.random.RandomState(self.seed+404); self.params=(rng.normal(0,.05,(128,self.input_dim)).astype(np.float32),np.zeros(128,np.float32),np.ones(128,np.float32),np.zeros(128,np.float32),rng.normal(0,.05,(self.dim,128)).astype(np.float32),np.zeros(self.dim,np.float32)); self.stats={"random_frozen":True,"samples":0}
    def predict_batch(self,x):
        if self.params is None: return norm_rows(x[:,-self.dim:])
        cur=x[:,self.k*self.dim:self.k*self.dim+self.dim]
        raw=self._forward_np(x); return norm_rows(cur+self.alpha*np.tanh(raw))
    def predict(self,hist: np.ndarray, cur: np.ndarray, g: np.ndarray)->np.ndarray:
        h=np.zeros((self.k,self.dim),np.float32); hh=np.asarray(hist,np.float32)[-self.k:]
        if len(hh): h[-len(hh):]=hh
        x=np.concatenate([h.reshape(-1),norm(cur),g]).astype(np.float32)[None]
        return self.predict_batch(x)[0]


def norm_rows(x):
    x=np.asarray(x,np.float32); return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def build_samples(seqs: Sequence[str], max_frames: int, dim: int, k: int, seed: int, feature_source: str, cache_dir: Optional[Path]):
    xs=[]; ys=[]; ns=[]; meta=[]; rng=np.random.RandomState(seed+13)
    for seq in seqs:
        sd=MOT_ROOT/"train"/seq; det=parse_det(sd/"det/det.txt",max_frames); gt=parse_gt(sd/"gt/gt.txt",max_frames)
        raw=load_features(seq,det,feature_source,cache_dir); feats,R=project_features(raw,dim,seed)
        # Build per-ID matched observations.
        byid={}
        for fr,gs in gt.items():
            for g in gs:
                d=best_det(det.get(fr,[]),g.box,.30)
                if d is None: continue
                # Dataclass equality on ``np.ndarray`` boxes is ambiguous;
                # recover the selected row by identity/IoU instead of list.index.
                idx=max(range(len(det[fr])), key=lambda ii: iou(det[fr][ii].box, g.box))
                byid.setdefault(g.gid,[]).append((fr,feats[fr][idx],d.box,d.score))
        for gid,obs in byid.items():
            obs=sorted(obs)
            for j,(fr,cur,box,score) in enumerate(obs[:-1]):
                nxt=obs[j+1]
                if nxt[0] != fr+1: continue
                hist=[o[1] for o in obs[:j][-k:]]
                if not hist: continue
                # Same-frame hard negative: closest descriptor among other GT
                # detections at t+1, falling back to a deterministic zero.
                negs=[]
                for ogid,oobs in byid.items():
                    if ogid==gid: continue
                    for ofr,of,ob,oscore in oobs:
                        if ofr==nxt[0]: negs.append(of); break
                if negs:
                    na=np.asarray(negs); ni=int(np.argmax(na.dot(nxt[1]))); neg=na[ni]
                else: neg=np.zeros(dim,np.float32)
                # Teacher target is the clean positive-vs-negative association
                # filter for the next frame, not merely the positive embedding.
                y=norm(nxt[1]-neg)
                h=np.zeros((k,dim),np.float32); h[-len(hist):]=hist
                g=geom(box,1920.,1080.,0.,1.,np.zeros(2),score,0.)
                c=cur.copy(); hh=h.copy()
                # Training corruption models occlusion/missing observations;
                # deployment remains clean detector features.
                if rng.rand()<.7: c*=float(rng.rand()>.12)
                c += rng.normal(0,.03,dim).astype(np.float32)
                if rng.rand()<.35: hh[rng.randint(0,k)] = 0.
                xs.append(np.concatenate([hh.reshape(-1),norm(c),g])); ys.append(y); ns.append(neg); meta.append((seq,fr,gid))
    if not xs: raise RuntimeError("no train samples")
    return {"x":np.asarray(xs,np.float32),"y":np.asarray(ys,np.float32),"neg":np.asarray(ns,np.float32),"meta":meta}


@dataclass
class State:
    cid: int
    last_frame: int
    box: np.ndarray
    feat_hist: List[np.ndarray] = field(default_factory=list)
    omega: Optional[np.ndarray] = None
    native_id: int = -1
    score: float = 1.
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2,np.float32))


def run_ocsort_rows(seq: str, max_frames: int, det: Dict[int,List[Det]], params: Dict[str,float]):
    # Exactly the same local OC-SORT public adapter as baseline runner.
    tr=OCSort(det_thresh=params["track_thresh"],max_age=int(params["max_age"]),min_hits=int(params["min_hits"]),
              iou_threshold=params["iou_threshold"],delta_t=int(params["delta_t"]),asso_func="iou",
              inertia=params["inertia"],use_byte=False)
    rows={}; t0=time.perf_counter()
    for fr in range(1,max_frames+1):
        ds=det.get(fr,[]); boxes=np.asarray([d.box for d in ds],np.float32) if ds else np.empty((0,4),np.float32)
        scores=np.asarray([d.score for d in ds],np.float32) if ds else np.empty((0,),np.float32); cats=np.zeros(len(ds),np.float32)
        ret=tr.update_public(boxes,cats,scores)
        cur=[]
        for r in ret:
            if len(r)<7: continue
            x1,y1,x2,y2,tid,_,lag=[float(x) for x in r[:7]]; ofr=fr+int(round(lag))
            if ofr!=fr: continue  # min_hits=1 in formal runs; avoid HP duplicates
            b=np.asarray([x1,y1,x2,y2],np.float32); d=best_det(ds,b,.20)
            cur.append({"frame":fr,"box":b,"native_id":int(round(tid)),"score":float(d.score if d else 1.),"feat":d.feat if d else None})
        rows[fr]=cur
    return rows,time.perf_counter()-t0


def adapt_rows(raw_rows: Dict[int,List[dict]], predictor: Optional[FilterJEPA], mode: str, max_frames: int, iw: float, ih: float, seed: int, max_age: int=30):
    """Causal identity adapter; boxes/emissions come from OC-SORT rows."""
    states: List[State]=[]; next_id=1; out={}; rng=np.random.RandomState(seed+818)
    # Native IDs are only a continuity hint, never a GT signal.
    for fr in range(1,max_frames+1):
        cur=raw_rows.get(fr,[])
        # Candidate features may be unavailable only for malformed crops.
        active=[s for s in states if fr-s.last_frame<=max_age]
        if active and cur:
            # Pre-registered occlusion repair: preserve a continuous native
            # OC-SORT identity when its geometry is unambiguous.  JEPA is
            # allowed to act only on rows left after this continuity pass or
            # on a short-gap/reappearance candidate.  This prevents a weak
            # provisional filter from perturbing every easy frame.
            forced=[]; forced_s=set(); forced_r=set()
            if mode.endswith("_gate"):
                for ii,s in enumerate(active):
                    best_j=-1; best_q=.0
                    for jj,r in enumerate(cur):
                        if jj in forced_r or int(r["native_id"]) != int(s.native_id): continue
                        q=iou(s.box,r["box"])
                        if q>best_q: best_q,best_j=q,jj
                    if best_j>=0 and best_q>=.20:
                        forced.append((ii,best_j)); forced_s.add(ii); forced_r.add(best_j)
            rem_active=[ii for ii in range(len(active)) if ii not in forced_s]
            rem_cur=[jj for jj in range(len(cur)) if jj not in forced_r]
            cost=np.full((len(rem_active),len(rem_cur)),1e3,np.float32)
            for ai,i in enumerate(rem_active):
                s=active[i]
                for aj,j in enumerate(rem_cur):
                    r=cur[j]
                    q=iou(s.box,r["box"])
                    app=0.
                    if r.get("feat") is not None:
                        if mode=="raw": app=0.
                        elif mode=="history": app=float(np.dot(norm(s.feat_hist[-1]),norm(r["feat"]))) if s.feat_hist else 0.
                        elif s.omega is not None: app=float(np.dot(s.omega,norm(r["feat"])))
                    native=1. if int(r["native_id"])==int(s.native_id) else 0.
                    # Keep OC-SORT geometry dominant; filter resolves close
                    # competitors and reappearance after a gap.
                    if mode=="raw": score=0.90*q+0.10*native
                    elif mode=="history": score=0.55*q+0.35*((app+1)/2)+0.10*native
                    else: score=0.52*q+0.38*((app+1)/2)+0.10*native
                    cost[ai,aj]=-score
            rr,cc=linear_sum_assignment(cost) if cost.size else (np.empty((0,),int),np.empty((0,),int)); matches=list(forced)
            for i,j in zip(rr,cc):
                i,j=rem_active[int(i)],rem_cur[int(j)]
                s=active[i]; r=cur[j]; q=iou(s.box,r["box"]); app=0.
                if r.get("feat") is not None and s.omega is not None: app=float(np.dot(s.omega,norm(r["feat"])))
                # Conservative gate: don't attach an unrelated row just for
                # appearance; allow a strong filter to bridge a short gap.
                ok=(q>=.05 and (mode=="raw" or app>=-.05)) or (mode not in ("raw",) and app>=.45 and fr-s.last_frame<=8)
                if ok: matches.append((i,j))
        else: matches=[]
        used_s={i for i,j in matches}; used_r={j for i,j in matches}
        # Matched rows retain canonical ID and update predictor for next frame.
        for i,j in matches:
            s=active[i]; r=cur[j]; old=s.box.copy(); s.box=r["box"].copy(); s.last_frame=fr; s.native_id=int(r["native_id"]); s.score=float(r["score"])
            c=norm(r["feat"]) if r.get("feat") is not None else (s.feat_hist[-1] if s.feat_hist else np.zeros(predictor.dim if predictor else 32,np.float32))
            if len(s.feat_hist):
                oc=np.asarray([(old[0]+old[2])/2,(old[1]+old[3])/2]); nc=np.asarray([(s.box[0]+s.box[2])/2,(s.box[1]+s.box[3])/2]); s.vel=nc-oc
            s.feat_hist=(s.feat_hist+[c])[-(predictor.k if predictor else 4):]
            if predictor is not None and mode not in ("raw","history"):
                gg=geom(s.box,iw,ih,0.,1.,s.vel,s.score,0.); s.omega=predictor.predict(np.asarray(s.feat_hist[:-1]),c,gg)
            elif mode=="history": s.omega=c
            elif mode=="raw": s.omega=None
            r["cid"]=s.cid
        # Unmatched rows create identities; unmatched states remain alive.
        for j,r in enumerate(cur):
            if j in used_r: continue
            c=norm(r["feat"]) if r.get("feat") is not None else np.zeros(predictor.dim if predictor else 32,np.float32)
            s=State(next_id,fr,r["box"].copy(),[c],None,int(r["native_id"]),float(r["score"]),np.zeros(2,np.float32)); next_id+=1
            if mode=="history": s.omega=c
            elif predictor is not None and mode not in ("raw",): s.omega=predictor.predict([],c,geom(s.box,iw,ih,0.,1.,s.vel,s.score,0.))
            states.append(s); r["cid"]=s.cid
        # Emit rows with canonical ID; raw mode uses native IDs exactly.
        out[fr]=[]
        for r in cur:
            tid=int(r["native_id"] if mode=="raw" else r.get("cid",r["native_id"]))
            out[fr].append((r["box"],tid,float(r["score"])))
    return out


def write_tracks(path: Path, rows: Dict[int,List[Tuple[np.ndarray,int,float]]]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w") as f:
        for fr in sorted(rows):
            for b,tid,s in rows[fr]:
                x1,y1,x2,y2=map(float,b); f.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},-1,-1,-1\n".format(fr,tid,x1,y1,x2-x1,y2-y1,s))


def run_one(seq: str, max_frames: int, out: Path, methods: Sequence[str], predictor_map: Dict[str,FilterJEPA], feature_source: str, cache_dir: Optional[Path], params: Dict[str,float], seed: int):
    sd=MOT_ROOT/"train"/seq; det=parse_det(sd/"det/det.txt",max_frames)
    raw_feats=load_features(seq,det,feature_source,cache_dir)
    # The train-side random projection is deterministic and shared across all
    # sequences/methods.  Keep detector rows unchanged, replacing only the
    # optional feature attached to each row.
    dim=next(iter(predictor_map.values())).dim if predictor_map else 32
    projected,_=project_features(raw_feats,dim,seed)
    for fr,ds in det.items():
        for idx,d in enumerate(ds): d.feat=projected[fr][idx]
    raw,sec=run_ocsort_rows(seq,max_frames,det,params); info={}
    for line in (sd/"seqinfo.ini").read_text(errors="replace").splitlines():
        if "=" in line: k,v=line.split("=",1); info[k.strip()]=v.strip()
    iw,ih=float(info.get("imWidth",1920)),float(info.get("imHeight",1080)); stats=[]
    for m in methods:
        if m=="raw": rows={fr:[(r["box"],int(r["native_id"]),r["score"]) for r in rs] for fr,rs in raw.items()}
        else: rows=adapt_rows(raw,predictor_map.get(m),m,max_frames,iw,ih,seed)
        p=out/"mot17"/"tracks"/m/(seq+".txt"); write_tracks(p,rows)
        stats.append({"sequence":seq,"method":m,"frames":max_frames,"det_rows":sum(len(v) for v in det.values()),"track_rows":sum(len(v) for v in rows.values()),"fps_ocsort":max_frames/max(sec,1e-9),"track_sha256":sha256(p)})
    return stats


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-dir",default="experiments/results/mot_ocsort_jepa_mot20"); ap.add_argument("--mot-train",default="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-09-FRCNN"); ap.add_argument("--mot-eval",default="MOT17-05-FRCNN"); ap.add_argument("--max-frames",type=int,default=20); ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--seed",type=int,default=17); ap.add_argument("--feature-source",choices=["handcrafted","fastreid"],default="handcrafted"); ap.add_argument("--feature-cache",default=""); ap.add_argument("--methods",default="raw,history,jepa,no_teacher,random_frozen"); ap.add_argument("--dim",type=int,default=32); ap.add_argument("--k",type=int,default=4); ap.add_argument("--track-thresh",type=float,default=.2); ap.add_argument("--iou-threshold",type=float,default=.3); ap.add_argument("--max-age",type=int,default=30); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True); train=[x for x in args.mot_train.split(",") if x]; ev=[x for x in args.mot_eval.split(",") if x]; methods=[x for x in args.methods.split(",") if x]; cache=Path(args.feature_cache).resolve() if args.feature_cache else None
    # Build/freeze the projection and predictor from train sequences only.
    samples=build_samples(train,args.max_frames,args.dim,args.k,args.seed,args.feature_source,cache); predictors={}
    for m in methods:
        base_mode=m[:-5] if m.endswith("_gate") else m
        if base_mode in ("jepa","no_teacher"):
            p=FilterJEPA(args.dim,args.k,args.seed,args.epochs,base_mode); p.fit(samples); predictors[m]=p
        elif base_mode=="random_frozen":
            p=FilterJEPA(args.dim,args.k,args.seed,args.epochs,m); p.fit_random(); predictors[m]=p
    params={"track_thresh":args.track_thresh,"iou_threshold":args.iou_threshold,"max_age":args.max_age,"min_hits":1,"delta_t":3,"inertia":.2}
    rows=[]
    for seq in train+ev: rows.extend(run_one(seq,args.max_frames,out,methods,predictors,args.feature_source,cache,params,args.seed))
    payload={"schema":"mot-ocsort-jepa-v1","status":"complete","dataset":"mot17","train_sequences":train,"eval_sequences":ev,"max_frames":args.max_frames,"methods":methods,"feature_source":args.feature_source,"feature_cache":str(cache) if cache else None,"teacher":"train-only hard-negative association filter; frozen after sample fit","student":"residual predictor; lagged t→t+1 adapter","eval_gt_read_by_tracker":False,"candidate_in_predictor":False,"ocsort_params":params,"predictor_train_stats":{m:p.stats for m,p in predictors.items()},"rows":rows,"command":" ".join(sys.argv),"cwd":os.getcwd(),"python":sys.version,"platform":platform.platform(),"source_sha256":sha256(Path(__file__).resolve())}
    # Keep a machine-readable reflection next to the raw run.  This is a
    # bounded smoke (not a leaderboard claim), and explicitly records the
    # causal/evaluation-GT boundary for downstream audits.
    (out/"results.json").write_text(json.dumps(payload,indent=2,sort_keys=True,default=float)+"\n")
    reflection={
        "schema":"mot-ocsort-jepa-reflection-v1",
        "status":"complete",
        "result_path":str(out/"results.json"),
        "traceback":None,
        "eval_gt_read_by_tracker":False,
        "candidate_in_predictor":False,
        "teacher_fit_scope":"train_sequences_only",
        "protocol":"OC-SORT detector rows with a causal lagged t-to-t+1 association adapter; predictor is not given next-frame candidates",
        "finding":"Smoke completed; handcrafted association-filter target is weak and this run is diagnostic only. Official TrackEval must be run separately on the emitted MOT files.",
        "next_step":"Run the vendored TrackEval audit, then compare raw/history/JEPA/no-teacher/random on the same staged detector rows before any online integration.",
        "source_sha256":payload["source_sha256"],
    }
    (out/"reflection.json").write_text(json.dumps(reflection,indent=2,sort_keys=True,default=float)+"\n")
    print(json.dumps(payload,indent=2,sort_keys=True,default=float))


if __name__=="__main__": main()
