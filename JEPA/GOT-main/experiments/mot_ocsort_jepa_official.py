#!/usr/bin/env python3
"""Bounded OC-SORT + frozen official GOT-JEPA latent smoke.

The public GOT-JEPA checkpoint is a single-object tracker, not an MOT model.
This experiment therefore uses only its frozen DINO/head representation and
serialized JEPA classification predictor as a *causal appearance filter* for
OC-SORT identities.  Detector boxes and OC-SORT lifecycle are unchanged.

For a row accepted at frame ``t`` we store the pooled output of
``JEPA_predictor_cls``.  That output is used only to score candidates at
``t+1`` (never the current candidate set).  A native-ID/IoU continuity gate
keeps easy OC-SORT frames identical to the baseline.  This is intentionally a
small bounded smoke, not a leaderboard implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "pytracking"
HYBRID_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT")
MOT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17")
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mot_ocsort_jepa as base  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class OfficialLatent:
    """Loader and frozen feature API for GOT_JEPA_378.tar."""

    def __init__(self, checkpoint: Path, device: str, batch: int = 8):
        import torch
        import tomp_jepa_feature_adapter_smoke as smoke

        self.torch = torch
        self.checkpoint_path = checkpoint.resolve()
        self.checkpoint_sha256 = sha256(self.checkpoint_path)
        self.device = torch.device(device)
        self.batch = int(batch)
        smoke.install_shims(REPO)
        # The archive pickles its serialized constructor.  build_net installs
        # local DINO/CoTracker constructors and verifies an exact state load.
        checkpoint_obj = torch.load(str(self.checkpoint_path), map_location="cpu")
        import ltr.models.tracking.tompnet_PT as tompnet_pt
        net = smoke.build_net(checkpoint_obj, tompnet_pt, torch)
        miss, unexp = net.load_state_dict(checkpoint_obj["net"], strict=False)
        if miss or unexp:
            raise RuntimeError("official checkpoint state mismatch: {} {}".format(miss, unexp))
        self.net = net.to(self.device).eval()
        self.feature_shape = [256, 27, 27]
        self.predictor_name = "JEPA_predictor_cls (official frozen checkpoint)"
        self.load_meta = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint": str(self.checkpoint_path),
            "epoch": checkpoint_obj.get("epoch"),
            "constructed_type": type(net).__name__,
            "parameters": int(sum(p.numel() for p in net.parameters())),
            "feature_shape": self.feature_shape,
            # The DINO/head crop map is 256x27x27.  The semantic JEPA call in
            # ``predict_filter_batch`` is instead made on the filter predictor
            # output, whose per-track shape is 256x1x1.
            "feature_map_shape": [256, 27, 27],
            "filter_predictor_input_shape": [1, "N_tracks", 256, 27, 27],
            "filter_predictor_output_shape": ["N_tracks", 256, 1, 1],
            "jepa_predictor_input_shape": ["N_tracks", 256, 1, 1],
            "jepa_predictor_output_shape": ["N_tracks", 256, 1, 1],
            "device": str(self.device),
            "gpu_name": (torch.cuda.get_device_name(self.device)
                         if self.device.type == "cuda" else None),
        }
        # The checkpoint object contains a second copy of all parameters.
        # Release it after the exact load to keep the bounded run small.
        del checkpoint_obj

    @staticmethod
    def crop_image(img: np.ndarray, box: np.ndarray, size: int = 378) -> np.ndarray:
        """Same square/context convention as the official ToMP smoke."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
        side = max(4.0, (bw * bh) ** 0.5 * 5.0)
        cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
        xa, ya = int(round(cx - side / 2.0)), int(round(cy - side / 2.0))
        xb, yb = xa + int(round(side)), ya + int(round(side))
        pl, pt, pr, pb = max(0, -xa), max(0, -ya), max(0, xb - w), max(0, yb - h)
        if pl or pt or pr or pb:
            img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
            xa, xb, ya, yb = xa + pl, xb + pl, ya + pt, yb + pt
        return cv2.resize(img[ya:yb, xa:xb], (size, size), interpolation=cv2.INTER_LINEAR)

    def _tensor_crop(self, img: np.ndarray, box: np.ndarray):
        torch = self.torch
        rgb = cv2.cvtColor(self.crop_image(img, box), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        return (x - mean) / std

    def extract_sequence(self, seq: str, det: Dict[int, List[base.Det]], max_frames: int,
                         cache_root: Path) -> Tuple[Dict[int, np.ndarray], Dict[str, object]]:
        """Extract/cache one sequence. Return frame -> [rows,C,H,W] maps."""
        seq_dir = MOT_ROOT / "train" / seq
        det_path = seq_dir / "det" / "det.txt"
        cache_root.mkdir(parents=True, exist_ok=True)
        cp = cache_root / (seq + ".npz")
        det_hash = sha256(det_path)
        expected_refs = [(fr, j) for fr in range(1, max_frames + 1)
                         for j in range(len(det.get(fr, [])))]
        if cp.exists():
            try:
                z = np.load(cp, allow_pickle=False)
                if (str(z["det_sha256"]) == det_hash and int(z["max_frames"]) == max_frames
                        and int(z["rows"]) == len(expected_refs)
                        and tuple(z["feature_shape"].tolist()) == (256, 27, 27)):
                    maps = np.asarray(z["features"], np.float32)
                    out = {}; off = 0
                    for fr in range(1, max_frames + 1):
                        n = len(det.get(fr, [])); out[fr] = maps[off:off+n]; off += n
                    return out, {"cache": str(cp), "cache_hit": True, "rows": len(maps),
                                 "feature_shape": list(maps.shape)}
            except Exception:
                pass

        torch = self.torch
        crops, refs = [], []
        t0 = time.perf_counter()
        for fr in range(1, max_frames + 1):
            rows = det.get(fr, [])
            if not rows:
                continue
            img = cv2.imread(str(seq_dir / "img1" / ("{:06d}.jpg".format(fr))))
            if img is None:
                raise FileNotFoundError(seq_dir / "img1" / ("{:06d}.jpg".format(fr)))
            for j, d in enumerate(rows):
                crops.append(self._tensor_crop(img, d.box))
                refs.append((fr, j))
        if not crops:
            raise RuntimeError("no detections in requested prefix")
        feats = []
        with torch.inference_mode():
            for st in range(0, len(crops), self.batch):
                xb = torch.stack(crops[st:st+self.batch]).to(self.device, non_blocking=True)
                z = self.net.extract_dino_features_spatial_intermediate_layers(xb, None)
                if tuple(z.shape[1:]) != (1024, 27, 27):
                    raise RuntimeError("unexpected DINO/bkMlp shape {}".format(tuple(z.shape)))
                z = self.net.head.extract_head_feat(z)
                if tuple(z.shape[1:]) != (256, 27, 27):
                    raise RuntimeError("unexpected head feature shape {}".format(tuple(z.shape)))
                if not bool(torch.isfinite(z).all()):
                    raise FloatingPointError("non-finite official head feature")
                feats.append(z.detach().cpu().float())
        maps = torch.cat(feats, dim=0).numpy().astype(np.float16)
        np.savez_compressed(cp, features=maps, det_sha256=np.asarray(det_hash),
                            max_frames=np.asarray(max_frames), rows=np.asarray(len(maps)),
                            feature_shape=np.asarray([256,27,27]))
        out = {}; off = 0
        for fr in range(1, max_frames + 1):
            n = len(det.get(fr, [])); out[fr] = maps[off:off+n].astype(np.float32); off += n
        return out, {"cache": str(cp), "cache_hit": False, "rows": len(maps),
                     "feature_shape": list(maps.shape), "extract_seconds": time.perf_counter()-t0}

    def predict_pool(self, maps: Sequence[np.ndarray]) -> np.ndarray:
        """Frozen JEPA predictor, pooled to one per-track appearance vector."""
        if not maps:
            return np.zeros(256, np.float32)
        torch = self.torch
        x = torch.from_numpy(np.stack(maps).astype(np.float32)).to(self.device)
        with torch.inference_mode():
            y = self.net.JEPA_predictor_cls(x)
            if not bool(torch.isfinite(y).all()):
                raise FloatingPointError("non-finite official JEPA predictor output")
            p = y.mean(dim=(-1, -2))
            p = torch.nn.functional.normalize(p, dim=-1)
        return p[-1].detach().cpu().numpy().astype(np.float32)

    def predict_filter_batch(self, context_maps: Sequence[np.ndarray], current_maps: Sequence[np.ndarray]) -> np.ndarray:
        """Run the *serialized* ToMP filter predictor, then the JEPA head.

        ``filter_predictor`` is the official semantic boundary: it consumes a
        train/context feature map and a current/test feature map and emits a
        1x1 classification filter.  The JEPA predictor is applied to that
        filter (not to a raw backbone map).  Each list entry is one live MOT
        track, so the sequence dimension is the track batch and no candidate
        boxes enter this call.
        """
        if not context_maps:
            return np.empty((0, 256), np.float32)
        torch = self.torch
        n = len(context_maps)
        # One context frame per track is a pooled history state; the maps
        # themselves are means of up to four accepted earlier observations.
        tr = torch.from_numpy(np.stack(context_maps).astype(np.float32)).to(self.device).unsqueeze(0)
        te = torch.from_numpy(np.stack(current_maps).astype(np.float32)).to(self.device).unsqueeze(0)
        h = w = 27
        yy, xx = torch.meshgrid(torch.arange(h, device=self.device), torch.arange(w, device=self.device), indexing="ij")
        # Object crops are centered by construction.  A fixed Gaussian is the
        # same label convention used by ToMP and contains no MOT GT.
        g = torch.exp(-((xx - 13.0) ** 2 + (yy - 13.0) ** 2) / (2.0 * 4.0 ** 2)).float()
        label = g.view(1, 1, h, w).expand(1, n, h, w).contiguous()
        # Normalized center-crop l/t/r/b target; this is a deterministic crop
        # coordinate prior, not an evaluation annotation.
        ltrb = torch.tensor([.25, .25, .75, .75], device=self.device).view(1, 1, 4, 1, 1).expand(1, n, 4, h, w).contiguous()
        with torch.inference_mode():
            cls_filter, _ = self.net.head.filter_predictor(tr, te, label, ltrb)
            if tuple(cls_filter.shape) != (n, 256, 1, 1):
                raise RuntimeError("unexpected official cls_filter shape {}".format(tuple(cls_filter.shape)))
            omega = self.net.JEPA_predictor_cls(cls_filter)
            if not bool(torch.isfinite(omega).all()):
                raise FloatingPointError("non-finite official JEPA filter output")
            # Keep the learned filter scale for the official classifier.  A
            # separate normalized copy can be formed by callers for cosine
            # diagnostics, but normalizing here would silently change the
            # serialized classifier interface.
            pooled = omega.flatten(1)
        return pooled.detach().cpu().numpy().astype(np.float32)

    def classifier_scores(self, omega: np.ndarray, candidate_maps: Sequence[np.ndarray]) -> np.ndarray:
        """Score current candidates with a frozen official classifier filter."""
        if not candidate_maps:
            return np.empty((0,), np.float32)
        torch = self.torch
        feat = torch.from_numpy(np.stack(candidate_maps).astype(np.float32)).to(self.device).unsqueeze(1)
        filt = torch.from_numpy(np.asarray(omega, np.float32)).to(self.device).view(1, 256, 1, 1)
        with torch.inference_mode():
            score_map = self.net.head.classifier(feat, filt)
            # A crop is centered, but max pooling is less sensitive to a few
            # pixels of crop quantization.  This call sees only candidates at
            # t+1 and never feeds them back to filter prediction.
            score = score_map.flatten(2).amax(dim=2).flatten()
            if not bool(torch.isfinite(score).all()):
                raise FloatingPointError("non-finite official classifier score")
        return score.detach().cpu().numpy().astype(np.float32)


@dataclass
class OState:
    cid: int
    last_frame: int
    box: np.ndarray
    native_id: int
    score: float
    maps: List[np.ndarray] = field(default_factory=list)
    omega: Optional[np.ndarray] = None


def attach_maps(raw: Dict[int, List[dict]], maps: Dict[int, np.ndarray], det: Dict[int, List[base.Det]]):
    for fr, rows in raw.items():
        dm = det.get(fr, [])
        for r in rows:
            if not dm:
                r["feat_map"] = np.zeros((256,27,27), np.float32)
                continue
            j = max(range(len(dm)), key=lambda q: base.iou(dm[q].box, r["box"]))
            r["feat_map"] = maps.get(fr, np.empty((0,256,27,27), np.float32))[j]
            r["feat_vec"] = norm_rows(r["feat_map"].mean((1,2), keepdims=True).reshape(1,-1))[0]


def adapt_official(raw: Dict[int, List[dict]], official: OfficialLatent, max_frames: int,
                   gate: bool = True, max_age: int = 30) -> Tuple[Dict[int, List[Tuple[np.ndarray,int,float]]], Dict[str, object]]:
    """Causal pooled-latent identity pass; no GT and no future candidates."""
    states: List[OState] = []; next_id = 1; out = {}; matched_count = 0; reassigned = 0
    for fr in range(1, max_frames + 1):
        cur = raw.get(fr, [])
        active = [s for s in states if fr - s.last_frame <= max_age]
        forced, used_s, used_r = [], set(), set()
        if gate:
            for i, s in enumerate(active):
                best = (-1, 0.0)
                for j, r in enumerate(cur):
                    if j in used_r or int(r["native_id"]) != int(s.native_id):
                        continue
                    q = base.iou(s.box, r["box"])
                    if q > best[1]: best = (j, q)
                if best[0] >= 0 and best[1] >= .20:
                    forced.append((i, best[0])); used_s.add(i); used_r.add(best[0])
        rem_s = [i for i in range(len(active)) if i not in used_s]
        rem_r = [j for j in range(len(cur)) if j not in used_r]
        cost = np.full((len(rem_s), len(rem_r)), 1e3, np.float32)
        for ai, i in enumerate(rem_s):
            s = active[i]
            for aj, j in enumerate(rem_r):
                r = cur[j]; q = base.iou(s.box, r["box"])
                app = float(np.dot(s.omega, r.get("feat_vec", np.zeros(256, np.float32)))) if s.omega is not None else 0.
                native = 1. if int(r["native_id"]) == int(s.native_id) else 0.
                # Fixed, pre-registered fusion: geometry remains dominant;
                # official JEPA is only a tie-breaker/reappearance cue.
                score = .58*q + .32*((app+1.)/2.) + .10*native
                cost[ai, aj] = -score
        rr, cc = linear_sum_assignment(cost) if cost.size else (np.empty(0, int), np.empty(0, int))
        matches = list(forced)
        for a, b in zip(rr, cc):
            i, j = rem_s[int(a)], rem_r[int(b)]; s, r = active[i], cur[j]
            q = base.iou(s.box, r["box"])
            app = float(np.dot(s.omega, r.get("feat_vec", np.zeros(256, np.float32)))) if s.omega is not None else 0.
            gap = fr - s.last_frame
            if q >= .05 or (app >= .35 and gap <= 8):
                matches.append((i,j))
        used_s = {i for i,j in matches}; used_r = {j for i,j in matches}
        updated = []
        for i, j in matches:
            s, r = active[i], cur[j]
            if int(s.native_id) != int(r["native_id"]): reassigned += 1
            s.box = r["box"].copy(); s.last_frame = fr; s.native_id = int(r["native_id"]); s.score = float(r["score"])
            s.maps = (s.maps + [r["feat_map"].astype(np.float32)])[-4:]
            updated.append(s); r["cid"] = s.cid; matched_count += 1
        for j, r in enumerate(cur):
            if j in used_r: continue
            # Preserve the native OC-SORT label whenever it is available for
            # a newly born track.  Arbitrarily renumbering the first frame can
            # globally permute otherwise-correct identities and would create
            # an artificial IDF1 loss unrelated to the JEPA adapter.  If a
            # native label is already occupied by a still-live state, fall
            # back to a fresh canonical ID (without consulting GT).
            preferred = int(r["native_id"])
            occupied = {int(q.cid) for q in states if fr - q.last_frame <= max_age}
            cid = preferred if preferred > 0 and preferred not in occupied else next_id
            while cid in occupied:
                cid += 1
            next_id = max(next_id, cid + 1)
            s = OState(cid, fr, r["box"].copy(), int(r["native_id"]), float(r["score"]), [r["feat_map"].astype(np.float32)], None)
            states.append(s); updated.append(s); r["cid"] = s.cid; next_id += 1
        # Crucially, this predictor call happens after frame-t assignment and
        # its output is consumed only by frame t+1.
        if updated:
            for s in updated:
                s.omega = official.predict_pool(s.maps)
        out[fr] = [(r["box"], int(r.get("cid", r["native_id"])), float(r["score"])) for r in cur]
    return out, {"matched_rows": matched_count, "native_reassignments": reassigned,
                 "lag": "t->t+1", "gate": gate, "history_k": 4}


def write_tracks(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for fr in sorted(rows):
            for b, tid, score in rows[fr]:
                x1,y1,x2,y2 = map(float, b)
                f.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},-1,-1,-1\n".format(fr,tid,x1,y1,x2-x1,y2-y1,score))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequence", default="MOT17-05-FRCNN")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    payload = {"schema":"mot-ocsort-official-jepa-v1", "status":"started",
               "dataset":"MOT17", "sequence":args.sequence, "max_frames":args.max_frames,
               "methods":["raw","official_pool"], "feature_source":"official GOT-JEPA checkpoint",
               "checkpoint":str(Path(args.checkpoint).resolve()), "command":" ".join(sys.argv),
               "cwd":os.getcwd(), "python":sys.version, "platform":platform.platform(),
               "eval_gt_read_by_tracker":False, "candidate_in_predictor":False,
               "production_tracker_modified":False, "association_lag":"t->t+1"}
    t_all = time.perf_counter()
    try:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        official = OfficialLatent(Path(args.checkpoint), args.device, args.batch)
        payload["official_load"] = official.load_meta
        sd = MOT_ROOT / "train" / args.sequence
        det = base.parse_det(sd / "det" / "det.txt", args.max_frames)
        params = {"track_thresh":.2, "iou_threshold":.3, "max_age":30, "min_hits":1, "delta_t":3, "inertia":.2}
        cache_root = out / "cache"
        maps, cache_meta = official.extract_sequence(args.sequence, det, args.max_frames, cache_root)
        payload["feature_cache"] = cache_meta
        raw, sec = base.run_ocsort_rows(args.sequence, args.max_frames, det, params)
        attach_maps(raw, maps, det)
        raw_rows = {fr:[(r["box"], int(r["native_id"]), float(r["score"])) for r in rs] for fr,rs in raw.items()}
        official_rows, adapt_meta = adapt_official(raw, official, args.max_frames, gate=True, max_age=30)
        write_tracks(out/"mot17"/"tracks"/"raw"/(args.sequence+".txt"), raw_rows)
        write_tracks(out/"mot17"/"tracks"/"official_pool"/(args.sequence+".txt"), official_rows)
        payload["ocsort_seconds"] = sec; payload["det_rows"] = int(sum(len(v) for v in det.values()))
        payload["track_rows"] = {"raw":sum(len(v) for v in raw_rows.values()), "official_pool":sum(len(v) for v in official_rows.values())}
        payload["adapter"] = adapt_meta
        payload["status"] = "complete"
        payload["finding"] = "Bounded official pooled-latent smoke complete; run vendored TrackEval before any claim."
    except Exception as exc:
        payload["status"] = "failure"; payload["error_type"] = type(exc).__name__; payload["error"] = str(exc); payload["traceback"] = traceback.format_exc()
        (out/"failure.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float)+"\n")
    payload["elapsed_seconds"] = time.perf_counter() - t_all
    (out/"results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float)+"\n")
    reflection = {"schema":"mot-ocsort-official-jepa-reflection-v1", "status":payload["status"],
                  "result_path":str(out/"results.json"), "eval_gt_read_by_tracker":False,
                  "candidate_in_predictor":False, "protocol":"fixed OC-SORT rows; frozen official DINO/head + JEPA pooled filter generated after t and consumed at t+1",
                  "next_step":"If TrackEval shows no drop on this bounded smoke, repeat on preregistered MOT17/DanceTrack splits; otherwise record NO-GO and train a MOT-specific JEPA filter.",
                  "error":payload.get("error"), "checkpoint_sha256":payload.get("official_load",{}).get("checkpoint_sha256")}
    (out/"reflection.json").write_text(json.dumps(reflection, indent=2, ensure_ascii=False, default=float)+"\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
