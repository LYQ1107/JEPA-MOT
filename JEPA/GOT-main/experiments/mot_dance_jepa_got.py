#!/usr/bin/env python3
"""Bounded DanceTrack MOT experiment aligned to the public GOT-JEPA encoder.

This is an isolated attribution harness; it does not edit LocateMOT, MASA, or
the earlier crop80 experiment.  All detector and clean-teacher appearances are
produced by the *same frozen* DINOv2 + ToMP head from ``GOT_JEPA_378.tar``:
the native 256x27x27 map is spatially pooled and L2-normalised to 256-D.

The MOT student is the existing strict history-only ``TemporalPredictor``.
Its K=4 input tokens contain only earlier fixed-detector observations,
observable geometry/score, and explicit missing masks.  Current candidates
are consumed only by the Hungarian association cost.  Train GT is used only
to select clean teacher targets and train identity negatives.  Evaluation
tracking is completed and hashed before validation GT is parsed for reporting;
the official TrackEval audit remains a separate command.
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
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# The vendored motmetrics copy used by the experiment predates NumPy 1.24.
# Install compatibility aliases in this process only; no dependency checkout
# or production tracker source is edited.
for _name, _value in {"float": float, "int": int, "bool": bool, "object": object, "str": str}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# The serialized checkpoint imports ``ltr.*`` and the vendored CoTracker
# package during construction.  Keep these paths local to this experiment;
# no installed/production tracker package is modified.
PYTRACKING_ROOT = HERE.parent / "pytracking"
if str(PYTRACKING_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTRACKING_ROOT))
COTRACKER_ROOT = PYTRACKING_ROOT / "ltr" / "cotracker2" / "co_tracker"
if str(COTRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_ROOT))

import mot_dance_jepa_real as dance  # noqa: E402
import mot_ocsort_jepa_official_v2 as official_api  # noqa: E402


CHECKPOINT_DEFAULT = "/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar"
TRAIN_DEFAULT = "dancetrack0001,dancetrack0002,dancetrack0006"
EVAL_DEFAULT = "dancetrack0004,dancetrack0007,dancetrack0014"
METHODS = ("raw", "jepa_temporal", "no_teacher", "random_frozen")
FEATURE_DIM = 256
FEATURE_MAP_SHAPE = (256, 27, 27)
HISTORY_K = 4


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def hash_records(records: Sequence[Tuple[int, int, np.ndarray, float]]) -> str:
    h = hashlib.sha256()
    for fr, key, box, score in records:
        h.update(np.asarray([int(fr), int(key)], np.int64).tobytes())
        h.update(np.asarray(box, np.float32).tobytes())
        h.update(np.asarray([float(score)], np.float32).tobytes())
    return h.hexdigest()


def norm(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v, np.float32)
    return a / max(float(np.linalg.norm(a)), 1e-8)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def list_names(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def gt_records(gt: Dict[int, List[dance.GT]]) -> List[Tuple[int, int, np.ndarray, float]]:
    return [(int(fr), int(g.gid), np.asarray(g.box, np.float32), 1.0)
            for fr in sorted(gt) for g in sorted(gt[fr], key=lambda x: int(x.gid))]


def detector_records(det: Dict[int, List[dance.Det]]) -> List[Tuple[int, int, np.ndarray, float]]:
    return [(int(fr), int(j), np.asarray(d.box, np.float32), float(d.score))
            for fr in sorted(det) for j, d in enumerate(det[fr])]


def _cache_meta(path: Path) -> Dict[str, object]:
    return json.loads(path.with_suffix(".json").read_text())


def extract_pooled_records(
    encoder: official_api.OfficialLatent,
    seq_dir: Path,
    records: Sequence[Tuple[int, int, np.ndarray, float]],
    cache_path: Path,
    role: str,
    max_frames: int,
    batch: int,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[str, object]]:
    """Extract frozen GOT-JEPA maps and cache only their 256-D pooled vectors."""
    record_hash = hash_records(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = cache_path.with_suffix(".json")
    expected = {
        "schema": "got-jepa-pooled-cache-v1", "sequence": seq_dir.name, "role": role,
        "max_frames": int(max_frames), "record_sha256": record_hash,
        "checkpoint_sha256": encoder.checkpoint_sha256,
        "source_feature_shape": list(FEATURE_MAP_SHAPE), "pooled_dim": FEATURE_DIM,
        "crop_size": 378, "context_factor": 5.0,
    }
    if cache_path.exists() and meta_path.exists():
        try:
            meta = _cache_meta(cache_path)
            z = np.load(cache_path, allow_pickle=False)
            valid = all(meta.get(k) == v for k, v in expected.items())
            valid = valid and tuple(z["features"].shape) == (len(records), FEATURE_DIM)
            valid = valid and tuple(z["frames"].shape) == (len(records),)
            valid = valid and tuple(z["keys"].shape) == (len(records),)
            if valid:
                feats = np.asarray(z["features"], np.float32)
                feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-8)
                if np.isfinite(feats).all():
                    out = {(int(fr), int(key)): feats[i] for i, (fr, key, _, _) in enumerate(records)}
                    return out, {**meta, "cache": str(cache_path), "cache_sha256": sha256(cache_path),
                                 "cache_hit": True, "rows": len(feats), "pooled_shape": list(feats.shape)}
        except Exception:
            pass
    if not records:
        raise RuntimeError("no records for {} {}".format(seq_dir.name, role))

    torch = encoder.torch
    files = dance.image_files(seq_dir)
    features: List[np.ndarray] = []
    pending = []
    map_shapes = set()
    t0 = time.perf_counter()

    def flush() -> None:
        if not pending:
            return
        xb = torch.stack(pending).to(encoder.device, non_blocking=True)
        with torch.inference_mode():
            z = encoder.net.extract_dino_features_spatial_intermediate_layers(xb, None)
            if tuple(z.shape[1:]) != (1024, 27, 27):
                raise RuntimeError("unexpected DINO map {}".format(tuple(z.shape)))
            z = encoder.net.head.extract_head_feat(z)
            map_shapes.add(tuple(int(x) for x in z.shape[1:]))
            if tuple(z.shape[1:]) != FEATURE_MAP_SHAPE or not bool(torch.isfinite(z).all()):
                raise FloatingPointError("invalid GOT-JEPA head map {}".format(tuple(z.shape)))
            pooled = torch.nn.functional.normalize(z.mean(dim=(-1, -2)), dim=-1)
            if not bool(torch.isfinite(pooled).all()):
                raise FloatingPointError("non-finite pooled GOT-JEPA feature")
            features.append(pooled.detach().cpu().float().numpy())
        pending.clear()

    current_frame = None
    image = None
    for fr, _, box, _ in records:
        if int(fr) != current_frame:
            current_frame = int(fr)
            if current_frame < 1 or current_frame > len(files):
                raise IndexError("frame {} outside image list for {}".format(current_frame, seq_dir))
            image = cv2.imread(str(files[current_frame - 1]))
            if image is None:
                raise FileNotFoundError(files[current_frame - 1])
        pending.append(encoder._tensor_crop(image, np.asarray(box, np.float32)))
        if len(pending) >= int(batch):
            flush()
    flush()
    feats = np.concatenate(features, axis=0).astype(np.float32)
    if feats.shape != (len(records), FEATURE_DIM) or not np.isfinite(feats).all():
        raise RuntimeError("pooled feature shape/finite failure {}".format(feats.shape))
    if encoder.device.type == "cuda":
        torch.cuda.synchronize(encoder.device)
    elapsed = time.perf_counter() - t0
    frames = np.asarray([x[0] for x in records], np.int32)
    keys = np.asarray([x[1] for x in records], np.int32)
    np.savez_compressed(cache_path, features=feats.astype(np.float16), frames=frames, keys=keys)
    meta = {
        **expected, "rows": len(records), "pooled_shape": list(feats.shape),
        "observed_source_feature_shapes": [list(x) for x in sorted(map_shapes)],
        "finite": True, "extract_seconds": elapsed,
        "rows_per_second": float(len(records) / max(elapsed, 1e-9)),
        "cache": str(cache_path), "cache_sha256": sha256(cache_path), "cache_hit": False,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    out = {(int(fr), int(key)): feats[i] for i, (fr, key, _, _) in enumerate(records)}
    return out, meta


def attach_detector_features(det: Dict[int, List[dance.Det]], feats: Dict[Tuple[int, int], np.ndarray]) -> None:
    for fr, rows in det.items():
        for j, d in enumerate(rows):
            f = feats.get((int(fr), int(j)))
            if f is None or f.shape != (FEATURE_DIM,) or not np.isfinite(f).all():
                raise RuntimeError("missing/invalid detector feature {}:{}".format(fr, j))
            d.feat = norm(f)


def one_to_one_matches(gt: Dict[int, List[dance.GT]], det: Dict[int, List[dance.Det]], threshold: float = .30):
    by_gid: Dict[Tuple[int, int], Tuple[int, dance.Det]] = {}
    stats = {"gt_rows": 0, "det_rows": 0, "matches": 0, "duplicate_detector_matches": 0}
    for fr in sorted(gt):
        gs, ds = gt.get(fr, []), det.get(fr, [])
        stats["gt_rows"] += len(gs); stats["det_rows"] += len(ds)
        if not gs or not ds:
            continue
        mat = np.asarray([[dance.box_iou(g.box, d.box) for d in ds] for g in gs], np.float32)
        rr, cc = linear_sum_assignment(-mat)
        used = set()
        for i, j in zip(rr.tolist(), cc.tolist()):
            if float(mat[i, j]) < threshold:
                continue
            if j in used:
                stats["duplicate_detector_matches"] += 1; continue
            used.add(j); g = gs[i]
            by_gid[(int(fr), int(g.gid))] = (int(j), ds[j]); stats["matches"] += 1
    return by_gid, stats


def build_temporal_samples_got(
    items: Sequence[Tuple[str, Path, Dict[int, List[dance.GT]], Dict[int, List[dance.Det]], Dict[Tuple[int, int], np.ndarray]]],
    max_frames: int,
    seed: int,
    max_samples: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """Build K fixed-offset detector histories and clean GOT-JEPA targets."""
    rng = np.random.RandomState(seed)
    xs: List[np.ndarray] = []; ys: List[np.ndarray] = []; poss: List[np.ndarray] = []
    negs: List[np.ndarray] = []; pair_masks: List[float] = []
    futures: List[np.ndarray] = []; future_masks: List[float] = []; qualities: List[float] = []
    stats: Dict[str, object] = {"sequences": {}, "history_future_violations": 0,
                                "student_gt_box_fields": 0, "visibility_fields": 0,
                                "current_candidate_in_x": False, "teacher_encoder_same_as_detector": True}
    for seq, seq_dir, gt, det, teacher in items:
        matched, match_stats = one_to_one_matches(gt, det)
        iw, ih = dance.sequence_wh(seq_dir)
        gid_frames: Dict[int, List[int]] = {}
        for (fr, gid) in matched:
            gid_frames.setdefault(int(gid), []).append(int(fr))
        for gid in gid_frames:
            gid_frames[gid].sort()
        seq_start = len(xs)
        for fr in sorted(gt):
            if fr <= 1 or (max_frames and fr > max_frames):
                continue
            for g in gt[fr]:
                target = teacher.get((int(fr), int(g.gid)))
                if target is None:
                    continue
                token_parts: List[np.ndarray] = []; valid = 0
                for offset in range(HISTORY_K, 0, -1):
                    old = int(fr) - int(offset)
                    rec = matched.get((old, int(g.gid)))
                    previous = [q for q in gid_frames.get(int(g.gid), []) if q < old]
                    prev_rec = matched.get((previous[-1], int(g.gid))) if previous else None
                    if rec is not None:
                        _, d = rec
                        alternatives = [dd.feat for dd in det.get(old, []) if dd is not d and dd.feat is not None]
                        feat = dance._corrupt_history_feature(d.feat, alternatives, rng)
                        box, score, obs_frame, missing = d.box, d.score, old, False
                        valid += 1
                    else:
                        prior = [q for q in gid_frames.get(int(g.gid), []) if q <= old]
                        fallback = matched.get((prior[-1], int(g.gid))) if prior else None
                        feat = np.zeros(FEATURE_DIM, np.float32)
                        box = fallback[1].box if fallback is not None else None
                        score = 0.0; obs_frame = prior[-1] if prior else old; missing = True
                    scalars = dance._temporal_token_scalars(
                        box, prev_rec[1].box if prev_rec is not None else None,
                        int(fr), int(obs_frame), float(score), bool(missing), float(iw), float(ih))
                    token_parts.append(np.concatenate([np.asarray(feat, np.float32), scalars]))
                    if int(obs_frame) >= int(fr):
                        stats["history_future_violations"] += 1
                x = np.concatenate(token_parts).astype(np.float32)
                if x.shape != (dance.temporal_input_dim(FEATURE_DIM, HISTORY_K),):
                    raise RuntimeError("bad temporal input {}".format(x.shape))
                cur = matched.get((int(fr), int(g.gid)))
                if cur is not None:
                    pos = cur[1].feat.copy()
                    competing = [d.feat for j, d in enumerate(det.get(fr, [])) if j != cur[0] and d.feat is not None]
                else:
                    pos = target.copy(); competing = [d.feat for d in det.get(fr, []) if d.feat is not None]
                if competing:
                    neg = competing[int(rng.randint(len(competing)))].copy(); pair_mask = 1.0
                else:
                    neg = target.copy(); pair_mask = 0.0
                future = teacher.get((int(fr) + 1, int(g.gid)), target)
                future_mask = 1.0 if (int(fr) + 1, int(g.gid)) in teacher else 0.0
                xs.append(x); ys.append(norm(target)); poss.append(norm(pos)); negs.append(norm(neg))
                pair_masks.append(pair_mask); futures.append(norm(future)); future_masks.append(future_mask)
                qualities.append(float(valid / HISTORY_K))
                if len(xs) >= max_samples:
                    break
            if len(xs) >= max_samples:
                break
        stats["sequences"][seq] = {"samples": len(xs) - seq_start, "matches": match_stats}
        if len(xs) >= max_samples:
            break
    if not xs:
        raise RuntimeError("no GOT-JEPA temporal samples")
    samples = {
        "x": np.asarray(xs, np.float32), "y": dance.normalize(np.asarray(ys, np.float32)),
        "pos": dance.normalize(np.asarray(poss, np.float32)), "neg": dance.normalize(np.asarray(negs, np.float32)),
        "pair_mask": np.asarray(pair_masks, np.float32), "future_y": dance.normalize(np.asarray(futures, np.float32)),
        "future_mask": np.asarray(future_masks, np.float32), "quality": np.asarray(qualities, np.float32),
    }
    for k, v in samples.items():
        if not np.isfinite(v).all():
            raise FloatingPointError("non-finite training field {}".format(k))
    stats.update({"count": len(xs), "input_shape": list(samples["x"].shape),
                  "target_shape": list(samples["y"].shape), "finite": True})
    return samples, stats


def tracks_hash(outputs: Dict[int, List[Tuple[int, np.ndarray, float]]]) -> str:
    h = hashlib.sha256()
    for fr in sorted(outputs):
        for tid, box, score in outputs[fr]:
            h.update(np.asarray([int(fr), int(tid)], np.int64).tobytes())
            h.update(np.asarray(box, np.float32).tobytes())
            h.update(np.asarray([float(score)], np.float32).tobytes())
    return h.hexdigest()


def track_eval_without_gt(
    eval_items: Sequence[Tuple[str, Path, Dict[int, List[dance.Det]]]],
    predictors: Dict[str, Optional[dance.TemporalPredictor]],
    methods: Sequence[str],
    out: Path,
    seed: int,
) -> Tuple[Dict[str, Dict[str, Dict[str, object]]], Dict[str, object]]:
    """Run all trackers with a signature that cannot receive GT."""
    outputs: Dict[str, Dict[str, Dict[str, object]]] = {}
    audit: Dict[str, object] = {"eval_uses_gt": False, "gt_argument_in_tracker": False,
                                "candidate_in_predictor": False, "tracker_completed_before_eval_gt_parse": True,
                                "sequence_hashes": {}}
    for seq, seq_dir, det in eval_items:
        iw, ih = dance.sequence_wh(seq_dir); outputs[seq] = {}
        audit["sequence_hashes"][seq] = {}
        for method in methods:
            tracker = dance.SimpleMOT(method, predictors.get(method), seed=seed,
                                      image_width=iw, image_height=ih)
            t0 = time.perf_counter(); result = tracker.run(det); elapsed = time.perf_counter() - t0
            path = out / "dance" / "tracks" / method / (seq + ".txt")
            dance.save_track_file(path, result)
            rec = {
                "outputs": result, "track_path": str(path), "track_sha256": sha256(path),
                "semantic_sha256": tracks_hash(result), "track_rows": len(dance.output_rows(result)),
                "association_seconds": elapsed, "temporal_prediction_count": int(tracker.temporal_prediction_count),
                "quality_mean": float(np.mean(tracker.quality_values)) if tracker.quality_values else None,
                "uncertainty_mean": float(np.mean(tracker.uncertainty_values)) if tracker.uncertainty_values else None,
                "memory_gate_accept": int(tracker.memory_gate_accept),
                "memory_gate_reject": int(tracker.memory_gate_reject),
            }
            outputs[seq][method] = rec
            audit["sequence_hashes"][seq][method] = rec["semantic_sha256"]
    return outputs, audit


def run_smoke(args: argparse.Namespace, out: Path, encoder: official_api.OfficialLatent) -> Dict[str, object]:
    """One-sequence feature/cache and detector-only tracker invariant smoke."""
    seq = args.smoke_sequence
    split = "train" if (dance.DANCE_ROOT / "train" / seq).exists() else "val"
    seq_dir = dance.DANCE_ROOT / split / seq
    det = dance.parse_frame_det_cache(dance.DANCE_CTRL_CACHE / seq, args.max_frames)
    drecs = detector_records(det)
    dfeat, dmeta = extract_pooled_records(encoder, seq_dir, drecs,
                                           Path(args.cache_dir) / (seq + "_detector.npz"),
                                           "detector", args.max_frames, args.batch)
    attach_detector_features(det, dfeat)
    tmeta = None; target_cosine = None
    if split == "train":
        ann = dance.limit_frame_dicts(
            dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/train.json", "train", [seq]),
            args.max_frames)[seq]
        trecs = gt_records(ann)
        tfeat, tmeta = extract_pooled_records(encoder, seq_dir, trecs,
                                              Path(args.cache_dir) / (seq + "_teacher.npz"),
                                              "clean_train_teacher", args.max_frames, args.batch)
        matched, _ = one_to_one_matches(ann, det)
        cos = [float(np.dot(norm(tfeat[(fr, gid)]), norm(d.feat)))
               for (fr, gid), (_, d) in matched.items() if (fr, gid) in tfeat]
        target_cosine = {"n": len(cos), "mean": float(np.mean(cos)) if cos else None,
                         "p10": float(np.percentile(cos, 10)) if cos else None}
    # Detector-only no-GT tracker smoke.  ``raw`` has no predictor and this
    # call occurs without constructing/parsing validation annotations.
    tracker = dance.SimpleMOT("raw", None, seed=args.seed,
                              image_width=dance.sequence_wh(seq_dir)[0],
                              image_height=dance.sequence_wh(seq_dir)[1])
    raw = tracker.run(det)
    smoke = {
        "schema": "dance-got-jepa-feature-smoke-v1", "status": "complete",
        "sequence": seq, "split": split, "max_frames": args.max_frames,
        "checkpoint_sha256": encoder.checkpoint_sha256, "official_load": encoder.load_meta,
        "detector_cache": dmeta, "teacher_cache": tmeta, "teacher_detector_cosine": target_cosine,
        "source_feature_shape": list(FEATURE_MAP_SHAPE), "pooled_shape": [FEATURE_DIM],
        "detector_rows": len(drecs), "all_detector_features_finite": bool(all(d.feat is not None and np.isfinite(d.feat).all() for rows in det.values() for d in rows)),
        "raw_track_rows": len(dance.output_rows(raw)), "raw_semantic_sha256": tracks_hash(raw),
        "eval_gt_parsed": False, "tracker_gt_argument": False, "candidate_in_predictor": False,
        "dctrl_identity_columns": 0, "dctrl_schema": "frame,x,y,w,h,score",
    }
    if smoke["source_feature_shape"] != [256, 27, 27] or smoke["pooled_shape"] != [256] or not smoke["all_detector_features_finite"]:
        raise AssertionError("GOT-JEPA feature smoke invariant failed")
    (out / "smoke.json").write_text(json.dumps(smoke, indent=2, sort_keys=True, default=float) + "\n")
    return smoke


def run_full(args: argparse.Namespace, out: Path, encoder: official_api.OfficialLatent) -> Dict[str, object]:
    train_names, eval_names = list_names(args.dance_train), list_names(args.dance_eval)
    train_gt = dance.limit_frame_dicts(
        dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/train.json", "train", train_names),
        args.max_frames)
    cache_root = Path(args.cache_dir); feature_meta: Dict[str, Dict[str, object]] = {}
    train_items = []
    for seq in train_names:
        seq_dir = dance.DANCE_ROOT / "train" / seq
        det = dance.parse_frame_det_cache(dance.DANCE_CTRL_CACHE / seq, args.max_frames)
        dfeat, dm = extract_pooled_records(encoder, seq_dir, detector_records(det),
                                            cache_root / (seq + "_detector.npz"),
                                            "detector", args.max_frames, args.batch)
        attach_detector_features(det, dfeat)
        teacher, tm = extract_pooled_records(encoder, seq_dir, gt_records(train_gt[seq]),
                                              cache_root / (seq + "_teacher.npz"),
                                              "clean_train_teacher", args.max_frames, args.batch)
        feature_meta[seq] = {"detector": dm, "teacher": tm}
        train_items.append((seq, seq_dir, train_gt[seq], det, teacher))
    samples, sample_meta = build_temporal_samples_got(train_items, args.max_frames, args.seed, args.max_samples)
    predictors: Dict[str, Optional[dance.TemporalPredictor]] = {"raw": None}
    for method in ("jepa_temporal", "no_teacher"):
        pred = dance.TemporalPredictor(FEATURE_DIM, method, args.seed, args.epochs,
                                       "cuda" if args.device.startswith("cuda") else "cpu")
        pred.fit(samples); predictors[method] = pred
    rnd = dance.TemporalPredictor(FEATURE_DIM, "random_frozen", args.seed, 0, "cpu")
    rnd.fit_random(); predictors["random_frozen"] = rnd

    # Evaluation feature extraction and tracking use only images + six-column
    # D-CTRL rows.  Validation annotations are intentionally parsed later.
    eval_items = []
    for seq in eval_names:
        seq_dir = dance.DANCE_ROOT / "val" / seq
        det = dance.parse_frame_det_cache(dance.DANCE_CTRL_CACHE / seq, args.max_frames)
        dfeat, dm = extract_pooled_records(encoder, seq_dir, detector_records(det),
                                            cache_root / (seq + "_detector.npz"),
                                            "detector", args.max_frames, args.batch)
        attach_detector_features(det, dfeat); feature_meta[seq] = {"detector": dm, "teacher": None}
        eval_items.append((seq, seq_dir, det))
    tracked_unlabeled, no_gt_audit = track_eval_without_gt(eval_items, predictors, METHODS, out, args.seed)
    tracker_completed_unix = time.time()

    # Post-tracking metrics only.  This block cannot alter saved tracker rows.
    eval_gt = dance.limit_frame_dicts(
        dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/val.json", "val", eval_names),
        args.max_frames)
    eval_gt_parsed_unix = time.time()
    rows = []; sequences: Dict[str, Dict[str, object]] = {}
    for seq, seq_dir, det in eval_items:
        sequences[seq] = {}
        det_digest = hash_records(detector_records(det))
        for method in METHODS:
            rec = tracked_unlabeled[seq][method]
            metrics = dance.evaluate_mot(eval_gt[seq], rec["outputs"])
            public = {k: v for k, v in rec.items() if k != "outputs"}
            public.update(metrics); public.update({"detector_digest": det_digest,
                                                   "detector_recall_iou50": dance.detector_recall(eval_gt[seq], det)})
            sequences[seq][method] = public
            rows.append({"dataset": "dance", "sequence": seq, "method": method, **public})
    no_gt_audit.update({"tracker_completed_unix": tracker_completed_unix,
                        "eval_gt_parsed_unix": eval_gt_parsed_unix,
                        "tracking_precedes_gt_parse": bool(tracker_completed_unix <= eval_gt_parsed_unix)})
    protocol = {
        "teacher_gt_only": True, "teacher_scope": "train sequences only",
        "teacher_encoder": "frozen GOT_JEPA_378 DINOv2 + ToMP head; 256x27x27 mean-pooled L2 256-D",
        "detector_encoder": "same frozen checkpoint/crop/head/pooling as teacher",
        "student_context_gt": False, "gt_geometry_in_student": False,
        "gt_geometry_in_student_x": False, "gt_visibility_in_student": False,
        "eval_uses_gt": False, "eval_gt_parsed_after_tracking": True,
        "candidate_in_predictor": False, "predictor_api": "TemporalPredictor.predict(K detector-history tokens)",
        "strict_temporal_methods": ["jepa_temporal", "no_teacher", "random_frozen"],
        "temporal_k": HISTORY_K,
        "token_scalar_schema": ["age_dt", "center_x", "center_y", "scale_w", "scale_h", "velocity_x", "velocity_y", "detector_score", "missing_mask"],
        "current_corruption": ["history feature mask", "gaussian noise", "identity mix"],
        "uncertainty_gate": "quality>=0.35, sigmoid(log_variance)<=0.65, detector_score>=0.35, prior_misses<=4",
        "association_negative": "same-frame D-CTRL candidate, train label only",
        "cross_time_target": "current and next-frame clean train-GT crop pooled checkpoint feature (stop-gradient)",
        "tracker_input": "D-CTRL six-column detections + frozen pooled checkpoint features only",
    }
    dance_block = {
        "dataset": "dance", "train_sequences": train_names, "eval_sequences": eval_names,
        "detector": {"type": "LocateMOT D-CTRL fixed boxes (per-frame cache; no track IDs)",
                     "cache_root": str(dance.DANCE_CTRL_CACHE), "oracle_boxes": False,
                     "weight_sha256": dance.DANCE_CTRL_WEIGHT_SHA256,
                     "feature_encoder": protocol["detector_encoder"]},
        "training_protocol": protocol, "no_gt_audit": no_gt_audit,
        "feature_cache": feature_meta, "train_samples": {k: list(v.shape) for k, v in samples.items()},
        "sample_audit": sample_meta,
        "predictor_stats": {m: (predictors[m].train_stats if predictors.get(m) is not None else None) for m in METHODS},
        "sequences": sequences, "rows": rows,
    }
    return dance_block


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--dance-train", default=TRAIN_DEFAULT)
    ap.add_argument("--dance-eval", default=EVAL_DEFAULT)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--max-samples", type=int, default=12000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--smoke-sequence", default="dancetrack0001")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    if not args.cache_dir:
        args.cache_dir = str(out / "feature_cache")
    seed_all(args.seed)
    checkpoint = Path(args.checkpoint).resolve()
    payload: Dict[str, object] = {
        "schema": "dance-got-jepa-temporal-v1", "status": "started",
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
        "platform": platform.platform(), "seed": args.seed, "max_frames": args.max_frames,
        "methods": list(METHODS), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "script_sha256": sha256(Path(__file__).resolve()), "production_tracker_modified": False,
        "old_crop80_script_modified": False, "smoke_only": bool(args.smoke_only),
    }
    t0 = time.perf_counter()
    try:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        encoder = official_api.OfficialLatent(checkpoint, args.device, args.batch)
        payload["official_load"] = encoder.load_meta
        if args.smoke_only:
            payload["smoke"] = run_smoke(args, out, encoder)
        else:
            payload["dance"] = run_full(args, out, encoder)
        payload["status"] = "complete"
    except Exception as exc:
        payload["status"] = "failure"; payload["error_type"] = type(exc).__name__; payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
        (out / "failure.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    payload["elapsed_seconds"] = time.perf_counter() - t0
    payload["script_sha256_final"] = sha256(Path(__file__).resolve())
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    if payload.get("dance", {}).get("rows"):
        rows = payload["dance"]["rows"]
        with (out / "per_sequence.csv").open("w", newline="") as f:
            keys = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    # Immutable source copies are kept inside each run for provenance.
    prov = out / "provenance"; prov.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).resolve(), prov / Path(__file__).name)
    shutil.copy2(HERE / "mot_dance_jepa_real.py", prov / "mot_dance_jepa_real.py")
    shutil.copy2(HERE / "mot_ocsort_jepa_official_v2.py", prov / "mot_ocsort_jepa_official_v2.py")
    print(json.dumps({"status": payload["status"], "out": str(out), "elapsed_seconds": payload["elapsed_seconds"],
                      "error": payload.get("error"), "smoke": payload.get("smoke")},
                     indent=2, sort_keys=True, default=float))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
