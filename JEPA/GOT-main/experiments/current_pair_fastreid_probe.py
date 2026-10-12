#!/usr/bin/env python3
"""FastReID-backed current-pair JEPA representation probe.

This is a separate diagnostic from ``current_pair_jepa_probe.py``.  It keeps
the latter's one-to-one Hungarian sample construction and frozen
teacher/student objective, but replaces the handcrafted crop encoder with the
dataset-specific FastReID SBS-S50 checkpoint used by Deep-OC-SORT.  The
weights are *in-domain pretrained* and therefore an engineering upper bound,
not a no-leakage leaderboard claim.  Identity histories are oracle histories
for this representation probe; online tracker integration is explicitly out
of scope here.

The student is trained with a deterministic crop corruption, while evaluation
defaults to the uncorrupted detector crop (``--eval-corruption off``).  This
mirrors deployment, where detections are not deliberately corrupted.  A
teacher is fit on clean GT-box crops first and copied to a frozen NumPy branch;
the student regresses its stop-gradient output.  Controls and paired
bootstrap differences are written to JSON for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Reuse only the audited, data-loading/matching and generic two-phase model
# primitives.  All FastReID extraction and experiment orchestration below are
# new and live in this file; no old tracker script is modified.
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
import current_pair_jepa_probe as base  # noqa: E402


MOT_CONFIG_DEFAULT = Path("/data1/LWR/vranlee/M4FTMoveOut4Doing/HybridSORT/fast_reid/configs/MOT17/sbs_S50.yml")
DANCE_CONFIG_DEFAULT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT/fast_reid/configs/DanceTrack/sbs_S50.yml")
MOT_WEIGHTS_DEFAULT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT/pretrained/mot17_sbs_S50.pth")
DANCE_WEIGHTS_DEFAULT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/HybridSORT/pretrained/dancetrack_sbs_S50.pth")
# Use the complete vendored FastReID tree (the masa copy omits
# fastreid/data/datasets).  Checkpoint files remain the pinned masa assets.
FASTREID_ROOT = Path("/data1/LWR/vranlee/M4FTMoveOut4Doing/HybridSORT/fast_reid")
HYBRIDSORT_ROOT = FASTREID_ROOT.parent


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class FastReIDEncoder:
    """Batched wrapper around the vendored FastReID interface."""

    def __init__(self, config: Path, weights: Path, device: str, batch_size: int = 32):
        if not config.exists():
            raise FileNotFoundError(config)
        if not weights.exists():
            raise FileNotFoundError(weights)
        # The local interface imports torch._six, removed in torch 2.x.
        # Keep the compatibility alias process-local and record it in output.
        try:
            import torch
            if "torch._six" not in sys.modules:
                sys.modules["torch._six"] = types.SimpleNamespace(string_classes=(str,))
        except Exception as exc:
            raise RuntimeError("FastReID requires torch") from exc
        # The vendored package uses both ``fastreid`` and the historical
        # ``fast_reid.fastreid`` namespace; expose both its package directory
        # and the HybridSORT parent before importing the interface.
        sys.path.insert(0, str(HYBRIDSORT_ROOT))
        sys.path.insert(0, str(FASTREID_ROOT))
        from fast_reid_interfece import FastReIDInterface
        self.device = device
        self.batch_size = int(batch_size)
        t0 = time.perf_counter()
        self.model = FastReIDInterface(str(config), str(weights), device, batch_size=self.batch_size)
        self.init_seconds = time.perf_counter() - t0
        self.config = config; self.weights = weights
        self.calls = 0; self.crops = 0; self.seconds = 0.0

    def encode(self, image: np.ndarray, boxes: Sequence[np.ndarray]) -> np.ndarray:
        if not boxes:
            return np.zeros((0, 2048), np.float32)
        arr = np.asarray([list(map(float, b[:4])) + [1.0] for b in boxes], np.float32)
        t0 = time.perf_counter()
        feat = np.asarray(self.model.inference(image, arr), dtype=np.float32)
        self.seconds += time.perf_counter() - t0; self.calls += 1; self.crops += len(boxes)
        if feat.ndim != 2 or feat.shape[0] != len(boxes):
            raise RuntimeError("FastReID output shape {} for {} boxes".format(tuple(feat.shape), len(boxes)))
        if feat.shape[1] != 2048:
            raise RuntimeError("expected FastReID SBS-S50 dim 2048, got {}".format(feat.shape[1]))
        feat = base.normalize(np.nan_to_num(feat, nan=0., posinf=0., neginf=0.), axis=1).astype(np.float32)
        if not np.isfinite(feat).all():
            raise FloatingPointError("non-finite FastReID embedding")
        return feat

    def attach(self, seq_dir: Path, det: Dict[int, List[base.Det]], max_frames: int = 0) -> None:
        fs = base.image_files(seq_dir)
        for fr, rows in det.items():
            if fr < 1 or fr > len(fs) or (max_frames and fr > max_frames):
                continue
            im = cv2.imread(str(fs[fr - 1]))
            if im is None:
                continue
            valid = [d for d in rows]
            feats = self.encode(im, [d.box for d in valid])
            for d, f in zip(valid, feats):
                d.feat = f

    def clean(self, seq_dir: Path, gt: Dict[int, List[base.GT]], max_frames: int = 0) -> Dict[Tuple[int, int], np.ndarray]:
        fs = base.image_files(seq_dir); out: Dict[Tuple[int, int], np.ndarray] = {}
        for fr, rows in gt.items():
            if fr < 1 or fr > len(fs) or (max_frames and fr > max_frames):
                continue
            im = cv2.imread(str(fs[fr - 1]))
            if im is None:
                continue
            feats = self.encode(im, [g.box for g in rows])
            for g, f in zip(rows, feats):
                out[(fr, g.gid)] = f
        return out


def _install_fast_feature_hooks(encoder: FastReIDEncoder) -> None:
    """Route base.build_samples feature calls to FastReID in this process."""
    base.FEATURE_DIM = 2048
    base.attach_features = encoder.attach
    base.clean_gt_features = encoder.clean


def _record_scores(pred: base.CurrentPairJEPA, records: Sequence[dict], clean_eval: bool = False) -> Dict[str, np.ndarray]:
    pos = []; margin = []; r1 = []
    for rec in records:
        z = pred.predict_probe(rec, clean=clean_eval)
        y = base.normalize(rec["target"]); neg = base.normalize(rec["negative"])
        cand = base.normalize(np.asarray(rec["all_candidates"], np.float32)); pc = int(rec["positive_column"])
        sims = cand @ z
        pos.append(float(z @ y)); margin.append(float(z @ y - z @ neg)); r1.append(float(pc >= 0 and int(1 + np.sum(sims > sims[pc])) == 1))
    return {"positive_cosine": np.asarray(pos, np.float64), "margin": np.asarray(margin, np.float64), "recall_at_1": np.asarray(r1, np.float64)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 2000) -> Dict[str, float]:
    """Paired bootstrap CI for method A minus B on identical records."""
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0., "mean_diff": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    d = np.asarray(a[:n] - b[:n], np.float64)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(int(n_boot), n))
    means = d[idx].mean(axis=1)
    return {"n": float(n), "mean_diff": float(d.mean()), "ci95_low": float(np.percentile(means, 2.5)), "ci95_high": float(np.percentile(means, 97.5)), "paired_bootstrap_seed": int(seed), "n_boot": int(n_boot)}


def _make_args(ns, dataset: str):
    """Adapt the base loader's small Namespace contract."""
    return ns


def run_dataset(dataset: str, args, out_root: Path) -> Dict[str, object]:
    # Select dataset-specific in-domain checkpoint/config.  These are not
    # treated as causal/no-leakage evidence; hashes and warning metadata are
    # mandatory in the result.
    if dataset == "mot17":
        config, weights = Path(args.mot_config), Path(args.mot_weights)
    else:
        config, weights = Path(args.dance_config), Path(args.dance_weights)
    encoder = FastReIDEncoder(config, weights, args.device, args.batch_size)
    _install_fast_feature_hooks(encoder)
    train_items, eval_items, detector_meta = base.load_items(dataset, args)
    train_arr, train_rec, train_meta = base.build_samples(train_items, args.max_frames, args.seed, args.max_samples)
    eval_arr, eval_rec, eval_meta = base.build_samples(eval_items, args.max_frames, args.seed + 7001, min(args.max_samples, 12000))
    if not (int(train_meta.get("duplicate_count", -1)) == 0 and int(eval_meta.get("duplicate_count", -1)) == 0):
        raise AssertionError("one-to-one matching audit failed")
    for bundle in (train_arr, eval_arr):
        if any(not np.isfinite(v).all() for v in bundle.values()):
            raise FloatingPointError("non-finite FastReID probe field")
    # Deployment default: train with corruption, evaluate the raw detector
    # crop.  ``on`` is retained as a robustness ablation.
    if args.eval_corruption == "off":
        for r in eval_rec:
            r["corrupt"] = np.asarray(r["candidate"], np.float32).copy()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    predictors: Dict[str, base.CurrentPairJEPA] = {}; train_stats = {}
    for m in methods:
        p = base.CurrentPairJEPA(m, args.seed, args.epochs)
        if m == "random_frozen": p.fit_random()
        else: p.fit(train_arr)
        if not p.train_stats.get("finite", False):
            raise FloatingPointError("non-finite predictor stats for {}".format(m))
        predictors[m] = p; train_stats[m] = p.train_stats
    validation = {m: base.metric_predictor(p, eval_rec, clean_eval=(m == "both_clean")) for m, p in predictors.items()}
    if any(not v.get("candidate_order_invariant", False) for v in validation.values()):
        raise AssertionError("candidate permutation audit failed")
    train_diag = {m: base.metric_predictor(p, train_rec, clean_eval=False) for m, p in predictors.items()}
    validation_by_sequence = {}
    for seq in sorted({str(r["sequence"]) for r in eval_rec}):
        rr = [r for r in eval_rec if str(r["sequence"]) == seq]
        validation_by_sequence[seq] = {m: base.metric_predictor(p, rr, clean_eval=(m == "both_clean")) for m, p in predictors.items()}
    teacher = next((p for m, p in predictors.items() if m != "random_frozen"), None)
    teacher_validation = base.metric_teacher(teacher, eval_rec) if teacher is not None else None
    teacher_train = base.metric_teacher(teacher, train_rec) if teacher is not None else None
    teacher_by_sequence = {seq: base.metric_teacher(teacher, [r for r in eval_rec if str(r["sequence"]) == seq]) for seq in sorted({str(r["sequence"]) for r in eval_rec})} if teacher is not None else {}
    # Paired bootstrap uses exactly the same eval records and compares the
    # intended JEPA branch with each control, not independently sampled sets.
    scores = {m: _record_scores(p, eval_rec, clean_eval=(m == "both_clean")) for m, p in predictors.items()}
    paired = {}
    if "current_jepa" in scores:
        for ctrl in ("current_only", "full_mask", "no_teacher", "random_frozen", "both_clean"):
            if ctrl in scores:
                paired["current_jepa_minus_" + ctrl] = {metric: paired_bootstrap(scores["current_jepa"][metric], scores[ctrl][metric], args.seed + 8100 + i) for i, metric in enumerate(("recall_at_1", "margin", "positive_cosine"))}
    # A deliberately conservative, predeclared gate for the next lagged
    # tracker stage.  It is informational here; no thresholds are tuned from
    # the held-out split.  A tracker may proceed only if the frozen teacher is
    # itself useful and JEPA beats both direct/no-teacher and random controls.
    teacher_ok = bool(teacher_validation and teacher_validation.get("target_cosine_mean", -1.) >= .50 and teacher_validation.get("finite", False))
    def _wins(metric: str, ctrl: str) -> bool:
        return bool("current_jepa" in validation and ctrl in validation and validation["current_jepa"].get(metric, -1e9) > validation[ctrl].get(metric, -1e9))
    quality_gate = {
        "teacher_target_cosine_threshold": .50, "teacher_target_cosine_mean": float(teacher_validation.get("target_cosine_mean", float("nan"))) if teacher_validation else None,
        "teacher_useful": teacher_ok, "current_beats_no_teacher_r1": _wins("recall_at_1", "no_teacher"),
        "current_beats_random_r1": _wins("recall_at_1", "random_frozen"), "current_beats_no_teacher_margin": _wins("hard_negative_margin_mean", "no_teacher"),
        "proceed_lagged_tracker": bool(teacher_ok and _wins("recall_at_1", "no_teacher") and _wins("recall_at_1", "random_frozen")),
        "decision_rule_preregistered": True,
    }
    protocol = {
        "schema": "current-pair-fastreid-probe-v1", "teacher": "clean GT crop target input, fit first then frozen (stop-gradient to student)",
        "student": "corrupted detector crop during training; eval corruption={} (deployment default off)".format(args.eval_corruption), "lag": 1,
        "matching": "per-frame Hungarian IoU>=0.30, no detection reuse", "duplicate_count": 0,
        "history_oracle_for_probe": True, "history_frames_strictly_before_current": True, "student_gt_inputs": False,
        "controls": ["current_jepa", "current_only", "full_mask", "no_teacher", "random_frozen", "both_clean"],
        "in_domain_pretrained_upper_bound": True,
        "warning": "MOT17/DanceTrack SBS-S50 weights are dataset-specific pretrained checkpoints; this diagnostic is not a no-leakage leaderboard/causal claim.",
        "deployment_corruption_default": "off", "candidate_order_audit": "true list permutation + positive-column remap; pair latent invariant",
    }
    return {
        "dataset": dataset, "detector": detector_meta, "train_sequences": [x[0] for x in train_items], "eval_sequences": [x[0] for x in eval_items],
        "train_samples": len(train_rec), "eval_samples": len(eval_rec), "train_meta": train_meta, "eval_meta": eval_meta,
        "predictor_stats": train_stats, "validation": validation, "validation_by_sequence": validation_by_sequence,
        "teacher_train": teacher_train, "teacher_validation": teacher_validation, "teacher_validation_by_sequence": teacher_by_sequence,
        "train_diagnostics": train_diag, "paired_bootstrap": paired, "quality_gate": quality_gate, "protocol": protocol,
        "fastreid": {"config": str(config), "config_sha256": file_sha256(config), "weights": str(weights), "weights_sha256": file_sha256(weights), "device": args.device, "batch_size": args.batch_size, "init_seconds": encoder.init_seconds, "inference_calls": encoder.calls, "inference_crops": encoder.crops, "inference_seconds": encoder.seconds, "feature_dim": 2048},
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("mot17", "dance", "both"), default="mot17")
    p.add_argument("--mot-train", default="MOT17-02-FRCNN,MOT17-04-FRCNN,MOT17-09-FRCNN")
    p.add_argument("--mot-eval", default="MOT17-05-FRCNN,MOT17-11-FRCNN")
    p.add_argument("--dance-train", default="dancetrack0001,dancetrack0002,dancetrack0006")
    p.add_argument("--dance-eval", default="dancetrack0004,dancetrack0007,dancetrack0014")
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=30000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-corruption", choices=("off", "on"), default="off")
    p.add_argument("--mot-config", default=str(MOT_CONFIG_DEFAULT)); p.add_argument("--dance-config", default=str(DANCE_CONFIG_DEFAULT))
    p.add_argument("--mot-weights", default=str(MOT_WEIGHTS_DEFAULT)); p.add_argument("--dance-weights", default=str(DANCE_WEIGHTS_DEFAULT))
    p.add_argument("--methods", default="current_jepa,current_only,full_mask,no_teacher,both_clean,random_frozen")
    p.add_argument("--out-dir", default="experiments/results/current_pair_fastreid_probe_v1")
    return p


def main() -> None:
    args = build_parser().parse_args(); base.seed_all(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "current-pair-fastreid-probe-v1", "command": " ".join(sys.argv), "cwd": os.getcwd(), "generated_unix": time.time(), "script_sha256": file_sha256(Path(__file__)), "base_script_sha256": file_sha256(EXP_DIR / "current_pair_jepa_probe.py"), "seed": args.seed, "max_frames": args.max_frames, "methods": [m for m in args.methods.split(",") if m]}
    for ds in (("mot17", "dance") if args.dataset == "both" else (args.dataset,)):
        payload[ds] = run_dataset(ds, args, out)
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({ds: payload[ds]["validation"] for ds in ("mot17", "dance") if ds in payload}, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
