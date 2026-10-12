#!/usr/bin/env python3
"""Train-only pre-registration diagnostic for GOT-JEPA ambiguity fusion.

The script deliberately stops before loading validation detections or
annotations.  It reuses the immutable 256-D pooled GOT-JEPA cache, trains the
small history-only predictor on the three fixed DanceTrack train sequences,
and derives one native top1--top2 cost-margin threshold from the suffix
(frames 61--120) at p25.  The recommended strict gate is fixed here as
``quality >= .80`` and ``sigmoid(log_variance) <= .40`` (plus the existing
history miss limit).  The resulting JSON is the threshold/affected-event
record to review before any 120-frame evaluation is launched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

# Importing the isolated runner installs NumPy compatibility aliases and
# exposes cache/cost helpers; it does not construct the encoder or read eval.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mot_dance_jepa_got_ambiguity as amb  # noqa: E402
import mot_dance_jepa_real as dance  # noqa: E402


TRAIN = ("dancetrack0001", "dancetrack0002", "dancetrack0006")
MAX_FRAMES = 120
SEED = 17
EPOCHS = 8
MAX_SAMPLES = 12000
SUFFIX_START = 61
MARGIN_QUANTILE = 25.0
QUALITY_GATE = 0.80
UNCERTAINTY_PROB_GATE = 0.40
METHODS = ("ambiguity_jepa", "no_teacher", "random_frozen")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n")


def strict_gate(track, pred):
    q = float(pred[1]); up = amb.sigmoid(float(pred[2]))
    ok = bool(q >= QUALITY_GATE and up <= UNCERTAINTY_PROB_GATE and int(track.misses) <= 4)
    return ok, q, up


def pct(values: List[float], qs=(0, 10, 25, 50, 75, 90, 100)) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    names = ("min", "p10", "p25", "p50", "p75", "p90", "max")
    return {n: float(v) for n, v in zip(names, np.percentile(arr, qs))} if arr.size else {n: None for n in names}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=amb.CHECKPOINT_DEFAULT)
    ap.add_argument("--cache-dir", default=amb.CACHE_DEFAULT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve(); cache_root = Path(args.cache_dir).resolve()
    t0 = time.perf_counter()
    base = {
        "schema": "dance-got-jepa-ambiguity-train-diagnostic-v2", "status": "started",
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
        "platform": platform.platform(), "train_sequences": list(TRAIN), "max_frames": MAX_FRAMES,
        "suffix_start_frame": SUFFIX_START, "suffix_definition": "last 50% of first 120 frames",
        "margin_quantile": MARGIN_QUANTILE, "quality_gate": QUALITY_GATE,
        "uncertainty_measure": "sigmoid(log_variance)", "uncertainty_prob_gate": UNCERTAINTY_PROB_GATE,
        "seed": SEED, "epochs": EPOCHS, "max_samples": MAX_SAMPLES,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "cache_dir": str(cache_root), "eval_gt_read": False, "eval_detections_read": False,
        "eval_tracking_run": False, "threshold_tuning_on_eval": False,
    }
    try:
        amb.seed_all(SEED)
        train_gt = dance.limit_frame_dicts(
            dance.parse_dance_coco(dance.DANCE_ROOT / "annotations/train.json", "train", list(TRAIN)), MAX_FRAMES)
        train_items = []; cache_meta = {}
        for seq in TRAIN:
            item, meta = amb.load_train_item(seq, MAX_FRAMES, cache_root, base["checkpoint_sha256"], train_gt)
            train_items.append(item); cache_meta[seq] = meta
        samples, sample_meta = amb.got.build_temporal_samples_got(train_items, MAX_FRAMES, SEED, MAX_SAMPLES)
        predictors = {}
        for method, mode in (("ambiguity_jepa", "jepa_temporal"), ("no_teacher", "no_teacher")):
            p = dance.TemporalPredictor(256, mode, SEED, EPOCHS, args.device)
            p.fit(samples); predictors[method] = p
        p = dance.TemporalPredictor(256, "random_frozen", SEED, 0, "cpu"); p.fit_random(); predictors["random_frozen"] = p

        margins: List[float] = []
        events: List[Dict[str, object]] = []
        pred_stats = {m: {"quality": [], "uncertainty_prob": [], "gate_pass": 0,
                          "prefers_top2": 0, "both_viable": 0, "ambiguous_and_gate": 0,
                          "potential_changed_events": 0} for m in METHODS}
        # Raw tracker state is advanced through all 120 train frames; only
        # suffix rows contribute to the threshold/affected-event statistic.
        for seq, seq_dir, gt, det, teacher in train_items:
            iw, ih = dance.sequence_wh(seq_dir)
            probe = dance.SimpleMOT("raw", None, seed=SEED, image_width=iw, image_height=ih)
            for fr in sorted(det):
                active, native, pred_boxes = amb.native_cost_matrix(probe, int(fr), det[fr])
                if int(fr) >= SUFFIX_START:
                    for i, tr in enumerate(probe.tracks):
                        finite = np.where(np.isfinite(native[i]) & (native[i] < 1e5))[0]
                        if finite.size < 2:
                            continue
                        order = finite[np.argsort(native[i, finite], kind="mergesort")]
                        j1, j2 = int(order[0]), int(order[1])
                        margin = float(native[i, j2] - native[i, j1]); margins.append(margin)
                        row = {"sequence": seq, "frame": int(fr), "track_index": int(i),
                               "track_id": int(tr.tid), "top1": j1, "top2": j2,
                               "native_top1": float(native[i, j1]), "native_top2": float(native[i, j2]),
                               "native_margin": margin}
                        for method in METHODS:
                            pp = predictors[method].predict(probe._temporal_tokens(tr, int(fr)))
                            ok, q, up = strict_gate(tr, pp)
                            c1 = amb.predictor_cost(probe, tr, pp, active[j1], pred_boxes[i])
                            c2 = amb.predictor_cost(probe, tr, pp, active[j2], pred_boxes[i])
                            pref2 = bool(c2 + 1e-7 < c1)
                            both = bool(native[i, j1] < amb.MAX_ASSIGN_COST and native[i, j2] < amb.MAX_ASSIGN_COST)
                            st = pred_stats[method]; st["quality"].append(q); st["uncertainty_prob"].append(up)
                            st["gate_pass"] += int(ok); st["prefers_top2"] += int(pref2); st["both_viable"] += int(both)
                            row[method] = {"quality": q, "uncertainty_prob": up, "gate_pass": ok,
                                           "predictor_cost_top1": float(c1), "predictor_cost_top2": float(c2),
                                           "prefers_top2": pref2, "both_viable": both}
                        events.append(row)
                probe.step(int(fr), det[fr])
        if not margins:
            raise RuntimeError("no train suffix rows with two native candidates")
        marr = np.asarray(margins, dtype=np.float64)
        threshold = float(np.percentile(marr, MARGIN_QUANTILE))
        for method in METHODS:
            st = pred_stats[method]
            # Derive thresholded counts after the single fixed p25 statistic;
            # no eval values enter this calculation.
            for row in events:
                x = row[method]
                if row["native_margin"] <= threshold and x["gate_pass"]:
                    st["ambiguous_and_gate"] += 1
                    if x["both_viable"] and x["prefers_top2"]:
                        st["potential_changed_events"] += 1
            for key in ("quality", "uncertainty_prob"):
                vals = st.pop(key)
                st[key + "_summary"] = pct(vals)
            st["total_suffix_rows"] = len(events)
            st["ambiguous_rows_at_p25"] = int(np.sum(marr <= threshold))
            st["gate_pass_rate"] = float(st["gate_pass"] / max(len(events), 1))
            st["potential_changed_fraction_of_ambiguous"] = float(st["potential_changed_events"] / max(st["ambiguous_rows_at_p25"], 1))
        result = dict(base)
        result.update({"status": "complete", "margin_threshold_p25": threshold,
                       "margin_count": int(marr.size), "margin_summary": pct(margins),
                       "cache_meta": cache_meta, "sample_meta": sample_meta,
                       "predictor_stats": {m: predictors[m].train_stats for m in METHODS},
                       "affected_event_diagnostic": pred_stats, "events": events,
                       "events_sha256": hashlib.sha256(json.dumps(events, sort_keys=True, default=float, separators=(",", ":")).encode()).hexdigest(),
                       "elapsed_seconds": time.perf_counter() - t0,
                       "next_action": "review this file; no evaluation was launched"})
        dump(out / "train_only_diagnostic.json", result)
        # Small marker with source/checkpoint hashes keeps the directory
        # self-describing without duplicating large feature arrays.
        prov = {"schema": "dance-got-jepa-ambiguity-train-diagnostic-provenance-v1",
                "diagnostic_sha256": sha256(out / "train_only_diagnostic.json"),
                "script_sha256": sha256(Path(__file__).resolve()),
                "runner_helper_sha256": sha256(HERE / "mot_dance_jepa_got_ambiguity.py"),
                "checkpoint_sha256": base["checkpoint_sha256"], "cache_dir": str(cache_root),
                "eval_gt_read": False, "eval_tracking_run": False}
        dump(out / "provenance.json", prov)
        print(json.dumps({"status": "complete", "out": str(out), "margin_threshold_p25": threshold,
                          "margin_summary": result["margin_summary"],
                          "affected_event_diagnostic": pred_stats}, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        result = dict(base); result.update({"status": "failure", "error": repr(exc), "elapsed_seconds": time.perf_counter() - t0})
        dump(out / "train_only_diagnostic.json", result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
