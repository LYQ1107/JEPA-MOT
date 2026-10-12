#!/usr/bin/env python3
"""Pre-registered gate sweep for the v4 MOT-JEPA lifecycle adapter.

This is an exploratory, bounded follow-up to the v4 lifecycle run.  It keeps
the corrected lifecycle rule and the frozen v3 student checkpoint, and varies
only the observable repair gate.  Candidate A is the original v4 gate; B/C
restrict gap/ambiguity repairs to short, high-confidence matches; D disables
gap repairs and permits only high-confidence ambiguity repairs.  No GT is
read by the tracker.  Fit-video suffix runs provide repair-count/distribution
audits before the no-GT 05/11 run; official TrackEval is invoked separately.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mot_jepa_adapter_v3_direct as v3  # noqa: E402
import mot_jepa_adapter_v3_stress_protocol as stress  # noqa: E402


FIT_SEQUENCES = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN")
EVAL_SEQUENCES = ("MOT17-05-FRCNN", "MOT17-11-FRCNN")
DEFAULT_SOURCE = Path("/data2/usr_for_deadline/mot_jepa_adapter_v3_stress_120")
DEFAULT_CACHE = Path("/data2/usr_for_deadline/mot_jepa_adapter_v3_direct_120/cache")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True,
                               ensure_ascii=False, default=float) + "\n")


@dataclass
class State:
    cid: int
    native_id: int
    last_frame: int
    box: np.ndarray
    score: float
    history: List[Tuple[int, np.ndarray, np.ndarray, float]]
    prototype: Optional[np.ndarray] = None
    visibility: float = 1.0
    uncertainty: float = 0.0


def _margin(values: np.ndarray, j: int) -> float:
    if len(values) <= 1:
        return float("inf")
    return float(values[j] - np.max(np.delete(values, j)))


# The four configurations are written in the artifact before any data is
# loaded.  A is byte/logic-equivalent to v4's fixed thresholds; B/C/D are the
# requested exploratory candidates and are not selected using evaluation GT.
CANDIDATES: Dict[str, Dict[str, object]] = {
    "A": {
        "description": "original v4 gate",
        "allow_gap": True, "max_gap": 8,
        "gap_iou_min": .02, "gap_app_min": .25, "gap_margin_min": 0.,
        "gap_score_or": True,
        "allow_ambiguous": True, "ambiguous_fused_margin": .10,
        "ambiguous_iou_min": .05, "ambiguous_app_min": .20,
        "ambiguous_margin_min": 0.,
    },
    "B": {
        "description": "gap<=3, q>=.20, app>=.60, row/col margin>=.05",
        "allow_gap": True, "max_gap": 3,
        "gap_iou_min": .20, "gap_app_min": .60, "gap_margin_min": .05,
        "gap_score_or": False,
        "allow_ambiguous": True, "ambiguous_fused_margin": .10,
        "ambiguous_iou_min": .20, "ambiguous_app_min": .60,
        "ambiguous_margin_min": .05,
    },
    "C": {
        "description": "gap<=3, q>=.20, app>=.80, row/col margin>=.05",
        "allow_gap": True, "max_gap": 3,
        "gap_iou_min": .20, "gap_app_min": .80, "gap_margin_min": .05,
        "gap_score_or": False,
        "allow_ambiguous": True, "ambiguous_fused_margin": .10,
        "ambiguous_iou_min": .20, "ambiguous_app_min": .80,
        "ambiguous_margin_min": .05,
    },
    "D": {
        "description": "ambiguous-only, q>=.20, app>=.60, row/col margin>=.05",
        "allow_gap": False, "max_gap": 0,
        "gap_iou_min": .20, "gap_app_min": .60, "gap_margin_min": .05,
        "gap_score_or": False,
        "allow_ambiguous": True, "ambiguous_fused_margin": .10,
        "ambiguous_iou_min": .20, "ambiguous_app_min": .60,
        "ambiguous_margin_min": .05,
    },
}


def online_track_sweep(
    p: v3.v2lib.PreparedSequence,
    cfg: Dict[str, object],
    model: Optional[v3.PrototypeNet],
    max_age: int = 30,
    disable_repairs: bool = False,
) -> Tuple[Dict[int, List[Tuple[np.ndarray, int, float]]], Dict[str, object]]:
    """One trained-adapter pass with a candidate-specific repair gate.

    The implementation follows v4_lifecycle exactly up to the two gate
    checks.  In particular, stale canonical IDs are not considered occupied
    when a new native row is born, and the prototype is updated after frame t
    then consumed only at t+1.
    """
    nframes = v3.v2lib.max_frames_for(p)
    states: List[State] = []
    next_fresh = 1_000_000
    out: Dict[int, List[Tuple[np.ndarray, int, float]]] = {}
    d: Dict[str, object] = {
        "candidate": cfg.get("name"), "repairs_disabled": bool(disable_repairs),
        "lag": "t->t+1", "eval_gt_read_by_tracker": False,
        "native_continuations": 0, "ambiguous_opened": 0,
        "gap_candidates": 0, "accepted_repairs": 0, "accepted_ambiguous": 0,
        "accepted_gap": 0, "gate_rejects": 0, "filter_updates": 0,
        "candidate_pairs": 0, "repair_events": [],
    }
    for fr in range(1, nframes + 1):
        cur = p.raw.get(fr, [])
        # Deduplicate live cids while retaining the most recent state.  A
        # stale predecessor may still be considered for a gap repair, but two
        # live states must never emit the same ID in one timestep.
        by_cid: Dict[int, State] = {}
        for old in states:
            if fr - old.last_frame > max_age:
                continue
            prev = by_cid.get(int(old.cid))
            if prev is None or old.last_frame > prev.last_frame:
                by_cid[int(old.cid)] = old
        active = list(by_cid.values())
        used_s: set = set(); used_r: set = set()
        matches: List[Tuple[int, int, str, float, float, float, float]] = []
        cvec = (np.stack([v3.pooled_map(r["feat_map"]) for r in cur]).astype(np.float32)
                if cur else np.empty((0, v3.FEATURE_DIM), np.float32))
        iou_cache = {i: np.asarray([v3.box_iou(s.box, r["box"]) for r in cur], np.float32)
                     for i, s in enumerate(active)}
        app_cache: Dict[int, np.ndarray] = {}
        for i, s in enumerate(active):
            app_cache[i] = (np.zeros(len(cur), np.float32) if s.prototype is None or not len(cur)
                            else cvec.dot(v3.norm(s.prototype)))

        ambiguous: set = set()
        native_opts: List[Tuple[float, int, int]] = []
        for i, s in enumerate(active):
            same = [j for j, r in enumerate(cur)
                    if int(r["native_id"]) == int(s.native_id) and iou_cache[i][j] >= .20]
            if not same:
                continue
            j = max(same, key=lambda q: float(iou_cache[i][q]))
            fused = .45 * iou_cache[i] + .55 * ((app_cache[i] + 1.) / 2.)
            mar = _margin(fused, j)
            gap = fr - s.last_frame
            if (not disable_repairs and bool(cfg.get("allow_ambiguous", True))
                    and mar < float(cfg.get("ambiguous_fused_margin", .10))
                    and s.prototype is not None and len(cur) > 1):
                if (float(s.visibility) >= .25 and abs(float(s.uncertainty)) <= 1.0):
                    ambiguous.add(i); d["ambiguous_opened"] = int(d["ambiguous_opened"]) + 1
                else:
                    d["gate_rejects"] = int(d["gate_rejects"]) + 1
            if i not in ambiguous and gap <= 1:
                native_opts.append((float(iou_cache[i][j]), i, j))
            elif gap > 1 and bool(cfg.get("allow_gap", True)):
                d["gap_candidates"] = int(d["gap_candidates"]) + 1
        for _q, i, j in sorted(native_opts, reverse=True):
            if i in used_s or j in used_r:
                continue
            matches.append((i, j, "native", 0., 0., 0., 0.)); used_s.add(i); used_r.add(j)
            d["native_continuations"] = int(d["native_continuations"]) + 1

        rem_s: List[int] = []
        if not disable_repairs:
            for i, s in enumerate(active):
                if i in used_s or s.prototype is None:
                    continue
                gap = fr - s.last_frame
                if i in ambiguous or (bool(cfg.get("allow_gap", True))
                                      and 2 <= gap <= int(cfg.get("max_gap", 8))):
                    rem_s.append(i)
        rem_r = [j for j in range(len(cur)) if j not in used_r]
        d["candidate_pairs"] = int(d["candidate_pairs"]) + len(rem_s) * len(rem_r)
        if rem_s and rem_r:
            fused = np.full((len(rem_s), len(rem_r)), -np.inf, np.float32)
            for a, i in enumerate(rem_s):
                for b, j in enumerate(rem_r):
                    fused[a, b] = .45 * float(iou_cache[i][j]) + .55 * float((app_cache[i][j] + 1.) / 2.)
            rr, cc = linear_sum_assignment(-fused)
            for a, b in zip(rr, cc):
                a, b = int(a), int(b); i, j = rem_s[a], rem_r[b]
                q = float(iou_cache[i][j]); ap = float(app_cache[i][j])
                row_mar = _margin(fused[a], b); col_mar = _margin(fused[:, b], a)
                is_amb = i in ambiguous
                if is_amb:
                    ok = (q >= float(cfg.get("ambiguous_iou_min", .05))
                          and ap >= float(cfg.get("ambiguous_app_min", .20))
                          and row_mar >= float(cfg.get("ambiguous_margin_min", 0.)))
                    # ``col_mar`` is checked separately to retain the
                    # requested row/column margin semantics.
                    ok = ok and col_mar >= float(cfg.get("ambiguous_margin_min", 0.))
                else:
                    score_ok = ((q >= float(cfg.get("gap_iou_min", .02))
                                 or ap >= float(cfg.get("gap_app_min", .25)))
                                if bool(cfg.get("gap_score_or", False)) else
                                (q >= float(cfg.get("gap_iou_min", .02))
                                 and ap >= float(cfg.get("gap_app_min", .25))))
                    ok = (bool(cfg.get("allow_gap", True)) and score_ok
                          and row_mar >= float(cfg.get("gap_margin_min", 0.))
                          and col_mar >= float(cfg.get("gap_margin_min", 0.)))
                if ok:
                    kind = "ambiguous" if is_amb else "gap"
                    matches.append((i, j, kind, q, ap, row_mar, col_mar))
                    used_s.add(i); used_r.add(j)
                    d["accepted_repairs"] = int(d["accepted_repairs"]) + 1
                    d["accepted_ambiguous"] = int(d["accepted_ambiguous"]) + int(is_amb)
                    d["accepted_gap"] = int(d["accepted_gap"]) + int(not is_amb)
                    d["repair_events"].append({"frame": int(fr), "kind": kind,
                                                "gap": int(fr - active[i].last_frame),
                                                "q": q, "app": ap,
                                                "row_margin": row_mar, "col_margin": col_mar,
                                                "native_id_before": int(active[i].native_id),
                                                "native_id_after": int(cur[j]["native_id"]),
                                                "cid": int(active[i].cid)})
                else:
                    d["gate_rejects"] = int(d["gate_rejects"]) + 1

        updated: List[Tuple[State, np.ndarray]] = []
        for i, j, _kind, _q, _ap, _rm, _cm in matches:
            s, r = active[i], cur[j]
            box = np.asarray(r["box"], np.float32); vec = v3.pooled_map(r["feat_map"])
            s.history = (s.history + [(fr, np.asarray(r["feat_map"], np.float32), box.copy(),
                                        float(r["score"]))])[-v3.HISTORY_K:]
            s.box = box.copy(); s.last_frame = fr; s.native_id = int(r["native_id"]); s.score = float(r["score"])
            x = v3.feature_input(s.history[:-1], vec, box, p.width, p.height, fr, s.score, 0.)
            updated.append((s, x)); r["cid_sweep"] = s.cid

        # Lifecycle-safe birth: stale cids do not reserve native labels, but
        # duplicate labels in this current frame do.
        occupied = {int(active[i].cid) for i in used_s}
        for j, r in enumerate(cur):
            if j in used_r:
                continue
            native = int(r["native_id"])
            if native > 0 and native not in occupied:
                cid = native
                states = [old for old in states if int(old.cid) != cid or old.last_frame >= fr]
            else:
                cid = next_fresh; all_cids = {int(old.cid) for old in states}
                while cid in occupied or cid in all_cids:
                    next_fresh += 1; cid = next_fresh
            next_fresh = max(next_fresh, cid + 1); occupied.add(cid)
            vec = v3.pooled_map(r["feat_map"])
            s = State(cid, native, fr, np.asarray(r["box"], np.float32).copy(), float(r["score"]),
                      [(fr, np.asarray(r["feat_map"], np.float32), np.asarray(r["box"], np.float32).copy(),
                        float(r["score"]))])
            states.append(s); r["cid_sweep"] = cid
            updated.append((s, v3.feature_input([], vec, r["box"], p.width, p.height, fr,
                                                  float(r["score"]), 0.)))
        for s, x in updated:
            if model is None:
                s.prototype = v3.pooled_map(s.history[-1][1]); s.visibility = 1.; s.uncertainty = 0.
            else:
                pred, vlog, ulog = model.predict(np.asarray(x, np.float32)[None])
                s.prototype = pred[0]; s.visibility = float(1. / (1. + np.exp(-vlog[0]))); s.uncertainty = float(ulog[0])
            d["filter_updates"] = int(d["filter_updates"]) + 1
        out[fr] = [(r["box"], int(r.get("cid_sweep", r["native_id"])), float(r["score"])) for r in cur]

    raw_ids = [int(r["native_id"]) for fr in range(1, nframes + 1) for r in p.raw.get(fr, [])]
    ids = [int(t[1]) for fr in range(1, nframes + 1) for t in out.get(fr, [])]
    d["id_differences_vs_raw"] = int(sum(a != b for a, b in zip(raw_ids, ids)))
    return out, d


def write_tracks(path: Path, rows: Dict[int, List[Tuple[np.ndarray, int, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for fr in sorted(rows):
            for box, tid, score in rows[fr]:
                x1, y1, x2, y2 = map(float, box)
                f.write("{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.6f},-1,-1,-1\n".format(
                    int(fr), int(tid), x1, y1, max(0., x2 - x1), max(0., y2 - y1), float(score)))


def _suffix_events(diag: Dict[str, object], fit_end: int) -> List[dict]:
    return [e for e in diag.get("repair_events", []) if int(e.get("frame", 0)) >= int(fit_end)]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=stress.CHECKPOINT_DEFAULT)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--fit-end-frame", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    source = Path(args.source_dir).resolve(); cache = Path(args.cache_dir).resolve(); out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    payload: Dict[str, object] = {
        "schema": "mot-jepa-adapter-v4-gate-sweep-v1", "status": "started",
        "dataset": "MOT17", "fit_sequences": list(FIT_SEQUENCES),
        "eval_sequences": list(EVAL_SEQUENCES), "max_frames": int(args.max_frames),
        "fit_end_frame": int(args.fit_end_frame), "quality_suffix": "fit sequences, t>=fit_end; read_gt=False",
        "source_dir": str(source), "cache_dir": str(cache), "checkpoint": str(Path(args.checkpoint).resolve()),
        "candidate_order": ["A", "B", "C", "D"], "candidates_preregistered": CANDIDATES,
        "lifecycle_fix": "v4: stale max_age cids do not reserve native labels; current-frame occupied only",
        "association_lag": "t->t+1", "candidate_in_predictor": False,
        "eval_gt_read_by_tracker": False, "thresholds_common": {
            "ambiguous_fused_margin": .10, "visibility_min": .25, "uncertainty_max": 1.0,
        }, "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
    }
    try:
        source_gate = json.loads((source / "quality_gate.json").read_text())
        if source_gate.get("stress_gate", {}).get("status") != "pass":
            raise RuntimeError("source stress gate is not pass")
        payload["source_stress_gate_status"] = source_gate.get("stress_gate", {}).get("status")
        payload["source_native_gate_status"] = source_gate.get("original_native_hard_gate", {}).get("status")
        src_train = json.loads((source.parent / "mot_jepa_adapter_v3_direct_120" / "train_results.json").read_text())
        expected = src_train.get("official_load", {}).get("checkpoint_sha256")
        if expected and sha256(Path(args.checkpoint).resolve()) != expected:
            raise RuntimeError("checkpoint hash differs from source")

        student_path = source.parent / "mot_jepa_adapter_v3_direct_120" / "checkpoints" / "trained_direct_prototype.pt"
        student = stress.load_prototype(student_path)
        payload["student_checkpoint_sha256"] = sha256(student_path)
        extractor = v3.official_api.OfficialLatent(Path(args.checkpoint), args.device, 8)
        payload["official_load"] = extractor.load_meta

        # ---- train-only suffix repair audit (no GT, no eval sequence) ----
        fit_prepared = [v3.v2lib.prepare_sequence(seq, int(args.max_frames), extractor, cache, read_gt=False)
                        for seq in FIT_SEQUENCES]
        train_audit: Dict[str, object] = {}
        for name, raw_cfg in CANDIDATES.items():
            cfg = dict(raw_cfg); cfg["name"] = name
            per_seq = {}; all_events = []
            for p in fit_prepared:
                _, diag = online_track_sweep(p, cfg, student, disable_repairs=False)
                ev = _suffix_events(diag, int(args.fit_end_frame)); all_events.extend(ev)
                per_seq[p.name] = {"events": ev, "accepted_repairs": len(ev),
                                   "accepted_gap": sum(e["kind"] == "gap" for e in ev),
                                   "accepted_ambiguous": sum(e["kind"] == "ambiguous" for e in ev),
                                   "id_differences_vs_raw": diag.get("id_differences_vs_raw")}
            train_audit[name] = {"config": cfg, "events": all_events,
                                 "accepted_repairs_suffix": len(all_events), "per_sequence": per_seq,
                                 "gt_read": False}
        payload["train_suffix_repair_audit"] = train_audit
        del fit_prepared; gc.collect()

        # ---- evaluation streams are opened only after the train-only audit ----
        eval_prepared = [v3.v2lib.prepare_sequence(seq, int(args.max_frames), extractor, cache, read_gt=False)
                         for seq in EVAL_SEQUENCES]
        payload["eval_manifest"] = {p.name: {"frames": int(v3.v2lib.max_frames_for(p)),
                                               "det_rows": int(sum(len(v) for v in p.det.values())),
                                               "metadata": p.metadata} for p in eval_prepared}

        # No-repair invariant is common to all candidates and runs before
        # candidate tracks are written.  This is an actual 05/11 check with
        # GT disabled, not a synthetic assumption.
        invariant = {}
        for p in eval_prepared:
            cfg = dict(CANDIDATES["A"]); cfg["name"] = "A"
            _, dg = online_track_sweep(p, cfg, student, disable_repairs=True)
            invariant[p.name] = int(dg.get("id_differences_vs_raw", -1))
            if invariant[p.name] != 0:
                raise AssertionError("no-repair identity invariant failed: {}".format(invariant))
        payload["no_repair_identity_invariant"] = {"status": "pass", "per_sequence": invariant}

        rows_meta = []; diagnostics: Dict[str, object] = {}
        # Raw rows are written once; all candidates use exactly the same boxes.
        for p in eval_prepared:
            raw_rows = {fr: [(r["box"], int(r["native_id"]), float(r["score"])) for r in rs]
                        for fr, rs in p.raw.items()}
            path = out / "mot17" / "tracks" / "raw" / (p.name + ".txt"); write_tracks(path, raw_rows)
            rows_meta.append({"sequence": p.name, "method": "raw", "frames": int(args.max_frames),
                              "det_rows": int(sum(len(v) for v in p.det.values())),
                              "track_rows": int(sum(len(v) for v in raw_rows.values())),
                              "track_sha256": sha256(path), "eval_gt_read_by_tracker": False})
        for name, raw_cfg in CANDIDATES.items():
            cfg = dict(raw_cfg); cfg["name"] = name
            diagnostics[name] = {}
            for p in eval_prepared:
                tracks, diag = online_track_sweep(p, cfg, student, disable_repairs=False)
                diagnostics[name][p.name] = diag
                path = out / "mot17" / "tracks" / name / (p.name + ".txt"); write_tracks(path, tracks)
                rows_meta.append({"sequence": p.name, "method": name, "frames": int(args.max_frames),
                                  "det_rows": int(sum(len(v) for v in p.det.values())),
                                  "track_rows": int(sum(len(v) for v in tracks.values())),
                                  "track_sha256": sha256(path), "diagnostics": diag,
                                  "eval_gt_read_by_tracker": False})
        payload["eval_diagnostics"] = diagnostics
        payload["methods"] = ["raw", "A", "B", "C", "D"]
        payload["rows"] = rows_meta
        payload["mot17"] = {"eval_sequences": list(EVAL_SEQUENCES), "rows": rows_meta}
        payload["eval_gt_read"] = False; payload["status"] = "complete"
        payload["elapsed_seconds"] = time.perf_counter() - t0
        write_json(out / "results.json", payload); write_json(out / "audit.json", {
            "schema": "mot-jepa-adapter-v4-gate-sweep-audit-v1",
            "protocol": "pre-registered A/B/C/D repair gate; lifecycle fixed; tracker read_gt=False",
            "candidates": CANDIDATES, "train_suffix_repair_audit": train_audit,
            "no_repair_identity_invariant": payload["no_repair_identity_invariant"],
            "eval_gt_read_by_tracker": False, "rows": rows_meta,
            "student_checkpoint_sha256": payload["student_checkpoint_sha256"],
            "source_stress_gate_status": payload["source_stress_gate_status"],
        })
        shutil.copy2(Path(__file__).resolve(), out / Path(__file__).name)
        shutil.copy2(HERE / "mot_jepa_adapter_v4_lifecycle.py", out / "mot_jepa_adapter_v4_lifecycle.py")
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        payload.update({"status": "failure", "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - t0})
        write_json(out / "failure.json", payload); write_json(out / "results.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
