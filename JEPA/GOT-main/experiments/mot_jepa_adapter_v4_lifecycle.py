#!/usr/bin/env python3
"""Lifecycle-only v4 follow-up for the stress-qualified MOT-JEPA adapter.

The v3 integration accidentally treated every stale state within ``max_age``
as an occupied ID.  A detector/native ID that legitimately restarted after a
gap was therefore renamed to a fresh ID even when JEPA made no repair.  This
isolated follow-up changes only that lifecycle rule: a native ID is considered
occupied when it is emitted by a state matched in the current frame (or by a
new row already emitted in that frame).  All association thresholds remain
fixed at the v3 values.  A no-repair invariant is checked before any repaired
outputs are written: with repairs disabled, emitted IDs must exactly equal
the native OC-SORT IDs.

The source v3 deterministic-stress gate is required to have passed.  The
05/11 streams are then loaded with ``read_gt=False`` and evaluated by the
separate vendored TrackEval audit.  This script never reads evaluation GT.
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

import mot_jepa_adapter_v3_stress_protocol as stress  # noqa: E402
import mot_jepa_adapter_v3_direct as v3  # noqa: E402


SOURCE_DEFAULT = Path("/data2/usr_for_deadline/mot_jepa_adapter_v3_stress_120")
MAX_FRAMES = 120
FIT_END = 96
FIT_SEQUENCES = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN")
EVAL_SEQUENCES = ("MOT17-05-FRCNN", "MOT17-11-FRCNN")


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


def margin(values: np.ndarray, j: int) -> float:
    if len(values) <= 1:
        return float("inf")
    return float(values[j] - np.max(np.delete(values, j)))


def online_track_lifecycle(
    p: v3.v2lib.PreparedSequence,
    models: Dict[str, Optional[v3.PrototypeNet]],
    thresholds: Dict[str, float],
    max_age: int = 30,
    disable_repairs: bool = False,
) -> Tuple[Dict[str, Dict[int, List[Tuple[np.ndarray, int, float]]]], Dict[str, dict]]:
    """Run causal association with lifecycle-safe native ID reuse.

    ``disable_repairs`` is used solely by the preflight identity invariant.
    It still executes the same state bookkeeping, but never opens a JEPA/IoU
    repair.  The corrected occupied set is deliberately *current-frame only*;
    stale states remain available for a later gap repair but do not force a
    fresh ID on an otherwise native row.
    """
    outputs: Dict[str, Dict[int, List[Tuple[np.ndarray, int, float]]]] = {}
    diags: Dict[str, dict] = {}
    nframes = v3.v2lib.max_frames_for(p)
    for mode, model in models.items():
        if mode == "raw":
            outputs[mode] = {fr: [(r["box"], int(r["native_id"]), float(r["score"]))
                             for r in rs] for fr, rs in p.raw.items()}
            diags[mode] = {"mode": mode, "eval_gt_read_by_tracker": False,
                           "lag": "t->t+1", "repairs_disabled": bool(disable_repairs)}
            continue
        states: List[State] = []
        next_fresh = 1_000_000
        out: Dict[int, List[Tuple[np.ndarray, int, float]]] = {}
        d = {"mode": mode, "lag": "t->t+1", "eval_gt_read_by_tracker": False,
             "repairs_disabled": bool(disable_repairs), "native_continuations": 0,
             "ambiguous_opened": 0, "gap_candidates": 0, "accepted_repairs": 0,
             "accepted_ambiguous": 0, "gate_rejects": 0, "filter_updates": 0,
             "candidate_pairs": 0}
        for fr in range(1, nframes + 1):
            cur = p.raw.get(fr, [])
            # Keep one live state per canonical ID.  This matters after a
            # native-ID restart: the stale predecessor may still be inside
            # ``max_age`` for gap repair, but it must not coexist with the
            # newly emitted native ID and create duplicate tracker rows.
            by_cid: Dict[int, State] = {}
            for old in states:
                if fr - old.last_frame > max_age:
                    continue
                prev = by_cid.get(int(old.cid))
                if prev is None or int(old.last_frame) > int(prev.last_frame):
                    by_cid[int(old.cid)] = old
            active = list(by_cid.values())
            used_s: set = set(); used_r: set = set()
            matches: List[Tuple[int, int, str]] = []
            cvec = (np.stack([v3.pooled_map(r["feat_map"]) for r in cur]).astype(np.float32)
                    if cur else np.empty((0, v3.FEATURE_DIM), np.float32))
            iou_cache = {i: np.asarray([v3.box_iou(s.box, r["box"]) for r in cur], np.float32)
                         for i, s in enumerate(active)}
            app_cache: Dict[int, np.ndarray] = {}
            for i, s in enumerate(active):
                if s.prototype is None or not len(cur):
                    app_cache[i] = np.zeros(len(cur), np.float32)
                else:
                    app_cache[i] = cvec.dot(v3.norm(s.prototype))

            ambiguous: set = set()
            native_opts: List[Tuple[float, int, int]] = []
            amb_thr = float(thresholds.get("ambiguous_fused_margin", .10))
            for i, s in enumerate(active):
                same = [j for j, r in enumerate(cur)
                        if int(r["native_id"]) == int(s.native_id)
                        and iou_cache[i][j] >= .20]
                if not same:
                    continue
                j = max(same, key=lambda q: float(iou_cache[i][q]))
                fused = .45 * iou_cache[i] + .55 * ((app_cache[i] + 1.) / 2.)
                mar = margin(fused, j)
                gap = fr - s.last_frame
                if (not disable_repairs and mar < amb_thr and s.prototype is not None
                        and len(cur) > 1):
                    if (float(s.visibility) >= float(thresholds.get("visibility_min", .25))
                            and abs(float(s.uncertainty)) <= float(thresholds.get("uncertainty_max", 1.0))):
                        ambiguous.add(i); d["ambiguous_opened"] += 1
                    else:
                        d["gate_rejects"] += 1
                if i not in ambiguous and gap <= 1:
                    native_opts.append((float(iou_cache[i][j]), i, j))
                elif gap > 1:
                    d["gap_candidates"] += 1
            for _q, i, j in sorted(native_opts, reverse=True):
                if i in used_s or j in used_r:
                    continue
                matches.append((i, j, "native")); used_s.add(i); used_r.add(j)
                d["native_continuations"] += 1

            rem_s: List[int] = []
            if not disable_repairs:
                for i, s in enumerate(active):
                    if i in used_s or s.prototype is None:
                        continue
                    gap = fr - s.last_frame
                    if i in ambiguous or 2 <= gap <= 8:
                        rem_s.append(i)
            rem_r = [j for j in range(len(cur)) if j not in used_r]
            d["candidate_pairs"] += len(rem_s) * len(rem_r)
            if rem_s and rem_r:
                fused = np.full((len(rem_s), len(rem_r)), -np.inf, np.float32)
                for a, i in enumerate(rem_s):
                    for b, j in enumerate(rem_r):
                        fused[a, b] = (.45 * float(iou_cache[i][j]) +
                                       .55 * float((app_cache[i][j] + 1.) / 2.))
                rr, cc = linear_sum_assignment(-fused)
                for a, b in zip(rr, cc):
                    a, b = int(a), int(b); i, j = rem_s[a], rem_r[b]
                    q = float(iou_cache[i][j]); ap = float(app_cache[i][j])
                    row_mar = margin(fused[a], b); col_mar = margin(fused[:, b], a)
                    is_amb = i in ambiguous
                    if is_amb:
                        ok = (q >= float(thresholds.get("ambiguous_iou_min", .05))
                              and ap >= float(thresholds.get("ambiguous_app_min", .20))
                              and row_mar >= 0. and col_mar >= 0.)
                    else:
                        ok = ((q >= .02 or ap >= .25) and row_mar >= 0. and col_mar >= 0.)
                    if ok:
                        matches.append((i, j, "ambiguous" if is_amb else "gap"))
                        used_s.add(i); used_r.add(j); d["accepted_repairs"] += 1
                        d["accepted_ambiguous"] += int(is_amb)
                    else:
                        d["gate_rejects"] += 1

            updated: List[Tuple[State, np.ndarray]] = []
            for i, j, _kind in matches:
                s, r = active[i], cur[j]
                old = s.box.copy(); box = np.asarray(r["box"], np.float32)
                vec = v3.pooled_map(r["feat_map"])
                s.history = (s.history + [(fr, np.asarray(r["feat_map"], np.float32),
                                            box.copy(), float(r["score"]))])[-v3.HISTORY_K:]
                s.box = box.copy(); s.last_frame = fr
                s.native_id = int(r["native_id"]); s.score = float(r["score"])
                x = v3.feature_input(s.history[:-1], vec, box, p.width, p.height,
                                     fr, s.score, 0.)
                updated.append((s, x)); r["cid_v4"] = s.cid

            # Lifecycle fix: only IDs emitted in this frame reserve a value.
            # Stale max_age states are intentionally excluded, so a native ID
            # can be reused after an un-repaired gap exactly as in raw OC-SORT.
            occupied = {int(active[i].cid) for i in used_s}
            for j, r in enumerate(cur):
                if j in used_r:
                    continue
                native = int(r["native_id"])
                if native > 0 and native not in occupied:
                    cid = native
                    # A stale state with the same native/canonical ID is a
                    # genuine lifecycle restart when no repair claimed this
                    # row.  Retire it before appending the new state so the
                    # next frame cannot emit the ID twice.
                    states = [old for old in states if int(old.cid) != int(cid) or old.last_frame >= fr]
                else:
                    cid = next_fresh
                    all_cids = {int(old.cid) for old in states}
                    while cid in occupied or cid in all_cids:
                        next_fresh += 1; cid = next_fresh
                next_fresh = max(next_fresh, cid + 1)
                occupied.add(cid)
                vec = v3.pooled_map(r["feat_map"])
                s = State(cid, native, fr, np.asarray(r["box"], np.float32).copy(),
                          float(r["score"]), [(fr, np.asarray(r["feat_map"], np.float32),
                                                np.asarray(r["box"], np.float32).copy(),
                                                float(r["score"]))])
                states.append(s); r["cid_v4"] = cid
                x = v3.feature_input([], vec, r["box"], p.width, p.height,
                                     fr, float(r["score"]), 0.)
                updated.append((s, x))

            for s, x in updated:
                if mode == "history":
                    s.prototype = v3.pooled_map(s.history[-1][1]); s.visibility = 1.; s.uncertainty = 0.
                else:
                    pred, vlog, ulog = model.predict(np.asarray(x, np.float32)[None])
                    s.prototype = pred[0]
                    s.visibility = float(1. / (1. + np.exp(-vlog[0])))
                    s.uncertainty = float(ulog[0])
                d["filter_updates"] += 1
            out[fr] = [(r["box"], int(r.get("cid_v4", r["native_id"])),
                        float(r["score"])) for r in cur]
        outputs[mode] = out; diags[mode] = d

    raw_ids = [int(r["native_id"]) for fr in range(1, nframes + 1) for r in p.raw.get(fr, [])]
    for mode in outputs:
        if mode != "raw":
            ids = [int(t[1]) for fr in range(1, nframes + 1)
                   for t in outputs[mode].get(fr, [])]
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default=str(SOURCE_DEFAULT))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=str(stress.CHECKPOINT_DEFAULT))
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    source = Path(args.source_dir).resolve(); out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir).resolve() if args.cache_dir else source.parent / "mot_jepa_adapter_v3_direct_120" / "cache"
    t0 = time.perf_counter()
    payload: Dict[str, object] = {
        "schema": "mot-jepa-adapter-v4-lifecycle-v1", "status": "started",
        "dataset": "MOT17", "fit_sequences": list(FIT_SEQUENCES),
        "eval_sequences_reserved": list(EVAL_SEQUENCES), "max_frames": int(args.max_frames),
        "source_stress_dir": str(source), "cache_dir": str(cache),
        "association_lag": "t->t+1", "candidate_in_predictor": False,
        "eval_gt_read_by_tracker": False,
        "lifecycle_fix": "occupied IDs are current-frame emitted cids only; stale max_age states do not force fresh renumbering",
        "thresholds": {"ambiguous_fused_margin": .10, "ambiguous_iou_min": .05,
                       "ambiguous_app_min": .20, "visibility_min": .25,
                       "uncertainty_max": 1.0, "source": "fixed before eval"},
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
    }
    try:
        src_gate_payload = json.loads((source / "quality_gate.json").read_text())
        stress_gate = dict(src_gate_payload.get("stress_gate", {}))
        native_gate = dict(src_gate_payload.get("original_native_hard_gate", {}))
        payload["source_stress_gate"] = stress_gate
        payload["source_native_gate"] = native_gate
        if stress_gate.get("status") != "pass":
            raise RuntimeError("source stress gate is not pass; refusing integration eval")
        student_path = source.parent / "mot_jepa_adapter_v3_direct_120" / "checkpoints" / "trained_direct_prototype.pt"
        no_teacher_path = source.parent / "mot_jepa_adapter_v3_direct_120" / "checkpoints" / "no_teacher_direct_prototype.pt"
        if not student_path.exists() or not no_teacher_path.exists():
            raise FileNotFoundError("source v3 checkpoints")
        student = stress.load_prototype(student_path)
        no_teacher = stress.load_prototype(no_teacher_path)
        random_model = v3.PrototypeNet(student.input_dim, 37 + 404)
        payload["checkpoint_sha256"] = {"student": sha256(student_path), "no_teacher": sha256(no_teacher_path)}

        extractor = v3.official_api.OfficialLatent(Path(args.checkpoint), args.device, 8)
        eval_prepared = [v3.v2lib.prepare_sequence(seq, int(args.max_frames), extractor,
                                                     cache, read_gt=False) for seq in EVAL_SEQUENCES]
        payload["eval_gt_read"] = False
        payload["eval_manifest"] = {p.name: {"frames": int(v3.v2lib.max_frames_for(p)),
                                               "det_rows": int(sum(len(v) for v in p.det.values())),
                                               "metadata": p.metadata} for p in eval_prepared}

        models = {"raw": None, "history": None, "trained_adapter": student,
                  "no_teacher": no_teacher, "random_frozen": random_model}
        # Preflight on both sequences: with no repairs all emitted IDs must
        # exactly match native OC-SORT.  This catches lifecycle regressions
        # before repaired outputs can influence the audit directory.
        invariant = {}
        for p in eval_prepared:
            sanity_models = {"raw": None, "history": None}
            sanity_out, sanity_diag = online_track_lifecycle(
                p, sanity_models, payload["thresholds"], disable_repairs=True)
            invariant[p.name] = {m: sanity_diag[m].get("id_differences_vs_raw")
                                 for m in ("history",)}
            if any(v != 0 for v in invariant[p.name].values()):
                raise AssertionError("no-repair identity invariant failed: {}".format(invariant))
        payload["no_repair_identity_invariant"] = {"status": "pass", "per_sequence": invariant}

        rows_meta = []; tracks_block = {}
        for p in eval_prepared:
            outputs, diags = online_track_lifecycle(p, models, payload["thresholds"], disable_repairs=False)
            for mode, tracks in outputs.items():
                path = out / "mot17" / "tracks" / mode / (p.name + ".txt")
                write_tracks(path, tracks)
                item = {"sequence": p.name, "method": mode, "frames": int(args.max_frames),
                        "det_rows": int(sum(len(v) for v in p.det.values())),
                        "track_rows": int(sum(len(v) for v in tracks.values())),
                        "track_sha256": sha256(path), "diagnostics": diags.get(mode, {}),
                        "eval_gt_read_by_tracker": False}
                rows_meta.append(item); tracks_block.setdefault(mode, {})[p.name] = str(path)
        payload["methods"] = list(models.keys())
        payload["rows"] = rows_meta
        payload["mot17"] = {"eval_sequences": list(EVAL_SEQUENCES), "rows": rows_meta,
                            "tracks": tracks_block}
        payload["eval"] = {"eval_sequences": list(EVAL_SEQUENCES), "rows": rows_meta,
                            "tracks": tracks_block, "eval_gt_read": False,
                            "thresholds": payload["thresholds"]}
        payload["eval_started_after_stress_gate"] = True
        payload["status"] = "complete"; payload["elapsed_seconds"] = time.perf_counter() - t0
        write_json(out / "results.json", payload)
        write_json(out / "train_results.json", payload)
        write_json(out / "audit.json", {
            "schema": "mot-jepa-adapter-v4-lifecycle-audit-v1",
            "source_stress_gate": stress_gate, "source_native_gate": native_gate,
            "no_repair_identity_invariant": payload["no_repair_identity_invariant"],
            "eval_started_after_stress_gate": True, "eval_gt_read_by_tracker": False,
            "thresholds": payload["thresholds"], "lifecycle_fix": payload["lifecycle_fix"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "rows": rows_meta,
        })
        for name in ("mot_jepa_adapter_v4_lifecycle.py", "mot_jepa_adapter_v3_direct.py",
                     "mot_jepa_adapter_v3_stress_protocol.py", "audit_trackeval_real.py"):
            src = HERE / name
            if src.exists(): shutil.copy2(src, out / name)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        payload.update({"status": "failure", "error_type": type(exc).__name__,
                        "error": str(exc), "traceback": traceback.format_exc(),
                        "elapsed_seconds": time.perf_counter() - t0})
        write_json(out / "failure.json", payload); write_json(out / "train_results.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
