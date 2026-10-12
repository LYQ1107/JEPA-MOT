#!/usr/bin/env python3
"""Read-only causal diagnosis for the bounded official-filter v2 run.

Replays the v2 association order against cached crop maps and records native
continuations, missing-track gaps, classifier scores, rank changes, and gate
predicates.  It never opens MOT GT and never writes tracker files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mot_ocsort_jepa as base  # noqa: E402
import mot_ocsort_jepa_official as v1  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class State:
    cid: int
    native_id: int
    last_frame: int
    box: np.ndarray
    maps: List[np.ndarray] = field(default_factory=list)
    omega: Optional[np.ndarray] = None


def second_margin(values: np.ndarray, chosen: int) -> float:
    if values.size <= 1:
        return float("inf")
    return float(values[chosen] - np.max(np.delete(values, chosen)))


def order(values: np.ndarray) -> List[int]:
    return list(np.argsort(-np.asarray(values), kind="mergesort").astype(int))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequence", default="MOT17-05-FRCNN")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mot-ocsort-official-filter-v2-diagnosis-v1",
        "status": "started", "dataset": "MOT17", "sequence": args.sequence,
        "max_frames": args.max_frames, "command": " ".join(sys.argv),
        "cwd": os.getcwd(), "eval_gt_read_by_tracker": False,
        "candidate_in_filter_predictor": False,
        "source_script_sha256": sha256(HERE / "mot_ocsort_jepa_official_filter_v2.py"),
        "checkpoint_sha256": sha256(Path(args.checkpoint).resolve()),
        "protocol": "replay v2 causal gate; no tracker output mutation",
        "thresholds": {"gap_min": 2, "gap_max": 8, "iou_min": .05,
                       "classifier_score_min": 0., "margin_min": .03},
    }
    frame_events: List[dict] = []; candidate_events: List[dict] = []
    score_values: List[float] = []; margin_values: List[float] = []
    t0 = time.perf_counter()
    try:
        official = v1.OfficialLatent(Path(args.checkpoint), args.device, args.batch)
        payload["official_load"] = official.load_meta
        seq_dir = base.MOT_ROOT / "train" / args.sequence
        det = base.parse_det(seq_dir / "det" / "det.txt", args.max_frames)
        maps, cache_meta = official.extract_sequence(args.sequence, det, args.max_frames,
                                                     Path(args.cache_dir).resolve())
        payload["feature_cache"] = cache_meta
        params = {"track_thresh": .2, "iou_threshold": .3, "max_age": 30,
                  "min_hits": 1, "delta_t": 3, "inertia": .2}
        raw, _ = base.run_ocsort_rows(args.sequence, args.max_frames, det, params)
        v1.attach_maps(raw, maps, det)
        states: List[State] = []; next_fresh = 1_000_000
        counts = {"raw_rows": 0, "active_states": 0, "native_continuations": 0,
                  "missing_state_events": 0, "eligible_states": 0,
                  "candidate_rows": 0, "classifier_state_calls": 0,
                  "candidate_pairs": 0, "iou_pass": 0, "score_pass": 0,
                  "row_margin_pass": 0, "col_margin_pass": 0,
                  "all_gate_pass": 0, "accepted_repairs": 0,
                  "app_rank_diff": 0, "fused_choice_diff_iou": 0,
                  "output_id_differences": 0}
        reconstructed: Dict[int, List[tuple]] = {}
        for fr in range(1, args.max_frames + 1):
            cur = raw.get(fr, []); counts["raw_rows"] += len(cur)
            active = [s for s in states if fr - s.last_frame <= 30]
            counts["active_states"] += len(active)
            used_s, used_r = set(), set(); matches = []
            native_ids = [int(r["native_id"]) for r in cur]
            for j, row in enumerate(cur):
                cand = [i for i, s in enumerate(active)
                        if i not in used_s and int(s.native_id) == int(row["native_id"])]
                if not cand:
                    continue
                i = max(cand, key=lambda q: base.iou(active[q].box, row["box"]))
                matches.append((i, j, "native")); used_s.add(i); used_r.add(j)
                counts["native_continuations"] += 1
            missing = [{"cid": int(s.cid), "native_id": int(s.native_id),
                        "gap": int(fr - s.last_frame)}
                       for i, s in enumerate(active) if i not in used_s]
            counts["missing_state_events"] += len(missing)
            rem_s = [i for i, s in enumerate(active)
                     if i not in used_s and 2 <= fr - s.last_frame <= 8 and s.omega is not None]
            rem_r = [j for j in range(len(cur)) if j not in used_r]
            counts["eligible_states"] += len(rem_s); counts["candidate_rows"] += len(rem_r)
            event = {"frame": fr, "raw_rows": len(cur), "native_ids": native_ids,
                     "active": [{"cid": int(s.cid), "native_id": int(s.native_id),
                                 "last_frame": int(s.last_frame), "gap": int(fr-s.last_frame)} for s in active],
                     "native_matches": [{"state_index": int(i), "row_index": int(j)} for i,j,_ in matches],
                     "missing_states": missing, "eligible_state_indices": rem_s,
                     "candidate_row_indices": rem_r}
            if rem_s and rem_r:
                counts["classifier_state_calls"] += len(rem_s)
                c_maps = [cur[j]["feat_map"].astype(np.float32) for j in rem_r]
                app = np.full((len(rem_s), len(rem_r)), -np.inf, np.float32)
                ious = np.zeros_like(app)
                for a, si in enumerate(rem_s):
                    app[a] = official.classifier_scores(active[si].omega, c_maps)
                    for b, rj in enumerate(rem_r):
                        ious[a,b] = base.iou(active[si].box, cur[rj]["box"])
                    score_values.extend([float(x) for x in app[a] if np.isfinite(x)])
                fused = .70 * ious + .30 * ((np.tanh(app) + 1.) / 2.)
                rr, cc = linear_sum_assignment(-fused)
                counts["candidate_pairs"] += int(app.size)
                event["candidate_matrix"] = []
                # Preserve the complete state×candidate score matrix, not
                # only the Hungarian-selected edge.  This lets the audit
                # distinguish a genuinely changed ranking from a gate that
                # simply had no eligible candidate.
                event["score_matrix"] = {
                    "state_indices": [int(x) for x in rem_s],
                    "candidate_indices": [int(x) for x in rem_r],
                    "classifier": [[float(x) for x in row] for row in app],
                    "iou": [[float(x) for x in row] for row in ious],
                    "fused": [[float(x) for x in row] for row in fused],
                }
                for a, b in zip(rr, cc):
                    a, b = int(a), int(b)
                    row_margin = second_margin(app[a], b); col_margin = second_margin(app[:, b], a)
                    iou_ok = float(ious[a,b]) >= .05; score_ok = float(app[a,b]) >= 0.
                    row_ok = row_margin >= .03; col_ok = col_margin >= .03
                    all_ok = iou_ok and score_ok and row_ok and col_ok
                    counts["iou_pass"] += int(iou_ok); counts["score_pass"] += int(score_ok)
                    counts["row_margin_pass"] += int(row_ok); counts["col_margin_pass"] += int(col_ok)
                    counts["all_gate_pass"] += int(all_ok); counts["accepted_repairs"] += int(all_ok)
                    margin_values.extend([row_margin, col_margin])
                    app_order = order(app[a]); iou_order = order(ious[a]); fused_order = order(fused[a])
                    counts["app_rank_diff"] += int(app_order != iou_order)
                    counts["fused_choice_diff_iou"] += int(fused_order[0] != iou_order[0])
                    ce = {"state_index": int(rem_s[a]), "state_cid": int(active[rem_s[a]].cid),
                          "state_native_id": int(active[rem_s[a]].native_id),
                          "row_index": int(rem_r[b]), "candidate_native_id": int(cur[rem_r[b]]["native_id"]),
                          "gap": int(fr-active[rem_s[a]].last_frame), "iou": float(ious[a,b]),
                          "classifier_score": float(app[a,b]), "fused_score": float(fused[a,b]),
                          "row_margin": float(row_margin), "col_margin": float(col_margin),
                          "iou_ok": iou_ok, "score_ok": score_ok, "row_margin_ok": row_ok,
                          "col_margin_ok": col_ok, "accepted": all_ok,
                          "app_order": app_order, "iou_order": iou_order, "fused_order": fused_order,
                          "all_app_scores": [float(x) for x in app[a]],
                          "all_ious": [float(x) for x in ious[a]],
                          "all_fused": [float(x) for x in fused[a]]}
                    event["candidate_matrix"].append(ce); candidate_events.append(ce)
                for a, b in zip(rr, cc):
                    a, b = int(a), int(b)
                    row_margin = second_margin(app[a], b); col_margin = second_margin(app[:,b], a)
                    if (float(ious[a,b]) >= .05 and float(app[a,b]) >= 0.
                            and row_margin >= .03 and col_margin >= .03):
                        si, rj = rem_s[a], rem_r[b]
                        matches.append((si, rj, "jepa_repair")); used_s.add(si); used_r.add(rj)
            frame_events.append(event)
            updated, contexts, currents = [], [], []
            for si, rj, _ in matches:
                s, row = active[si], cur[rj]; current = row["feat_map"].astype(np.float32)
                context = np.mean(np.stack(s.maps), axis=0).astype(np.float32) if s.maps else current
                s.native_id = int(row["native_id"]); s.last_frame = fr; s.box = row["box"].copy()
                s.maps = (s.maps + [current])[-4:]; row["cid_v2"] = s.cid
                updated.append(s); contexts.append(context); currents.append(current)
            active_cids = {int(s.cid) for s in active if fr-s.last_frame <= 30}
            for j, row in enumerate(cur):
                if j in used_r:
                    continue
                cid = int(row["native_id"])
                if cid <= 0 or cid in active_cids:
                    while next_fresh in active_cids:
                        next_fresh += 1
                    cid = next_fresh; next_fresh += 1
                current = row["feat_map"].astype(np.float32)
                s = State(cid, int(row["native_id"]), fr, row["box"].copy(), [current])
                states.append(s); active_cids.add(cid); row["cid_v2"] = cid
                updated.append(s); contexts.append(current); currents.append(current)
            if updated:
                omegas = official.predict_filter_batch(contexts, currents)
                for s, omega in zip(updated, omegas):
                    s.omega = omega
            reconstructed[fr] = [(row["box"], int(row.get("cid_v2", row["native_id"])), float(row["score"])) for row in cur]
        raw_ids = [int(r["native_id"]) for fr in range(1,args.max_frames+1) for r in raw.get(fr,[])]
        rec_ids = [int(t[1]) for fr in range(1,args.max_frames+1) for t in reconstructed.get(fr,[])]
        counts["output_id_differences"] = int(sum(a != b for a,b in zip(raw_ids, rec_ids)))
        payload["counts"] = counts
        payload["score_summary"] = {"n": len(score_values), "finite": bool(np.isfinite(score_values).all()) if score_values else True,
                                     "min": float(np.min(score_values)) if score_values else None,
                                     "max": float(np.max(score_values)) if score_values else None,
                                     "mean": float(np.mean(score_values)) if score_values else None,
                                     "std": float(np.std(score_values)) if score_values else None}
        payload["margin_summary"] = {"n": len(margin_values), "min": float(np.min(margin_values)) if margin_values else None,
                                      "max": float(np.max(margin_values)) if margin_values else None,
                                      "mean": float(np.mean(margin_values)) if margin_values else None}
        payload["candidate_events"] = candidate_events; payload["frame_event_count"] = len(frame_events)
        payload["status"] = "complete"
        payload["recommendation"] = "B: train a MOT-specific JEPA adapter before formal expansion" if counts["accepted_repairs"] == 0 else "A: bounded repair is active; run preregistered formal evaluation"
        payload["recommendation_reason"] = ("Frozen official filter produced no gate-passing repair in the 20-frame prefix; a formal run would test mostly the unchanged OC-SORT baseline." if counts["accepted_repairs"] == 0 else "At least one causal repair passed all pre-registered predicates.")
    except Exception as exc:
        payload["status"] = "failure"; payload["error_type"] = type(exc).__name__; payload["error"] = str(exc); payload["traceback"] = traceback.format_exc()
    payload["elapsed_seconds"] = time.perf_counter() - t0
    (out / "diagnosis.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
    (out / "frame_events.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, default=float) + "\n" for x in frame_events))
    if candidate_events:
        keys = sorted({k for x in candidate_events for k in x})
        with (out / "candidate_scores.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(candidate_events)
    summary = "# v2 frozen-filter diagnosis\n\n" + json.dumps({k: payload.get(k) for k in ["status", "counts", "score_summary", "margin_summary", "recommendation", "recommendation_reason"]}, indent=2, ensure_ascii=False, default=float) + "\n"
    (out / "summary.md").write_text(summary)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
