#!/usr/bin/env python3
"""Semantic-correct GOT-JEPA filter adapter for a bounded MOT smoke.

This adapter keeps the public detections, OC-SORT boxes, emissions and native
identity continuity unchanged by default.  It only attempts to join a newly
appearing OC-SORT identity to a recently missing track when the frozen official
GOT-JEPA filter is confident.

The causal order is strict.  After frame-t assignment, accepted history and the
accepted current crop are passed to the serialized ToMP ``filter_predictor``.
Its [B,256,1,1] classification filter is transformed by the official JEPA
predictor and stored.  At frame t+1 candidate maps are passed only to the frozen
classifier; they never enter the filter predictor that produced the stored
filter.  Evaluation GT is not opened by this program.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mot_ocsort_jepa as base  # noqa: E402
import mot_ocsort_jepa_official as v1  # noqa: E402


@dataclass
class FilterState:
    cid: int
    native_id: int
    last_frame: int
    box: np.ndarray
    maps: List[np.ndarray] = field(default_factory=list)
    omega: Optional[np.ndarray] = None


def _second_margin(values: np.ndarray, chosen: int) -> float:
    if values.size <= 1:
        return float("inf")
    keep = np.delete(values, chosen)
    return float(values[chosen] - np.max(keep))


def adapt_filter_v2(
    raw: Dict[int, List[dict]],
    official: v1.OfficialLatent,
    max_frames: int,
    max_age: int = 30,
    repair_gap_min: int = 2,
    repair_gap_max: int = 8,
    repair_iou_min: float = 0.05,
    repair_margin_min: float = 0.03,
    repair_score_min: float = 0.0,
) -> Tuple[Dict[int, List[Tuple[np.ndarray, int, float]]], Dict[str, object]]:
    """Baseline-preserving, occlusion-only semantic filter adapter."""
    states: List[FilterState] = []
    out: Dict[int, List[Tuple[np.ndarray, int, float]]] = {}
    next_fresh = 1_000_000
    diag = {
        "baseline_native_continuations": 0,
        "repair_candidates": 0,
        "accepted_repairs": 0,
        "filter_updates": 0,
        "classifier_state_calls": 0,
        "association_lag": "t->t+1",
        "candidate_in_filter_predictor": False,
        "history_k": 4,
        "repair_gate": {
            "gap_min": repair_gap_min,
            "gap_max": repair_gap_max,
            "iou_min": repair_iou_min,
            "two_sided_classifier_margin_min": repair_margin_min,
            "classifier_score_min": repair_score_min,
        },
    }

    for fr in range(1, max_frames + 1):
        cur = raw.get(fr, [])
        active = [s for s in states if fr - s.last_frame <= max_age]
        used_s, used_r = set(), set()
        matches: List[Tuple[int, int, str]] = []

        # Preserve every unambiguous native OC-SORT continuation.  This makes
        # no-repair output byte-equivalent in identity semantics to the raw
        # baseline instead of re-running all easy assignments.
        for j, row in enumerate(cur):
            candidates = [i for i, s in enumerate(active)
                          if i not in used_s and int(s.native_id) == int(row["native_id"])]
            if not candidates:
                continue
            i = max(candidates, key=lambda q: base.iou(active[q].box, row["box"]))
            matches.append((i, j, "native"))
            used_s.add(i); used_r.add(j)
            diag["baseline_native_continuations"] += 1

        # Only new native IDs versus recently missing states enter the JEPA
        # repair branch.  Candidate maps are scored by the *stored* t-1 filter.
        rem_s = [i for i, s in enumerate(active)
                 if i not in used_s and repair_gap_min <= fr - s.last_frame <= repair_gap_max
                 and s.omega is not None]
        rem_r = [j for j in range(len(cur)) if j not in used_r]
        if rem_s and rem_r:
            app = np.full((len(rem_s), len(rem_r)), -np.inf, np.float32)
            ious = np.zeros_like(app)
            candidate_maps = [cur[j]["feat_map"].astype(np.float32) for j in rem_r]
            for a, si in enumerate(rem_s):
                app[a] = official.classifier_scores(active[si].omega, candidate_maps)
                diag["classifier_state_calls"] += 1
                for b, rj in enumerate(rem_r):
                    ious[a, b] = base.iou(active[si].box, cur[rj]["box"])
            # Fixed score fusion.  tanh bounds the pretrained classifier scale;
            # all release decisions below additionally require two-sided raw
            # classifier margins and an IoU/gap gate.
            fused = 0.70 * ious + 0.30 * ((np.tanh(app) + 1.0) / 2.0)
            rr, cc = linear_sum_assignment(-fused)
            diag["repair_candidates"] += int(len(rr))
            for a, b in zip(rr, cc):
                a, b = int(a), int(b)
                row_margin = _second_margin(app[a], b)
                col_margin = _second_margin(app[:, b], a)
                if (float(ious[a, b]) >= repair_iou_min
                        and float(app[a, b]) >= repair_score_min
                        and row_margin >= repair_margin_min
                        and col_margin >= repair_margin_min):
                    si, rj = rem_s[a], rem_r[b]
                    matches.append((si, rj, "jepa_repair"))
                    used_s.add(si); used_r.add(rj)
                    diag["accepted_repairs"] += 1

        # Apply assignments.  Context is captured before writing the accepted
        # frame-t map, then all filters are predicted in one track batch.
        update_states: List[FilterState] = []
        contexts: List[np.ndarray] = []
        currents: List[np.ndarray] = []
        for si, rj, reason in matches:
            state, row = active[si], cur[rj]
            current = row["feat_map"].astype(np.float32)
            context = (np.mean(np.stack(state.maps), axis=0).astype(np.float32)
                       if state.maps else current)
            state.native_id = int(row["native_id"])
            state.last_frame = fr
            state.box = row["box"].copy()
            state.maps = (state.maps + [current])[-4:]
            row["cid_v2"] = state.cid
            update_states.append(state); contexts.append(context); currents.append(current)

        # Unmatched rows retain their raw native ID whenever possible.  This
        # is the baseline-preserving fallback and uses no learned score.
        active_cids = {int(s.cid) for s in active if fr - s.last_frame <= max_age}
        for j, row in enumerate(cur):
            if j in used_r:
                continue
            cid = int(row["native_id"])
            if cid <= 0 or cid in active_cids:
                while next_fresh in active_cids:
                    next_fresh += 1
                cid = next_fresh; next_fresh += 1
            current = row["feat_map"].astype(np.float32)
            state = FilterState(cid=cid, native_id=int(row["native_id"]),
                                last_frame=fr, box=row["box"].copy(), maps=[current])
            states.append(state); active_cids.add(cid); row["cid_v2"] = cid
            update_states.append(state); contexts.append(current); currents.append(current)

        if update_states:
            omegas = official.predict_filter_batch(contexts, currents)
            if omegas.shape != (len(update_states), 256):
                raise RuntimeError("unexpected batched omega shape {}".format(omegas.shape))
            for state, omega in zip(update_states, omegas):
                state.omega = omega
            diag["filter_updates"] += len(update_states)

        out[fr] = [(row["box"], int(row.get("cid_v2", row["native_id"])),
                    float(row["score"])) for row in cur]

    diag["states_created"] = len(states)
    return out, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequence", default="MOT17-05-FRCNN")
    ap.add_argument("--max-frames", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="/data2/usr_for_deadline/got_jepa_official_mot20/cache")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve()
    payload = {
        "schema": "mot-ocsort-official-jepa-filter-v2",
        "status": "started",
        "dataset": "MOT17",
        "sequence": args.sequence,
        "max_frames": args.max_frames,
        "methods": ["raw", "official_filter_v2"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "source_sha256": v1.sha256(source),
        "parent_v1_sha256": v1.sha256(HERE / "mot_ocsort_jepa_official.py"),
        "eval_gt_read_by_tracker": False,
        "candidate_in_filter_predictor": False,
        "production_tracker_modified": False,
        "association_lag": "t->t+1",
        "semantic_correction": "official FilterPredictor -> JEPA cls filter -> next-frame frozen classifier",
    }
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
        raw, ocsort_seconds = base.run_ocsort_rows(args.sequence, args.max_frames, det, params)
        v1.attach_maps(raw, maps, det)
        raw_rows = {fr: [(r["box"], int(r["native_id"]), float(r["score"])) for r in rows]
                    for fr, rows in raw.items()}
        v2_rows, diag = adapt_filter_v2(raw, official, args.max_frames)
        v1.write_tracks(out / "mot17" / "tracks" / "raw" / (args.sequence + ".txt"), raw_rows)
        v1.write_tracks(out / "mot17" / "tracks" / "official_filter_v2" / (args.sequence + ".txt"), v2_rows)
        payload["ocsort_seconds"] = ocsort_seconds
        payload["det_rows"] = int(sum(len(x) for x in det.values()))
        payload["track_rows"] = {"raw": sum(len(x) for x in raw_rows.values()),
                                 "official_filter_v2": sum(len(x) for x in v2_rows.values())}
        payload["adapter"] = diag
        payload["status"] = "complete"
    except Exception as exc:
        payload["status"] = "failure"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
        (out / "failure.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
    payload["elapsed_seconds"] = time.perf_counter() - t0
    (out / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
    reflection = {
        "schema": "mot-ocsort-official-jepa-filter-v2-reflection",
        "status": payload["status"],
        "semantic_interface": payload["semantic_correction"],
        "causal_lag": "t->t+1",
        "eval_gt_read_by_tracker": False,
        "candidate_in_filter_predictor": False,
        "decision_rule": "Run official TrackEval before any release claim; stop if HOTA/IDF1/MOTA drop or IDSW increases.",
        "error": payload.get("error"),
    }
    (out / "reflection.json").write_text(json.dumps(reflection, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=float))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
