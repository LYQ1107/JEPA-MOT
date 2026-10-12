#!/usr/bin/env python3
"""Post-hoc, read-only diagnosis for the MOT17 temporal-JEPA run.

This script never feeds GT back into tracking.  It only joins the frozen
detector stream, GT, and already-written raw/temporal track files to quantify
history coverage, detector gaps, candidate crowding, and ID continuity by
coverage bucket.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import mot_dance_jepa_real as m


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_track(path: Path):
    out = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        p = line.strip().split(",")
        if len(p) < 7:
            continue
        try:
            fr, tid = int(float(p[0])), int(float(p[1]))
            box = m.xywh_to_xyxy([float(p[2]), float(p[3]), float(p[4]), float(p[5])])
            score = float(p[6])
        except (TypeError, ValueError):
            continue
        out[fr].append((tid, box, score))
    return out


def quantiles(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "min": float(np.min(a)), "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)), "max": float(np.max(a)),
            "mean": float(np.mean(a))}


def best_det_index(box: np.ndarray, dets: List[m.Det]) -> Tuple[int, float]:
    if not dets:
        return -1, 0.0
    vals = [m.box_iou(box, d.box) for d in dets]
    j = int(np.argmax(vals))
    return (j, float(vals[j])) if vals[j] >= .95 else (-1, float(vals[j]))


def dominant_ids(assignments: Dict[int, Dict[int, int]]) -> Dict[int, int]:
    counts = defaultdict(Counter)
    for fr, rows in assignments.items():
        for gid, tid in rows.items():
            if tid is not None:
                counts[gid][tid] += 1
    return {gid: c.most_common(1)[0][0] for gid, c in counts.items() if c}


def sequence_diag(seq: str, max_frames: int, track_root: Path | None) -> Dict[str, object]:
    seq_dir, gt, det, _ = m.load_mot_sequence(seq, max_frames)
    # (frame,gid) -> detector index; matching is post-hoc and used only for
    # coverage/error stratification.
    gdet: Dict[Tuple[int, int], int] = {}
    coverage: Dict[Tuple[int, int], int] = {}
    miss_run: Dict[Tuple[int, int], int] = {}
    score_all: List[float] = []; score_match: List[float] = []
    candidate_counts: List[int] = []
    for fr in sorted(gt):
        ds = det.get(fr, [])
        candidate_counts.append(len(ds)); score_all.extend(float(d.score) for d in ds)
        for g in gt[fr]:
            vals = [m.box_iou(g.box, d.box) for d in ds]
            j = int(np.argmax(vals)) if vals else -1
            iou = float(vals[j]) if j >= 0 else 0.0
            if j >= 0 and iou >= .30:
                gdet[(fr, g.gid)] = j; score_match.append(float(ds[j].score))
            hits = 0
            for off in range(1, 5):
                old = fr - off
                old_ds = det.get(old, [])
                old_gt = {gg.gid: gg for gg in gt.get(old, [])}
                gg = old_gt.get(g.gid)
                ok = False
                if gg is not None and old_ds:
                    ov = [m.box_iou(gg.box, dd.box) for dd in old_ds]
                    ok = bool(ov and max(ov) >= .30)
                hits += int(ok)
            coverage[(fr, g.gid)] = hits
            run = 0
            for back in range(1, min(fr, 40) + 1):
                old = fr - back
                old_gt = {gg.gid: gg for gg in gt.get(old, [])}
                gg = old_gt.get(g.gid)
                old_ds = det.get(old, [])
                ok = False
                if gg is not None and old_ds:
                    ok = max([m.box_iou(gg.box, dd.box) for dd in old_ds] or [0.0]) >= .30
                if ok:
                    break
                run += 1
            miss_run[(fr, g.gid)] = run

    result: Dict[str, object] = {
        "sequence": seq, "frames": len(set(gt) | set(det)), "gt_frames": len(gt), "det_frames": len(det),
        "gt_rows": int(sum(len(v) for v in gt.values())), "detections": int(sum(len(v) for v in det.values())),
        "candidate_count": quantiles(candidate_counts), "detector_score_all": quantiles(score_all),
        "detector_score_gt_matched": quantiles(score_match),
        "history_coverage": {str(k): int(sum(1 for v in coverage.values() if v == k)) for k in range(5)},
        "history_coverage_fraction": {str(k): float(sum(1 for v in coverage.values() if v == k) / max(len(coverage), 1)) for k in range(5)},
        "missing_run_length": quantiles([float(v) for v in miss_run.values()]),
    }
    if track_root is None:
        return result

    method_assign: Dict[str, Dict[int, Dict[int, int]]] = {}
    method_paths = {name: track_root / name / f"{seq}.txt" for name in ("raw", "jepa_temporal")}
    for method, path in method_paths.items():
        tracks = read_track(path)
        by_frame_gid: Dict[int, Dict[int, int]] = defaultdict(dict)
        by_frame_det: Dict[int, Dict[int, int]] = defaultdict(dict)
        for fr, rows in tracks.items():
            ds = det.get(fr, [])
            for tid, box, _ in rows:
                j, _ = best_det_index(box, ds)
                if j < 0:
                    continue
                vals = [m.box_iou(g.box, box) for g in gt.get(fr, [])]
                if not vals:
                    continue
                gi = int(np.argmax(vals)); iou = float(vals[gi])
                if iou < .50:
                    continue
                gid = int(gt[fr][gi].gid)
                by_frame_gid[fr][gid] = int(tid)
                by_frame_det[fr][j] = int(tid)
        method_assign[method] = by_frame_gid
        dom = dominant_ids(by_frame_gid)
        buckets = {str(k): {"gt_frames": 0, "assigned": 0, "dominant_correct": 0, "switch_events": 0, "prev_pairs": 0} for k in range(5)}
        prev_by_gid: Dict[int, int] = {}
        for fr in sorted(gt):
            for g in gt[fr]:
                b = str(coverage.get((fr, g.gid), 0)); z = buckets[b]; z["gt_frames"] += 1
                tid = by_frame_gid.get(fr, {}).get(int(g.gid))
                if tid is not None:
                    z["assigned"] += 1
                    z["dominant_correct"] += int(dom.get(int(g.gid), -1) == tid)
                    if int(g.gid) in prev_by_gid:
                        z["prev_pairs"] += 1
                        z["switch_events"] += int(prev_by_gid[int(g.gid)] != tid)
                    prev_by_gid[int(g.gid)] = tid
        for z in buckets.values():
            z["assignment_rate"] = float(z["assigned"] / max(z["gt_frames"], 1))
            z["dominant_accuracy"] = float(z["dominant_correct"] / max(z["assigned"], 1))
            z["switch_rate"] = float(z["switch_events"] / max(z["prev_pairs"], 1))
        result.setdefault("methods", {})[method] = {"coverage_buckets": buckets, "dominant_ids": len(dom)}

    # Compare raw/JЕPA detector-column ownership only where both files carry
    # the same frozen detection.  This is an attribution diagnostic, not GT
    # input to the tracker.
    changes = {str(k): {"shared_detections": 0, "changed": 0} for k in range(5)}
    raw_tracks = read_track(method_paths["raw"]); jepa_tracks = read_track(method_paths["jepa_temporal"])
    for fr in sorted(set(raw_tracks) | set(jepa_tracks)):
        ds = det.get(fr, [])
        maps = {}
        for method, rows in (("raw", raw_tracks.get(fr, [])), ("jepa", jepa_tracks.get(fr, []))):
            mm = {}
            for tid, box, _ in rows:
                j, _ = best_det_index(box, ds)
                if j >= 0: mm[j] = int(tid)
            maps[method] = mm
        for j in set(maps["raw"]) & set(maps["jepa"]):
            # Find the GT/coverage bucket for this detector column.
            bucket = None
            for g in gt.get(fr, []):
                if gdet.get((fr, g.gid)) == j:
                    bucket = coverage.get((fr, g.gid), 0); break
            if bucket is None:
                continue
            z = changes[str(bucket)]; z["shared_detections"] += 1
            z["changed"] += int(maps["raw"][j] != maps["jepa"][j])
    for z in changes.values():
        z["change_rate"] = float(z["changed"] / max(z["shared_detections"], 1))
    result["assignment_changes_by_coverage"] = changes
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/mot_temporal_history_diagnostics_v1.json")
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--track-root", default="experiments/results/real_mot_temporal_v3/mot17/tracks")
    args = ap.parse_args()
    seqs = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN", "MOT17-05-FRCNN", "MOT17-11-FRCNN"]
    track_root = Path(args.track_root)
    per = [sequence_diag(s, args.max_frames, track_root if s in ("MOT17-05-FRCNN", "MOT17-11-FRCNN") else None) for s in seqs]
    out = {
        "schema": "mot-history-diagnostics-v1", "max_frames": args.max_frames,
        "sequences": per, "train_sequences": seqs[:3], "eval_sequences": seqs[3:],
        "track_results_sha256": sha256(track_root.parent.parent / "results.json") if (track_root.parent.parent / "results.json").exists() else None,
        "script_sha256": sha256(Path(__file__)),
        "posthoc_gt_only": True,
        "note": "No field from this file is fed back into tracker training/inference; coverage uses IoU>=0.30 detector-to-GT joins.",
    }
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
