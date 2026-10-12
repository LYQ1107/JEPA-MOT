#!/usr/bin/env python3
"""Read-only protocol audit for the real DanceTrack temporal-JEPA harness.

The tracker itself is deliberately not imported or changed.  This utility
records the detector/cache and protocol declarations from ``results.json``
and checks a few source/cache invariants that are easy to accidentally break
when extending a run: D-CTRL files must be per-frame six-column detections
without an identity field, evaluation must not use oracle boxes, and the
strict teacher/no-teacher/random controls must be present.  It writes a
machine-readable report suitable for a run's provenance bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


REQUIRED_METHODS = {"raw", "jepa_temporal", "no_teacher", "random_frozen"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_cache(cache_root: Path, sequences: list[str], max_frames: int) -> dict:
    """Check representative D-CTRL files and prove they have no ID column."""
    checks: dict[str, object] = {"root": str(cache_root), "sequences": {}, "pass": True}
    for seq in sequences:
        seq_root = cache_root / seq
        files = sorted(seq_root.glob("*.txt")) if seq_root.is_dir() else []
        files = [p for p in files if not max_frames or int(p.stem) <= max_frames]
        rows = 0
        bad_width = 0
        bad_numeric = 0
        widths: set[int] = set()
        for path in files:
            for line in path.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                vals = [x.strip() for x in line.split(",")]
                widths.add(len(vals)); rows += 1
                if len(vals) != 6:
                    bad_width += 1
                    continue
                try:
                    [float(v) for v in vals]
                except ValueError:
                    bad_numeric += 1
        ok = bool(files) and rows > 0 and widths == {6} and bad_width == 0 and bad_numeric == 0
        checks["sequences"][seq] = {
            "files_checked": len(files), "rows_checked": rows,
            "column_widths": sorted(widths), "bad_width_rows": bad_width,
            "bad_numeric_rows": bad_numeric, "six_column_frame_xywh_score": ok,
        }
        checks["pass"] = bool(checks["pass"] and ok)
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--source-script", default="experiments/mot_dance_jepa_real.py")
    ap.add_argument("--cache-root", default="/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/detections_ctrl")
    ap.add_argument("--sequences", default="dancetrack0001,dancetrack0002,dancetrack0006,dancetrack0004,dancetrack0007,dancetrack0014")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    result_dir = Path(args.result_dir).resolve()
    result_path = result_dir / "results.json"
    source = Path(args.source_script).resolve()
    payload = json.loads(result_path.read_text()) if result_path.exists() else {}
    dance = payload.get("dance", {}) if isinstance(payload, dict) else {}
    methods = set(str(x) for x in payload.get("methods", [])) if isinstance(payload, dict) else set()
    if not methods and isinstance(dance, dict):
        methods = {str(r.get("method")) for r in dance.get("rows", []) if isinstance(r, dict)}
    protocol = dance.get("training_protocol", {}) if isinstance(dance, dict) else {}
    detector = dance.get("detector", {}) if isinstance(dance, dict) else {}
    max_frames = int(args.max_frames or payload.get("max_frames", 0) or 0)
    seqs = [x for x in args.sequences.split(",") if x]
    checks = {
        "result_exists": result_path.exists(),
        "source_exists": source.exists(),
        "methods_required_present": REQUIRED_METHODS.issubset(methods),
        "methods": sorted(methods),
        "detector_type_dctrl": "D-CTRL" in str(detector.get("type", "")),
        "oracle_boxes_false": detector.get("oracle_boxes") is False,
        "protocol_teacher_gt_only": protocol.get("teacher_gt_only") is True,
        "protocol_student_context_gt_false": protocol.get("student_context_gt") is False,
        "protocol_gt_geometry_false": protocol.get("gt_geometry_in_student") is False and protocol.get("gt_geometry_in_student_x") is False,
        "protocol_gt_visibility_false": protocol.get("gt_visibility_in_student") is False,
        "protocol_eval_uses_gt_false": protocol.get("eval_uses_gt") is False,
        "strict_temporal_controls": set(protocol.get("strict_temporal_methods", [])) >= {"jepa_temporal", "no_teacher", "random_frozen"},
    }
    cache = check_cache(Path(args.cache_root), seqs, max_frames)
    checks["cache_six_column_no_id"] = cache["pass"]
    checks["all_pass"] = bool(all(v for k, v in checks.items() if k not in {"methods"}) and cache["pass"])
    report = {
        "schema": "dance-temporal-protocol-audit-v1",
        "result_dir": str(result_dir), "result_sha256": sha256(result_path) if result_path.exists() else None,
        "source_script": str(source), "source_script_sha256": sha256(source) if source.exists() else None,
        "cache": cache, "checks": checks,
        "payload_command": payload.get("command"), "seed": payload.get("seed"), "max_frames": max_frames,
        "python": sys.version, "platform": platform.platform(),
    }
    out = result_dir / "protocol_audit.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_pass": checks["all_pass"], "out": str(out), "checks": checks}, indent=2, sort_keys=True))
    if not checks["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
