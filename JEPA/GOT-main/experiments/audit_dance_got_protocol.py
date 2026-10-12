#!/usr/bin/env python3
"""Audit a ``mot_dance_jepa_got.py`` run without changing any artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


METHODS = {"raw", "jepa_temporal", "no_teacher", "random_frozen"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--source-script", default="experiments/mot_dance_jepa_got.py")
    args = ap.parse_args()
    out = Path(args.result_dir).resolve(); result_path = out / "results.json"
    payload = json.loads(result_path.read_text())
    dance = payload.get("dance", {}); protocol = dance.get("training_protocol", {})
    detector = dance.get("detector", {}); ng = dance.get("no_gt_audit", {})
    checks: dict[str, object] = {}
    checks["status_complete"] = payload.get("status") == "complete"
    checks["methods_exact"] = set(payload.get("methods", [])) == METHODS
    checks["oracle_boxes_false"] = detector.get("oracle_boxes") is False
    checks["dctrl_detector"] = "D-CTRL" in str(detector.get("type", ""))
    checks["checkpoint_hash_recorded"] = bool(payload.get("checkpoint_sha256"))
    ckpt = Path(str(payload.get("checkpoint", "")))
    checks["checkpoint_exists"] = ckpt.exists()
    checks["checkpoint_hash_matches"] = bool(ckpt.exists() and sha256(ckpt) == payload.get("checkpoint_sha256"))
    load = payload.get("official_load", {})
    checks["official_load_hash_matches"] = load.get("checkpoint_sha256") == payload.get("checkpoint_sha256")
    checks["feature_map_256x27x27"] = load.get("feature_map_shape") == [256, 27, 27]
    checks["feature_shape_256x27x27"] = load.get("feature_shape") == [256, 27, 27]
    checks["pooled_encoder_same"] = "same frozen checkpoint" in str(protocol.get("detector_encoder", ""))
    checks["teacher_gt_only"] = protocol.get("teacher_gt_only") is True and protocol.get("teacher_scope") == "train sequences only"
    checks["student_context_gt_false"] = protocol.get("student_context_gt") is False
    checks["gt_geometry_visibility_false"] = (protocol.get("gt_geometry_in_student") is False and
                                                protocol.get("gt_geometry_in_student_x") is False and
                                                protocol.get("gt_visibility_in_student") is False)
    checks["eval_uses_gt_false"] = protocol.get("eval_uses_gt") is False and ng.get("eval_uses_gt") is False
    checks["no_gt_tracker_argument"] = ng.get("gt_argument_in_tracker") is False
    checks["candidate_not_in_predictor"] = protocol.get("candidate_in_predictor") is False and ng.get("candidate_in_predictor") is False
    checks["tracking_precedes_eval_gt_parse"] = bool(ng.get("tracking_precedes_gt_parse"))
    checks["temporal_controls_present"] = set(protocol.get("strict_temporal_methods", [])) >= {"jepa_temporal", "no_teacher", "random_frozen"}
    # Validate every recorded cache, including the sidecar hash and checkpoint
    # binding.  A cache may live in the shared smoke directory by design.
    cache_checks = {}; cache_ok = True
    for seq, roles in dance.get("feature_cache", {}).items():
        cache_checks[seq] = {}
        for role, meta in roles.items():
            if not meta:
                continue
            cp = Path(str(meta.get("cache", ""))); expected_hash = meta.get("cache_sha256")
            ok = (cp.exists() and meta.get("checkpoint_sha256") == payload.get("checkpoint_sha256") and
                  meta.get("source_feature_shape") == [256, 27, 27] and meta.get("pooled_dim") == 256 and
                  meta.get("pooled_shape", [0, 0])[-1:] == [256] and
                  bool(expected_hash) and sha256(cp) == expected_hash)
            cache_checks[seq][role] = {"cache": str(cp), "exists": cp.exists(), "hash_matches": bool(cp.exists() and sha256(cp) == expected_hash), "pass": ok}
            cache_ok = cache_ok and ok
    checks["feature_cache_hashes_and_shapes"] = cache_ok and bool(cache_checks)
    te = out / "trackeval_official" / "audit.json"
    checks["trackeval_complete"] = bool(te.exists() and json.loads(te.read_text()).get("status") == "complete")
    source = Path(args.source_script).resolve(); copy = out / "provenance" / source.name
    checks["source_copy_hash_matches"] = bool(source.exists() and copy.exists() and sha256(source) == sha256(copy))
    report = {
        "schema": "dance-got-jepa-protocol-audit-v1", "result_dir": str(out),
        "result_sha256": sha256(result_path), "checkpoint": str(ckpt),
        "checkpoint_sha256": payload.get("checkpoint_sha256"), "cache_checks": cache_checks,
        "checks": checks, "all_pass": bool(all(bool(v) for v in checks.values())),
        "python": sys.version, "platform": platform.platform(),
    }
    path = out / "got_protocol_audit.json"; path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"all_pass": report["all_pass"], "path": str(path), "checks": checks}, indent=2, sort_keys=True))
    return 0 if report["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
