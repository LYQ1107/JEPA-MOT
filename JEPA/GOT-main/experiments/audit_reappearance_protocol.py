#!/usr/bin/env python3
"""Read-only protocol and lifecycle audit for the invariant reappearance run.

This audit deliberately does not read DanceTrack images or ground truth.  It
checks the committed tracker/trace artifacts, cache/checkpoint provenance, and
the independent official TrackEval report.  The no-GT replay is performed by
``audit_reappearance_invariant.py`` before TrackEval; its report is treated as
an input here and is checked for a passing status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


METHODS = ("raw", "reappearance_jepa", "no_teacher", "random_frozen")
PREDICTOR_METHODS = set(METHODS) - {"raw"}
EXPECTED_SEQUENCES = ("dancetrack0004", "dancetrack0007", "dancetrack0014")


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=float, separators=(",", ":")).encode()
    ).hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


class Audit:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.checks: Dict[str, Any] = {}

    def check(self, name: str, ok: bool, detail: Any = None) -> None:
        self.checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            self.failures.append(name if detail is None else f"{name}: {detail}")


def audit_trace(a: Audit, path: Path, expected_sha: str, expected_rows: int,
                sequence: str, method: str, lifecycle: bool) -> List[Dict[str, Any]]:
    label = f"trace.{sequence}.{method}.{'lifecycle' if lifecycle else 'events'}"
    if not path.exists():
        a.check(label + ".exists", False, str(path))
        return []
    a.check(label + ".file_sha256", sha256(path) == expected_sha,
            {"expected": expected_sha, "observed": sha256(path)})
    payload = load(path)
    rows = payload.get("rows")
    ok_rows = isinstance(rows, list) and len(rows) == int(expected_rows)
    a.check(label + ".row_count", ok_rows,
            {"expected": int(expected_rows), "observed": len(rows) if isinstance(rows, list) else None})
    a.check(label + ".rows_sha256", isinstance(rows, list) and payload.get("rows_sha256") == hash_json(rows),
            {"expected": payload.get("rows_sha256"), "observed": hash_json(rows) if isinstance(rows, list) else None})
    a.check(label + ".identity", payload.get("sequence") == sequence and payload.get("method") == method)
    return rows if isinstance(rows, list) else []


def audit_event_rows(a: Audit, rows: List[Dict[str, Any]], sequence: str, method: str) -> Dict[str, int]:
    label = f"events.{sequence}.{method}"
    failures: List[str] = []
    changed = 0
    outside = 0
    direct = 0
    jepa_by_frame: Dict[int, int] = {}
    frame_rows: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        try:
            frame = int(row["frame"])
            misses = int(row["misses_before"])
            gap_frames = int(row["gap_frames"])
            gap_ok = bool(row["gap_ok"])
            gate = bool(row["gate_pass"])
            used = bool(row["jepa_used"])
            native = row.get("native_assignment")
            assigned = row.get("used_assignment")
        except Exception as exc:
            failures.append(f"malformed:{exc}")
            continue
        expected_gap = bool(1 <= misses <= 4 and gap_frames == misses)
        if gap_ok != expected_gap:
            failures.append(f"bad_gap@{frame}")
        if method == "raw" and used:
            failures.append(f"raw_jepa_used@{frame}")
        if used and (method not in PREDICTOR_METHODS or not gap_ok or not gate):
            failures.append(f"illegal_jepa_use@{frame}")
        if misses == 0 and used:
            failures.append(f"continuous_jepa_use@{frame}")
        # The tracker records assignment_changed_rows while iterating the
        # *used* Hungarian matches.  Therefore an otherwise-native row that
        # becomes unmatched is not included in that counter; mirror that
        # definition here so the report can reproduce results.json exactly.
        if assigned is not None and native != assigned:
            changed += 1
            if used:
                direct += 1
            else:
                outside += 1
        frame_rows.setdefault(frame, []).append(row)
        if used:
            jepa_by_frame[frame] = jepa_by_frame.get(frame, 0) + 1
    # A non-JEPA assignment can differ only as a global Hungarian cascade on a
    # frame where a JEPA row was actually active.  Frames without JEPA rows
    # must be byte-for-byte native assignment semantics.
    for frame, fr_rows in frame_rows.items():
        has_jepa = bool(jepa_by_frame.get(frame, 0))
        for row in fr_rows:
            if row.get("native_assignment") != row.get("used_assignment") and not bool(row.get("jepa_used")) and not has_jepa:
                failures.append(f"unexplained_outside_change@{frame}")
            if not has_jepa and row.get("native_assignment") != row.get("used_assignment"):
                failures.append(f"no_jepa_frame_change@{frame}")
    a.check(label + ".contract", not failures, failures)
    return {"rows": len(rows), "changed": changed, "outside": outside, "direct": direct,
            "jepa_frames": len(jepa_by_frame), "jepa_rows": sum(jepa_by_frame.values())}


def audit_lifecycle_rows(a: Audit, rows: List[Dict[str, Any]], sequence: str, method: str) -> Dict[str, int]:
    label = f"lifecycle.{sequence}.{method}"
    failures: List[str] = []
    accepts = rejects = 0
    for row in rows:
        try:
            used = bool(row["jepa_used"])
            score = float(row["score"])
            update = bool(row["update_ok"])
            missing = bool(row["missing"])
            source = str(row["memory_source"])
            legal = bool(row["legal_raw_observation"])
            gate = bool(row["gate_pass"])
        except Exception as exc:
            failures.append(f"malformed:{exc}")
            continue
        if update:
            accepts += 1
        else:
            rejects += 1
        if not used:
            # This is the key lifecycle-invariant contract: all rows that did
            # not actually use a predictor copy raw SimpleMOT semantics.
            if source != "raw_exact" or not update or missing != bool(not (score >= 0.35)):
                failures.append(f"raw_semantics@{row.get('frame')}")
        else:
            if source != "jepa_gate" or update != bool(legal and gate) or missing != bool(not update):
                failures.append(f"jepa_gate_semantics@{row.get('frame')}")
    a.check(label + ".contract", not failures, failures)
    return {"rows": len(rows), "accept": accepts, "reject": rejects}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--source", default=None, help="tracker source used for provenance comparison")
    args = ap.parse_args()
    out = Path(args.result_dir).resolve()
    a = Audit()
    required = ["results.json", "protocol.json", "summary.json", "reflection.json", "provenance.json",
                "preflight/invariant_audit.json", "preflight/provenance.json", "trackeval_official/audit.json"]
    for rel in required:
        a.check("artifact." + rel, (out / rel).exists(), str(out / rel))
    if a.failures:
        report = {"schema": "dance-got-jepa-reappearance-protocol-audit-v1", "status": "failure",
                  "result_dir": str(out), "failures": a.failures, "checks": a.checks}
        (out / "reappearance_protocol_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True)); return 1

    results = load(out / "results.json"); protocol = load(out / "protocol.json")
    summary = load(out / "summary.json"); reflection = load(out / "reflection.json")
    provenance = load(out / "provenance.json")
    preflight = load(out / "preflight/invariant_audit.json")
    preflight_prov = load(out / "preflight/provenance.json")
    te = load(out / "trackeval_official/audit.json")

    a.check("results.status", results.get("status") == "complete", results.get("status"))
    a.check("results.methods", tuple(results.get("methods", [])) == METHODS, results.get("methods"))
    dance = results.get("dance", {})
    seqs = tuple(dance.get("eval_sequences", []))
    a.check("results.eval_sequences", seqs == EXPECTED_SEQUENCES, seqs)
    a.check("protocol.fixed_max_frames", protocol.get("max_frames") == 120)
    a.check("protocol.no_eval_gt", protocol.get("eval_gt_used_for_tracking") is False)
    a.check("protocol.no_posthoc", protocol.get("lifecycle", "").find("no posthoc") >= 0)
    a.check("protocol.candidate_independent", protocol.get("predictor_candidate_independent") is True)
    a.check("protocol.gap_rule", protocol.get("gap_rule") == "track misses>=1, frame_gap==misses, <=4; task-semantic not tuned")
    a.check("protocol.gates", protocol.get("quality_gate") == "quality>=0.35" and protocol.get("uncertainty_gate") == "sigmoid(log_variance)<=0.65")
    a.check("summary.status", summary.get("status") == "positive", summary.get("status"))
    a.check("summary.official", te.get("status") == "complete" and te.get("failures") == [],
            {"status": te.get("status"), "failures": te.get("failures")})
    a.check("reflection.official", reflection.get("official_trackeval_status") == "complete")
    a.check("preflight.pass", preflight.get("status") == "complete" and preflight.get("preflight_pass") is True and preflight.get("trackeval_invoked") is False,
            {"status": preflight.get("status"), "preflight_pass": preflight.get("preflight_pass"), "trackeval_invoked": preflight.get("trackeval_invoked")})
    a.check("preflight.no_gt", preflight.get("eval_gt_read") is False and preflight_prov.get("eval_gt_read") is False)

    # Check checkpoint-bound cache metadata without loading image/GT data.
    checkpoint = Path(str(results.get("checkpoint", "")))
    expected_ckpt = str(results.get("checkpoint_sha256", ""))
    a.check("checkpoint.exists", checkpoint.exists(), str(checkpoint))
    observed_ckpt = sha256(checkpoint) if checkpoint.exists() else None
    a.check("checkpoint.sha256", observed_ckpt == expected_ckpt, {"expected": expected_ckpt, "observed": observed_ckpt})
    cache_checks = []
    for seq, roles in (dance.get("feature_cache", {}) or {}).items():
        for role, meta in (roles or {}).items():
            ok = bool(meta.get("schema") == "got-jepa-pooled-cache-v1" and meta.get("checkpoint_sha256") == expected_ckpt
                       and meta.get("cache_sha256") == meta.get("cache_sha256_observed")
                       and meta.get("source_feature_shape") == [256, 27, 27]
                       and meta.get("pooled_dim") == 256 and meta.get("finite") is True)
            cache_checks.append({"sequence": seq, "role": role, "pass": ok})
    a.check("feature_cache.bound_and_shape", bool(cache_checks) and all(x["pass"] for x in cache_checks), cache_checks)

    no_gt = dance.get("no_gt_audit", {}) or {}
    a.check("no_gt.tracking_precedes_gt_parse", no_gt.get("eval_uses_gt") is False and no_gt.get("gt_argument_in_tracker") is False and no_gt.get("tracking_precedes_gt_parse") is True,
            {k: no_gt.get(k) for k in ("eval_uses_gt", "gt_argument_in_tracker", "tracking_precedes_gt_parse")})

    trace_stats: Dict[str, Any] = {}
    for seq in EXPECTED_SEQUENCES:
        trace_stats[seq] = {}
        seq_data = (dance.get("sequences", {}) or {}).get(seq, {})
        for method in METHODS:
            rec = seq_data.get(method, {})
            ev_ref = rec.get("event_trace", {})
            lc_ref = rec.get("lifecycle_trace", {})
            ev_rows = audit_trace(a, Path(str(ev_ref.get("path", ""))), str(ev_ref.get("sha256", "")), int(ev_ref.get("rows", -1)), seq, method, False)
            lc_rows = audit_trace(a, Path(str(lc_ref.get("path", ""))), str(lc_ref.get("sha256", "")), int(lc_ref.get("rows", -1)), seq, method, True)
            ev_stat = audit_event_rows(a, ev_rows, seq, method)
            lc_stat = audit_lifecycle_rows(a, lc_rows, seq, method)
            a.check(f"stats.{seq}.{method}.assignment_changed_rows", int(rec.get("assignment_changed_rows", -1)) == ev_stat["changed"], {"record": rec.get("assignment_changed_rows"), "trace": ev_stat["changed"]})
            a.check(f"stats.{seq}.{method}.outside_changes", int(rec.get("assignment_changed_outside_reappearance", -1)) == ev_stat["outside"], {"record": rec.get("assignment_changed_outside_reappearance"), "trace": ev_stat["outside"]})
            a.check(f"stats.{seq}.{method}.jepa_rows", int(rec.get("jepa_used_rows", -1)) == ev_stat["jepa_rows"], {"record": rec.get("jepa_used_rows"), "trace": ev_stat["jepa_rows"]})
            a.check(f"stats.{seq}.{method}.lifecycle_counts", int(rec.get("memory_update_accept", -1)) == lc_stat["accept"] and int(rec.get("memory_update_reject", -1)) == lc_stat["reject"], {"record": [rec.get("memory_update_accept"), rec.get("memory_update_reject")], "trace": [lc_stat["accept"], lc_stat["reject"]]})
            trace_stats[seq][method] = {"events": ev_stat, "lifecycle": lc_stat}
            if method == "raw":
                a.check(f"raw.{seq}.semantic_matches_preflight", rec.get("semantic_sha256") == ((preflight.get("eval", {}) or {}).get(seq, {}).get("raw", {}) or {}).get("raw_shadow_semantic_sha256"), {"record": rec.get("semantic_sha256"), "preflight": ((preflight.get("eval", {}) or {}).get(seq, {}).get("raw", {}) or {}).get("raw_shadow_semantic_sha256")})

    # Provenance copies and the tracker source hash must remain reproducible.
    prov_dir = out / "provenance"
    for name, recorded in (provenance.get("source_files", {}) or {}).items():
        p = prov_dir / name
        a.check("provenance." + name, p.exists() and sha256(p) == recorded, {"exists": p.exists(), "recorded": recorded, "observed": sha256(p) if p.exists() else None})
    if args.source:
        source = Path(args.source).resolve()
        recorded = provenance.get("source_files", {}).get(source.name)
        a.check("provenance.current_source", source.exists() and recorded == sha256(source), {"recorded": recorded, "observed": sha256(source) if source.exists() else None})

    combined = (te.get("combined", {}) or {})
    raw = combined.get("raw", {}) or {}
    cand = combined.get("reappearance_jepa", {}) or {}
    delta = {k: float(cand[k]) - float(raw[k]) for k in ("HOTA_official_mean", "HOTA_official_alpha0.50", "AssA_official_alpha0.50", "DetA_official_alpha0.50", "IDF1_official", "MOTA_official", "IDSW_official")}
    reported = ((summary.get("rows", {}) or {}).get("reappearance_jepa", {}) or {}).get("delta_vs_raw", {})
    a.check("metrics.delta_reproduced", all(abs(delta[k] - float(reported.get(k))) < 1e-12 for k in delta), {"computed": delta, "reported": reported})
    a.check("metrics.positive_hota_idf1", delta["HOTA_official_mean"] > 0 and delta["IDF1_official"] > 0)

    report = {"schema": "dance-got-jepa-reappearance-protocol-audit-v1",
              "status": "pass" if not a.failures else "failure", "result_dir": str(out),
              "dataset": "DanceTrack", "sequences": list(EXPECTED_SEQUENCES), "methods": list(METHODS),
              "gt_read": False, "failures": a.failures, "checks": a.checks,
              "trace_stats": trace_stats, "official_combined_delta_reappearance_vs_raw": delta,
              "python": sys.version, "platform": platform.platform()}
    (out / "reappearance_protocol_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    print(json.dumps({"status": report["status"], "out": str(out / "reappearance_protocol_audit.json"),
                      "failures": a.failures, "delta": delta}, indent=2, sort_keys=True, default=float))
    return 0 if not a.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
