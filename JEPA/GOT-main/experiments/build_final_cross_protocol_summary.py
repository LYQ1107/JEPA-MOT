#!/usr/bin/env python3
"""Build the immutable cross-protocol MOT+JEPA result index.

This is a read-only collector: it never opens images/GT and never reruns a
tracker.  The generated JSON/CSV keep the official TrackEval rows together
with protocol, audit, and provenance paths so that the final report cannot
silently mix the OC-SORT baseline, the MOT17 diagnostic harness, and the
DanceTrack GOT-JEPA harness.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main")
OUT_JSON = ROOT / "experiments/final_cross_protocol_summary.json"
OUT_CSV = ROOT / "experiments/final_cross_protocol_summary.csv"
CHECKPOINT = Path("/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar")
TRACK_EVAL_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/TrackEval-official")

METRIC_COLUMNS = (
    "HOTAmean",
    "HOTA@0.5",
    "AssA",
    "DetA",
    "IDF1",
    "MOTA",
    "IDSW",
)
CSV_METRIC_COLUMNS = {
    "HOTAmean": "HOTA_official_mean",
    "HOTA@0.5": "HOTA_official_alpha0.50",
    "AssA": "AssA_official_alpha0.50",
    "DetA": "DetA_official_alpha0.50",
    "IDF1": "IDF1_official",
    "MOTA": "MOTA_official",
    "IDSW": "IDSW_official",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def metric_row(raw: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"sequence": raw["sequence"], "method": raw["method"]}
    for name, column in CSV_METRIC_COLUMNS.items():
        value = float(raw[column])
        out[name] = int(value) if name == "IDSW" and value.is_integer() else value
    return out


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        return [metric_row(r) for r in csv.DictReader(f)]


def with_deltas(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Official CSV has one raw row per sequence.  Keep deltas local to a
    # protocol so a baseline from another harness can never become a control.
    raw_by_seq = {
        r["sequence"]: r for r in rows if r["method"] == "raw"
    }
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        raw = raw_by_seq.get(row["sequence"])
        if raw is not None and row["method"] != "raw":
            item["delta_vs_raw"] = {
                key: row[key] - raw[key] for key in METRIC_COLUMNS
            }
        else:
            item["delta_vs_raw"] = None
        out.append(item)
    return out


def audit_status(path: Path, status_keys: Iterable[str] = ("status",)) -> Any:
    if not path.exists():
        return "missing"
    try:
        payload = load_json(path)
    except Exception as exc:  # pragma: no cover - defensive provenance path
        return f"unreadable:{exc}"
    for key in status_keys:
        if key in payload:
            return payload[key]
    # A few early audits predate the uniform top-level ``status`` field.  Do
    # not label those as missing when their explicit contract says all checks
    # passed; retain the distinction in the referenced JSON itself.
    if payload.get("all_pass") is True or payload.get("preflight_pass") is True:
        return "pass"
    if (
        payload.get("source_stress_gate_status") == "pass"
        and payload.get("no_repair_identity_invariant", {}).get("status") == "pass"
    ):
        return "pass"
    if payload.get("failures") == []:
        return "pass"
    return "no_top_level_status"


def check_entry(
    name: str,
    path: Path,
    *,
    result: Optional[Path] = None,
    trackeval: Optional[Path] = None,
    protocol_audit: Optional[Path] = None,
    preflight: Optional[Path] = None,
    reflection: Optional[Path] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"name": name, "artifact_dir": str(path)}
    refs = {
        "result": result,
        "trackeval_audit": trackeval,
        "protocol_audit": protocol_audit,
        "preflight_audit": preflight,
        "reflection": reflection,
    }
    entry["audit_paths"] = {k: str(v) for k, v in refs.items() if v is not None}
    entry["audit_status"] = {
        k: audit_status(v) for k, v in refs.items() if v is not None
    }
    return entry


def make_spec(
    protocol_id: str,
    dataset: str,
    frames: int,
    seed: Optional[int],
    artifact_dir: str,
    validity: str,
    note: str,
    *,
    csv_path: Optional[str] = None,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    artifact = Path(artifact_dir)
    if csv_path is None:
        csv_file = artifact / "trackeval_official/official_metrics.csv"
    else:
        csv_file = Path(csv_path)
    rows = with_deltas(load_metrics(csv_file))
    for row in rows:
        row.update(
            {
                "protocol_id": protocol_id,
                "dataset": dataset,
                "frames": frames,
                "seed": seed,
                "validity": validity,
            }
        )
    spec: Dict[str, Any] = {
        "protocol_id": protocol_id,
        "dataset": dataset,
        "frames": frames,
        "seed": seed,
        "artifact_dir": artifact_dir,
        "official_metrics_csv": str(csv_file),
        "official_metrics_csv_sha256": sha256(csv_file),
        "validity": validity,
        "note": note,
        "methods": methods or sorted({r["method"] for r in rows}),
        "rows": rows,
    }
    return spec


def combined(spec: Dict[str, Any], method: str) -> Optional[Dict[str, Any]]:
    for row in spec["rows"]:
        if row["sequence"] == "COMBINED_SEQ" and row["method"] == method:
            return row
    return None


def excluded_crop80() -> List[Dict[str, Any]]:
    # These are retained as provenance only.  They use the old 80-pixel crop
    # path and therefore must not be pooled with strict GOT-JEPA 256-D rows.
    out = []
    for seed, directory in (
        (17, "/data2/usr_for_deadline/real_dance_temporal_repro_300_s17"),
        (23, "/data2/usr_for_deadline/real_dance_temporal_repro_300_s23"),
    ):
        csv_path = Path(directory) / "trackeval_official/official_metrics.csv"
        rows = load_metrics(csv_path)
        for row in rows:
            if row["sequence"] == "COMBINED_SEQ":
                out.append(
                    {
                        "dataset": "DanceTrack",
                        "frames": 300,
                        "seed": seed,
                        "method": row["method"],
                        "metrics": {k: row[k] for k in METRIC_COLUMNS},
                        "artifact_dir": directory,
                        "official_metrics_csv": str(csv_path),
                        "official_metrics_csv_sha256": sha256(csv_path),
                        "validity": "excluded_crop80_not_comparable",
                    }
                )
    return out


def main() -> None:
    specs = [
        make_spec(
            "mot17_ocsort_baseline_120",
            "MOT17",
            120,
            None,
            "/data1/LWR/vranlee/SERVER_ONLY/avis/JEPA/GOT-main/experiments/results/baseline_ocsort_mot120",
            "official_baseline",
            "Formal public-FRCNN OC-SORT run on five MOT17 sequences; no JEPA.",
            methods=["ocsort"],
        ),
        make_spec(
            "mot17_jepa_gate_sweep_120",
            "MOT17",
            120,
            None,
            "/data2/usr_for_deadline/mot_jepa_adapter_v4_gate_sweep_120",
            "official_negative_or_noop",
            "MOT17 diagnostic association harness; A is negative, B/C/D are exact raw no-ops.",
            methods=["raw", "A", "B", "C", "D"],
        ),
        make_spec(
            "dance_got_unconditional_120_s17",
            "DanceTrack",
            120,
            17,
            "/data2/usr_for_deadline/mot_dance_jepa_got_full120_s17",
            "official_no_go",
            "Strict GOT-JEPA 256-D pooled appearance with unconditional temporal replacement.",
            methods=["raw", "jepa_temporal", "no_teacher", "random_frozen"],
        ),
        make_spec(
            "dance_got_ambiguity_120_s17",
            "DanceTrack",
            120,
            17,
            "/data2/usr_for_deadline/mot_dance_jepa_got_ambiguity120_s17",
            "official_no_op",
            "Strict 256-D ambiguity-only gate; zero eligible eval applications and exact raw output.",
            methods=["raw", "ambiguity_jepa", "no_teacher", "random_frozen"],
        ),
        make_spec(
            "dance_reappearance_120_s17",
            "DanceTrack",
            120,
            17,
            "/data2/usr_for_deadline/mot_dance_jepa_got_reappearance120_s17_invariant",
            "official_boundary_positive",
            "Lifecycle-invariant reappearance-only JEPA adapter; fixed gap/quality/uncertainty gate.",
            methods=["raw", "reappearance_jepa", "no_teacher", "random_frozen"],
        ),
        make_spec(
            "dance_reappearance_120_s23",
            "DanceTrack",
            120,
            23,
            "/data2/usr_for_deadline/mot_dance_jepa_got_reappearance120_s23_invariant",
            "official_boundary_positive",
            "Second fixed seed of the lifecycle-invariant reappearance-only adapter.",
            methods=["raw", "reappearance_jepa", "no_teacher", "random_frozen"],
        ),
        make_spec(
            "dance_reappearance_300_s17",
            "DanceTrack",
            300,
            17,
            "/data2/usr_for_deadline/mot_dance_jepa_got_reappearance300_s17_invariant",
            "official_mixed",
            "Longer 300-frame confirmation; positive candidate direction but small effect.",
            methods=["raw", "reappearance_jepa", "no_teacher", "random_frozen"],
        ),
        make_spec(
            "dance_reappearance_300_s23",
            "DanceTrack",
            300,
            23,
            "/data2/usr_for_deadline/mot_dance_jepa_got_reappearance300_s23_invariant",
            "official_negative",
            "Longer 300-frame confirmation; candidate fails the HOTA+IDF1 positivity rule.",
            methods=["raw", "reappearance_jepa", "no_teacher", "random_frozen"],
        ),
    ]

    audit_entries = [
        check_entry(
            "mot17_ocsort_baseline_120",
            Path(specs[0]["artifact_dir"]),
            result=Path(specs[0]["artifact_dir"]) / "results.json",
            trackeval=Path(specs[0]["artifact_dir"]) / "trackeval_official/audit.json",
            reflection=Path(specs[0]["artifact_dir"]) / "reflection.json",
        ),
        check_entry(
            "mot17_jepa_gate_sweep_120",
            Path(specs[1]["artifact_dir"]),
            result=Path(specs[1]["artifact_dir"]) / "results.json",
            trackeval=Path(specs[1]["artifact_dir"]) / "trackeval_official/audit.json",
        ),
        check_entry(
            "dance_got_unconditional_120_s17",
            Path(specs[2]["artifact_dir"]),
            result=Path(specs[2]["artifact_dir"]) / "results.json",
            trackeval=Path(specs[2]["artifact_dir"]) / "trackeval_official/audit.json",
            protocol_audit=Path(specs[2]["artifact_dir"]) / "got_protocol_audit.json",
            reflection=Path(specs[2]["artifact_dir"]) / "reflection.json",
        ),
        check_entry(
            "dance_got_ambiguity_120_s17",
            Path(specs[3]["artifact_dir"]),
            result=Path(specs[3]["artifact_dir"]) / "results.json",
            trackeval=Path(specs[3]["artifact_dir"]) / "trackeval_official/audit.json",
            protocol_audit=Path("/data2/usr_for_deadline/mot_dance_jepa_got_ambiguity_event_audit_s17/event_reason_audit.json"),
            reflection=Path(specs[3]["artifact_dir"]) / "reflection.json",
        ),
    ]
    for spec in specs[4:]:
        artifact = Path(spec["artifact_dir"])
        audit_entries.append(
            check_entry(
                spec["protocol_id"],
                artifact,
                result=artifact / "results.json",
                trackeval=artifact / "trackeval_official/audit.json",
                protocol_audit=artifact / "reappearance_protocol_audit.json",
                preflight=artifact / "preflight/invariant_audit.json",
                reflection=artifact / "reflection.json",
            )
        )

    pair_audits = {
        "120": "/data2/usr_for_deadline/mot_dance_jepa_got_reappearance120_seedpair_invariant/seed_pair_audit.json",
        "300": "/data2/usr_for_deadline/mot_dance_jepa_got_reappearance300_seedpair_invariant/seed_pair_audit.json",
    }
    paired = {}
    for frames, path in pair_audits.items():
        payload = load_json(Path(path))
        paired[frames] = {
            "path": path,
            "status": payload.get("status"),
            "candidate_positive_both_seeds": payload.get("candidate_positive_both_seeds"),
            "paired_reappearance_vs_raw": payload.get("paired_reappearance_vs_raw"),
            "run_dirs": payload.get("run_dirs"),
        }

    manifest = Path("/data2/usr_for_deadline/mot_dance_jepa_got_300/cache_manifest.json")
    training = {
        "encoder_retrained": False,
        "frozen_checkpoint": str(CHECKPOINT),
        "student_temporal_predictor_epochs": 8,
        "teacher": "clean frozen encoder/teacher; GT labels used only on train sequences",
        "student_inputs": "detector-history appearance + observable box/velocity/age/score/missing mask; current candidate excluded",
        "train_sequences_dance": ["dancetrack0001", "dancetrack0002", "dancetrack0006"],
        "eval_sequences_dance": ["dancetrack0004", "dancetrack0007", "dancetrack0014"],
        "eval_gt_read_by_tracker": False,
        "note": "The public GOT-JEPA encoder is not retrained in this PoC; only the MOT temporal adapter is trained.",
    }

    summary: Dict[str, Any] = {
        "schema": "mot-jepa-final-cross-protocol-summary-v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workspace": str(ROOT),
        "metric_definition": "Official vendored TrackEval HOTA/CLEAR/Identity; sequence-truncated staging; no proxy aliases.",
        "harness_boundary": {
            "mot17_baseline": "OC-SORT public-detection formal run",
            "mot17_jepa": "isolated MOT17 association diagnostic, not OC-SORT leaderboard integration",
            "dance_jepa": "SimpleMOT/Hungarian association harness with fixed detector rows and shared lifecycle",
            "comparison_rule": "Only within-protocol deltas versus that protocol's raw row are valid; absolute OC-SORT-vs-JEPA comparison is not claimed.",
        },
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": sha256(CHECKPOINT),
        },
        "cache": {
            "strict_120_dir": "/data2/usr_for_deadline/mot_dance_jepa_got_120/feature_cache",
            "strict_300_dir": str(manifest.parent / "feature_cache"),
            "strict_300_manifest": str(manifest),
            "strict_300_manifest_sha256": sha256(manifest),
            "strict_feature_shape": [256, 27, 27],
            "pooled_dim": 256,
            "cache_manifest_status": audit_status(manifest),
        },
        "training": training,
        "audit_statuses": audit_entries,
        "pair_audits": paired,
        "protocols": specs,
        "rows": [row for spec in specs for row in spec["rows"]],
        "excluded_evidence": {
            "crop80_temporal": excluded_crop80(),
            "reason": "Old crop80 path is a different representation/protocol and is not pooled with strict GOT-JEPA 256-D results.",
        },
        "decisions": {
            "baseline": "OC-SORT",
            "mot17_jepa_gate_sweep": "NO-GO (A negative; B/C/D exact no-op)",
            "dance_unconditional_temporal": "NO-GO",
            "dance_ambiguity_only": "NO-OP",
            "dance_reappearance_120": "boundary-positive in both fixed seeds; modest effect",
            "dance_reappearance_300": "mixed; stop expansion, retain 120-frame boundary evidence",
            "teacher_attribution": "not established; no_teacher is required control",
            "next_engineering_step": "Port the invariant reappearance adapter into OC-SORT/BoT-SORT and run full MOT17/DanceTrack benchmark.",
        },
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    headers = [
        "protocol_id", "dataset", "frames", "seed", "sequence", "method", "validity",
        *METRIC_COLUMNS,
        *(f"delta_{k}_vs_raw" for k in METRIC_COLUMNS),
        "artifact_dir", "official_metrics_csv",
    ]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in summary["rows"]:
            out = {k: row.get(k) for k in headers}
            delta = row.get("delta_vs_raw") or {}
            for key in METRIC_COLUMNS:
                out[f"delta_{key}_vs_raw"] = delta.get(key)
            spec = next(s for s in specs if s["protocol_id"] == row["protocol_id"])
            out["artifact_dir"] = spec["artifact_dir"]
            out["official_metrics_csv"] = spec["official_metrics_csv"]
            writer.writerow(out)
    print(json.dumps({"json": str(OUT_JSON), "csv": str(OUT_CSV), "rows": len(summary["rows"])}, indent=2))


if __name__ == "__main__":
    main()
