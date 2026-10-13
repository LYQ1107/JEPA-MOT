"""Static and runtime preflight for the registered seed17 H2 Wave-A run.

The preflight deliberately does not train, open the tracker, or read target
identity annotations.  It verifies that the local prepared streams and dense
feature caches are internally consistent, records the source snapshots used by
the four-cell D/P factorial, and reports whether the current host can execute
the CUDA and NAS dependent rollout.
"""

from __future__ import annotations

import argparse
import datetime as dt
import errno
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
H2_CELLS = [
    {"id": "D_to_D", "source_initialization": "discriminative", "online_update": "discriminative"},
    {"id": "D_to_P", "source_initialization": "discriminative", "online_update": "predictive"},
    {"id": "P_to_D", "source_initialization": "predictive", "online_update": "discriminative"},
    {"id": "P_to_P", "source_initialization": "predictive", "online_update": "predictive"},
]


def stat_probe(path: Path) -> dict:
    """Return a serialisable probe without turning a stale FUSE mount into a crash."""
    try:
        info = path.stat()
    except OSError as error:
        kind = "transport_endpoint_not_connected" if error.errno == errno.ENOTCONN else "os_error"
        return {"status": kind, "errno": error.errno}
    return {"status": "ok", "size": info.st_size}


def file_record(path: Path, role: str) -> dict:
    record = stat_probe(path)
    record["role"] = role
    return record


def command_probe(arguments: list[str]) -> dict:
    try:
        result = subprocess.run(arguments, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": arguments, "status": "unavailable", "error": str(error)}
    return {
        "command": arguments,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout.strip()[-2000:],
        "stderr": result.stderr.strip()[-2000:],
    }


def git_value(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(arguments, cwd=WORKSPACE, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def runtime_probe() -> dict:
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as error:  # pragma: no cover - protects the audit on minimal hosts
        torch_info = {"status": "import_error", "error": repr(error)}
    return {
        "torch": torch_info,
        "nvidia_smi": command_probe(["nvidia-smi", "-L"]),
        "active_owned_processes": active_owned_processes(),
    }


def active_owned_processes() -> list[dict]:
    patterns = re.compile(r"(?:jepa_cmot\.(?:run_stream|continual_sequence|h2_fixed_teacher|h2_mechanism_probe)|ua_common_state_replay)")
    try:
        result = subprocess.run(["ps", "-eo", "pid=,args="], check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    own_pid = str(os.getpid())
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(own_pid + " ") or not patterns.search(line):
            continue
        pid, _, command = line.partition(" ")
        rows.append({"pid": pid, "command": command})
    return rows


def annotation_and_feature_record(entry: dict) -> dict:
    prepared = Path(entry["prepared"])
    features_path = Path(entry["features"])
    annotation_path = prepared / "annotations" / "val.json"
    result = {
        "video": entry["video"],
        "domain": entry["domain"],
        "declared_frames": int(entry["frames"]),
        "annotation": file_record(annotation_path, "prepared_detection_annotation"),
        "features": file_record(features_path, "dense_feature_cache"),
    }
    if result["annotation"]["status"] != "ok":
        result["annotation_checks"] = {"status": "unreadable"}
    else:
        annotation = json.loads(annotation_path.read_text())
        images = annotation.get("images", [])
        detections = sum(len(image.get("detection", [])) for image in images)
        flags = {
            "gt_read": annotation.get("gt_read"),
            "native_full_frames": annotation.get("native_full_frames"),
            "engineering_smoke_only": annotation.get("engineering_smoke_only"),
            "annotations_key_count": len(annotation.get("annotations", [])),
        }
        result["annotation_checks"] = {
            "status": "ok",
            "image_count": len(images),
            "detection_count": detections,
            "flags": flags,
            "target_gt_isolated": flags["gt_read"] is False and flags["annotations_key_count"] == 0,
            "native_frame_complete": flags["native_full_frames"] is True and len(images) == int(entry["frames"]),
        }
        if images:
            result["image_sample"] = stat_probe(Path(images[0]["file_name"]))
    if result["features"]["status"] != "ok":
        result["feature_checks"] = {"status": "unreadable"}
    else:
        with np.load(features_path, allow_pickle=False) as cache:
            features = cache["features"]
            offsets = cache["offsets"]
            image_ids = cache["image_ids"]
            consistent = (
                features.ndim == 2
                and features.shape[1] == 1024
                and features.dtype == np.float32
                and offsets.ndim == 1
                and len(offsets) == int(entry["frames"]) + 1
                and image_ids.ndim == 1
                and len(image_ids) == int(entry["frames"])
                and int(offsets[-1]) == len(features)
            )
            result["feature_checks"] = {
                "status": "ok",
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "offsets_shape": list(offsets.shape),
                "image_ids_shape": list(image_ids.shape),
                "consistent_with_prepared_stream": bool(consistent),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        default=WORKSPACE / "runs/JEPA_CMOT/continual_competition_v1/bdd100k_forward_seed17/inputs/development.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=WORKSPACE / "runs/JEPA_CMOT/continual_competition_v1/bdd100k_forward_seed17/protocol.json",
    )
    parser.add_argument(
        "--readiness",
        type=Path,
        default=WORKSPACE / "docs/JEPA_IDENTITY/H2_PROTOCOL_READINESS.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=WORKSPACE / "docs/JEPA_IDENTITY/H2_WAVE_A_PREFLIGHT_SEED17.json",
    )
    parser.add_argument(
        "--nas-root",
        type=Path,
        default=Path(os.environ.get("JEPA_NAS_ROOT", "nas/A100")),
        help="NAS root to probe; the value is intentionally omitted from the report.",
    )
    args = parser.parse_args()

    inputs = json.loads(args.inputs.read_text())
    readiness = json.loads(args.readiness.read_text()) if args.readiness.exists() else {}
    source_root = WORKSPACE / "runs/JEPA_CMOT"
    source_files = {
        "D_checkpoint": source_root / "source_history_v1/bdd_dense_current/seed17/discriminative/final.pt",
        "P_checkpoint": source_root / "source_history_v1/bdd_dense_current/seed17/predictive/final.pt",
        "D_calibration": source_root / "source_calibration_v1/bdd_dense_current/seed17/discriminative.json",
        "P_calibration": source_root / "source_calibration_v1/bdd_dense_current/seed17/predictive.json",
        "frozen_teacher_calibration": source_root / "source_calibration_v1/bdd_dense_current/seed17/frozen_last.json",
    }
    stream_records = [annotation_and_feature_record(entry) for entry in inputs]
    source_records = {name: file_record(path, name) for name, path in source_files.items()}

    nas_probe = stat_probe(args.nas_root)
    all_streams_structurally_ready = all(
        row["annotation"]["status"] == "ok"
        and row["features"]["status"] == "ok"
        and row["annotation_checks"].get("target_gt_isolated")
        and row["annotation_checks"].get("native_frame_complete")
        and row["feature_checks"].get("consistent_with_prepared_stream")
        for row in stream_records
    )
    images_accessible = all(row.get("image_sample", {}).get("status") == "ok" for row in stream_records)
    source_ready = all(row["status"] == "ok" for row in source_records.values())
    cuda = runtime_probe()
    cuda_ready = bool(cuda.get("torch", {}).get("cuda_available")) and cuda.get("nvidia_smi", {}).get("status") == "ok"
    result = {
        "status": "INPUTS_VALID_RUNTIME_BLOCKED" if all_streams_structurally_ready and source_ready else "INPUT_PREFLIGHT_FAILED",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": {
            "commit": git_value(["git", "rev-parse", "HEAD"]),
            "branch": git_value(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "registered_protocol": {
            "hypothesis": "H2",
            "wave": "A",
            "dataset": "bdd100k",
            "seed": 17,
            "domain_order": ["clear", "overcast", "rainy", "night"],
            "cells": H2_CELLS,
            "same_initialization": True,
            "readiness_status": readiness.get("status"),
        },
        "inputs": {
            "manifest": file_record(args.inputs, "development_stream_manifest"),
            "legacy_protocol_excluded_from_causal_wave_a": {
                "record": file_record(args.protocol, "legacy_continual_competition_protocol"),
                "reason": "Its four controls all use predictive_ranking with different retention graphs; it is not the registered D/P 2x2 factorial.",
            },
            "stream_count": len(stream_records),
            "declared_frame_count": sum(int(entry["frames"]) for entry in inputs),
            "stream_records": stream_records,
            "all_streams_structurally_ready": all_streams_structurally_ready,
        },
        "source_artifacts": source_records,
        "leakage_audit": {
            "target_gt_used_for_training_or_gate": False,
            "all_prepared_annotations_gt_read_false": all(
                row.get("annotation_checks", {}).get("flags", {}).get("gt_read") is False for row in stream_records
            ),
            "all_prepared_annotations_have_zero_gt_annotations": all(
                row.get("annotation_checks", {}).get("flags", {}).get("annotations_key_count") == 0 for row in stream_records
            ),
        },
        "runtime": {
            "nas": nas_probe,
            "nas_images_accessible_from_prepared_sample": images_accessible,
            "cuda_ready_for_formal_run": cuda_ready,
            **cuda,
        },
        "decision": {
            "source_checkpoints_and_calibrations_ready": source_ready,
            "local_detection_and_feature_inputs_ready": all_streams_structurally_ready,
            "formal_wave_a_started": False,
            "formal_wave_a_allowed_now": bool(source_ready and all_streams_structurally_ready and images_accessible and cuda_ready),
            "blockers": [
                reason
                for reason, failed in [
                    ("NAS_FUSE_ENDPOINT_DISCONNECTED", not images_accessible),
                    ("NVIDIA_DRIVER_UNAVAILABLE", not cuda_ready),
                ]
                if failed
            ],
            "next_action": "Re-run this preflight after the NAS SSHFS endpoint and NVIDIA driver are restored; then execute only one seed17 forward-order Wave-A factorial.",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    print(json.dumps({"status": result["status"], "out": str(args.out), "blockers": result["decision"]["blockers"]}, indent=2))


if __name__ == "__main__":
    main()
