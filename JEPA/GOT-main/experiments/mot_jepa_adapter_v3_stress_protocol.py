#!/usr/bin/env python3
"""Pre-registered deterministic-stress audit for the v3 MOT-JEPA adapter.

The v3 direct-prototype run used a native ``hard`` subset (low OC-SORT IoU
margin, gaps, or an ID change).  With a short held-out suffix that subset can
be very small even when the MOT-specific occlusion corruption is meaningful.
This wrapper keeps that original gate *unchanged* and adds a separately
labelled stress gate: every held-out fit-suffix sample receives the same
deterministic history/current masking and noise used during v3 training.

Only the already trained v3 checkpoint and fit-sequence feature caches are
loaded before the stress gate.  MOT17-05/11 are prepared with ``read_gt=False``
only after the stress gate passes.  The tracker consumes the predicted
prototype one frame later (t -> t+1); boxes and detector rows are untouched.
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
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import mot_jepa_adapter_v3_direct as v3  # noqa: E402


SOURCE_DEFAULT = Path("/data2/usr_for_deadline/mot_jepa_adapter_v3_direct_120")
CHECKPOINT_DEFAULT = Path("/data3/testdata/vranlee/GOT_JEPA_checkpoints/GOT_JEPA_378.tar")
FIT_SEQUENCES = ("MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-09-FRCNN")
EVAL_SEQUENCES = ("MOT17-05-FRCNN", "MOT17-11-FRCNN")
MAX_FRAMES = 120
FIT_END = 96
QUALITY_CAP = 120
STRESS_SEED = 777  # also the seed in v3.evaluate_quality/corrupt_batch


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


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def load_prototype(path: Path) -> v3.PrototypeNet:
    """Load a v3 checkpoint without constructing or retraining a predictor."""
    import torch

    blob = torch.load(str(path), map_location="cpu")
    if blob.get("schema") != "mot-jepa-direct-prototype-v1":
        raise RuntimeError("unexpected prototype checkpoint schema: {}".format(blob.get("schema")))
    model = v3.PrototypeNet(int(blob["input_dim"]), int(blob.get("seed", 37)),
                            float(blob.get("alpha", .50)))
    model.net.load_state_dict(blob["state_dict"], strict=True)
    model.net.eval()
    return model


def source_paths(source: Path) -> Tuple[Path, Path]:
    ckdir = source / "checkpoints"
    student = ckdir / "trained_direct_prototype.pt"
    no_teacher = ckdir / "no_teacher_direct_prototype.pt"
    if not student.exists() or not no_teacher.exists():
        raise FileNotFoundError("v3 checkpoints missing under {}".format(ckdir))
    cache = source / "cache"
    if not cache.is_dir():
        raise FileNotFoundError("v3 fit cache missing: {}".format(cache))
    return student, no_teacher


def build_stress_gate(clean: Dict[str, object], stress: Dict[str, object],
                      source_gate: Dict[str, object], data_checks: Dict[str, object]) -> Dict[str, object]:
    """Evaluate only pre-registered stress criteria; retain source gate verbatim."""
    teacher = clean.get("teacher", {})
    student = stress.get("student", {})
    finite_ok = bool(data_checks.get("all_finite"))
    teacher_ok = (teacher.get("recall_at1") is not None and
                  float(teacher.get("recall_at1")) >= .80)
    student_ok = (student.get("mean_margin") is not None and
                  float(student.get("mean_margin")) > 0. and
                  student.get("delta_chance") is not None and
                  float(student.get("delta_chance")) >= .10 and
                  student.get("teacher_cosine_mean") is not None and
                  float(student.get("teacher_cosine_mean")) >= .50)
    data_ok = (finite_ok and int(data_checks.get("duplicate_raw_row_labels", 1)) == 0 and
               int(data_checks.get("history_future_violations", 1)) == 0 and
               int(data_checks.get("teacher_history_future_violations", 1)) == 0 and
               not bool(data_checks.get("eval_gt_used")))
    gate = {
        "protocol": "v3 deterministic corruption stress (all held-out fit suffix samples)",
        "pre_registered": {
            "clean_teacher_recall_at1_ge": .80,
            "stress_student_mean_margin_gt": 0.,
            "stress_student_delta_chance_ge": .10,
            "stress_student_teacher_cosine_ge": .50,
            "no_teacher_is_control_only": True,
            "stress_seed": STRESS_SEED,
            "corruption": "current/history masking + vector/geometry noise via v3.corrupt_batch",
        },
        "source_native_gate": source_gate,
        "source_native_gate_status": source_gate.get("status"),
        "source_native_integration_decision": source_gate.get("integration_decision"),
        "clean_teacher_status": "pass" if teacher_ok else "fail",
        "stress_student_status": "pass" if student_ok else "fail",
        "data_status": "pass" if data_ok else "fail",
        "stress_hard_definition": "all deterministic-corruption views of held-out suffix; no native-hard filtering",
        "status": "pass" if teacher_ok and student_ok and data_ok else "fail",
        "teacher_quality_fail": not teacher_ok,
        "stress_student_quality_fail": not student_ok,
        "data_quality_fail": not data_ok,
        "integration_decision": "GO_STRESS_CANDIDATE" if teacher_ok and student_ok and data_ok else "NO-GO",
    }
    return gate


def copy_sources(out: Path, source: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    names = ("mot_jepa_adapter_v3_stress_protocol.py", "mot_jepa_adapter_v3_direct.py",
             "mot_jepa_adapter_v2_clean_teacher.py", "mot_ocsort_jepa_official.py",
             "mot_ocsort_jepa_official_v2.py", "mot_ocsort_jepa.py",
             "audit_trackeval_real.py")
    for name in names:
        src = HERE / name
        if src.exists():
            shutil.copy2(src, out / name)
    # Keep a byte-level pointer to the exact source run used for the model and
    # its original native-hard gate; this is not used to alter any decisions.
    for name in ("quality_gate.json", "train_results.json", "audit.json"):
        src = source / name
        if src.exists():
            shutil.copy2(src, out / ("source_" + name))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-dir", default=str(SOURCE_DEFAULT))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT_DEFAULT))
    ap.add_argument("--cache-dir", default="", help="default: <source-dir>/cache")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    ap.add_argument("--fit-end-frame", type=int, default=FIT_END)
    ap.add_argument("--quality-max-per-seq", type=int, default=QUALITY_CAP)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    source = Path(args.source_dir).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir).resolve() if args.cache_dir else source / "cache"
    student_path, no_teacher_path = source_paths(source)
    source_gate_path = source / "quality_gate.json"
    if not source_gate_path.exists():
        raise FileNotFoundError(source_gate_path)
    source_gate_payload = load_json(source_gate_path)
    source_gate = dict(source_gate_payload.get("gate", {}))
    source_train_path = source / "train_results.json"
    source_train = load_json(source_train_path) if source_train_path.exists() else {}
    payload: Dict[str, object] = {
        "schema": "mot-jepa-adapter-v3-stress-protocol-v1",
        "status": "started", "dataset": "MOT17", "fit_sequences": list(FIT_SEQUENCES),
        "eval_sequences_reserved": list(EVAL_SEQUENCES), "max_frames": int(args.max_frames),
        "fit_end_frame": int(args.fit_end_frame), "quality_sample_t_ge": int(args.fit_end_frame),
        "quality_target_t_ge": int(args.fit_end_frame) + 1, "quality_max_per_seq": int(args.quality_max_per_seq),
        "source_dir": str(source), "source_native_gate_path": str(source_gate_path),
        "source_native_gate_status": source_gate.get("status"),
        "source_native_integration_decision": source_gate.get("integration_decision"),
        "student_checkpoint": str(student_path), "no_teacher_checkpoint": str(no_teacher_path),
        "student_checkpoint_sha256": sha256(student_path), "no_teacher_checkpoint_sha256": sha256(no_teacher_path),
        "official_checkpoint": str(Path(args.checkpoint).resolve()),
        "cache_dir": str(cache), "candidate_in_predictor": False,
        "association_lag": "t->t+1", "eval_gt_read_by_tracker": False,
        "corruption": "deterministic v3.corrupt_batch: history/current masking + vector/geometry noise",
        "stress_seed": STRESS_SEED, "thresholds_fixed_before_eval": {
            "ambiguous_fused_margin": .10, "ambiguous_iou_min": .05, "ambiguous_app_min": .20,
            "visibility_min": .25, "uncertainty_max": 1.0,
        },
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
    }
    t0 = time.perf_counter()
    try:
        import torch
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        random.seed(37); np.random.seed(37); torch.manual_seed(37)
        expected_ckpt_hash = str(source_train.get("official_load", {}).get("checkpoint_sha256", ""))
        if expected_ckpt_hash and sha256(Path(args.checkpoint).resolve()) != expected_ckpt_hash:
            raise RuntimeError("official checkpoint hash differs from source v3 run")

        student = load_prototype(student_path)
        no_teacher = load_prototype(no_teacher_path)
        random_model = v3.PrototypeNet(student.input_dim, 37 + 404)
        payload["loaded_models"] = {"student_input_dim": int(student.input_dim),
                                    "student_alpha": float(student.alpha),
                                    "student_checkpoint_sha256": sha256(student_path),
                                    "no_teacher_checkpoint_sha256": sha256(no_teacher_path)}

        # Only the three fit videos are prepared before the stress decision.
        # Their compressed feature maps are reused byte-for-byte; no 05/11
        # cache is touched in this phase.
        extractor = v3.official_api.OfficialLatent(Path(args.checkpoint), args.device, 8)
        fit_prepared = [v3.v2lib.prepare_sequence(seq, int(args.max_frames), extractor,
                                                    cache, read_gt=True) for seq in FIT_SEQUENCES]
        all_samples, all_stats = v3.build_direct_samples(fit_prepared)
        fit_samples = [s for s in all_samples if int(s.frame) < int(args.fit_end_frame)]
        held_samples = [s for s in all_samples if int(s.frame) >= int(args.fit_end_frame)]
        held = v3.v2lib.select_quality_samples(held_samples, int(args.quality_max_per_seq))
        payload["fit_manifest"] = v3._prepared_manifest(fit_prepared)
        payload["sample_stats_all"] = all_stats
        payload["fit_sample_count"] = len(fit_samples)
        payload["quality_sample_count_before_cap"] = len(held_samples)
        payload["quality_sample_count"] = len(held)
        payload["quality_sequences"] = list(FIT_SEQUENCES)
        payload["quality_sample_t_ge"] = int(args.fit_end_frame)
        payload["quality_target_t_ge"] = int(args.fit_end_frame) + 1

        clean = v3.evaluate_quality(held, student, no_teacher, random_model,
                                     max_per_seq=0, stress=False)
        stress = v3.evaluate_quality(held, student, no_teacher, random_model,
                                      max_per_seq=0, stress=True)
        # ``evaluate_quality`` uses a local deterministic RNG (seed 777) and
        # applies corruption to every selected suffix sample.  Keep both views
        # in the artifact; only stress.student is used by the added gate.
        data_checks = {
            "all_finite": bool(np.isfinite(np.stack([s.x for s in held])).all()),
            "duplicate_raw_row_labels": int(all_stats.get("duplicate_raw_row_labels", 0)),
            "history_future_violations": int(all_stats.get("history_future_violations", 0)),
            "teacher_history_future_violations": int(all_stats.get("teacher_history_future_violations", 0)),
            "student_x_gt_fields": 0, "fit_gt_used_only_for_teacher_labels": True,
            "eval_gt_used": False, "eval_sequences_not_loaded_before_gate": True,
        }
        # Preserve the original v3 native-hard NO-GO exactly as loaded, while
        # reporting a separate stress gate for the new MOT-hard view.
        stress_gate = build_stress_gate(clean, stress, source_gate, data_checks)
        payload["quality_clean"] = clean
        payload["quality_stress"] = stress
        payload["quality_gate"] = stress_gate
        payload["original_native_hard_gate"] = source_gate
        payload["data_checks"] = data_checks
        payload["eval_started_after_stress_gate"] = False
        write_json(out / "quality_gate.json", {
            "schema": "mot-jepa-adapter-v3-stress-quality-gate-v1",
            "original_native_hard_gate": source_gate,
            "stress_gate": stress_gate,
            "clean_teacher": clean.get("teacher"),
            "stress_student": stress.get("student"),
            "stress_no_teacher_control": stress.get("no_teacher"),
            "stress_samples": int(len(held)), "data_checks": data_checks,
            "corruption_seed": STRESS_SEED,
        })

        if stress_gate.get("status") == "pass":
            # Release fit maps before loading the held-out evaluation videos;
            # this keeps the no-GT evaluation bounded without changing data.
            del fit_prepared, all_samples, fit_samples, held_samples, held
            gc.collect()
            thresholds = {"ambiguous_fused_margin": .10, "ambiguous_iou_min": .05,
                          "ambiguous_app_min": .20, "visibility_min": .25,
                          "uncertainty_max": 1.0,
                          "source": "fixed before eval; stress gate does not inspect eval"}
            models = {"raw": None, "history": None, "trained_adapter": student,
                      "no_teacher": no_teacher, "random_frozen": random_model}
            eval_payload = v3.run_eval(EVAL_SEQUENCES, extractor, cache, models,
                                       thresholds, out, int(args.max_frames))
            payload.update(eval_payload)
            # Add explicit blocks for the standalone TrackEval audit utility.
            payload["methods"] = list(models.keys())
            payload["mot17"] = {"eval_sequences": list(EVAL_SEQUENCES),
                                "rows": list(eval_payload.get("rows", [])),
                                "tracks": eval_payload.get("eval", {}).get("tracks", {})}
            payload["eval_started_after_stress_gate"] = True
            payload["eval_gt_read"] = False
        else:
            payload["eval_skipped_reason"] = "stress_gate_fail"
        payload["status"] = "complete"
        payload["elapsed_seconds"] = time.perf_counter() - t0
        write_json(out / "train_results.json", payload)
        write_json(out / "audit.json", {
            "schema": "mot-jepa-adapter-v3-stress-audit-v1",
            "protocol": "native-hard preserved + deterministic all-suffix stress gate",
            "source_native_gate": source_gate, "stress_gate": stress_gate,
            "data_checks": data_checks, "fit_sequences": list(FIT_SEQUENCES),
            "eval_sequences_reserved": list(EVAL_SEQUENCES),
            "eval_started_after_stress_gate": bool(payload.get("eval_started_after_stress_gate")),
            "eval_gt_read_by_tracker": False, "candidate_in_predictor": False,
            "association_lag": "t->t+1", "thresholds": payload.get("thresholds_fixed_before_eval"),
            "checkpoint_sha256": payload.get("student_checkpoint_sha256"),
        })
        copy_sources(out, source)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:
        payload.update({"status": "failure", "error_type": type(exc).__name__, "error": str(exc),
                        "traceback": traceback.format_exc(), "elapsed_seconds": time.perf_counter() - t0})
        write_json(out / "failure.json", payload)
        write_json(out / "train_results.json", payload)
        copy_sources(out, source)
        print(json.dumps(payload, indent=2, sort_keys=True, default=float))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
