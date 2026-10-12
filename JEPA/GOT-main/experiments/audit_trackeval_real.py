#!/usr/bin/env python3
"""Run the vendored official TrackEval on an existing real-MOT smoke.

This is an *audit* utility, not a tracker and not a modification of
``mot_dance_jepa_real.py``.  It stages a frame-truncated MOTChallenge copy
under the requested output directory, because TrackEval rejects GT rows past
the configured sequence length.  The source GT and tracker files are read
only; staged files are hashed and listed in the resulting JSON.

Example (DanceTrack v2):

    python experiments/audit_trackeval_real.py \
      --result-dir experiments/results/real_dance_sanity_v2 \
      --dataset dance --max-frames 120 \
      --out-dir experiments/results/real_dance_sanity_v2/trackeval_official

The vendored TrackEval checkout currently uses removed NumPy aliases
(``np.float``, ``np.int``).  The script installs local compatibility aliases
in memory before importing TrackEval; no reference checkout is edited.
"""
from __future__ import print_function

import argparse
import configparser
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DANCE_ROOT = "/data1/LWR/vranlee/DATASETS/JDE/dancetrack"
DEFAULT_MOT_ROOT = "/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/MOT17"
DEFAULT_TRACKEVAL = "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT/references/TrackEval-official"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_text(path: Path) -> str:
    return sha256_file(path) if path.exists() else "missing"


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_seqinfo(path: Path) -> Dict[str, str]:
    cfg = configparser.ConfigParser()
    if cfg.read(str(path)) and "Sequence" in cfg:
        return {k: v for k, v in cfg["Sequence"].items()}
    # A tiny fallback for malformed but otherwise readable seqinfo files.
    out: Dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def infer_sequences(payload: Dict[str, object], dataset: str, result_dir: Path) -> List[str]:
    block = payload.get(dataset)
    if isinstance(block, dict):
        vals = block.get("eval_sequences")
        if isinstance(vals, list) and vals:
            return [str(x) for x in vals]
        rows = block.get("rows")
        if isinstance(rows, list):
            seqs = sorted({str(r.get("sequence")) for r in rows if isinstance(r, dict) and r.get("sequence")})
            if seqs:
                return seqs
    # Last-resort discovery from track files.  This does not invent data; it
    # only enumerates files already present in the supplied result directory.
    roots = [result_dir / dataset / "tracks", result_dir / "tracks"]
    for root in roots:
        if not root.is_dir():
            continue
        files = list(root.glob("*/*.txt"))
        seqs = sorted({p.stem for p in files})
        if seqs:
            return seqs
    return []


def infer_methods(payload: Dict[str, object], dataset: str, result_dir: Path) -> List[str]:
    vals = payload.get("methods")
    if isinstance(vals, list) and vals:
        return [str(x) for x in vals]
    block = payload.get(dataset)
    if isinstance(block, dict):
        rows = block.get("rows")
        if isinstance(rows, list):
            methods = sorted({str(r.get("method")) for r in rows if isinstance(r, dict) and r.get("method")})
            if methods:
                return methods
    roots = [result_dir / dataset / "tracks", result_dir / "tracks"]
    for root in roots:
        if root.is_dir():
            methods = sorted(p.name for p in root.iterdir() if p.is_dir())
            if methods:
                return methods
    return []


def choose_gt_root(dataset: str, args: argparse.Namespace) -> Path:
    if dataset == "dance":
        return Path(args.dance_root)
    return Path(args.mot_root)


def source_seq_dir(dataset: str, seq: str, args: argparse.Namespace) -> Path:
    root = choose_gt_root(dataset, args)
    # The current smoke uses DanceTrack validation and MOT17 train videos.
    split = "val" if dataset == "dance" else "train"
    return root / split / seq


def tracker_source_path(result_dir: Path, dataset: str, method: str, seq: str) -> Path:
    candidates = [
        result_dir / dataset / "tracks" / method / (seq + ".txt"),
        result_dir / "tracks" / method / (seq + ".txt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def truncate_mot_file(src: Path, dst: Path, max_frame: int, is_gt: bool) -> Dict[str, object]:
    """Copy only frames <= max_frame and enforce numeric MOT rows."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    max_seen = 0
    with src.open("r", encoding="utf-8", errors="replace") as fi, dst.open("w", encoding="utf-8") as fo:
        for line_no, line in enumerate(fi, 1):
            s = line.strip()
            if not s:
                continue
            vals = [x.strip() for x in s.split(",")]
            if len(vals) < 6:
                skipped += 1
                continue
            try:
                frame = int(float(vals[0]))
                float(vals[1])
                for x in vals[2:]:
                    float(x)
            except Exception:
                skipped += 1
                continue
            max_seen = max(max_seen, frame)
            if frame < 1 or frame > max_frame:
                continue
            # TrackEval requires GT class/mark columns.  Official DanceTrack
            # and MOT17 GT already have them; retain all source columns and
            # append ignored world coordinates only when needed.
            if is_gt and len(vals) < 8:
                vals += ["1"] * (8 - len(vals))
            if len(vals) < 10:
                vals += ["-1"] * (10 - len(vals))
            fo.write(",".join(vals) + "\n")
            kept += 1
    return {"source": str(src), "destination": str(dst), "kept_rows": kept,
            "skipped_rows": skipped, "source_max_frame_seen": max_seen,
            "source_sha256": hash_text(src), "staged_sha256": hash_text(dst)}


def stage_dataset(
    result_dir: Path,
    out_dir: Path,
    dataset: str,
    sequences: Sequence[str],
    methods: Sequence[str],
    args: argparse.Namespace,
    max_frames: int,
) -> Tuple[Path, Path, Dict[str, object]]:
    stage = out_dir / "staging"
    gt_root = stage / "gt"
    tracker_root = stage / "trackers"
    if stage.exists():
        # This directory is owned by this audit invocation.  Remove only the
        # prior staging subtree, never the supplied result directory or data.
        shutil.rmtree(str(stage))
    gt_root.mkdir(parents=True, exist_ok=True)
    tracker_root.mkdir(parents=True, exist_ok=True)
    provenance: Dict[str, object] = {"dataset": dataset, "max_frames": int(max_frames), "sequences": {}, "trackers": {}}
    actual_lengths: Dict[str, int] = {}
    for seq in sequences:
        seq_dir = source_seq_dir(dataset, seq, args)
        gt_src = seq_dir / "gt" / "gt.txt"
        if not gt_src.exists():
            raise FileNotFoundError("GT not found: {}".format(gt_src))
        info = parse_seqinfo(seq_dir / "seqinfo.ini")
        seq_len_src = int(float(info.get("seqlength", max_frames))) if info.get("seqlength") else max_frames
        length = min(max_frames, seq_len_src) if max_frames > 0 else seq_len_src
        actual_lengths[seq] = length
        gt_dst = gt_root / seq / "gt" / "gt.txt"
        gt_meta = truncate_mot_file(gt_src, gt_dst, length, is_gt=True)
        ini_dst = gt_root / seq / "seqinfo.ini"
        ini_dst.write_text("[Sequence]\nname={}\nimDir=img1\nframeRate={}\nseqLength={}\nimWidth={}\nimHeight={}\nimExt=.jpg\n".format(
            seq, info.get("framerate", "20"), length, info.get("imwidth", "0"), info.get("imheight", "0")))
        provenance["sequences"][seq] = {"source_seq_dir": str(seq_dir), "source_seqinfo": info,
                                         "effective_length": length, "gt": gt_meta,
                                         "staged_seqinfo_sha256": hash_text(ini_dst)}
        for method in methods:
            src = tracker_source_path(result_dir, dataset, method, seq)
            if not src.exists():
                raise FileNotFoundError("tracker file not found: {}".format(src))
            dst = tracker_root / method / "data" / (seq + ".txt")
            meta = truncate_mot_file(src, dst, length, is_gt=False)
            provenance["trackers"].setdefault(method, {})[seq] = meta
    provenance["seq_info"] = actual_lengths
    return gt_root, tracker_root, provenance


def install_numpy_compat() -> None:
    # TrackEval's pinned code predates NumPy 1.24.  Keep this process-local.
    aliases = {"float": float, "int": int, "bool": bool, "object": object, "str": str}
    for name, value in aliases.items():
        # Checking ``__dict__`` avoids NumPy's FutureWarning-emitting
        # ``hasattr`` shim for the removed aliases.
        if name not in np.__dict__:
            setattr(np, name, value)


def git_head(path: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def run_official(
    gt_root: Path,
    tracker_root: Path,
    out_dir: Path,
    sequences: Sequence[str],
    methods: Sequence[str],
    lengths: Dict[str, int],
    trackeval_root: Path,
) -> Dict[str, object]:
    install_numpy_compat()
    sys.path.insert(0, str(trackeval_root))
    import trackeval  # noqa: E402

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({
        "USE_PARALLEL": False, "BREAK_ON_ERROR": True, "RETURN_ON_ERROR": False,
        "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": True, "PRINT_CONFIG": False,
        "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True, "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
        "LOG_ON_ERROR": str(out_dir / "trackeval_error.log"),
    })
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({
        "GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(tracker_root),
        "OUTPUT_FOLDER": str(out_dir / "official_output"), "TRACKERS_TO_EVAL": list(methods),
        "TRACKER_DISPLAY_NAMES": list(methods), "CLASSES_TO_EVAL": ["pedestrian"],
        # MOT17 semantics are appropriate for person boxes and retain the
        # official preprocessing path.  DanceTrack has only class=1 people.
        "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train", "SEQ_INFO": dict(lengths),
        "SKIP_SPLIT_FOL": True, "TRACKER_SUB_FOLDER": "data", "OUTPUT_SUB_FOLDER": "",
        "DO_PREPROC": True, "PRINT_CONFIG": False,
    })
    metric_cfg = {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    metrics = [trackeval.metrics.HOTA(metric_cfg), trackeval.metrics.CLEAR(metric_cfg), trackeval.metrics.Identity(metric_cfg)]
    evaluator = trackeval.Evaluator(eval_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    result, messages = evaluator.evaluate([dataset], metrics)
    dataset_name = dataset.get_name()
    return {"dataset_name": dataset_name, "result": result, "messages": messages}


def scalar(x: object) -> Optional[float]:
    try:
        a = np.asarray(x)
        if a.size == 0:
            return None
        return float(a.reshape(-1)[0])
    except Exception:
        return None


def metric_at(values: object, alpha: float = 0.5) -> Optional[float]:
    try:
        a = np.asarray(values, dtype=np.float64).reshape(-1)
        if a.size == 0:
            return None
        # HOTA labels in official TrackEval are 0.05, ..., 0.95.
        labels = np.arange(0.05, 0.99, 0.05)
        idx = int(np.argmin(np.abs(labels[: a.size] - alpha)))
        return float(a[idx])
    except Exception:
        return scalar(values)


def metric_mean(values: object) -> Optional[float]:
    try:
        a = np.asarray(values, dtype=np.float64).reshape(-1)
        return float(np.mean(a)) if a.size else None
    except Exception:
        return scalar(values)


def extract_rows(official: Dict[str, object], sequences: Sequence[str], methods: Sequence[str]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    ds_name = official["dataset_name"]
    result = official["result"][ds_name]
    rows: List[Dict[str, object]] = []
    combined: Dict[str, object] = {}
    for method in methods:
        mres = result.get(method)
        if not isinstance(mres, dict):
            continue
        for seq in list(sequences) + ["COMBINED_SEQ"]:
            sres = mres.get(seq)
            if not isinstance(sres, dict):
                continue
            cls = sres.get("pedestrian", {})
            if not isinstance(cls, dict):
                continue
            hota = cls.get("HOTA", {})
            clear = cls.get("CLEAR", {})
            ident = cls.get("Identity", {})
            row = {
                "method": method, "sequence": seq,
                "HOTA_official_mean": metric_mean(hota.get("HOTA")) if isinstance(hota, dict) else None,
                "HOTA_official_alpha0.50": metric_at(hota.get("HOTA")) if isinstance(hota, dict) else None,
                "DetA_official_alpha0.50": metric_at(hota.get("DetA")) if isinstance(hota, dict) else None,
                "AssA_official_alpha0.50": metric_at(hota.get("AssA")) if isinstance(hota, dict) else None,
                "LocA_official_alpha0.50": metric_at(hota.get("LocA")) if isinstance(hota, dict) else None,
                "MOTA_official": scalar(clear.get("MOTA")) if isinstance(clear, dict) else None,
                "MOTP_official": scalar(clear.get("MOTP")) if isinstance(clear, dict) else None,
                "IDSW_official": scalar(clear.get("IDSW")) if isinstance(clear, dict) else None,
                "IDF1_official": scalar(ident.get("IDF1")) if isinstance(ident, dict) else None,
                "IDTP_official": scalar(ident.get("IDTP")) if isinstance(ident, dict) else None,
                "IDFP_official": scalar(ident.get("IDFP")) if isinstance(ident, dict) else None,
                "IDFN_official": scalar(ident.get("IDFN")) if isinstance(ident, dict) else None,
            }
            rows.append(row)
            if seq == "COMBINED_SEQ":
                combined[method] = row
    return rows, combined


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result-dir", required=True)
    ap.add_argument("--dataset", choices=["dance", "mot17"], default="dance")
    ap.add_argument("--dance-root", default=DEFAULT_DANCE_ROOT)
    ap.add_argument("--mot-root", default=DEFAULT_MOT_ROOT)
    ap.add_argument("--trackeval-root", default=DEFAULT_TRACKEVAL)
    ap.add_argument("--sequences", default="", help="comma-separated; default inferred from results.json")
    ap.add_argument("--methods", default="", help="comma-separated; default inferred from results.json")
    ap.add_argument("--max-frames", type=int, default=0, help="0 uses result metadata or source seq length")
    ap.add_argument("--out-dir", default="", help="default: <result-dir>/trackeval_official")
    args = ap.parse_args()
    result_dir = Path(args.result_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else result_dir / "trackeval_official"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "audit.log"
    payload_path = result_dir / "results.json"
    payload = read_json(payload_path) if payload_path.exists() else {}
    sequences = [x for x in args.sequences.split(",") if x] or infer_sequences(payload, args.dataset, result_dir)
    methods = [x for x in args.methods.split(",") if x] or infer_methods(payload, args.dataset, result_dir)
    if not sequences or not methods:
        raise RuntimeError("could not infer sequences/methods; pass --sequences/--methods explicitly")
    metadata_max = int(payload.get("max_frames", 0) or 0)
    max_frames = int(args.max_frames or metadata_max or 0)
    if max_frames <= 0:
        max_frames = 10 ** 9
    failures: List[Dict[str, object]] = []
    base: Dict[str, object] = {
        "schema": "trackeval-real-audit-v1", "status": "failed", "result_dir": str(result_dir),
        "dataset": args.dataset, "sequences": sequences, "methods": methods, "max_frames": max_frames,
        "command": " ".join(sys.argv), "cwd": os.getcwd(), "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__,
        "trackeval_root": str(Path(args.trackeval_root).resolve()),
        "trackeval_git_head": git_head(Path(args.trackeval_root)),
        "source_results_sha256": hash_text(payload_path),
        "staging": None, "official_rows": [], "combined": {}, "failures": failures,
        "metric_definition": "Official vendored TrackEval HOTA/CLEAR/Identity on a frame-truncated MOTChallenge staging copy; no proxy aliases.",
    }
    try:
        gt_root, tracker_root, provenance = stage_dataset(result_dir, out_dir, args.dataset, sequences, methods, args, max_frames)
        base["staging"] = provenance
        official = run_official(gt_root, tracker_root, out_dir, sequences, methods, provenance["seq_info"], Path(args.trackeval_root).resolve())
        rows, combined = extract_rows(official, sequences, methods)
        base["official_rows"] = rows
        base["combined"] = combined
        base["official_messages"] = official.get("messages")
        base["status"] = "complete"
        print(json.dumps({"status": "complete", "combined": combined}, indent=2, sort_keys=True, default=float))
    except Exception as exc:
        failures.append({"error": repr(exc), "traceback": traceback.format_exc()})
        print("[audit-failure] {}".format(exc), file=sys.stderr)
        traceback.print_exc()
    out_json = out_dir / "audit.json"
    out_json.write_text(json.dumps(base, indent=2, sort_keys=True, default=float) + "\n")
    if base["official_rows"]:
        keys = sorted({k for r in base["official_rows"] for k in r})
        with (out_dir / "official_metrics.csv").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(base["official_rows"])
    # The shell tee captures full evaluator output; this small marker makes
    # the audit directory self-describing even when invoked without tee.
    log_path.write_text("command: {}\nstatus: {}\nsee audit.json\n".format(base["command"], base["status"]))
    if base["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
