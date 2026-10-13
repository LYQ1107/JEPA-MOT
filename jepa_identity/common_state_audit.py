"""CPU common-state identity audit over cached detector features.

This evaluator reuses the state recorded by the predictive-residual tracer:
one history, one frame, and the tracer's top candidate subset are exposed to
every control.  It never uses target labels to construct a query or a score;
labels already attached to the offline case file are consumed only after all
scores have been produced.  The state is therefore common across controls but
the candidate subset remains the tracer's recorded subset rather than every
post-gate detector candidate.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jepa_cmot.history import pack_history
from jepa_cmot.models import HistoryPredictor, PredictorConfig

from .identity_metrics import bootstrap_interval, wilson_interval


COCO6_TO_NATIVE8 = (0, 2, 3, 4, 6, 7)
EVALUABLE = {
    "corrected_local_identity",
    "harmful_local_identity",
    "both_correct",
    "neither_correct",
}
CONTROL_NAMES = (
    "geometry_iou",
    "history_last",
    "history_mean4",
    "history_ema",
    "native_recorded",
    "mlp_ce",
    "mlp_jepa_predictive",
    "mlp_jepa_ranking",
    "transformer_ce",
    "transformer_jepa_predictive",
    "transformer_jepa_ranking",
    "feature_shuffle",
    "feature_random",
    "constant_affine",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xyxy_iou(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, left)
    bx1, by1, bx2, by2 = map(float, right)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def unit(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def load_predictor(path: Path) -> tuple[HistoryPredictor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = HistoryPredictor(PredictorConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def track_births(path: Path) -> dict[int, int]:
    births: dict[int, int] = {}
    if not path.exists():
        return births
    for line in path.read_text().splitlines():
        fields = line.split(",")
        if len(fields) < 2:
            continue
        try:
            frame, track_id = int(float(fields[0])), int(float(fields[1]))
        except ValueError:
            continue
        births[track_id] = min(frame, births.get(track_id, frame))
    return births


class VideoFeatureIndex:
    """Frame/detection/feature lookup aligned with formal val.json and npz."""

    def __init__(self, entry: dict[str, Any]):
        self.entry = entry
        self.prepared = Path(entry["prepared"])
        annotation_path = self.prepared / "annotations" / "val.json"
        self.annotation_sha256 = sha256_file(annotation_path)
        raw = json.loads(annotation_path.read_text())
        self.images = {int(row["frame_id"]): row for row in raw["images"]}
        with np.load(entry["features"], allow_pickle=False) as cache:
            self.features = np.asarray(cache["features"], dtype=np.float32)
            self.offsets = np.asarray(cache["offsets"], dtype=np.int64)
            self.image_ids = np.asarray(cache["image_ids"], dtype=np.int64)
        self.image_positions = {int(image_id): index for index, image_id in enumerate(self.image_ids)}
        self.frame_rows: dict[int, list[dict[str, Any]]] = {}
        self.image_rows: dict[int, list[dict[str, Any]]] = {}
        for frame, image in self.images.items():
            position = self.image_positions[int(image["id"])]
            start, stop = int(self.offsets[position]), int(self.offsets[position + 1])
            rows = []
            for index, detection in enumerate(image.get("detection", [])):
                if len(detection) < 6 or start + index >= stop:
                    continue
                category = int(float(detection[5]))
                native_class = COCO6_TO_NATIVE8[category] if 0 <= category < len(COCO6_TO_NATIVE8) else category
                rows.append({
                    "index": index,
                    "box": np.asarray(detection[:4], dtype=np.float32),
                    "native_class": native_class,
                    "feature": self.features[start + index],
                    "image_id": int(image["id"]),
                })
            self.frame_rows[int(frame)] = rows
            self.image_rows[int(image["id"])] = rows
        self.random_cache: dict[int, np.ndarray] = {}
        self.shuffle_cache: dict[int, np.ndarray] = {}

    def match(self, frame: int, box: list[float], native_class: int) -> dict[str, Any] | None:
        candidates = self.frame_rows.get(int(frame), [])
        if not candidates:
            return None
        scored = sorted(
            ((xyxy_iou(box, row["box"]), row) for row in candidates),
            key=lambda item: (-item[0], abs(int(item[1]["native_class"]) - int(native_class)), int(item[1]["index"])),
        )
        score, row = scored[0]
        if score < 0.90:
            return None
        return dict(row, match_iou=float(score))

    def by_detection_index(self, frame: int, detection_index: int, box: list[float]) -> dict[str, Any] | None:
        for row in self.frame_rows.get(int(frame), []):
            if int(row["index"]) == int(detection_index) and xyxy_iou(box, row["box"]) >= 0.90:
                return dict(row, match_iou=float(xyxy_iou(box, row["box"])))
        return None

    def transformed_feature(self, row: dict[str, Any], control: str) -> np.ndarray:
        if control not in {"feature_shuffle", "feature_random"}:
            return np.asarray(row["feature"], dtype=np.float32)
        image_id = int(row["image_id"])
        rows = self.image_rows.get(image_id, [])
        if image_id not in self.random_cache:
            random_rng = np.random.RandomState(17 + image_id)
            shuffle_rng = np.random.RandomState(17 + image_id)
            random_features = random_rng.normal(size=(len(rows), self.features.shape[1])).astype(np.float32)
            self.random_cache[image_id] = unit(random_features)
            permutation = shuffle_rng.permutation(len(rows)) if rows else np.asarray([], dtype=np.int64)
            self.shuffle_cache[image_id] = permutation
        index = int(row["index"])
        if control == "feature_random":
            transformed = self.random_cache[image_id][index]
        else:
            permutation = self.shuffle_cache[image_id]
            transformed = rows[int(permutation[index])]["feature"] if len(permutation) else row["feature"]
        return np.asarray(transformed, dtype=np.float32)


def cosine_cost(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query = unit(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
    candidates = unit(np.asarray(candidates, dtype=np.float32))
    return 1.0 - candidates @ query


def ema_query(features: list[np.ndarray], beta: float = 0.25) -> np.ndarray:
    value = None
    for feature in features:
        feature = unit(feature.reshape(1, -1))[0]
        value = feature if value is None else unit((beta * feature + (1.0 - beta) * value).reshape(1, -1))[0]
    return unit(value.reshape(1, -1))[0] if value is not None else value


def model_query(model: HistoryPredictor, observations: list[dict[str, Any]], current_frame: int,
                width: int, height: int) -> np.ndarray:
    features, metadata, mask = pack_history(
        observations, current_frame, model.config.history, model.config.feature_dim, width, height
    )
    with torch.no_grad():
        query = model(
            torch.from_numpy(features[None]),
            torch.from_numpy(metadata[None]),
            torch.from_numpy(mask[None]),
        )[0].numpy()
    return np.asarray(query, dtype=np.float32)


def score_event(case: dict[str, Any], index: VideoFeatureIndex, births: dict[int, int],
                models: dict[str, HistoryPredictor]) -> tuple[dict[str, list[float]], dict[str, Any]] | None:
    history = case.get("history", [])
    if not history:
        return None
    history_rows = []
    for row in history:
        match = index.match(int(row["frame"]), row["box"], int(case.get("native_class", 0)))
        if match is None:
            return None
        history_rows.append((row, match))
    candidate_rows = []
    for candidate in case.get("candidate_gt_ids", []):
        # Tracer indices are local to the post-score-threshold detector list;
        # they are not raw val.json indices.  Geometry is the stable join key.
        match = index.match(int(case["frame"]), candidate["box"], int(candidate.get("native_class", case.get("native_class", 0))))
        if match is None:
            return None
        candidate_rows.append((candidate, match))
    if not candidate_rows:
        return None
    image = index.images[int(case["frame"])]
    width, height = int(image["width"]), int(image["height"])
    birth = int(births.get(int(case["track_id"]), min(int(row["frame"]) for row in history)))

    def observations_for(control: str) -> list[dict[str, Any]]:
        observations = []
        for row, match in history_rows:
            observations.append({
                "feature": index.transformed_feature(match, control),
                "box": np.asarray(row["box"], dtype=np.float32),
                "frame": int(row["frame"]),
                "birth": birth,
                "confidence": float(row.get("confidence", 1.0)),
            })
        return observations

    original_history = observations_for("original")
    last = unit(original_history[-1]["feature"].reshape(1, -1))[0]
    mean4 = unit(np.sum([row["feature"] for row in original_history[-4:]], axis=0).reshape(1, -1))[0]
    ema = ema_query([row["feature"] for row in original_history])
    random_history = observations_for("feature_random")
    shuffle_history = observations_for("feature_shuffle")
    candidate_features = np.asarray([match["feature"] for _, match in candidate_rows], dtype=np.float32)
    random_candidates = np.asarray([index.transformed_feature(match, "feature_random") for _, match in candidate_rows], dtype=np.float32)
    shuffle_candidates = np.asarray([index.transformed_feature(match, "feature_shuffle") for _, match in candidate_rows], dtype=np.float32)
    last_box = np.asarray(history[-1]["box"], dtype=np.float32)
    candidate_boxes = [np.asarray(candidate["box"], dtype=np.float32) for candidate, _ in candidate_rows]

    scores: dict[str, list[float]] = {
        "geometry_iou": [1.0 - xyxy_iou(last_box, box) for box in candidate_boxes],
        "history_last": cosine_cost(last, candidate_features).tolist(),
        "history_mean4": cosine_cost(mean4, candidate_features).tolist(),
        "history_ema": cosine_cost(ema, candidate_features).tolist(),
        "native_recorded": [float(candidate.get("native_cost", 0.0)) for candidate, _ in candidate_rows],
        "feature_shuffle": cosine_cost(unit(shuffle_history[-1]["feature"].reshape(1, -1))[0], shuffle_candidates).tolist(),
        "feature_random": cosine_cost(unit(random_history[-1]["feature"].reshape(1, -1))[0], random_candidates).tolist(),
        "constant_affine": [0.0 for _ in candidate_rows],
    }
    for name, model in models.items():
        control = "feature_random" if name.endswith("_random") else "original"
        query = model_query(model, observations_for(control), int(case["frame"]), width, height)
        scores[name] = cosine_cost(query, candidate_features if control == "original" else random_candidates).tolist()
    metadata = {
        "history_rows": len(history_rows),
        "candidate_rows": len(candidate_rows),
        "min_history_match_iou": min(float(match["match_iou"]) for _, match in history_rows),
        "min_candidate_match_iou": min(float(match["match_iou"]) for _, match in candidate_rows),
        "birth_source": "residual_tracks_txt" if int(case["track_id"]) in births else "first_history_frame_fallback",
    }
    return scores, metadata


def event_metric(case: dict[str, Any], control: str) -> dict[str, Any]:
    expected = case.get("expected_gt_id")
    candidates = case.get("candidate_gt_ids", [])
    scores = case["scores"][control]
    ordered = sorted(range(len(candidates)), key=lambda index: (float(scores[index]), int(candidates[index].get("index", index))))
    positive = [position for position, index in enumerate(ordered) if candidates[index].get("gt_id") == expected]
    if not positive:
        return {"candidate_present": False, "top1": None, "rank": None, "mrr": None,
                "recall_at": {"1": False, "3": False, "5": False}, "pairwise_auc": None,
                "average_precision": None, "margin": None, "nll": None, "brier": None,
                "winner_confidence": None, "winner_correct": None, "winner_gt_id": None}
    rank = positive[0] + 1
    positive_scores = [float(scores[index]) for index in ordered if candidates[index].get("gt_id") == expected]
    negative_scores = [float(scores[index]) for index in ordered if candidates[index].get("gt_id") != expected]
    pairwise = [1.0 if p < n else 0.5 if p == n else 0.0 for p in positive_scores for n in negative_scores]
    precision = []
    seen = 0
    for position, index in enumerate(ordered, start=1):
        if candidates[index].get("gt_id") == expected:
            seen += 1
            precision.append(seen / position)
    margin = None if not negative_scores else min(negative_scores) - min(positive_scores)
    shifted = np.asarray([-float(scores[index]) for index in ordered], dtype=np.float64)
    shifted -= shifted.max()
    probabilities = np.exp(np.clip(shifted, -80, 0))
    probabilities /= max(probabilities.sum(), 1e-12)
    positive_probability = float(sum(probability for probability, index in zip(probabilities, ordered)
                                     if candidates[index].get("gt_id") == expected))
    winner_correct = candidates[ordered[0]].get("gt_id") == expected
    return {
        "candidate_present": True,
        "top1": bool(winner_correct),
        "rank": rank,
        "mrr": 1.0 / rank,
        "recall_at": {str(k): rank <= k for k in (1, 3, 5)},
        "pairwise_auc": sum(pairwise) / len(pairwise) if pairwise else None,
        "average_precision": sum(precision) / len(precision) if precision else None,
        "margin": margin,
        "nll": -math.log(max(1e-12, positive_probability)),
        "brier": (positive_probability - 1.0) ** 2,
        "winner_confidence": float(probabilities[0]),
        "winner_correct": bool(winner_correct),
        "winner_gt_id": candidates[ordered[0]].get("gt_id"),
    }


def summarize(cases: list[dict[str, Any]], control: str, seed: int = 17) -> dict[str, Any]:
    results = [event_metric(case, control) for case in cases]
    eligible = [result for result in results if result["candidate_present"]]
    coverage = len(eligible)

    def values(key: str) -> list[float]:
        return [float(result[key]) for result in eligible if result.get(key) is not None]

    top1 = sum(bool(result["top1"]) for result in eligible)
    corrected = sum(
        bool(case.get("native_gt_id") != case.get("expected_gt_id") and result.get("winner_gt_id") == case.get("expected_gt_id"))
        for case, result in zip(cases, results) if result["candidate_present"]
    )
    harmful = sum(
        bool(case.get("native_gt_id") == case.get("expected_gt_id") and result.get("winner_gt_id") != case.get("expected_gt_id"))
        for case, result in zip(cases, results) if result["candidate_present"]
    )
    output = {
        "evaluable_common_state_events": len(results),
        "correct_candidate_present": coverage,
        "correct_candidate_absent": len(results) - coverage,
        "candidate_coverage_rate": coverage / len(results) if results else None,
        "candidate_coverage_95ci": wilson_interval(coverage, len(results)),
        "top1_identity_accuracy_conditional": top1 / coverage if coverage else None,
        "top1_identity_accuracy_95ci": wilson_interval(top1, coverage),
        "local_corrected_candidate_present": corrected,
        "local_harmful_candidate_present": harmful,
        "local_net_correction_candidate_present": corrected - harmful,
        "local_corrected_harmful_ratio_candidate_present": corrected / harmful if harmful else None,
        "mrr_conditional": sum(values("mrr")) / len(values("mrr")) if values("mrr") else None,
        "mrr_95ci": bootstrap_interval(values("mrr"), seed + 1),
        "recall_at_k_conditional": {
            str(k): sum(bool(result["recall_at"][str(k)]) for result in eligible) / coverage if coverage else None
            for k in (1, 3, 5)
        },
        "pairwise_auc": sum(values("pairwise_auc")) / len(values("pairwise_auc")) if values("pairwise_auc") else None,
        "pairwise_auc_95ci": bootstrap_interval(values("pairwise_auc"), seed + 2),
        "average_precision": sum(values("average_precision")) / len(values("average_precision")) if values("average_precision") else None,
        "average_precision_95ci": bootstrap_interval(values("average_precision"), seed + 3),
        "positive_vs_best_negative_margin": sum(values("margin")) / len(values("margin")) if values("margin") else None,
        "positive_vs_best_negative_margin_95ci": bootstrap_interval(values("margin"), seed + 4),
        "candidate_nll_uncalibrated": sum(values("nll")) / len(values("nll")) if values("nll") else None,
        "candidate_brier_uncalibrated": sum(values("brier")) / len(values("brier")) if values("brier") else None,
        "score_definition": "lower cost is better; all controls use one recorded history and one tracer candidate subset",
        "probability_warning": "NLL/Brier use softmax(-cost) without fitting a probability calibration on this audit",
    }
    ece = 0.0
    bins = [[] for _ in range(10)]
    for result in eligible:
        bins[min(9, int(float(result["winner_confidence"]) * 10))].append(result)
    for bucket in bins:
        if bucket:
            confidence = sum(float(result["winner_confidence"]) for result in bucket) / len(bucket)
            accuracy = sum(bool(result["winner_correct"]) for result in bucket) / len(bucket)
            ece += len(bucket) / coverage * abs(confidence - accuracy)
    output["winner_ece_uncalibrated"] = ece if coverage else None
    return output


def run_audit(cases_path: Path, inputs_dir: Path, residual_root: Path, out_cases: Path,
              max_events: int | None = None) -> dict[str, Any]:
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    cases = [case for case in cases if case.get("method") == "predictive_residual_audit" and case.get("diagnosis") in EVALUABLE]
    if max_events is not None:
        cases = cases[:max_events]
    entries = {}
    for path in sorted(inputs_dir.glob("*.json")):
        rows = json.loads(path.read_text())
        if rows:
            entries[rows[0]["video"]] = rows[0]
    models_paths = {
        "mlp_ce": Path("runs/JEPA_CMOT/source_history_v1/bdd_dense_current/seed17/discriminative/final.pt"),
        "mlp_jepa_predictive": Path("runs/JEPA_CMOT/source_history_v1/bdd_dense_current/seed17/predictive/final.pt"),
        "mlp_jepa_ranking": Path("runs/JEPA_CMOT/source_history_v1/bdd_dense_current/seed17/predictive_ranking/final.pt"),
        "transformer_ce": Path("runs/JEPA_CMOT/transformer_v1/models/bdd100k/seed17/discriminative/final.pt"),
        "transformer_jepa_predictive": Path("runs/JEPA_CMOT/transformer_v1/models/bdd100k/seed17/predictive/final.pt"),
        "transformer_jepa_ranking": Path("runs/JEPA_CMOT/transformer_v1/models/bdd100k/seed17/predictive_ranking/final.pt"),
    }
    models = {}
    for name, path in models_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        models[name], _ = load_predictor(path)
    indices: dict[str, VideoFeatureIndex] = {}
    births: dict[str, dict[int, int]] = {}
    scored_cases = []
    skip_reasons = Counter()
    mapping_iou = []
    for case in cases:
        video = str(case["video"])
        if video not in entries:
            skip_reasons["missing_input_entry"] += 1
            continue
        if video not in indices:
            indices[video] = VideoFeatureIndex(entries[video])
            births[video] = track_births(residual_root / "videos" / video / "predictive_residual_audit" / video / "tracks.txt")
        try:
            result = score_event(case, indices[video], births[video], models)
        except (KeyError, IndexError, ValueError, StopIteration):
            result = None
        if result is None:
            skip_reasons["feature_box_mapping_failed"] += 1
            continue
        scores, metadata = result
        enriched = dict(case)
        enriched["scores"] = scores
        enriched["common_state_metadata"] = metadata
        scored_cases.append(enriched)
        mapping_iou.append(float(metadata["min_candidate_match_iou"]))
    out_cases.parent.mkdir(parents=True, exist_ok=True)
    with out_cases.open("w") as stream:
        for case in scored_cases:
            stream.write(json.dumps(case, separators=(",", ":"), ensure_ascii=False) + "\n")
    controls = {control: summarize(scored_cases, control, 17 + index * 101)
                for index, control in enumerate(CONTROL_NAMES)}
    by_domain = {}
    for domain in sorted({case["domain"] for case in scored_cases}):
        subset = [case for case in scored_cases if case["domain"] == domain]
        by_domain[domain] = {control: summarize(subset, control, 1000 + index * 101)
                             for index, control in enumerate(CONTROL_NAMES)}
    gap_bins = {
        "1": lambda gap: gap == 1,
        "2-4": lambda gap: 2 <= gap <= 4,
        "5-8": lambda gap: 5 <= gap <= 8,
        ">8": lambda gap: gap > 8,
        "unknown": lambda gap: gap is None,
    }
    by_gap = {}
    for label, predicate in gap_bins.items():
        subset = [case for case in scored_cases if predicate(case.get("last_observation_gap"))]
        if subset:
            by_gap[label] = {control: summarize(subset, control, 2000 + index * 101)
                             for index, control in enumerate(CONTROL_NAMES)}
    return {
        "status": "COMPLETE_COMMON_STATE_CPU_IDENTITY_AUDIT",
        "scope": "evaluable predictive-residual events only; one recorded history and one tracer top-candidate subset shared by every control",
        "source_cases": str(cases_path),
        "source_case_count": len(cases),
        "scored_case_count": len(scored_cases),
        "skipped_cases": dict(skip_reasons),
        "videos": sorted({case["video"] for case in scored_cases}),
        "domains": sorted({case["domain"] for case in scored_cases}),
        "controls": controls,
        "by_domain": by_domain,
        "by_observation_gap": by_gap,
        "model_checkpoints": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in models_paths.items()},
        "feature_source": "formal_detection_streams_v1 dense_current features; target GT is not in val.json",
        "mean_min_candidate_match_iou": sum(mapping_iou) / len(mapping_iou) if mapping_iou else None,
        "target_GT_used_for_training_or_gate": False,
        "offline_GT_used_for_diagnosis": True,
        "sealed_used": False,
        "limitations": [
            "common state is the predictive-residual rollout state, not a newly replayed independent baseline state",
            "candidate set is the tracer top subset (normally six native-ranked rows), not every detector output",
            "geometry_iou is a transparent last-box IoU control, not the full GeneralTrack motion model",
            "birth age is read from the residual track file when available and otherwise falls back to the first recorded history frame",
            "constant, NLL, Brier, and ECE are scale/calibration diagnostics; NLL/Brier/ECE use uncalibrated softmax(-cost)",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--residual-root", type=Path, required=True)
    parser.add_argument("--out-cases", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    report = run_audit(args.cases, args.inputs_dir, args.residual_root, args.out_cases, args.max_events)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("COMMON_STATE_CPU_AUDIT_COMPLETE", report["scored_case_count"], flush=True)


if __name__ == "__main__":
    main()
