"""CPU audit of recorded one-step association changes.

The audit intentionally separates two scopes:

* detector-level GT coverage over every development frame, and
* identity diagnoses for the assignment changes that the existing causal
  tracker recorded.

The latter is a common-state local counterfactual on one rollout.  It is not
an independent baseline rollout and it is not a claim about all association
opportunities.  Target labels are read only by this offline audit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


BDD_COCO6_TO_NATIVE8 = (0, 2, 3, 4, 6, 7)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_to_xyxy(box: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, w, h = map(float, box)
    return x, y, x + w, y + h


def iou_xyxy(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, left)
    bx1, by1, bx2, by2 = map(float, right)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def native_group(native_class: int) -> int:
    native_class = int(native_class)
    return 0 if native_class < 2 else 1 if native_class < 6 else 2


def bdd_category_group(category_id: int) -> int:
    category_id = int(category_id)
    if 1 <= category_id <= len(BDD_COCO6_TO_NATIVE8):
        return native_group(BDD_COCO6_TO_NATIVE8[category_id - 1])
    return native_group(category_id)


def load_labels(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    images = {int(row["id"]): row for row in raw["images"]}
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in raw.get("annotations", []):
        if annotation.get("iscrowd"):
            continue
        image = images[int(annotation["image_id"])]
        by_frame[int(image["frame_id"])].append(annotation)
    return {
        "raw": raw,
        "images": images,
        "by_frame": by_frame,
        "known_frames": set(map(int, raw.get("evaluation_frame_ids", []))),
    }


def gt_identity(labels: dict[str, Any], frame: int, box: Iterable[float], native_class: int) -> tuple[int | None, float]:
    if int(frame) not in labels["known_frames"]:
        return None, 0.0
    best_id, best_iou = None, 0.0
    target_group = native_group(native_class)
    for annotation in labels["by_frame"].get(int(frame), []):
        if bdd_category_group(annotation["category_id"]) != target_group:
            continue
        score = iou_xyxy(box, xywh_to_xyxy(annotation["bbox"]))
        if score > best_iou:
            best_id, best_iou = int(annotation["instance_id"]), score
    return (best_id, best_iou) if best_iou >= 0.5 else (None, best_iou)


def dominant_history(labels: dict[str, Any], event: dict[str, Any]) -> tuple[int | None, float, str]:
    matched = []
    for row in event.get("history", []):
        identity, _ = gt_identity(labels, int(row["frame"]), row["box"], int(event["native_class"]))
        if identity is not None:
            matched.append(identity)
    if not matched or int(event["frame"]) not in labels["known_frames"]:
        return None, 0.0, "unscored_history_or_current"
    counts = Counter(matched)
    expected, frequency = counts.most_common(1)[0]
    purity = frequency / len(matched)
    if purity < 0.75:
        return int(expected), purity, "mixed_identity_history"
    return int(expected), purity, "evaluable"


def choice_identity(labels: dict[str, Any], frame: int, choice: dict[str, Any] | None) -> tuple[int | None, float]:
    if choice is None:
        return None, 0.0
    return gt_identity(labels, frame, choice["box"], int(choice["native_class"]))


def diagnose_event(labels: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    expected, purity, status = dominant_history(labels, event)
    frame = int(event["frame"])
    native_id, native_iou = choice_identity(labels, frame, event.get("native_choice"))
    jepa_id, jepa_iou = choice_identity(labels, frame, event.get("jepa_choice"))
    candidate_ids = []
    for candidate in event.get("candidates", []):
        candidate_id, candidate_iou = choice_identity(labels, frame, candidate)
        candidate_ids.append({
            "index": int(candidate["index"]),
            "gt_id": candidate_id,
            "iou": candidate_iou,
            "native_cost": candidate.get("native_cost"),
            "fused_cost": candidate.get("fused_cost"),
            "semantic": candidate.get("semantic"),
            "detection_score": candidate.get("score"),
        })
    if status != "evaluable":
        diagnosis = status
    elif native_id != expected and jepa_id == expected:
        diagnosis = "corrected_local_identity"
    elif native_id == expected and jepa_id != expected:
        diagnosis = "harmful_local_identity"
    elif native_id == expected and jepa_id == expected:
        diagnosis = "both_correct"
    else:
        diagnosis = "neither_correct"
    history_frames = [int(row["frame"]) for row in event.get("history", [])]
    return {
        "video": event.get("video"),
        "frame": frame,
        "branch": event.get("branch"),
        "track_id": int(event.get("track_id", -1)),
        "diagnosis": diagnosis,
        "history_purity": purity,
        "expected_gt_id": expected,
        "native_gt_id": native_id,
        "jepa_gt_id": jepa_id,
        "native_iou": native_iou,
        "jepa_iou": jepa_iou,
        "candidate_gt_ids": candidate_ids,
        "correct_candidate_present": expected is not None and any(row["gt_id"] == expected for row in candidate_ids),
        "candidate_count": len(candidate_ids),
        "last_observation_gap": frame - max(history_frames) if history_frames else None,
    }


def detector_coverage(prepared_annotations: Path, labels: dict[str, Any]) -> dict[str, Any]:
    prepared = json.loads(prepared_annotations.read_text())
    image_by_frame = {int(row["frame_id"]): row for row in prepared["images"]}
    totals = Counter()
    by_group: dict[str, Counter] = defaultdict(Counter)
    for frame, annotations in labels["by_frame"].items():
        image = image_by_frame.get(int(frame))
        if image is None:
            continue
        detections = image.get("detection", [])
        for annotation in annotations:
            totals["gt_objects"] += 1
            gt_box = xywh_to_xyxy(annotation["bbox"])
            group = str(bdd_category_group(annotation["category_id"]))
            best = 0.0
            for detection in detections:
                if len(detection) < 6:
                    continue
                native = BDD_COCO6_TO_NATIVE8[int(float(detection[5]))]
                if native_group(native) != int(group):
                    continue
                best = max(best, iou_xyxy(detection[:4], gt_box))
            if best >= 0.5:
                totals["gt_with_iou05_detection"] += 1
                by_group[group]["gt_with_iou05_detection"] += 1
            else:
                totals["gt_without_iou05_detection"] += 1
                by_group[group]["gt_without_iou05_detection"] += 1
            by_group[group]["gt_objects"] += 1
    denominator = totals["gt_objects"]
    return {
        "gt_objects": totals["gt_objects"],
        "gt_with_iou05_detection": totals["gt_with_iou05_detection"],
        "gt_without_iou05_detection": totals["gt_without_iou05_detection"],
        "coverage_rate": totals["gt_with_iou05_detection"] / denominator if denominator else None,
        "by_native_group": {key: dict(value) for key, value in sorted(by_group.items())},
        "definition": "at least one same-native-group detector box with IoU >= 0.5; this is detector presence, not post-matcher candidate eligibility",
    }


def counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def run_audit(root: Path, manifest_path: Path, cases_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    manifest_videos = {
        row["name"]: row
        for row in manifest["videos"]
        if row.get("dataset") == "bdd100k" and row.get("split") == "development"
    }
    inputs = {}
    for path in sorted((root / "inputs").glob("*.json")):
        entries = json.loads(path.read_text())
        if entries:
            inputs[entries[0]["video"]] = entries[0]
    video_names = sorted(set(manifest_videos) & set(inputs))
    methods = [row["name"] for row in protocol["controls"]]
    method_counts = {method: Counter() for method in methods}
    method_domains = {method: defaultdict(Counter) for method in methods}
    method_gaps = {method: defaultdict(Counter) for method in methods}
    cases = []
    coverage_by_domain: dict[str, dict[str, Any]] = {}
    missing_event_logs = Counter()
    for video in video_names:
        entry = inputs[video]
        manifest_row = manifest_videos[video]
        labels_path = Path(manifest_row["labels"])
        labels = load_labels(labels_path)
        prepared_annotations = Path(entry["prepared"]) / "annotations" / "val.json"
        domain = manifest_row["domain"]
        if domain not in coverage_by_domain:
            coverage_by_domain[domain] = detector_coverage(prepared_annotations, labels)
        for method in methods:
            event_path = root / "videos" / video / method / video / "association_changes.jsonl"
            if not event_path.exists():
                missing_event_logs[method] += 1
                events = []
            else:
                events = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
            method_counts[method]["videos_total"] += 1
            method_counts[method]["event_logs_present"] += int(event_path.exists())
            method_counts[method]["recorded_changes"] += len(events)
            for event in events:
                event["video"] = video
                diagnosis = diagnose_event(labels, event)
                method_counts[method][diagnosis["diagnosis"]] += 1
                method_domains[method][domain][diagnosis["diagnosis"]] += 1
                gap = diagnosis["last_observation_gap"]
                method_gaps[method][str(gap) if gap is not None else "unknown"][diagnosis["diagnosis"]] += 1
                if diagnosis["diagnosis"] in {"unscored_history_or_current", "mixed_identity_history"}:
                    method_counts[method]["non_evaluable_events"] += 1
                else:
                    method_counts[method]["evaluable_events"] += 1
                    method_counts[method]["correct_candidate_present"] += int(diagnosis["correct_candidate_present"])
                    method_counts[method]["correct_candidate_absent"] += int(not diagnosis["correct_candidate_present"])
                if len(cases) < 5000:
                    cases.append({"method": method, "domain": domain, **diagnosis})
    taxonomy = {
        "status": "COMPLETE_OFFLINE_H1_EVENT_AUDIT",
        "dataset": "bdd100k",
        "split": "development",
        "seed": 17,
        "scope": "existing development_rank_reward_v1 causal one-step assignment-change logs plus detector-level coverage",
        "protocol_sha256": sha256_file(protocol_path),
        "manifest_sha256": sha256_file(manifest_path),
        "methods": methods,
        "videos": video_names,
        "video_count": len(video_names),
        "coverage_by_domain": coverage_by_domain,
        "missing_event_logs_by_method": counter_to_dict(missing_event_logs),
        "method_counts": {method: counter_to_dict(counter) for method, counter in method_counts.items()},
        "method_domain_diagnosis": {
            method: {domain: counter_to_dict(counter) for domain, counter in sorted(domains.items())}
            for method, domains in method_domains.items()
        },
        "method_gap_diagnosis": {
            method: {gap: counter_to_dict(counter) for gap, counter in sorted(gaps.items())}
            for method, gaps in method_gaps.items()
        },
        "case_sample_count": len(cases),
        "case_sample_path": str(cases_path),
        "target_GT_used_for_training_or_gate": False,
        "sealed_used": False,
        "limitations": [
            "association diagnoses cover only changed assignments recorded by the existing tracer; absent event files are not evidence of zero opportunities",
            "the common state is the method's own rollout state, so this is not an independent full-baseline rollout",
            "candidate lists are the tracer's recorded top candidates and do not expose every detector or post-matcher gate",
            "this Phase-1 run reuses BDD development evidence; UA event-level logs require a GPU-backed rerun and are not imputed",
        ],
    }
    headroom_methods = {}
    for method, counter in method_counts.items():
        evaluable = counter["evaluable_events"]
        corrected = counter["corrected_local_identity"]
        harmful = counter["harmful_local_identity"]
        headroom_methods[method] = {
            "recorded_changed_events": counter["recorded_changes"],
            "evaluable_changed_events": evaluable,
            "correct_candidate_present": counter["correct_candidate_present"],
            "correct_candidate_absent": counter["correct_candidate_absent"],
            "local_corrected_events": corrected,
            "local_harmful_events": harmful,
            "corrected_minus_harmful": corrected - harmful,
            "corrected_harmful_ratio": corrected / harmful if harmful else None,
            "observed_event_headroom_rate": corrected / evaluable if evaluable else None,
            "interpretation": "lower-bound local headroom among recorded changed assignments; not all candidate opportunities",
        }
    headroom = {
        "status": "COMPLETE_OBSERVED_LOCAL_HEADROOM_NOT_GLOBAL_UPPER_BOUND",
        "dataset": "bdd100k",
        "split": "development",
        "seed": 17,
        "taxonomy_reference": str(cases_path.with_name("IDENTITY_ERROR_TAXONOMY.json")),
        "methods": headroom_methods,
        "strong_control_comparisons": [
            "predictive_rank_reward vs frozen_last_rank_reward",
            "predictive_rank_reward vs discriminative_rank_reward",
            "predictive_rank_reward vs random_rank_reward",
            "predictive_residual_audit vs predictive_rank_reward",
        ],
        "candidate_coverage_reference": "IDENTITY_ERROR_TAXONOMY.json::coverage_by_domain",
        "target_GT_used_for_training_or_gate": False,
        "sealed_used": False,
        "limitations": taxonomy["limitations"],
    }
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_path.open("w") as stream:
        for row in cases:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return taxonomy, headroom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-taxonomy", type=Path, required=True)
    parser.add_argument("--out-headroom", type=Path, required=True)
    parser.add_argument("--out-cases", type=Path, required=True)
    args = parser.parse_args()
    taxonomy, headroom = run_audit(args.root, args.manifest, args.out_cases)
    args.out_taxonomy.parent.mkdir(parents=True, exist_ok=True)
    args.out_taxonomy.write_text(json.dumps(taxonomy, indent=2, ensure_ascii=False) + "\n")
    args.out_headroom.parent.mkdir(parents=True, exist_ok=True)
    args.out_headroom.write_text(json.dumps(headroom, indent=2, ensure_ascii=False) + "\n")
    print("IDENTITY_EVENT_AUDIT_COMPLETE", taxonomy["video_count"], taxonomy["methods"], flush=True)


if __name__ == "__main__":
    main()
