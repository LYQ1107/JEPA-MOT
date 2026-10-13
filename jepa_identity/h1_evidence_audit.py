"""Machine-check the completed seed17 H1 evidence bundle.

This is a consistency gate over compact reports already produced by the H1
audits.  It does not train, read raw labels, rerun a tracker, or turn a
development result into a general scientific claim.  A passing audit means
the registered static evidence is internally complete; the runtime and
generalization limits remain explicit in the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COMMON_CONTROLS = (
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
PAIR_CONTROLS = (
    "history_last",
    "mlp_ce",
    "mlp_jepa_predictive",
    "mlp_jepa_ranking",
    "transformer_ce",
    "transformer_jepa_predictive",
    "transformer_jepa_ranking",
)
COMMON_METRICS = (
    "top1_identity_accuracy_conditional",
    "mrr_conditional",
    "pairwise_auc",
    "local_net_correction_candidate_present",
)


def _read(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / name).read_text())


def _check(checks: list[dict[str, Any]], name: str, passed: bool,
           observed: Any, expected: Any) -> None:
    checks.append({
        "name": name,
        "pass": bool(passed),
        "observed": observed,
        "expected": expected,
    })


def audit(root: Path) -> dict[str, Any]:
    common = _read(root, "COMMON_STATE_IDENTITY_METRICS.json")
    paired = _read(root, "PAIRED_COMMON_STATE_COMPARISONS_SEED17.json")
    ua_paired = _read(root, "UA_PAIRED_COMMON_STATE_COMPARISONS_SEED17.json")
    bdd_cluster = _read(root, "BDD_VIDEO_CLUSTER_PAIRED_SEED17.json")
    ua_cluster = _read(root, "UA_VIDEO_CLUSTER_PAIRED_SEED17.json")
    stratified = _read(root, "H1_DOMAIN_STRATIFIED_PAIRED_SEED17.json")
    structural = _read(root, "STRUCTURAL_FORMAL_SEED17.json")
    ua = _read(root, "UA_LEGACY_TRACK_REPLAY_SEED17.json")
    taxonomy = _read(root, "IDENTITY_ERROR_TAXONOMY.json")
    headroom = _read(root, "ASSOCIATION_HEADROOM.json")
    ranking = _read(root, "IDENTITY_INFORMATION_METRICS.json")

    checks: list[dict[str, Any]] = []
    _check(checks, "common_state_status", common.get("status") == "COMPLETE_COMMON_STATE_CPU_IDENTITY_AUDIT",
           common.get("status"), "COMPLETE_COMMON_STATE_CPU_IDENTITY_AUDIT")
    _check(checks, "common_state_case_count", common.get("source_case_count") == 1394 and common.get("scored_case_count") == 1394,
           [common.get("source_case_count"), common.get("scored_case_count")], [1394, 1394])
    common_controls = tuple(common.get("controls", {}).keys())
    _check(checks, "required_common_controls", set(COMMON_CONTROLS).issubset(common_controls),
           sorted(common_controls), list(COMMON_CONTROLS))
    coverage = {
        name: {
            "present": common["controls"][name].get("correct_candidate_present"),
            "absent": common["controls"][name].get("correct_candidate_absent"),
            "rate": common["controls"][name].get("candidate_coverage_rate"),
        }
        for name in COMMON_CONTROLS
    }
    _check(checks, "shared_candidate_coverage", len({json.dumps(value, sort_keys=True) for value in coverage.values()}) == 1
           and next(iter(coverage.values()))["present"] == 1177
           and next(iter(coverage.values()))["absent"] == 217,
           coverage, {"present": 1177, "absent": 217, "rate": 1177 / 1394})
    metric_presence = {
        name: sorted(metric for metric in COMMON_METRICS if metric in common["controls"][name])
        for name in COMMON_CONTROLS
    }
    _check(checks, "ranking_and_correction_metrics", all(set(COMMON_METRICS) == set(values) for values in metric_presence.values()),
           metric_presence, list(COMMON_METRICS))
    _check(checks, "common_state_no_training_gt", common.get("target_GT_used_for_training_or_gate") is False
           and common.get("sealed_used") is False,
           [common.get("target_GT_used_for_training_or_gate"), common.get("sealed_used")], [False, False])

    _check(checks, "paired_status", paired.get("status") == "COMPLETE_PAIRED_COMMON_STATE_COMPARISON_SEED17",
           paired.get("status"), "COMPLETE_PAIRED_COMMON_STATE_COMPARISON_SEED17")
    _check(checks, "paired_counts", [paired.get("source_case_count"), paired.get("candidate_present_case_count"),
                                      paired.get("candidate_absent_case_count")] == [1394, 1177, 217],
           [paired.get("source_case_count"), paired.get("candidate_present_case_count"), paired.get("candidate_absent_case_count")],
           [1394, 1177, 217])
    _check(checks, "paired_controls", set(PAIR_CONTROLS).issubset(set(paired.get("controls", []))),
           paired.get("controls", []), list(PAIR_CONTROLS))
    _check(checks, "paired_comparison_count", len(paired.get("comparisons", [])) == 6,
           len(paired.get("comparisons", [])), 6)
    _check(checks, "paired_no_training_gt", paired.get("target_GT_used_for_training_or_gate") is False
           and paired.get("sealed_used") is False,
           [paired.get("target_GT_used_for_training_or_gate"), paired.get("sealed_used")], [False, False])
    _check(checks, "ua_paired_status_and_counts", ua_paired.get("status") == "COMPLETE_PAIRED_COMMON_STATE_COMPARISON_SEED17"
           and [ua_paired.get("source_case_count"), ua_paired.get("candidate_present_case_count"),
                ua_paired.get("candidate_absent_case_count")] == [4199, 4159, 40],
           [ua_paired.get("status"), ua_paired.get("source_case_count"),
            ua_paired.get("candidate_present_case_count"), ua_paired.get("candidate_absent_case_count")],
           ["COMPLETE_PAIRED_COMMON_STATE_COMPARISON_SEED17", 4199, 4159, 40])
    _check(checks, "ua_paired_controls_and_limit", set(PAIR_CONTROLS).issubset(set(ua_paired.get("controls", [])))
           and ua_paired.get("target_GT_used_for_training_or_gate") is False
           and ua_paired.get("sealed_used") is False,
           {"controls": ua_paired.get("controls", []),
            "target_GT_used_for_training_or_gate": ua_paired.get("target_GT_used_for_training_or_gate"),
            "sealed_used": ua_paired.get("sealed_used")},
           "required controls / false / false")
    _check(checks, "bdd_cluster_status_and_counts", bdd_cluster.get("status") == "COMPLETE_VIDEO_CLUSTER_PAIRED_AUDIT_SEED17"
           and bdd_cluster.get("dataset") == "bdd100k"
           and [bdd_cluster.get("source_case_count"), bdd_cluster.get("candidate_present_case_count"), bdd_cluster.get("candidate_absent_case_count"), bdd_cluster.get("video_count")] == [1394, 1177, 217, 19],
           [bdd_cluster.get("status"), bdd_cluster.get("dataset"), bdd_cluster.get("source_case_count"),
            bdd_cluster.get("candidate_present_case_count"), bdd_cluster.get("candidate_absent_case_count"), bdd_cluster.get("video_count")],
           ["COMPLETE_VIDEO_CLUSTER_PAIRED_AUDIT_SEED17", "bdd100k", 1394, 1177, 217, 19])
    _check(checks, "ua_cluster_status_and_counts", ua_cluster.get("status") == "COMPLETE_VIDEO_CLUSTER_PAIRED_AUDIT_SEED17"
           and ua_cluster.get("dataset") == "ua_detrac_legacy_replay"
           and [ua_cluster.get("source_case_count"), ua_cluster.get("candidate_present_case_count"), ua_cluster.get("candidate_absent_case_count"), ua_cluster.get("video_count")] == [4199, 4159, 40, 17],
           [ua_cluster.get("status"), ua_cluster.get("dataset"), ua_cluster.get("source_case_count"),
            ua_cluster.get("candidate_present_case_count"), ua_cluster.get("candidate_absent_case_count"), ua_cluster.get("video_count")],
           ["COMPLETE_VIDEO_CLUSTER_PAIRED_AUDIT_SEED17", "ua_detrac_legacy_replay", 4199, 4159, 40, 17])
    _check(checks, "cluster_audits_no_training_gt", all(
        report.get("target_GT_used_for_training_or_gate") is False and report.get("sealed_used") is False
        and set(PAIR_CONTROLS).issubset(set(report.get("controls", [])))
        for report in (bdd_cluster, ua_cluster)),
        {"bdd": [bdd_cluster.get("target_GT_used_for_training_or_gate"), bdd_cluster.get("sealed_used")],
         "ua": [ua_cluster.get("target_GT_used_for_training_or_gate"), ua_cluster.get("sealed_used")]},
        "required controls / false / false")
    _check(checks, "domain_stratified_status_and_count", stratified.get("status") == "COMPLETE_BDD_DOMAIN_STRATIFIED_PAIRED_COMPARISON_SEED17"
           and stratified.get("source_case_count") == 1394
           and set(stratified.get("domains", {})) == {"clear", "night", "overcast", "rainy"},
           [stratified.get("status"), stratified.get("source_case_count"), sorted(stratified.get("domains", {}))],
           ["COMPLETE_BDD_DOMAIN_STRATIFIED_PAIRED_COMPARISON_SEED17", 1394, ["clear", "night", "overcast", "rainy"]])
    stratified_counts = [value.get("source_case_count", 0) for value in stratified.get("domains", {}).values()]
    _check(checks, "domain_stratified_controls_and_coverage", stratified.get("target_GT_used_for_training_or_gate") is False
           and stratified.get("sealed_used") is False
           and set(PAIR_CONTROLS).issubset(set(stratified.get("controls", [])))
           and sum(stratified_counts) == 1394,
           {"controls": stratified.get("controls", []), "target_GT_used_for_training_or_gate": stratified.get("target_GT_used_for_training_or_gate"),
            "sealed_used": stratified.get("sealed_used"), "domain_counts": stratified_counts},
           "required controls / false / false / domain counts sum to 1394")

    training = structural.get("training_budget", {})
    structural_runs = structural.get("runs", {})
    _check(checks, "structural_status", structural.get("status") == "COMPLETE_SEED17_CPU_STRUCTURAL_COMMON_STATE_COMPARISON",
           structural.get("status"), "COMPLETE_SEED17_CPU_STRUCTURAL_COMMON_STATE_COMPARISON")
    _check(checks, "structural_budget", [training.get("seed"), training.get("steps"), training.get("multi_seed"), training.get("device")] == [17, 1000, False, "cpu"],
           [training.get("seed"), training.get("steps"), training.get("multi_seed"), training.get("device")], [17, 1000, False, "cpu"])
    _check(checks, "structural_candidates_and_controls", set(structural_runs) == {
        "identity_residual_predictive", "identity_residual_ranking_only",
        "candidate_relative_predictive", "candidate_relative_ranking_only",
    }, sorted(structural_runs), [
        "candidate_relative_predictive", "candidate_relative_ranking_only",
        "identity_residual_predictive", "identity_residual_ranking_only",
    ])
    _check(checks, "structural_fixed_state", [structural.get("common_state_events"), structural.get("candidate_present_events")] == [1394, 1177],
           [structural.get("common_state_events"), structural.get("candidate_present_events")], [1394, 1177])
    _check(checks, "structural_no_gt_or_claim", structural.get("identity_evidence_claim") is False
           and structural.get("target_gt_used_for_training_or_gate") is False
           and all(run.get("target_gt_used_for_training_or_gate") is False and run.get("candidate_input_to_history_encoder") is False
                   and run.get("preflight_status") == "PASS_STRUCTURAL_CANDIDATE_PREFLIGHT"
                   for run in structural_runs.values()),
           {"identity_evidence_claim": structural.get("identity_evidence_claim"),
            "target_gt_used_for_training_or_gate": structural.get("target_gt_used_for_training_or_gate"),
            "per_run": {name: [run.get("target_gt_used_for_training_or_gate"), run.get("candidate_input_to_history_encoder"), run.get("preflight_status")]
                        for name, run in structural_runs.items()}},
           "false / false / no candidate in history encoder / all preflights pass")

    _check(checks, "ua_supplement_status", ua.get("status") == "COMPLETE_UA_LEGACY_TRACK_CPU_COMMON_STATE_REPLAY_SEED17"
           and ua.get("input_video_count") == 17 and ua.get("scored_case_count") == 4199,
           [ua.get("status"), ua.get("input_video_count"), ua.get("scored_case_count")],
           ["COMPLETE_UA_LEGACY_TRACK_CPU_COMMON_STATE_REPLAY_SEED17", 17, 4199])
    _check(checks, "ua_no_training_or_candidate_gt", ua.get("target_GT_used_for_training_or_candidate_selection") is False
           and ua.get("target_GT_used_for_evaluability_filter") is True and ua.get("sealed_used") is False,
           [ua.get("target_GT_used_for_training_or_candidate_selection"), ua.get("target_GT_used_for_evaluability_filter"), ua.get("sealed_used")],
           [False, True, False])

    for name, report in [("event_taxonomy", taxonomy), ("headroom", headroom), ("ranking", ranking)]:
        _check(checks, f"{name}_status_present", str(report.get("status", "")).startswith("COMPLETE_"),
               report.get("status"), "COMPLETE_*")
    _check(checks, "all_registered_offline_audits_no_training_gt", all(
        report.get("target_GT_used_for_training_or_gate") is False and report.get("sealed_used") is False
        for report in (taxonomy, headroom)),
        {"taxonomy": [taxonomy.get("target_GT_used_for_training_or_gate"), taxonomy.get("sealed_used")],
         "headroom": [headroom.get("target_GT_used_for_training_or_gate"), headroom.get("sealed_used")]},
        "false / false")

    passed = all(item["pass"] for item in checks)
    return {
        "status": "COMPLETE_STATIC_H1_EVIDENCE_AUDIT_WITH_RUNTIME_LIMITS" if passed else "FAILED_STATIC_H1_EVIDENCE_AUDIT",
        "scope": "seed17 BDD common-state and structural evidence plus supplementary UA legacy-track replay",
        "static_checks_pass": passed,
        "checks": checks,
        "scientific_claim": "H1_GENERAL_CLAIM_NOT_ESTABLISHED",
        "remaining_runtime_or_scope_limits": [
            "UA registered fresh event tracer remains unexecuted because the current run requires restored NAS/GPU runtime",
            "the BDD common state is a recorded predictive-residual state and top candidate subset, not an all-opportunity independent rollout",
            "the structural comparison is one development seed; no multi-seed confidence claim is made",
            "sealed evaluation remains frozen",
            "registered full-video H2 Wave-A remains unexecuted",
        ],
        "next_allowed_action": "Restore the NAS SSHFS endpoint and NVIDIA driver, rerun the registered UA event preflight, then run only the already registered one-seed forward-order H2 Wave-A factorial; do not add seeds or threshold sweeps.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=Path, default=Path("docs/JEPA_IDENTITY"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.docs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("H1_EVIDENCE_AUDIT", report["status"], "checks", len(report["checks"]), flush=True)
    if not report["static_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
