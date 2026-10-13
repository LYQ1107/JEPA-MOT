import unittest

from .paired_common_state import compare


def toy_case(frame, expected, native, predictive_scores, reference_scores):
    return {
        "video": "toy",
        "frame": frame,
        "track_id": 1,
        "branch": "high",
        "expected_gt_id": expected,
        "native_gt_id": native,
        "candidate_gt_ids": [
            {"index": 0, "gt_id": 7, "native_cost": 0.5},
            {"index": 1, "gt_id": 3, "native_cost": 0.6},
        ],
        "scores": {
            "mlp_jepa_predictive": predictive_scores,
            "mlp_ce": reference_scores,
        },
    }


class PairedCommonStateTest(unittest.TestCase):
    def test_pairs_same_candidate_present_mask_and_reports_correction_delta(self):
        cases = [
            toy_case(1, 7, 3, [0.1, 0.8], [0.8, 0.1]),
            toy_case(2, 7, 7, [0.8, 0.1], [0.1, 0.8]),
        ]
        report = compare(
            cases,
            controls=("mlp_jepa_predictive", "mlp_ce"),
            comparisons=(("mlp_jepa_predictive", "mlp_ce"),),
        )
        self.assertEqual(report["source_case_count"], 2)
        self.assertEqual(report["candidate_present_case_count"], 2)
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["metrics"]["top1"]["paired_cases"], 2)
        self.assertAlmostEqual(comparison["metrics"]["net_correction"]["mean_improvement"], 0.0)
        self.assertEqual(report["control_summary_candidate_present"]["mlp_jepa_predictive"]["net_correction"], 0.0)

    def test_candidate_absence_is_reported_and_excluded(self):
        case = toy_case(1, 9, 3, [0.1, 0.8], [0.8, 0.1])
        report = compare(
            [case],
            controls=("mlp_jepa_predictive", "mlp_ce"),
            comparisons=(("mlp_jepa_predictive", "mlp_ce"),),
        )
        self.assertEqual(report["candidate_present_case_count"], 0)
        self.assertEqual(report["candidate_absent_case_count"], 1)
        self.assertEqual(report["comparisons"][0]["metrics"]["top1"]["paired_cases"], 0)


if __name__ == "__main__":
    unittest.main()
