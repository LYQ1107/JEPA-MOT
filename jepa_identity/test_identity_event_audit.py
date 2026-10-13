import unittest

from .identity_event_audit import bdd_category_group, iou_xyxy, native_group
from .identity_metrics import ranking_case, summarize_cases


class IdentityEventAuditTest(unittest.TestCase):
    def test_iou_and_native_groups(self):
        self.assertAlmostEqual(iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(native_group(0), native_group(1))
        self.assertNotEqual(native_group(0), native_group(2))

    def test_bdd_category_mapping(self):
        self.assertEqual(bdd_category_group(1), native_group(0))
        self.assertEqual(bdd_category_group(6), native_group(7))

    def test_identity_ranking_separates_candidate_coverage(self):
        case = {
            "method": "toy",
            "domain": "clear",
            "diagnosis": "both_correct",
            "expected_gt_id": 7,
            "candidate_gt_ids": [
                {"index": 0, "gt_id": 3, "native_cost": 0.1, "fused_cost": 0.2},
                {"index": 1, "gt_id": 7, "native_cost": 0.3, "fused_cost": 0.05},
            ],
        }
        result = ranking_case(case, "fused_cost")
        self.assertTrue(result["candidate_present"])
        self.assertTrue(result["top1"])
        report = summarize_cases([case])
        self.assertEqual(report["methods"]["toy"]["intervention"]["correct_candidate_present"], 1)
        self.assertEqual(report["methods"]["toy"]["intervention"]["top1_identity_accuracy_conditional"], 1.0)


if __name__ == "__main__":
    unittest.main()
