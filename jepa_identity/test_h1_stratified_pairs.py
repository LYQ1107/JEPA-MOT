import unittest

from .h1_stratified_pairs import stratify


def case(domain, frame, predictive, reference):
    return {
        "domain": domain,
        "video": "toy",
        "frame": frame,
        "track_id": frame,
        "branch": "high",
        "expected_gt_id": 7,
        "native_gt_id": 3,
        "candidate_gt_ids": [{"index": 0, "gt_id": 7}, {"index": 1, "gt_id": 3}],
        "scores": {"mlp_jepa_predictive": predictive, "mlp_ce": reference},
    }


class H1StratifiedPairsTest(unittest.TestCase):
    def test_domain_groups_are_kept_separate(self):
        report = stratify(
            [case("clear", 1, [0.1, 0.9], [0.9, 0.1]),
             case("night", 2, [0.9, 0.1], [0.1, 0.9])],
            controls=("mlp_jepa_predictive", "mlp_ce"),
            comparisons=(("mlp_jepa_predictive", "mlp_ce"),),
        )
        self.assertEqual(set(report["domains"]), {"clear", "night"})
        self.assertEqual(report["domains"]["clear"]["candidate_present_case_count"], 1)
        self.assertEqual(report["domains"]["night"]["candidate_present_case_count"], 1)
        self.assertEqual(report["status"], "COMPLETE_BDD_DOMAIN_STRATIFIED_PAIRED_COMPARISON_SEED17")


if __name__ == "__main__":
    unittest.main()
