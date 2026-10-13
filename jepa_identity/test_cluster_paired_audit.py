import unittest

from .cluster_paired_audit import audit


def case(video, frame, predictive, reference):
    return {
        "video": video,
        "frame": frame,
        "track_id": frame,
        "branch": "high",
        "expected_gt_id": 7,
        "native_gt_id": 3,
        "candidate_gt_ids": [{"index": 0, "gt_id": 7}, {"index": 1, "gt_id": 3}],
        "scores": {"mlp_jepa_predictive": predictive, "mlp_ce": reference},
    }


class ClusterPairedAuditTest(unittest.TestCase):
    def test_video_is_the_resampling_unit(self):
        report = audit(
            [case("a", 1, [0.1, 0.9], [0.9, 0.1]),
             case("a", 2, [0.1, 0.9], [0.9, 0.1]),
             case("b", 3, [0.9, 0.1], [0.1, 0.9])],
            dataset="toy",
            controls=("mlp_jepa_predictive", "mlp_ce"),
            comparisons=(("mlp_jepa_predictive", "mlp_ce"),),
        )
        metric = report["comparisons"][0]["metrics"]["top1"]
        self.assertEqual(report["video_count"], 2)
        self.assertEqual(metric["clusters"], 2)
        self.assertEqual(metric["positive_clusters"], 1)
        self.assertEqual(metric["negative_clusters"], 1)


if __name__ == "__main__":
    unittest.main()
