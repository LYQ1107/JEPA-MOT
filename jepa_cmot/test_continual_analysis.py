import json
import tempfile
import unittest
from pathlib import Path

from .continual_analysis import update_magnitude_summary


def report(records, updates=None):
    return {
        "learner_stats": {
            "updates": len(records) if updates is None else updates,
            "update_magnitude": records,
        }
    }


class ContinualAnalysisUpdateMagnitudeTest(unittest.TestCase):
    def test_collects_per_video_reports_and_checks_matched_counts(self):
        records = [
            {"update": 1, "frame": 8, "l2": 0.1, "relative": 0.01, "parameter_l2": 10.0},
            {"update": 2, "frame": 16, "l2": 0.2, "relative": 0.02, "parameter_l2": 10.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "complete.json").write_text(json.dumps({
                "protocol": {"controls": [{"name": "D_to_D"}, {"name": "D_to_P"}]}
            }))
            for control in ("D_to_D", "D_to_P"):
                path = root / "adaptation" / "stage1_overcast" / "video1" / control / "video1" / "complete.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(report(records)))
            summary = update_magnitude_summary(root)
        self.assertEqual(summary["status"], "COMPLETE_UPDATE_MAGNITUDE_AUDIT")
        self.assertEqual(summary["record_count"], 4)
        self.assertTrue(summary["matched_update_counts"])
        self.assertEqual(summary["expected_controls"], ["D_to_D", "D_to_P"])
        self.assertAlmostEqual(summary["overall"]["mean_l2"], 0.15)
        self.assertEqual(summary["by_domain"]["overcast"]["record_count"], 4)

    def test_old_report_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "adaptation" / "stage1_overcast" / "video1" / "D_to_D" / "video1" / "complete.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"learner_stats": {"updates": 2}}))
            summary = update_magnitude_summary(root)
        self.assertEqual(summary["status"], "UPDATE_MAGNITUDE_UNAVAILABLE")
        self.assertEqual(summary["record_count"], 0)
        self.assertIsNone(summary["matched_update_counts"])
        self.assertEqual(summary["missing_update_magnitude_reports"], 1)

    def test_mismatched_control_counts_are_reported(self):
        records = [
            {"update": 1, "frame": 8, "l2": 0.1, "relative": 0.01, "parameter_l2": 10.0},
            {"update": 2, "frame": 16, "l2": 0.2, "relative": 0.02, "parameter_l2": 10.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "complete.json").write_text(json.dumps({
                "protocol": {"controls": [{"name": "D_to_D"}, {"name": "D_to_P"}]}
            }))
            first = root / "adaptation" / "stage1_overcast" / "video1" / "D_to_D" / "video1" / "complete.json"
            second = root / "adaptation" / "stage1_overcast" / "video1" / "D_to_P" / "video1" / "complete.json"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_text(json.dumps(report(records)))
            second.write_text(json.dumps(report(records[:1])))
            summary = update_magnitude_summary(root)
        self.assertFalse(summary["matched_update_counts"])
        self.assertEqual(summary["update_count_mismatches"][0]["counts"], {"D_to_D": 2, "D_to_P": 1})


if __name__ == "__main__":
    unittest.main()
