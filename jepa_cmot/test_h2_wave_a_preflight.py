import json
import tempfile
import unittest
from pathlib import Path

from .h2_wave_a_preflight import H2_CELLS, control_protocol_audit


def controls():
    source = {
        "discriminative": "/work/source/discriminative/final.pt",
        "predictive": "/work/source/predictive/final.pt",
    }
    shared = {
        "enabled": True,
        "replay": 128,
        "batch": 8,
        "interval": 8,
        "learning_rate": 0.0001,
        "reliability": "confidence",
        "retention": "none",
        "seed": 17,
        "need_threshold": 0.05,
        "retention_weight": 1.0,
    }
    output = []
    for cell in H2_CELLS:
        config = dict(shared, method=cell["online_update"])
        output.append({
            "name": cell["id"],
            "method": "online",
            "checkpoint": source[cell["source_initialization"]],
            "geometry_duplicates": True,
            "online_config": config,
        })
    return output


class H2WaveAPreflightTest(unittest.TestCase):
    def test_registered_controls_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.json"
            path.write_text(json.dumps(controls()))
            result = control_protocol_audit(path, H2_CELLS)
        self.assertEqual(result["status"], "COMPLETE_H2_CONTROL_PROTOCOL_AUDIT")
        self.assertTrue(all(result["checks"].values()))

    def test_method_drift_fails(self):
        items = controls()
        items[0]["online_config"]["method"] = "predictive"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controls.json"
            path.write_text(json.dumps(items))
            result = control_protocol_audit(path, H2_CELLS)
        self.assertEqual(result["status"], "H2_CONTROL_PROTOCOL_MISMATCH")
        self.assertFalse(result["checks"]["cell_mapping"])


if __name__ == "__main__":
    unittest.main()
