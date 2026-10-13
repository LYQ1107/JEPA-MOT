import unittest
from pathlib import Path

from .h1_evidence_audit import audit


class H1EvidenceAuditTest(unittest.TestCase):
    def test_registered_seed17_bundle_passes_static_gate(self):
        report = audit(Path("docs/JEPA_IDENTITY"))
        self.assertTrue(report["static_checks_pass"])
        self.assertEqual(report["status"], "COMPLETE_STATIC_H1_EVIDENCE_AUDIT_WITH_RUNTIME_LIMITS")
        self.assertEqual(len(report["checks"]), 26)
        self.assertEqual(report["scientific_claim"], "H1_GENERAL_CLAIM_NOT_ESTABLISHED")


if __name__ == "__main__":
    unittest.main()
