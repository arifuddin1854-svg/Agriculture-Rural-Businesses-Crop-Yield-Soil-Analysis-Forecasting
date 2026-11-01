"""
Unit and Integration Tests for Day 17 Containerization, CI/CD, and System Health Audit
Axlero Agri-Tech Platform / Member 1
"""

import unittest
from pathlib import Path
from run_system_health_audit import SystemHealthAuditor


class TestSystemHealthAudit(unittest.TestCase):
    """Test suite covering the platform health audit and containerization readiness."""

    @classmethod
    def setUpClass(cls):
        cls.auditor = SystemHealthAuditor()

    def test_01_auditor_initialization(self):
        """Verify auditor paths and category containers initialize properly."""
        self.assertTrue(self.auditor.models_dir.exists())
        self.assertTrue(self.auditor.data_dir.exists())
        self.assertIn("overall_status", self.auditor.audit_results)

    def test_02_model_checkpoint_audit(self):
        """Verify model checkpoint auditing passes for registered regressors."""
        self.auditor.audit_model_checkpoints()
        categories = self.auditor.audit_results["categories"]
        self.assertIn("Model Checkpoints", categories)
        checks = categories["Model Checkpoints"]
        # Ensure at least the primary production model passed
        rf_checks = [c for c in checks if "Production Regressor" in c["check"]]
        self.assertTrue(len(rf_checks) > 0)
        self.assertEqual(rf_checks[0]["status"], "PASS")

    def test_03_dataset_audit(self):
        """Verify feature layers and datasets exist and conform to schemas."""
        self.auditor.audit_datasets_and_feature_stores()
        categories = self.auditor.audit_results["categories"]
        self.assertIn("Data Store & Feature Layers", categories)
        checks = categories["Data Store & Feature Layers"]
        self.assertGreaterEqual(len(checks), 5)
        # Check that unified dataset passed
        unified = [c for c in checks if "Unified Agronomic Dataset" in c["check"]]
        self.assertTrue(len(unified) > 0)
        self.assertEqual(unified[0]["status"], "PASS")

    def test_04_full_audit_execution_and_reports(self):
        """Execute full health audit and verify report output files."""
        fresh_auditor = SystemHealthAuditor()
        results = fresh_auditor.run_full_audit(skip_latency=True)

        self.assertIn(results["overall_status"], ["HEALTHY", "DEGRADED"])
        self.assertGreater(results["checks_passed"], 10)
        self.assertEqual(results["checks_failed"], 0)

        json_rep = Path("reports/system_health_audit.json")
        md_rep = Path("reports/system_health_audit.md")
        self.assertTrue(json_rep.exists())
        self.assertTrue(md_rep.exists())
        self.assertGreater(json_rep.stat().st_size, 100)
        self.assertGreater(md_rep.stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
