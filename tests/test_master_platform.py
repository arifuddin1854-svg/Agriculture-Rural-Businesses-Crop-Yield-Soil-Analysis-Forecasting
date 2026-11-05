"""
Unit and Integration Tests for Day 18 Master Platform Orchestrator
Member 1: Systems & ML Infrastructure Engineer
Axlero Agri-Tech Platform
"""

import unittest
from run_master_platform import MasterPlatformOrchestrator, PLATFORM_STAGES


class TestMasterPlatformOrchestrator(unittest.TestCase):
    """Test suite covering the Master Platform Orchestration CLI."""

    def setUp(self):
        self.orchestrator = MasterPlatformOrchestrator(verbose=False)

    def test_01_stages_registration(self):
        """Verify all enterprise pipeline stages are registered with valid modules."""
        stages = self.orchestrator.list_stages()
        self.assertEqual(len(stages), 9)

        expected_ids = ["pipeline", "features", "models", "neural", "satellite", "inference", "climate", "economics", "audit"]
        stage_ids = [s["id"] for s in stages]
        self.assertListEqual(stage_ids, expected_ids)

        for s in stages:
            self.assertIn("name", s)
            self.assertIn("module", s)
            self.assertIn("function", s)
            self.assertIn("description", s)

    def test_02_dry_run_all(self):
        """Verify full pipeline dry run simulates topological stage sequence cleanly."""
        summary = self.orchestrator.run_all(dry_run=True)
        self.assertEqual(summary["overall_status"], "SUCCESS")
        self.assertIsNone(summary["failed_stage"])
        self.assertEqual(len(summary["stage_results"]), 9)

        for res in summary["stage_results"]:
            self.assertEqual(res["status"], "DRY_RUN")

    def test_03_dry_run_individual_stages(self):
        """Verify individual stages execute properly in dry run mode."""
        for stage_id in ["pipeline", "models", "satellite", "climate", "audit"]:
            res = self.orchestrator.execute_stage(stage_id, dry_run=True)
            self.assertEqual(res["stage"], stage_id)
            self.assertEqual(res["status"], "DRY_RUN")

    def test_04_unknown_stage_raises_error(self):
        """Verify requesting an invalid stage raises ValueError."""
        with self.assertRaises(ValueError):
            self.orchestrator.execute_stage("non_existent_stage", dry_run=True)

    def test_05_execute_audit_stage_live(self):
        """Verify live execution of system health audit stage succeeds."""
        res = self.orchestrator.execute_stage("audit", dry_run=False)
        self.assertEqual(res["stage"], "audit")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertGreater(res["duration_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
