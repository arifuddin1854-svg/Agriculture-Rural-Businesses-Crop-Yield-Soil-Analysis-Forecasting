"""
Unit and Integration Tests for Day 16 Climate Change Scenario Simulation Engine
Member 1 / Production Climate Resilience Module
"""

import unittest
from pathlib import Path
import pandas as pd

from src.climate_scenarios import ClimateScenarioSimulator, CLIMATE_PATHWAYS


class TestClimateScenarios(unittest.TestCase):
    """Test suite covering the Day 16 Climate Change Simulation Engine."""

    @classmethod
    def setUpClass(cls):
        cls.simulator = ClimateScenarioSimulator()

    def test_01_pathway_definitions(self):
        """Verify SSP2-4.5 and SSP5-8.5 have positive temperature shifts."""
        self.assertIn("SSP2_45", CLIMATE_PATHWAYS)
        self.assertIn("SSP5_85", CLIMATE_PATHWAYS)
        self.assertGreater(CLIMATE_PATHWAYS["SSP5_85"]["temp_shift_c"], CLIMATE_PATHWAYS["SSP2_45"]["temp_shift_c"])
        self.assertGreater(CLIMATE_PATHWAYS["SSP5_85"]["heat_days_multiplier"], 1.0)

    def test_02_single_scenario_simulation(self):
        """Verify single pathway simulation produces expected schema and row count."""
        df_base = self.simulator.load_baseline().head(10)
        df_sim = self.simulator.simulate_pathway(df_base, "SSP2_45")

        self.assertEqual(len(df_sim), 10)
        self.assertIn("projected_yield_mt_ha", df_sim.columns)
        self.assertIn("yield_change_pct", df_sim.columns)
        self.assertIn("risk_level", df_sim.columns)
        self.assertTrue((df_sim["projected_yield_mt_ha"] > 0).all())

    def test_03_resilience_reports_emission(self):
        """Verify resilience reports are created and contain adaptation policies."""
        df_base = self.simulator.load_baseline().head(10)
        df_ssp2 = self.simulator.simulate_pathway(df_base, "SSP2_45")
        df_ssp5 = self.simulator.simulate_pathway(df_base, "SSP5_85")
        df_all = pd.concat([df_ssp2, df_ssp5], ignore_index=True)

        res = self.simulator.generate_resilience_reports(df_all, 0.5)
        self.assertIn("crop_impacts", res)
        self.assertIn("recommended_adaptation_policies", res)
        self.assertGreaterEqual(len(res["recommended_adaptation_policies"]), 1)
        self.assertTrue(Path("reports/climate_resilience_report.json").exists())
        self.assertTrue(Path("reports/climate_resilience_report.md").exists())


if __name__ == "__main__":
    unittest.main()
