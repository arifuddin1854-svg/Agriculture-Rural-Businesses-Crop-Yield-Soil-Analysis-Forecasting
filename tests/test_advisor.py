"""
Unit Tests for Prescriptive Agronomic Advisory Engine
Day 10 Milestone
"""

import unittest
from pathlib import Path
from src.advisor import generate_advisory_report, build_feature_vector


class TestAdvisorEngine(unittest.TestCase):
    """Test suite verifying agronomic advisory diagnostics and yield forecasting."""

    def test_acidic_soil_diagnostic(self):
        report = generate_advisory_report(
            crop="Wheat",
            nitrogen=120.0,
            phosphorus=45.0,
            potassium=130.0,
            ph=5.2,  # Acidic
            verbose=False
        )
        self.assertIn("recommendations", report)
        self.assertGreater(report["total_potential_gain_mt_ha"], 0.0)
        self.assertGreater(report["projected_yield_mt_ha"], report["baseline_yield_mt_ha"])

        ph_rec = next(r for r in report["recommendations"] if r["factor"] == "Soil pH")
        self.assertIn("Moderately Acidic", ph_rec["status"])
        self.assertIn("limestone", ph_rec["recommendation"])
        self.assertGreater(ph_rec["impact_mt_ha"], 0.0)

    def test_nitrogen_deficiency_diagnostic(self):
        report = generate_advisory_report(
            crop="Rice",
            nitrogen=75.0,  # Deficient
            phosphorus=45.0,
            potassium=130.0,
            ph=6.5,  # Optimal
            verbose=False
        )
        n_rec = next(r for r in report["recommendations"] if "Nitrogen" in r["factor"])
        self.assertIn("Deficiency", n_rec["status"])
        self.assertIn("Top-dress", n_rec["recommendation"])
        self.assertGreater(n_rec["impact_mt_ha"], 0.0)

    def test_optimal_soil_profile(self):
        report = generate_advisory_report(
            crop="Maize",
            nitrogen=125.0,
            phosphorus=45.0,
            potassium=130.0,
            ph=6.6,
            verbose=False
        )
        self.assertGreaterEqual(report["baseline_yield_mt_ha"], 1.0)
        self.assertIsNotNone(report["projected_yield_mt_ha"])


if __name__ == "__main__":
    unittest.main()
