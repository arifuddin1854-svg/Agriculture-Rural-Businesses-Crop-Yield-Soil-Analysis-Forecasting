"""
Unit and Integration Tests for Day 19 Agricultural Economics & Portfolio Optimization
Member 1 / Rural Business Intelligence Deliverable
"""

import unittest
from pathlib import Path
import pandas as pd

from src.economic_optimizer import AgriEconomicOptimizer, COMMODITY_PRICES_USD_PER_MT


class TestAgriEconomicOptimizer(unittest.TestCase):
    """Test suite covering the Day 19 Agricultural Economics & Portfolio Optimizer."""

    @classmethod
    def setUpClass(cls):
        cls.optimizer = AgriEconomicOptimizer()
        cls.df_uni = pd.read_csv("data/processed/unified_agri_dataset.csv")

    def test_01_farm_financial_calculation(self):
        """Verify farm financial calculations produce positive revenues and coherent margins."""
        fin = self.optimizer.calculate_farm_financials(
            crop="Wheat",
            predicted_yield_mt_ha=4.2,
            planted_area_ha=50.0,
            nitrogen_kg_ha=110.0,
            phosphorus_kg_ha=45.0,
            potassium_kg_ha=45.0
        )
        self.assertEqual(fin["crop_type"], "Wheat")
        self.assertGreater(fin["per_hectare"]["gross_revenue_usd"], 0)
        self.assertGreater(fin["per_hectare"]["total_production_cost_usd"], 0)
        self.assertIn("net_profit_usd", fin["per_hectare"])
        self.assertAlmostEqual(
            fin["farm_totals"]["gross_revenue_usd"],
            fin["per_hectare"]["gross_revenue_usd"] * 50.0,
            places=1
        )

    def test_02_prescriptive_amendment_roi(self):
        """Verify soil amendment ROI calculates positive upside for viable interventions."""
        prescriptions = [
            {"action": "Apply Agricultural Limestone (CaCO3)", "category": "Soil Amendment", "expected_yield_gain_mt_ha": 0.5},
            {"action": "Synchronized Nitrogen Top-Dressing", "category": "Fertilizer", "expected_yield_gain_mt_ha": 0.3}
        ]
        roi_res = self.optimizer.evaluate_amendment_roi("Wheat", 4.0, prescriptions, 50.0)
        self.assertEqual(len(roi_res), 2)
        for r in roi_res:
            self.assertIn("amendment_investment_usd_ha", r)
            self.assertIn("incremental_revenue_usd_ha", r)
            self.assertIn("roi_percentage", r)
            self.assertIn(r["payback_economic_rating"], ["EXCELLENT", "POSITIVE", "SUBOPTIMAL"])

    def test_03_markowitz_crop_portfolio_weights(self):
        """Verify portfolio optimization weights sum to 1.0 (100% acreage) within bounds."""
        opt = self.optimizer.optimize_crop_portfolio(self.df_uni, target_risk_tolerance=0.5)
        weights = opt["optimal_acreage_weights"]

        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        for c in COMMODITY_PRICES_USD_PER_MT:
            self.assertIn(c, weights)
            self.assertGreaterEqual(weights[c], 0.04)

        self.assertGreater(opt["portfolio_expected_net_profit_usd_ha"], 0)
        self.assertGreater(len(opt["frontier_curve"]), 5)

    def test_04_full_economic_pipeline_reports(self):
        """Verify full economic pipeline generates figures and reports."""
        res = self.optimizer.run_full_economic_pipeline()
        self.assertIn("sample_farm_financial_profile", res)
        self.assertIn("markowitz_crop_portfolio_optimization", res)
        self.assertTrue(Path("reports/figures/08_crop_economic_frontier.png").exists())
        self.assertTrue(Path("reports/figures/08_fertilizer_roi_response.png").exists())
        self.assertTrue(Path("reports/economic_optimization_report.json").exists())
        self.assertTrue(Path("reports/economic_optimization_report.md").exists())

    def test_05_calculate_farm_financials_all_crops(self):
        """Verify farm financial valuation evaluates cleanly for all supported crops."""
        for crop in COMMODITY_PRICES_USD_PER_MT.keys():
            fin = self.optimizer.calculate_farm_financials(
                crop=crop,
                predicted_yield_mt_ha=3.5,
                planted_area_ha=20.0
            )
            self.assertEqual(fin["crop_type"], crop)
            self.assertGreater(fin["farm_totals"]["gross_revenue_usd"], 0)
            self.assertGreater(fin["farm_totals"]["total_production_cost_usd"], 0)
            self.assertIn("profit_margin_pct", fin["per_hectare"])


if __name__ == "__main__":
    unittest.main()

