"""
Integration Tests for End-to-End Master Data Pipeline
Member 1: Data Integration & Data Engineer (Day 6 Deliverable)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd

from run_data_pipeline import DataPipelineOrchestrator


class TestMasterDataPipeline(unittest.TestCase):
    """Integration test suite executing full pipeline in temporary isolated environment."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.interim_dir = Path(self.temp_dir) / "interim"
        self.processed_dir = Path(self.temp_dir) / "processed"
        self.report_dir = Path(self.temp_dir) / "reports"
        self.figures_dir = Path(self.temp_dir) / "reports" / "figures"

        self.orchestrator = DataPipelineOrchestrator(
            raw_dir=str(self.raw_dir),
            interim_dir=str(self.interim_dir),
            processed_dir=str(self.processed_dir),
            report_dir=str(self.report_dir),
            figures_dir=str(self.figures_dir),
            seed=42
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_all_pipeline(self):
        # Run all 5 stages end-to-end
        self.orchestrator.run_all()

        # 1. Verify Raw Ingestion Artifacts
        self.assertTrue((self.raw_dir / "soil_chemistry.csv").exists())
        self.assertTrue((self.raw_dir / "historical_yields.csv").exists())
        self.assertTrue((self.raw_dir / "weather_daily.csv").exists())

        # 2. Verify Interim Clean Artifacts
        self.assertTrue((self.interim_dir / "clean_soil_chemistry.csv").exists())
        self.assertTrue((self.interim_dir / "clean_historical_yields.csv").exists())
        self.assertTrue((self.interim_dir / "clean_weather_daily.csv").exists())

        # 3. Verify Processed Weather Aggregations
        self.assertTrue((self.processed_dir / "weather_aggregated.csv").exists())
        self.assertTrue((self.processed_dir / "weather_aggregated_annual.csv").exists())

        # 4. Verify Final Master Unified Dataset
        master_csv = self.processed_dir / "unified_agri_dataset.csv"
        self.assertTrue(master_csv.exists())

        df = pd.read_csv(master_csv)
        self.assertEqual(len(df), 225)
        self.assertEqual(df.isnull().sum().sum(), 0)
        self.assertGreaterEqual(len(df.columns), 30)

        # 5. Verify Reports & Visual Artifacts
        self.assertTrue((self.report_dir / "data_cleaning_validation_report.json").exists())
        self.assertTrue((self.report_dir / "data_cleaning_validation_report.md").exists())
        self.assertTrue((self.report_dir / "weather_aggregation_report.json").exists())
        self.assertTrue((self.report_dir / "weather_aggregation_report.md").exists())
        self.assertTrue((self.report_dir / "data_fusion_report.json").exists())
        self.assertTrue((self.report_dir / "data_fusion_report.md").exists())
        self.assertTrue((self.report_dir / "eda_summary_report.md").exists())

        for fig_name in [
            "yield_distributions_by_crop.png",
            "soil_chemistry_correlation_heatmap.png",
            "weather_stress_vs_yield.png",
            "district_spatial_yield_distribution.png",
            "temporal_yield_trends_2019_2023.png"
        ]:
            self.assertTrue((self.figures_dir / fig_name).exists(), f"Missing figure: {fig_name}")


if __name__ == "__main__":
    unittest.main()
