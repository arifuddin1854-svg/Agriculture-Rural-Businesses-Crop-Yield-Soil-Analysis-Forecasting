"""
Unit and Integration Tests for Geospatial Data Fusion Pipeline
Member 1: Data Integration & Data Engineer (Day 4 Milestone)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_fusion import (
    verify_referential_integrity,
    fuse_agricultural_datasets,
    generate_data_fusion_report,
    run_data_fusion_pipeline
)


class TestReferentialIntegrity(unittest.TestCase):
    """Test suite for foreign key validation across datasets."""

    def setUp(self):
        self.yields = pd.DataFrame([
            {"district_id": "DIST_101", "year": 2022, "crop_type": "Wheat", "crop_yield_mt_ha": 4.5},
            {"district_id": "DIST_102", "year": 2022, "crop_type": "Rice", "crop_yield_mt_ha": 3.8},
        ])
        self.soil = pd.DataFrame([
            {"district_id": "DIST_101", "soil_type": "Black Soil", "soil_ph": 6.5},
            {"district_id": "DIST_102", "soil_type": "Alluvial", "soil_ph": 7.1},
        ])
        self.weather = pd.DataFrame([
            {"district_id": "DIST_101", "year": 2022, "crop_type": "Wheat", "cumulative_rainfall_mm": 250.0},
            {"district_id": "DIST_102", "year": 2022, "crop_type": "Rice", "cumulative_rainfall_mm": 1100.0},
        ])

    def test_referential_integrity_success(self):
        result = verify_referential_integrity(self.yields, self.soil, self.weather)
        self.assertTrue(result["integrity_passed"])
        self.assertEqual(len(result["issues"]), 0)

    def test_missing_district_in_soil(self):
        # Remove DIST_102 from soil
        soil_missing = self.soil[self.soil["district_id"] == "DIST_101"]
        result = verify_referential_integrity(self.yields, soil_missing, self.weather)
        self.assertFalse(result["integrity_passed"])
        self.assertTrue(any("Missing 1 districts in soil chemistry" in issue for issue in result["issues"]))

    def test_missing_combo_in_weather(self):
        # Remove DIST_102-Rice from weather
        weather_missing = self.weather[self.weather["district_id"] == "DIST_101"]
        result = verify_referential_integrity(self.yields, self.soil, weather_missing)
        self.assertFalse(result["integrity_passed"])
        self.assertTrue(any("Missing 1 (district, year, crop) combinations" in issue for issue in result["issues"]))


class TestDataFusionMerging(unittest.TestCase):
    """Test suite for dataframe merging and schema ordering."""

    def setUp(self):
        self.yields = pd.DataFrame([
            {"district_id": "DIST_101", "year": 2022, "crop_type": "Wheat", "planted_area_ha": 1000.0, "production_mt": 4500.0, "crop_yield_mt_ha": 4.5},
            {"district_id": "DIST_102", "year": 2022, "crop_type": "Rice", "planted_area_ha": 2000.0, "production_mt": 7600.0, "crop_yield_mt_ha": 3.8},
        ])
        self.soil = pd.DataFrame([
            {"district_id": "DIST_101", "soil_type": "Black Soil", "nitrogen_kg_ha": 120.0, "soil_ph": 6.5},
            {"district_id": "DIST_102", "soil_type": "Alluvial", "nitrogen_kg_ha": 95.0, "soil_ph": 7.1},
        ])
        self.weather = pd.DataFrame([
            {"district_id": "DIST_101", "year": 2022, "crop_type": "Wheat", "cumulative_rainfall_mm": 250.0, "gdd_cumulative": 2300.0},
            {"district_id": "DIST_102", "year": 2022, "crop_type": "Rice", "cumulative_rainfall_mm": 1100.0, "gdd_cumulative": 3100.0},
        ])

    def test_successful_fusion(self):
        fused_df, summary = fuse_agricultural_datasets(self.yields, self.soil, self.weather)

        self.assertEqual(len(fused_df), 2)
        self.assertEqual(summary["null_values_count"], 0)
        self.assertIn("district_id", fused_df.columns)
        self.assertIn("soil_type", fused_df.columns)
        self.assertIn("cumulative_rainfall_mm", fused_df.columns)
        self.assertIn("crop_yield_mt_ha", fused_df.columns)

        # Check column ordering: core keys first
        self.assertEqual(fused_df.columns[0], "district_id")
        self.assertEqual(fused_df.columns[1], "year")
        self.assertEqual(fused_df.columns[2], "crop_type")


class TestDataFusionPipelineIntegration(unittest.TestCase):
    """End-to-end integration test for Day 4 Data Fusion."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.interim_dir = Path(self.temp_dir) / "interim"
        self.processed_dir = Path(self.temp_dir) / "processed"
        self.report_dir = Path(self.temp_dir) / "reports"
        self.interim_dir.mkdir(parents=True)
        self.processed_dir.mkdir(parents=True)

        # Copy existing interim and processed inputs
        shutil.copy("data/interim/clean_historical_yields.csv", self.interim_dir / "clean_historical_yields.csv")
        shutil.copy("data/interim/clean_soil_chemistry.csv", self.interim_dir / "clean_soil_chemistry.csv")
        shutil.copy("data/processed/weather_aggregated.csv", self.processed_dir / "weather_aggregated.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_fusion_pipeline(self):
        results = run_data_fusion_pipeline(
            yields_path=str(self.interim_dir / "clean_historical_yields.csv"),
            soil_path=str(self.interim_dir / "clean_soil_chemistry.csv"),
            weather_path=str(self.processed_dir / "weather_aggregated.csv"),
            output_dir=str(self.processed_dir),
            report_dir=str(self.report_dir)
        )

        csv_file = self.processed_dir / "unified_agri_dataset.csv"
        json_report = self.report_dir / "data_fusion_report.json"
        md_report = self.report_dir / "data_fusion_report.md"

        self.assertTrue(csv_file.exists())
        self.assertTrue(json_report.exists())
        self.assertTrue(md_report.exists())

        fused_df = pd.read_csv(csv_file)
        self.assertEqual(len(fused_df), 225)
        self.assertEqual(fused_df.isnull().sum().sum(), 0)
        self.assertGreaterEqual(len(fused_df.columns), 30)


if __name__ == "__main__":
    unittest.main()
