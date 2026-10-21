"""
Unit Tests for Agronomic Feature Engineering Pipeline
Day 7 Milestone
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from src.features import (
    compute_stoichiometric_features,
    compute_hydrothermal_indices,
    encode_categorical_features,
    run_feature_engineering_pipeline
)


class TestFeatureEngineering(unittest.TestCase):
    """Test suite verifying Day 7 agronomic transformations."""

    def setUp(self):
        self.sample_df = pd.DataFrame([{
            "district_id": "DIST_101",
            "year": 2022,
            "crop_type": "Wheat",
            "soil_type": "Alluvial",
            "planted_area_ha": 25000.0,
            "production_mt": 112500.0,
            "crop_yield_mt_ha": 4.5,
            "nitrogen_kg_ha": 120.0,
            "phosphorus_kg_ha": 40.0,
            "potassium_kg_ha": 60.0,
            "soil_ph": 6.8,
            "organic_matter_pct": 1.5,
            "electrical_conductivity_ds_m": 0.45,
            "growing_season_days": 150,
            "cumulative_rainfall_mm": 450.0,
            "mean_temp_max": 28.5,
            "heat_stress_days": 8,
            "consecutive_dry_days_max": 14,
            "gdd_cumulative": 1850.0,
            "cumulative_solar_radiation_mj": 2200.0,
            "mean_vpd_kpa": 1.4,
            "mean_diurnal_range": 12.0
        }])

    def test_stoichiometric_features(self):
        res = compute_stoichiometric_features(self.sample_df)
        self.assertIn("npk_total_kg_ha", res.columns)
        self.assertAlmostEqual(res["npk_total_kg_ha"].iloc[0], 220.0)
        self.assertIn("n_ratio", res.columns)
        self.assertIn("n_to_p_ratio", res.columns)
        self.assertAlmostEqual(res["n_to_p_ratio"].iloc[0], 3.0, places=1)
        self.assertIn("soil_ph_deviation_from_neutral", res.columns)
        self.assertAlmostEqual(res["soil_ph_deviation_from_neutral"].iloc[0], 0.3, places=2)

    def test_hydrothermal_indices(self):
        res = compute_hydrothermal_indices(self.sample_df)
        self.assertIn("hydro_thermal_stress_index", res.columns)
        self.assertIn("rainfall_per_season_day", res.columns)
        self.assertAlmostEqual(res["rainfall_per_season_day"].iloc[0], 3.0, places=1)
        self.assertIn("drought_severity_index", res.columns)
        self.assertIn("atmospheric_aridity_index", res.columns)

    def test_categorical_encoding(self):
        res, encoded_cols = encode_categorical_features(self.sample_df)
        self.assertIn("crop_type_Wheat", res.columns)
        self.assertIn("soil_type_Alluvial", res.columns)
        # Check spatial keys are preserved intact
        self.assertIn("district_id", res.columns)
        self.assertIn("year", res.columns)
        self.assertEqual(res["district_id"].iloc[0], "DIST_101")


class TestFeaturePipelineExecution(unittest.TestCase):
    """Integration test verifying end-to-end execution with temporary directories."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_file = Path(self.temp_dir) / "unified_test.csv"
        self.output_dir = Path(self.temp_dir) / "processed"
        self.report_dir = Path(self.temp_dir) / "reports"

        # Copy unified dataset to temp
        shutil.copy("data/processed/unified_agri_dataset.csv", self.input_file)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_feature_engineering_pipeline(self):
        artifacts = run_feature_engineering_pipeline(
            input_path=str(self.input_file),
            output_dir=str(self.output_dir),
            report_dir=str(self.report_dir)
        )
        self.assertTrue(artifacts["csv"].exists())
        self.assertTrue(artifacts["json_report"].exists())
        self.assertTrue(artifacts["md_report"].exists())

        df = pd.read_csv(artifacts["csv"])
        self.assertEqual(len(df), 225)
        self.assertGreater(len(df.columns), 45)
        self.assertEqual(df.isnull().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
