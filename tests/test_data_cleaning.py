"""
Unit and Integration Tests for Data Cleaning & Missing Value Imputation Pipeline
Member 1: Data Integration & Data Engineer (Day 2 Milestone)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_cleaning import (
    clean_soil_chemistry,
    clean_weather_daily,
    clean_historical_yields,
    run_data_cleaning_pipeline,
    QualityAuditTracker,
    BOUNDARIES
)


class TestSoilDataCleaning(unittest.TestCase):
    """Test suite for soil chemistry cleaning and boundary enforcement."""

    def setUp(self):
        self.raw_soil = pd.DataFrame([
            {
                "district_id": "DIST_101",
                "soil_type": "Black Soil",
                "nitrogen_kg_ha": 120.0,
                "phosphorus_kg_ha": 45.0,
                "potassium_kg_ha": 150.0,
                "soil_ph": 11.5,  # Out of bounds (> 9.5)
                "organic_matter_pct": 1.2,
                "electrical_conductivity_ds_m": 0.4
            },
            {
                "district_id": "DIST_102",
                "soil_type": "Black Soil",
                "nitrogen_kg_ha": np.nan,  # Missing value
                "phosphorus_kg_ha": 55.0,
                "potassium_kg_ha": 170.0,
                "soil_ph": 2.1,  # Out of bounds (< 3.5)
                "organic_matter_pct": 1.4,
                "electrical_conductivity_ds_m": 0.5
            },
            {
                "district_id": "DIST_103",
                "soil_type": "Black Soil",
                "nitrogen_kg_ha": 140.0,
                "phosphorus_kg_ha": 50.0,
                "potassium_kg_ha": 160.0,
                "soil_ph": 6.8,
                "organic_matter_pct": 1.3,
                "electrical_conductivity_ds_m": 0.45
            }
        ])

    def test_soil_imputation_and_boundaries(self):
        tracker = QualityAuditTracker("test_soil")
        clean_df = clean_soil_chemistry(self.raw_soil, tracker)

        # 1. Check missing nitrogen imputed from same soil_type median (median of 120, 140 = 130)
        self.assertFalse(clean_df["nitrogen_kg_ha"].isnull().any())
        self.assertEqual(clean_df.loc[1, "nitrogen_kg_ha"], 130.0)

        # 2. Check pH clamping (11.5 clamped to 9.5, 2.1 clamped to 3.5)
        self.assertEqual(clean_df.loc[0, "soil_ph"], 9.5)
        self.assertEqual(clean_df.loc[1, "soil_ph"], 3.5)
        self.assertEqual(clean_df.loc[2, "soil_ph"], 6.8)

        # 3. Check tracker records
        self.assertEqual(tracker.imputed_nulls.get("nitrogen_kg_ha"), 1)
        self.assertEqual(tracker.boundary_violations.get("soil_ph"), 2)


class TestWeatherDataCleaning(unittest.TestCase):
    """Test suite for weather data cleaning, rolling imputation, and temperature sanity."""

    def setUp(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        self.raw_weather = pd.DataFrame({
            "district_id": ["DIST_101"] * 10,
            "date": dates.strftime("%Y-%m-%d"),
            "temp_max": [25.0, 26.0, 15.0, 27.0, 28.0, np.nan, 29.0, 30.0, 31.0, 32.0],  # row 2 has temp_max < temp_min
            "temp_min": [12.0, 13.0, 22.0, 14.0, 15.0, 16.0, 16.0, 17.0, 18.0, 19.0],  # row 2 has temp_min > temp_max
            "precipitation_mm": [0.0, -5.0, 12.0, 0.0, np.nan, 2.0, 0.0, 0.0, 5.0, 0.0],  # row 1 has negative rain
            "relative_humidity": [50.0, 120.0, 45.0, 60.0, 55.0, 50.0, 48.0, 52.0, 58.0, 50.0],  # row 1 has >100%
            "solar_radiation_mj": [18.0, 19.0, 17.0, 20.0, 21.0, 22.0, 20.0, 19.0, 21.0, 20.0]
        })

    def test_weather_cleaning_logic(self):
        tracker = QualityAuditTracker("test_weather")
        clean_df = clean_weather_daily(self.raw_weather, tracker)

        # 1. Verify temperature inversion swapping (row 2: min was 22, max was 15 -> swapped to min 15, max 22)
        self.assertTrue((clean_df["temp_max"] >= clean_df["temp_min"]).all())
        self.assertEqual(tracker.swapped_temp_inversions, 1)

        # 2. Verify negative precipitation clamped to 0.0
        self.assertEqual(clean_df.loc[1, "precipitation_mm"], 0.0)
        self.assertTrue((clean_df["precipitation_mm"] >= 0.0).all())

        # 3. Verify humidity clamped to 100.0
        self.assertEqual(clean_df.loc[1, "relative_humidity"], 100.0)

        # 4. Verify rolling median imputation filled missing temp_max and precipitation
        self.assertFalse(clean_df["temp_max"].isnull().any())
        self.assertFalse(clean_df["precipitation_mm"].isnull().any())


class TestYieldDataCleaning(unittest.TestCase):
    """Test suite for crop yields cleaning and arithmetic reconciliation."""

    def setUp(self):
        self.raw_yield = pd.DataFrame([
            {
                "district_id": "DIST_101",
                "year": 2022,
                "crop_type": "Wheat",
                "planted_area_ha": 1000.0,
                "production_mt": 4500.0,
                "crop_yield_mt_ha": np.nan  # Missing yield, reconstructable: 4500 / 1000 = 4.5
            },
            {
                "district_id": "DIST_102",
                "year": 2022,
                "crop_type": "Wheat",
                "planted_area_ha": 2000.0,
                "production_mt": np.nan,  # Missing production, reconstructable: 5.0 * 2000 = 10000
                "crop_yield_mt_ha": 5.0
            },
            {
                "district_id": "DIST_103",
                "year": 2022,
                "crop_type": "Wheat",
                "planted_area_ha": 1500.0,
                "production_mt": 6000.0,
                "crop_yield_mt_ha": 4.0
            }
        ])

    def test_yield_arithmetic_reconstruction(self):
        tracker = QualityAuditTracker("test_yield")
        clean_df = clean_historical_yields(self.raw_yield, tracker)

        # 1. Missing yield reconstructed: 4500 / 1000 = 4.5
        self.assertAlmostEqual(clean_df.loc[0, "crop_yield_mt_ha"], 4.5, places=2)

        # 2. Missing production reconstructed: 5.0 * 2000 = 10000.0
        self.assertAlmostEqual(clean_df.loc[1, "production_mt"], 10000.0, places=1)


class TestFullDataCleaningPipeline(unittest.TestCase):
    """Integration test running the full pipeline end-to-end on temporary directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = Path(self.temp_dir) / "raw"
        self.interim_dir = Path(self.temp_dir) / "interim"
        self.report_dir = Path(self.temp_dir) / "reports"
        self.raw_dir.mkdir(parents=True)

        # Copy existing raw files
        shutil.copy("data/raw/soil_chemistry.csv", self.raw_dir / "soil_chemistry.csv")
        shutil.copy("data/raw/historical_yields.csv", self.raw_dir / "historical_yields.csv")
        shutil.copy("data/raw/weather_daily.csv", self.raw_dir / "weather_daily.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_pipeline(self):
        results = run_data_cleaning_pipeline(
            raw_dir=str(self.raw_dir),
            interim_dir=str(self.interim_dir),
            report_dir=str(self.report_dir)
        )

        # Check clean files exist
        self.assertTrue((self.interim_dir / "clean_soil_chemistry.csv").exists())
        self.assertTrue((self.interim_dir / "clean_historical_yields.csv").exists())
        self.assertTrue((self.interim_dir / "clean_weather_daily.csv").exists())

        # Check report files exist
        self.assertTrue((self.report_dir / "data_cleaning_validation_report.json").exists())
        self.assertTrue((self.report_dir / "data_cleaning_validation_report.md").exists())

        # Verify no NaN values in any cleaned dataset
        for filename in ["clean_soil_chemistry.csv", "clean_historical_yields.csv", "clean_weather_daily.csv"]:
            df = pd.read_csv(self.interim_dir / filename)
            self.assertEqual(df.isnull().sum().sum(), 0, f"Found unexpected NaNs in {filename}")


if __name__ == "__main__":
    unittest.main()
