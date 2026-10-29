"""
Unit and Integration Tests for Phenological Weather Aggregations Pipeline
Member 1: Data Integration & Data Engineer (Day 3 Milestone)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from src.weather_aggregation import (
    CropPhenologyConfig,
    DEFAULT_PHENOLOGY_PROFILES,
    calculate_daily_gdd,
    calculate_vapor_pressure_deficit,
    calculate_max_consecutive_dry_days,
    aggregate_weather_window,
    extract_phenological_window,
    aggregate_crop_phenology,
    aggregate_annual_weather,
    run_weather_aggregation_pipeline
)


class TestAgronomicCalculations(unittest.TestCase):
    """Test suite for core agro-meteorological transformation equations."""

    def test_gdd_standard_calculation(self):
        # T_max = 28, T_min = 16, T_base = 10, T_upper = 35
        # T_eff_max = 28, T_eff_min = 16, T_mean = 22.0, GDD = 22.0 - 10.0 = 12.0
        gdd = calculate_daily_gdd(np.array([28.0]), np.array([16.0]), t_base=10.0, t_upper=35.0)
        self.assertAlmostEqual(float(gdd[0]), 12.0, places=2)

    def test_gdd_below_base_temp(self):
        # T_max = 8, T_min = 2, T_base = 10 -> T_eff clipped to base -> GDD = 0.0
        gdd = calculate_daily_gdd(np.array([8.0]), np.array([2.0]), t_base=10.0, t_upper=35.0)
        self.assertAlmostEqual(float(gdd[0]), 0.0, places=2)

    def test_gdd_above_upper_cutoff(self):
        # T_max = 42, T_min = 24, T_base = 10, T_upper = 35
        # T_eff_max = 35, T_eff_min = 24, T_mean = 29.5 -> GDD = 19.5
        gdd = calculate_daily_gdd(np.array([42.0]), np.array([24.0]), t_base=10.0, t_upper=35.0)
        self.assertAlmostEqual(float(gdd[0]), 19.5, places=2)

    def test_vpd_saturated_air(self):
        # 100% RH should result in VPD = 0.0
        vpd = calculate_vapor_pressure_deficit(np.array([30.0]), np.array([20.0]), np.array([100.0]))
        self.assertAlmostEqual(float(vpd[0]), 0.0, places=2)

    def test_vpd_positive_deficit(self):
        # T_max = 35.0, T_min = 25.0, RH = 50.0% -> Non-zero positive VPD
        vpd = calculate_vapor_pressure_deficit(np.array([35.0]), np.array([25.0]), np.array([50.0]))
        self.assertGreater(float(vpd[0]), 1.0)

    def test_consecutive_dry_days(self):
        # 3 dry, 2 wet, 5 dry, 1 wet
        precip = pd.Series([0.0, 0.5, 0.0, 15.0, 30.0, 0.0, 0.0, 0.0, 0.2, 0.0, 20.0])
        max_dry = calculate_max_consecutive_dry_days(precip, dry_threshold=1.0)
        self.assertEqual(max_dry, 5)

        # All wet days -> 0 consecutive dry days
        precip_wet = pd.Series([10.0, 15.0, 25.0])
        self.assertEqual(calculate_max_consecutive_dry_days(precip_wet, dry_threshold=1.0), 0)


class TestPhenologicalWindowExtraction(unittest.TestCase):
    """Test suite for seasonal extraction and temporal boundary handling."""

    def setUp(self):
        # Generate 2 years of daily weather (2022 and 2023)
        dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
        self.weather_df = pd.DataFrame({
            "district_id": ["DIST_101"] * len(dates),
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "temp_max": np.random.uniform(25, 35, len(dates)),
            "temp_min": np.random.uniform(15, 22, len(dates)),
            "precipitation_mm": np.random.uniform(0, 10, len(dates)),
            "relative_humidity": np.random.uniform(40, 80, len(dates)),
            "solar_radiation_mj": np.random.uniform(15, 25, len(dates)),
        })

    def test_kharif_crop_window(self):
        # Rice active in months [6, 7, 8, 9, 10, 11] of harvest year 2023
        config = DEFAULT_PHENOLOGY_PROFILES["Rice"]
        window = extract_phenological_window(self.weather_df, "DIST_101", 2023, config)
        self.assertTrue((window["month"].isin([6, 7, 8, 9, 10, 11])).all())
        self.assertTrue((window["year"] == 2023).all())
        self.assertEqual(len(window), 30 + 31 + 31 + 30 + 31 + 30)  # Jun-Nov = 183 days

    def test_cross_year_wheat_window(self):
        # Wheat active in months [11, 12] of 2022 and [1, 2, 3, 4] of 2023
        config = DEFAULT_PHENOLOGY_PROFILES["Wheat"]
        window = extract_phenological_window(self.weather_df, "DIST_101", 2023, config)
        self.assertFalse(window.empty)

        months_in_window = set(window["month"].unique())
        self.assertEqual(months_in_window, {1, 2, 3, 4, 11, 12})
        # Check Nov & Dec are from 2022
        nov_dec = window[window["month"].isin([11, 12])]
        self.assertTrue((nov_dec["year"] == 2022).all())
        # Check Jan-Apr are from 2023
        jan_apr = window[window["month"].isin([1, 2, 3, 4])]
        self.assertTrue((jan_apr["year"] == 2023).all())


class TestWeatherAggregatorIntegration(unittest.TestCase):
    """Integration test suite for phenological weather aggregation pipeline."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.interim_dir = Path(self.temp_dir) / "interim"
        self.processed_dir = Path(self.temp_dir) / "processed"
        self.report_dir = Path(self.temp_dir) / "reports"
        self.interim_dir.mkdir(parents=True)

        # Copy existing clean datasets
        shutil.copy("data/interim/clean_weather_daily.csv", self.interim_dir / "clean_weather_daily.csv")
        shutil.copy("data/interim/clean_historical_yields.csv", self.interim_dir / "clean_historical_yields.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_weather_aggregation_pipeline(self):
        results = run_weather_aggregation_pipeline(
            clean_weather_path=str(self.interim_dir / "clean_weather_daily.csv"),
            clean_yields_path=str(self.interim_dir / "clean_historical_yields.csv"),
            output_dir=str(self.processed_dir),
            report_dir=str(self.report_dir)
        )

        # Verify output files exist
        pheno_file = self.processed_dir / "weather_aggregated.csv"
        annual_file = self.processed_dir / "weather_aggregated_annual.csv"
        json_report = self.report_dir / "weather_aggregation_report.json"
        md_report = self.report_dir / "weather_aggregation_report.md"

        self.assertTrue(pheno_file.exists())
        self.assertTrue(annual_file.exists())
        self.assertTrue(json_report.exists())
        self.assertTrue(md_report.exists())

        # Load and verify phenological aggregations
        pheno_df = pd.read_csv(pheno_file)
        self.assertEqual(len(pheno_df), 225)
        self.assertEqual(pheno_df.isnull().sum().sum(), 0)

        # Check required columns
        required_cols = [
            "district_id", "year", "crop_type", "cumulative_rainfall_mm",
            "gdd_cumulative", "mean_temp_max", "heat_stress_days",
            "mean_vpd_kpa", "consecutive_dry_days_max"
        ]
        for col in required_cols:
            self.assertIn(col, pheno_df.columns)

        # Verify agronomic values are physically reasonable
        self.assertTrue((pheno_df["cumulative_rainfall_mm"] >= 0.0).all())
        self.assertTrue((pheno_df["gdd_cumulative"] > 0.0).all())
        self.assertTrue((pheno_df["mean_temp_max"] > 10.0).all())
        self.assertTrue((pheno_df["heat_stress_days"] >= 0).all())
        self.assertTrue((pheno_df["mean_vpd_kpa"] >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
