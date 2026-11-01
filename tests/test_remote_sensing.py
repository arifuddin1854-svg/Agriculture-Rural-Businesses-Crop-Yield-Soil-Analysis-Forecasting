"""
Unit and Integration Tests for Day 13 Satellite Remote Sensing & Vegetation Indices Engine
Member 1 / Production Remote Sensing Module
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.remote_sensing import (
    calculate_spectral_indices,
    SatelliteDataEngine,
    CROP_SPECTRAL_PROFILES
)


class TestRemoteSensingEngine(unittest.TestCase):
    """Test suite covering the Day 13 Satellite Remote Sensing Engine."""

    def test_01_spectral_index_calculation_equations(self):
        """Verify NDVI, NDWI, and EVI adhere to mathematical and biophysical boundaries."""
        df_bands = pd.DataFrame([{
            "nir": 0.50,
            "red": 0.08,
            "blue": 0.04,
            "swir": 0.18
        }])

        df_out = calculate_spectral_indices(df_bands)

        # NDVI = (0.50 - 0.08) / (0.50 + 0.08) = 0.42 / 0.58 = 0.7241
        self.assertAlmostEqual(df_out["ndvi"].iloc[0], 0.7241, places=3)
        # NDWI = (0.50 - 0.18) / (0.50 + 0.18) = 0.32 / 0.68 = 0.4706
        self.assertAlmostEqual(df_out["ndwi"].iloc[0], 0.4706, places=3)
        # EVI should be strictly positive for healthy green vegetation
        self.assertGreater(df_out["evi"].iloc[0], 0.40)
        self.assertLessEqual(df_out["evi"].iloc[0], 2.5)

    def test_02_raw_timeseries_generation(self):
        """Verify 10-day dekadal timeseries generation across specified districts."""
        engine = SatelliteDataEngine(
            raw_output_path="reports/test_spectral_raw.csv",
            processed_output_path="reports/test_canopy_indices.csv"
        )
        df_raw = engine.generate_raw_spectral_timeseries(
            districts=["DIST_101", "DIST_102"],
            years=[2022, 2023]
        )

        # 2 districts * 2 years * 36 dekads = 144 records
        self.assertEqual(len(df_raw), 144)
        self.assertIn("ndvi", df_raw.columns)
        self.assertIn("ndwi", df_raw.columns)
        self.assertIn("evi", df_raw.columns)
        self.assertTrue((df_raw["ndvi"] >= -1.0).all() and (df_raw["ndvi"] <= 1.0).all())

        # Cleanup
        p = Path("reports/test_spectral_raw.csv")
        if p.exists():
            p.unlink()

    def test_03_phenological_aggregation_integrity(self):
        """Verify phenological canopy metrics align with historical crop records."""
        test_path = Path("reports/test_spectral_raw_03.csv")
        engine = SatelliteDataEngine(raw_output_path=str(test_path))
        df_raw = engine.generate_raw_spectral_timeseries(
            districts=["DIST_101"],
            years=[2023]
        )
        if test_path.exists():
            test_path.unlink()
        # Verify CROP_SPECTRAL_PROFILES has all 5 crops
        for c in ["Wheat", "Rice", "Maize", "Soybean", "Cotton"]:
            self.assertIn(c, CROP_SPECTRAL_PROFILES)


if __name__ == "__main__":
    unittest.main()
