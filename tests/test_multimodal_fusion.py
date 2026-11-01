"""
Unit and Integration Tests for Day 14 Multi-Modal Data Fusion
Member 1 / Production Multi-Modal Pipeline
"""

import unittest
from pathlib import Path
import pandas as pd

from src.multimodal_fusion import fuse_multimodal_datasets, generate_multimodal_report


class TestMultiModalFusion(unittest.TestCase):
    """Test suite covering the Day 14 Multi-Modal Data Fusion Module."""

    def test_01_referential_integrity_and_shape(self):
        """Verify multi-modal fusion executes with 100% referential integrity (225 rows)."""
        df_fused = fuse_multimodal_datasets(
            agri_dataset_path="data/processed/unified_agri_dataset.csv",
            satellite_indices_path="data/processed/canopy_vegetation_indices.csv",
            output_csv_path="reports/test_multimodal_output.csv",
            output_parquet_path="reports/test_multimodal_output.parquet"
        )

        self.assertEqual(len(df_fused), 225)
        self.assertEqual(df_fused.isnull().sum().sum(), 0)
        # Should include both original and satellite columns + interactions
        self.assertIn("satellite_peak_ndvi", df_fused.columns)
        self.assertIn("photosynthetic_radiation_efficiency", df_fused.columns)
        self.assertIn("soil_nitrogen_canopy_coupling", df_fused.columns)
        self.assertIn("canopy_hydration_deficit_index", df_fused.columns)

        # Cleanup
        for p in ["reports/test_multimodal_output.csv", "reports/test_multimodal_output.parquet"]:
            Path(p).unlink(missing_ok=True)

    def test_02_derived_interaction_metrics(self):
        """Verify derived interaction features are mathematically sound and non-negative."""
        df_fused = pd.read_csv("data/processed/multimodal_agri_dataset.csv")

        # Photosynthetic radiation efficiency should be positive
        self.assertTrue((df_fused["photosynthetic_radiation_efficiency"] > 0).all())
        # Peak NDVI should be in [0.3, 0.95] for agricultural canopies
        self.assertTrue((df_fused["satellite_peak_ndvi"] >= 0.3).all())
        self.assertTrue((df_fused["satellite_peak_ndvi"] <= 0.95).all())

    def test_03_report_metrics_emission(self):
        """Verify report generation produces non-empty structured data."""
        df_fused = pd.read_csv("data/processed/multimodal_agri_dataset.csv")
        res = generate_multimodal_report(df_fused, 0.25, report_dir="reports")

        self.assertEqual(res["total_records"], 225)
        self.assertIn("peak_ndvi_vs_yield", res["multi_modal_correlations"])
        self.assertTrue(Path("reports/multimodal_fusion_report.json").exists())
        self.assertTrue(Path("reports/multimodal_fusion_report.md").exists())


if __name__ == "__main__":
    unittest.main()
