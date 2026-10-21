"""
Unit Tests for Spatial Cross-Validation and Regression Modeling Suite
Day 9 Milestone
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from src.models import (
    compute_metrics,
    evaluate_spatial_cv,
    run_spatial_modeling_pipeline
)
from sklearn.linear_model import Ridge


class TestSpatialModelingMetrics(unittest.TestCase):
    """Test suite verifying metric calculations and spatial splits."""

    def test_compute_metrics_perfect(self):
        y_true = np.array([4.0, 5.0, 6.0])
        y_pred = np.array([4.0, 5.0, 6.0])
        scores = compute_metrics(y_true, y_pred)
        self.assertEqual(scores["RMSE"], 0.0)
        self.assertEqual(scores["MAE"], 0.0)
        self.assertEqual(scores["MAPE (%)"], 0.0)
        self.assertEqual(scores["R2"], 1.0)

    def test_spatial_cv_partitioning(self):
        # 10 records across 2 districts
        X = pd.DataFrame({"feat1": np.arange(10), "feat2": np.arange(10) * 2})
        y = pd.Series(np.arange(10) * 1.5)
        districts = pd.Series(["D1", "D1", "D1", "D1", "D1", "D2", "D2", "D2", "D2", "D2"])

        models = {"Ridge": Ridge(alpha=1.0)}
        benchmark, oof = evaluate_spatial_cv(X, y, districts, models, n_splits=2)
        self.assertIn("Ridge", benchmark)
        self.assertEqual(len(oof["Ridge"]), 10)


class TestModelingPipelineExecution(unittest.TestCase):
    """Integration test verifying end-to-end pipeline execution with model checkpointing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.features_file = Path(self.temp_dir) / "test_features.csv"
        self.models_dir = Path(self.temp_dir) / "models"
        self.figures_dir = Path(self.temp_dir) / "figures"
        self.reports_dir = Path(self.temp_dir) / "reports"

        shutil.copy("data/processed/train_features.csv", self.features_file)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_spatial_modeling_pipeline(self):
        results = run_spatial_modeling_pipeline(
            features_path=str(self.features_file),
            models_dir=str(self.models_dir),
            figures_dir=str(self.figures_dir),
            report_dir=str(self.reports_dir)
        )
        self.assertTrue(results["checkpoint"].exists())
        self.assertTrue(results["json_report"].exists())
        self.assertTrue(results["md_report"].exists())

        checkpoint = joblib.load(results["checkpoint"])
        self.assertIn("model", checkpoint)
        self.assertIn("feature_names", checkpoint)
        self.assertIn("spatial_metrics", checkpoint)
        self.assertGreater(checkpoint["spatial_metrics"]["R2"], 0.75)


if __name__ == "__main__":
    unittest.main()
