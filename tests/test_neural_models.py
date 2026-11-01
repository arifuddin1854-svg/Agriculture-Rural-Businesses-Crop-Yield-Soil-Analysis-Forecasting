"""
Unit and Integration Tests for Day 15 Tabular Neural Regressor Benchmark
Member 1 / Production ML Modeling Engine
"""

import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.neural_models import TabularNeuralPipeline


class TestNeuralModels(unittest.TestCase):
    """Test suite covering the Day 15 Tabular Deep Neural Regressor Pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = TabularNeuralPipeline()
        cls.X, cls.y, cls.groups, cls.years, cls.features = cls.pipeline.load_data()

    def test_01_data_loading_and_shape(self):
        """Verify feature dimensions, target column extraction, and group labels."""
        self.assertEqual(len(self.X), 225)
        self.assertEqual(len(self.y), 225)
        self.assertEqual(len(np.unique(self.groups)), 15)
        self.assertGreaterEqual(len(self.features), 50)

    def test_02_spatial_cv_execution(self):
        """Verify 2-fold spatial evaluation returns valid R2 and RMSE metrics."""
        # Use 2 splits for rapid deterministic testing
        metrics = self.pipeline.evaluate_spatial_cv(self.X, self.y, self.groups, n_splits=2)

        self.assertIn("R2", metrics["spatial_metrics"])
        self.assertIn("RMSE", metrics["spatial_metrics"])
        self.assertGreater(metrics["spatial_metrics"]["R2"], 0.15)
        self.assertLess(metrics["spatial_metrics"]["RMSE"], 1.60)

    def test_03_neural_checkpoint_serialization(self):
        """Verify champion neural model checkpoint saves with scaler and feature schema."""
        chk_path = Path("models/neural_yield_regressor.joblib")
        self.assertTrue(chk_path.exists())

        chk = joblib.load(chk_path)
        self.assertEqual(chk["model_name"], "Tabular_Neural_Network_MLP")
        self.assertIn("scaler", chk)
        self.assertIn("architecture", chk)
        self.assertIsInstance(chk["model"], MLPRegressor)

        # Test inference with saved model and scaler
        X_test = self.X.iloc[:5]
        X_scaled = chk["scaler"].transform(X_test)
        preds = chk["model"].predict(X_scaled)
        self.assertEqual(len(preds), 5)
        self.assertTrue(np.all(preds > 0))


if __name__ == "__main__":
    unittest.main()
