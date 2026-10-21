"""
Unit and Integration Tests for Day 11 Production Inference Engine & REST API
Member 1 / Production ML Service Deliverable
"""

import io
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.inference import CropYieldInferenceEngine
from src.api_service import InferenceRequestHandler


class DummyHTTPHandler(InferenceRequestHandler):
    """
    Subclass of InferenceRequestHandler to mock socket connections for deterministic testing.
    """
    def __init__(self, method: str, path: str, body: Optional[Dict] = None):
        self.rfile = io.BytesIO(json.dumps(body).encode("utf-8") if body else b"")
        self.wfile = io.BytesIO()
        self.headers = {
            "Content-Length": str(len(self.rfile.getvalue())),
            "Content-Type": "application/json"
        }
        self.path = path
        self.command = method
        self.response_status = None
        self.response_headers = {}

    def send_response(self, code, message=None):
        self.response_status = code

    def send_header(self, keyword, value):
        self.response_headers[keyword] = value

    def end_headers(self):
        pass

    def get_json_response(self) -> Dict:
        self.wfile.seek(0)
        raw = self.wfile.read().decode("utf-8")
        return json.loads(raw)


class TestInferenceEngineDay11(unittest.TestCase):
    """Test suite covering the Day 11 Production Inference Engine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = CropYieldInferenceEngine("models/best_yield_regressor.joblib")

    def test_01_engine_initialization(self):
        """Verify model loads with correct feature count and performance metrics."""
        self.assertIsNotNone(self.engine.model)
        self.assertEqual(len(self.engine.feature_names), 57)
        self.assertIn("R2", self.engine.spatial_metrics)
        self.assertGreaterEqual(self.engine.spatial_metrics["R2"], 0.75)

    def test_02_feature_vector_alignment(self):
        """Verify minimal raw farm inputs generate exactly 57 strictly aligned columns."""
        raw_farm = {
            "crop_type": "Wheat",
            "soil_type": "Alluvial",
            "district_id": "DIST_101",
            "planted_area_ha": 60.0,
            "nitrogen_kg_ha": 115.0,
            "phosphorus_kg_ha": 48.0,
            "potassium_kg_ha": 45.0,
            "soil_ph": 6.4,
            "organic_matter_pct": 1.7
        }
        df_vec = self.engine.build_feature_vector(raw_farm)
        self.assertEqual(df_vec.shape, (1, 57))
        self.assertListEqual(df_vec.columns.tolist(), self.engine.feature_names)
        self.assertEqual(df_vec.isnull().sum().sum(), 0)

    def test_03_single_farm_prediction_and_intervals(self):
        """Verify point yield prediction and 95% confidence intervals adhere to boundaries."""
        raw_farm = {
            "crop_type": "Wheat",
            "planted_area_ha": 50.0,
            "nitrogen_kg_ha": 120.0,
            "phosphorus_kg_ha": 50.0,
            "potassium_kg_ha": 50.0,
            "soil_ph": 6.5,
            "organic_matter_pct": 2.0
        }
        result = self.engine.predict_single(raw_farm)

        pred_yield = result["predicted_yield_mt_ha"]
        ci_lower = result["confidence_interval_95"]["lower_bound_mt_ha"]
        ci_upper = result["confidence_interval_95"]["upper_bound_mt_ha"]
        std_err = result["uncertainty_std_mt_ha"]

        # Agronomic range checks for Wheat
        self.assertGreaterEqual(pred_yield, 2.0)
        self.assertLessEqual(pred_yield, 6.5)

        # Interval integrity: lower <= pred <= upper
        self.assertLessEqual(ci_lower, pred_yield)
        self.assertGreaterEqual(ci_upper, pred_yield)
        self.assertGreater(std_err, 0.0)

        # Production consistency: area * yield
        expected_prod = result["harvest_production_mt"]["expected_mt"]
        self.assertAlmostEqual(expected_prod, pred_yield * 50.0, places=1)

    def test_04_prescriptive_advisory_soil_acidity(self):
        """Verify severely acidic soil triggers limestone prescription with positive upside."""
        acidic_farm = {
            "crop_type": "Maize",
            "soil_ph": 5.1,
            "nitrogen_kg_ha": 80.0,
            "phosphorus_kg_ha": 25.0,
            "potassium_kg_ha": 35.0,
            "organic_matter_pct": 1.4
        }
        result = self.engine.predict_single(acidic_farm)

        prescriptions = result["prescriptive_advisory"]["prescriptions"]
        self.assertGreaterEqual(len(prescriptions), 1)

        # First prescription should address acidity or nutrient deficiency
        actions = [p["action"] for p in prescriptions]
        self.assertTrue(any("Limestone" in a or "Nitrogen" in a for a in actions))

        # Expected gain should be positive
        top_gain = prescriptions[0]["expected_gain_pct"]
        self.assertGreater(top_gain, 0.0)

    def test_05_risk_assessment_classification(self):
        """Verify environmental risk assessment flags high stress conditions."""
        stressed_farm = {
            "crop_type": "Cotton",
            "soil_ph": 4.8,  # Severe acidity
            "consecutive_dry_days_max": 28.0,  # Severe drought
            "heat_stress_days": 14.0,  # Heat shock
            "cumulative_rainfall_mm": 250.0
        }
        result = self.engine.predict_single(stressed_farm)
        risk = result["risk_assessment"]

        self.assertIn(risk["level"], ["HIGH", "CRITICAL"])
        self.assertGreaterEqual(len(risk["risk_factors"]), 2)

    def test_06_batch_inference_execution(self):
        """Verify high-throughput batch inference over multiple farm inputs."""
        farms = [
            {"crop_type": "Wheat", "soil_ph": 6.5, "nitrogen_kg_ha": 120.0, "planted_area_ha": 40.0},
            {"crop_type": "Rice", "soil_ph": 6.2, "nitrogen_kg_ha": 110.0, "planted_area_ha": 55.0},
            {"crop_type": "Maize", "soil_ph": 6.8, "nitrogen_kg_ha": 130.0, "planted_area_ha": 30.0},
            {"crop_type": "Soybean", "soil_ph": 6.4, "nitrogen_kg_ha": 90.0, "planted_area_ha": 25.0},
            {"crop_type": "Cotton", "soil_ph": 7.1, "nitrogen_kg_ha": 100.0, "planted_area_ha": 50.0},
        ]
        df_batch = self.engine.predict_batch(farms)

        self.assertEqual(len(df_batch), 5)
        self.assertIn("predicted_yield_mt_ha", df_batch.columns)
        self.assertIn("yield_ci_lower_mt_ha", df_batch.columns)
        self.assertIn("yield_ci_upper_mt_ha", df_batch.columns)
        self.assertIn("expected_production_mt", df_batch.columns)
        self.assertIn("risk_level", df_batch.columns)
        self.assertIn("top_prescriptive_action", df_batch.columns)
        self.assertEqual(df_batch.isnull().sum().sum(), 0)

    def test_07_api_health_endpoint(self):
        """Verify REST API GET /health returns status 200 and healthy metadata."""
        handler = DummyHTTPHandler(method="GET", path="/health")
        handler.do_GET()

        self.assertEqual(handler.response_status, 200)
        data = handler.get_json_response()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["features_count"], 57)
        self.assertIn("spatial_r2_score", data)

    def test_08_api_predict_endpoint(self):
        """Verify REST API POST /api/v1/predict returns point inference and intervals."""
        farm_body = {
            "crop_type": "Rice",
            "nitrogen_kg_ha": 110.0,
            "soil_ph": 6.3,
            "planted_area_ha": 45.0
        }
        handler = DummyHTTPHandler(method="POST", path="/api/v1/predict", body=farm_body)
        handler.do_POST()

        self.assertEqual(handler.response_status, 200)
        data = handler.get_json_response()
        self.assertEqual(data["crop_type"], "Rice")
        self.assertIn("predicted_yield_mt_ha", data)
        self.assertIn("confidence_interval_95", data)

    def test_09_api_prescribe_endpoint(self):
        """Verify REST API POST /api/v1/prescribe returns full predictions + advisory."""
        farm_body = {
            "crop_type": "Wheat",
            "soil_ph": 5.3,
            "nitrogen_kg_ha": 85.0
        }
        handler = DummyHTTPHandler(method="POST", path="/api/v1/prescribe", body=farm_body)
        handler.do_POST()

        self.assertEqual(handler.response_status, 200)
        data = handler.get_json_response()
        self.assertIn("prescriptive_advisory", data)
        self.assertGreaterEqual(len(data["prescriptive_advisory"]["prescriptions"]), 1)


if __name__ == "__main__":
    unittest.main()
