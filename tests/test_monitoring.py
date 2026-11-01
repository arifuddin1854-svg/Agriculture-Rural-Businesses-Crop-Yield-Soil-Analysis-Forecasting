"""
Unit and Integration Tests for Day 12 Covariate Drift Monitoring Engine & Web Dashboard
Member 1 / Production ML Service Deliverable
"""

import io
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.monitoring import ModelMonitoringEngine
from src.api_service import InferenceRequestHandler


class DummyHTTPHandler(InferenceRequestHandler):
    """
    Subclass of InferenceRequestHandler to mock socket connections for deterministic testing.
    """
    def __init__(self, method: str, path: str, body: dict = None):
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

    def get_json_response(self) -> dict:
        self.wfile.seek(0)
        raw = self.wfile.read().decode("utf-8")
        return json.loads(raw)

    def get_raw_response(self) -> bytes:
        self.wfile.seek(0)
        return self.wfile.read()


class TestMonitoringEngineDay12(unittest.TestCase):
    """Test suite covering the Day 12 Covariate Drift Monitoring Engine & Web Assets."""

    @classmethod
    def setUpClass(cls):
        cls.engine = ModelMonitoringEngine("data/processed/train_features.csv")

    def test_01_calculate_psi_identical_distribution(self):
        """Verify identical distributions yield zero or near-zero PSI (< 0.05)."""
        np.random.seed(42)
        base = np.random.normal(100, 15, size=1000)
        current = np.random.normal(100, 15, size=1000)

        psi = self.engine.calculate_psi(base, current)
        self.assertLess(psi, 0.05)

    def test_02_calculate_psi_shifted_distribution(self):
        """Verify significantly shifted distribution triggers PSI >= 0.25."""
        np.random.seed(42)
        base = np.random.normal(100, 10, size=1000)
        current = np.random.normal(140, 10, size=1000)  # Significant +40 mean shift

        psi = self.engine.calculate_psi(base, current)
        self.assertGreaterEqual(psi, 0.25)

    def test_03_ks_test_divergence(self):
        """Verify Kolmogorov-Smirnov test correctly flags divergent distributions."""
        np.random.seed(42)
        dist1 = np.random.exponential(scale=10, size=500)
        dist2 = np.random.normal(loc=50, scale=10, size=500)

        stat, pval = self.engine.calculate_ks_test(dist1, dist2)
        self.assertGreater(stat, 0.40)
        self.assertLess(pval, 0.01)

    def test_04_evaluate_covariate_drift_structure(self):
        """Verify full drift evaluation returns valid structure with expected metrics."""
        summary = self.engine.evaluate_covariate_drift("data/processed/unified_agri_dataset.csv")

        self.assertIn("health_score_pct", summary)
        self.assertIn("overall_status", summary)
        self.assertIn("features", summary)
        self.assertGreaterEqual(summary["health_score_pct"], 80.0)

        for feat_name, d in summary["features"].items():
            self.assertIn("psi", d)
            self.assertIn("ks_statistic", d)
            self.assertIn("status", d)
            self.assertIn(d["status"], ["STABLE", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"])

    def test_05_generate_drift_reports(self):
        """Verify JSON and Markdown reports are generated and contain expected headers."""
        json_rep = "reports/test_drift_report.json"
        md_rep = "reports/test_drift_report.md"

        self.engine.generate_audit_reports(
            current_data="data/processed/unified_agri_dataset.csv",
            output_json=json_rep,
            output_md=md_rep
        )

        p_json = Path(json_rep)
        p_md = Path(md_rep)
        self.assertTrue(p_json.exists())
        self.assertTrue(p_md.exists())

        # Clean up temporary test report files
        p_json.unlink()
        p_md.unlink()

    def test_06_api_serves_dashboard_html(self):
        """Verify GET / returns 200 OK and serves index.html."""
        handler = DummyHTTPHandler(method="GET", path="/")
        handler.do_GET()

        self.assertEqual(handler.response_status, 200)
        self.assertIn("text/html", handler.response_headers.get("Content-Type", ""))
        content = handler.get_raw_response().decode("utf-8")
        self.assertIn("AXLERO AGRI-TECH", content)
        self.assertIn("Farm Telemetry & Soil Profile", content)

    def test_07_api_serves_static_assets(self):
        """Verify GET /style.css and /app.js return 200 OK and correct mime types."""
        h_css = DummyHTTPHandler(method="GET", path="/style.css")
        h_css.do_GET()
        self.assertEqual(h_css.response_status, 200)
        self.assertIn("text/css", h_css.response_headers.get("Content-Type", ""))

        h_js = DummyHTTPHandler(method="GET", path="/app.js")
        h_js.do_GET()
        self.assertEqual(h_js.response_status, 200)
        self.assertIn("javascript", h_js.response_headers.get("Content-Type", ""))

    def test_08_api_monitoring_endpoint(self):
        """Verify GET /api/v1/monitoring returns 200 OK with drift scores."""
        handler = DummyHTTPHandler(method="GET", path="/api/v1/monitoring")
        handler.do_GET()

        self.assertEqual(handler.response_status, 200)
        data = handler.get_json_response()
        self.assertIn("health_score_pct", data)
        self.assertIn("overall_status", data)
        self.assertIn("features", data)


if __name__ == "__main__":
    unittest.main()
