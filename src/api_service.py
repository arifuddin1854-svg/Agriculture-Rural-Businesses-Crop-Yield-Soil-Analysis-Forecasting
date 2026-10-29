"""
Day 11: Production REST API Service & Real-Time Model Serving
Member 1 / Production ML Service Deliverable

Responsible for:
- Providing real-time HTTP REST endpoints for single-farm yield inference and soil advisory
- Dual compatibility: Native Python HTTP server (zero external dependencies) + optional FastAPI adapter
- Endpoints:
  * GET  /health -> Service health, model status, and spatial CV benchmarks
  * GET  /model/metadata -> Full feature list and cross-validation performance metrics
  * POST /api/v1/predict -> Real-time yield inference with 95% confidence interval
  * POST /api/v1/prescribe -> Comprehensive yield prediction + counterfactual prescriptive soil advisory
  * POST /api/v1/batch -> Multi-farm batch predictions
"""

import argparse
from datetime import datetime
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.inference import CropYieldInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("APIService_Day11")


class InferenceRequestHandler(BaseHTTPRequestHandler):
    """
    Standard library HTTP handler delivering production REST endpoints for crop yield forecasting.
    """
    engine: Optional[CropYieldInferenceEngine] = None

    @classmethod
    def get_engine(cls) -> CropYieldInferenceEngine:
        if cls.engine is None:
            cls.engine = CropYieldInferenceEngine()
        return cls.engine

    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw_body = self.rfile.read(content_length).decode("utf-8")
        return json.loads(raw_body)

    def _send_json(self, data: Any, status_code: int = 200):
        self._set_headers(status_code)
        payload = json.dumps(data, indent=2).encode("utf-8")
        self.wfile.write(payload)

    def _send_error(self, message: str, status_code: int = 400):
        self._send_json({"error": True, "status_code": status_code, "message": message}, status_code=status_code)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        engine = self.get_engine()

        if path in ["", "/health", "/api/health"]:
            response = {
                "status": "healthy",
                "service": "Crop Yield Forecasting & Soil Advisory API",
                "version": "1.1.0",
                "timestamp": datetime.now().isoformat(),
                "model_champion": engine.model_name,
                "features_count": len(engine.feature_names),
                "spatial_r2_score": engine.spatial_metrics.get("R2", 0.8033),
                "spatial_rmse_mt_ha": engine.spatial_metrics.get("RMSE", 0.5882)
            }
            self._send_json(response, 200)

        elif path in ["/model/metadata", "/api/v1/model"]:
            response = {
                "model_name": engine.model_name,
                "training_timestamp": engine.timestamp,
                "feature_count": len(engine.feature_names),
                "feature_schema": engine.feature_names,
                "spatial_group_kfold_metrics": engine.spatial_metrics,
                "temporal_loyo_metrics": engine.temporal_metrics,
                "supported_crops": engine.SUPPORTED_CROPS,
                "supported_soils": engine.SUPPORTED_SOILS
            }
            self._send_json(response, 200)

        else:
            self._send_error(f"Endpoint '{path}' not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        engine = self.get_engine()

        try:
            body = self._read_json_body()
        except Exception as e:
            self._send_error(f"Invalid JSON payload: {str(e)}", 400)
            return

        try:
            if path in ["/api/v1/predict", "/predict"]:
                if not body:
                    self._send_error("Request body cannot be empty", 400)
                    return
                result = engine.predict_single(body)
                # Omit prescriptions from basic predict endpoint
                result_clean = {
                    "crop_type": result["crop_type"],
                    "planted_area_ha": result["planted_area_ha"],
                    "predicted_yield_mt_ha": result["predicted_yield_mt_ha"],
                    "uncertainty_std_mt_ha": result["uncertainty_std_mt_ha"],
                    "confidence_interval_95": result["confidence_interval_95"],
                    "harvest_production_mt": result["harvest_production_mt"],
                    "risk_assessment": result["risk_assessment"],
                    "model_metadata": result["model_metadata"]
                }
                self._send_json(result_clean, 200)

            elif path in ["/api/v1/prescribe", "/prescribe", "/api/v1/advisor"]:
                if not body:
                    self._send_error("Request body cannot be empty", 400)
                    return
                result = engine.predict_single(body)
                self._send_json(result, 200)

            elif path in ["/api/v1/batch", "/batch"]:
                farms = body.get("farms", body)
                if not isinstance(farms, list) or len(farms) == 0:
                    self._send_error("Batch endpoint requires a non-empty list of farm objects under 'farms'", 400)
                    return
                df_out = engine.predict_batch(farms)
                batch_records = df_out.to_dict(orient="records")
                response = {
                    "total_records": len(batch_records),
                    "regional_mean_yield_mt_ha": round(float(df_out["predicted_yield_mt_ha"].mean()), 3),
                    "regional_total_production_mt": round(float(df_out["expected_production_mt"].sum()), 2),
                    "predictions": batch_records
                }
                self._send_json(response, 200)

            else:
                self._send_error(f"POST endpoint '{path}' not found", 404)

        except Exception as e:
            logger.error(f"Internal server error handling {path}: {str(e)}", exc_info=True)
            self._send_error(f"Inference error: {str(e)}", 500)


def run_api_server(host: str = "127.0.0.1", port: int = 8000):
    """
    Launches the standalone REST API server.
    """
    server_address = (host, port)
    InferenceRequestHandler.get_engine()  # Pre-warm model cache
    httpd = HTTPServer(server_address, InferenceRequestHandler)
    logger.info("=" * 75)
    logger.info(f"Day 11 Crop Yield Inference REST API running at http://{host}:{port}")
    logger.info("Available Endpoints:")
    logger.info(f"  • GET  http://{host}:{port}/health")
    logger.info(f"  • GET  http://{host}:{port}/model/metadata")
    logger.info(f"  • POST http://{host}:{port}/api/v1/predict")
    logger.info(f"  • POST http://{host}:{port}/api/v1/prescribe")
    logger.info(f"  • POST http://{host}:{port}/api/v1/batch")
    logger.info("=" * 75)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down API server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 11: Production REST API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    run_api_server(host=args.host, port=args.port)
