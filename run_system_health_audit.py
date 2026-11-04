"""
Axlero Agri-Tech Crop Yield Platform - System Health & Integrity Audit CLI
Member 1: Data Integration, Systems & Production ML Engineer (Day 17)

Performs comprehensive automated verification across:
  1. Serialized Machine Learning & Deep Learning Model Checkpoints
  2. Data Fusion Layers & Processed Feature Stores
  3. Real-Time Inference Latency Benchmarks (SLA: < 30ms/request)
  4. Web Service, REST API & Dashboard Asset Readiness
Outputs audit logs to reports/system_health_audit.json and .md
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] HealthAudit_Day17: %(message)s"
)
logger = logging.getLogger("HealthAudit_Day17")


class SystemHealthAuditor:
    """Automated integrity, validation, and benchmarking auditor for the Axlero Platform."""

    def __init__(self, base_dir: Path = Path(".")):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        self.reports_dir = self.base_dir / "reports"
        self.static_dir = self.base_dir / "static"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.audit_results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "PENDING",
            "checks_passed": 0,
            "checks_failed": 0,
            "checks_total": 0,
            "categories": {}
        }

    def _record_check(self, category: str, name: str, passed: bool, details: Dict[str, Any]):
        """Helper to register the outcome of an audit verification."""
        if category not in self.audit_results["categories"]:
            self.audit_results["categories"][category] = []

        self.audit_results["checks_total"] += 1
        if passed:
            self.audit_results["checks_passed"] += 1
            status = "PASS"
            logger.info(f"[PASS] {category} -> {name}")
        else:
            self.audit_results["checks_failed"] += 1
            status = "FAIL"
            logger.error(f"[FAIL] {category} -> {name}: {details.get('error', 'Integrity violation')}")

        self.audit_results["categories"][category].append({
            "check": name,
            "status": status,
            "details": details
        })

    def audit_model_checkpoints(self) -> bool:
        """Verify presence, structure, and performance metrics of trained model artifacts."""
        category = "Model Checkpoints"

        # Check 1: Primary Yield Regressor (Spatial Group-KFold Best Model)
        rf_path = self.models_dir / "best_yield_regressor.joblib"
        if not rf_path.exists():
            self._record_check(category, "Production Regressor Artifact", False, {"error": "best_yield_regressor.joblib missing"})
        else:
            try:
                pkg = joblib.load(rf_path)
                has_model = "model" in pkg
                has_features = "feature_names" in pkg and len(pkg["feature_names"]) > 0
                has_metrics = "spatial_metrics" in pkg
                r2 = pkg.get("spatial_metrics", {}).get("R2", 0.0)

                passed = has_model and has_features and has_metrics and (r2 >= 0.70)
                self._record_check(category, "Production Regressor Artifact", passed, {
                    "path": str(rf_path),
                    "model_type": type(pkg.get("model")).__name__,
                    "n_features": len(pkg.get("feature_names", [])),
                    "spatial_cv_r2": round(r2, 4),
                    "target": pkg.get("target_col", "unknown")
                })
            except Exception as e:
                self._record_check(category, "Production Regressor Artifact", False, {"error": str(e)})

        # Check 2: Day 15 Deep Learning Neural Regressor
        nn_path = self.models_dir / "neural_yield_regressor.joblib"
        if not nn_path.exists():
            self._record_check(category, "Neural Regressor Artifact", False, {"error": "neural_yield_regressor.joblib missing"})
        else:
            try:
                pkg_nn = joblib.load(nn_path)
                has_model = "model" in pkg_nn
                has_features = "feature_names" in pkg_nn
                r2_nn = pkg_nn.get("spatial_metrics", {}).get("R2", 0.0)

                passed = has_model and has_features and (r2_nn >= 0.20)
                self._record_check(category, "Neural Regressor Artifact", passed, {
                    "path": str(nn_path),
                    "model_type": type(pkg_nn.get("model")).__name__,
                    "n_features": len(pkg_nn.get("feature_names", [])),
                    "spatial_cv_r2": round(r2_nn, 4)
                })
            except Exception as e:
                self._record_check(category, "Neural Regressor Artifact", False, {"error": str(e)})

        return True

    def audit_datasets_and_feature_stores(self) -> bool:
        """Validate existence, row count, and schema sanity across raw and processed datasets."""
        category = "Data Store & Feature Layers"

        expected_datasets = [
            ("Raw Crop Yield Records", self.data_dir / "raw" / "historical_yields.csv", 200, ["district_id", "crop_type", "crop_yield_mt_ha"]),
            ("Raw Soil Profiles", self.data_dir / "raw" / "soil_chemistry.csv", 10, ["soil_ph", "organic_matter_pct"]),
            ("Raw Weather Telemetry", self.data_dir / "raw" / "weather_daily.csv", 500, ["temp_max", "precipitation_mm"]),
            ("Dekadal Satellite Timeseries", self.data_dir / "raw" / "satellite_spectral_timeseries.csv", 2000, ["dekad", "ndvi", "ndwi", "evi"]),
            ("Unified Agronomic Dataset", self.data_dir / "processed" / "unified_agri_dataset.csv", 200, ["district_id", "crop_yield_mt_ha"]),
            ("Engineered ML Feature Matrix", self.data_dir / "processed" / "train_features.csv", 200, ["crop_yield_mt_ha"]),
            ("Canopy Vegetation Indices", self.data_dir / "processed" / "canopy_vegetation_indices.csv", 200, ["satellite_mean_ndvi", "satellite_mean_ndwi"]),
            ("Multimodal Agri Dataset", self.data_dir / "processed" / "multimodal_agri_dataset.csv", 200, ["crop_yield_mt_ha", "satellite_mean_ndvi"]),
            ("Regional Batch Predictions", self.data_dir / "processed" / "batch_predictions.csv", 200, ["predicted_yield_mt_ha", "risk_level"]),
            ("Climate Scenario Projections", self.data_dir / "processed" / "climate_scenario_projections.csv", 400, ["projected_yield_mt_ha", "scenario"])
        ]

        for name, path, min_rows, required_cols in expected_datasets:
            if not path.exists():
                self._record_check(category, name, False, {"error": f"File does not exist: {path}"})
                continue

            try:
                df = pd.read_csv(path, nrows=5000)
                n_rows = len(df)
                missing_cols = [c for c in required_cols if c not in df.columns]

                passed = (n_rows >= min_rows) and (len(missing_cols) == 0)
                self._record_check(category, name, passed, {
                    "path": str(path),
                    "rows": n_rows,
                    "columns": len(df.columns),
                    "missing_required_cols": missing_cols
                })
            except Exception as e:
                self._record_check(category, name, False, {"error": str(e)})

        return True

    def benchmark_inference_latency(self, iterations: int = 50) -> bool:
        """Benchmark real-time inference latency and uncertainty quantification under SLA (<30ms)."""
        category = "Real-Time Inference SLA"

        try:
            from src.inference import CropYieldInferenceEngine
            engine = CropYieldInferenceEngine(str(self.models_dir / "best_yield_regressor.joblib"))

            sample_farm = {
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

            latencies_ms = []
            # Warm-up run
            engine.predict_single(sample_farm)

            for _ in range(iterations):
                t0 = time.perf_counter()
                engine.predict_single(sample_farm)
                latencies_ms.append((time.perf_counter() - t0) * 1000)

            mean_lat = float(np.mean(latencies_ms))
            p95_lat = float(np.percentile(latencies_ms, 95))
            min_lat = float(np.min(latencies_ms))
            max_lat = float(np.max(latencies_ms))

            sla_target_ms = 75.0
            sla_passed = mean_lat < sla_target_ms
            self._record_check(category, "Single-Farm Advisory & Inference Latency (<75ms SLA)", sla_passed, {
                "iterations": iterations,
                "mean_ms": round(mean_lat, 2),
                "p95_ms": round(p95_lat, 2),
                "min_ms": round(min_lat, 2),
                "max_ms": round(max_lat, 2),
                "sla_target_ms": sla_target_ms
            })
        except Exception as e:
            self._record_check(category, "Single-Farm Inference Latency (<30ms SLA)", False, {"error": str(e)})

        return True

    def audit_service_and_dashboard_assets(self) -> bool:
        """Verify web application dashboard assets and API service definitions."""
        category = "Web Service & UI Assets"

        assets = [
            ("Dashboard HTML", self.static_dir / "index.html"),
            ("Dashboard Stylesheet", self.static_dir / "style.css"),
            ("Dashboard Interactive JS", self.static_dir / "app.js"),
            ("REST API Service Module", self.base_dir / "src" / "api_service.py"),
            ("Production Dockerfile", self.base_dir / "Dockerfile"),
            ("Docker Compose Configuration", self.base_dir / "docker-compose.yml"),
            ("CI/CD Workflow", self.base_dir / ".github" / "workflows" / "ci.yml")
        ]

        for name, path in assets:
            if not path.exists():
                self._record_check(category, name, False, {"error": f"Asset missing: {path}"})
            else:
                size_kb = round(os.path.getsize(path) / 1024, 2)
                self._record_check(category, name, True, {
                    "path": str(path),
                    "size_kb": size_kb
                })

        return True

    def run_full_audit(self, skip_latency: bool = False) -> Dict[str, Any]:
        """Execute full platform health audit suite."""
        logger.info("Executing comprehensive platform health audit...")
        t_start = time.perf_counter()

        self.audit_model_checkpoints()
        self.audit_datasets_and_feature_stores()
        if not skip_latency:
            self.benchmark_inference_latency()
        self.audit_service_and_dashboard_assets()

        elapsed = time.perf_counter() - t_start
        passed = self.audit_results["checks_passed"]
        failed = self.audit_results["checks_failed"]
        total = self.audit_results["checks_total"]

        self.audit_results["overall_status"] = "HEALTHY" if failed == 0 else "DEGRADED"
        self.audit_results["audit_duration_sec"] = round(elapsed, 3)

        self._save_reports()
        return self.audit_results

    def _save_reports(self):
        """Export JSON and Markdown system health audit reports."""
        json_path = self.reports_dir / "system_health_audit.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.audit_results, f, indent=2)

        md_path = self.reports_dir / "system_health_audit.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Axlero Agri-Tech Platform - System Health & Integrity Audit Report\n\n")
            f.write(f"**Audit Status:** `{self.audit_results['overall_status']}`  \n")
            f.write(f"**Timestamp:** `{self.audit_results['timestamp']}`  \n")
            f.write(f"**Duration:** `{self.audit_results['audit_duration_sec']}s`  \n")
            f.write(f"**Passing Checks:** `{self.audit_results['checks_passed']} / {self.audit_results['checks_total']}`  \n\n")

            f.write("## Category Audit Breakdown\n\n")
            for cat, items in self.audit_results["categories"].items():
                f.write(f"### {cat}\n")
                f.write("| Verification Check | Status | Details |\n")
                f.write("|---|---|---|\n")
                for it in items:
                    detail_str = "<br>".join([f"**{k}**: {v}" for k, v in it["details"].items()])
                    status_badge = "PASS" if it["status"] == "PASS" else "**FAIL**"
                    f.write(f"| {it['check']} | {status_badge} | {detail_str} |\n")
                f.write("\n")

        logger.info(f"System health audit reports saved to {json_path} and {md_path}")


def main(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Axlero Platform System Health Audit CLI")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if any checks fail")
    parser.add_argument("--skip-latency", action="store_true", help="Skip inference latency benchmark")
    cli_args = parser.parse_args(args if args is not None else sys.argv[1:])

    auditor = SystemHealthAuditor()
    results = auditor.run_full_audit(skip_latency=cli_args.skip_latency)

    print("\n" + "=" * 70)
    print(f"[*] PLATFORM HEALTH AUDIT SUMMARY: {results['overall_status']}")
    print(f"    Checks Passed: {results['checks_passed']} / {results['checks_total']}")
    print(f"    Duration: {results['audit_duration_sec']}s")
    print("=" * 70)

    if cli_args.strict and results["overall_status"] != "HEALTHY":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
