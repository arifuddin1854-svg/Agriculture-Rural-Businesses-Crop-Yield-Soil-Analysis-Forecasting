"""
Axlero Agri-Tech Crop Yield & Soil Analysis Platform
Master Platform Orchestration & Integration CLI
Member 1: Systems & ML Infrastructure Engineer (Day 18 Deliverable)

Provides centralized, unified lifecycle orchestration across:
  - Stage 1: Data Pipeline (Ingestion, Cleaning, Weather Aggregation, Geospatial Fusion)
  - Stage 2: Feature Engineering (Biophysical features, interaction indices, lag features)
  - Stage 3: Spatial ML Modeling (Spatial Group-KFold CV, Model Selection, Serialization)
  - Stage 4: Neural Regressor Benchmark (Deep Learning MLP Tabular Regressor)
  - Stage 5: Satellite Remote Sensing & Multimodal Fusion (NDVI, NDWI, EVI, PRUE)
  - Stage 6: Regional Batch Inference & Prescriptive Advisory
  - Stage 7: Climate Scenario Simulation Engine (CMIP6 SSP2-4.5 & SSP5-8.5)
  - Stage 8: System Health & Integrity Audit (Model benchmarks, SLA latency)
"""

import argparse
import importlib
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] MasterPlatform_Day18: %(message)s"
)
logger = logging.getLogger("MasterPlatform_Day18")


PLATFORM_STAGES = [
    {
        "id": "pipeline",
        "name": "Data Ingestion & Multimodal Data Pipeline",
        "module": "run_data_pipeline",
        "function": "main",
        "description": "Clean raw data, aggregate daily weather telemetry, fuse spatial soil profiles"
    },
    {
        "id": "features",
        "name": "Biophysical & Agro-Climatic Feature Engineering",
        "module": "src.features",
        "function": "main",
        "description": "Derive GDD, heat stress index, nutrient ratios, and soil-climate interaction terms"
    },
    {
        "id": "models",
        "name": "Spatial Group-KFold Machine Learning Modeling",
        "module": "src.models",
        "function": "main",
        "description": "Train candidate regressors under spatial cross-validation and export production model"
    },
    {
        "id": "neural",
        "name": "Deep Learning Tabular Neural Regressor Benchmark",
        "module": "src.neural_models",
        "function": "main",
        "description": "Train Multi-Layer Perceptron tabular regressor with residual depth"
    },
    {
        "id": "satellite",
        "name": "Satellite Remote Sensing & Multimodal Canopy Fusion",
        "module": "src.multimodal_fusion",
        "function": "main",
        "description": "Extract dekadal spectral indices (NDVI/EVI/NDWI) and fuse multimodal dataset"
    },
    {
        "id": "inference",
        "name": "Regional Batch Inference & Prescriptive Advisory",
        "module": "run_batch_inference",
        "function": "main",
        "description": "Generate point predictions, confidence intervals, and prescriptive soil amendments"
    },
    {
        "id": "climate",
        "name": "Climate Change Scenario Simulation Engine",
        "module": "src.climate_scenarios",
        "function": "main",
        "description": "Simulate CMIP6 SSP2-4.5 and SSP5-8.5 climate stress projections and vulnerability"
    },
    {
        "id": "economics",
        "name": "Agricultural Economics & Portfolio Optimization",
        "module": "src.economic_optimizer",
        "function": "main",
        "description": "Calculate farm profit margins, prescriptive amendment ROI, and Markowitz crop portfolios"
    },
    {
        "id": "audit",
        "name": "System Health & SLA Performance Audit",
        "module": "run_system_health_audit",
        "function": "main",
        "description": "Validate model artifacts, dataset integrity, and sub-75ms inference latency SLA"
    }
]


class MasterPlatformOrchestrator:
    """Enterprise orchestrator executing end-to-end stages of the Axlero Platform."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        self.stage_map = {s["id"]: s for s in PLATFORM_STAGES}

    def list_stages(self) -> List[Dict[str, str]]:
        """Return configured pipeline stages."""
        return PLATFORM_STAGES

    def execute_stage(self, stage_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """Execute a single pipeline stage by ID."""
        if stage_id not in self.stage_map:
            raise ValueError(f"Unknown stage: '{stage_id}'. Valid stages: {list(self.stage_map.keys())}")

        stage = self.stage_map[stage_id]
        logger.info(f"[*] ENTERING STAGE: [{stage_id.upper()}] - {stage['name']}")

        if dry_run:
            logger.info(f"    [DRY RUN] Would execute function '{stage['function']}' from module '{stage['module']}'")
            return {
                "stage": stage_id,
                "status": "DRY_RUN",
                "duration_sec": 0.0,
                "description": stage["description"]
            }

        t0 = time.perf_counter()
        status = "SUCCESS"
        error_msg = None
        orig_argv = list(sys.argv)

        try:
            sys.argv = [stage["module"]]
            # Special case for modules that need satellite timeseries before multimodal fusion
            if stage_id == "satellite":
                rs_module = importlib.import_module("src.remote_sensing")
                if hasattr(rs_module, "main"):
                    rs_module.main()

            mod = importlib.import_module(stage["module"])
            fn = getattr(mod, stage["function"])
            fn()
        except SystemExit as se:
            if se.code not in (0, None):
                status = "FAILED"
                error_msg = f"SystemExit with code {se.code}"
                logger.error(f"[!] STAGE [{stage_id.upper()}] FAILED with exit code: {se.code}")
        except Exception as e:
            status = "FAILED"
            error_msg = str(e)
            logger.error(f"[!] STAGE [{stage_id.upper()}] ENCOUNTERED EXCEPTION: {e}", exc_info=self.verbose)
        finally:
            sys.argv = orig_argv

        elapsed = round(time.perf_counter() - t0, 3)
        if status == "SUCCESS":
            logger.info(f"[+] STAGE [{stage_id.upper()}] COMPLETED in {elapsed}s")
        else:
            logger.error(f"[-] STAGE [{stage_id.upper()}] FAILED after {elapsed}s: {error_msg}")

        return {
            "stage": stage_id,
            "status": status,
            "duration_sec": elapsed,
            "error": error_msg
        }

    def run_all(self, dry_run: bool = False, skip_audit: bool = False) -> Dict[str, Any]:
        """Execute all platform stages in topological sequence."""
        logger.info("==================================================================")
        logger.info("   AXLERO AGRI-TECH PLATFORM: FULL PIPELINE EXECUTION STARTED    ")
        logger.info("==================================================================")

        t_start = time.perf_counter()
        results = []
        stages_to_run = [s["id"] for s in PLATFORM_STAGES]
        if skip_audit and "audit" in stages_to_run:
            stages_to_run.remove("audit")

        failed_stage = None
        for stage_id in stages_to_run:
            res = self.execute_stage(stage_id, dry_run=dry_run)
            results.append(res)
            if res["status"] == "FAILED":
                failed_stage = stage_id
                logger.critical(f"ABORTING PIPELINE: Stage '{stage_id}' encountered an error.")
                break

        total_elapsed = round(time.perf_counter() - t_start, 3)
        overall_status = "SUCCESS" if failed_stage is None else "FAILED"

        logger.info("==================================================================")
        logger.info(f"   AXLERO AGRI-TECH PLATFORM: EXECUTION COMPLETED ({overall_status})")
        logger.info(f"   Total Execution Duration: {total_elapsed}s")
        logger.info("==================================================================")

        return {
            "overall_status": overall_status,
            "total_duration_sec": total_elapsed,
            "failed_stage": failed_stage,
            "stage_results": results
        }


def main():
    parser = argparse.ArgumentParser(
        description="Axlero Agri-Tech Platform Master Orchestration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_master_platform.py --stage all
  python run_master_platform.py --stage audit
  python run_master_platform.py --stage satellite
  python run_master_platform.py --stage all --dry-run
        """
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "pipeline", "features", "models", "neural", "satellite", "inference", "climate", "economics", "audit"],
        help="Specify the pipeline stage to execute (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate stage sequence without running execution functions"
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the system health audit when running the full pipeline"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debugging output"
    )

    args = parser.parse_args()
    orchestrator = MasterPlatformOrchestrator(verbose=args.verbose)

    if args.stage == "all":
        summary = orchestrator.run_all(dry_run=args.dry_run, skip_audit=args.skip_audit)
        if summary["overall_status"] != "SUCCESS" and not args.dry_run:
            sys.exit(1)
    else:
        res = orchestrator.execute_stage(args.stage, dry_run=args.dry_run)
        if res["status"] != "SUCCESS" and not args.dry_run:
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
