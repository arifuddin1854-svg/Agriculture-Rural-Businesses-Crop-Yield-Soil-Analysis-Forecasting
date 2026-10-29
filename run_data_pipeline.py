"""
🌱 Agriculture & Rural Businesses: Crop Yield & Soil Analysis Forecasting System
Member 1: Data Integration & Data Engineer
Pipeline Integration & Master CLI Orchestrator (Day 6 Deliverable)

Orchestrates the entire end-to-end data pipeline:
  Stage 1: Multi-Source Data Ingestion (src/data_loader.py)
  Stage 2: Data Cleaning & Missing Value Imputation (src/data_cleaning.py)
  Stage 3: Phenological Daily Weather Aggregations (src/weather_aggregation.py)
  Stage 4: Geospatial Data Fusion (src/data_fusion.py)
  Stage 5: Exploratory Data Analysis & Visual Artifact Generation (src/eda_generator.py)
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Dict, Optional

# Core pipeline imports
from src.data_loader import generate_raw_datasets
from src.data_cleaning import run_data_cleaning_pipeline
from src.weather_aggregation import run_weather_aggregation_pipeline
from src.data_fusion import run_data_fusion_pipeline
from src.eda_generator import generate_eda_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MasterPipelineOrchestrator")


class DataPipelineOrchestrator:
    """End-to-end orchestrator for data engineering, cleaning, and fusion workflows."""

    def __init__(
        self,
        raw_dir: str = "data/raw",
        interim_dir: str = "data/interim",
        processed_dir: str = "data/processed",
        report_dir: str = "reports",
        figures_dir: str = "reports/figures",
        seed: int = 42
    ):
        self.raw_dir = Path(raw_dir)
        self.interim_dir = Path(interim_dir)
        self.processed_dir = Path(processed_dir)
        self.report_dir = Path(report_dir)
        self.figures_dir = Path(figures_dir)
        self.seed = seed

        # Ensure all directory structures exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def run_stage_1_ingestion(self) -> Dict[str, Path]:
        """Stage 1: Multi-source raw data ingestion & realistic dataset synthesis."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 1: MULTI-SOURCE RAW DATA INGESTION")
        logger.info("=" * 80)
        t0 = time.time()
        files = generate_raw_datasets(output_dir=str(self.raw_dir), seed=self.seed)
        elapsed = time.time() - t0
        logger.info(f"✅ Stage 1 completed in {elapsed:.2f}s")
        return files

    def run_stage_2_cleaning(self) -> Dict[str, Path]:
        """Stage 2: Missing value imputation, temperature inversion fixes, and QA."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 2: DATA CLEANING & PHYSICAL BOUNDARY IMPUTATION")
        logger.info("=" * 80)
        t0 = time.time()
        files = run_data_cleaning_pipeline(
            raw_dir=str(self.raw_dir),
            interim_dir=str(self.interim_dir),
            report_dir=str(self.report_dir)
        )
        elapsed = time.time() - t0
        logger.info(f"✅ Stage 2 completed in {elapsed:.2f}s")
        return files

    def run_stage_3_weather_aggregation(self) -> Dict[str, Path]:
        """Stage 3: Agro-meteorological phenological aggregation (GDD, rain, VPD, heat stress)."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 3: PHENOLOGICAL DAILY WEATHER AGGREGATIONS")
        logger.info("=" * 80)
        t0 = time.time()
        files = run_weather_aggregation_pipeline(
            clean_weather_path=str(self.interim_dir / "clean_weather_daily.csv"),
            clean_yields_path=str(self.interim_dir / "clean_historical_yields.csv"),
            output_dir=str(self.processed_dir),
            report_dir=str(self.report_dir)
        )
        elapsed = time.time() - t0
        logger.info(f"✅ Stage 3 completed in {elapsed:.2f}s")
        return files

    def run_stage_4_fusion(self) -> Dict[str, Path]:
        """Stage 4: Multi-source geospatial alignment and master dataset fusion."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 4: GEOSPATIAL DATA FUSION & REFERENTIAL INTEGRITY")
        logger.info("=" * 80)
        t0 = time.time()
        files = run_data_fusion_pipeline(
            yields_path=str(self.interim_dir / "clean_historical_yields.csv"),
            soil_path=str(self.interim_dir / "clean_soil_chemistry.csv"),
            weather_path=str(self.processed_dir / "weather_aggregated.csv"),
            output_dir=str(self.processed_dir),
            report_dir=str(self.report_dir)
        )
        elapsed = time.time() - t0
        logger.info(f"✅ Stage 4 completed in {elapsed:.2f}s")
        return files

    def run_stage_5_eda(self) -> None:
        """Stage 5: Exploratory Data Analysis & visual artifact rendering."""
        logger.info("\n" + "=" * 80)
        logger.info("STAGE 5: EXPLORATORY DATA ANALYSIS & VISUAL ARTIFACT GENERATION")
        logger.info("=" * 80)
        t0 = time.time()
        generate_eda_figures(
            dataset_path=str(self.processed_dir / "unified_agri_dataset.csv"),
            output_fig_dir=str(self.figures_dir),
            output_report_path=str(self.report_dir / "eda_summary_report.md")
        )
        elapsed = time.time() - t0
        logger.info(f"✅ Stage 5 completed in {elapsed:.2f}s")

    def run_all(self):
        """Executes all 5 stages in end-to-end sequence."""
        start_time = time.time()
        logger.info("🚀 Starting Full End-to-End Agricultural Data Pipeline...")

        self.run_stage_1_ingestion()
        self.run_stage_2_cleaning()
        self.run_stage_3_weather_aggregation()
        self.run_stage_4_fusion()
        self.run_stage_5_eda()

        total_elapsed = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info(f"🎉 MASTER DATA PIPELINE SUCCESSFULLY EXECUTED in {total_elapsed:.2f}s")
        logger.info(f"Master Dataset Deliverable: {self.processed_dir / 'unified_agri_dataset.csv'}")
        logger.info(f"EDA Summary Report:        {self.report_dir / 'eda_summary_report.md'}")
        logger.info(f"Visual Artifacts:          {self.figures_dir}/")
        logger.info("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Axlero Agri-Tech End-to-End Data Pipeline Orchestrator"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "ingest", "clean", "aggregate", "fuse", "eda"],
        default="all",
        help="Specify which pipeline stage to execute (default: 'all')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline stages end-to-end"
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--interim-dir", type=str, default="data/interim", help="Path to interim cleaned data directory")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--report-dir", type=str, default="reports", help="Path to reports directory")
    parser.add_argument("--figures-dir", type=str, default="reports/figures", help="Path to figures directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    orchestrator = DataPipelineOrchestrator(
        raw_dir=args.raw_dir,
        interim_dir=args.interim_dir,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        figures_dir=args.figures_dir,
        seed=args.seed
    )

    if args.all or args.stage == "all":
        orchestrator.run_all()
    elif args.stage == "ingest":
        orchestrator.run_stage_1_ingestion()
    elif args.stage == "clean":
        orchestrator.run_stage_2_cleaning()
    elif args.stage == "aggregate":
        orchestrator.run_stage_3_weather_aggregation()
    elif args.stage == "fuse":
        orchestrator.run_stage_4_fusion()
    elif args.stage == "eda":
        orchestrator.run_stage_5_eda()


if __name__ == "__main__":
    main()
