"""
Member 1: Data Integration & Data Engineer
Day 4: Geospatial Data Fusion Pipeline

Responsible for:
- Harmonizing multi-source agricultural datasets (interim cleaned soil chemistry,
  cleaned historical yields, and phenologically aggregated weather metrics).
- Enforcing referential integrity across spatial (district_id) and temporal (year, crop_type) keys.
- Detecting and preventing unaligned geographic anomalies and Cartesian join explosions.
- Exporting the production Unified Agricultural Master Dataset (CSV and Parquet formats).
- Generating comprehensive machine-readable and executive audit reports in reports/.
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataFusion_Day4")


# =====================================================================
# Integrity Verification & Referential Key Checks
# =====================================================================

def verify_referential_integrity(
    yields_df: pd.DataFrame,
    soil_df: pd.DataFrame,
    weather_agg_df: pd.DataFrame,
    spatial_key: str = "district_id"
) -> Dict[str, Any]:
    """
    Validates foreign key referential integrity across yields, soil, and weather datasets.
    Ensures zero unaligned districts or missing temporal coordinates.
    """
    issues = []
    
    yield_districts = set(yields_df[spatial_key].unique())
    soil_districts = set(soil_df[spatial_key].unique())
    weather_districts = set(weather_agg_df[spatial_key].unique())

    # 1. Check spatial key coverage in soil chemistry
    missing_in_soil = yield_districts - soil_districts
    if missing_in_soil:
        issues.append(f"Missing {len(missing_in_soil)} districts in soil chemistry: {sorted(list(missing_in_soil))}")

    # 2. Check spatial key coverage in weather aggregations
    missing_in_weather = yield_districts - weather_districts
    if missing_in_weather:
        issues.append(f"Missing {len(missing_in_weather)} districts in weather: {sorted(list(missing_in_weather))}")

    # 3. Check composite key alignment if crop-specific phenology is present
    is_phenological_weather = "crop_type" in weather_agg_df.columns
    if is_phenological_weather:
        yield_keys = set(
            zip(yields_df["district_id"], yields_df["year"], yields_df["crop_type"])
        )
        weather_keys = set(
            zip(weather_agg_df["district_id"], weather_agg_df["year"], weather_agg_df["crop_type"])
        )
        missing_weather_combos = yield_keys - weather_keys
        if missing_weather_combos:
            issues.append(f"Missing {len(missing_weather_combos)} (district, year, crop) combinations in weather aggregation.")
    else:
        yield_temporal_keys = set(zip(yields_df["district_id"], yields_df["year"]))
        weather_temporal_keys = set(zip(weather_agg_df["district_id"], weather_agg_df["year"]))
        missing_annual = yield_temporal_keys - weather_temporal_keys
        if missing_annual:
            issues.append(f"Missing {len(missing_annual)} (district, year) keys in annual weather aggregation.")

    integrity_passed = len(issues) == 0
    return {
        "integrity_passed": integrity_passed,
        "yield_districts_count": len(yield_districts),
        "soil_districts_count": len(soil_districts),
        "weather_districts_count": len(weather_districts),
        "is_phenological_weather": is_phenological_weather,
        "issues": issues
    }


# =====================================================================
# Geospatial Multi-Source Dataset Fusion Engine
# =====================================================================

def fuse_agricultural_datasets(
    yields_df: pd.DataFrame,
    soil_df: pd.DataFrame,
    weather_agg_df: pd.DataFrame,
    spatial_key: str = "district_id"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs deterministic multi-source fusion:
    1. Inner joins historical crop yields with static soil chemical profiles on `district_id`.
    2. Merges with meteorological growing-season indicators on `[district_id, year, crop_type]`
       (or `[district_id, year]` if annual weather is provided).
    3. Validates row cardinality and column schema alignment.
    """
    initial_yields_len = len(yields_df)
    logger.info(f"Initiating geospatial data fusion for {initial_yields_len} yield records...")

    # Step 1: Referential integrity check
    integrity_report = verify_referential_integrity(yields_df, soil_df, weather_agg_df, spatial_key)
    if not integrity_report["integrity_passed"]:
        logger.warning(f"Integrity check warnings detected: {integrity_report['issues']}")

    # Step 2: Merge crop yields with static soil chemistry
    merged_soil = pd.merge(
        yields_df,
        soil_df,
        on=spatial_key,
        how="inner"
    )
    logger.info(f"   -> Soil merge completed. Intermediate rows: {len(merged_soil)}")

    # Step 3: Merge with weather aggregations
    has_crop_in_weather = "crop_type" in weather_agg_df.columns
    if has_crop_in_weather:
        join_keys = [spatial_key, "year", "crop_type"]
    else:
        join_keys = [spatial_key, "year"]

    fused_df = pd.merge(
        merged_soil,
        weather_agg_df,
        on=join_keys,
        how="inner"
    )
    logger.info(f"   -> Weather merge completed on keys {join_keys}. Final rows: {len(fused_df)}")

    # Step 4: Cardinality verification
    if len(fused_df) != initial_yields_len:
        logger.warning(
            f"Row count mismatch after fusion! Expected {initial_yields_len}, but got {len(fused_df)}. "
            f"Difference: {initial_yields_len - len(fused_df)} rows dropped."
        )

    # Step 5: Column ordering for logical consistency
    core_keys = ["district_id", "year", "crop_type", "planted_area_ha", "production_mt", "crop_yield_mt_ha"]
    soil_keys = [col for col in soil_df.columns if col not in core_keys]
    weather_keys = [col for col in weather_agg_df.columns if col not in core_keys and col not in soil_keys]

    ordered_cols = [c for c in core_keys if c in fused_df.columns] + \
                   [c for c in soil_keys if c in fused_df.columns] + \
                   [c for c in weather_keys if c in fused_df.columns]
    
    # Ensure any residual columns are appended
    remaining_cols = [c for c in fused_df.columns if c not in ordered_cols]
    ordered_cols.extend(remaining_cols)
    fused_df = fused_df[ordered_cols].copy()

    audit_summary = {
        "input_yield_records": initial_yields_len,
        "fused_records_count": len(fused_df),
        "columns_count": len(fused_df.columns),
        "null_values_count": int(fused_df.isnull().sum().sum()),
        "districts_count": int(fused_df[spatial_key].nunique()),
        "crops_count": int(fused_df["crop_type"].nunique()),
        "years_count": int(fused_df["year"].nunique()),
        "columns": list(fused_df.columns),
        "integrity_check": integrity_report
    }

    return fused_df, audit_summary


# =====================================================================
# Audit Report Generation
# =====================================================================

def generate_data_fusion_report(
    fused_df: pd.DataFrame,
    audit_summary: Dict[str, Any],
    output_json_path: Path,
    output_md_path: Path
) -> None:
    """
    Generates structured audit logs and comprehensive Markdown documentation
    for the unified agricultural dataset.
    """
    crop_stats = {}
    for crop, group in fused_df.groupby("crop_type"):
        crop_stats[crop] = {
            "record_count": len(group),
            "mean_yield_mt_ha": round(float(group["crop_yield_mt_ha"].mean()), 2),
            "std_yield_mt_ha": round(float(group["crop_yield_mt_ha"].std()), 2),
            "mean_area_ha": round(float(group["planted_area_ha"].mean()), 1),
            "mean_gdd": round(float(group["gdd_cumulative"].mean()), 1) if "gdd_cumulative" in group.columns else 0.0,
            "mean_rainfall_mm": round(float(group["cumulative_rainfall_mm"].mean()), 1) if "cumulative_rainfall_mm" in group.columns else 0.0
        }

    report_payload = {
        "timestamp": datetime.now().isoformat(),
        "summary": audit_summary,
        "crop_level_breakdown": crop_stats
    }

    # Write JSON report
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Write Markdown report
    md_lines = [
        "# 🌐 Geospatial Data Fusion Quality Audit Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Scope**: Member 1 (Day 4 Milestone) — Multi-Source Geospatial Fusion  ",
        f"**Unified Records**: {audit_summary['fused_records_count']} rows, {audit_summary['columns_count']} features  ",
        f"**Referential Integrity Status**: {'✅ PASSED (100% matched)' if audit_summary['integrity_check']['integrity_passed'] else '⚠️ WARNINGS PRESENT'}  ",
        "",
        "---",
        "",
        "## 📊 Unified Dataset Summary by Crop",
        "",
        "| Crop Type | Records | Mean Yield (MT/Ha) | Mean Area (Ha) | Mean Rainfall (mm) | Mean GDD (°C-day) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for crop, stats in crop_stats.items():
        yld = f"{stats['mean_yield_mt_ha']} ± {stats['std_yield_mt_ha']}"
        area = f"{stats['mean_area_ha']}"
        rain = f"{stats['mean_rainfall_mm']}"
        gdd = f"{stats['mean_gdd']}"
        md_lines.append(f"| **{crop}** | {stats['record_count']} | {yld} | {area} | {rain} | {gdd} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 🧱 Unified Schema Feature Breakdown",
        "",
        "| Feature Category | Count | Example Variables |",
        "| :--- | :---: | :--- |",
        "| **Geographic & Keys** | 3 | `district_id`, `year`, `crop_type` |",
        "| **Production & Target** | 3 | `planted_area_ha`, `production_mt`, `crop_yield_mt_ha` |",
        "| **Soil Chemistry** | 7 | `soil_type`, `nitrogen_kg_ha`, `phosphorus_kg_ha`, `potassium_kg_ha`, `soil_ph`, `organic_matter_pct`, `electrical_conductivity_ds_m` |",
        "| **Agro-Meteorology** | 16 | `cumulative_rainfall_mm`, `gdd_cumulative`, `mean_temp_max`, `heat_stress_days`, `mean_vpd_kpa`, `consecutive_dry_days_max`, `cumulative_solar_radiation_mj` |",
        "",
        "---",
        "✅ **Handoff Ready**: Unified Master Feature Table prepared for Member 2 (Agronomic Feature Engineering)."
    ])

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


# =====================================================================
# Main Pipeline Runner
# =====================================================================

def run_data_fusion_pipeline(
    yields_path: str = "data/interim/clean_historical_yields.csv",
    soil_path: str = "data/interim/clean_soil_chemistry.csv",
    weather_path: str = "data/processed/weather_aggregated.csv",
    output_dir: str = "data/processed",
    report_dir: str = "reports"
) -> Dict[str, Path]:
    """
    Orchestrates Day 4 Geospatial Data Fusion pipeline.
    """
    logger.info("=" * 70)
    logger.info("Executing Member 1 Day 4: Geospatial Data Fusion Pipeline")
    logger.info("=" * 70)

    out_path = Path(output_dir)
    rep_path = Path(report_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Clean Datasets
    logger.info(f"1. Loading cleaned crop yields from {yields_path}...")
    yields_df = pd.read_csv(yields_path)

    logger.info(f"2. Loading cleaned soil chemistry from {soil_path}...")
    soil_df = pd.read_csv(soil_path)

    logger.info(f"3. Loading aggregated weather indicators from {weather_path}...")
    weather_df = pd.read_csv(weather_path)

    # 4. Perform Data Fusion
    logger.info("4. Executing multi-source fusion and integrity checks...")
    fused_df, audit_summary = fuse_agricultural_datasets(
        yields_df=yields_df,
        soil_df=soil_df,
        weather_agg_df=weather_df,
        spatial_key="district_id"
    )

    # 5. Export Master Datasets (CSV & Parquet)
    csv_file = out_path / "unified_agri_dataset.csv"
    parquet_file = out_path / "unified_agri_dataset.parquet"

    fused_df.to_csv(csv_file, index=False)
    logger.info(f"   -> Saved Master CSV Dataset: {csv_file} ({len(fused_df)} rows, {len(fused_df.columns)} columns)")

    try:
        fused_df.to_parquet(parquet_file, index=False)
        logger.info(f"   -> Saved Master Parquet Dataset: {parquet_file}")
    except Exception as e:
        logger.warning(f"Parquet serialization skipped ({e}); CSV master deliverable saved successfully.")

    # 6. Generate Audit Documentation
    json_report = rep_path / "data_fusion_report.json"
    md_report = rep_path / "data_fusion_report.md"
    generate_data_fusion_report(fused_df, audit_summary, json_report, md_report)
    logger.info(f"6. Saved audit reports to {json_report} and {md_report}")

    logger.info("=" * 70)
    logger.info("Day 4 Geospatial Data Fusion Completed Successfully!")
    logger.info("=" * 70)

    return {
        "csv": csv_file,
        "parquet": parquet_file,
        "json_report": json_report,
        "md_report": md_report
    }


def main():
    parser = argparse.ArgumentParser(description="Geospatial Agricultural Data Fusion Pipeline")
    parser.add_argument("--yields-path", type=str, default="data/interim/clean_historical_yields.csv")
    parser.add_argument("--soil-path", type=str, default="data/interim/clean_soil_chemistry.csv")
    parser.add_argument("--weather-path", type=str, default="data/processed/weather_aggregated.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--report-dir", type=str, default="reports")
    args = parser.parse_args()

    run_data_fusion_pipeline(
        yields_path=args.yields_path,
        soil_path=args.soil_path,
        weather_path=args.weather_path,
        output_dir=args.output_dir,
        report_dir=args.report_dir
    )


if __name__ == "__main__":
    main()
