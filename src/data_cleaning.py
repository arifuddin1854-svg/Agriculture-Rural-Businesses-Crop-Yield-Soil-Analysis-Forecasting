"""
Member 1: Data Integration & Data Engineer
Day 2: Data Cleaning, Missing Value Imputation & Quality Assurance Pipeline

Responsible for:
1. Physical boundary verification & clamping (pH in [3.5, 9.5], positive rainfall, NPK bounds, etc.)
2. Domain-aware missing value imputation:
   - Weather: 7-day rolling median per district, district-month median fallback
   - Soil Chemistry: Spatial/soil-type median imputation, global fallback
   - Crop Yields: Agronomic arithmetic reconstruction (Production / Area) & crop-year median
3. Outlier detection & sanity checks (e.g., temp_min <= temp_max consistency)
4. Persisting cleaned interim datasets to `data/interim/`
5. Generating comprehensive QA validation reports (JSON & Markdown)
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

# Set thread count for BLAS to avoid Windows allocation issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DataCleaning_Day2")

# =====================================================================
# PHYSICAL DOMAIN BOUNDARIES & SCHEMA DEFINITIONS
# =====================================================================

BOUNDARIES = {
    "soil": {
        "soil_ph": (3.5, 9.5),                       # Agronomically feasible soil pH
        "nitrogen_kg_ha": (0.0, 500.0),              # Available N in kg/ha
        "phosphorus_kg_ha": (0.0, 300.0),            # Available P in kg/ha
        "potassium_kg_ha": (0.0, 600.0),             # Available K in kg/ha
        "organic_matter_pct": (0.05, 15.0),          # SOM %
        "electrical_conductivity_ds_m": (0.0, 10.0), # EC dS/m
    },
    "weather": {
        "temp_min": (-30.0, 55.0),                   # Min temp °C
        "temp_max": (-25.0, 60.0),                   # Max temp °C
        "precipitation_mm": (0.0, 500.0),            # Non-negative daily rain in mm
        "relative_humidity": (0.0, 100.0),           # Percentage [0, 100]
        "solar_radiation_mj": (0.0, 50.0),           # MJ/m2
    },
    "yield": {
        "planted_area_ha": (1.0, 100000.0),          # Hectares
        "production_mt": (0.0, 1000000.0),           # Metric Tonnes
        "crop_yield_mt_ha": (0.1, 50.0),             # MT/ha
    }
}


class QualityAuditTracker:
    """Tracks quality issues, missing values, anomalies, and transformations."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.initial_rows = 0
        self.final_rows = 0
        self.initial_nulls: Dict[str, int] = {}
        self.imputed_nulls: Dict[str, int] = {}
        self.boundary_violations: Dict[str, int] = {}
        self.swapped_temp_inversions = 0
        self.outliers_flagged: Dict[str, int] = {}
        self.before_stats: Dict[str, Dict[str, float]] = {}
        self.after_stats: Dict[str, Dict[str, float]] = {}
        self.notes: List[str] = []

    def record_before_stats(self, df: pd.DataFrame):
        self.initial_rows = len(df)
        self.initial_nulls = {col: int(df[col].isnull().sum()) for col in df.columns}
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if not s.empty:
                self.before_stats[col] = {
                    "min": round(float(s.min()), 2),
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
                    "max": round(float(s.max()), 2)
                }

    def record_after_stats(self, df: pd.DataFrame):
        self.final_rows = len(df)
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if not s.empty:
                self.after_stats[col] = {
                    "min": round(float(s.min()), 2),
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2) if len(s) > 1 else 0.0,
                    "max": round(float(s.max()), 2)
                }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "initial_records": self.initial_rows,
            "final_records": self.final_rows,
            "initial_nulls": self.initial_nulls,
            "imputed_nulls": self.imputed_nulls,
            "boundary_violations_corrected": self.boundary_violations,
            "temperature_inversions_swapped": self.swapped_temp_inversions,
            "outliers_flagged_iqr": self.outliers_flagged,
            "before_stats": self.before_stats,
            "after_stats": self.after_stats,
            "audit_notes": self.notes
        }


# =====================================================================
# DATA CLEANING ENGINES
# =====================================================================

def clean_soil_chemistry(
    df: pd.DataFrame,
    tracker: Optional[QualityAuditTracker] = None
) -> pd.DataFrame:
    """
    Cleans soil chemistry datasets:
    - Enforces schema and types
    - Imputes missing chemical properties using soil_type group median, falling back to global median
    - Validates and clamps to physical agronomic boundaries (pH, NPK, SOM, EC)
    - Detects IQR statistical anomalies
    """
    if tracker is None:
        tracker = QualityAuditTracker("soil_chemistry")
    
    clean_df = df.copy()
    tracker.record_before_stats(clean_df)

    expected_cols = [
        "district_id", "soil_type", "nitrogen_kg_ha", "phosphorus_kg_ha",
        "potassium_kg_ha", "soil_ph", "organic_matter_pct", "electrical_conductivity_ds_m"
    ]
    for col in expected_cols:
        if col not in clean_df.columns:
            raise ValueError(f"Missing required soil column: {col}")

    # 1. Type casting
    clean_df["district_id"] = clean_df["district_id"].astype(str).str.strip()
    clean_df["soil_type"] = clean_df["soil_type"].astype(str).str.strip()
    numeric_cols = [c for c in expected_cols if c not in ["district_id", "soil_type"]]
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # 2. Imputation via spatial / soil_type median
    for col in numeric_cols:
        missing_cnt = int(clean_df[col].isnull().sum())
        if missing_cnt > 0:
            # Group median by soil_type
            group_median = clean_df.groupby("soil_type")[col].transform("median")
            overall_median = clean_df[col].median()
            clean_df[col] = clean_df[col].fillna(group_median).fillna(overall_median)
            # If still null (e.g. all values missing), fallback to boundary midpoint
            if clean_df[col].isnull().any():
                low, high = BOUNDARIES["soil"][col]
                clean_df[col] = clean_df[col].fillna((low + high) / 2.0)
            tracker.imputed_nulls[col] = missing_cnt
            tracker.notes.append(f"Imputed {missing_cnt} missing values in '{col}' using soil_type group median.")

    # 3. Physical Boundary Enforcement & Clamping
    for col, (low, high) in BOUNDARIES["soil"].items():
        if col in clean_df.columns:
            violations = int(((clean_df[col] < low) | (clean_df[col] > high)).sum())
            if violations > 0:
                tracker.boundary_violations[col] = violations
                clean_df[col] = clean_df[col].clip(lower=low, upper=high)
                tracker.notes.append(f"Clamped {violations} out-of-boundary entries in '{col}' to [{low}, {high}].")

    # 4. Outlier detection (IQR method for reporting)
    for col in numeric_cols:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = int(((clean_df[col] < (q1 - 1.5 * iqr)) | (clean_df[col] > (q3 + 1.5 * iqr))).sum())
        tracker.outliers_flagged[col] = outliers

    # Round numeric values
    for col in numeric_cols:
        clean_df[col] = clean_df[col].round(2)

    tracker.record_after_stats(clean_df)
    return clean_df


def clean_weather_daily(
    df: pd.DataFrame,
    tracker: Optional[QualityAuditTracker] = None
) -> pd.DataFrame:
    """
    Cleans daily weather time-series records:
    - Parses dates and sorts chronologically per district
    - Inverts temp_min and temp_max if sensor inverted them (temp_min > temp_max)
    - Imputes missing values via 7-day rolling median per district, followed by district-month median fallback
    - Clamps to physical atmospheric boundaries (e.g., non-negative rain, relative humidity in [0, 100])
    - Detects IQR statistical anomalies
    """
    if tracker is None:
        tracker = QualityAuditTracker("weather_daily")

    clean_df = df.copy()
    tracker.record_before_stats(clean_df)

    expected_cols = [
        "district_id", "date", "temp_max", "temp_min",
        "precipitation_mm", "relative_humidity", "solar_radiation_mj"
    ]
    for col in expected_cols:
        if col not in clean_df.columns:
            raise ValueError(f"Missing required weather column: {col}")

    # 1. Date normalization & chronological sorting
    clean_df["district_id"] = clean_df["district_id"].astype(str).str.strip()
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    clean_df["year"] = clean_df["date"].dt.year
    clean_df["month"] = clean_df["date"].dt.month
    clean_df = clean_df.sort_values(by=["district_id", "date"]).reset_index(drop=True)

    numeric_cols = [
        "temp_max", "temp_min", "precipitation_mm", "relative_humidity", "solar_radiation_mj"
    ]
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # 2. Consistency Check: Swap temp_min and temp_max if inverted (temp_min > temp_max)
    inverted_mask = clean_df["temp_min"] > clean_df["temp_max"]
    inversion_count = int(inverted_mask.sum())
    if inversion_count > 0:
        tracker.swapped_temp_inversions = inversion_count
        tracker.notes.append(f"Swapped {inversion_count} inverted temp_min/temp_max records.")
        clean_df.loc[inverted_mask, ["temp_max", "temp_min"]] = clean_df.loc[
            inverted_mask, ["temp_min", "temp_max"]
        ].values

    # 3. Time-Series Missing Value Imputation
    # Strategy: Rolling 7-day median per district -> District/Month median -> Global median
    for col in numeric_cols:
        missing_cnt = int(clean_df[col].isnull().sum())
        if missing_cnt > 0:
            tracker.imputed_nulls[col] = missing_cnt
            # Rolling 7-day median per district
            clean_df[col] = (
                clean_df.groupby("district_id")[col]
                .transform(lambda s: s.rolling(window=7, min_periods=1, center=True).median())
            )
            # Forward & Backward fill per district
            clean_df[col] = clean_df.groupby("district_id")[col].ffill().bfill()

            # District x Month median fallback
            if clean_df[col].isnull().any():
                dist_month_med = clean_df.groupby(["district_id", "month"])[col].transform("median")
                clean_df[col] = clean_df[col].fillna(dist_month_med)

            # Global median fallback
            if clean_df[col].isnull().any():
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())

            tracker.notes.append(f"Imputed {missing_cnt} missing values in '{col}' using 7-day rolling median & month fallback.")

    # 4. Physical Boundary Clamping
    for col, (low, high) in BOUNDARIES["weather"].items():
        if col in clean_df.columns:
            violations = int(((clean_df[col] < low) | (clean_df[col] > high)).sum())
            if violations > 0:
                tracker.boundary_violations[col] = violations
                clean_df[col] = clean_df[col].clip(lower=low, upper=high)
                tracker.notes.append(f"Clamped {violations} out-of-boundary records in '{col}' to [{low}, {high}].")

    # 5. Outlier Detection (IQR)
    for col in numeric_cols:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = int(((clean_df[col] < (q1 - 1.5 * iqr)) | (clean_df[col] > (q3 + 1.5 * iqr))).sum())
        tracker.outliers_flagged[col] = outliers

    # Round numeric values
    for col in numeric_cols:
        clean_df[col] = clean_df[col].round(1)

    # Convert date back to ISO string for standard serialization
    clean_df["date"] = clean_df["date"].dt.strftime("%Y-%m-%d")

    tracker.record_after_stats(clean_df)
    return clean_df


def clean_historical_yields(
    df: pd.DataFrame,
    tracker: Optional[QualityAuditTracker] = None
) -> pd.DataFrame:
    """
    Cleans historical crop yield records:
    - Normalizes crop names and district IDs
    - Agronomic arithmetic consistency: validates and reconciles `crop_yield_mt_ha = production_mt / planted_area_ha`
    - Imputes missing values using crop-type by year medians
    - Enforces physical boundaries on area, production, and yield
    - Detects IQR statistical anomalies
    """
    if tracker is None:
        tracker = QualityAuditTracker("historical_yields")

    clean_df = df.copy()
    tracker.record_before_stats(clean_df)

    expected_cols = [
        "district_id", "year", "crop_type",
        "planted_area_ha", "production_mt", "crop_yield_mt_ha"
    ]
    for col in expected_cols:
        if col not in clean_df.columns:
            raise ValueError(f"Missing required yield column: {col}")

    # 1. Type casting & text normalization
    clean_df["district_id"] = clean_df["district_id"].astype(str).str.strip()
    clean_df["crop_type"] = clean_df["crop_type"].astype(str).str.strip().str.title()
    clean_df["year"] = pd.to_numeric(clean_df["year"], errors="coerce").astype(int)

    numeric_cols = ["planted_area_ha", "production_mt", "crop_yield_mt_ha"]
    for col in numeric_cols:
        clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    # 2. Arithmetic Reconciliation
    # Case A: Missing yield but present production and area
    missing_yield = clean_df["crop_yield_mt_ha"].isnull() & clean_df["production_mt"].notnull() & clean_df["planted_area_ha"].notnull()
    if missing_yield.any():
        clean_df.loc[missing_yield, "crop_yield_mt_ha"] = (
            clean_df.loc[missing_yield, "production_mt"] / clean_df.loc[missing_yield, "planted_area_ha"]
        )
        tracker.notes.append(f"Reconstructed {missing_yield.sum()} yield values from Production / Area.")

    # Case B: Missing production but present yield and area
    missing_prod = clean_df["production_mt"].isnull() & clean_df["crop_yield_mt_ha"].notnull() & clean_df["planted_area_ha"].notnull()
    if missing_prod.any():
        clean_df.loc[missing_prod, "production_mt"] = (
            clean_df.loc[missing_prod, "crop_yield_mt_ha"] * clean_df.loc[missing_prod, "planted_area_ha"]
        )
        tracker.notes.append(f"Reconstructed {missing_prod.sum()} production values from Yield * Area.")

    # 3. Imputation for remaining nulls using crop_type x year median
    for col in numeric_cols:
        missing_cnt = int(clean_df[col].isnull().sum())
        if missing_cnt > 0:
            tracker.imputed_nulls[col] = missing_cnt
            group_med = clean_df.groupby(["crop_type", "year"])[col].transform("median")
            crop_med = clean_df.groupby("crop_type")[col].transform("median")
            clean_df[col] = clean_df[col].fillna(group_med).fillna(crop_med).fillna(clean_df[col].median())
            tracker.notes.append(f"Imputed {missing_cnt} missing values in '{col}' using crop x year median.")

    # 4. Physical Boundary Clamping
    for col, (low, high) in BOUNDARIES["yield"].items():
        if col in clean_df.columns:
            violations = int(((clean_df[col] < low) | (clean_df[col] > high)).sum())
            if violations > 0:
                tracker.boundary_violations[col] = violations
                clean_df[col] = clean_df[col].clip(lower=low, upper=high)
                tracker.notes.append(f"Clamped {violations} out-of-boundary entries in '{col}' to [{low}, {high}].")

    # 5. Reconcile final yield = production / area if discrepancy > 10%
    ratio = clean_df["production_mt"] / clean_df["planted_area_ha"].replace(0, np.nan)
    discrepancy = (np.abs(clean_df["crop_yield_mt_ha"] - ratio) / clean_df["crop_yield_mt_ha"]) > 0.10
    if discrepancy.any():
        clean_df.loc[discrepancy, "crop_yield_mt_ha"] = ratio[discrepancy]
        tracker.notes.append(f"Reconciled {discrepancy.sum()} yield values with production/area ratio.")

    # 6. Outlier Detection (IQR)
    for col in numeric_cols:
        q1 = clean_df[col].quantile(0.25)
        q3 = clean_df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = int(((clean_df[col] < (q1 - 1.5 * iqr)) | (clean_df[col] > (q3 + 1.5 * iqr))).sum())
        tracker.outliers_flagged[col] = outliers

    # Round numeric values
    clean_df["planted_area_ha"] = clean_df["planted_area_ha"].round(1)
    clean_df["production_mt"] = clean_df["production_mt"].round(1)
    clean_df["crop_yield_mt_ha"] = clean_df["crop_yield_mt_ha"].round(2)

    tracker.record_after_stats(clean_df)
    return clean_df


# =====================================================================
# REPORT GENERATORS
# =====================================================================

def generate_markdown_report(audit_results: Dict[str, Any], output_file: Path):
    """Generates a comprehensive Markdown validation and QA report."""
    md_lines = [
        "# 🛡️ Data Cleaning & Quality Assurance Validation Report",
        "",
        "> **Generated by**: Member 1 (Data Integration & Data Engineer) — Day 2 Milestone  ",
        "> **Status**: ✅ Quality Validated & Interim Data Persisted  ",
        "> **Scope**: Soil Chemistry, Daily Meteorology, Historical Crop Yields  ",
        "",
        "---",
        "",
        "## 📊 Executive Summary of Data Cleaning",
        "",
        "| Dataset | Initial Records | Cleaned Records | Initial Nulls | Imputed Nulls | Boundary Clamps | Inversions Corrected |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for name, data in audit_results.items():
        init_null_sum = sum(data.get("initial_nulls", {}).values())
        imputed_sum = sum(data.get("imputed_nulls", {}).values())
        bounds_sum = sum(data.get("boundary_violations_corrected", {}).values())
        inversions = data.get("temperature_inversions_swapped", 0)

        md_lines.append(
            f"| **{name}** | {data['initial_records']:,} | {data['final_records']:,} | "
            f"{init_null_sum} | {imputed_sum} | {bounds_sum} | {inversions} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 🔬 Detailed Dataset Audits",
        ""
    ])

    for name, data in audit_results.items():
        md_lines.extend([
            f"### 📁 `{name}`",
            "",
            f"- **Records Processed**: `{data['final_records']:,}`",
            f"- **Audit Notes**: {len(data.get('audit_notes', []))} actionable audit events recorded",
            ""
        ])

        if data.get("audit_notes"):
            md_lines.append("**Transformation Log:**")
            for note in data["audit_notes"]:
                md_lines.append(f"- 🔹 {note}")
            md_lines.append("")

        # Distribution comparison table
        md_lines.extend([
            "**Pre- vs. Post-Cleaning Feature Distribution:**",
            "",
            "| Feature | Pre-Clean [Min, Mean, Max] | Post-Clean [Min, Mean, Max] | IQR Outliers Flagged |",
            "| :--- | :---: | :---: | :---: |"
        ])

        all_cols = sorted(list(data.get("after_stats", {}).keys()))
        for col in all_cols:
            b = data.get("before_stats", {}).get(col, {})
            a = data.get("after_stats", {}).get(col, {})
            b_str = f"[{b.get('min', 'N/A')}, {b.get('mean', 'N/A')}, {b.get('max', 'N/A')}]"
            a_str = f"[{a.get('min', 'N/A')}, {a.get('mean', 'N/A')}, {a.get('max', 'N/A')}]"
            outlier_cnt = data.get("outliers_flagged_iqr", {}).get(col, 0)
            md_lines.append(f"| `{col}` | {b_str} | {a_str} | {outlier_cnt} |")

        md_lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


# =====================================================================
# PIPELINE ORCHESTRATOR
# =====================================================================

def run_data_cleaning_pipeline(
    raw_dir: str = "data/raw",
    interim_dir: str = "data/interim",
    report_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Orchestrates end-to-end data cleaning and validation for all Day 2 datasets.
    """
    raw_path = Path(raw_dir)
    interim_path = Path(interim_dir)
    report_path = Path(report_dir)

    interim_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    audit_summary: Dict[str, Any] = {}

    # 1. Clean Soil Chemistry
    soil_file = raw_path / "soil_chemistry.csv"
    logger.info(f"1. Processing Soil Chemistry: {soil_file}")
    if not soil_file.exists():
        raise FileNotFoundError(f"Soil file not found at {soil_file}")
    soil_raw = pd.read_csv(soil_file)
    soil_tracker = QualityAuditTracker("soil_chemistry")
    clean_soil = clean_soil_chemistry(soil_raw, soil_tracker)
    clean_soil.to_csv(interim_path / "clean_soil_chemistry.csv", index=False)
    audit_summary["soil_chemistry"] = soil_tracker.to_dict()
    logger.info(f"   -> Saved: {interim_path / 'clean_soil_chemistry.csv'} ({len(clean_soil)} rows)")

    # 2. Clean Historical Yields
    yield_file = raw_path / "historical_yields.csv"
    logger.info(f"2. Processing Historical Yields: {yield_file}")
    if not yield_file.exists():
        raise FileNotFoundError(f"Yield file not found at {yield_file}")
    yield_raw = pd.read_csv(yield_file)
    yield_tracker = QualityAuditTracker("historical_yields")
    clean_yield = clean_historical_yields(yield_raw, yield_tracker)
    clean_yield.to_csv(interim_path / "clean_historical_yields.csv", index=False)
    audit_summary["historical_yields"] = yield_tracker.to_dict()
    logger.info(f"   -> Saved: {interim_path / 'clean_historical_yields.csv'} ({len(clean_yield)} rows)")

    # 3. Clean Daily Weather
    weather_file = raw_path / "weather_daily.csv"
    logger.info(f"3. Processing Daily Weather: {weather_file}")
    if not weather_file.exists():
        raise FileNotFoundError(f"Weather file not found at {weather_file}")
    weather_raw = pd.read_csv(weather_file)
    weather_tracker = QualityAuditTracker("weather_daily")
    clean_weather = clean_weather_daily(weather_raw, weather_tracker)
    clean_weather.to_csv(interim_path / "clean_weather_daily.csv", index=False)
    audit_summary["weather_daily"] = weather_tracker.to_dict()
    logger.info(f"   -> Saved: {interim_path / 'clean_weather_daily.csv'} ({len(clean_weather)} rows)")

    # 4. Generate Reports
    json_report_file = report_path / "data_cleaning_validation_report.json"
    md_report_file = report_path / "data_cleaning_validation_report.md"

    with open(json_report_file, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    logger.info(f"4. JSON validation report written to {json_report_file}")

    generate_markdown_report(audit_summary, md_report_file)
    logger.info(f"5. Markdown validation report written to {md_report_file}")

    return {
        "clean_files": {
            "soil": interim_path / "clean_soil_chemistry.csv",
            "yield": interim_path / "clean_historical_yields.csv",
            "weather": interim_path / "clean_weather_daily.csv"
        },
        "reports": {
            "json": json_report_file,
            "markdown": md_report_file
        },
        "audit": audit_summary
    }


def main():
    parser = argparse.ArgumentParser(description="Day 2: Data Cleaning & Missing Value Imputation")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw datasets directory")
    parser.add_argument("--interim-dir", type=str, default="data/interim", help="Path to save cleaned interim datasets")
    parser.add_argument("--report-dir", type=str, default="reports", help="Path to save validation reports")
    args = parser.parse_args()

    logger.info("Executing Day 2 Data Cleaning & Missing Value Imputation Pipeline...")
    results = run_data_cleaning_pipeline(
        raw_dir=args.raw_dir,
        interim_dir=args.interim_dir,
        report_dir=args.report_dir
    )
    logger.info("Day 2 pipeline execution completed successfully.")
    for k, v in results["clean_files"].items():
        logger.info(f" Clean file [{k}]: {v}")


if __name__ == "__main__":
    main()
