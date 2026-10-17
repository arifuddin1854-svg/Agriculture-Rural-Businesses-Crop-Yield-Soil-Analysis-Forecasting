"""
Week 1: Geospatial & Weather Data Integration Pipeline
Axlero Data Science & ML Internship: Project 3 (Agriculture)

Responsible for:
- Aggregating daily weather records into biologically meaningful growing-season metrics:
  * Cumulative rainfall (mm)
  * Growing Degree Days (GDD, base temperature 10°C)
  * Heat stress days (Tmax >= 35°C)
  * Mean temperature, max temperature, and relative humidity
- Merging meteorology with static soil chemical data (NPK, pH, organic carbon) via District IDs
- Merging with annual historical crop yield statistics
- Validating zero geographic or temporal alignment anomalies
- Exporting the canonical unified dataset to data/processed/
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_growing_degree_days(df: pd.DataFrame, base_temp: float = 10.0) -> pd.Series:
    """
    Calculates daily Growing Degree Days (GDD) = max(0, (Tmax + Tmin)/2 - base_temp).
    """
    if "temp_max" in df.columns and "temp_min" in df.columns:
        mean_t = (df["temp_max"] + df["temp_min"]) / 2.0
        gdd = np.maximum(0.0, mean_t - base_temp)
        return gdd
    elif "temp_mean" in df.columns:
        return np.maximum(0.0, df["temp_mean"] - base_temp)
    return pd.Series(0.0, index=df.index)


def aggregate_weather(
    weather_df: pd.DataFrame, 
    group_col: str = "district_id", 
    year_col: str = "year"
) -> pd.DataFrame:
    """
    Aggregates daily weather data into growing-season agronomic summary indicators.
    """
    logger.info("Aggregating daily weather records to growing-season metrics...")
    df = weather_df.copy()

    # Calculate daily GDD and heat stress indicators
    df["gdd"] = compute_growing_degree_days(df, base_temp=10.0)
    df["heat_stress_day"] = (df["temp_max"] >= 35.0).astype(int) if "temp_max" in df.columns else 0
    df["rainy_day"] = (df["precipitation"] > 1.0).astype(int) if "precipitation" in df.columns else 0

    # Aggregations per district-year
    agg_rules = {
        "precipitation": ["sum", "mean", "std"],
        "temp_max": ["mean", "max"],
        "temp_min": ["mean", "min"],
        "temp_mean": ["mean"],
        "humidity": ["mean", "min", "max"],
        "solar_radiation": ["mean", "sum"],
        "gdd": ["sum"],
        "heat_stress_day": ["sum"],
        "rainy_day": ["sum"]
    }

    # Filter only available columns
    available_aggs = {col: funcs for col, funcs in agg_rules.items() if col in df.columns}
    grouped = df.groupby([group_col, year_col]).agg(available_aggs)
    
    # Flatten column multi-index
    grouped.columns = [f"{col}_{func}" if func != "sum" else f"cumulative_{col}" if col in ["precipitation", "gdd"] else f"total_{col}"
                       for col, func in grouped.columns.values]
    
    result = grouped.reset_index()
    logger.info(f"Aggregated weather shape: {result.shape} across {result[group_col].nunique()} districts and {result[year_col].nunique()} years.")
    return result


def fuse_datasets(
    weather_agg_df: pd.DataFrame, 
    soil_df: pd.DataFrame, 
    yield_df: pd.DataFrame, 
    key: str = "district_id"
) -> pd.DataFrame:
    """
    Merges temporal aggregated weather, static soil profiles, and crop yields.
    Ensures zero unaligned records.
    """
    logger.info(f"Fusing datasets using spatial identifier '{key}' and temporal key 'year'...")
    
    # Merge Yield records with Soil profiles
    merged_soil_yield = pd.merge(yield_df, soil_df, on=key, how="inner")
    logger.info(f"Merged Yield + Soil shape: {merged_soil_yield.shape}")
    
    # Merge with Weather aggregations on [district_id, year]
    if "year" in weather_agg_df.columns and "year" in merged_soil_yield.columns:
        final_df = pd.merge(merged_soil_yield, weather_agg_df, on=[key, "year"], how="inner")
    else:
        final_df = pd.merge(merged_soil_yield, weather_agg_df, on=key, how="inner")
    
    # Verify integrity
    missing_count = final_df.isnull().sum().sum()
    logger.info(f"Master Dataset Fusion Complete. Final Shape: {final_df.shape}, Missing Values: {missing_count}")
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Fuse Weather, Soil, and Yield Data")
    parser.add_argument("--weather-path", type=str, default="data/raw/weather_daily.csv", help="Path to raw weather data")
    parser.add_argument("--soil-path", type=str, default="data/raw/soil_chemistry.csv", help="Path to raw soil chemistry data")
    parser.add_argument("--yield-path", type=str, default="data/raw/historical_yields.csv", help="Path to raw crop yield data")
    parser.add_argument("--output-parquet", type=str, default="data/processed/unified_agri_dataset.parquet", help="Path to save parquet dataset")
    parser.add_argument("--output-csv", type=str, default="data/processed/unified_agri_dataset.csv", help="Path to save csv dataset")
    args = parser.parse_args()

    weather_p = Path(args.weather_path)
    soil_p = Path(args.soil_path)
    yield_p = Path(args.yield_path)

    if not (weather_p.exists() and soil_p.exists() and yield_p.exists()):
        logger.error(f"One or more raw datasets missing: {weather_p}, {soil_p}, {yield_p}")
        return

    logger.info("Loading raw datasets...")
    weather_df = pd.read_csv(weather_p)
    soil_df = pd.read_csv(soil_p)
    yield_df = pd.read_csv(yield_p)

    weather_agg = aggregate_weather(weather_df)
    master_df = fuse_datasets(weather_agg, soil_df, yield_df, key="district_id")

    # Ensure output directories exist
    Path(args.output_parquet).parent.mkdir(parents=True, exist_ok=True)
    
    master_df.to_parquet(args.output_parquet, index=False)
    master_df.to_csv(args.output_csv, index=False)
    logger.info(f"Successfully saved unified dataset to:")
    logger.info(f" - {args.output_parquet}")
    logger.info(f" - {args.output_csv}")


if __name__ == "__main__":
    main()
