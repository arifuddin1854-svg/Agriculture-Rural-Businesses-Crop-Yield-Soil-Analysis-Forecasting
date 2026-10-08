"""
Week 1: Geospatial & Weather Data Integration Pipeline

Responsible for:
- Aggregating daily weather data into phenological growing-season metrics
- Merging meteorology with static soil chemical data (NPK, pH) via District/County IDs
- Producing the unified master dataset
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def aggregate_weather(weather_df: pd.DataFrame, group_col: str = "district_id", year_col: str = "year") -> pd.DataFrame:
    """
    Aggregates daily weather data into growing-season agronomic summary indicators.
    Calculates cumulative rainfall, heat stress days, and mean temperatures.
    """
    logger.info("Aggregating daily weather records to growing-season metrics...")
    
    # Example aggregations (adjust column names based on dataset)
    agg_dict = {}
    if "precipitation" in weather_df.columns:
        agg_dict["precipitation"] = ["sum", "mean", "std"]
    if "temp_max" in weather_df.columns:
        agg_dict["temp_max"] = ["mean", "max"]
    if "temp_min" in weather_df.columns:
        agg_dict["temp_min"] = ["mean", "min"]
    if "humidity" in weather_df.columns:
        agg_dict["humidity"] = ["mean"]

    if not agg_dict:
        logger.warning("Standard weather columns not found, using generic aggregations.")
        return weather_df

    grouped = weather_df.groupby([group_col, year_col]).agg(agg_dict)
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    return grouped.reset_index()


def fuse_datasets(weather_agg_df: pd.DataFrame, soil_df: pd.DataFrame, yield_df: pd.DataFrame, key: str = "district_id") -> pd.DataFrame:
    """
    Merges temporal aggregated weather, static soil profiles, and crop yields.
    """
    logger.info(f"Merging datasets using spatial key '{key}'...")
    merged = pd.merge(yield_df, soil_df, on=key, how="inner")
    if "year" in weather_agg_df.columns and "year" in merged.columns:
        final_df = pd.merge(merged, weather_agg_df, on=[key, "year"], how="inner")
    else:
        final_df = pd.merge(merged, weather_agg_df, on=key, how="inner")
    
    logger.info(f"Fusion complete. Shape: {final_df.shape}")
    return final_df


def main():
    parser = argparse.ArgumentParser(description="Fuse Weather, Soil, and Yield Data")
    parser.add_argument("--weather-path", type=str, default="data/raw/weather_daily.csv", help="Path to raw weather data")
    parser.add_argument("--soil-path", type=str, default="data/raw/soil_chemistry.csv", help="Path to raw soil chemistry data")
    parser.add_argument("--yield-path", type=str, default="data/raw/historical_yields.csv", help="Path to raw crop yield data")
    parser.add_argument("--output-path", type=str, default="data/processed/unified_raw.parquet", help="Path to save fused dataset")
    args = parser.parse_args()

    logger.info("Week 1 Data Fusion pipeline initialized.")


if __name__ == "__main__":
    main()
