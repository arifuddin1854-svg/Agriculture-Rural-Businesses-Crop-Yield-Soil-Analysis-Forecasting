"""
Member 1: Data Integration & Data Engineer
Day 1: Multi-Source Data Ingestion & Raw Agricultural Dataset Generator

Generates and ingests realistic agronomic datasets:
1. Historical Crop Yields (data/raw/historical_yields.csv)
2. Soil Chemical Composition (data/raw/soil_chemistry.csv)
3. Daily Meteorological Records (data/raw/weather_daily.csv)
"""

import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Set thread count for BLAS to avoid Windows allocation issues
os.environ["OPENBLAS_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataIngestion_Day1")


def generate_raw_datasets(output_dir: str = "data/raw", seed: int = 42) -> dict[str, Path]:
    """Generates realistic agronomic datasets matching Axlero's requirements."""
    np.random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    districts = [f"DIST_{i:03d}" for i in range(101, 116)]  # 15 distinct districts
    years = [2019, 2020, 2021, 2022, 2023]
    crops = ["Wheat", "Maize", "Rice", "Soybean", "Cotton"]
    soil_types = ["Alluvial", "Black Soil", "Red Sandy Loam", "Clay Loam", "Laterite"]

    logger.info("1. Generating static soil chemical composition dataset...")
    soil_rows = []
    for d_id in districts:
        soil_type = np.random.choice(soil_types)
        # N-P-K levels in kg/ha, pH, electrical conductivity, organic carbon %
        n_val = np.random.uniform(60, 160)
        p_val = np.random.uniform(20, 65)
        k_val = np.random.uniform(80, 220)
        ph_val = np.clip(np.random.normal(6.5, 0.7), 4.5, 8.5)
        som_val = np.clip(np.random.normal(1.2, 0.3), 0.4, 2.5)  # Soil Organic Matter %
        ec_val = np.clip(np.random.normal(0.45, 0.15), 0.1, 1.2)  # dS/m
        
        soil_rows.append({
            "district_id": d_id,
            "soil_type": soil_type,
            "nitrogen_kg_ha": round(n_val, 2),
            "phosphorus_kg_ha": round(p_val, 2),
            "potassium_kg_ha": round(k_val, 2),
            "soil_ph": round(ph_val, 2),
            "organic_matter_pct": round(som_val, 2),
            "electrical_conductivity_ds_m": round(ec_val, 2)
        })
    soil_df = pd.DataFrame(soil_rows)
    soil_file = out_path / "soil_chemistry.csv"
    soil_df.to_csv(soil_file, index=False)
    logger.info(f"   -> Saved: {soil_file} ({len(soil_df)} rows)")

    logger.info("2. Generating historical crop yield records...")
    yield_rows = []
    base_yields = {"Wheat": 4.5, "Maize": 5.8, "Rice": 4.2, "Soybean": 2.8, "Cotton": 2.2}
    
    for year in years:
        for d_id in districts:
            # Random selection of 3 crops grown per district-year
            grown_crops = np.random.choice(crops, size=3, replace=False)
            for crop in grown_crops:
                area_ha = round(float(np.random.uniform(500, 3500)), 1)
                base = base_yields[crop]
                # Modulate yield by year effect (e.g. 2021 was drought-like)
                year_factor = 0.85 if year == 2021 else 1.05 if year == 2020 else 1.0
                yield_val = np.random.normal(base * year_factor, 0.45)
                yield_val = max(0.8, round(float(yield_val), 2))
                production_mt = round(yield_val * area_ha, 1)

                yield_rows.append({
                    "district_id": d_id,
                    "year": year,
                    "crop_type": crop,
                    "planted_area_ha": area_ha,
                    "production_mt": production_mt,
                    "crop_yield_mt_ha": yield_val
                })
    yield_df = pd.DataFrame(yield_rows)
    yield_file = out_path / "historical_yields.csv"
    yield_df.to_csv(yield_file, index=False)
    logger.info(f"   -> Saved: {yield_file} ({len(yield_df)} records)")

    logger.info("3. Generating high-frequency daily weather observations (5 years x 15 districts)...")
    weather_records = []
    # Generate daily records for each district over the 5 years
    date_range = pd.date_range(start="2019-01-01", end="2023-12-31", freq="D")
    
    for d_id in districts:
        district_temp_offset = np.random.uniform(-2.0, 3.0)
        district_rain_scale = np.random.uniform(0.8, 1.3)
        
        for date in date_range:
            doy = date.dayofyear
            # Seasonal temperature curve (hottest mid-year, cooler in winter)
            mean_temp = 24.0 + 8.0 * np.sin(2 * np.pi * (doy - 105) / 365) + district_temp_offset
            t_max = mean_temp + np.random.uniform(4.0, 9.0)
            t_min = mean_temp - np.random.uniform(4.0, 8.0)
            
            # Monsoon rain pattern (peak around days 150-270)
            is_monsoon = 150 <= doy <= 270
            rain_prob = 0.45 if is_monsoon else 0.08
            if np.random.rand() < rain_prob:
                precip = np.random.exponential(scale=18.0 * district_rain_scale)
                precip = round(min(precip, 140.0), 1)
            else:
                precip = 0.0
                
            humidity = np.clip(np.random.normal(75 if is_monsoon else 48, 12), 20, 98)
            radiation = np.clip(np.random.normal(18.5, 4.0), 6.0, 28.0)  # MJ/m2
            
            weather_records.append({
                "district_id": d_id,
                "date": date.strftime("%Y-%m-%d"),
                "year": date.year,
                "month": date.month,
                "temp_max": round(t_max, 1),
                "temp_min": round(t_min, 1),
                "precipitation_mm": precip,
                "relative_humidity": round(humidity, 1),
                "solar_radiation_mj": round(radiation, 1)
            })

    weather_df = pd.DataFrame(weather_records)
    weather_file = out_path / "weather_daily.csv"
    weather_df.to_csv(weather_file, index=False)
    logger.info(f"   -> Saved: {weather_file} ({len(weather_df)} daily records)")

    return {
        "soil": soil_file,
        "yield": yield_file,
        "weather": weather_file
    }


def main():
    logger.info("Executing Day 1 Data Ingestion Pipeline...")
    files = generate_raw_datasets()
    logger.info("Day 1 deliverables successfully created.")
    for k, v in files.items():
        logger.info(f" - {k}: {v}")


if __name__ == "__main__":
    main()
