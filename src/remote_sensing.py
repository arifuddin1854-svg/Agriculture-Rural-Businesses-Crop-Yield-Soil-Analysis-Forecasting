"""
Day 13: Satellite Remote Sensing & Vegetation Indices Engine (NDVI, NDWI, EVI)
Member 1 / Production Remote Sensing Module

Responsible for:
- Generating realistic Sentinel-2 / Landsat-8 optical spectral reflectance timeseries
  (Bands: Blue B2, Green B3, Red B4, NIR B8, SWIR B11) at 10-day dekadal compositing intervals
- Deriving biophysical vegetation and canopy water indices:
  * Normalized Difference Vegetation Index (NDVI)
  * Normalized Difference Water Index (NDWI / Gao)
  * Enhanced Vegetation Index (EVI)
- Aggregating phenological canopy metrics:
  * Peak NDVI (maximum vegetative canopy vigor at reproductive phase)
  * Integrated NDVI Area Under Curve (AUC - total photosynthetic biomass proxy)
  * Critical flowering-window NDWI (canopy water stress indicator)
  * Vegetation Condition Index (VCI) normalized against district climatological bounds
- Emitting raw spectral timeseries and processed seasonal canopy feature sets
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
logger = logging.getLogger("RemoteSensing_Day13")


# Crop specific optical signatures and baseline parameters
CROP_SPECTRAL_PROFILES = {
    "Wheat": {"peak_doy": 75, "duration": 150, "peak_nir": 0.52, "peak_red": 0.06, "swir_base": 0.16},
    "Rice": {"peak_doy": 240, "duration": 125, "peak_nir": 0.56, "peak_red": 0.05, "swir_base": 0.14},
    "Maize": {"peak_doy": 225, "duration": 115, "peak_nir": 0.58, "peak_red": 0.05, "swir_base": 0.15},
    "Soybean": {"peak_doy": 230, "duration": 110, "peak_nir": 0.54, "peak_red": 0.06, "swir_base": 0.17},
    "Cotton": {"peak_doy": 260, "duration": 165, "peak_nir": 0.50, "peak_red": 0.08, "swir_base": 0.20},
}


def calculate_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes biophysical vegetation and canopy hydration indices from surface reflectance.
    """
    data = df.copy()
    nir = data["nir"]
    red = data["red"]
    blue = data.get("blue", red * 0.8)
    swir = data.get("swir", red * 2.2)

    # 1. Normalized Difference Vegetation Index (NDVI)
    data["ndvi"] = (nir - red) / (nir + red + 1e-5)
    data["ndvi"] = data["ndvi"].clip(-1.0, 1.0)

    # 2. Normalized Difference Water Index (NDWI)
    data["ndwi"] = (nir - swir) / (nir + swir + 1e-5)
    data["ndwi"] = data["ndwi"].clip(-1.0, 1.0)

    # 3. Enhanced Vegetation Index (EVI)
    evi_denom = nir + 6.0 * red - 7.5 * blue + 1.0
    data["evi"] = 2.5 * (nir - red) / (evi_denom + 1e-5)
    data["evi"] = data["evi"].clip(-1.0, 2.5)

    return data


class SatelliteDataEngine:
    """
    Orchestrates the ingestion, index computation, and phenological aggregation
    of optical satellite remote sensing data for agricultural territories.
    """

    def __init__(
        self,
        raw_output_path: str = "data/raw/satellite_spectral_timeseries.csv",
        processed_output_path: str = "data/processed/canopy_vegetation_indices.csv",
        report_dir: str = "reports",
        seed: int = 42
    ):
        self.raw_output_path = Path(raw_output_path)
        self.processed_output_path = Path(processed_output_path)
        self.report_dir = Path(report_dir)
        self.seed = seed
        np.random.seed(seed)

    def generate_raw_spectral_timeseries(
        self,
        districts: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Simulates 10-day dekadal surface reflectance composites (36 observations/year)
        across agricultural districts over 5 historical years.
        """
        if districts is None:
            districts = [f"DIST_{i:03d}" for i in range(101, 116)]
        if years is None:
            years = [2019, 2020, 2021, 2022, 2023]

        logger.info(f"Generating 10-day satellite spectral timeseries across {len(districts)} districts and {len(years)} years...")

        rows = []
        for dist in districts:
            # District soil background reflectance profile
            soil_brightness = 0.15 + (int(dist.split("_")[1]) % 5) * 0.02
            is_dry_district = dist in ["DIST_102", "DIST_107", "DIST_112"]

            for yr in years:
                is_drought_year = (yr == 2021)
                for dekad in range(1, 37):
                    doy = dekad * 10 - 5  # Day of year center

                    # Seasonal sinusoidal greenup curve (Kharif peak around DOY 235, Rabi peak around DOY 75)
                    kharif_greenup = np.exp(-((doy - 235) ** 2) / (2 * 45 ** 2))
                    rabi_greenup = np.exp(-((doy - 75) ** 2) / (2 * 35 ** 2))
                    combined_greenness = max(kharif_greenup, rabi_greenup * 0.85)

                    if is_drought_year:
                        combined_greenness *= 0.78
                    if is_dry_district:
                        combined_greenness *= 0.88

                    # Reflectance bands with realistic atmospheric noise
                    noise = np.random.normal(0, 0.015)
                    red_refl = max(0.02, soil_brightness * (1.0 - combined_greenness * 0.70) + noise)
                    nir_refl = min(0.65, 0.18 + combined_greenness * 0.40 + noise)
                    blue_refl = max(0.01, red_refl * 0.65 + np.random.normal(0, 0.008))
                    swir_refl = min(0.40, 0.22 - combined_greenness * 0.10 + (0.06 if is_drought_year else 0.0) + noise)

                    date_str = f"{yr}-{(doy // 31) + 1:02d}-{(doy % 28) + 1:02d}"

                    rows.append({
                        "district_id": dist,
                        "year": yr,
                        "dekad": dekad,
                        "day_of_year": doy,
                        "observation_date": date_str,
                        "blue": round(blue_refl, 4),
                        "red": round(red_refl, 4),
                        "nir": round(nir_refl, 4),
                        "swir": round(swir_refl, 4),
                        "cloud_cover_pct": round(np.random.uniform(0.0, 12.0), 1)
                    })

        df_raw = pd.DataFrame(rows)
        # Compute spectral indices
        df_raw = calculate_spectral_indices(df_raw)

        self.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(self.raw_output_path, index=False)
        logger.info(f"Saved raw satellite spectral timeseries: {self.raw_output_path} ({len(df_raw)} records)")
        return df_raw

    def aggregate_phenological_canopy_metrics(
        self,
        spectral_df: pd.DataFrame,
        yields_path: str = "data/processed/unified_agri_dataset.csv"
    ) -> pd.DataFrame:
        """
        Aligns dekadal satellite observations with historical crop seasons
        to compute seasonal canopy indicators for each district, year, and crop.
        """
        y_path = Path(yields_path)
        if not y_path.exists():
            raise FileNotFoundError(f"Unified agricultural dataset required at {y_path}")

        logger.info(f"Loading agricultural master records from {y_path} to align phenological satellite metrics...")
        df_master = pd.read_csv(y_path)
        logger.info(f"Aligning with {len(df_master)} crop instances across 5 crops...")

        canopy_rows = []
        for idx, row in df_master.iterrows():
            dist = row["district_id"]
            yr = int(row["year"])
            crop = row["crop_type"]

            # Filter spectral observations for this district & year
            dist_spectral = spectral_df[
                (spectral_df["district_id"] == dist) &
                (spectral_df["year"] == yr)
            ]

            if dist_spectral.empty:
                # Fallback to general district mean
                dist_spectral = spectral_df[spectral_df["district_id"] == dist]

            # Crop-specific phenology window
            prof = CROP_SPECTRAL_PROFILES.get(crop, CROP_SPECTRAL_PROFILES["Wheat"])
            peak_doy = prof["peak_doy"]

            # Rabi crops (Wheat) peak in DOY 40-110; Kharif crops peak in DOY 200-270
            if crop == "Wheat":
                season_sub = dist_spectral[(dist_spectral["day_of_year"] <= 120) | (dist_spectral["day_of_year"] >= 310)]
            else:
                season_sub = dist_spectral[(dist_spectral["day_of_year"] >= 160) & (dist_spectral["day_of_year"] <= 300)]

            if season_sub.empty:
                season_sub = dist_spectral

            # 1. Peak NDVI & Timing
            peak_ndvi = float(season_sub["ndvi"].max())
            peak_ndvi_doy = int(season_sub.loc[season_sub["ndvi"].idxmax(), "day_of_year"])

            # 2. Mean & Integrated NDVI (AUC)
            mean_ndvi = float(season_sub["ndvi"].mean())
            ndvi_auc = float(season_sub["ndvi"].sum() * 10.0)  # Dekadal integral

            # 3. Enhanced Vegetation Index (EVI)
            peak_evi = float(season_sub["evi"].max())
            mean_evi = float(season_sub["evi"].mean())

            # 4. Canopy Water Index (NDWI)
            mean_ndwi = float(season_sub["ndwi"].mean())
            flowering_ndwi = float(season_sub.loc[(season_sub["day_of_year"] - peak_doy).abs().idxmin(), "ndwi"])

            # 5. Canopy Stress Indicators
            water_stress_dekads = int((season_sub["ndwi"] < 0.10).sum())
            canopy_senescence_rate = float(max(0.0, (peak_ndvi - season_sub["ndvi"].iloc[-1]) / 40.0))

            # 6. Vegetation Condition Index (VCI)
            # Relative to historical max/min NDVI for the district
            dist_all_ndvi = spectral_df[spectral_df["district_id"] == dist]["ndvi"]
            min_hist_ndvi = float(dist_all_ndvi.min())
            max_hist_ndvi = float(dist_all_ndvi.max())
            vci_score = float(np.clip(((peak_ndvi - min_hist_ndvi) / (max_hist_ndvi - min_hist_ndvi + 1e-5)) * 100.0, 0.0, 100.0))

            canopy_rows.append({
                "district_id": dist,
                "year": yr,
                "crop_type": crop,
                "satellite_peak_ndvi": round(peak_ndvi, 4),
                "satellite_peak_ndvi_doy": peak_ndvi_doy,
                "satellite_mean_ndvi": round(mean_ndvi, 4),
                "satellite_ndvi_auc": round(ndvi_auc, 2),
                "satellite_peak_evi": round(peak_evi, 4),
                "satellite_mean_evi": round(mean_evi, 4),
                "satellite_mean_ndwi": round(mean_ndwi, 4),
                "satellite_flowering_ndwi": round(flowering_ndwi, 4),
                "canopy_water_stress_dekads": water_stress_dekads,
                "canopy_senescence_rate": round(canopy_senescence_rate, 4),
                "vegetation_condition_index_vci": round(vci_score, 2)
            })

        df_canopy = pd.DataFrame(canopy_rows)
        self.processed_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_canopy.to_csv(self.processed_output_path, index=False)
        logger.info(f"Saved processed canopy vegetation indices: {self.processed_output_path} ({len(df_canopy)} rows)")
        return df_canopy

    def generate_audit_reports(
        self,
        df_canopy: pd.DataFrame,
        elapsed_sec: float
    ) -> Dict[str, Any]:
        """
        Emits JSON and Markdown audit reports for the satellite remote sensing engine.
        """
        self.report_dir.mkdir(parents=True, exist_ok=True)
        total_records = len(df_canopy)

        # Crop statistics
        crop_summary = {}
        for crop, grp in df_canopy.groupby("crop_type"):
            crop_summary[crop] = {
                "records": len(grp),
                "mean_peak_ndvi": round(float(grp["satellite_peak_ndvi"].mean()), 3),
                "mean_ndvi_auc": round(float(grp["satellite_ndvi_auc"].mean()), 1),
                "mean_flowering_ndwi": round(float(grp["satellite_flowering_ndwi"].mean()), 3),
                "mean_vci": round(float(grp["vegetation_condition_index_vci"].mean()), 1)
            }

        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_stage": "Day 13 - Satellite Remote Sensing Engine",
            "total_records": total_records,
            "districts_covered": int(df_canopy["district_id"].nunique()),
            "crops_covered": int(df_canopy["crop_type"].nunique()),
            "execution_seconds": round(elapsed_sec, 2),
            "spectral_kpis": {
                "regional_mean_peak_ndvi": round(float(df_canopy["satellite_peak_ndvi"].mean()), 3),
                "regional_mean_ndvi_auc": round(float(df_canopy["satellite_ndvi_auc"].mean()), 1),
                "regional_mean_ndwi": round(float(df_canopy["satellite_mean_ndwi"].mean()), 3),
                "regional_mean_vci": round(float(df_canopy["vegetation_condition_index_vci"].mean()), 1)
            },
            "crop_summary": crop_summary
        }

        # 1. JSON Report
        json_path = self.report_dir / "remote_sensing_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2)

        # 2. Markdown Report
        md_path = self.report_dir / "remote_sensing_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 🛰️ Day 13: Satellite Remote Sensing & Vegetation Indices Audit Report\n\n")
            f.write(f"**Execution Timestamp**: {audit_data['timestamp']}  \n")
            f.write(f"**Coverage**: {total_records} seasonal instances across {audit_data['districts_covered']} districts  \n\n")

            f.write("## 📊 Regional Biophysical Canopy KPIs\n\n")
            f.write("| Biophysical Indicator | Regional Value | Interpretation |\n")
            f.write("| :--- | :---: | :--- |\n")
            f.write(f"| **Mean Peak NDVI** | **{audit_data['spectral_kpis']['regional_mean_peak_ndvi']:.3f}** | Robust vegetative canopy density at peak reproductive stage |\n")
            f.write(f"| **Mean Integrated NDVI (AUC)** | **{audit_data['spectral_kpis']['regional_mean_ndvi_auc']:.1f}** | Total seasonal accumulated photosynthetic biomass |\n")
            f.write(f"| **Mean Canopy NDWI** | **{audit_data['spectral_kpis']['regional_mean_ndwi']:.3f}** | Canopy water content & cellular hydration status |\n")
            f.write(f"| **Vegetation Condition Index (VCI)** | **{audit_data['spectral_kpis']['regional_mean_vci']:.1f}%** | Vigor ranking against 5-year historical district envelope |\n\n")

            f.write("## 🌾 Crop-Specific Canopy Spectral Response\n\n")
            f.write("| Crop Species | Records | Peak NDVI | NDVI AUC | Flowering NDWI | VCI Score |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
            for crop, s in crop_summary.items():
                f.write(f"| **{crop}** | {s['records']} | {s['mean_peak_ndvi']:.3f} | {s['mean_ndvi_auc']:.1f} | {s['mean_flowering_ndwi']:.3f} | **{s['mean_vci']:.1f}%** |\n")
            f.write("\n")
            f.write("```bash\n# Reproduce Day 13 Remote Sensing Pipeline\npython -m src.remote_sensing\n```\n")

        logger.info(f"Saved audit reports to {json_path} and {md_path}")
        return audit_data


def main():
    import time
    t0 = time.time()
    engine = SatelliteDataEngine()
    df_raw = engine.generate_raw_spectral_timeseries()
    df_canopy = engine.aggregate_phenological_canopy_metrics(df_raw)
    elapsed = time.time() - t0
    res = engine.generate_audit_reports(df_canopy, elapsed)

    print("\n" + "=" * 75)
    print(f"[*] DAY 13 SATELLITE ENGINE: Processed {res['total_records']} records in {elapsed:.2f}s")
    print(f"[*] Mean Peak NDVI: {res['spectral_kpis']['regional_mean_peak_ndvi']:.3f} | VCI Score: {res['spectral_kpis']['regional_mean_vci']:.1f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
