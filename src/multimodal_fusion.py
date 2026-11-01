"""
Day 14: Multi-Modal Data Fusion with Satellite Vegetation Features
Member 1 / Production Multi-Modal Pipeline

Responsible for:
- Harmonizing multi-source modalities:
  * Static soil chemistry (N-P-K, pH, SOM, EC)
  * Growing-season meteorological aggregations (GDD, rain, VPD, heat stress)
  * Optical satellite remote sensing (Peak NDVI, NDWI, EVI, VCI)
  * Annual historical crop yields
- Validating 100% referential integrity across composite keys (district_id, year, crop_type)
- Deriving multi-modal agronomic interaction metrics:
  * Photosynthetic Radiation Use Efficiency (PRUE proxy: NDVI AUC / GDD)
  * Canopy Hydration Deficit Index (NDWI x VPD interaction)
  * Soil-Canopy Coupling Ratio (Root-zone Nitrogen x Peak Canopy Greenness)
- Exporting master multi-modal dataset (CSV & Parquet)
- Rendering publication-quality visual artifacts in reports/figures/
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MultiModalFusion_Day14")


def fuse_multimodal_datasets(
    agri_dataset_path: str = "data/processed/unified_agri_dataset.csv",
    satellite_indices_path: str = "data/processed/canopy_vegetation_indices.csv",
    output_csv_path: str = "data/processed/multimodal_agri_dataset.csv",
    output_parquet_path: str = "data/processed/multimodal_agri_dataset.parquet"
) -> pd.DataFrame:
    """
    Fuses unified agricultural records with satellite vegetation indices
    and derives multi-modal interaction features.
    """
    agri_file = Path(agri_dataset_path)
    sat_file = Path(satellite_indices_path)

    if not agri_file.exists():
        raise FileNotFoundError(f"Agricultural dataset not found: {agri_file}")
    if not sat_file.exists():
        raise FileNotFoundError(f"Satellite indices dataset not found: {sat_file}")

    logger.info(f"Loading agricultural master records from {agri_file}...")
    df_agri = pd.read_csv(agri_file)
    logger.info(f"Loading satellite canopy metrics from {sat_file}...")
    df_sat = pd.read_csv(sat_file)

    join_keys = ["district_id", "year", "crop_type"]
    logger.info(f"Executing referential join on {join_keys}...")

    df_fused = pd.merge(df_agri, df_sat, on=join_keys, how="inner")

    if len(df_fused) != len(df_agri):
        logger.warning(f"Row count mismatch after join: {len(df_fused)} fused vs {len(df_agri)} initial.")
    else:
        logger.info(f"Referential integrity verified: 100% match ({len(df_fused)} rows).")

    # -------------------------------------------------------------
    # Derive Multi-Modal Agronomic Interaction Features
    # -------------------------------------------------------------
    logger.info("Deriving multi-modal soil-weather-satellite interaction metrics...")

    # 1. Photosynthetic Radiation Use Efficiency (PRUE)
    # Total accumulated greenness per degree day of thermal development
    gdd = df_fused["gdd_cumulative"]
    df_fused["photosynthetic_radiation_efficiency"] = (df_fused["satellite_ndvi_auc"] / (gdd + 1e-5)).round(4)

    # 2. Canopy Hydration Deficit Index
    # When NDWI is low and atmospheric vapor pressure deficit (VPD) is high
    vpd = df_fused["mean_vpd_kpa"]
    df_fused["canopy_hydration_deficit_index"] = ((1.0 - df_fused["satellite_mean_ndwi"]) * vpd).round(4)

    # 3. Soil-Canopy Coupling Ratio
    # Nitrogen availability converted to actual green canopy density
    n = df_fused["nitrogen_kg_ha"]
    df_fused["soil_nitrogen_canopy_coupling"] = ((n / 120.0) * df_fused["satellite_peak_ndvi"]).round(4)

    # 4. Thermal Senescence Acceleration
    # Ratio of heat shock days to peak vegetative health
    heat_days = df_fused["heat_stress_days"]
    df_fused["thermal_senescence_risk"] = (heat_days / (df_fused["satellite_peak_ndvi"] * 100.0 + 1e-5)).round(4)

    # Export
    out_csv = Path(output_csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_fused.to_csv(out_csv, index=False)
    logger.info(f"Saved multi-modal CSV dataset to {out_csv} ({df_fused.shape[0]} rows, {df_fused.shape[1]} cols)")

    out_pq = Path(output_parquet_path)
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    df_fused.to_parquet(out_pq, index=False)
    logger.info(f"Saved multi-modal Parquet dataset to {out_pq}")

    return df_fused


def render_multimodal_visualizations(
    df_multimodal: pd.DataFrame,
    figures_dir: str = "reports/figures"
):
    """
    Renders publication-quality multi-modal visualization artifacts.
    """
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]

    # -------------------------------------------------------------
    # Figure 1: Peak NDVI vs Crop Yield across Species
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    palette = {"Wheat": "#d4a373", "Rice": "#2a9d8f", "Maize": "#e76f51", "Soybean": "#588157", "Cotton": "#3a86ff"}

    for crop in df_multimodal["crop_type"].unique():
        sub = df_multimodal[df_multimodal["crop_type"] == crop]
        color = palette.get(crop, "#333333")
        sns.regplot(
            data=sub,
            x="satellite_peak_ndvi",
            y="crop_yield_mt_ha",
            label=f"{crop} (r = {sub['satellite_peak_ndvi'].corr(sub['crop_yield_mt_ha']):.2f})",
            ax=ax,
            color=color,
            scatter_kws={"alpha": 0.65, "s": 50},
            line_kws={"linewidth": 2}
        )

    ax.set_title("Multi-Modal Agronomic Alignment: Satellite Peak NDVI vs. Crop Yield", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Sentinel-2 / Landsat-8 Peak NDVI", fontsize=11, labelpad=8)
    ax.set_ylabel("Reported Harvest Yield (MT/Ha)", fontsize=11, labelpad=8)
    ax.legend(title="Crop Species & Correlation", frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.tight_layout()

    fig_path1 = fig_dir / "05_vegetation_index_vs_yield.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {fig_path1}")

    # -------------------------------------------------------------
    # Figure 2: Photosynthetic Efficiency Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.boxplot(
        data=df_multimodal,
        x="crop_type",
        y="photosynthetic_radiation_efficiency",
        hue="crop_type",
        palette="Greens_d",
        legend=False,
        ax=ax,
        boxprops=dict(alpha=0.8)
    )
    sns.stripplot(
        data=df_multimodal,
        x="crop_type",
        y="photosynthetic_radiation_efficiency",
        color="#1b4332",
        alpha=0.45,
        jitter=0.2,
        ax=ax
    )
    ax.set_title("Photosynthetic Radiation Efficiency (NDVI AUC / GDD) Across Crop Species", fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Crop Species", fontsize=11)
    ax.set_ylabel("PRUE Metric (Biomass / Thermal Unit)", fontsize=11)
    plt.tight_layout()

    fig_path2 = fig_dir / "05_ndvi_seasonal_trajectories.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    logger.info(f"Saved figure: {fig_path2}")


def generate_multimodal_report(
    df: pd.DataFrame,
    elapsed_sec: float,
    report_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Generates machine-readable JSON and Markdown multi-modal fusion audit reports.
    """
    rep_dir = Path(report_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    corr_peak_ndvi = float(df["satellite_peak_ndvi"].corr(df["crop_yield_mt_ha"]))
    corr_prue = float(df["photosynthetic_radiation_efficiency"].corr(df["crop_yield_mt_ha"]))

    crop_metrics = {}
    for crop, grp in df.groupby("crop_type"):
        crop_metrics[crop] = {
            "records": len(grp),
            "mean_yield_mt_ha": round(float(grp["crop_yield_mt_ha"].mean()), 2),
            "mean_peak_ndvi": round(float(grp["satellite_peak_ndvi"].mean()), 3),
            "ndvi_yield_correlation": round(float(grp["satellite_peak_ndvi"].corr(grp["crop_yield_mt_ha"])), 3),
            "mean_prue": round(float(grp["photosynthetic_radiation_efficiency"].mean()), 4)
        }

    audit_data = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_stage": "Day 14 - Multi-Modal Data Fusion",
        "total_records": len(df),
        "total_features": len(df.columns),
        "execution_seconds": round(elapsed_sec, 2),
        "multi_modal_correlations": {
            "peak_ndvi_vs_yield": round(corr_peak_ndvi, 3),
            "prue_vs_yield": round(corr_prue, 3)
        },
        "crop_breakdown": crop_metrics
    }

    # 1. JSON Report
    json_path = rep_dir / "multimodal_fusion_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    # 2. Markdown Report
    md_path = rep_dir / "multimodal_fusion_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🛰️🌱 Day 14: Multi-Modal Data Fusion Audit Report\n\n")
        f.write(f"**Execution Timestamp**: {audit_data['timestamp']}  \n")
        f.write(f"**Unified Records**: {audit_data['total_records']} | **Total Modalities**: Soil + Weather + Satellite Optical Bands  \n\n")

        f.write("## 📊 Multi-Modal Agronomic Predictive Power\n\n")
        f.write("| Feature Category | Pearson Correlation (r) vs Yield | Predictive Impact |\n")
        f.write("| :--- | :---: | :--- |\n")
        f.write(f"| **Satellite Peak NDVI** | **r = {corr_peak_ndvi:+.3f}** | Direct biophysical proxy for vegetative canopy vigor at reproduction |\n")
        f.write(f"| **Photosynthetic Efficiency (PRUE)** | **r = {corr_prue:+.3f}** | Normalized thermal biomass accumulation per degree-day |\n\n")

        f.write("## 🌾 Crop-Specific Multi-Modal Response Matrix\n\n")
        f.write("| Crop Species | Records | Mean Yield (MT/Ha) | Mean Peak NDVI | NDVI-Yield Correlation (r) | PRUE Metric |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for crop, m in crop_metrics.items():
            f.write(f"| **{crop}** | {m['records']} | {m['mean_yield_mt_ha']:.2f} | {m['mean_peak_ndvi']:.3f} | **{m['ndvi_yield_correlation']:+.3f}** | {m['mean_prue']:.4f} |\n")
        f.write("\n")
        f.write("```bash\n# Reproduce Day 14 Multi-Modal Fusion\npython -m src.multimodal_fusion\n```\n")

    logger.info(f"Saved audit reports to {json_path} and {md_path}")
    return audit_data


def main():
    t0 = time.time()
    df_fused = fuse_multimodal_datasets()
    render_multimodal_visualizations(df_fused)
    elapsed = time.time() - t0
    res = generate_multimodal_report(df_fused, elapsed)

    print("\n" + "=" * 75)
    print(f"[*] DAY 14 MULTI-MODAL FUSION: Fused {res['total_records']} records across {res['total_features']} features in {elapsed:.2f}s")
    print(f"[*] Peak NDVI vs Yield Correlation: r = {res['multi_modal_correlations']['peak_ndvi_vs_yield']:+.3f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
