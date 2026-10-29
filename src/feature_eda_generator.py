"""
Day 8: Feature Analysis, Visual Artifacts & Notebook 02 Generator
Generates:
- 02_npk_stoichiometry_vs_yield.png
- 02_hydrothermal_stress_surface.png
- 02_feature_correlation_ranking.png
- notebooks/02_feature_engineering_eda.ipynb
- reports/feature_analysis_eda_report.md
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FeatureEDA_Day8")


def generate_feature_eda_artifacts(
    features_path: str = "data/processed/train_features.csv",
    figures_dir: str = "reports/figures",
    report_dir: str = "reports"
) -> Dict[str, Path]:
    """
    Renders publication visual figures and analysis summary for Day 8.
    """
    feat_p = Path(features_path)
    fig_d = Path(figures_dir)
    rep_d = Path(report_dir)

    fig_d.mkdir(parents=True, exist_ok=True)
    rep_d.mkdir(parents=True, exist_ok=True)

    if not feat_p.exists():
        raise FileNotFoundError(f"Feature store not found at: {feat_p}")

    df = pd.read_csv(feat_p)
    logger.info(f"Loaded {len(df)} engineered records with {len(df.columns)} features.")

    sns.set_theme(style="whitegrid", font="sans-serif")

    # -----------------------------------------------------------------
    # Figure 1: NPK Stoichiometry vs. Crop Yield
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(
        data=df, x="n_to_p_ratio", y="crop_yield_mt_ha", hue="crop_type",
        palette="tab10", s=60, alpha=0.8, ax=axes[0]
    )
    sns.regplot(
        data=df, x="n_to_p_ratio", y="crop_yield_mt_ha",
        scatter=False, color="darkred", ax=axes[0],
        line_kws={"linewidth": 2, "linestyle": "--"}
    )
    axes[0].set_title("N:P Stoichiometric Ratio vs. Crop Yield", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Nitrogen to Phosphorus Ratio (N:P)", fontsize=10)
    axes[0].set_ylabel("Yield (Metric Tons / Hectare)", fontsize=10)

    sns.scatterplot(
        data=df, x="n_ratio", y="crop_yield_mt_ha", hue="crop_type",
        palette="tab10", s=60, alpha=0.8, ax=axes[1]
    )
    sns.regplot(
        data=df, x="n_ratio", y="crop_yield_mt_ha",
        scatter=False, color="darkblue", ax=axes[1],
        line_kws={"linewidth": 2, "linestyle": "--"}
    )
    axes[1].set_title("Nitrogen Fraction (N / Total NPK) vs. Yield", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Nitrogen Proportion", fontsize=10)
    axes[1].set_ylabel("Yield (Metric Tons / Hectare)", fontsize=10)

    fig1_path = fig_d / "02_npk_stoichiometry_vs_yield.png"
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig1_path}")

    # -----------------------------------------------------------------
    # Figure 2: Hydrothermal Stress Surface
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df["mean_vpd_kpa"],
        df["hydro_thermal_stress_index"],
        c=df["crop_yield_mt_ha"],
        cmap="viridis",
        s=70,
        alpha=0.85,
        edgecolors="w"
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Yield (Metric Tons / Hectare)", fontsize=10)

    ax.set_title("Hydrothermal Stress Index vs. Vapor Pressure Deficit (VPD)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean VPD (kPa) [Atmospheric Aridity]", fontsize=10)
    ax.set_ylabel("Hydrothermal Stress Index [Moisture / Thermal Stress]", fontsize=10)

    fig2_path = fig_d / "02_hydrothermal_stress_surface.png"
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig2_path}")

    # -----------------------------------------------------------------
    # Figure 3: Feature Correlation Ranking
    # -----------------------------------------------------------------
    numeric_df = df.select_dtypes(include=[np.number])
    corrs = numeric_df.corr()["crop_yield_mt_ha"].drop(["crop_yield_mt_ha", "production_mt"], errors="ignore")
    top_corrs = corrs.abs().sort_values(ascending=False).head(15)
    top_corrs_signed = corrs.loc[top_corrs.index]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#2ecc71" if val > 0 else "#e74c3c" for val in top_corrs_signed]
    bars = ax.barh(top_corrs_signed.index[::-1], top_corrs_signed.values[::-1], color=colors[::-1], alpha=0.85)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Top 15 Engineered Features Correlated with Crop Yield", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient (r)", fontsize=10)
    ax.set_xlim(-0.9, 0.9)

    for bar, val in zip(bars, top_corrs_signed.values[::-1]):
        x_pos = val + (0.02 if val >= 0 else -0.06)
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.2f}", va="center", fontsize=9, fontweight="bold")

    fig3_path = fig_d / "02_feature_correlation_ranking.png"
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig3_path}")

    # -----------------------------------------------------------------
    # Markdown Summary Report
    # -----------------------------------------------------------------
    rep_lines = [
        "# 📈 Day 8: Agronomic Feature Analysis & EDA Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Scope**: Day 8 Milestone — Feature Correlation Screening & Agronomic Response Curves  ",
        f"**Records Analyzed**: {len(df)} across 15 districts and 5 crops  ",
        "",
        "---",
        "",
        "## 🔍 Key Findings & Feature Correlation Insights",
        "",
        "| Rank | Feature | Pearson $r$ | Agronomic Meaning & Model Relevance |",
        "| :---: | :--- | :---: | :--- |"
    ]

    for idx, (feat, val) in enumerate(top_corrs_signed.items(), 1):
        meaning = "Strong positive driver for vegetative canopy biomass" if val > 0.4 else "Critical thermal/moisture deficit penalty" if val < -0.4 else "Nutrient stoichiometry balance indicator"
        rep_lines.append(f"| {idx} | `{feat}` | **{val:+.3f}** | {meaning} |")

    rep_lines.extend([
        "",
        "---",
        "",
        "## 📊 Generated Visual Artifacts",
        f"1. `reports/figures/02_npk_stoichiometry_vs_yield.png`: Stoichiometric response curves.",
        f"2. `reports/figures/02_hydrothermal_stress_surface.png`: Joint VPD and moisture stress landscape.",
        f"3. `reports/figures/02_feature_correlation_ranking.png`: Top 15 ranked feature attributions.",
        "",
        "---",
        "✅ **Ready for Day 9**: Features selected for Spatial Block Group-KFold Model Training."
    ])

    md_report_path = rep_d / "feature_analysis_eda_report.md"
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rep_lines))
    logger.info(f"Saved: {md_report_path}")

    return {
        "fig1": fig1_path,
        "fig2": fig2_path,
        "fig3": fig3_path,
        "report": md_report_path
    }


def build_feature_engineering_notebook(
    output_path: str = "notebooks/02_feature_engineering_eda.ipynb"
) -> Path:
    """
    Constructs comprehensive Jupyter notebook for Day 8 EDA.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🌾 Week 2 / Day 8: Agronomic Feature Engineering & Correlation Analysis\n",
                    "\n",
                    "**Project**: Agriculture & Rural Businesses: Crop Yield & Soil Analysis Forecasting System  \n",
                    "**Deliverable**: Day 8 Milestone — Advanced Feature Space & Non-Linear Response Exploration  \n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 📌 Objectives\n",
                    "1. Explore stoichiometric nutrient ratios ($N:P$, $N:K$, $P:K$).\n",
                    "2. Investigate hydrothermal aridity indices and thermal stress penalties.\n",
                    "3. Quantify feature-target linear and non-linear correlations with end-of-season crop yield.\n",
                    "4. Validate feature store integrity for Spatial Block Cross-Validation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "sns.set_theme(style='whitegrid')\n",
                    "df = pd.read_csv('../data/processed/train_features.csv')\n",
                    "print(f'Feature Store Loaded: {df.shape[0]} rows x {df.shape[1]} columns')\n",
                    "df[['crop_type', 'soil_type', 'crop_yield_mt_ha', 'n_to_p_ratio', 'hydro_thermal_stress_index']].head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Stoichiometric Response: N:P Ratio vs. Yield"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, ax = plt.subplots(figsize=(10, 5))\n",
                    "sns.scatterplot(data=df, x='n_to_p_ratio', y='crop_yield_mt_ha', hue='crop_type', s=70, ax=ax)\n",
                    "sns.regplot(data=df, x='n_to_p_ratio', y='crop_yield_mt_ha', scatter=False, color='red', ax=ax)\n",
                    "ax.set_title('Crop Yield as a Function of Nitrogen-to-Phosphorus Stoichiometry', fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Hydrothermal Stress Landscape: Moisture vs. Evaporative Demand"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "fig, ax = plt.subplots(figsize=(10, 5))\n",
                    "sc = ax.scatter(df['mean_vpd_kpa'], df['hydro_thermal_stress_index'], c=df['crop_yield_mt_ha'], cmap='viridis', s=70)\n",
                    "plt.colorbar(sc, label='Crop Yield (MT/Ha)')\n",
                    "ax.set_xlabel('Vapor Pressure Deficit (kPa)')\n",
                    "ax.set_ylabel('Hydrothermal Stress Index')\n",
                    "ax.set_title('Yield Distribution across Climate Stress Space', fontweight='bold')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Pearson Correlation Ranking across All Engineered Features"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "numeric_df = df.select_dtypes(include=[np.number])\n",
                    "corrs = numeric_df.corr()['crop_yield_mt_ha'].drop(['crop_yield_mt_ha', 'production_mt'], errors='ignore')\n",
                    "top_corrs = corrs.abs().sort_values(ascending=False).head(15)\n",
                    "print('Top 15 Most Correlated Features with Crop Yield:')\n",
                    "print(corrs.loc[top_corrs.index])"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    logger.info(f"Built notebook: {out_p}")
    return out_p


def main():
    generate_feature_eda_artifacts()
    build_feature_engineering_notebook()


if __name__ == "__main__":
    main()
