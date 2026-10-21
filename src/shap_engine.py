"""
Day 10: SHAP Factor Attribution & Exogenous vs. Controllable Disentanglement Engine
Member 4 Scope / Sprint Extension (Days 7–10)

Responsible for:
- Computing Shapley attribution values via TreeExplainer on the production model
- Decoupling farmer-controllable factors (N, P, K, pH, SOM) from exogenous weather drivers
- Generating SHAP summary beeswarm plots and factor category attribution breakdowns
- Exporting machine-readable attribution scores to reports/shap_attribution_report.json
- Building Jupyter Notebook: notebooks/04_shap_explainability.ipynb
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SHAPEngine_Day10")


CONTROLLABLE_KEYWORDS = [
    "nitrogen", "phosphorus", "potassium", "ph", "som", "organic_matter",
    "npk", "n_ratio", "p_ratio", "k_ratio", "n_to_p", "acidity", "alkalinity"
]

EXOGENOUS_KEYWORDS = [
    "rain", "precip", "temp", "heat", "cold", "gdd", "vpd", "solar",
    "dry_days", "aridity", "humidity", "diurnal", "hydro_thermal"
]


def classify_feature_domain(feature_name: str) -> str:
    """
    Classifies an input feature into Controllable (Farmer-Managed) or Exogenous (Weather).
    """
    f_lower = feature_name.lower()
    if any(k in f_lower for k in CONTROLLABLE_KEYWORDS):
        return "Controllable (Soil / Nutrient)"
    if any(k in f_lower for k in EXOGENOUS_KEYWORDS):
        return "Exogenous (Meteorological)"
    return "Agronomic Baseline / Spatial"


def compute_shap_explanations(
    model: Any,
    X: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Calculates SHAP values for the feature matrix using TreeExplainer.
    """
    logger.info(f"Initializing SHAP TreeExplainer for feature space: {X.shape}...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap,
        "domain": [classify_feature_domain(c) for c in X.columns]
    }).sort_values(by="mean_abs_shap", ascending=False)

    return shap_values, importance_df


def render_shap_figures(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    importance_df: pd.DataFrame,
    figures_dir: Path
) -> Tuple[Path, Path]:
    """
    Renders SHAP beeswarm summary and controllable vs. exogenous factor comparison.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. SHAP Beeswarm Summary Plot
    fig1_path = figures_dir / "04_shap_summary_beeswarm.png"
    plt.figure(figsize=(11, 7))
    shap.summary_plot(shap_values, X, max_display=15, show=False)
    plt.title("SHAP Global Feature Attribution (Top 15 Drivers of Crop Yield)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig1_path}")

    # 2. Controllable vs. Exogenous Relative Importance
    fig2_path = figures_dir / "04_shap_controllable_vs_exogenous.png"
    domain_agg = importance_df.groupby("domain")["mean_abs_shap"].sum().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Pie Chart
    colors = ["#27ae60", "#2980b9", "#f39c12"]
    axes[0].pie(
        domain_agg["mean_abs_shap"],
        labels=domain_agg["domain"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=[0.05, 0.05, 0.05]
    )
    axes[0].set_title("Aggregate Attribution: Controllable vs. Exogenous", fontsize=12, fontweight="bold")

    # Bar chart of top controllable features
    top_controllable = importance_df[importance_df["domain"] == "Controllable (Soil / Nutrient)"].head(8)
    sns.barplot(
        data=top_controllable,
        x="mean_abs_shap",
        y="feature",
        palette="Greens_r",
        hue="feature",
        legend=False,
        ax=axes[1]
    )
    axes[1].set_title("Top Controllable Farm Factors (Targeted Interventions)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean |SHAP Value| (Yield Impact in MT/Ha)")
    axes[1].set_ylabel("Soil & Nutrient Feature")

    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved: {fig2_path}")

    return fig1_path, fig2_path


def build_shap_notebook(output_path: Path) -> None:
    """
    Constructs Day 10 Jupyter notebook for SHAP attribution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🌾 Week 4 / Day 10: SHAP Feature Attribution & Prescriptive Agronomy\n",
                    "\n",
                    "**Project**: Agriculture & Rural Businesses: Crop Yield & Soil Analysis Forecasting System  \n",
                    "**Deliverable**: Day 10 Milestone — TreeExplainer Interpretability & Controllable Factor Disentanglement  \n",
                    "\n",
                    "---\n",
                    "\n",
                    "## 📌 Objectives\n",
                    "1. Compute global and local feature attribution using **SHAP TreeExplainer**.\n",
                    "2. Decouple **controllable management factors** ($N, P, K$, pH, SOM) from **exogenous weather**.\n",
                    "3. Validate actionable intervention thresholds for the Prescriptive Farmer Advisory Script."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import joblib\n",
                    "import pandas as pd\n",
                    "import shap\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "# Load model and feature store\n",
                    "checkpoint = joblib.load('../models/best_yield_regressor.joblib')\n",
                    "model = checkpoint['model']\n",
                    "features = checkpoint['feature_names']\n",
                    "\n",
                    "df = pd.read_csv('../data/processed/train_features.csv')\n",
                    "X = df[features]\n",
                    "print(f\"Loaded feature matrix: {X.shape}\")\n",
                    "\n",
                    "# TreeExplainer attribution\n",
                    "explainer = shap.TreeExplainer(model)\n",
                    "shap_values = explainer.shap_values(X)\n",
                    "print(\"SHAP attribution complete!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Global SHAP Summary Beeswarm Plot"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(10, 6))\n",
                    "shap.summary_plot(shap_values, X, max_display=12)\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)


def run_shap_pipeline(
    checkpoint_path: str = "models/best_yield_regressor.joblib",
    features_path: str = "data/processed/train_features.csv",
    figures_dir: str = "reports/figures",
    report_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Executes the full SHAP explainability pipeline.
    """
    logger.info("=" * 70)
    logger.info("Executing Day 10: SHAP Feature Attribution Pipeline")
    logger.info("=" * 70)

    chk_p = Path(checkpoint_path)
    feat_p = Path(features_path)
    fig_d = Path(figures_dir)
    rep_d = Path(report_dir)

    fig_d.mkdir(parents=True, exist_ok=True)
    rep_d.mkdir(parents=True, exist_ok=True)

    checkpoint = joblib.load(chk_p)
    model = checkpoint["model"]
    feature_names = checkpoint["feature_names"]

    df = pd.read_csv(feat_p)
    X = df[feature_names].copy()

    # Compute SHAP
    shap_vals, imp_df = compute_shap_explanations(model, X)

    # Render Figures
    fig1, fig2 = render_shap_figures(shap_vals, X, imp_df, fig_d)

    # Build Notebook 04
    nb_path = Path("notebooks/04_shap_explainability.ipynb")
    build_shap_notebook(nb_path)

    # Machine-readable JSON Report
    top_10 = imp_df.head(10).to_dict(orient="records")
    report_payload = {
        "timestamp": datetime.now().isoformat(),
        "model_name": checkpoint["model_name"],
        "total_features": len(feature_names),
        "top_features_ranked": top_10,
        "domain_breakdown": imp_df.groupby("domain")["mean_abs_shap"].sum().to_dict()
    }

    json_report = rep_d / "shap_attribution_report.json"
    with open(json_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    # Executive Markdown Report
    md_lines = [
        "# 🔍 Day 10: SHAP Feature Attribution & Factor Disentanglement Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Scope**: Day 10 Milestone — TreeExplainer Global Importance & Controllable Factor Isolation  ",
        f"**Evaluated Model**: `{checkpoint['model_name']}`  ",
        "",
        "---",
        "",
        "## 🌾 Top 10 Drivers of Agricultural Yield (TreeSHAP)",
        "",
        "| Rank | Feature | Mean |SHAP| (MT/Ha Impact) | Factor Domain | Agronomic Interpretation |",
        "| :---: | :--- | :---: | :--- | :--- |"
    ]

    for rank, row in enumerate(top_10, 1):
        domain = row["domain"]
        val = row["mean_abs_shap"]
        feat = row["feature"]
        interp = "Farmer manageable via fertilizer scheduling" if "Controllable" in domain else "Exogenous macro-climatic atmospheric driver"
        md_lines.append(f"| {rank} | `{feat}` | **{val:.4f}** | {domain} | {interp} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## ⚖️ Factor Domain Disentanglement Summary",
        ""
    ])

    for dom, score in report_payload["domain_breakdown"].items():
        md_lines.append(f"- **{dom}**: {score:.4f} cumulative SHAP impact")

    md_lines.extend([
        "",
        "---",
        "✅ **Advisory Ready**: Top controllable factors mapped to prescriptive recommendations in `src/advisor.py`."
    ])

    md_report = rep_d / "shap_attribution_report.md"
    with open(md_report, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info("=" * 70)
    logger.info("Day 10 SHAP Attribution Pipeline Completed Successfully!")
    logger.info("=" * 70)

    return {
        "json_report": json_report,
        "md_report": md_report,
        "fig1": fig1,
        "fig2": fig2,
        "importance_df": imp_df
    }


def main():
    parser = argparse.ArgumentParser(description="SHAP Attribution Pipeline")
    parser.add_argument("--checkpoint", type=str, default="models/best_yield_regressor.joblib")
    parser.add_argument("--features", type=str, default="data/processed/train_features.csv")
    parser.add_argument("--figures-dir", type=str, default="reports/figures")
    parser.add_argument("--report-dir", type=str, default="reports")
    args = parser.parse_args()

    run_shap_pipeline(
        checkpoint_path=args.checkpoint,
        features_path=args.features,
        figures_dir=args.figures_dir,
        report_dir=args.report_dir
    )


if __name__ == "__main__":
    main()
