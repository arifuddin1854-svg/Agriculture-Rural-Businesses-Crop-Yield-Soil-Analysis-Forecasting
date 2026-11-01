"""
Day 16: Climate Change Scenario Simulation Engine (CMIP6: SSP2-4.5 & SSP5-8.5)
Member 1 / Production Climate Resilience Module

Responsible for:
- Simulating future climate projections based on IPCC CMIP6 Shared Socioeconomic Pathways:
  * SSP2-4.5 ("Middle of the Road"): Moderate emissions (+1.5°C mean temp, +35% heat days, erratic rain)
  * SSP5-8.5 ("Fossil-fueled Development"): High emissions (+2.8°C mean temp, +75% heat days, severe drought)
- Applying projected climate perturbations to regional farm baselines
- Simulating 2025–2040 crop yield trajectories using production model (models/best_yield_regressor.joblib)
- Quantifying crop-specific vulnerabilities and district climate resilience rankings
- Prescribing agronomic adaptation strategies:
  * Crop switching (e.g., transitioning heat-stressed wheat to drought-tolerant sorghum/pearl millet)
  * Sowing calendar shifts (14-21 day earlier planting to escape terminal heat)
  * Organic mulching and water-harvesting soil amendments
- Exporting projected scenario datasets and rendering analytical figures
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.inference import CropYieldInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ClimateScenarios_Day16")


# CMIP6 Climatological Perturbation Factors (Projections for 2030-2040 relative to 2019-2023 baseline)
CLIMATE_PATHWAYS = {
    "SSP2_45": {
        "name": "SSP2-4.5 (Middle of the Road)",
        "temp_shift_c": 1.5,
        "heat_days_multiplier": 1.35,
        "rainfall_multiplier": 0.94,
        "dry_spell_increase_days": 4.0,
        "vpd_increase_pct": 0.12,
        "color": "#e76f51"
    },
    "SSP5_85": {
        "name": "SSP5-8.5 (High Emissions)",
        "temp_shift_c": 2.8,
        "heat_days_multiplier": 1.75,
        "rainfall_multiplier": 0.86,
        "dry_spell_increase_days": 8.0,
        "vpd_increase_pct": 0.28,
        "color": "#d90429"
    }
}


class ClimateScenarioSimulator:
    """
    Simulates agricultural yield vulnerability and resilience under IPCC CMIP6 climate pathways.
    """

    def __init__(
        self,
        model_path: str = "models/best_yield_regressor.joblib",
        baseline_data_path: str = "data/processed/unified_agri_dataset.csv",
        output_dir: str = "data/processed",
        figures_dir: str = "reports/figures",
        report_dir: str = "reports"
    ):
        self.engine = CropYieldInferenceEngine(model_path=model_path)
        self.baseline_path = Path(baseline_data_path)
        self.output_dir = Path(output_dir)
        self.figures_dir = Path(figures_dir)
        self.report_dir = Path(report_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def load_baseline(self) -> pd.DataFrame:
        if not self.baseline_path.exists():
            raise FileNotFoundError(f"Baseline dataset not found at {self.baseline_path}")
        return pd.read_csv(self.baseline_path)

    def simulate_pathway(
        self,
        df_base: pd.DataFrame,
        scenario_key: str
    ) -> pd.DataFrame:
        """
        Applies climate scenario perturbations to baseline records and simulates yield impact.
        """
        cfg = CLIMATE_PATHWAYS[scenario_key]
        logger.info(f"Simulating climate scenario: {cfg['name']} (+{cfg['temp_shift_c']}°C, +{(cfg['heat_days_multiplier']-1)*100:.0f}% heat days)...")

        sim_records = []
        for idx, row in df_base.iterrows():
            rec = row.to_dict()

            # Apply meteorological perturbations
            base_tmax = float(rec.get("mean_temp_max", 32.5))
            base_rain = float(rec.get("cumulative_rainfall_mm", 850.0))
            base_heat = float(rec.get("heat_stress_days", 5.0))
            base_vpd = float(rec.get("mean_vpd_kpa", 1.25))
            base_dry = float(rec.get("consecutive_dry_days_max", 12.0))

            pert_tmax = base_tmax + cfg["temp_shift_c"]
            pert_rain = max(100.0, base_rain * cfg["rainfall_multiplier"])
            pert_heat = base_heat * cfg["heat_days_multiplier"]
            pert_vpd = base_vpd * (1.0 + cfg["vpd_increase_pct"])
            pert_dry = base_dry + cfg["dry_spell_increase_days"]

            sim_farm = rec.copy()
            sim_farm["mean_temp_max"] = pert_tmax
            sim_farm["cumulative_rainfall_mm"] = pert_rain
            sim_farm["heat_stress_days"] = pert_heat
            sim_farm["mean_vpd_kpa"] = pert_vpd
            sim_farm["consecutive_dry_days_max"] = pert_dry

            # Run inference under climate stress
            pred_res = self.engine.predict_single(sim_farm)
            sim_yield = pred_res["predicted_yield_mt_ha"]
            base_yield = float(rec.get("crop_yield_mt_ha", pred_res["predicted_yield_mt_ha"]))
            yield_change_pct = ((sim_yield - base_yield) / max(0.1, base_yield)) * 100.0

            sim_records.append({
                "scenario": scenario_key,
                "scenario_name": cfg["name"],
                "district_id": rec["district_id"],
                "crop_type": rec["crop_type"],
                "baseline_yield_mt_ha": round(base_yield, 3),
                "projected_yield_mt_ha": round(sim_yield, 3),
                "yield_delta_mt_ha": round(sim_yield - base_yield, 3),
                "yield_change_pct": round(yield_change_pct, 2),
                "projected_temp_max": round(pert_tmax, 2),
                "projected_rainfall_mm": round(pert_rain, 1),
                "projected_heat_stress_days": round(pert_heat, 1),
                "risk_level": pred_res["risk_assessment"]["level"]
            })

        return pd.DataFrame(sim_records)

    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Executes both SSP2-4.5 and SSP5-8.5 climate pathway simulations.
        """
        df_base = self.load_baseline()
        df_ssp2 = self.simulate_pathway(df_base, "SSP2_45")
        df_ssp5 = self.simulate_pathway(df_base, "SSP5_85")

        df_all = pd.concat([df_ssp2, df_ssp5], ignore_index=True)

        out_path = self.output_dir / "climate_scenario_projections.csv"
        df_all.to_csv(out_path, index=False)
        logger.info(f"Saved climate scenario projections to {out_path} ({len(df_all)} records)")
        return df_all

    def render_visual_artifacts(self, df_scenarios: pd.DataFrame):
        """
        Renders scenario yield comparisons and district vulnerability heatmaps.
        """
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # -------------------------------------------------------------
        # Figure 1: Projected Yield Change by Crop under Climate Pathways
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        crop_order = ["Wheat", "Rice", "Maize", "Soybean", "Cotton"]
        summary = df_scenarios.groupby(["crop_type", "scenario_name"])["yield_change_pct"].mean().reset_index()

        sns.barplot(
            data=summary,
            x="crop_type",
            y="yield_change_pct",
            hue="scenario_name",
            order=crop_order,
            palette={"SSP2-4.5 (Middle of the Road)": "#f39c12", "SSP5-8.5 (High Emissions)": "#c0392b"},
            ax=ax
        )

        ax.axhline(0, color="#2c3e50", linestyle="--", linewidth=1.2)
        ax.set_title("Projected Agricultural Yield Impact Under CMIP6 Climate Pathways (2025–2040)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Crop Species", fontsize=10, labelpad=8)
        ax.set_ylabel("Mean Projected Yield Change (%)", fontsize=10, labelpad=8)
        ax.legend(title="Climate Scenario Pathway", frameon=True, facecolor="#ffffff")
        plt.tight_layout()

        fig_path1 = self.figures_dir / "07_climate_scenario_yield_projections.png"
        plt.savefig(fig_path1, dpi=300)
        plt.close()
        logger.info(f"Saved figure: {fig_path1}")

        # -------------------------------------------------------------
        # Figure 2: District Vulnerability Matrix under SSP5-8.5
        # -------------------------------------------------------------
        df_high = df_scenarios[df_scenarios["scenario"] == "SSP5_85"]
        pivot_vuln = df_high.pivot_table(
            index="district_id",
            columns="crop_type",
            values="yield_change_pct",
            aggfunc="mean"
        ).fillna(0.0)

        fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
        sns.heatmap(
            pivot_vuln,
            cmap="RdYlGn",
            center=0.0,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Yield Change (%)"},
            ax=ax,
            linewidths=0.5
        )

        ax.set_title("Regional Vulnerability Matrix: District x Crop Yield Shifts (SSP5-8.5 High Emissions)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Crop Species", fontsize=10)
        ax.set_ylabel("Agricultural District ID", fontsize=10)
        plt.tight_layout()

        fig_path2 = self.figures_dir / "07_district_vulnerability_heatmap.png"
        plt.savefig(fig_path2, dpi=300)
        plt.close()
        logger.info(f"Saved figure: {fig_path2}")

    def generate_resilience_reports(
        self,
        df_scenarios: pd.DataFrame,
        elapsed_sec: float
    ) -> Dict[str, Any]:
        """
        Emits machine-readable JSON and Markdown resilience audit reports.
        """
        crop_impacts = {}
        for crop, grp in df_scenarios.groupby("crop_type"):
            ssp2_sub = grp[grp["scenario"] == "SSP2_45"]
            ssp5_sub = grp[grp["scenario"] == "SSP5_85"]
            crop_impacts[crop] = {
                "ssp2_mean_change_pct": round(float(ssp2_sub["yield_change_pct"].mean()), 2),
                "ssp5_mean_change_pct": round(float(ssp5_sub["yield_change_pct"].mean()), 2),
                "vulnerability_tier": "HIGH_VULNERABILITY" if ssp5_sub["yield_change_pct"].mean() < -8.0 else ("MODERATE_VULNERABILITY" if ssp5_sub["yield_change_pct"].mean() < -3.0 else "RESILIENT")
            }

        # Most vulnerable districts
        dist_impacts = df_scenarios[df_scenarios["scenario"] == "SSP5_85"].groupby("district_id")["yield_change_pct"].mean().sort_values()
        top_vulnerable = {str(k): round(float(v), 2) for k, v in dist_impacts.head(5).items()}

        audit_data = {
            "timestamp": datetime.now().isoformat(),
            "pipeline_stage": "Day 16 - Climate Change Scenario Simulation Engine",
            "scenarios_evaluated": ["SSP2-4.5", "SSP5-8.5"],
            "total_simulations": len(df_scenarios),
            "execution_seconds": round(elapsed_sec, 2),
            "crop_impacts": crop_impacts,
            "most_vulnerable_districts": top_vulnerable,
            "recommended_adaptation_policies": [
                "Early Sowing Calendar Shift: Advance sowing dates by 14-21 days for Wheat to avoid grain-filling heat shocks.",
                "Crop Diversification: Transition high-vulnerability districts to drought-hardy millets, sorghum, and pulses.",
                "Soil Water Retention Buffering: Incorporate 8 MT/ha organic compost and biochar to maintain soil moisture during prolonged dry spells."
            ]
        }

        # 1. JSON Report
        json_path = self.report_dir / "climate_resilience_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(audit_data, f, indent=2)

        # 2. Markdown Report
        md_path = self.report_dir / "climate_resilience_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 🌦️ Day 16: Climate Change Scenario Simulation & Agronomic Resilience Report\n\n")
            f.write(f"**Execution Timestamp**: {audit_data['timestamp']}  \n")
            f.write(f"**Pathways Evaluated**: IPCC CMIP6 Shared Socioeconomic Pathways (SSP2-4.5 & SSP5-8.5)  \n\n")

            f.write("## 📊 Crop Vulnerability Under Climate Pathways (2025–2040)\n\n")
            f.write("| Crop Species | SSP2-4.5 (+1.5°C) Impact | SSP5-8.5 (+2.8°C) Impact | Vulnerability Tier | Primary Climate Driver |\n")
            f.write("| :--- | :---: | :---: | :---: | :--- |\n")
            for crop, d in crop_impacts.items():
                tier_badge = "🔴 **HIGH**" if "HIGH" in d["vulnerability_tier"] else ("🟡 **MODERATE**" if "MODERATE" in d["vulnerability_tier"] else "🟢 **RESILIENT**")
                f.write(f"| **{crop}** | {d['ssp2_mean_change_pct']:+.1f}% | {d['ssp5_mean_change_pct']:+.1f}% | {tier_badge} | Terminal heat stress & water deficit |\n")
            f.write("\n")

            f.write("## ⚠️ Top 5 Climate-Vulnerable Agricultural Districts (SSP5-8.5)\n\n")
            f.write("| District ID | Mean Projected Yield Shift | Climatological Driver |\n")
            f.write("| :--- | :---: | :--- |\n")
            for dist, change in top_vulnerable.items():
                f.write(f"| **{dist}** | **{change:+.1f}%** | Extreme heat wave frequency and dry spell duration |\n")
            f.write("\n")

            f.write("## 🌾 Strategic Agronomic Adaptation Prescriptions\n")
            for p in audit_data["recommended_adaptation_policies"]:
                f.write(f"- {p}\n")
            f.write("\n")
            f.write("```bash\n# Reproduce Day 16 Climate Scenarios\npython -m src.climate_scenarios\n```\n")

        logger.info(f"Saved audit reports to {json_path} and {md_path}")
        return audit_data


def main():
    t0 = time.time()
    simulator = ClimateScenarioSimulator()
    df_scenarios = simulator.run_all_scenarios()
    simulator.render_visual_artifacts(df_scenarios)
    elapsed = time.time() - t0
    res = simulator.generate_resilience_reports(df_scenarios, elapsed)

    print("\n" + "=" * 75)
    print(f"[*] DAY 16 CLIMATE SIMULATOR: Simulated {res['total_simulations']} farm scenarios in {elapsed:.2f}s")
    for crop, d in res["crop_impacts"].items():
        print(f"    -> {crop}: SSP2-4.5 ({d['ssp2_mean_change_pct']:+.1f}%), SSP5-8.5 ({d['ssp5_mean_change_pct']:+.1f}%) [{d['vulnerability_tier']}]")
    print("=" * 75)


if __name__ == "__main__":
    main()
