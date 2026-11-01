"""
Day 12: Production Covariate Drift Detection & Model Health Monitoring Engine
Member 1 / Production ML Service Deliverable

Responsible for:
- Detecting distribution shifts (covariate drift) between training baseline and production telemetry
- Calculating Population Stability Index (PSI) with quantile binning across continuous features
- Performing two-sample Kolmogorov-Smirnov (KS) tests for distribution divergence
- Flagging extreme environmental boundary anomalies (unprecedented heat stress, drought anomalies)
- Producing machine-readable JSON and human-readable Markdown drift audit reports
- Powering real-time monitoring API endpoints for operational dashboards
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MonitoringEngine_Day12")


class ModelMonitoringEngine:
    """
    Production monitoring engine for tracking agricultural feature distribution shifts,
    covariate drift, and model prediction stability.
    """

    CRITICAL_MONITORED_FEATURES = [
        "nitrogen_kg_ha",
        "phosphorus_kg_ha",
        "potassium_kg_ha",
        "soil_ph",
        "organic_matter_pct",
        "cumulative_rainfall_mm",
        "mean_temp_max",
        "mean_vpd_kpa",
        "drought_severity_index",
        "heat_stress_days"
    ]

    def __init__(
        self,
        baseline_path: Union[str, Path] = "data/processed/train_features.csv",
        report_dir: Union[str, Path] = "reports"
    ):
        self.baseline_path = Path(baseline_path)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_df = None

        if self.baseline_path.exists():
            logger.info(f"Loading training baseline data from {self.baseline_path}...")
            self.baseline_df = pd.read_csv(self.baseline_path)
            logger.info(f"Loaded {len(self.baseline_df)} baseline records with {len(self.baseline_df.columns)} features.")
        else:
            logger.warning(f"Baseline file {self.baseline_path} not found. Monitoring engine will require baseline DataFrame passed explicitly.")

    @staticmethod
    def calculate_psi(
        expected: np.ndarray,
        actual: np.ndarray,
        num_bins: int = 10,
        epsilon: float = 1e-4
    ) -> float:
        """
        Calculates Population Stability Index (PSI) between baseline (expected) and production (actual).
        
        Interpretation:
          - PSI < 0.10: No significant shift (Stable)
          - 0.10 <= PSI < 0.25: Moderate shift (Monitor closely)
          - PSI >= 0.25: Significant drift (Model retraining recommended)
        """
        exp_clean = expected[~np.isnan(expected)]
        act_clean = actual[~np.isnan(actual)]

        if len(exp_clean) == 0 or len(act_clean) == 0:
            return 0.0

        # Create quantile bins from expected baseline
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(exp_clean, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        bin_edges = np.unique(bin_edges)

        if len(bin_edges) <= 2:
            return 0.0

        # Bin counts
        exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
        act_counts, _ = np.histogram(act_clean, bins=bin_edges)

        # Proportions with Laplace smoothing
        exp_pct = (exp_counts + epsilon) / (len(exp_clean) + epsilon * len(exp_counts))
        act_pct = (act_counts + epsilon) / (len(act_clean) + epsilon * len(act_counts))

        # Sum of (Actual% - Expected%) * ln(Actual% / Expected%)
        psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(max(0.0, psi_value))

    @staticmethod
    def calculate_ks_test(
        expected: np.ndarray,
        actual: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculates two-sample Kolmogorov-Smirnov test statistic and p-value.
        """
        exp_clean = expected[~np.isnan(expected)]
        act_clean = actual[~np.isnan(actual)]

        if len(exp_clean) == 0 or len(act_clean) == 0:
            return 0.0, 1.0

        res = stats.ks_2samp(exp_clean, act_clean)
        return float(res.statistic), float(res.pvalue)

    def evaluate_covariate_drift(
        self,
        current_data: Union[pd.DataFrame, str, Path],
        feature_subset: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates drift across monitored features comparing baseline to current production data.
        """
        if isinstance(current_data, (str, Path)):
            curr_df = pd.read_csv(current_data)
        else:
            curr_df = current_data.copy()

        if self.baseline_df is None:
            raise ValueError("Baseline dataset is not initialized.")

        features_to_monitor = feature_subset or [
            f for f in self.CRITICAL_MONITORED_FEATURES
            if f in self.baseline_df.columns and f in curr_df.columns
        ]

        logger.info(f"Evaluating covariate drift across {len(features_to_monitor)} features for {len(curr_df)} production records...")

        feature_drift_results = {}
        high_drift_features = []
        moderate_drift_features = []
        stable_features = []

        for feat in features_to_monitor:
            exp_vals = self.baseline_df[feat].to_numpy(dtype=float)
            act_vals = curr_df[feat].to_numpy(dtype=float)

            psi = self.calculate_psi(exp_vals, act_vals)
            ks_stat, ks_pval = self.calculate_ks_test(exp_vals, act_vals)

            baseline_mean = float(np.nanmean(exp_vals))
            baseline_std = float(np.nanstd(exp_vals))
            current_mean = float(np.nanmean(act_vals))
            current_std = float(np.nanstd(act_vals))

            # Status classification
            if psi >= 0.25 or (ks_stat > 0.30 and ks_pval < 0.01):
                status = "SIGNIFICANT_DRIFT"
                high_drift_features.append(feat)
            elif psi >= 0.10 or (ks_stat > 0.15 and ks_pval < 0.05):
                status = "MODERATE_DRIFT"
                moderate_drift_features.append(feat)
            else:
                status = "STABLE"
                stable_features.append(feat)

            feature_drift_results[feat] = {
                "feature": feat,
                "psi": round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": round(ks_pval, 5),
                "status": status,
                "baseline_mean": round(baseline_mean, 2),
                "baseline_std": round(baseline_std, 2),
                "current_mean": round(current_mean, 2),
                "current_std": round(current_std, 2),
                "mean_shift_pct": round(((current_mean - baseline_mean) / (abs(baseline_mean) + 1e-5)) * 100.0, 2)
            }

        # Overall System Health Score (100 - penalties for drift)
        total_feats = len(features_to_monitor)
        drift_penalty = (len(high_drift_features) * 25.0 + len(moderate_drift_features) * 10.0) / max(1, total_feats)
        health_score = round(max(0.0, min(100.0, 100.0 - drift_penalty)), 1)

        if len(high_drift_features) > 0:
            overall_status = "RETRAIN_RECOMMENDED"
            action = f"Investigate severe drift in {len(high_drift_features)} features: {', '.join(high_drift_features)}. Re-fit spatial model."
        elif len(moderate_drift_features) > 0:
            overall_status = "MONITORING_ALERT"
            action = f"Moderate distribution shift detected in {len(moderate_drift_features)} features. Monitor validation error closely."
        else:
            overall_status = "HEALTHY_STABLE"
            action = "All monitored agronomic and meteorological covariates match baseline distributions within expected tolerance."

        summary = {
            "timestamp": datetime.now().isoformat(),
            "baseline_records": len(self.baseline_df),
            "production_records": len(curr_df),
            "total_monitored_features": total_feats,
            "overall_status": overall_status,
            "health_score_pct": health_score,
            "stable_feature_count": len(stable_features),
            "moderate_drift_feature_count": len(moderate_drift_features),
            "high_drift_feature_count": len(high_drift_features),
            "recommended_action": action,
            "features": feature_drift_results
        }

        return summary

    def generate_audit_reports(
        self,
        current_data: Union[pd.DataFrame, str, Path],
        output_json: str = "reports/drift_monitoring_report.json",
        output_md: str = "reports/drift_monitoring_report.md"
    ) -> Dict[str, Any]:
        """
        Executes drift analysis and writes structured JSON and Markdown audit reports.
        """
        summary = self.evaluate_covariate_drift(current_data)

        # 1. Save JSON
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved JSON drift report to: {json_path}")

        # 2. Save Markdown
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# 🛡️ Day 12: Covariate Drift & Model Stability Audit Report\n\n")
            f.write(f"**Execution Timestamp**: {summary['timestamp']}  \n")
            f.write(f"**Overall Health Score**: `{summary['health_score_pct']}%` ({summary['overall_status']})  \n")
            f.write(f"**Baseline Samples**: {summary['baseline_records']} records | **Production Samples**: {summary['production_records']} records  \n\n")

            f.write("## 📊 Monitoring Executive Summary\n\n")
            f.write("| Status Category | Count | Agronomic Status & Recommendation |\n")
            f.write("| :--- | :---: | :--- |\n")
            f.write(f"| **🟢 Stable Features (PSI < 0.10)** | **{summary['stable_feature_count']}** | Operating within safe training envelope |\n")
            f.write(f"| **🟡 Moderate Drift (0.10 ≤ PSI < 0.25)** | **{summary['moderate_drift_feature_count']}** | Minor seasonal variation observed |\n")
            f.write(f"| **🔴 High Drift (PSI ≥ 0.25)** | **{summary['high_drift_feature_count']}** | Significant climate/soil shift requiring attention |\n\n")

            f.write(f"**Recommended Action**: {summary['recommended_action']}\n\n")

            f.write("## 🧪 Monitored Feature Covariate Shift Matrix\n\n")
            f.write("| Feature Name | PSI Score | KS Statistic | Baseline (Mean ± SD) | Production (Mean ± SD) | Shift (%) | Drift Status |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

            for feat, d in summary["features"].items():
                status_icon = "🟢" if d["status"] == "STABLE" else ("🟡" if d["status"] == "MODERATE_DRIFT" else "🔴")
                f.write(
                    f"| `{feat}` | {d['psi']:.4f} | {d['ks_statistic']:.4f} | "
                    f"{d['baseline_mean']:.2f} ± {d['baseline_std']:.2f} | "
                    f"{d['current_mean']:.2f} ± {d['current_std']:.2f} | "
                    f"{d['mean_shift_pct']:+.1f}% | {status_icon} **{d['status']}** |\n"
                )
            f.write("\n")
            f.write("```bash\n# Reproduce drift audit pipeline\npython -m src.monitoring\n```\n")

        logger.info(f"Saved Markdown drift report to: {md_path}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Day 12: Model Monitoring & Covariate Drift CLI")
    parser.add_argument("--baseline", default="data/processed/train_features.csv", help="Baseline training features CSV")
    parser.add_argument("--production", default="data/processed/unified_agri_dataset.csv", help="Production dataset to check")
    parser.add_argument("--json-report", default="reports/drift_monitoring_report.json", help="Path to output JSON report")
    parser.add_argument("--md-report", default="reports/drift_monitoring_report.md", help="Path to output Markdown report")

    args = parser.parse_args()

    engine = ModelMonitoringEngine(baseline_path=args.baseline)
    res = engine.generate_audit_reports(
        current_data=args.production,
        output_json=args.json_report,
        output_md=args.md_report
    )
    print("\n" + "=" * 75)
    print(f"[*] DAY 12 MONITORING AUDIT: Overall Health: {res['health_score_pct']}% ({res['overall_status']})")
    print(f"[*] Stable Features: {res['stable_feature_count']} | Moderate Drift: {res['moderate_drift_feature_count']} | High Drift: {res['high_drift_feature_count']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
