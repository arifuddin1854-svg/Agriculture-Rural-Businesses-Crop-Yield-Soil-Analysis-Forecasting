# 🛡️ Day 12: Covariate Drift & Model Stability Audit Report

**Execution Timestamp**: 2026-11-01T12:38:19.836648  
**Overall Health Score**: `100.0%` (HEALTHY_STABLE)  
**Baseline Samples**: 225 records | **Production Samples**: 225 records  

## 📊 Monitoring Executive Summary

| Status Category | Count | Agronomic Status & Recommendation |
| :--- | :---: | :--- |
| **🟢 Stable Features (PSI < 0.10)** | **9** | Operating within safe training envelope |
| **🟡 Moderate Drift (0.10 ≤ PSI < 0.25)** | **0** | Minor seasonal variation observed |
| **🔴 High Drift (PSI ≥ 0.25)** | **0** | Significant climate/soil shift requiring attention |

**Recommended Action**: All monitored agronomic and meteorological covariates match baseline distributions within expected tolerance.

## 🧪 Monitored Feature Covariate Shift Matrix

| Feature Name | PSI Score | KS Statistic | Baseline (Mean ± SD) | Production (Mean ± SD) | Shift (%) | Drift Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `nitrogen_kg_ha` | 0.0000 | 0.0000 | 112.70 ± 30.26 | 112.70 ± 30.26 | +0.0% | 🟢 **STABLE** |
| `phosphorus_kg_ha` | 0.0000 | 0.0000 | 44.13 ± 15.04 | 44.13 ± 15.04 | +0.0% | 🟢 **STABLE** |
| `potassium_kg_ha` | 0.0000 | 0.0000 | 160.18 ± 47.79 | 160.18 ± 47.79 | +0.0% | 🟢 **STABLE** |
| `soil_ph` | 0.0000 | 0.0000 | 6.32 ± 0.63 | 6.32 ± 0.63 | +0.0% | 🟢 **STABLE** |
| `organic_matter_pct` | 0.0000 | 0.0000 | 1.19 ± 0.28 | 1.19 ± 0.28 | +0.0% | 🟢 **STABLE** |
| `cumulative_rainfall_mm` | 0.0000 | 0.0000 | 884.10 ± 404.02 | 884.10 ± 404.02 | +0.0% | 🟢 **STABLE** |
| `mean_temp_max` | 0.0000 | 0.0000 | 32.47 ± 3.99 | 32.47 ± 3.99 | +0.0% | 🟢 **STABLE** |
| `mean_vpd_kpa` | 0.0000 | 0.0000 | 1.32 ± 0.13 | 1.32 ± 0.13 | +0.0% | 🟢 **STABLE** |
| `heat_stress_days` | 0.0000 | 0.0000 | 72.23 ± 40.97 | 72.23 ± 40.97 | +0.0% | 🟢 **STABLE** |

```bash
# Reproduce drift audit pipeline
python -m src.monitoring
```
