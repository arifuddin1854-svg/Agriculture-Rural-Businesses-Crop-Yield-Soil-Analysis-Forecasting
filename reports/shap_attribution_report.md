# 🔍 Day 10: SHAP Feature Attribution & Factor Disentanglement Report

**Generated**: 2026-10-21 15:18:20  
**Scope**: Day 10 Milestone — TreeExplainer Global Importance & Controllable Factor Isolation  
**Evaluated Model**: `Random_Forest`  

---

## 🌾 Top 10 Drivers of Agricultural Yield (TreeSHAP)

| Rank | Feature | Mean |SHAP| (MT/Ha Impact) | Factor Domain | Agronomic Interpretation |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `crop_type_Maize` | **0.5708** | Agronomic Baseline / Spatial | Exogenous macro-climatic atmospheric driver |
| 2 | `cumulative_solar_radiation_mj` | **0.2794** | Exogenous (Meteorological) | Exogenous macro-climatic atmospheric driver |
| 3 | `crop_type_Soybean` | **0.2168** | Agronomic Baseline / Spatial | Exogenous macro-climatic atmospheric driver |
| 4 | `mean_relative_humidity` | **0.1151** | Exogenous (Meteorological) | Exogenous macro-climatic atmospheric driver |
| 5 | `crop_type_Cotton` | **0.0874** | Agronomic Baseline / Spatial | Exogenous macro-climatic atmospheric driver |
| 6 | `growing_season_days` | **0.0736** | Agronomic Baseline / Spatial | Exogenous macro-climatic atmospheric driver |
| 7 | `crop_type_Rice` | **0.0484** | Agronomic Baseline / Spatial | Exogenous macro-climatic atmospheric driver |
| 8 | `mean_diurnal_range` | **0.0350** | Exogenous (Meteorological) | Exogenous macro-climatic atmospheric driver |
| 9 | `mean_temp_min` | **0.0324** | Exogenous (Meteorological) | Exogenous macro-climatic atmospheric driver |
| 10 | `mean_temp_max` | **0.0320** | Exogenous (Meteorological) | Exogenous macro-climatic atmospheric driver |

---

## ⚖️ Factor Domain Disentanglement Summary

- **Agronomic Baseline / Spatial**: 1.0568 cumulative SHAP impact
- **Controllable (Soil / Nutrient)**: 0.0890 cumulative SHAP impact
- **Exogenous (Meteorological)**: 0.6538 cumulative SHAP impact

---
✅ **Advisory Ready**: Top controllable factors mapped to prescriptive recommendations in `src/advisor.py`.