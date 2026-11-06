# 🤖 Day 9: Spatial Block Cross-Validation & Regression Model Benchmark Report

**Generated**: 2026-11-06 09:55:09  
**Scope**: Day 9 Milestone — Supervised Ensemble Regression & Leak-Free Spatial Validation  
**Champion Production Model**: **Random_Forest**  
**Artifact Location**: `models\best_yield_regressor.joblib`  

---

## 📊 Spatial Block Group-KFold (5 Folds by District) Benchmark

| Model Algorithm | Spatial $R^2$ | Spatial RMSE (MT/Ha) | Spatial MAE (MT/Ha) | Spatial MAPE (%) | Benchmark Target Met? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge** | **0.6771** | 0.7535 | 0.6026 | 17.88% | ⚠️ Acceptable |
| **Random_Forest** ⭐ **Champion** | **0.8035** | 0.5879 | 0.4563 | 12.98% | ⚠️ Acceptable |
| **Gradient_Boosting** | **0.7639** | 0.6455 | 0.5011 | 14.00% | ⚠️ Acceptable |
| **Hist_Gradient_Boosting** | **0.7717** | 0.6332 | 0.5087 | 14.65% | ⚠️ Acceptable |

---

## ⏳ Temporal Leave-One-Year-Out (LOYO) Generalizability

| Test Year | LOYO $R^2$ | LOYO RMSE (MT/Ha) | LOYO MAPE (%) | Climate Context |
| :---: | :---: | :---: | :---: | :--- |
| **2019** | 0.7967 | 0.6119 | 12.12% | Normal Agronomic Trajectory |
| **2020** | 0.7302 | 0.7000 | 14.57% | Normal Agronomic Trajectory |
| **2021** | 0.3112 | 0.9071 | 23.73% | 2021 Severe Heatwave & Drought Anomaly |
| **2022** | 0.8693 | 0.4807 | 10.40% | Normal Agronomic Trajectory |
| **2023** | 0.8734 | 0.5029 | 13.38% | Normal Agronomic Trajectory |

**Average LOYO Performance**: $R^2 = 0.7162$ | $\text{RMSE} = 0.6405$ | $\text{MAPE} = 14.84\%$

---
✅ **Ready for Day 10**: Production model checkpoint ready for SHAP factor attribution and the Prescriptive Farmer Advisory Engine.