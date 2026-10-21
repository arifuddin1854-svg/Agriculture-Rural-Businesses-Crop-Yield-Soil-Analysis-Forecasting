# 🤖 Day 9: Spatial Block Cross-Validation & Regression Model Benchmark Report

**Generated**: 2026-10-21 15:16:03  
**Scope**: Day 9 Milestone — Supervised Ensemble Regression & Leak-Free Spatial Validation  
**Champion Production Model**: **Random_Forest**  
**Artifact Location**: `models\best_yield_regressor.joblib`  

---

## 📊 Spatial Block Group-KFold (5 Folds by District) Benchmark

| Model Algorithm | Spatial $R^2$ | Spatial RMSE (MT/Ha) | Spatial MAE (MT/Ha) | Spatial MAPE (%) | Benchmark Target Met? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge** | **0.6771** | 0.7535 | 0.6026 | 17.88% | ⚠️ Acceptable |
| **Random_Forest** ⭐ **Champion** | **0.8033** | 0.5882 | 0.4558 | 12.97% | ⚠️ Acceptable |
| **Gradient_Boosting** | **0.7634** | 0.6461 | 0.5017 | 14.02% | ⚠️ Acceptable |
| **Hist_Gradient_Boosting** | **0.7717** | 0.6332 | 0.5087 | 14.65% | ⚠️ Acceptable |

---

## ⏳ Temporal Leave-One-Year-Out (LOYO) Generalizability

| Test Year | LOYO $R^2$ | LOYO RMSE (MT/Ha) | LOYO MAPE (%) | Climate Context |
| :---: | :---: | :---: | :---: | :--- |
| **2019** | 0.7983 | 0.6095 | 12.07% | Normal Agronomic Trajectory |
| **2020** | 0.7292 | 0.7012 | 14.60% | Normal Agronomic Trajectory |
| **2021** | 0.3129 | 0.9060 | 23.66% | 2021 Severe Heatwave & Drought Anomaly |
| **2022** | 0.8696 | 0.4800 | 10.43% | Normal Agronomic Trajectory |
| **2023** | 0.8724 | 0.5047 | 13.45% | Normal Agronomic Trajectory |

**Average LOYO Performance**: $R^2 = 0.7165$ | $\text{RMSE} = 0.6403$ | $\text{MAPE} = 14.84\%$

---
✅ **Ready for Day 10**: Production model checkpoint ready for SHAP factor attribution and the Prescriptive Farmer Advisory Engine.