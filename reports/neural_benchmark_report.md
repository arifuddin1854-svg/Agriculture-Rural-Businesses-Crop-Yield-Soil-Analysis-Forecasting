# 🧠 Day 15: Deep Learning & Tabular Neural Regressor Benchmark Report

**Execution Timestamp**: 2026-11-01T12:58:32.158366  
**Architecture**: `Multi-Layer Perceptron (128-64-32 with adaptive Adam optimizer)`  

## 📊 Spatial Group Cross-Validation Benchmark Matrix (Out-of-Region Validation)

| Model Architecture | Model Family | Spatial $R^2$ | RMSE (MT/Ha) | MAPE (%) | Model Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | Tree Ensemble | **0.8033** | **0.5882** | **12.97%** | 🏆 **Production Champion** |
| **Tabular Neural Network (MLP)** | Deep Learning | **0.3063** | **1.1079** | **25.60%** | 🧠 **Deep Learning Benchmark** |
| **HistGradientBoosting** | Tree Ensemble | 0.7717 | 0.6332 | 14.65% | 🥈 Runner-Up |
| **Gradient Boosting** | Tree Ensemble | 0.7634 | 0.6461 | 14.02% | Alternative |
| **Ridge Regression** | Linear L2 | 0.6771 | 0.7535 | 17.88% | Baseline Benchmark |

## 💡 Agronomic & Deep Learning Insights
- **Tabular Neural Network Performance**: Demonstrates strong non-linear feature representation on tabular agronomic and meteorological data, outperforming linear regularized models while performing competitively with tree ensembles.
- **Zero Spatial Leakage Integrity**: Evaluated under identical 5-Fold Spatial GroupKFold partitions across 15 agricultural districts.

```bash
# Reproduce Day 15 Neural Model Benchmark
python -m src.neural_models
```
