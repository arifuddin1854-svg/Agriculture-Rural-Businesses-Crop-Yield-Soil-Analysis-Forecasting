# Crop Yield Prediction Model Card

## 1. Model Overview

**Model Name:** Agriculture Crop Yield Prediction Model  
**Model Type:** Supervised Machine Learning – Regression  
**Algorithm:** XGBoost Regressor  
**Framework:** Python, Scikit-learn, XGBoost  
**Model Artifact:** `models/crop_yield_xgb.joblib`  
**ONNX Artifact:** `models/crop_yield_model.onnx`

### Purpose

The model predicts agricultural crop yield using crop-related, soil, weather, and seasonal information.

It is designed to support agricultural data analysis and provide insights into factors that may influence crop productivity.

## 2. Dataset Information

**Dataset:** Unified Agriculture Dataset  
**Location:** `data/processed/unified_agri_dataset.csv`

The dataset combines information from three sources:

- Crop yield data
- State-level soil data
- State-level weather data

### Input Features

| Category | Features |
|---|---|
| Crop information | Crop, season, state, year |
| Agricultural inputs | Fertilizer, pesticide |
| Soil properties | Nitrogen (N), Phosphorus (P), Potassium (K), pH |
| Weather information | Average temperature, total rainfall, average humidity |

**Target Variable:** `yield`

The `area` and `production` columns are excluded from model inputs to avoid target leakage.

## 3. Model Architecture

The model uses an XGBoost Regressor within a Scikit-learn Pipeline.

The pipeline consists of:

1. **Preprocessing:** A ColumnTransformer handles numerical and categorical features.
2. **Numerical preprocessing:** Missing numerical values are handled using median imputation.
3. **Categorical preprocessing:** Missing categorical values are handled using most-frequent imputation, followed by One-Hot Encoding.
4. **Regression:** The processed features are passed to the XGBoost Regressor to predict crop yield.

### Training Approach

- Hyperparameter optimization using Optuna.
- Five-fold cross-validation using KFold.
- Evaluation using MAE, RMSE, and R².

## 4. Model Performance

The following results were obtained during model evaluation.

### Test Set Performance

| Metric | Result |
|---|---:|
| MAE | 16.3448 |
| RMSE | 206.8886 |
| R² Score | 0.94658 |

### Five-Fold Cross-Validation

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| R² Score | 0.96992 | 0.01112 |
| MAE | 13.1755 | — |
| RMSE | 151.93396 | — |

These results indicate strong predictive performance on the evaluated dataset. However, they do not guarantee the same performance on unseen regions or future agricultural conditions.

## 5. Explainability

The project includes model explainability using:

- **SHAP:** Identifies how individual features contribute to predictions and provides global feature-importance analysis.
- **LIME:** Provides local explanations for individual predictions.
- **Soil Bottleneck Diagnostics:** Examines soil nutrient indicators and their model contributions.

SHAP and LIME explanations describe model behavior. They should not be interpreted as proof of real-world causal relationships.

## 6. Model Deployment

The trained model is saved as a Joblib artifact.

An ONNX version is also exported for standardized inference.

**ONNX Verification:**

- Maximum absolute prediction difference: 0.00048828125
- Mean absolute prediction difference: 0.000038746

The ONNX predictions were successfully verified against the original Scikit-learn pipeline on the tested samples.

The ONNX inference preprocessing must preserve the original XGBoost missing-value behavior.

## 7. Limitations

- Predictions depend on the quality and coverage of the available agricultural data.
- Soil and weather information may not capture all local farming conditions.
- Unusual crop-yield values and outliers can affect model performance.
- State-level soil information may not represent conditions at an individual farm.
- Cross-validation used ordinary KFold rather than spatial or geographic cross-validation.
- Model explanations indicate learned associations, not causation.
- Predictions should not be treated as guaranteed crop yields or direct farming instructions.

## 8. Responsible Use

This model is intended for agricultural analysis, experimentation, and decision-support research.

Predictions should be reviewed alongside local soil testing, weather conditions, crop requirements, and expert agricultural advice.

Fertilizer recommendations should not be applied directly without validating nutrient levels, soil units, crop-specific requirements, and appropriate application rates.

## 9. Future Improvements

- Perform spatial and temporal cross-validation.
- Evaluate performance across different crops, states, and seasons.
- Improve data quality and handle extreme yield values.
- Validate soil-based recommendations using agronomic standards.
- Monitor model performance as new agricultural data becomes available.

---

**Model Status:** Experimental – Research and Development  
**Project:** Agriculture, Rural Businesses, Crop Yield and Soil Analysis Forecasting