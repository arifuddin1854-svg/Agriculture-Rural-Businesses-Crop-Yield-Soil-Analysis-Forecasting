"""
Week 3: Supervised Regression Modeling & Spatial Cross-Validation

Responsible for:
- Ensemble regressors: Random Forest, Gradient Boosting, XGBoost/LightGBM
- Spatial Block Cross-Validation (GroupKFold by region/district)
- Evaluation metrics: RMSE, MAE, MAPE, R-squared
"""

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_metrics(y_true, y_pred) -> dict:
    """Computes RMSE, MAE, MAPE, and R2."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape, "R2": r2}


def run_spatial_cross_validation(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5, model_type: str = "rf"):
    """
    Executes Spatial Group-KFold CV to ensure neighboring farms do not contaminate test folds.
    """
    gkf = GroupKFold(n_splits=n_splits)
    metrics_list = []

    logger.info(f"Starting {n_splits}-fold Spatial Cross-Validation using groups: {groups.nunique()} unique clusters.")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        if model_type == "rf":
            model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        else:
            model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores = evaluate_metrics(y_val, preds)
        metrics_list.append(scores)
        logger.info(f"Fold {fold} - RMSE: {scores['RMSE']:.4f}, MAPE: {scores['MAPE (%)']:.2f}%, R2: {scores['R2']:.4f}")

    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    logger.info(f"Average Spatial CV Metrics: {avg_metrics}")
    return avg_metrics, model


def main():
    parser = argparse.ArgumentParser(description="Spatial Cross-Validation & Regression Training")
    parser.add_argument("--train-data", type=str, default="data/processed/train_features.parquet")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--save-path", type=str, default="models/best_yield_regressor.joblib")
    args = parser.parse_args()

    logger.info("Week 3 Modeling pipeline initialized.")


if __name__ == "__main__":
    main()
