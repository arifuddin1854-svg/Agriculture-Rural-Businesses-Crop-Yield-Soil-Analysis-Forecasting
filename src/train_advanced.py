
import os
import json
import joblib
import numpy as np
import pandas as pd
import optuna

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Create output folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# 2. Load dataset
DATA_PATH = "data/processed/unified_agri_dataset.csv"

if not os.path.exists(DATA_PATH):
    DATA_PATH = "data/unified_agri_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)
print("Columns:", df.columns.tolist())


# 3. Prepare features and target
target = "yield"

if target not in df.columns:
    raise ValueError("Target column 'yield' not found in dataset.")

# Exclude area and production to prevent target leakage
excluded = [target, "area", "production"]

X = df.drop(columns=excluded, errors="ignore")
y = pd.to_numeric(df[target], errors="coerce")

# Remove invalid target rows
valid = y.notna() & (y >= 0)
X = X.loc[valid].copy()
y = y.loc[valid].copy()

categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

print("Features:", X.shape[1])
print("Categorical:", categorical_cols)
print("Numeric:", numeric_cols)


# 4. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])


# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. Define models
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

lgbm_model = LGBMRegressor(
    n_estimators=300,
    max_depth=-1,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


# 7. Train and evaluate initial models
models = {
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model
}

metrics = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics[name] = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions))
    }

    print(name, metrics[name])


# 8. Optuna tuning for XGBoost
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1
    }

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(**params))
    ])

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="r2",
        n_jobs=1
    )

    return float(np.mean(scores["test_score"]))


print("\nStarting Optuna tuning...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print("Best Optuna parameters:", study.best_params)
print("Best Optuna CV R2:", study.best_value)


# 9. Train final tuned XGBoost
best_xgb = XGBRegressor(
    **study.best_params,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_xgb)
])

final_pipeline.fit(X_train, y_train)

test_predictions = final_pipeline.predict(X_test)

metrics["Tuned XGBoost"] = {
    "MAE": float(mean_absolute_error(y_test, test_predictions)),
    "RMSE": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
    "R2": float(r2_score(y_test, test_predictions))
}

print("\nTuned XGBoost test metrics:", metrics["Tuned XGBoost"])


# 10. Five-fold cross-validation
print("\nRunning 5-fold cross-validation...")

cv_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_xgb)
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_validate(
    cv_pipeline,
    X,
    y,
    cv=cv,
    scoring={
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error"
    },
    n_jobs=1
)

cv_summary = {
    "5_fold_R2_mean": float(np.mean(cv_scores["test_r2"])),
    "5_fold_R2_std": float(np.std(cv_scores["test_r2"])),
    "5_fold_MAE_mean": float(-np.mean(cv_scores["test_mae"])),
    "5_fold_RMSE_mean": float(np.sqrt(-np.mean(cv_scores["test_mse"])))
}

print("5-fold CV summary:", cv_summary)


# 11. Save model and results
joblib.dump(final_pipeline, "models/crop_yield_xgb.joblib")

pd.DataFrame(metrics).T.to_csv(
    "results/advanced_model_metrics.csv"
)

with open("results/metrics_summary.json", "w") as file:
    json.dump({
        "model_metrics": metrics,
        "cross_validation": cv_summary,
        "best_optuna_parameters": study.best_params
    }, file, indent=4)

study.trials_dataframe().to_csv(
    "results/optuna_trials.csv", index=False
)

print("\nAll tasks completed successfully!")
print("Saved: models/crop_yield_xgb.joblib")
print("Saved: results/advanced_model_metrics.csv")
print("Saved: results/metrics_summary.json")
print("Saved: results/optuna_trials.csv")
