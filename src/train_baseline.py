
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load datasets
crop_df = pd.read_csv("data/crop_yield.csv")
soil_df = pd.read_csv("data/state_soil_data.csv")
weather_df = pd.read_csv("data/state_weather_data_1997_2020.csv")

# 2. Merge datasets
df = crop_df.merge(soil_df, on="state", how="left")
df = df.merge(weather_df, on=["state", "year"], how="left")

# 3. Save the unified dataset
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/unified_agri_dataset.csv", index=False)

# 4. Select features and target
# Exclude area and production to avoid direct yield leakage.
target = "yield"

categorical = ["crop", "season", "state"]
numeric = [
    "year", "fertilizer", "pesticide",
    "N", "P", "K", "pH",
    "avg_temp_c", "total_rainfall_mm",
    "avg_humidity_percent"
]

X = df[categorical + numeric]
y = pd.to_numeric(df[target], errors="coerce")

# Remove rows with missing or invalid target values
valid = y.notna() & y.map(lambda v: pd.notna(v) and abs(v) != float("inf"))
X = X.loc[valid]
y = y.loc[valid]

# 5. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Preprocess the features
preprocessor = ColumnTransformer([
    ("categorical",
     Pipeline([
         ("imputer", SimpleImputer(strategy="most_frequent")),
         ("encoder", OneHotEncoder(handle_unknown="ignore"))
     ]),
     categorical),

    ("numeric",
     SimpleImputer(strategy="median"),
     numeric)
])

# 7. Train and compare baseline models
models = {
    "Dummy Baseline": DummyRegressor(strategy="median"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
}

results = []
trained_models = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = pipeline

# 8. Display and save the benchmark results
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\nBaseline Model Comparison:")
print(results_df.to_string(index=False))

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/baseline_metrics.csv", index=False)

# Save the lowest-RMSE model as a provisional best model
best_name = results_df.iloc[0]["Model"]
os.makedirs("models", exist_ok=True)
joblib.dump(trained_models[best_name], "models/best_baseline_model.joblib")

print("\nProvisional best model:", best_name)
print("Saved dataset, metrics, and model.")
