"""
Train and export RandomForestRegressor baseline model for yield prediction.
"""
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

def main():
    parquet_path = Path("data/processed/unified_agri_dataset.parquet")
    if not parquet_path.exists():
        print(f"Dataset not found at {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    features = ["ph", "nitrogen", "phosphorus", "potassium", "cumulative_precipitation", "temp_max_mean"]
    
    # One-hot encode crop_type
    crop_dummies = pd.get_dummies(df["crop_type"], prefix="crop", dtype=float)
    X = pd.concat([df[features], crop_dummies], axis=1)
    y = df["yield_mt_per_ha"]

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X, y)

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "best_yield_regressor.joblib"

    artifact = {
        "model": model,
        "features": features,
        "dummy_columns": crop_dummies.columns.tolist(),
        "crops": df["crop_type"].unique().tolist(),
        "all_feature_names": X.columns.tolist(),
        "version": "rf_ensemble_v1.0.0",
    }
    joblib.dump(artifact, out_path)
    print(f"Model saved successfully to {out_path} with {len(X)} training samples.")

if __name__ == "__main__":
    main()
