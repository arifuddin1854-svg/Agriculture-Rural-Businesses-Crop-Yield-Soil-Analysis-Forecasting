
import os
import joblib
import numpy as np
import pandas as pd

from explainability import (
    explainer,
    prepare_features,
    feature_names,
    DATA_PATH
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

OUTPUT_PATH = "models/shap_cache.joblib"
SAMPLE_SIZE = 100

# --------------------------------------------------
# 2. Load and prepare actual dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

excluded = ["yield", "area", "production"]

X = df.drop(columns=excluded, errors="ignore")

y = pd.to_numeric(df["yield"], errors="coerce")
valid = y.notna() & (y >= 0)

X = X.loc[valid].copy()

# Select representative records from actual data
sample = X.sample(
    n=min(SAMPLE_SIZE, len(X)),
    random_state=42
)

print("Selected records:", len(sample))

# --------------------------------------------------
# 3. Calculate SHAP values
# --------------------------------------------------

transformed = prepare_features(sample)

shap_values = explainer.shap_values(transformed)

shap_values = np.asarray(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[0]

# --------------------------------------------------
# 4. Generate predictions
# --------------------------------------------------

pipeline = joblib.load("models/crop_yield_xgb.joblib")

predictions = pipeline.predict(sample)

# --------------------------------------------------
# 5. Create SHAP cache
# --------------------------------------------------

shap_cache = {
    "sample_indices": sample.index.to_list(),
    "input_records": sample.reset_index(drop=True),
    "feature_names": list(feature_names),
    "shap_values": shap_values,
    "predictions": predictions,
    "sample_size": len(sample),
    "model_path": "models/crop_yield_xgb.joblib",
    "description": "Precomputed SHAP explanations for representative agricultural records"
}

# --------------------------------------------------
# 6. Save cache
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(shap_cache, OUTPUT_PATH)

print("\nSHAP cache generated successfully!")
print("Saved at:", OUTPUT_PATH)
print("SHAP values shape:", shap_values.shape)
print("Number of predictions:", len(predictions))
