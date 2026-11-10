import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort

# Load original pipeline
pipeline = joblib.load("models/crop_yield_xgb.joblib")

# Load dataset
data_path = "data/processed/unified_agri_dataset.csv"
df = pd.read_csv(data_path)

# Prepare input features
X = df.drop(
    columns=["yield", "area", "production"],
    errors="ignore"
)

y = pd.to_numeric(df["yield"], errors="coerce")
X = X.loc[y.notna() & (y >= 0)]

# Select 20 test samples
sample = X.head(20)

# Original model predictions
original_predictions = pipeline.predict(sample)

# Apply original preprocessing
preprocessor = pipeline.named_steps["preprocessor"]
transformed = preprocessor.transform(sample)

if hasattr(transformed, "toarray"):
    transformed = transformed.toarray()

transformed = np.asarray(transformed, dtype=np.float32)

# Preserve XGBoost sparse missing-value behavior
transformed[transformed == 0] = np.nan

# Load ONNX model
session = ort.InferenceSession(
    "models/crop_yield_model.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ONNX predictions
onnx_predictions = session.run(
    [output_name],
    {input_name: transformed}
)[0].reshape(-1)

# Compare predictions
difference = np.abs(original_predictions - onnx_predictions)

print("Original predictions:", original_predictions[:5])
print("ONNX predictions:", onnx_predictions[:5])

print("\nMaximum absolute difference:", difference.max())
print("Mean absolute difference:", difference.mean())

if np.allclose(
    original_predictions,
    onnx_predictions,
    rtol=1e-4,
    atol=1e-4
):
    print("\nSUCCESS: ONNX predictions match the original model.")
else:
    print("\nWARNING: Prediction differences detected.")