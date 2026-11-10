import os
import joblib
import onnxmltools

from onnxmltools.convert.common.data_types import FloatTensorType

# Paths
MODEL_PATH = "models/crop_yield_xgb.joblib"
OUTPUT_PATH = "models/crop_yield_model.onnx"

# Create models folder if needed
os.makedirs("models", exist_ok=True)

# Load trained pipeline
pipeline = joblib.load(MODEL_PATH)

# Extract trained XGBoost model
xgb_model = pipeline.named_steps["model"]

# Get number of features after preprocessing
preprocessor = pipeline.named_steps["preprocessor"]
feature_count = len(preprocessor.get_feature_names_out())

print("Transformed feature count:", feature_count)

# Define ONNX input
initial_types = [
    ("input", FloatTensorType([None, feature_count]))
]

# Convert XGBoost model
onnx_model = onnxmltools.convert_xgboost(
    xgb_model,
    initial_types=initial_types,
    target_opset=15
)

# Save ONNX model
onnxmltools.utils.save_model(
    onnx_model,
    OUTPUT_PATH
)

print("ONNX model exported successfully!")
print("Saved at:", OUTPUT_PATH)