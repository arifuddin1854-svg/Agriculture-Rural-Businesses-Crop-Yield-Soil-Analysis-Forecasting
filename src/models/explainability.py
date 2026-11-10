from lime.lime_tabular import LimeTabularExplainer
import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Paths and model loading
# --------------------------------------------------

MODEL_PATH = "models/crop_yield_xgb.joblib"
DATA_PATH = "data/processed/unified_agri_dataset.csv"

os.makedirs("reports", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)

# Extract preprocessing and trained XGBoost model
preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["model"]

# TreeExplainer works directly with the trained tree model
explainer = shap.TreeExplainer(xgb_model)

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()


# --------------------------------------------------
# 2. Prepare input data
# --------------------------------------------------

def prepare_features(input_data):
    """
    Convert raw input features into the format
    expected by the trained XGBoost model.
    """

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    transformed = preprocessor.transform(input_data)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    return np.asarray(transformed)


# --------------------------------------------------
# 3. Individual prediction explanation
# --------------------------------------------------

def explain_prediction(input_data):
    """
    Predict crop yield and explain the contribution
    of each feature using SHAP.
    """

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    prediction = float(pipeline.predict(input_data)[0])

    transformed = prepare_features(input_data)

    shap_values = explainer.shap_values(transformed)

    contributions = np.asarray(shap_values)

    if contributions.ndim == 3:
        contributions = contributions[0]

    contributions = contributions.reshape(-1)

    explanation = pd.DataFrame({
        "feature": feature_names,
        "feature_value": transformed[0],
        "shap_value": contributions
    })

    explanation["impact"] = np.where(
        explanation["shap_value"] > 0,
        "Increases predicted yield",
        "Decreases predicted yield"
    )

    explanation = explanation.sort_values(
        by="shap_value",
        key=lambda values: values.abs(),
        ascending=False
    )

    return prediction, explanation


# --------------------------------------------------
# 4. Soil bottleneck diagnostic
# --------------------------------------------------

def diagnose_soil_bottlenecks(explanation):
    """
    Identify soil features with negative SHAP
    contributions to the predicted yield.
    """

    soil_features = ["N", "P", "K", "pH"]

    bottlenecks = []

    for _, row in explanation.iterrows():

        feature_name = str(row["feature"]).split("__")[-1]

        if feature_name in soil_features and row["shap_value"] < 0:

            bottlenecks.append(
                f"{feature_name} has a negative contribution "
                f"to predicted yield "
                f"(SHAP: {row['shap_value']:.4f}). "
                f"Observed value: {row['feature_value']:.4f}."
            )

    if not bottlenecks:
        bottlenecks.append(
            "No negative SHAP contributions were detected "
            "for the soil features."
        )

    return bottlenecks


# --------------------------------------------------
# 5. Global SHAP summary plot
# --------------------------------------------------

def generate_shap_summary(sample_size=500):

    df = pd.read_csv(DATA_PATH)

    target = "yield"
    excluded = [target, "area", "production"]

    X = df.drop(columns=excluded, errors="ignore")

    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna() & (y >= 0)

    X = X.loc[valid].copy()

    # Use a representative sample for plotting
    sample = X.sample(
        n=min(sample_size, len(X)),
        random_state=42
    )

    transformed = prepare_features(sample)

    shap_values = explainer.shap_values(transformed)

    plt.figure(figsize=(12, 8))

    shap.summary_plot(
        shap_values,
        transformed,
        feature_names=feature_names,
        show=False,
        max_display=20
    )

    plt.tight_layout()

    output_path = "reports/shap_summary.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"SHAP summary plot saved: {output_path}")


# --------------------------------------------------
# 6. Main execution
# --------------------------------------------------
# --------------------------------------------------
# 6. LIME local explanation fallback
# --------------------------------------------------

def lime_explanation(input_data, num_features=10, sample_size=500):
    """
    Explain an individual prediction using LIME.
    Uses the transformed features and trained XGBoost model.
    """

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    # Load original training data
    df = pd.read_csv(DATA_PATH)

    excluded = ["yield", "area", "production"]
    X = df.drop(columns=excluded, errors="ignore")

    y = pd.to_numeric(df["yield"], errors="coerce")
    valid = y.notna() & (y >= 0)
    X = X.loc[valid]

    # Create background sample
    background = X.sample(
        n=min(sample_size, len(X)),
        random_state=42
    )

    # Transform background and input using fitted preprocessing
    background_transformed = prepare_features(background)
    input_transformed = prepare_features(input_data)

    # Initialize LIME
    lime_explainer = LimeTabularExplainer(
        training_data=background_transformed,
        feature_names=list(feature_names),
        mode="regression",
        discretize_continuous=True,
        random_state=42
    )

    # Generate local explanation
    explanation = lime_explainer.explain_instance(
        data_row=input_transformed[0],
        predict_fn=xgb_model.predict,
        num_features=min(num_features, len(feature_names))
    )

    return explanation

if __name__ == "__main__":

    df = pd.read_csv(DATA_PATH)

    target = "yield"
    excluded = [target, "area", "production"]

    X = df.drop(columns=excluded, errors="ignore")

    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna() & (y >= 0)

    X = X.loc[valid].copy()

    # Explain one example from the dataset
    sample_input = X.iloc[[0]]

    prediction, explanation = explain_prediction(sample_input)

    print("\nPredicted crop yield:", prediction)

    print("\nTop SHAP feature contributions:")
    print(explanation.head(10).to_string(index=False))

    print("\nSoil bottleneck diagnostic:")

    for message in diagnose_soil_bottlenecks(explanation):
        print("-", message)

    # Generate global SHAP visualization
    generate_shap_summary()

        # LIME local explanation
    print("\nLIME local explanation:")

    lime_result = lime_explanation(sample_input)

    for feature, contribution in lime_result.as_list():
        print(f"{feature}: {contribution:.4f}")