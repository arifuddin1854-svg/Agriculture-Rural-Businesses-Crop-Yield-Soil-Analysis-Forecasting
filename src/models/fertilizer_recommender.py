import pandas as pd
import numpy as np

SOIL_DATA_PATH = "data/raw/state_soil_data.csv"

# Load soil dataset
soil_data = pd.read_csv(SOIL_DATA_PATH)

# Dataset-relative thresholds.
# These identify comparatively low values, NOT confirmed deficiencies.
LOW_NUTRIENT_THRESHOLDS = {
    "N": soil_data["N"].quantile(0.25),
    "P": soil_data["P"].quantile(0.25),
    "K": soil_data["K"].quantile(0.25)
}


def recommend_fertilizer(state, explanation):

    state_soil = soil_data[
        soil_data["state"].str.lower() == state.lower()
    ]

    if state_soil.empty:
        return ["Soil data unavailable for this state."]

    soil = state_soil.iloc[0]

    recommendations = []

    for nutrient in ["N", "P", "K"]:

        feature_name = f"numeric__{nutrient}"

        shap_rows = explanation[
            explanation["feature"] == feature_name
        ]

        if shap_rows.empty:
            continue

        shap_value = float(shap_rows.iloc[0]["shap_value"])
        nutrient_value = float(soil[nutrient])

        threshold = LOW_NUTRIENT_THRESHOLDS[nutrient]

        if nutrient_value < threshold and shap_value < 0:

            recommendations.append(
                f"{nutrient}: Relatively low soil value "
                f"({nutrient_value}) and negative model contribution "
                f"({shap_value:.2f}). Consider a soil-test-based "
                f"{nutrient} fertilizer assessment."
            )

        elif shap_value < 0:

            recommendations.append(
                f"{nutrient}: Negative SHAP contribution detected, "
                f"but the state-level soil value is not below the "
                f"dataset's lower-quartile threshold. "
                f"Further assessment is required."
            )

    if not recommendations:
        recommendations.append(
            "No NPK recommendations triggered by the current rules."
        )

    recommendations.append(
        "Application rates in kg/ha require verified soil-test units, "
        "crop-specific requirements and agronomic recommendations."
    )

    return recommendations