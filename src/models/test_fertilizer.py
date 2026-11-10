import pandas as pd

from fertilizer_recommender import recommend_fertilizer

# Example SHAP contributions for testing
example_explanation = pd.DataFrame({
    "feature": [
        "numeric__N",
        "numeric__P",
        "numeric__K"
    ],
    "shap_value": [
        -1198.85,
        -19.58,
        10.25
    ]
})

results = recommend_fertilizer(
    "Assam",
    example_explanation
)

print("\nFertilizer Recommendations:\n")

for result in results:
    print("-", result)