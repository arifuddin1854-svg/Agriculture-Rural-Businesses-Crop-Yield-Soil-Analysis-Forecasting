from pathlib import Path
from typing import Tuple, Union

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERICAL_FEATURES = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "npk_total",
    "nk_ratio",
    "pk_ratio",
    "aridity_index",
    "ph_optimal_dev",
]
CATEGORICAL_FEATURES = ["soil_type", "crop_type"]
TARGET_COLUMN = "yield_ton_per_ha"


def build_preprocessing_pipeline() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num_pipeline", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat_pipeline", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def process_and_save_data(
    dataframe: pd.DataFrame,
    output_dir: Union[str, Path],
    pipeline_save_path: Union[str, Path],
) -> Tuple[pd.DataFrame, ColumnTransformer]:
    output_directory = Path(output_dir)
    pipeline_path = Path(pipeline_save_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)

    features = dataframe[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    target = dataframe[TARGET_COLUMN]
    pipeline = build_preprocessing_pipeline()
    transformed_features = pipeline.fit_transform(features)

    categorical_encoder = pipeline.named_transformers_["cat_pipeline"].named_steps[
        "encoder"
    ]
    feature_names = NUMERICAL_FEATURES + list(
        categorical_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    )
    processed_dataframe = pd.DataFrame(transformed_features, columns=feature_names)
    processed_dataframe[TARGET_COLUMN] = target.reset_index(drop=True)
    processed_dataframe.to_csv(
        output_directory / "processed_agri_data.csv", index=False
    )
    joblib.dump(pipeline, pipeline_path)
    return processed_dataframe, pipeline