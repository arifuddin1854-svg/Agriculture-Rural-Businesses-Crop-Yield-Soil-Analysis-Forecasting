"""
Week 2: EDA & Spatial-Temporal Feature Engineering Pipeline

Responsible for:
- Computing agronomic interaction features (Rainfall * Temperature index, N:P:K ratios)
- Encoding categorical variables (crop type, soil type)
- Exporting the normalized training dataset to data/processed/
"""

import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_agronomic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes agronomic interaction indices and nutrient balance ratios.
    """
    data = df.copy()

    # N-P-K Stoichiometric ratio features
    if all(col in data.columns for col in ["nitrogen", "phosphorus", "potassium"]):
        logger.info("Computing N:P:K nutrient balance features...")
        data["npk_total"] = data["nitrogen"] + data["phosphorus"] + data["potassium"]
        data["n_ratio"] = data["nitrogen"] / (data["npk_total"] + 1e-5)
        data["p_ratio"] = data["phosphorus"] / (data["npk_total"] + 1e-5)
        data["k_ratio"] = data["potassium"] / (data["npk_total"] + 1e-5)
        data["n_to_p"] = data["nitrogen"] / (data["phosphorus"] + 1e-5)

    # Weather & Thermal-Hydrological interaction indices
    precip_col = [c for c in data.columns if "precip" in c.lower() or "rain" in c.lower()]
    temp_col = [c for c in data.columns if "temp" in c.lower() and "max" in c.lower()]
    
    if precip_col and temp_col:
        p_c = precip_col[0]
        t_c = temp_col[0]
        logger.info(f"Computing Rainfall * Temperature interaction using {p_c} and {t_c}...")
        data["hydro_thermal_stress_index"] = data[p_c] / (data[t_c] + 1e-5)

    return data


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str] = None) -> pd.DataFrame:
    """
    Applies one-hot or ordinal encoding to categorical attributes.
    """
    if cat_cols is None:
        cat_cols = [c for c in ["crop_type", "soil_type", "season"] if c in df.columns]

    if not cat_cols:
        return df

    logger.info(f"Encoding categorical columns: {cat_cols}")
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--input", type=str, default="data/processed/unified_raw.parquet", help="Input unified dataset")
    parser.add_argument("--output", type=str, default="data/processed/train_features.parquet", help="Output engineered dataset")
    args = parser.parse_args()

    logger.info("Week 2 Feature Engineering pipeline initialized.")


if __name__ == "__main__":
    main()
