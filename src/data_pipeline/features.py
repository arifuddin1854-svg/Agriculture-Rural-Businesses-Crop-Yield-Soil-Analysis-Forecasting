import numpy as np
import pandas as pd


def create_agronomic_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    data = dataframe.copy()
    data["npk_total"] = data["nitrogen"] + data["phosphorus"] + data["potassium"]
    data["nk_ratio"] = data["nitrogen"] / (data["potassium"] + 1e-5)
    data["pk_ratio"] = data["phosphorus"] / (data["potassium"] + 1e-5)
    data["aridity_index"] = (data["temperature"] * 100.0) / (
        data["humidity"] + data["rainfall"] + 1e-5
    )
    data["ph_optimal_dev"] = np.abs(data["ph"] - 6.75)
    return data