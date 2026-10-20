from pathlib import Path
from typing import Union

import pandas as pd


REQUIRED_COLUMNS = [
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "soil_type",
    "crop_type",
    "yield_ton_per_ha",
]


class DataIngestionError(Exception):
    pass


def load_raw_data(file_path: Union[str, Path]) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"Raw data file not found at: {path.resolve()}")

    try:
        data = pd.read_csv(path)
    except Exception as error:
        raise DataIngestionError(f"Failed to read CSV file: {error}") from error

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise DataIngestionError(f"Dataset missing required columns: {missing_columns}")

    return data