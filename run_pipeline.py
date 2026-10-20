from pathlib import Path

import numpy as np
import pandas as pd

from src.data_pipeline import (
    create_agronomic_features,
    load_raw_data,
    process_and_save_data,
)


def create_sample_data(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    random_generator = np.random.default_rng(42)
    sample_count = 200
    pd.DataFrame(
        {
            "nitrogen": random_generator.uniform(10, 140, sample_count),
            "phosphorus": random_generator.uniform(5, 90, sample_count),
            "potassium": random_generator.uniform(10, 200, sample_count),
            "temperature": random_generator.uniform(15, 38, sample_count),
            "humidity": random_generator.uniform(30, 90, sample_count),
            "ph": random_generator.uniform(4.5, 8.5, sample_count),
            "rainfall": random_generator.uniform(20, 300, sample_count),
            "soil_type": random_generator.choice(
                ["Sandy", "Loamy", "Clay", "Black"], sample_count
            ),
            "crop_type": random_generator.choice(
                ["Wheat", "Maize", "Rice", "Cotton"], sample_count
            ),
            "yield_ton_per_ha": random_generator.uniform(1.2, 7.5, sample_count),
        }
    ).to_csv(file_path, index=False)


def main() -> None:
    raw_path = Path("data/raw/sample_agriculture_data.csv")
    output_directory = Path("data/processed")
    pipeline_path = output_directory / "preprocessor_pipeline.joblib"

    if not raw_path.exists():
        create_sample_data(raw_path)

    dataframe = create_agronomic_features(load_raw_data(raw_path))
    processed_dataframe, _ = process_and_save_data(
        dataframe, output_directory, pipeline_path
    )
    print(f"Pipeline executed successfully. Output shape: {processed_dataframe.shape}")


if __name__ == "__main__":
    main()