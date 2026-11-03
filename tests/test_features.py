import numpy as np
import pandas as pd

from src.data_pipeline.features import create_agronomic_features


def sample_dataframe():
    return pd.DataFrame(
        {
            "nitrogen": [10],
            "phosphorus": [20],
            "potassium": [5],
            "temperature": [25],
            "humidity": [50],
            "rainfall": [10],
            "ph": [7.25],
        }
    )


def test_agronomic_features_are_calculated_correctly():
    dataframe = sample_dataframe()

    result = create_agronomic_features(dataframe)

    assert result.loc[0, "npk_total"] == 35
    assert np.isclose(result.loc[0, "nk_ratio"], 10 / (5 + 1e-5))
    assert np.isclose(result.loc[0, "pk_ratio"], 20 / (5 + 1e-5))
    assert np.isclose(
        result.loc[0, "aridity_index"],
        (25 * 100) / (50 + 10 + 1e-5),
    )
    assert np.isclose(result.loc[0, "ph_optimal_dev"], 0.5)


def test_original_dataframe_is_not_modified():
    dataframe = sample_dataframe()
    original = dataframe.copy()

    create_agronomic_features(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)


def test_zero_potassium_does_not_cause_division_by_zero():
    dataframe = sample_dataframe()
    dataframe.loc[0, "potassium"] = 0

    result = create_agronomic_features(dataframe)

    assert np.isfinite(result.loc[0, "nk_ratio"])
    assert np.isfinite(result.loc[0, "pk_ratio"])
