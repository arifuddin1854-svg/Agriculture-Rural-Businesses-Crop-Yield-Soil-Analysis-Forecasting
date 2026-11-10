
import joblib
import numpy as np
import pandas as pd
import pytest
import onnxruntime as ort

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "crop_yield_xgb.joblib"
ONNX_PATH = ROOT / "models" / "crop_yield_model.onnx"
DATA_PATH = ROOT / "data" / "processed" / "unified_agri_dataset.csv"


@pytest.fixture(scope="module")
def pipeline():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(
        columns=["yield", "area", "production"],
        errors="ignore"
    )

    y = pd.to_numeric(df["yield"], errors="coerce")
    valid = y.notna() & (y >= 0)

    return X.loc[valid].head(5).copy()


def test_model_loads(pipeline):
    assert pipeline is not None
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_prediction_shape(pipeline, sample_data):
    predictions = pipeline.predict(sample_data)

    assert predictions.shape == (len(sample_data),)


def test_predictions_are_finite(pipeline, sample_data):
    predictions = pipeline.predict(sample_data)

    assert np.all(np.isfinite(predictions))


def test_predictions_are_repeatable(pipeline, sample_data):
    first = pipeline.predict(sample_data)
    second = pipeline.predict(sample_data)

    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)


def test_extreme_soil_inputs_do_not_crash(pipeline, sample_data):
    low_sample = sample_data.iloc[[0]].copy()
    high_sample = sample_data.iloc[[0]].copy()

    soil_columns = ["N", "P", "K", "pH"]

    for column in soil_columns:
        if column in low_sample.columns:
            low_sample[column] = 0
            high_sample[column] = 1000

    low_prediction = pipeline.predict(low_sample)
    high_prediction = pipeline.predict(high_sample)

    assert np.all(np.isfinite(low_prediction))
    assert np.all(np.isfinite(high_prediction))


def test_onnx_predictions_match_pipeline(pipeline, sample_data):
    if not ONNX_PATH.exists():
        pytest.skip("ONNX model not found")

    transformed = pipeline.named_steps["preprocessor"].transform(
        sample_data
    )

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=np.float32)

    # Preserve sparse XGBoost missing-value behavior
    transformed[transformed == 0] = np.nan

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    onnx_predictions = session.run(
        None,
        {input_name: transformed}
    )[0].flatten()

    original_predictions = pipeline.predict(sample_data)

    np.testing.assert_allclose(
        onnx_predictions,
        original_predictions,
        rtol=1e-4,
        atol=1e-3
    )
