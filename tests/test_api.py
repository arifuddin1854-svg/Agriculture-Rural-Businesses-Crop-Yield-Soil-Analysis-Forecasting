from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

from src.api.routes import predict as predict_route


def mock_inference(**kwargs):
    """Return a predictable inference result for API testing."""
    return {
        "crop": kwargs["crop"],
        "predicted_yield": 5.25,
        "baseline_yield": 4.50,
        "potential_gain": 0.75,
        "confidence_score": 0.93,
        "model_version": "test_model_v1",
        "recommendations": [],
        "shap_values": {
            "nitrogen": 0.12,
            "ph": -0.03,
        },
        "input_features": {
            "nitrogen": kwargs["nitrogen"],
            "phosphorus": kwargs["phosphorus"],
            "potassium": kwargs["potassium"],
            "ph": kwargs["ph"],
            "rainfall": kwargs["rainfall"],
            "temp_max": kwargs["temp_max"],
        },
    }


def mock_database_logging(**kwargs):
    """Return mock database records without writing to the database."""
    return (
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(id=uuid4(), created_at=datetime.utcnow()),
    )


def test_yield_prediction_success(client, monkeypatch):
    """Verify that a valid prediction request returns the expected response."""
    monkeypatch.setattr(predict_route, "predict_yield", mock_inference)
    monkeypatch.setattr(
        predict_route,
        "log_assessment_and_prediction",
        mock_database_logging,
    )

    payload = {
        "crop": "Wheat",
        "nitrogen": 85,
        "phosphorus": 42,
        "potassium": 120,
        "ph": 6.4,
    }

    response = client.post("/predict/yield", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert result["crop"] == "Wheat"
    assert result["predicted_yield"] == 5.25
    assert result["baseline_yield"] == 4.50
    assert result["potential_gain"] == 0.75
    assert result["confidence_score"] == 0.93
    assert result["model_version"] == "test_model_v1"
    assert "prediction_id" in result
    assert "soil_log_id" in result


def test_yield_prediction_rejects_missing_fields(client):
    """Verify that required soil parameters cannot be omitted."""
    payload = {
        "crop": "Wheat",
        "nitrogen": 85,
        "phosphorus": 42,
    }

    response = client.post("/predict/yield", json=payload)

    assert response.status_code == 422


def test_yield_prediction_rejects_invalid_ph(client):
    """Verify that pH values outside the allowed range are rejected."""
    payload = {
        "crop": "Wheat",
        "nitrogen": 85,
        "phosphorus": 42,
        "potassium": 120,
        "ph": 15,
    }

    response = client.post("/predict/yield", json=payload)

    assert response.status_code == 422


def test_yield_prediction_rejects_negative_nitrogen(client):
    """Verify that negative nitrogen values are rejected."""
    payload = {
        "crop": "Wheat",
        "nitrogen": -10,
        "phosphorus": 42,
        "potassium": 120,
        "ph": 6.4,
    }

    response = client.post("/predict/yield", json=payload)

    assert response.status_code == 422
