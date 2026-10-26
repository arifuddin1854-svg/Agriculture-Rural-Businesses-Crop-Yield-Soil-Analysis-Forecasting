"""
Comprehensive tests for Engineer 3 deliverables:
1. FastAPI Inference Route (POST /predict/yield)
2. Pydantic Data Contracts & Strict Range Constraints (pH 0-14, positive NPK)
3. Soil Recommendation Route (POST /recommend/crop)
4. Database Logging Handler (SQLAlchemy Yield & Soil Log Records)
5. Standardized Error & Exception Handling Middleware
"""

import uuid
from src.api.models.soil_log import SoilLog
from src.api.models.yield_prediction import YieldPrediction
from src.api.models.user import User


def test_predict_yield_endpoint_success(client, db_session):
    """
    Test POST /predict/yield endpoint:
    - Ingests real-time soil parameter payload
    - Computes yield estimates and agronomic prescriptions
    - Stores records in database tables (SoilLog and YieldPrediction)
    """
    payload = {
        "crop": "Wheat",
        "nitrogen": 85.0,
        "phosphorus": 42.0,
        "potassium": 120.0,
        "ph": 5.6,
        "rainfall": 550.0,
        "temp_max": 29.5,
        "field_name": "Field-North-Plot1",
        "district": "Patiala",
        "season": "Rabi 2026",
    }

    response = client.post("/predict/yield", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Verify prediction response contract
    assert data["crop"] == "Wheat"
    assert data["predicted_yield"] > 0
    assert data["baseline_yield"] > 0
    assert data["potential_gain"] >= 0
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert "model_version" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) > 0
    assert "shap_values" in data
    assert "ph" in data["shap_values"]
    assert "nitrogen" in data["shap_values"]
    assert "input_features" in data
    assert data["prediction_id"] is not None
    assert data["soil_log_id"] is not None

    # Verify database persistence
    pred_db = db_session.query(YieldPrediction).filter(YieldPrediction.id == uuid.UUID(data["prediction_id"])).first()
    assert pred_db is not None
    assert pred_db.crop == "Wheat"
    assert pred_db.predicted_yield == data["predicted_yield"]

    soil_db = db_session.query(SoilLog).filter(SoilLog.id == uuid.UUID(data["soil_log_id"])).first()
    assert soil_db is not None
    assert soil_db.nitrogen == 85.0
    assert soil_db.ph == 5.6
    assert soil_db.field_name == "Field-North-Plot1"


def test_predict_yield_with_registered_user(client, db_session):
    """
    Test that yield prediction properly associates with a registered user if provided.
    """
    # Create user
    user = User(
        email="farmer.singh@axlero.agri",
        hashed_password="hashed_secret_password",
        full_name="Harpreet Singh",
        role="farmer",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    payload = {
        "user_id": str(user.id),
        "crop": "Maize",
        "nitrogen": 130.0,
        "phosphorus": 55.0,
        "potassium": 80.0,
        "ph": 6.5,
        "rainfall": 700.0,
        "temp_max": 28.0,
    }

    response = client.post("/api/v1/predict/yield", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Maize"

    # Verify user association in database
    pred_db = db_session.query(YieldPrediction).filter(YieldPrediction.id == uuid.UUID(data["prediction_id"])).first()
    assert pred_db.user_id == user.id

    soil_db = db_session.query(SoilLog).filter(SoilLog.id == uuid.UUID(data["soil_log_id"])).first()
    assert soil_db.user_id == user.id


def test_pydantic_strict_ph_range_validation(client):
    """
    Test strict boundaries: pH must be between 0.0 and 14.0.
    Invalid inputs must return HTTP 422 with standardized error envelope.
    """
    # Test pH > 14.0
    payload_high_ph = {
        "crop": "Wheat",
        "nitrogen": 100.0,
        "phosphorus": 50.0,
        "potassium": 80.0,
        "ph": 15.5,  # Exceeds max pH
    }
    res_high = client.post("/predict/yield", json=payload_high_ph)
    assert res_high.status_code == 422
    err_data = res_high.json()
    assert "error" in err_data
    assert err_data["error"]["code"] == "VALIDATION_ERROR"
    assert any("ph" in str(d["field"]).lower() for d in err_data["error"]["details"])

    # Test pH < 0.0
    payload_neg_ph = {
        "crop": "Wheat",
        "nitrogen": 100.0,
        "phosphorus": 50.0,
        "potassium": 80.0,
        "ph": -1.2,  # Sub-zero pH
    }
    res_neg = client.post("/predict/yield", json=payload_neg_ph)
    assert res_neg.status_code == 422
    err_neg_data = res_neg.json()
    assert err_neg_data["error"]["code"] == "VALIDATION_ERROR"


def test_pydantic_strict_positive_npk_validation(client):
    """
    Test strict boundaries: NPK values must be positive / non-negative (>= 0).
    """
    # Negative Nitrogen
    payload_neg_n = {
        "crop": "Wheat",
        "nitrogen": -25.0,
        "phosphorus": 40.0,
        "potassium": 60.0,
        "ph": 6.5,
    }
    res = client.post("/predict/yield", json=payload_neg_n)
    assert res.status_code == 422
    data = res.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert any("nitrogen" in str(d["field"]).lower() for d in data["error"]["details"])

    # Negative Phosphorus
    payload_neg_p = {
        "crop": "Wheat",
        "nitrogen": 80.0,
        "phosphorus": -10.0,
        "potassium": 60.0,
        "ph": 6.5,
    }
    res_p = client.post("/predict/yield", json=payload_neg_p)
    assert res_p.status_code == 422

    # Negative Potassium
    payload_neg_k = {
        "crop": "Wheat",
        "nitrogen": 80.0,
        "phosphorus": 40.0,
        "potassium": -5.0,
        "ph": 6.5,
    }
    res_k = client.post("/predict/yield", json=payload_neg_k)
    assert res_k.status_code == 422


def test_pydantic_missing_attributes_validation(client):
    """
    Test missing mandatory attributes (e.g. missing crop or NPK) returns HTTP 422.
    """
    incomplete_payload = {
        "nitrogen": 80.0,
        "phosphorus": 40.0,
        # missing crop and ph
    }
    res = client.post("/predict/yield", json=incomplete_payload)
    assert res.status_code == 422
    data = res.json()
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert len(data["error"]["details"]) >= 2


def test_recommend_crop_endpoint(client, db_session):
    """
    Test POST /recommend/crop:
    - Evaluates NPK values against optimal crop lookup tables
    - Suggests top target crops ranked by suitability
    - Persists soil assessment log in database
    """
    payload = {
        "nitrogen": 120.0,
        "phosphorus": 52.0,
        "potassium": 55.0,
        "ph": 6.6,
        "rainfall": 650.0,
        "temperature": 22.0,
        "top_k": 3,
        "district": "Ludhiana",
    }

    response = client.post("/recommend/crop", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Response verification
    assert "top_crop" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) == 3

    # Wheat should be ranked near top given N=120, P=52, K=55, pH=6.6, Temp=22
    top_candidate = data["recommendations"][0]
    assert top_candidate["rank"] == 1
    assert top_candidate["suitability_score"] > 80.0
    assert top_candidate["match_status"] in ["Highly Suitable", "Moderately Suitable"]
    assert "optimal_ranges" in top_candidate
    assert "nitrogen_kg_ha" in top_candidate["optimal_ranges"]
    assert "diagnostic_notes" in top_candidate

    # Verify soil log persistence in DB
    assert data["soil_log_id"] is not None
    soil_log = db_session.query(SoilLog).filter(SoilLog.id == uuid.UUID(data["soil_log_id"])).first()
    assert soil_log is not None
    assert soil_log.nitrogen == 120.0
    assert soil_log.ph == 6.6


def test_recommend_crop_legume_profile(client):
    """
    Test crop recommendation with low nitrogen (legume profile like Soybean / Chickpea).
    """
    payload = {
        "nitrogen": 30.0,  # Low N, typical for nitrogen-fixing legumes
        "phosphorus": 60.0,
        "potassium": 45.0,
        "ph": 6.4,
        "top_k": 4,
    }
    response = client.post("/api/v1/recommend/crop", json=payload)
    assert response.status_code == 200
    data = response.json()

    recommended_crops = [r["crop"] for r in data["recommendations"]]
    # Should identify legumes (Soybean, Chickpea, Kidney Beans, Lentil) as highly suitable
    assert any(crop in recommended_crops for crop in ["Soybean", "Chickpea", "Kidney Beans", "Lentil"])


def test_standardized_error_response_format_404(client):
    """
    Test that 404 responses conform to the standardized error schema.
    """
    fake_id = uuid.uuid4()
    response = client.get(f"/api/v1/predictions/{fake_id}")
    assert response.status_code == 404
    data = response.json()

    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"
    assert data["error"]["status_code"] == 404
    assert "timestamp" in data
