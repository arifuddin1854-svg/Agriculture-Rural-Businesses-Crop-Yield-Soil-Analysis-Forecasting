"""Integration and unit tests for Axlero Agri-Tech API endpoints."""
import uuid


def test_root_endpoint(client):
    """Verify root greeting endpoint returns 200 and docs links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["docs"] == "/docs"
    assert data["health"] == "/health"


def test_get_health(client):
    """Verify GET /health returns 200 and complies with HealthResponse schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] in ["healthy", "degraded"]
    assert "timestamp" in data
    assert data["version"] == "1.0.0"
    assert "service" in data
    assert "database" in data
    assert "status" in data["database"]
    assert "environment" in data


def test_get_api_v1_health(client):
    """Verify GET /api/v1/health prefix endpoint functions identically."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["version"] == "1.0.0"


def test_user_lifecycle(client):
    """Test user registration, duplicate email rejection, and retrieval."""
    user_payload = {
        "email": "farmer.patel@axlero.agri",
        "password": "SecurePassword123!",
        "full_name": "Arif Patel",
        "role": "farmer",
        "farm_name": "Green Valley Organic Acres",
        "district": "Ludhiana",
    }
    
    # 1. Create user
    res = client.post("/api/v1/users/", json=user_payload)
    assert res.status_code == 201
    user_data = res.json()
    assert user_data["email"] == user_payload["email"]
    assert user_data["role"] == "farmer"
    user_id = user_data["id"]

    # 2. Reject duplicate email
    res_dup = client.post("/api/v1/users/", json=user_payload)
    assert res_dup.status_code == 400

    # 3. Retrieve user by ID
    res_get = client.get(f"/api/v1/users/{user_id}")
    assert res_get.status_code == 200
    assert res_get.json()["id"] == user_id

    # 4. List users
    res_list = client.get("/api/v1/users/")
    assert res_list.status_code == 200
    assert any(u["id"] == user_id for u in res_list.json())


def test_soil_log_and_prediction_pipeline(client):
    """Test full cycle: Create User -> Submit Soil Log -> Record Yield Prediction."""
    # 1. Create User
    user_res = client.post(
        "/api/v1/users/",
        json={
            "email": "agronomist.sharma@axlero.agri",
            "password": "StrongPassword456!",
            "full_name": "Dr. Sunita Sharma",
            "role": "agronomist",
            "farm_name": "Punjab Agro Lab",
            "district": "Patiala",
        },
    )
    assert user_res.status_code == 201
    user_id = user_res.json()["id"]

    # 2. Create Soil Log
    soil_payload = {
        "user_id": user_id,
        "field_name": "Plot 12-B South Tier",
        "district": "Patiala",
        "latitude": 30.3398,
        "longitude": 76.3869,
        "nitrogen": 85.0,
        "phosphorus": 42.0,
        "potassium": 120.0,
        "ph": 5.6,
        "organic_matter": 2.4,
        "moisture": 18.5,
        "soil_type": "Clay Loam",
        "notes": "Pre-sowing baseline soil test",
    }
    soil_res = client.post("/api/v1/soil-logs/", json=soil_payload)
    assert soil_res.status_code == 201
    soil_data = soil_res.json()
    soil_log_id = soil_data["id"]
    assert soil_data["ph"] == 5.6
    assert soil_data["nitrogen"] == 85.0

    # 3. Query Soil Logs by District
    list_res = client.get("/api/v1/soil-logs/?district=Patiala")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # 4. Record Yield Prediction linked to the Soil Log
    pred_payload = {
        "user_id": user_id,
        "soil_log_id": soil_log_id,
        "crop": "Wheat",
        "season": "Rabi 2026",
        "model_version": "rf_v1.0.0",
        "predicted_yield": 5.61,
        "confidence_score": 0.934,
        "baseline_yield": 4.82,
        "potential_gain": 0.79,
        "recommendations": [
            {
                "category": "Soil pH Alert",
                "current_level": 5.6,
                "optimal_range": "6.2 - 6.8",
                "recommendation": "Apply 1.2 tons/ha of agricultural limestone.",
                "expected_impact_mt_ha": 0.48,
                "percentage_gain": 10.0,
            },
            {
                "category": "Nitrogen (N) Status",
                "current_level": 85.0,
                "optimal_range": "110 - 130 kg/ha",
                "recommendation": "Top-dress with 35 kg/ha nitrogenous organic fertilizer.",
                "expected_impact_mt_ha": 0.31,
                "percentage_gain": 6.4,
            },
        ],
        "shap_values": {
            "ph": -0.48,
            "nitrogen": -0.31,
            "phosphorus": 0.12,
            "potassium": 0.18,
        },
        "input_features": {
            "nitrogen": 85.0,
            "phosphorus": 42.0,
            "potassium": 120.0,
            "ph": 5.6,
            "rainfall": 550.0,
            "temp_max": 29.5,
        },
    }
    pred_res = client.post("/api/v1/predictions/", json=pred_payload)
    assert pred_res.status_code == 201
    pred_data = pred_res.json()
    assert pred_data["crop"] == "Wheat"
    assert pred_data["predicted_yield"] == 5.61
    assert pred_data["potential_gain"] == 0.79

    # 5. Retrieve prediction by ID
    pred_id = pred_data["id"]
    get_pred_res = client.get(f"/api/v1/predictions/{pred_id}")
    assert get_pred_res.status_code == 200
    assert get_pred_res.json()["model_version"] == "rf_v1.0.0"
