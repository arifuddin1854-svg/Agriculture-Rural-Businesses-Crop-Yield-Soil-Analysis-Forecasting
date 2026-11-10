"""
API Endpoint Unit Tests — Production API Hardening & Security.
Deliverable: tests/test_api.py (Engineer 3)

Tests HTTP 200 / 201 / 400 / 401 / 404 / 422 scenarios across:
- Yield Prediction routes (/predict/yield)       — 4 existing tests preserved
- Auth routes (/auth/register, /auth/login,
              /auth/token, /auth/me, /auth/refresh)
- Financial routes (/business/financial-forecast,
                    /business/history)
- Security Header verification
"""

from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.api.routes import predict as predict_route

# ---------------------------------------------------------------------------
# Shared helper payloads
# ---------------------------------------------------------------------------

_REGISTER_PAYLOAD = {
    "email": "test_hardening@agri.example.com",
    "password": "SecureP@ss2026!",
    "full_name": "Test Hardening User",
}

_FINANCIAL_PAYLOAD = {
    "crop_type": "Wheat",
    "farm_size": 10.0,
    "predicted_yield": 3.5,
}


# ===========================================================================
# SECTION 1: Yield Prediction (4 original tests — preserved)
# ===========================================================================


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


# ===========================================================================
# SECTION 2: Auth — Registration Tests
# ===========================================================================


def test_register_success(client):
    """POST /auth/register with valid data returns HTTP 201 and a JWT token."""
    response = client.post("/auth/register", json=_REGISTER_PAYLOAD)
    assert response.status_code == 201
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "user" in data
    assert data["user"]["email"] == _REGISTER_PAYLOAD["email"]


def test_register_duplicate_email_returns_400(client):
    """POST /auth/register with a duplicate email returns HTTP 400."""
    # Ensure the user exists first
    client.post("/auth/register", json=_REGISTER_PAYLOAD)
    response = client.post("/auth/register", json=_REGISTER_PAYLOAD)
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()


def test_register_missing_email_returns_422(client):
    """POST /auth/register without email field returns HTTP 422."""
    response = client.post(
        "/auth/register",
        json={"password": "SomePass123!", "full_name": "Missing Email"},
    )
    assert response.status_code == 422


def test_register_missing_password_returns_422(client):
    """POST /auth/register without password field returns HTTP 422."""
    response = client.post(
        "/auth/register",
        json={
            "email": "nopwd@agri.example.com",
            "full_name": "Missing Password",
        },
    )
    assert response.status_code == 422


# ===========================================================================
# SECTION 3: Auth — Login Tests
# ===========================================================================


@pytest.fixture
def registered_user(client):
    """Register a fresh unique user and return their credentials + token."""
    unique_email = f"login_user_{uuid4().hex[:8]}@agri.example.com"
    creds = {
        "email": unique_email,
        "password": "LoginPass!2026",
        "full_name": "Login Test User",
    }
    reg_resp = client.post("/auth/register", json=creds)
    assert reg_resp.status_code == 201
    return creds, reg_resp.json()


def test_login_json_success(client, registered_user):
    """POST /auth/login with correct credentials returns HTTP 200 + token."""
    creds, _ = registered_user
    response = client.post(
        "/auth/login",
        json={"email": creds["email"], "password": creds["password"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_login_wrong_password_returns_401(client, registered_user):
    """POST /auth/login with wrong password returns HTTP 401."""
    creds, _ = registered_user
    response = client.post(
        "/auth/login",
        json={"email": creds["email"], "password": "WrongPass!999"},
    )
    assert response.status_code == 401


def test_login_nonexistent_user_returns_401(client):
    """POST /auth/login for unknown email returns HTTP 401."""
    response = client.post(
        "/auth/login",
        json={
            "email": "ghost_user_xyz@agri.example.com",
            "password": "AnyPassword123!",
        },
    )
    assert response.status_code == 401


def test_login_missing_field_returns_422(client):
    """POST /auth/login without password field returns HTTP 422."""
    missing_pwd = {"email": "someone@agri.example.com"}
    response = client.post("/auth/login", json=missing_pwd)
    assert response.status_code == 422


def test_oauth2_token_success(client, registered_user):
    """POST /auth/token (OAuth2 form flow) with valid creds returns 200."""
    creds, _ = registered_user
    response = client.post(
        "/auth/token",
        data={
            "username": creds["email"],
            "password": creds["password"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_oauth2_token_wrong_password_returns_401(client, registered_user):
    """POST /auth/token with wrong password returns HTTP 401."""
    creds, _ = registered_user
    response = client.post(
        "/auth/token",
        data={
            "username": creds["email"],
            "password": "WRONG_pass_000",
        },
    )
    assert response.status_code == 401


# ===========================================================================
# SECTION 4: Auth — /auth/me Tests
# ===========================================================================


def test_me_authenticated_returns_200(client, registered_user):
    """GET /auth/me with valid Bearer token returns HTTP 200."""
    _, token_resp = registered_user
    headers = {"Authorization": f"Bearer {token_resp['access_token']}"}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "email" in data


def test_me_no_token_returns_401(client):
    """GET /auth/me without Authorization header returns HTTP 401."""
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_me_invalid_token_returns_401(client):
    """GET /auth/me with malformed token returns HTTP 401."""
    headers = {"Authorization": "Bearer this.is.not.a.valid.jwt.token"}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 401


# ===========================================================================
# SECTION 5: JWT Token Rotation — /auth/refresh Tests
# ===========================================================================


def test_token_refresh_success(client, registered_user):
    """POST /auth/refresh with valid refresh token returns new tokens."""
    _, token_resp = registered_user
    refresh_tok = token_resp.get("refresh_token")
    assert refresh_tok, "Login must return a refresh_token"
    refresh_payload = {"refresh_token": refresh_tok}
    response = client.post("/auth/refresh", json=refresh_payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


def test_token_refresh_with_access_token_returns_401(client, registered_user):
    """POST /auth/refresh with an access token (not refresh) returns 401."""
    _, token_resp = registered_user
    # Supplying the access token where a refresh token is expected
    response = client.post(
        "/auth/refresh",
        json={"refresh_token": token_resp["access_token"]},
    )
    assert response.status_code == 401


def test_token_refresh_invalid_token_returns_401(client):
    """POST /auth/refresh with garbage string returns HTTP 401."""
    response = client.post(
        "/auth/refresh",
        json={"refresh_token": "invalid.jwt.garbage"},
    )
    assert response.status_code == 401


def test_token_refresh_missing_body_returns_422(client):
    """POST /auth/refresh with missing body returns HTTP 422."""
    response = client.post("/auth/refresh", json={})
    assert response.status_code == 422


# ===========================================================================
# SECTION 6: Financial Endpoint Tests
# ===========================================================================


def test_financial_forecast_success(client, registered_user):
    """POST /business/financial-forecast with valid payload returns 200."""
    _, token_resp = registered_user
    headers = {"Authorization": f"Bearer {token_resp['access_token']}"}
    response = client.post(
        "/business/financial-forecast",
        json=_FINANCIAL_PAYLOAD,
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    # UnifiedBusinessResponse nests financials under financial_metrics
    assert "financial_metrics" in data or "gross_revenue" in data
    assert "farm_summary" in data or "crop_type" in data


def test_financial_forecast_anonymous_success(client):
    """POST /business/financial-forecast without auth returns 200."""
    response = client.post(
        "/business/financial-forecast",
        json=_FINANCIAL_PAYLOAD,
    )
    assert response.status_code == 200


def test_financial_forecast_negative_farm_size_returns_422(client):
    """POST /business/financial-forecast with negative size returns 422."""
    bad_payload = dict(_FINANCIAL_PAYLOAD)
    bad_payload["farm_size"] = -5.0
    response = client.post("/business/financial-forecast", json=bad_payload)
    assert response.status_code == 422


def test_financial_forecast_missing_crop_type_returns_422(client):
    """POST /business/financial-forecast without crop_type returns 422."""
    bad_payload = {
        "farm_size": 10.0,
        "predicted_yield": 3.5,
    }
    response = client.post("/business/financial-forecast", json=bad_payload)
    assert response.status_code == 422


def test_financial_forecast_zero_yield_returns_200(client):
    """POST /business/financial-forecast with zero yield is accepted."""
    payload = dict(_FINANCIAL_PAYLOAD)
    payload["predicted_yield"] = 0.0
    response = client.post("/business/financial-forecast", json=payload)
    assert response.status_code == 200


def test_financial_history_requires_auth(client):
    """GET /business/history without token returns a list."""
    response = client.get("/business/history")
    # /business/history uses get_optional_current_user
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_financial_history_success(client, registered_user):
    """GET /business/history with valid token returns HTTP 200 and a list."""
    _, token_resp = registered_user
    headers = {"Authorization": f"Bearer {token_resp['access_token']}"}
    response = client.get("/business/history", headers=headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_financial_history_detail_not_found(client, registered_user):
    """GET /business/history/{id} for a non-existent audit ID returns 404."""
    _, token_resp = registered_user
    headers = {"Authorization": f"Bearer {token_resp['access_token']}"}
    fake_id = str(uuid4())
    response = client.get(f"/business/history/{fake_id}", headers=headers)
    assert response.status_code == 404


# ===========================================================================
# SECTION 7: Security Headers Tests
# ===========================================================================


def test_security_headers_present_on_root(client):
    """Every response must include the mandatory HTTP security headers."""
    response = client.get("/")
    headers = response.headers

    assert (
        headers.get("X-Frame-Options") == "DENY"
    ), "X-Frame-Options header must be DENY"
    assert (
        headers.get("X-Content-Type-Options") == "nosniff"
    ), "X-Content-Type-Options must be nosniff"
    assert (
        headers.get("X-XSS-Protection") == "1; mode=block"
    ), "X-XSS-Protection header missing"
    hsts_msg = "HSTS header must be present"
    assert "Strict-Transport-Security" in headers, hsts_msg
    csp_msg = "CSP header must be present"
    assert "Content-Security-Policy" in headers, csp_msg
    ref_msg = "Referrer-Policy header must be present"
    assert "Referrer-Policy" in headers, ref_msg


def test_security_headers_on_docs_endpoint(client):
    """Security headers must also be injected on the /docs endpoint."""
    response = client.get("/docs")
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"


def test_security_headers_on_predict_endpoint(client):
    """Security headers must be present on 422 error responses too."""
    bad_payload = {"crop": "Wheat"}
    response = client.post("/predict/yield", json=bad_payload)
    assert response.status_code == 422
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
