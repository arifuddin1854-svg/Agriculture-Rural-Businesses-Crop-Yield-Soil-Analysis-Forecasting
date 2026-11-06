"""
Comprehensive Test Suite for Engineer 3 Deliverables:
1. JWT Authentication Module (OAuth2, password hashing, token issuance, /auth endpoints)
2. Financial Forecast REST API (POST /business/financial-forecast)
3. ROI Calculation Engine (Gross Revenue, Operational Costs, Net Profit, ROI %, Break-even metrics)
4. SHAP Feedback API Integration (Unified Business Response JSON, Controllable vs Exogenous, Monetary impact)
5. Session History DB Persistence (Financial Audit Logs schema and /business/history endpoints)
"""

import uuid
import pytest
from src.api.auth import get_password_hash, verify_password, create_access_token, decode_access_token
from src.api.models.financial_log import FinancialAuditLog
from src.api.models.user import User
from src.financial_calculator import (
    calculate_roi_and_financials,
    normalize_farm_size,
    build_shap_diagnostic_feedback,
)


# ===========================================================================
# 1. JWT & Password Hashing Unit Tests
# ===========================================================================

def test_password_hashing_and_verification():
    raw_password = "FarmerSecretPassword@2026"
    hashed = get_password_hash(raw_password)

    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password(raw_password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False
    assert verify_password("", hashed) is False

    # Test backward-compatible legacy placeholder
    legacy_hash = "pbkdf2_sha256$legacy_plain_pass"
    assert verify_password("legacy_plain_pass", legacy_hash) is True
    assert verify_password("wrong", legacy_hash) is False


def test_jwt_token_creation_and_decoding():
    user_id = str(uuid.uuid4())
    token = create_access_token(
        data={"sub": user_id, "user_id": user_id, "email": "test@agri.com", "role": "farmer"}
    )
    assert isinstance(token, str)

    payload = decode_access_token(token)
    assert payload["sub"] == user_id
    assert payload["email"] == "test@agri.com"
    assert payload["role"] == "farmer"


# ===========================================================================
# 2. Authentication Route Tests (/auth/register, /auth/login, /auth/me)
# ===========================================================================

def test_user_registration_and_jwt_issuance(client, db_session):
    payload = {
        "email": "new.farmer@axlero.agri",
        "password": "SecurePassword123!",
        "full_name": "Ramesh Kumar",
        "role": "farmer",
        "farm_name": "Kumar Organic Fields",
        "district": "Ludhiana",
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == "new.farmer@axlero.agri"
    assert data["user"]["full_name"] == "Ramesh Kumar"

    # Verify user saved in DB
    db_user = db_session.query(User).filter(User.email == "new.farmer@axlero.agri").first()
    assert db_user is not None
    assert verify_password("SecurePassword123!", db_user.hashed_password) is True


def test_auth_login_json_and_token_flow(client, db_session):
    # Register user first
    email = "login.tester@axlero.agri"
    password = "LoginTestPwd#2026"
    client.post("/api/v1/auth/register", json={"email": email, "password": password, "full_name": "Test User"})

    # JSON login
    login_resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # Access protected route /auth/me
    headers = {"Authorization": f"Bearer {token}"}
    me_resp = client.get("/api/v1/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email

    # Access /auth/me without token -> 401
    unauth_resp = client.get("/api/v1/auth/me")
    assert unauth_resp.status_code == 401


def test_oauth2_form_token_endpoint(client, db_session):
    email = "oauth.tester@axlero.agri"
    password = "OAuthPassword456!"
    client.post("/api/v1/auth/register", json={"email": email, "password": password})

    # OAuth2 Password Form flow (form-encoded)
    form_data = {"username": email, "password": password}
    token_resp = client.post("/api/v1/auth/token", data=form_data)
    assert token_resp.status_code == 200
    data = token_resp.json()
    assert "access_token" in data


# ===========================================================================
# 3. ROI Calculation Engine Logic Unit Tests
# ===========================================================================

def test_roi_calculation_engine_math():
    farm_size_ha = 10.0
    predicted_yield = 4.5  # MT/ha -> 45 MT total
    metrics = calculate_roi_and_financials(
        crop_type="Wheat",
        farm_size_ha=farm_size_ha,
        predicted_yield_mt_ha=predicted_yield,
        currency="INR",
    )

    # 45 MT * 22,750 INR = 1,023,750 INR Gross Revenue
    expected_gross = 45.0 * 22750.0
    assert metrics.gross_revenue == expected_gross
    assert metrics.crop_price_per_mt == 22750.0

    # Operational costs check
    assert metrics.total_operational_costs > 0
    assert metrics.net_profit == metrics.gross_revenue - metrics.total_operational_costs
    expected_roi = (metrics.net_profit / metrics.total_operational_costs) * 100.0
    assert abs(metrics.roi_percentage - round(expected_roi, 2)) < 0.05
    assert metrics.break_even_yield_mt_ha > 0
    assert metrics.break_even_price_per_mt > 0
    assert len(metrics.cost_breakdown.itemized) > 0


def test_farm_size_normalization():
    # 10 acres = 4.04686 hectares
    ha = normalize_farm_size(10.0, "acres")
    assert round(ha, 2) == 4.05

    ha_direct = normalize_farm_size(5.0, "hectares")
    assert ha_direct == 5.0


# ===========================================================================
# 4. Financial Endpoint & Unified Business Response JSON Tests
# ===========================================================================

def test_post_financial_forecast_direct_endpoint(client, db_session):
    """
    Test direct POST /business/financial-forecast endpoint:
    - Ingests farm size, crop type, predicted yield
    - Calculates ROI, net profit, cost breakdown
    - Integrates SHAP explainability diagnostics
    - Persists audit record to PostgreSQL database
    """
    payload = {
        "crop_type": "Wheat",
        "farm_size": 12.0,
        "farm_size_unit": "hectares",
        "predicted_yield": 4.80,
        "ph": 5.4,  # Acidic: should trigger SHAP limitation and lime amendment
        "nitrogen": 75.0,
        "phosphorus": 35.0,
        "potassium": 80.0,
        "district": "Patiala",
        "notes": "Pre-Rabi season financial appraisal",
    }

    response = client.post("/business/financial-forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    # 1. Structure Assertions
    assert data["status"] == "success"
    assert data["audit_id"] is not None
    assert data["persisted"] is True

    # 2. Farm Summary
    farm_summary = data["farm_summary"]
    assert farm_summary["crop_type"] == "Wheat"
    assert farm_summary["farm_size_ha"] == 12.0
    assert farm_summary["predicted_yield_mt_ha"] == 4.80
    assert abs(farm_summary["total_production_mt"] - (12.0 * 4.80)) < 1e-3

    # 3. Financial Metrics
    fin = data["financial_metrics"]
    assert fin["currency"] == "INR"
    assert fin["gross_revenue"] > 0
    assert fin["total_operational_costs"] > 0
    assert fin["net_profit"] == round(fin["gross_revenue"] - fin["total_operational_costs"], 2)
    assert fin["roi_percentage"] > 0
    assert fin["break_even_yield_mt_ha"] > 0
    assert fin["break_even_price_per_mt"] > 0

    # Cost breakdown
    cb = fin["cost_breakdown"]
    assert cb["seeds"] > 0
    assert cb["fertilizers"] > 0
    assert cb["machinery_and_labor"] > 0
    assert len(cb["itemized"]) >= 5

    # 4. SHAP Diagnostic Feedback
    shap = data["shap_diagnostic_feedback"]
    assert "summary" in shap
    assert len(shap["feature_attributions"]) > 0
    assert any("Controllable" in f["domain"] for f in shap["feature_attributions"])
    assert any("Exogenous" in f["domain"] for f in shap["feature_attributions"])
    assert any("financial_impact" in f for f in shap["feature_attributions"])

    # Prescriptive amendments
    assert len(shap["prescriptive_amendments"]) > 0
    for amend in shap["prescriptive_amendments"]:
        assert amend["estimated_treatment_cost_total"] > 0
        assert amend["projected_revenue_gain_total"] >= 0
        assert "amendment_roi_percentage" in amend

    # 5. Database Persistence Verification
    audit_uuid = uuid.UUID(data["audit_id"])
    db_record = db_session.query(FinancialAuditLog).filter(FinancialAuditLog.id == audit_uuid).first()
    assert db_record is not None
    assert db_record.crop_type == "Wheat"
    assert db_record.farm_size_ha == 12.0
    assert db_record.gross_revenue == fin["gross_revenue"]
    assert db_record.net_profit == fin["net_profit"]


def test_post_financial_forecast_versioned_with_auth(client, db_session):
    """
    Test POST /api/v1/business/financial-forecast with JWT authentication and custom price overrides.
    """
    # Create user & get token
    reg_resp = client.post("/api/v1/auth/register", json={
        "email": "auth.farmer@axlero.agri",
        "password": "Password789!",
        "full_name": "Sukhwinder Kaur",
    })
    token = reg_resp.json()["access_token"]
    user_id = reg_resp.json()["user"]["id"]

    payload = {
        "crop_type": "Cotton",
        "farm_size": 25.0,
        "farm_size_unit": "acres",  # Tests acres conversion
        "predicted_yield": 2.85,
        "custom_crop_price_per_mt": 75000.0,  # Custom premium price
        "custom_operational_cost_per_ha": 50000.0,
    }

    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/v1/business/financial-forecast", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == user_id
    assert data["financial_metrics"]["crop_price_per_mt"] == 75000.0
    # Farm size in hectares should be normalized: 25 * 0.404686 = 10.1172 ha
    assert abs(data["farm_summary"]["farm_size_ha"] - 10.1172) < 0.01

    # Verify user relationship in DB
    audit_uuid = uuid.UUID(data["audit_id"])
    db_record = db_session.query(FinancialAuditLog).filter(FinancialAuditLog.id == audit_uuid).first()
    assert db_record.user_id == uuid.UUID(user_id)


# ===========================================================================
# 5. Session History DB Persistence & History Query Endpoints
# ===========================================================================

def test_financial_history_query_and_delete(client, db_session):
    """
    Test GET /api/v1/business/history and DELETE /api/v1/business/history/{id}.
    """
    # Create an assessment
    payload = {
        "crop_type": "Maize",
        "farm_size": 8.0,
        "farm_size_unit": "hectares",
        "predicted_yield": 6.2,
    }
    resp = client.post("/business/financial-forecast", json=payload)
    audit_id = resp.json()["audit_id"]

    # Query history list
    history_resp = client.get("/api/v1/business/history?crop_type=Maize")
    assert history_resp.status_code == 200
    items = history_resp.json()
    assert isinstance(items, list)
    matching = [item for item in items if item["id"] == audit_id]
    assert len(matching) == 1
    assert matching[0]["crop_type"] == "Maize"

    # Query individual history detail
    detail_resp = client.get(f"/api/v1/business/history/{audit_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["id"] == audit_id

    # Delete history record
    del_resp = client.delete(f"/api/v1/business/history/{audit_id}")
    assert del_resp.status_code == 204

    # Confirm deletion
    get_after_del = client.get(f"/api/v1/business/history/{audit_id}")
    assert get_after_del.status_code == 404


# ===========================================================================
# 6. Input Validation & Error Handling Tests
# ===========================================================================

def test_financial_forecast_validation_errors(client):
    # Negative farm size
    bad_payload_1 = {
        "crop_type": "Wheat",
        "farm_size": -5.0,
        "predicted_yield": 4.5,
    }
    resp1 = client.post("/business/financial-forecast", json=bad_payload_1)
    assert resp1.status_code == 422

    # Negative yield
    bad_payload_2 = {
        "crop_type": "Wheat",
        "farm_size": 5.0,
        "predicted_yield": -1.0,
    }
    resp2 = client.post("/business/financial-forecast", json=bad_payload_2)
    assert resp2.status_code == 422

    # Invalid farm size unit
    bad_payload_3 = {
        "crop_type": "Wheat",
        "farm_size": 5.0,
        "farm_size_unit": "lightyears",
        "predicted_yield": 4.5,
    }
    resp3 = client.post("/business/financial-forecast", json=bad_payload_3)
    assert resp3.status_code == 422
