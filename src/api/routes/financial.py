"""
FastAPI Routes for Financial Calculator, Agribusiness Analytics, and Historical Audit Sessions.
Deliverable: src/api/routes/financial.py (Engineer 3)

Endpoints:
- POST /business/financial-forecast: Ingests farm size, crop type, and predicted yield;
  computes ROI, gross revenue, cost breakdowns, and integrates SHAP diagnostic feedback.
- GET  /business/history: Lists paginated historical assessment logs.
- GET  /business/history/{audit_id}: Retrieves specific historical audit report.
- DELETE /business/history/{audit_id}: Deletes historical audit record.
"""

from datetime import datetime
import logging
from typing import List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from src.api.auth import get_optional_current_user
from src.api.database import get_db
from src.api.models.financial_log import FinancialAuditLog
from src.api.models.user import User
from src.api.schemas.financial import (
    FinancialAuditLogRead,
    FinancialForecastRequest,
    UnifiedBusinessResponse,
)
from src.api.services.financial_service import execute_financial_forecast

logger = logging.getLogger("axlero_agri_api.routes.financial")

router = APIRouter(prefix="/business", tags=["Financial Calculator & Agribusiness Analytics"])


@router.post(
    "/financial-forecast",
    response_model=UnifiedBusinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute farm financial forecast, net profit margin, and SHAP diagnostic feedback",
)
def compute_financial_forecast(
    payload: FinancialForecastRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Ingests:
    - **crop_type**: Target crop (e.g. 'Wheat', 'Maize', 'Rice', 'Cotton')
    - **farm_size**: Cultivated acreage or hectarage
    - **predicted_yield**: Forecasted yield in Metric Tons per Hectare
    - **custom pricing / costs** (optional)
    - **soil / weather parameters** (optional, for real-time SHAP factor attribution)

    Computes:
    - Gross Revenue, Itemized Operational Costs, Net Profit, and ROI percentage
    - Break-even yield and break-even commodity price
    - Local SHAP factor attributions (controllable vs. exogenous) and prescriptive ROI amendments

    Persists assessment in PostgreSQL database tables for historical audit tracking.
    """
    audit_id = uuid.uuid4()
    logger.info(
        f"Processing financial forecast request for Crop='{payload.crop_type}', "
        f"FarmSize={payload.farm_size} ({payload.farm_size_unit}), Yield={payload.predicted_yield} MT/ha"
    )

    # 1. Resolve Effective User Identity
    effective_user_id = None
    if current_user:
        effective_user_id = current_user.id
    elif payload.user_id:
        effective_user_id = payload.user_id

    # 2. Execute Financial Modeling & SHAP Diagnostics
    response = execute_financial_forecast(request=payload, audit_id=audit_id)
    if effective_user_id:
        response.user_id = effective_user_id

    # 3. Session History DB Persistence: Save to financial_audit_logs table
    try:
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")

        db_audit_log = FinancialAuditLog(
            id=audit_id,
            user_id=effective_user_id,
            soil_log_id=payload.soil_log_id,
            yield_prediction_id=payload.yield_prediction_id,
            crop_type=response.farm_summary.crop_type,
            farm_size_ha=response.farm_summary.farm_size_ha,
            predicted_yield=response.farm_summary.predicted_yield_mt_ha,
            total_production_mt=response.farm_summary.total_production_mt,
            currency=response.financial_metrics.currency,
            crop_price_per_mt=response.financial_metrics.crop_price_per_mt,
            gross_revenue=response.financial_metrics.gross_revenue,
            total_operational_costs=response.financial_metrics.total_operational_costs,
            net_profit=response.financial_metrics.net_profit,
            roi_percentage=response.financial_metrics.roi_percentage,
            profit_margin_percentage=response.financial_metrics.profit_margin_percentage,
            break_even_yield=response.financial_metrics.break_even_yield_mt_ha,
            break_even_price=response.financial_metrics.break_even_price_per_mt,
            cost_breakdown=response.financial_metrics.cost_breakdown.model_dump(),
            shap_diagnostic_feedback=response.shap_diagnostic_feedback.model_dump(),
            prescriptive_amendments=[
                a.model_dump() for a in response.shap_diagnostic_feedback.prescriptive_amendments
            ],
            assessment_notes=payload.notes,
            audit_metadata={
                "client_ip": client_host,
                "user_agent": user_agent,
                "district": payload.district,
                "field_name": payload.field_name,
            },
        )
        db.add(db_audit_log)
        db.commit()
        db.refresh(db_audit_log)
        response.persisted = True
        logger.info(f"Successfully persisted financial audit log {audit_id} in database.")
    except Exception as exc:
        db.rollback()
        logger.error(f"Failed to persist financial audit log to database: {exc}")
        response.persisted = False

    return response


@router.get(
    "/history",
    response_model=List[FinancialAuditLogRead],
    summary="List Historical Financial Calculation Logs",
)
def list_financial_history(
    crop_type: Optional[str] = Query(None, description="Filter logs by crop name"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter logs by user ID"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=200, description="Pagination limit"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Retrieves historical financial audit logs and user farm assessments.
    Supports filtering by crop type, authenticated user, and pagination.
    """
    query = db.query(FinancialAuditLog)

    # If authenticated non-superuser, restrict to own records unless explicitly searching
    if current_user and not current_user.is_superuser:
        query = query.filter(FinancialAuditLog.user_id == current_user.id)
    elif user_id:
        query = query.filter(FinancialAuditLog.user_id == user_id)

    if crop_type:
        query = query.filter(FinancialAuditLog.crop_type.ilike(f"%{crop_type}%"))

    logs = query.order_by(FinancialAuditLog.created_at.desc()).offset(skip).limit(limit).all()
    return logs


@router.get(
    "/history/{audit_id}",
    response_model=FinancialAuditLogRead,
    summary="Retrieve Specific Financial Assessment Details",
)
def get_financial_audit_log(
    audit_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Fetches the detailed record of an individual historical financial calculation.
    """
    record = db.query(FinancialAuditLog).filter(FinancialAuditLog.id == audit_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Financial audit log with ID '{audit_id}' was not found.",
        )

    # Security check: If authenticated non-superuser, verify ownership
    if current_user and not current_user.is_superuser:
        if record.user_id and record.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: You do not own this financial assessment report.",
            )

    return record


@router.delete(
    "/history/{audit_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Historical Financial Assessment Record",
)
def delete_financial_audit_log(
    audit_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Deletes a historical financial audit record.
    """
    record = db.query(FinancialAuditLog).filter(FinancialAuditLog.id == audit_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Financial audit log with ID '{audit_id}' was not found.",
        )

    if current_user and not current_user.is_superuser:
        if record.user_id and record.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: You cannot delete another user's assessment.",
            )

    db.delete(record)
    db.commit()
    return None

