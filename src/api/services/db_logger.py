"""
Database Logging Handler for Axlero Agri-Tech API.
Integrates SQLAlchemy persistence to store incoming soil assessment inputs
and model prediction results in PostgreSQL / relational database tables.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID
from sqlalchemy.orm import Session

from src.api.models.soil_log import SoilLog
from src.api.models.user import User
from src.api.models.yield_prediction import YieldPrediction

logger = logging.getLogger("axlero_agri_api.db_logger")


def log_assessment_and_prediction(
    db: Session,
    crop: str,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    predicted_yield: float,
    baseline_yield: float,
    potential_gain: float,
    confidence_score: float,
    model_version: str,
    recommendations: List[Dict[str, Any]],
    shap_values: Dict[str, float],
    input_features: Dict[str, Any],
    user_id: Optional[UUID] = None,
    field_name: Optional[str] = None,
    district: Optional[str] = None,
    season: Optional[str] = None,
    organic_matter: Optional[float] = None,
    moisture: Optional[float] = None,
    notes: Optional[str] = None,
) -> tuple[SoilLog, YieldPrediction]:
    """
    Persists incoming soil assessment parameters and resulting yield predictions
    into the database in a single atomic transaction.

    Returns:
        tuple[SoilLog, YieldPrediction]: The refreshed SQLAlchemy model instances.
    """
    valid_user_id: Optional[UUID] = None
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            valid_user_id = user.id
        else:
            logger.warning(
                f"Provided user_id {user_id} not found in database; logging with anonymous farmer context."
            )

    # 1. Store incoming assessment inputs in SoilLog table
    resolved_field_name = field_name or f"Field-{crop}-Assessment"
    resolved_district = district or "Standard-District"

    soil_log = SoilLog(
        user_id=valid_user_id,
        field_name=resolved_field_name,
        district=resolved_district,
        nitrogen=float(nitrogen),
        phosphorus=float(phosphorus),
        potassium=float(potassium),
        ph=float(ph),
        organic_matter=float(organic_matter) if organic_matter is not None else None,
        moisture=float(moisture) if moisture is not None else None,
        notes=notes or f"Automated soil test ingestion for {crop} yield forecasting",
    )
    db.add(soil_log)
    db.flush()  # Populates soil_log.id for foreign key linkage

    # 2. Store model prediction results in YieldPrediction table
    yield_record = YieldPrediction(
        user_id=valid_user_id,
        soil_log_id=soil_log.id,
        crop=crop,
        season=season or "Current Season",
        model_version=model_version,
        predicted_yield=float(predicted_yield),
        baseline_yield=float(baseline_yield),
        potential_gain=float(potential_gain),
        confidence_score=float(confidence_score),
        recommendations=recommendations,
        shap_values=shap_values,
        input_features=input_features,
    )
    db.add(yield_record)

    db.commit()
    db.refresh(soil_log)
    db.refresh(yield_record)

    logger.info(
        f"[DB Logging] Successfully persisted SoilLog (ID={soil_log.id}) and "
        f"YieldPrediction (ID={yield_record.id}, Crop={crop}, Yield={predicted_yield:.2f} MT/Ha)"
    )

    return soil_log, yield_record


def log_soil_assessment(
    db: Session,
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    user_id: Optional[UUID] = None,
    field_name: Optional[str] = None,
    district: Optional[str] = None,
    notes: Optional[str] = None,
) -> SoilLog:
    """
    Stores an incoming soil test payload in the database.
    """
    valid_user_id: Optional[UUID] = None
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            valid_user_id = user.id

    soil_log = SoilLog(
        user_id=valid_user_id,
        field_name=field_name or "Soil Recommendation Assessment",
        district=district or "Standard-District",
        nitrogen=float(nitrogen),
        phosphorus=float(phosphorus),
        potassium=float(potassium),
        ph=float(ph),
        notes=notes or "Automated soil chemistry log for crop recommendation",
    )
    db.add(soil_log)
    db.commit()
    db.refresh(soil_log)

    logger.info(f"[DB Logging] Logged standalone SoilLog ID={soil_log.id}")
    return soil_log
