"""
FastAPI Inference Route for Crop Yield Forecasting.
Endpoint: POST /predict/yield
Ingests real-time soil parameter payloads and responds with yield estimates,
SHAP feature attributions, and prescriptive soil amendments.
"""

from datetime import datetime
import logging
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.schemas.soil import (
    YieldPredictionRequest,
    YieldPredictionResponse,
    RecommendationDetail,
)
from src.api.services.inference import predict_yield
from src.api.services.db_logger import log_assessment_and_prediction

logger = logging.getLogger("axlero_agri_api.routes.predict")

router = APIRouter(tags=["Yield Prediction & Inference"])


@router.post(
    "/predict/yield",
    response_model=YieldPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest real-time soil parameters and estimate crop harvest yield",
)
def predict_crop_yield(
    request: YieldPredictionRequest,
    db: Session = Depends(get_db),
):
    """
    Ingests real-time soil test parameters (NPK, pH, moisture, rainfall, temperature)
    and produces:
    - End-of-season yield forecasts (Metric Tons / Hectare)
    - Actionable agronomic amendment recommendations
    - Potential yield gains
    - Local SHAP factor attributions

    Stores incoming assessment inputs and prediction results in the database.
    """
    logger.info(
        f"Yield inference requested for Crop='{request.crop}', N={request.nitrogen}, "
        f"P={request.phosphorus}, K={request.potassium}, pH={request.ph}"
    )

    # 1. Execute Real-Time Model Inference & Advisory Diagnostics
    inference_result = predict_yield(
        crop=request.crop,
        nitrogen=request.nitrogen,
        phosphorus=request.phosphorus,
        potassium=request.potassium,
        ph=request.ph,
        rainfall=request.rainfall or 550.0,
        temp_max=request.temp_max or 29.5,
    )

    # 2. Database Logging: Store incoming assessment inputs and prediction results
    soil_log, yield_record = log_assessment_and_prediction(
        db=db,
        crop=request.crop,
        nitrogen=request.nitrogen,
        phosphorus=request.phosphorus,
        potassium=request.potassium,
        ph=request.ph,
        predicted_yield=inference_result["predicted_yield"],
        baseline_yield=inference_result["baseline_yield"],
        potential_gain=inference_result["potential_gain"],
        confidence_score=inference_result["confidence_score"],
        model_version=inference_result["model_version"],
        recommendations=inference_result["recommendations"],
        shap_values=inference_result["shap_values"],
        input_features=inference_result["input_features"],
        user_id=request.user_id,
        field_name=request.field_name,
        district=request.district,
        season=request.season,
    )

    # 3. Construct validated response contract
    rec_items = [RecommendationDetail(**r) for r in inference_result["recommendations"]]

    return YieldPredictionResponse(
        crop=inference_result["crop"],
        predicted_yield=inference_result["predicted_yield"],
        baseline_yield=inference_result["baseline_yield"],
        potential_gain=inference_result["potential_gain"],
        confidence_score=inference_result["confidence_score"],
        model_version=inference_result["model_version"],
        recommendations=rec_items,
        shap_values=inference_result["shap_values"],
        input_features=inference_result["input_features"],
        prediction_id=yield_record.id,
        soil_log_id=soil_log.id,
        created_at=yield_record.created_at or datetime.utcnow(),
    )
