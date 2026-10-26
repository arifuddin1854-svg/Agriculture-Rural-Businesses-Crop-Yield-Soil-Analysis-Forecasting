"""
Soil Recommendation Route.
Handles POST /recommend/crop evaluating incoming soil parameters against optimal crop lookup tables.
"""

from datetime import datetime
import logging
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.schemas.soil import (
    CropRecommendationRequest,
    CropRecommendationResponse,
    CropRecommendationItem,
)
from src.api.services.crop_recommender import recommend_top_crops
from src.api.services.db_logger import log_soil_assessment

logger = logging.getLogger("axlero_agri_api.routes.recommend")

router = APIRouter(tags=["Soil & Crop Recommendation"])


@router.post(
    "/recommend/crop",
    response_model=CropRecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend optimal target crops based on soil chemistry",
)
def recommend_crops(
    request: CropRecommendationRequest,
    db: Session = Depends(get_db),
):
    """
    Evaluates soil Nitrogen, Phosphorus, Potassium, and pH against optimal crop
    lookup tables to identify and rank the most suitable target crops.

    Optionally logs the incoming assessment parameters into the PostgreSQL/relational database.
    """
    logger.info(
        f"Evaluating crop recommendation for N={request.nitrogen}, P={request.phosphorus}, "
        f"K={request.potassium}, pH={request.ph}"
    )

    # 1. Run Agronomic Recommendation Engine
    ranked_candidates = recommend_top_crops(
        nitrogen=request.nitrogen,
        phosphorus=request.phosphorus,
        potassium=request.potassium,
        ph=request.ph,
        rainfall=request.rainfall,
        temperature=request.temperature,
        top_k=request.top_k,
    )

    # 2. Database Logging: Store incoming soil assessment record
    soil_log = log_soil_assessment(
        db=db,
        nitrogen=request.nitrogen,
        phosphorus=request.phosphorus,
        potassium=request.potassium,
        ph=request.ph,
        user_id=request.user_id,
        district=request.district,
        field_name=request.field_name,
        notes=f"Crop recommendation query: top candidate={ranked_candidates[0]['crop'] if ranked_candidates else 'None'}",
    )

    # 3. Format response contract
    items = [CropRecommendationItem(**item) for item in ranked_candidates]
    top_crop = items[0].crop if items else "Wheat"

    return CropRecommendationResponse(
        top_crop=top_crop,
        recommendations=items,
        analyzed_inputs={
            "nitrogen": request.nitrogen,
            "phosphorus": request.phosphorus,
            "potassium": request.potassium,
            "ph": request.ph,
            "rainfall": request.rainfall or 0.0,
            "temperature": request.temperature or 0.0,
        },
        soil_log_id=soil_log.id,
        timestamp=datetime.utcnow(),
    )
