import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models.soil_log import SoilLog
from src.api.models.user import User
from src.api.models.yield_prediction import YieldPrediction
from src.api.schemas.yield_prediction import YieldPredictionCreate, YieldPredictionRead

router = APIRouter(prefix="/predictions", tags=["Yield Predictions"])


@router.post("/", response_model=YieldPredictionRead, status_code=status.HTTP_201_CREATED)
def record_prediction(pred_in: YieldPredictionCreate, db: Session = Depends(get_db)):
    """Save a model inference yield prediction and advisory prescription."""
    user = db.query(User).filter(User.id == pred_in.user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if pred_in.soil_log_id:
        soil_log = db.query(SoilLog).filter(SoilLog.id == pred_in.soil_log_id).first()
        if not soil_log:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Referenced Soil Log not found")

    prediction = YieldPrediction(**pred_in.model_dump())
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


@router.get("/", response_model=List[YieldPredictionRead])
def list_predictions(
    crop: Optional[str] = Query(None, description="Filter predictions by crop type"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter predictions by user ID"),
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Retrieve historical yield predictions and recommendations."""
    query = db.query(YieldPrediction)
    if crop:
        query = query.filter(YieldPrediction.crop.ilike(f"%{crop}%"))
    if user_id:
        query = query.filter(YieldPrediction.user_id == user_id)
    return query.order_by(YieldPrediction.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/{pred_id}", response_model=YieldPredictionRead)
def get_prediction(pred_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieve a single prediction by ID."""
    pred = db.query(YieldPrediction).filter(YieldPrediction.id == pred_id).first()
    if not pred:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Yield prediction not found")
    return pred
