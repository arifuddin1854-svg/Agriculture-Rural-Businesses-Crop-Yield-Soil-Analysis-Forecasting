import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models.soil_log import SoilLog
from src.api.models.user import User
from src.api.schemas.soil_log import SoilLogCreate, SoilLogRead

router = APIRouter(prefix="/soil-logs", tags=["Soil Logs"])


@router.post("/", response_model=SoilLogRead, status_code=status.HTTP_201_CREATED)
def create_soil_log(soil_in: SoilLogCreate, db: Session = Depends(get_db)):
    """Record a new soil chemistry analysis log."""
    user = db.query(User).filter(User.id == soil_in.user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    soil_log = SoilLog(**soil_in.model_dump())
    db.add(soil_log)
    db.commit()
    db.refresh(soil_log)
    return soil_log


@router.get("/", response_model=List[SoilLogRead])
def list_soil_logs(
    district: Optional[str] = Query(None, description="Filter logs by geographic district"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter logs by user ID"),
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Query soil sample logs with optional district or user filters."""
    query = db.query(SoilLog)
    if district:
        query = query.filter(SoilLog.district == district)
    if user_id:
        query = query.filter(SoilLog.user_id == user_id)
    return query.order_by(SoilLog.sampled_at.desc()).offset(skip).limit(limit).all()


@router.get("/{log_id}", response_model=SoilLogRead)
def get_soil_log(log_id: uuid.UUID, db: Session = Depends(get_db)):
    """Retrieve details for a specific soil log."""
    log = db.query(SoilLog).filter(SoilLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Soil log not found")
    return log
