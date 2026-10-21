from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class SoilLogBase(BaseModel):
    field_name: str = Field(..., max_length=150, description="Field or parcel identifier")
    district: str = Field(..., max_length=100, description="District/County for spatial grouping")
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    nitrogen: float = Field(..., ge=0, description="Soil Nitrogen (N) in kg/ha")
    phosphorus: float = Field(..., ge=0, description="Soil Phosphorus (P) in kg/ha")
    potassium: float = Field(..., ge=0, description="Soil Potassium (K) in kg/ha")
    ph: float = Field(..., ge=0.0, le=14.0, description="Soil pH level (0-14)")
    organic_matter: Optional[float] = Field(None, ge=0.0, le=100.0, description="Soil organic matter %")
    moisture: Optional[float] = Field(None, ge=0.0, le=100.0, description="Soil moisture %")
    soil_type: Optional[str] = Field(None, max_length=50, description="Soil taxonomy classification")
    sampled_at: Optional[datetime] = None
    notes: Optional[str] = None


class SoilLogCreate(SoilLogBase):
    user_id: UUID


class SoilLogUpdate(BaseModel):
    field_name: Optional[str] = None
    district: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    ph: Optional[float] = None
    organic_matter: Optional[float] = None
    moisture: Optional[float] = None
    soil_type: Optional[str] = None
    notes: Optional[str] = None


class SoilLogRead(SoilLogBase):
    id: UUID
    user_id: UUID
    sampled_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
