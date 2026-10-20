from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class RecommendationItem(BaseModel):
    category: str = Field(..., description="e.g. 'Soil pH Alert', 'Nitrogen (N) Status', 'Nutrient Balance'")
    current_level: float
    optimal_range: str
    recommendation: str
    expected_impact_mt_ha: float
    percentage_gain: float


class YieldPredictionBase(BaseModel):
    crop: str = Field(..., max_length=100, description="Target crop e.g. 'Wheat', 'Maize'")
    season: Optional[str] = Field(None, max_length=50)
    model_version: str = Field(default="v1.0.0")
    predicted_yield: float = Field(..., description="Forecasted yield in Metric Tons / Hectare")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    baseline_yield: Optional[float] = None
    potential_gain: Optional[float] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    shap_values: Optional[Dict[str, float]] = None
    input_features: Dict[str, Any] = Field(..., description="Snapshot of features fed into the model")

    model_config = ConfigDict(protected_namespaces=())


class YieldPredictionCreate(YieldPredictionBase):
    user_id: UUID
    soil_log_id: Optional[UUID] = None


class YieldPredictionRead(YieldPredictionBase):
    id: UUID
    user_id: UUID
    soil_log_id: Optional[UUID] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
