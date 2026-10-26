"""
Pydantic Data Contracts for Soil Chemistry Analysis, Yield Prediction, and Crop Recommendation.
Defines strict type boundaries, ranges, and agronomic constraints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator


class SoilParameters(BaseModel):
    """
    Standard soil test parameters with strict physical and chemical bounds.
    """
    nitrogen: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Soil Nitrogen (N) content in kg/ha. Must be non-negative (>= 0).",
        examples=[85.0],
    )
    phosphorus: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Soil Phosphorus (P) content in kg/ha. Must be non-negative (>= 0).",
        examples=[42.0],
    )
    potassium: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Soil Potassium (K) content in kg/ha. Must be non-negative (>= 0).",
        examples=[120.0],
    )
    ph: float = Field(
        ...,
        ge=0.0,
        le=14.0,
        description="Soil pH level. Strictly bounded between 0.0 (strongly acidic) and 14.0 (strongly alkaline).",
        examples=[6.4],
    )
    organic_matter: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Soil organic matter percentage (0-100%).",
        examples=[2.5],
    )
    moisture: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Soil moisture percentage (0-100%).",
        examples=[22.0],
    )
    soil_type: Optional[str] = Field(
        None,
        max_length=50,
        description="Taxonomic classification (e.g., 'Clay Loam', 'Alluvial', 'Sandy Loam', 'Black Soil').",
        examples=["Clay Loam"],
    )

    model_config = ConfigDict(extra="ignore")


class YieldPredictionRequest(BaseModel):
    """
    Request contract for real-time crop yield inference at POST /predict/yield.
    """
    crop: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target crop name (e.g. 'Wheat', 'Maize', 'Soybean', 'Rice', 'Cotton').",
        examples=["Wheat"],
    )
    nitrogen: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Soil Nitrogen (N) in kg/ha. Positive / non-negative value.",
        examples=[85.0],
    )
    phosphorus: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Soil Phosphorus (P) in kg/ha. Positive / non-negative value.",
        examples=[42.0],
    )
    potassium: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Soil Potassium (K) in kg/ha. Positive / non-negative value.",
        examples=[120.0],
    )
    ph: float = Field(
        ...,
        ge=0.0,
        le=14.0,
        description="Soil pH scale (0.0 to 14.0).",
        examples=[5.6],
    )
    rainfall: Optional[float] = Field(
        550.0,
        ge=0.0,
        le=5000.0,
        description="Cumulative or seasonal rainfall in mm.",
        examples=[550.0],
    )
    temp_max: Optional[float] = Field(
        29.5,
        ge=-50.0,
        le=60.0,
        description="Average maximum temperature in °C during growing season.",
        examples=[29.5],
    )
    season: Optional[str] = Field(
        None,
        max_length=50,
        description="Growing season identifier (e.g. 'Kharif', 'Rabi', 'Zaid', '2026').",
        examples=["Rabi 2026"],
    )
    district: Optional[str] = Field(
        None,
        max_length=100,
        description="District or geographic region identifier.",
        examples=["Patiala"],
    )
    field_name: Optional[str] = Field(
        None,
        max_length=150,
        description="Farm or plot parcel name.",
        examples=["Plot 4-East"],
    )
    user_id: Optional[UUID] = Field(
        None,
        description="Optional registered user/farmer ID for record association.",
    )

    @field_validator("crop")
    @classmethod
    def validate_crop_name(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Crop name cannot be empty or whitespace only.")
        return cleaned.title()

    model_config = ConfigDict(extra="ignore")


class RecommendationDetail(BaseModel):
    """
    Actionable soil amendment or nutrient diagnosis item.
    """
    category: str = Field(..., description="Nutrient or physical factor (e.g. 'Soil pH Alert', 'Nitrogen (N) Status').")
    current_level: float = Field(..., description="Observed input value.")
    optimal_range: str = Field(..., description="Agronomically recommended range.")
    recommendation: str = Field(..., description="Actionable prescription (fertilizer, lime, compost, etc.).")
    expected_impact_mt_ha: float = Field(..., description="Estimated yield gain in Metric Tons / Hectare.")
    percentage_gain: float = Field(..., description="Percentage yield gain.")


class YieldPredictionResponse(BaseModel):
    """
    Response contract for POST /predict/yield endpoint.
    """
    crop: str = Field(..., description="Target crop evaluated.")
    predicted_yield: float = Field(..., description="Estimated end-of-season yield in Metric Tons / Hectare.")
    baseline_yield: float = Field(..., description="Unoptimized baseline yield in MT/Ha.")
    potential_gain: float = Field(..., description="Projected gain from recommended amendments in MT/Ha.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Inference confidence metric (0.0 - 1.0).")
    model_version: str = Field(..., description="Model identifier used for inference.")
    recommendations: List[RecommendationDetail] = Field(
        default_factory=list,
        description="Prescriptive soil amendments.",
    )
    shap_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Local feature attributions for soil and climate factors.",
    )
    input_features: Dict[str, Any] = Field(
        ...,
        description="Snapshot of features evaluated by the model.",
    )
    prediction_id: Optional[UUID] = Field(None, description="Database ID of logged yield prediction record.")
    soil_log_id: Optional[UUID] = Field(None, description="Database ID of logged soil assessment record.")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Inference timestamp.")

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class CropRecommendationRequest(BaseModel):
    """
    Request contract for POST /recommend/crop evaluating NPK and soil properties.
    """
    nitrogen: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Soil Nitrogen (N) in kg/ha (must be positive / non-negative).",
        examples=[95.0],
    )
    phosphorus: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Soil Phosphorus (P) in kg/ha (must be positive / non-negative).",
        examples=[45.0],
    )
    potassium: float = Field(
        ...,
        ge=0.0,
        le=1000.0,
        description="Soil Potassium (K) in kg/ha (must be positive / non-negative).",
        examples=[130.0],
    )
    ph: float = Field(
        ...,
        ge=0.0,
        le=14.0,
        description="Soil pH level strictly within range 0.0 - 14.0.",
        examples=[6.5],
    )
    rainfall: Optional[float] = Field(
        None,
        ge=0.0,
        le=5000.0,
        description="Expected rainfall in mm (optional).",
        examples=[700.0],
    )
    temperature: Optional[float] = Field(
        None,
        ge=-50.0,
        le=60.0,
        description="Average temperature in °C (optional).",
        examples=[24.0],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of top suitable crops to return (1-10).",
        examples=[3],
    )
    district: Optional[str] = Field(None, max_length=100)
    field_name: Optional[str] = Field(None, max_length=150)
    user_id: Optional[UUID] = Field(None, description="Optional registered user/farmer ID.")

    model_config = ConfigDict(extra="ignore")


class CropRecommendationItem(BaseModel):
    """
    Candidate crop recommendation with suitability metrics.
    """
    crop: str = Field(..., description="Crop name.")
    suitability_score: float = Field(..., ge=0.0, le=100.0, description="Suitability match percentage (0-100%).")
    rank: int = Field(..., ge=1, description="Rank among suggested crops.")
    match_status: str = Field(..., description="Classification: 'Highly Suitable', 'Moderately Suitable', or 'Marginal'.")
    optimal_ranges: Dict[str, str] = Field(..., description="Optimal N, P, K, and pH benchmarks for this crop.")
    diagnostic_notes: List[str] = Field(default_factory=list, description="Specific nutrient alignment details.")


class CropRecommendationResponse(BaseModel):
    """
    Response contract for POST /recommend/crop endpoint.
    """
    top_crop: str = Field(..., description="Single most suitable crop.")
    recommendations: List[CropRecommendationItem] = Field(..., description="Ranked list of top candidate crops.")
    analyzed_inputs: Dict[str, float] = Field(..., description="Soil chemistry parameters evaluated.")
    soil_log_id: Optional[UUID] = Field(None, description="Database ID of logged assessment.")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Evaluation timestamp.")

    model_config = ConfigDict(from_attributes=True)
