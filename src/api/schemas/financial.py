"""
Pydantic Schemas for Financial Calculator API & Unified Business Response JSON.
Deliverable: Unified Business Response JSON Schema (Engineer 3)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CostBreakdownItem(BaseModel):
    category: str
    cost_per_ha: float
    total_cost: float
    percentage_of_total: float


class CostBreakdown(BaseModel):
    seeds: float = Field(..., description="Certified high-yield seed expenditure")
    fertilizers: float = Field(..., description="Nutrient and fertilizer applications (N, P, K, micronutrients)")
    irrigation: float = Field(..., description="Electricity, diesel, and water management costs")
    machinery_and_labor: float = Field(..., description="Tractor tillage, sowing, harvesting, and field labor")
    crop_protection: float = Field(..., description="Pesticides, herbicides, and crop health treatments")
    post_harvest: float = Field(..., description="Bagging, handling, and transportation to local mandi")
    overheads: float = Field(..., description="Crop insurance (PMFBY), land lease, and incidental overheads")
    total_cost: float = Field(..., description="Sum of all operational costs")
    cost_per_ha: float = Field(..., description="Total operational cost normalized per hectare")
    itemized: List[CostBreakdownItem] = Field(default_factory=list)


class FinancialMetrics(BaseModel):
    currency: str = Field(default="INR", description="Currency ISO code (INR, USD, EUR)")
    crop_price_per_mt: float = Field(..., description="Commodity price per Metric Ton")
    gross_revenue: float = Field(..., description="Total anticipated sales revenue")
    total_operational_costs: float = Field(..., description="Total operational expenses for the cultivation season")
    net_profit: float = Field(..., description="Gross revenue minus total operational costs")
    roi_percentage: float = Field(..., description="Return on Investment percentage: (Net Profit / Operational Costs) * 100")
    profit_margin_percentage: float = Field(..., description="Net Profit Margin percentage: (Net Profit / Gross Revenue) * 100")
    break_even_yield_mt_ha: float = Field(..., description="Minimum yield required to cover all operational costs (MT/Ha)")
    break_even_price_per_mt: float = Field(..., description="Minimum commodity price per MT needed to achieve break-even")
    revenue_per_ha: float = Field(..., description="Gross revenue normalized per hectare")
    cost_per_ha: float = Field(..., description="Operational cost normalized per hectare")
    net_profit_per_ha: float = Field(..., description="Net profit normalized per hectare")
    cost_breakdown: CostBreakdown


class SHAPFactorFeedback(BaseModel):
    feature: str = Field(..., description="Agronomic or environmental factor name")
    domain: str = Field(..., description="'Controllable (Soil / Nutrient)' or 'Exogenous (Meteorological)'")
    shap_impact_mt_ha: float = Field(..., description="Directional yield impact in MT/ha")
    financial_impact: float = Field(..., description="Directional monetary impact on gross revenue")
    direction: str = Field(..., description="'Positive' (yield booster) or 'Negative' (yield penalty)")
    interpretation: str = Field(..., description="Actionable agronomic interpretation")


class PrescriptiveAmendmentFinancial(BaseModel):
    category: str
    current_level: float
    optimal_range: str
    recommendation: str
    expected_impact_mt_ha: float
    estimated_treatment_cost_total: float
    projected_revenue_gain_total: float
    net_benefit_total: float
    amendment_roi_percentage: float


class SHAPDiagnosticFeedback(BaseModel):
    summary: str = Field(..., description="High-level diagnostic narrative of yield and profit drivers")
    controllable_impact_total: float = Field(..., description="Net yield impact of farmer-controllable soil factors (MT/ha)")
    exogenous_impact_total: float = Field(..., description="Net yield impact of exogenous weather factors (MT/ha)")
    controllable_financial_impact: float = Field(..., description="Total monetary impact of controllable soil conditions")
    exogenous_financial_impact: float = Field(..., description="Total monetary impact of weather and climate conditions")
    top_limiting_factors: List[SHAPFactorFeedback] = Field(default_factory=list)
    top_positive_drivers: List[SHAPFactorFeedback] = Field(default_factory=list)
    feature_attributions: List[SHAPFactorFeedback] = Field(default_factory=list)
    prescriptive_amendments: List[PrescriptiveAmendmentFinancial] = Field(default_factory=list)


class FarmSummary(BaseModel):
    crop_type: str
    farm_size_ha: float
    predicted_yield_mt_ha: float
    total_production_mt: float
    district: Optional[str] = None
    field_name: Optional[str] = None


class FinancialForecastRequest(BaseModel):
    """
    Request contract for POST /business/financial-forecast.
    Ingests farm size, crop type, and predicted yield values.
    """
    crop_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Target crop commodity (e.g. 'Wheat', 'Maize', 'Rice', 'Cotton', 'Soybean', 'Chickpea', 'Mustard')",
        examples=["Wheat"],
    )
    farm_size: float = Field(
        ...,
        gt=0.0,
        description="Cultivated land area. Must be strictly positive (> 0).",
        examples=[10.0],
    )
    farm_size_unit: str = Field(
        default="hectares",
        description="Unit of area measurement: 'hectares' or 'acres'. (1 acre = 0.404686 ha)",
        examples=["hectares"],
    )
    predicted_yield: float = Field(
        ...,
        ge=0.0,
        description="Predicted harvest yield in Metric Tons per Hectare (MT/Ha).",
        examples=[4.85],
    )

    # Optional Custom Financial Overrides
    custom_crop_price_per_mt: Optional[float] = Field(
        None,
        ge=0.0,
        description="Custom commodity market price per Metric Ton (overrides standard benchmark/MSP).",
        examples=[24500.0],
    )
    custom_operational_cost_per_ha: Optional[float] = Field(
        None,
        ge=0.0,
        description="Custom operational cost per hectare (overrides standard itemized cost model).",
        examples=[38000.0],
    )
    currency: str = Field(
        default="INR",
        description="Currency code for financial reporting ('INR', 'USD', 'EUR').",
        examples=["INR"],
    )

    # Optional Soil & Environmental Profile for Live SHAP Diagnostics
    nitrogen: Optional[float] = Field(None, ge=0.0, le=1000.0, description="Soil Nitrogen (N) kg/ha")
    phosphorus: Optional[float] = Field(None, ge=0.0, le=500.0, description="Soil Phosphorus (P) kg/ha")
    potassium: Optional[float] = Field(None, ge=0.0, le=1000.0, description="Soil Potassium (K) kg/ha")
    ph: Optional[float] = Field(None, ge=0.0, le=14.0, description="Soil pH level (0-14)")
    rainfall: Optional[float] = Field(None, ge=0.0, description="Seasonal rainfall in mm")
    temp_max: Optional[float] = Field(None, description="Average maximum temperature in °C")

    # Optional Metadata & Foreign Key Tracking
    user_id: Optional[UUID] = None
    soil_log_id: Optional[UUID] = None
    yield_prediction_id: Optional[UUID] = None
    field_name: Optional[str] = None
    district: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("farm_size_unit")
    @classmethod
    def validate_unit(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("hectares", "ha", "acres", "acre"):
            raise ValueError("farm_size_unit must be 'hectares' (or 'ha') or 'acres' (or 'acre')")
        return clean

    model_config = ConfigDict(extra="ignore")


class UnifiedBusinessResponse(BaseModel):
    """
    Unified Business Response JSON combining ROI calculation engine output,
    itemized costs, SHAP factor explainability, and database persistence audit details.
    """
    status: str = Field(default="success", description="Operational execution status")
    audit_id: UUID = Field(..., description="Unique persistent identifier in financial_audit_logs table")
    timestamp: datetime = Field(..., description="Assessment execution timestamp")
    farm_summary: FarmSummary = Field(..., description="Cultivation profile and volume metrics")
    financial_metrics: FinancialMetrics = Field(..., description="Executive financial return, revenue, cost, and margin forecasts")
    shap_diagnostic_feedback: SHAPDiagnosticFeedback = Field(..., description="Engineer 2 SHAP explainability & agronomic diagnostic feedback")
    persisted: bool = Field(default=True, description="Indicates whether audit log was stored in PostgreSQL database")
    user_id: Optional[UUID] = Field(None, description="Associated user identity if authenticated")

    model_config = ConfigDict(from_attributes=True)


class FinancialAuditLogRead(BaseModel):
    """
    Schema for reading persisted financial audit logs.
    """
    id: UUID
    user_id: Optional[UUID] = None
    soil_log_id: Optional[UUID] = None
    yield_prediction_id: Optional[UUID] = None
    crop_type: str
    farm_size_ha: float
    predicted_yield: float
    total_production_mt: float
    currency: str
    crop_price_per_mt: float
    gross_revenue: float
    total_operational_costs: float
    net_profit: float
    roi_percentage: float
    profit_margin_percentage: float
    break_even_yield: Optional[float] = None
    break_even_price: Optional[float] = None
    cost_breakdown: Optional[Dict[str, Any]] = None
    shap_diagnostic_feedback: Optional[Dict[str, Any]] = None
    prescriptive_amendments: Optional[List[Dict[str, Any]]] = None
    assessment_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

