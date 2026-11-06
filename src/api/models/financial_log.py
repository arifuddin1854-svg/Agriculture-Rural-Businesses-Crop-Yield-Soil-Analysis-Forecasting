"""
SQLAlchemy Model for Financial Audit Logs and Farm Business Assessments.
Deliverable: Financial Audit Logs Schema (Engineer 3)
"""

import uuid
from sqlalchemy import Column, String, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from src.api.database import Base
from src.api.models.base import GUID, TimestampMixin


class FinancialAuditLog(Base, TimestampMixin):
    """
    Persists historical financial assessments, ROI forecasts, operational cost breakdowns,
    and SHAP-driven diagnostic feedback for registered and guest farming enterprises.
    """
    __tablename__ = "financial_audit_logs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    soil_log_id = Column(GUID(), ForeignKey("soil_logs.id", ondelete="SET NULL"), nullable=True, index=True)
    yield_prediction_id = Column(GUID(), ForeignKey("yield_predictions.id", ondelete="SET NULL"), nullable=True, index=True)

    # Core Farm & Agronomic Inputs
    crop_type = Column(String(100), nullable=False, index=True)
    farm_size_ha = Column(Float, nullable=False, comment="Farm cultivation area in hectares")
    predicted_yield = Column(Float, nullable=False, comment="Estimated harvest yield in Metric Tons / Hectare")
    total_production_mt = Column(Float, nullable=False, comment="Total anticipated harvest volume in Metric Tons")

    # Financial Forecast & Return Metrics
    currency = Column(String(10), default="INR", nullable=False)
    crop_price_per_mt = Column(Float, nullable=False, comment="Commodity market/MSP rate per Metric Ton")
    gross_revenue = Column(Float, nullable=False, comment="Total gross sales revenue")
    total_operational_costs = Column(Float, nullable=False, comment="Sum of input, labor, machinery, and overhead costs")
    net_profit = Column(Float, nullable=False, comment="Gross revenue minus total operational costs")
    roi_percentage = Column(Float, nullable=False, comment="Return on Investment percentage: (Net Profit / Costs) * 100")
    profit_margin_percentage = Column(Float, nullable=False, comment="Net Profit Margin percentage: (Net Profit / Gross Revenue) * 100")
    break_even_yield = Column(Float, nullable=True, comment="Minimum yield required to cover all operational costs (MT/Ha)")
    break_even_price = Column(Float, nullable=True, comment="Minimum market price required to break even per MT")

    # Granular Itemized Breakdown & Explainability
    cost_breakdown = Column(JSON, nullable=True, comment="Itemized cost components (seeds, fertilizer, labor, irrigation, etc.)")
    shap_diagnostic_feedback = Column(JSON, nullable=True, comment="SHAP explainability attributions and diagnostic evaluation")
    prescriptive_amendments = Column(JSON, nullable=True, comment="Actionable agronomic interventions and projected gains")
    
    # Audit & Tracking Metadata
    assessment_notes = Column(String(255), nullable=True)
    audit_metadata = Column(JSON, nullable=True, comment="Session identifier, client IP, source client, environment details")

    # Relationships
    user = relationship("User", back_populates="financial_logs")
    soil_log = relationship("SoilLog", backref="financial_logs")
    yield_prediction = relationship("YieldPrediction", backref="financial_logs")

    def __repr__(self):
        return (
            f"<FinancialAuditLog(id={self.id}, crop={self.crop_type}, "
            f"farm_size={self.farm_size_ha}ha, net_profit={self.net_profit:,.2f} {self.currency}, "
            f"ROI={self.roi_percentage:.1f}%)>"
        )

