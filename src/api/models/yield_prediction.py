import uuid
from sqlalchemy import Column, String, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from src.api.database import Base
from src.api.models.base import GUID, TimestampMixin


class YieldPrediction(Base, TimestampMixin):
    __tablename__ = "yield_predictions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    soil_log_id = Column(GUID(), ForeignKey("soil_logs.id", ondelete="SET NULL"), nullable=True, index=True)

    # Prediction parameters & metadata
    crop = Column(String(100), nullable=False, index=True)
    season = Column(String(50), nullable=True)
    model_version = Column(String(50), default="v1.0.0", nullable=False)

    # Core predictions & impact
    predicted_yield = Column(Float, nullable=False, comment="Forecasted harvest volume in Metric Tons / Hectare")
    confidence_score = Column(Float, nullable=True, comment="Model confidence score (0.0 to 1.0)")
    baseline_yield = Column(Float, nullable=True, comment="Unoptimized baseline yield in MT/Ha")
    potential_gain = Column(Float, nullable=True, comment="Potential gain from recommended soil amendments in MT/Ha")

    # Prescriptive recommendations & Explainability (SHAP)
    recommendations = Column(JSON, nullable=True, comment="Structured agronomic amendments and actions")
    shap_values = Column(JSON, nullable=True, comment="Local feature attributions for controllable & exogenous factors")
    input_features = Column(JSON, nullable=False, comment="Snapshot of weather & soil feature inputs used for inference")

    # Relationships
    user = relationship("User", back_populates="yield_predictions")
    soil_log = relationship("SoilLog", back_populates="yield_predictions")

    def __repr__(self):
        return (
            f"<YieldPrediction(id={self.id}, crop={self.crop}, "
            f"predicted_yield={self.predicted_yield:.2f} MT/Ha)>"
        )
