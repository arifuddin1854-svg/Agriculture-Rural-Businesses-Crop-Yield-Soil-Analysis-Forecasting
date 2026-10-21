import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from src.api.database import Base
from src.api.models.base import GUID, TimestampMixin


class SoilLog(Base, TimestampMixin):
    __tablename__ = "soil_logs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    field_name = Column(String(150), nullable=False)
    district = Column(String(100), nullable=False, index=True)
    
    # Geographic coordinates
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Chemical & Physical Properties
    nitrogen = Column(Float, nullable=False, comment="Soil Nitrogen in kg/ha")
    phosphorus = Column(Float, nullable=False, comment="Soil Phosphorus in kg/ha")
    potassium = Column(Float, nullable=False, comment="Soil Potassium in kg/ha")
    ph = Column(Float, nullable=False, comment="Soil pH level (0-14)")
    organic_matter = Column(Float, nullable=True, comment="Soil Organic Matter %")
    moisture = Column(Float, nullable=True, comment="Soil Moisture %")
    soil_type = Column(String(50), nullable=True, comment="Classification e.g. Clay, Loam, Sandy")

    # Sample timestamp & metadata
    sampled_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    notes = Column(Text, nullable=True)

    # Check constraints
    __table_args__ = (
        CheckConstraint("ph >= 0 AND ph <= 14", name="chk_soil_ph_range"),
        CheckConstraint("nitrogen >= 0", name="chk_nitrogen_non_negative"),
        CheckConstraint("phosphorus >= 0", name="chk_phosphorus_non_negative"),
        CheckConstraint("potassium >= 0", name="chk_potassium_non_negative"),
    )

    # Relationships
    user = relationship("User", back_populates="soil_logs")
    yield_predictions = relationship("YieldPrediction", back_populates="soil_log")

    def __repr__(self):
        return f"<SoilLog(id={self.id}, field={self.field_name}, district={self.district}, pH={self.ph})>"
