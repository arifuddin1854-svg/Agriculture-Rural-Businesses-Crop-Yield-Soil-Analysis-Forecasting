import uuid
from sqlalchemy import Column, String, Boolean
from sqlalchemy.orm import relationship
from src.api.database import Base
from src.api.models.base import GUID, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(String(50), default="farmer", nullable=False)
    farm_name = Column(String(255), nullable=True)
    district = Column(String(100), nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)

    # Relationships
    soil_logs = relationship("SoilLog", back_populates="user", cascade="all, delete-orphan")
    yield_predictions = relationship("YieldPrediction", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"
