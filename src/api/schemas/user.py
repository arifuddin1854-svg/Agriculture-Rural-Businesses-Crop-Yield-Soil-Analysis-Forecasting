from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    role: str = Field(default="farmer", description="Role: 'farmer', 'agronomist', 'researcher', 'admin'")
    farm_name: Optional[str] = None
    district: Optional[str] = None
    is_active: bool = True


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="Plaintext password to be hashed")


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    farm_name: Optional[str] = None
    district: Optional[str] = None
    is_active: Optional[bool] = None


class UserRead(UserBase):
    id: UUID
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
