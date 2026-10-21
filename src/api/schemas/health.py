from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class DatabaseStatus(BaseModel):
    status: str = Field(..., description="Connection status: 'connected' or 'disconnected'")
    latency_ms: Optional[float] = Field(None, description="Round-trip latency in milliseconds if connected")
    engine: Optional[str] = Field(None, description="Database dialect/engine name")
    details: Optional[str] = Field(None, description="Diagnostic details or error message")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status: 'healthy' or 'degraded'")
    timestamp: datetime = Field(..., description="UTC timestamp of the health check")
    version: str = Field(..., description="API Version")
    service: str = Field(..., description="Name of the service")
    database: DatabaseStatus = Field(..., description="Database connectivity status")
    environment: str = Field(..., description="Runtime environment")
