import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.config import settings
from src.api.database import get_db, engine
from src.api.schemas.health import HealthResponse, DatabaseStatus

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health & DB Connectivity Check",
    description="Returns the operational status of the API, runtime environment, and PostgreSQL database connectivity.",
)
def check_health(db: Session = Depends(get_db)) -> HealthResponse:
    """
    Probes system availability and verifies active database connectivity with latency check.
    """
    db_status = "disconnected"
    latency_ms = None
    details = None
    engine_name = engine.dialect.name if engine else "unknown"

    try:
        start_time = time.perf_counter()
        db.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        db_status = "connected"
        details = f"Connection established to {engine_name} engine."
    except Exception as exc:
        db_status = "disconnected"
        details = f"Database probe failed: {str(exc)}"

    overall_status = "healthy" if db_status == "connected" else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version=settings.VERSION,
        service=settings.PROJECT_NAME,
        database=DatabaseStatus(
            status=db_status,
            latency_ms=latency_ms,
            engine=engine_name,
            details=details,
        ),
        environment=settings.ENVIRONMENT,
    )
