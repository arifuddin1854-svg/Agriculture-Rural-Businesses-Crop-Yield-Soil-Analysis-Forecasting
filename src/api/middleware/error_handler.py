"""
Standardized API Exception Handling and Middleware.
Provides unified HTTP status code responses for:
- Invalid inputs and schema validation boundaries (422)
- Missing resources and attributes (404)
- Model serving and inference failures (500 / 503)
- Unhandled internal exceptions (500)
"""

from datetime import datetime
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("axlero_agri_api.errors")


class ModelServingError(Exception):
    """Raised when real-time model inference or scoring pipeline fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 503):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.status_code = status_code


class AgronomicDataError(ValueError):
    """Raised when input soil parameters violate agronomic physical constraints."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


def build_error_response(
    status_code: int,
    code: str,
    message: str,
    details: Optional[Any] = None,
    path: Optional[str] = None,
) -> JSONResponse:
    """
    Constructs a standardized JSON response conforming to API error specifications.
    """
    error_payload = {
        "detail": message,
        "error": {
            "code": code,
            "message": message,
            "status_code": status_code,
            "details": details or [],
            "path": path,
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    return JSONResponse(status_code=status_code, content=error_payload)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handles Pydantic data validation errors (e.g., pH outside 0-14, negative NPK).
    Returns HTTP 422 with structured field details.
    """
    errors_list: List[Dict[str, Any]] = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors_list.append({
            "field": field_path,
            "message": err.get("msg", "Validation error"),
            "type": err.get("type", "value_error"),
        })

    logger.warning(
        f"Validation failed on {request.method} {request.url.path}: {len(errors_list)} violation(s)."
    )

    return build_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="VALIDATION_ERROR",
        message=f"Request payload validation failed with {len(errors_list)} error(s).",
        details=errors_list,
        path=str(request.url.path),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handles standard HTTP exceptions (404 Not Found, 400 Bad Request, etc.).
    """
    code_map = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        422: "UNPROCESSABLE_ENTITY",
    }
    code = code_map.get(exc.status_code, "HTTP_ERROR")
    logger.info(f"HTTPException [{exc.status_code}] on {request.url.path}: {exc.detail}")

    return build_error_response(
        status_code=exc.status_code,
        code=code,
        message=str(exc.detail),
        details={"status_code": exc.status_code},
        path=str(request.url.path),
    )


async def model_serving_exception_handler(request: Request, exc: ModelServingError) -> JSONResponse:
    """
    Handles ML inference and recommendation model serving errors.
    Returns HTTP 503 or specified error code.
    """
    logger.error(f"ModelServingError on {request.url.path}: {exc.message}", exc_info=True)
    return build_error_response(
        status_code=exc.status_code,
        code="MODEL_SERVING_ERROR",
        message=exc.message,
        details=exc.details,
        path=str(request.url.path),
    )


async def agronomic_data_exception_handler(request: Request, exc: AgronomicDataError) -> JSONResponse:
    """
    Handles domain-specific agronomic data errors.
    """
    logger.warning(f"AgronomicDataError on {request.url.path}: {exc.message}")
    return build_error_response(
        status_code=exc.status_code,
        code="AGRONOMIC_DATA_ERROR",
        message=exc.message,
        details=exc.details,
        path=str(request.url.path),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for unhandled internal server exceptions.
    Returns HTTP 500 with sanitized message.
    """
    logger.error(f"Unhandled Server Error on {request.url.path}: {exc}", exc_info=True)
    return build_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected server error occurred while processing the request.",
        details={"error_class": type(exc).__name__},
        path=str(request.url.path),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers all standardized exception handlers to the FastAPI application instance.
    """
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(ModelServingError, model_serving_exception_handler)
    app.add_exception_handler(AgronomicDataError, agronomic_data_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
