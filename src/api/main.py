import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.config import settings
from src.api.database import engine, Base
import src.api.models  # Registers all models with Base.metadata
from src.api.routes.health import router as health_router
from src.api.routes.users import router as users_router
from src.api.routes.soil_logs import router as soil_logs_router
from src.api.routes.predictions import router as predictions_router
from src.api.routes.predict import router as predict_router
from src.api.routes.recommend import router as recommend_router
from src.api.routes.financial import router as financial_router
from src.api.auth import router as auth_router
from src.api.middleware.error_handler import register_exception_handlers
from src.api.middleware import (
    limiter,
    _rate_limit_exceeded_handler,
    SecurityHeadersMiddleware,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("axlero_agri_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management:
    Initializes database tables in development mode and handles graceful shutdown.
    """
    logger.info("Initializing Axlero Agri-Tech API services...")
    # In development/test mode, create tables if they do not exist
    try:
        if settings.ENVIRONMENT == "development":
            logger.info("Verifying/creating database tables...")
            Base.metadata.create_all(bind=engine)
    except Exception as exc:
        logger.warning(f"Could not automatically create tables at startup: {exc}")

    yield
    logger.info("Shutting down Axlero Agri-Tech API services...")


_OPENAPI_TAGS = [
    {
        "name": "Authentication & User Sessions",
        "description": (
            "OAuth2 Password Flow and JWT token issuance, refresh, and revocation. "
            "Supports user registration, login, and authenticated session management."
        ),
    },
    {
        "name": "Financial Calculator & Agribusiness Analytics",
        "description": (
            "POST /business/financial-forecast computes Gross Revenue, "
            "Total Operational Costs, Net Profit, ROI percentage, break-even yield, "
            "and SHAP diagnostic factor attributions for agribusiness planning."
        ),
    },
    {
        "name": "Yield Prediction & Inference",
        "description": (
            "POST /predict/yield ingests soil chemistry (N, P, K, pH) and "
            "weather parameters to produce machine-learning yield forecasts "
            "with confidence scores and SHAP feature importances."
        ),
    },
    {
        "name": "Crop Recommendations",
        "description": (
            "POST /recommend/crop returns ranked crop suitability scores "
            "based on soil composition, climate data, and historical performance."
        ),
    },
    {
        "name": "Health",
        "description": (
            "Service liveness and readiness probes used by load balancers, "
            "orchestration platforms, and CI health checks."
        ),
    },
    {
        "name": "Users",
        "description": (
            "CRUD operations for farmer, agronomist, and business manager "
            "user profiles. Requires admin or authenticated access."
        ),
    },
    {
        "name": "Soil Logs",
        "description": (
            "Create and query historical soil chemistry measurement records "
            "tied to farm locations and assessment sessions."
        ),
    },
    {
        "name": "Predictions",
        "description": (
            "Retrieve historical yield prediction records and assessment "
            "reports linked to soil logs and farm sessions."
        ),
    },
    {"name": "Root", "description": "API root welcome and navigation endpoint."},
]


def create_application() -> FastAPI:
    """
    Factory function to construct and configure the FastAPI application.
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Axlero Agri-Tech Initiative: Predictive & Prescriptive API platform for "
            "Crop Yield Forecasting, Multi-Source Soil Chemistry Analysis, and SHAP-driven Advisory."
        ),
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
        openapi_tags=_OPENAPI_TAGS,
    )

    # Attach rate-limiter state so slowapi can intercept requests
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Security Headers Middleware (applied to all responses)
    app.add_middleware(SecurityHeadersMiddleware)

    # Register Standardized Exception Handlers
    register_exception_handlers(app)

    # CORS Middleware — restrict to configured origins in production
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "Accept"],
        )

    # Direct Routes (allowing POST /predict/yield, POST /recommend/crop, POST /business/financial-forecast without prefix)
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(predict_router)
    app.include_router(recommend_router)
    app.include_router(financial_router)

    # API v1 Versioned Routes
    app.include_router(health_router, prefix=settings.API_V1_STR)
    app.include_router(auth_router, prefix=settings.API_V1_STR)
    app.include_router(users_router, prefix=settings.API_V1_STR)
    app.include_router(soil_logs_router, prefix=settings.API_V1_STR)
    app.include_router(predictions_router, prefix=settings.API_V1_STR)
    app.include_router(predict_router, prefix=settings.API_V1_STR)
    app.include_router(recommend_router, prefix=settings.API_V1_STR)
    app.include_router(financial_router, prefix=settings.API_V1_STR)

    @app.get("/", tags=["Root"])
    def root():
        return {
            "message": "Welcome to Axlero Agri-Tech Crop Yield & Soil Analysis API",
            "version": settings.VERSION,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_application()
