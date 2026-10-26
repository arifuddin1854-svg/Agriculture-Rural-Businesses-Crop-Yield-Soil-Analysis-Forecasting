import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.database import engine, Base
from src.api.routes.health import router as health_router
from src.api.routes.users import router as users_router
from src.api.routes.soil_logs import router as soil_logs_router
from src.api.routes.predictions import router as predictions_router
from src.api.routes.predict import router as predict_router
from src.api.routes.recommend import router as recommend_router
from src.api.middleware.error_handler import register_exception_handlers

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
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
        if settings.ENVIRONMENT in ["development", "testing"]:
            logger.info("Verifying/creating database tables...")
            Base.metadata.create_all(bind=engine)
    except Exception as exc:
        logger.warning(f"Could not automatically create tables at startup: {exc}")
    
    yield
    logger.info("Shutting down Axlero Agri-Tech API services...")


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
    )

    # Register Standardized Exception Handlers
    register_exception_handlers(app)

    # CORS Middleware
    if settings.BACKEND_CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Direct Routes (allowing POST /predict/yield and POST /recommend/crop without prefix)
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(recommend_router)

    # API v1 Versioned Routes
    app.include_router(health_router, prefix=settings.API_V1_STR)
    app.include_router(users_router, prefix=settings.API_V1_STR)
    app.include_router(soil_logs_router, prefix=settings.API_V1_STR)
    app.include_router(predictions_router, prefix=settings.API_V1_STR)
    app.include_router(predict_router, prefix=settings.API_V1_STR)
    app.include_router(recommend_router, prefix=settings.API_V1_STR)

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
