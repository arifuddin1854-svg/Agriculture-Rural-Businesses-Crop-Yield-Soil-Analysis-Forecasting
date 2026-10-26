from src.api.routes.health import router as health_router
from src.api.routes.users import router as users_router
from src.api.routes.soil_logs import router as soil_logs_router
from src.api.routes.predictions import router as predictions_router
from src.api.routes.predict import router as predict_router
from src.api.routes.recommend import router as recommend_router

__all__ = [
    "health_router",
    "users_router",
    "soil_logs_router",
    "predictions_router",
    "predict_router",
    "recommend_router",
]
