"""
Axlero Agri-Tech Financial Calculation Engine & ROI Module.
Deliverable: Financial Calculation Logic (Engineer 3)

Provides direct programmatic access to:
- calculate_roi_and_financials
- execute_financial_forecast
- build_shap_diagnostic_feedback
- normalize_farm_size
- CROP_FINANCIAL_BENCHMARKS
"""

from src.api.services.financial_service import (
    calculate_roi_and_financials,
    execute_financial_forecast,
    build_shap_diagnostic_feedback,
    normalize_farm_size,
    get_crop_benchmark,
    CROP_FINANCIAL_BENCHMARKS,
    DEFAULT_BENCHMARK,
)

__all__ = [
    "calculate_roi_and_financials",
    "execute_financial_forecast",
    "build_shap_diagnostic_feedback",
    "normalize_farm_size",
    "get_crop_benchmark",
    "CROP_FINANCIAL_BENCHMARKS",
    "DEFAULT_BENCHMARK",
]

