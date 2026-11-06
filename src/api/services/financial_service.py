"""
Financial Calculation Engine & SHAP Explainability Feedback Module.
Deliverable: Financial Calculation Logic & Unified Business Response Integration (Engineer 3)
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

from src.api.schemas.financial import (
    CostBreakdown,
    CostBreakdownItem,
    FarmSummary,
    FinancialForecastRequest,
    FinancialMetrics,
    PrescriptiveAmendmentFinancial,
    SHAPDiagnosticFeedback,
    SHAPFactorFeedback,
    UnifiedBusinessResponse,
)
from src.api.services.inference import (
    CROP_BASELINES,
    compute_advisory_recommendations,
    compute_shap_attributions,
)

logger = logging.getLogger("axlero_agri_api.financial_service")

# ---------------------------------------------------------------------------
# Commodity Pricing Benchmarks (INR / Metric Ton) & Production Cost Models
# ---------------------------------------------------------------------------

CROP_FINANCIAL_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "Wheat": {
        "price_per_mt": 22750.0,  # Central MSP benchmark
        "baseline_yield": 4.50,
        "costs_per_ha": {
            "seeds": 4000.0,
            "fertilizers": 9000.0,
            "irrigation": 5500.0,
            "machinery_and_labor": 12000.0,
            "crop_protection": 3500.0,
            "post_harvest": 2500.0,
            "overheads": 2000.0,
        },
    },
    "Maize": {
        "price_per_mt": 20900.0,
        "baseline_yield": 6.35,
        "costs_per_ha": {
            "seeds": 4500.0,
            "fertilizers": 8500.0,
            "irrigation": 4500.0,
            "machinery_and_labor": 11000.0,
            "crop_protection": 3000.0,
            "post_harvest": 2000.0,
            "overheads": 1500.0,
        },
    },
    "Rice": {
        "price_per_mt": 23000.0,
        "baseline_yield": 5.40,
        "costs_per_ha": {
            "seeds": 4500.0,
            "fertilizers": 9500.0,
            "irrigation": 8500.0,
            "machinery_and_labor": 14000.0,
            "crop_protection": 4000.0,
            "post_harvest": 3000.0,
            "overheads": 2500.0,
        },
    },
    "Cotton": {
        "price_per_mt": 71200.0,
        "baseline_yield": 2.60,
        "costs_per_ha": {
            "seeds": 7500.0,
            "fertilizers": 11000.0,
            "irrigation": 6500.0,
            "machinery_and_labor": 16500.0,
            "crop_protection": 6500.0,
            "post_harvest": 3000.0,
            "overheads": 2500.0,
        },
    },
    "Soybean": {
        "price_per_mt": 48920.0,
        "baseline_yield": 3.20,
        "costs_per_ha": {
            "seeds": 5500.0,
            "fertilizers": 7000.0,
            "irrigation": 3500.0,
            "machinery_and_labor": 11500.0,
            "crop_protection": 3000.0,
            "post_harvest": 2500.0,
            "overheads": 1500.0,
        },
    },
    "Chickpea": {
        "price_per_mt": 54400.0,
        "baseline_yield": 1.95,
        "costs_per_ha": {
            "seeds": 5000.0,
            "fertilizers": 5500.0,
            "irrigation": 3000.0,
            "machinery_and_labor": 10000.0,
            "crop_protection": 2500.0,
            "post_harvest": 2000.0,
            "overheads": 1500.0,
        },
    },
    "Mustard": {
        "price_per_mt": 56500.0,
        "baseline_yield": 1.85,
        "costs_per_ha": {
            "seeds": 3500.0,
            "fertilizers": 6000.0,
            "irrigation": 3000.0,
            "machinery_and_labor": 9500.0,
            "crop_protection": 2000.0,
            "post_harvest": 2000.0,
            "overheads": 1500.0,
        },
    },
    "Lentil": {
        "price_per_mt": 64000.0,
        "baseline_yield": 1.65,
        "costs_per_ha": {
            "seeds": 4500.0,
            "fertilizers": 5000.0,
            "irrigation": 2500.0,
            "machinery_and_labor": 9000.0,
            "crop_protection": 2500.0,
            "post_harvest": 2000.0,
            "overheads": 1500.0,
        },
    },
    "Sugarcane": {
        "price_per_mt": 3150.0,
        "baseline_yield": 75.0,
        "costs_per_ha": {
            "seeds": 12000.0,
            "fertilizers": 18000.0,
            "irrigation": 15000.0,
            "machinery_and_labor": 25000.0,
            "crop_protection": 5000.0,
            "post_harvest": 6000.0,
            "overheads": 3000.0,
        },
    },
}

DEFAULT_BENCHMARK = {
    "price_per_mt": 25000.0,
    "baseline_yield": 4.00,
    "costs_per_ha": {
        "seeds": 4000.0,
        "fertilizers": 8000.0,
        "irrigation": 5000.0,
        "machinery_and_labor": 12000.0,
        "crop_protection": 3000.0,
        "post_harvest": 2000.0,
        "overheads": 1500.0,
    },
}

AMENDMENT_COST_BENCHMARKS: Dict[str, float] = {
    "Soil pH Alert": 3500.0,           # Lime application per hectare
    "Nitrogen (N) Deficiency": 1200.0,  # Urea top-dressing per hectare
    "Phosphorus (P) Booster": 1800.0,  # DAP application per hectare
    "Potassium (K) Balance": 1500.0,   # Muriate of Potash per hectare
}


# ---------------------------------------------------------------------------
# Unit Normalization & Helper Functions
# ---------------------------------------------------------------------------

def normalize_farm_size(farm_size: float, unit: str) -> float:
    """
    Normalizes land area into hectares.
    1 acre = 0.404686 hectares.
    """
    unit_clean = unit.lower().strip()
    if unit_clean in ("acre", "acres"):
        return round(farm_size * 0.404686, 4)
    return round(farm_size, 4)


def get_crop_benchmark(crop_type: str) -> Dict[str, Any]:
    """
    Fetches financial & cost profile for the given crop or falls back to default.
    """
    normalized = crop_type.strip().title()
    return CROP_FINANCIAL_BENCHMARKS.get(normalized, DEFAULT_BENCHMARK)


# ---------------------------------------------------------------------------
# ROI Calculation Engine
# ---------------------------------------------------------------------------

def calculate_roi_and_financials(
    crop_type: str,
    farm_size_ha: float,
    predicted_yield_mt_ha: float,
    custom_crop_price_per_mt: Optional[float] = None,
    custom_operational_cost_per_ha: Optional[float] = None,
    currency: str = "INR",
) -> FinancialMetrics:
    """
    Executes the ROI calculation engine:
    - Ingests farm size, crop type, and predicted yield values
    - Calculates total production volume (MT)
    - Computes Gross Revenue, Itemized Operational Costs, Net Profit, and ROI %
    - Computes Break-even Yield and Break-even Commodity Price
    """
    benchmark = get_crop_benchmark(crop_type)

    # 1. Commodity Pricing
    crop_price = (
        float(custom_crop_price_per_mt)
        if custom_crop_price_per_mt is not None and custom_crop_price_per_mt > 0
        else benchmark["price_per_mt"]
    )

    # 2. Production Volume & Gross Revenue
    total_production_mt = round(farm_size_ha * predicted_yield_mt_ha, 3)
    gross_revenue = round(total_production_mt * crop_price, 2)

    # 3. Itemized Operational Costs
    std_costs_per_ha = benchmark["costs_per_ha"]
    std_total_cost_per_ha = sum(std_costs_per_ha.values())

    if custom_operational_cost_per_ha is not None and custom_operational_cost_per_ha > 0:
        cost_scale = custom_operational_cost_per_ha / max(1.0, std_total_cost_per_ha)
        active_costs_per_ha = {k: round(v * cost_scale, 2) for k, v in std_costs_per_ha.items()}
        total_cost_per_ha = float(custom_operational_cost_per_ha)
    else:
        active_costs_per_ha = std_costs_per_ha.copy()
        total_cost_per_ha = std_total_cost_per_ha

    total_operational_costs = round(total_cost_per_ha * farm_size_ha, 2)

    # Itemized Breakdown
    itemized_list: List[CostBreakdownItem] = []
    for category, per_ha in active_costs_per_ha.items():
        cat_total = round(per_ha * farm_size_ha, 2)
        pct = round((cat_total / max(1e-5, total_operational_costs)) * 100.0, 1)
        itemized_list.append(CostBreakdownItem(
            category=category.replace("_", " ").title(),
            cost_per_ha=per_ha,
            total_cost=cat_total,
            percentage_of_total=pct,
        ))

    cost_breakdown = CostBreakdown(
        seeds=round(active_costs_per_ha.get("seeds", 0.0) * farm_size_ha, 2),
        fertilizers=round(active_costs_per_ha.get("fertilizers", 0.0) * farm_size_ha, 2),
        irrigation=round(active_costs_per_ha.get("irrigation", 0.0) * farm_size_ha, 2),
        machinery_and_labor=round(active_costs_per_ha.get("machinery_and_labor", 0.0) * farm_size_ha, 2),
        crop_protection=round(active_costs_per_ha.get("crop_protection", 0.0) * farm_size_ha, 2),
        post_harvest=round(active_costs_per_ha.get("post_harvest", 0.0) * farm_size_ha, 2),
        overheads=round(active_costs_per_ha.get("overheads", 0.0) * farm_size_ha, 2),
        total_cost=total_operational_costs,
        cost_per_ha=round(total_cost_per_ha, 2),
        itemized=itemized_list,
    )

    # 4. Net Profit & Margins
    net_profit = round(gross_revenue - total_operational_costs, 2)

    if total_operational_costs > 0:
        roi_percentage = round((net_profit / total_operational_costs) * 100.0, 2)
    else:
        roi_percentage = 0.0

    if gross_revenue > 0:
        profit_margin_percentage = round((net_profit / gross_revenue) * 100.0, 2)
    else:
        profit_margin_percentage = 0.0

    # 5. Break-Even Metrics
    if crop_price > 0 and farm_size_ha > 0:
        break_even_yield = round(total_operational_costs / (crop_price * farm_size_ha), 3)
    else:
        break_even_yield = 0.0

    if total_production_mt > 0:
        break_even_price = round(total_operational_costs / total_production_mt, 2)
    else:
        break_even_price = 0.0

    # 6. Normalized Per-Hectare Indicators
    revenue_per_ha = round(gross_revenue / max(1e-5, farm_size_ha), 2)
    profit_per_ha = round(net_profit / max(1e-5, farm_size_ha), 2)

    return FinancialMetrics(
        currency=currency,
        crop_price_per_mt=crop_price,
        gross_revenue=gross_revenue,
        total_operational_costs=total_operational_costs,
        net_profit=net_profit,
        roi_percentage=roi_percentage,
        profit_margin_percentage=profit_margin_percentage,
        break_even_yield_mt_ha=break_even_yield,
        break_even_price_per_mt=break_even_price,
        revenue_per_ha=revenue_per_ha,
        cost_per_ha=round(total_cost_per_ha, 2),
        net_profit_per_ha=profit_per_ha,
        cost_breakdown=cost_breakdown,
    )


# ---------------------------------------------------------------------------
# SHAP Feedback API Integration & Prescriptive Return Analyzer
# ---------------------------------------------------------------------------

def build_shap_diagnostic_feedback(
    crop_type: str,
    farm_size_ha: float,
    crop_price_per_mt: float,
    nitrogen: Optional[float] = None,
    phosphorus: Optional[float] = None,
    potassium: Optional[float] = None,
    ph: Optional[float] = None,
    rainfall: Optional[float] = None,
    temp_max: Optional[float] = None,
    currency: str = "INR",
) -> SHAPDiagnosticFeedback:
    """
    Integrates Engineer 2's explainability module directly into the API response schema:
    - Computes local Shapley factor attributions
    - Disentangles Controllable (Farmer-Managed) vs. Exogenous (Weather) drivers
    - Quantifies directional monetary gain/loss for each agronomic factor
    - Prescribes high-ROI soil interventions with net gain and amendment ROI
    """
    # Fallback to balanced representative values if not provided
    val_n = nitrogen if nitrogen is not None else 85.0
    val_p = phosphorus if phosphorus is not None else 42.0
    val_k = potassium if potassium is not None else 80.0
    val_ph = ph if ph is not None else 6.2
    val_rain = rainfall if rainfall is not None else 580.0
    val_temp = temp_max if temp_max is not None else 29.0

    # 1. Compute SHAP attributions via Engineer 2's logic
    shap_dict = compute_shap_attributions(
        crop=crop_type,
        nitrogen=val_n,
        phosphorus=val_p,
        potassium=val_k,
        ph=val_ph,
        rainfall=val_rain,
        temp_max=val_temp,
    )

    domain_mapping = {
        "ph": "Controllable (Soil / Nutrient)",
        "nitrogen": "Controllable (Soil / Nutrient)",
        "phosphorus": "Controllable (Soil / Nutrient)",
        "potassium": "Controllable (Soil / Nutrient)",
        "rainfall": "Exogenous (Meteorological)",
        "temp_max": "Exogenous (Meteorological)",
    }

    interpretations = {
        "ph": (
            f"Soil pH at {val_ph:.2f}. "
            + ("Acidic constraint restricts root uptake." if val_ph < 6.0 else "Optimal soil acidity maintained.")
        ),
        "nitrogen": (
            f"Available N is {val_n:.1f} kg/ha. "
            + ("Nitrogen deficit dampens vegetative biomass." if val_n < 80.0 else "Nitrogen supply supports robust canopy growth.")
        ),
        "phosphorus": (
            f"Available P is {val_p:.1f} kg/ha. "
            + ("Suboptimal P inhibits root elongation." if val_p < 30.0 else "Sufficient P promotes early flowering.")
        ),
        "potassium": (
            f"Available K is {val_k:.1f} kg/ha. "
            + ("Potassium deficit increases drought vulnerability." if val_k < 60.0 else "Adequate K optimizes cellular turgor.")
        ),
        "rainfall": f"Cumulative precipitation ({val_rain:.1f} mm) provides adequate seasonal moisture.",
        "temp_max": (
            f"Maximum ambient temperature ({val_temp:.1f} °C). "
            + ("High thermal stress during anthesis." if val_temp > 35.0 else "Favorable thermal range.")
        ),
    }

    feature_feedbacks: List[SHAPFactorFeedback] = []
    controllable_impact_total = 0.0
    exogenous_impact_total = 0.0
    controllable_fin_total = 0.0
    exogenous_fin_total = 0.0

    for feat, shap_val in shap_dict.items():
        domain = domain_mapping.get(feat, "Agronomic Baseline / Spatial")
        # Monetary impact = delta_yield (MT/ha) * farm_size (ha) * price_per_mt
        fin_impact = round(shap_val * farm_size_ha * crop_price_per_mt, 2)
        direction = "Positive" if shap_val >= 0 else "Negative"

        feedback = SHAPFactorFeedback(
            feature=feat,
            domain=domain,
            shap_impact_mt_ha=round(shap_val, 4),
            financial_impact=fin_impact,
            direction=direction,
            interpretation=interpretations.get(feat, f"{feat} SHAP attribution impact"),
        )
        feature_feedbacks.append(feedback)

        if "Controllable" in domain:
            controllable_impact_total += shap_val
            controllable_fin_total += fin_impact
        else:
            exogenous_impact_total += shap_val
            exogenous_fin_total += fin_impact

    # Sort into top limiting and positive drivers
    top_limiting = sorted(
        [f for f in feature_feedbacks if f.shap_impact_mt_ha < 0],
        key=lambda x: x.shap_impact_mt_ha,
    )
    top_positive = sorted(
        [f for f in feature_feedbacks if f.shap_impact_mt_ha >= 0],
        key=lambda x: x.shap_impact_mt_ha,
        reverse=True,
    )

    # 2. Prescriptive Soil Amendments & Quantified Financial Benefits
    raw_recs, _ = compute_advisory_recommendations(
        crop=crop_type,
        ph=val_ph,
        nitrogen=val_n,
        phosphorus=val_p,
        potassium=val_k,
    )

    prescriptive_amendments: List[PrescriptiveAmendmentFinancial] = []
    for rec in raw_recs:
        category = rec["category"]
        gain_mt_ha = float(rec.get("expected_impact_mt_ha", 0.0))
        unit_treatment_cost = AMENDMENT_COST_BENCHMARKS.get(category, 2000.0)
        total_treatment_cost = round(unit_treatment_cost * farm_size_ha, 2)
        proj_rev_gain = round(gain_mt_ha * farm_size_ha * crop_price_per_mt, 2)
        net_benefit = round(proj_rev_gain - total_treatment_cost, 2)
        amendment_roi = (
            round((net_benefit / max(1.0, total_treatment_cost)) * 100.0, 1)
            if total_treatment_cost > 0
            else 0.0
        )

        prescriptive_amendments.append(PrescriptiveAmendmentFinancial(
            category=category,
            current_level=float(rec.get("current_level", 0.0)),
            optimal_range=str(rec.get("optimal_range", "Target")),
            recommendation=str(rec.get("recommendation", "")),
            expected_impact_mt_ha=gain_mt_ha,
            estimated_treatment_cost_total=total_treatment_cost,
            projected_revenue_gain_total=proj_rev_gain,
            net_benefit_total=net_benefit,
            amendment_roi_percentage=amendment_roi,
        ))

    # 3. Diagnostic Narrative Synthesis
    if top_limiting:
        primary_constraint = top_limiting[0].feature
        diag_narrative = (
            f"Financial forecast for {crop_type} is constrained primarily by {primary_constraint} "
            f"causing an estimated gross revenue drag of {abs(top_limiting[0].financial_impact):,.2f} {currency}. "
            f"Addressing controllable soil deficiencies yields up to {sum(p.net_benefit_total for p in prescriptive_amendments):,.2f} {currency} net upside."
        )
    else:
        diag_narrative = (
            f"Optimal agronomic and climatic profile detected for {crop_type}. "
            f"Controllable factors contribute an estimated +{controllable_fin_total:,.2f} {currency} in gross revenue value."
        )

    return SHAPDiagnosticFeedback(
        summary=diag_narrative,
        controllable_impact_total=round(controllable_impact_total, 4),
        exogenous_impact_total=round(exogenous_impact_total, 4),
        controllable_financial_impact=round(controllable_fin_total, 2),
        exogenous_financial_impact=round(exogenous_fin_total, 2),
        top_limiting_factors=top_limiting,
        top_positive_drivers=top_positive,
        feature_attributions=feature_feedbacks,
        prescriptive_amendments=prescriptive_amendments,
    )


# ---------------------------------------------------------------------------
# High-Level Orchestrator
# ---------------------------------------------------------------------------

def execute_financial_forecast(
    request: FinancialForecastRequest,
    audit_id: Optional[uuid.UUID] = None,
) -> UnifiedBusinessResponse:
    """
    Executes end-to-end financial forecasting:
    - Normalizes farm land units
    - Calculates production, gross revenue, itemized costs, net profit, and ROI %
    - Computes SHAP explainability attributions and diagnostic feedback
    - Assembles Unified Business Response JSON contract
    """
    rec_id = audit_id or uuid.uuid4()
    farm_size_ha = normalize_farm_size(request.farm_size, request.farm_size_unit)

    # 1. Financial Metrics Computation
    fin_metrics = calculate_roi_and_financials(
        crop_type=request.crop_type,
        farm_size_ha=farm_size_ha,
        predicted_yield_mt_ha=request.predicted_yield,
        custom_crop_price_per_mt=request.custom_crop_price_per_mt,
        custom_operational_cost_per_ha=request.custom_operational_cost_per_ha,
        currency=request.currency,
    )

    # 2. SHAP Diagnostic Feedback Integration
    shap_feedback = build_shap_diagnostic_feedback(
        crop_type=request.crop_type,
        farm_size_ha=farm_size_ha,
        crop_price_per_mt=fin_metrics.crop_price_per_mt,
        nitrogen=request.nitrogen,
        phosphorus=request.phosphorus,
        potassium=request.potassium,
        ph=request.ph,
        rainfall=request.rainfall,
        temp_max=request.temp_max,
        currency=request.currency,
    )

    # 3. Farm Cultivation Summary
    farm_summary = FarmSummary(
        crop_type=request.crop_type,
        farm_size_ha=farm_size_ha,
        predicted_yield_mt_ha=request.predicted_yield,
        total_production_mt=round(farm_size_ha * request.predicted_yield, 3),
        district=request.district,
        field_name=request.field_name,
    )

    # 4. Construct Unified Business Response JSON
    return UnifiedBusinessResponse(
        status="success",
        audit_id=rec_id,
        timestamp=datetime.now(timezone.utc),
        farm_summary=farm_summary,
        financial_metrics=fin_metrics,
        shap_diagnostic_feedback=shap_feedback,
        persisted=True,
        user_id=request.user_id,
    )

