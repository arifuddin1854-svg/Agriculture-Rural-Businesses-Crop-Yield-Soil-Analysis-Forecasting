"""
Agronomic Crop Recommendation Module.
Evaluates soil NPK, pH, and environmental indicators against optimal crop lookup tables
to rank and recommend top target crops with prescriptive diagnostic notes.
"""

from typing import Any, Dict, List, Optional
import heapq

# Agronomic optimal parameter lookup table for primary agricultural crops
# Values represent scientifically established (min, max, ideal_center) agronomic thresholds.
OPTIMAL_CROP_PROFILES = {
    "Rice": {
        "n": (80.0, 120.0, 100.0),
        "p": (35.0, 55.0, 45.0),
        "k": (35.0, 55.0, 45.0),
        "ph": (5.5, 7.0, 6.2),
        "temp": (20.0, 35.0, 26.0),
        "rain": (1000.0, 2500.0, 1500.0),
        "category": "Cereal / Grain",
    },
    "Wheat": {
        "n": (100.0, 140.0, 120.0),
        "p": (40.0, 65.0, 52.0),
        "k": (40.0, 65.0, 50.0),
        "ph": (6.0, 7.5, 6.6),
        "temp": (15.0, 28.0, 22.0),
        "rain": (450.0, 850.0, 650.0),
        "category": "Cereal / Grain",
    },
    "Maize": {
        "n": (110.0, 150.0, 130.0),
        "p": (45.0, 70.0, 58.0),
        "k": (55.0, 85.0, 70.0),
        "ph": (5.8, 7.2, 6.5),
        "temp": (18.0, 32.0, 25.0),
        "rain": (500.0, 950.0, 750.0),
        "category": "Cereal / Coarse Grain",
    },
    "Soybean": {
        "n": (20.0, 50.0, 35.0),  # Legume fixates atmospheric nitrogen
        "p": (45.0, 75.0, 60.0),
        "k": (40.0, 65.0, 52.0),
        "ph": (6.0, 7.0, 6.5),
        "temp": (20.0, 30.0, 25.0),
        "rain": (600.0, 1000.0, 800.0),
        "category": "Legume / Oilseed",
    },
    "Cotton": {
        "n": (90.0, 130.0, 110.0),
        "p": (40.0, 65.0, 52.0),
        "k": (40.0, 70.0, 55.0),
        "ph": (6.0, 8.0, 7.0),
        "temp": (22.0, 36.0, 28.0),
        "rain": (500.0, 1100.0, 750.0),
        "category": "Cash / Fiber Crop",
    },
    "Chickpea": {
        "n": (20.0, 45.0, 30.0),
        "p": (40.0, 70.0, 55.0),
        "k": (20.0, 50.0, 35.0),
        "ph": (6.0, 7.8, 6.8),
        "temp": (16.0, 26.0, 21.0),
        "rain": (350.0, 700.0, 500.0),
        "category": "Pulse / Legume",
    },
    "Kidney Beans": {
        "n": (20.0, 45.0, 32.0),
        "p": (55.0, 80.0, 68.0),
        "k": (20.0, 45.0, 32.0),
        "ph": (5.7, 6.8, 6.2),
        "temp": (15.0, 28.0, 22.0),
        "rain": (600.0, 1200.0, 900.0),
        "category": "Pulse / Legume",
    },
    "Lentil": {
        "n": (20.0, 40.0, 28.0),
        "p": (30.0, 55.0, 42.0),
        "k": (20.0, 40.0, 30.0),
        "ph": (6.0, 7.5, 6.7),
        "temp": (14.0, 25.0, 19.0),
        "rain": (350.0, 650.0, 480.0),
        "category": "Pulse / Legume",
    },
    "Banana": {
        "n": (170.0, 260.0, 210.0),
        "p": (70.0, 110.0, 90.0),
        "k": (180.0, 300.0, 240.0),
        "ph": (6.0, 7.5, 6.7),
        "temp": (22.0, 35.0, 28.0),
        "rain": (1200.0, 2200.0, 1700.0),
        "category": "Horticultural / Fruit",
    },
    "Mango": {
        "n": (80.0, 130.0, 105.0),
        "p": (40.0, 65.0, 52.0),
        "k": (80.0, 130.0, 105.0),
        "ph": (5.5, 7.5, 6.5),
        "temp": (22.0, 35.0, 28.0),
        "rain": (750.0, 1500.0, 1100.0),
        "category": "Horticultural / Fruit",
    },
    "Grapes": {
        "n": (50.0, 90.0, 70.0),
        "p": (40.0, 65.0, 52.0),
        "k": (100.0, 160.0, 130.0),
        "ph": (6.2, 7.5, 6.8),
        "temp": (18.0, 32.0, 25.0),
        "rain": (500.0, 850.0, 650.0),
        "category": "Horticultural / Fruit",
    },
    "Mustard": {
        "n": (70.0, 100.0, 85.0),
        "p": (30.0, 50.0, 40.0),
        "k": (30.0, 50.0, 40.0),
        "ph": (6.0, 7.5, 6.8),
        "temp": (12.0, 25.0, 18.0),
        "rain": (300.0, 600.0, 450.0),
        "category": "Oilseed",
    },
}

# Pre-computed static optimal range representations to avoid repetitive runtime string formatting
PRECOMPUTED_OPTIMAL_RANGES = {
    crop: {
        "nitrogen_kg_ha": f"{prof['n'][0]:.0f} - {prof['n'][1]:.0f}",
        "phosphorus_kg_ha": f"{prof['p'][0]:.0f} - {prof['p'][1]:.0f}",
        "potassium_kg_ha": f"{prof['k'][0]:.0f} - {prof['k'][1]:.0f}",
        "ph": f"{prof['ph'][0]:.1f} - {prof['ph'][1]:.1f}",
    }
    for crop, prof in OPTIMAL_CROP_PROFILES.items()
}


def _calculate_factor_penalty(val: float, low: float, high: float, ideal: float, tolerance_scale: float) -> float:
    """
    Computes a normalized penalty between 0.0 (perfect) and 1.0 (unsuitable).
    If within [low, high], penalty is zero or minimal based on distance to ideal center.
    """
    if low <= val <= high:
        return 0.15 * abs(val - ideal) / max(1e-5, (high - low))
    elif val < low:
        dist = low - val
        return min(1.0, 0.2 + (dist / max(1e-5, low * tolerance_scale)))
    else:
        dist = val - high
        return min(1.0, 0.2 + (dist / max(1e-5, high * tolerance_scale)))


def evaluate_crop_suitability(
    crop_name: str,
    profile: Dict[str, Any],
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    rainfall: Optional[float] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluates soil parameters against a single crop's optimal lookup profile.
    """
    (n_min, n_max, n_ideal) = profile["n"]
    (p_min, p_max, p_ideal) = profile["p"]
    (k_min, k_max, k_ideal) = profile["k"]
    (ph_min, ph_max, ph_ideal) = profile["ph"]

    # Calculate individual parameter penalties
    n_pen = _calculate_factor_penalty(nitrogen, n_min, n_max, n_ideal, tolerance_scale=0.6)
    p_pen = _calculate_factor_penalty(phosphorus, p_min, p_max, p_ideal, tolerance_scale=0.6)
    k_pen = _calculate_factor_penalty(potassium, k_min, k_max, k_ideal, tolerance_scale=0.6)
    ph_pen = _calculate_factor_penalty(ph, ph_min, ph_max, ph_ideal, tolerance_scale=0.3)

    if rainfall is not None and temperature is not None:
        r_min, r_max, r_ideal = profile["rain"]
        t_min, t_max, t_ideal = profile["temp"]
        r_pen = _calculate_factor_penalty(rainfall, r_min, r_max, r_ideal, tolerance_scale=0.5)
        t_pen = _calculate_factor_penalty(temperature, t_min, t_max, t_ideal, tolerance_scale=0.4)
        composite_penalty = (
            0.20 * n_pen + 0.20 * p_pen + 0.20 * k_pen + 0.20 * ph_pen + 0.10 * r_pen + 0.10 * t_pen
        )
    elif rainfall is not None:
        r_min, r_max, r_ideal = profile["rain"]
        r_pen = _calculate_factor_penalty(rainfall, r_min, r_max, r_ideal, tolerance_scale=0.5)
        composite_penalty = (
            0.22 * n_pen + 0.22 * p_pen + 0.22 * k_pen + 0.22 * ph_pen + 0.12 * r_pen
        )
    elif temperature is not None:
        t_min, t_max, t_ideal = profile["temp"]
        t_pen = _calculate_factor_penalty(temperature, t_min, t_max, t_ideal, tolerance_scale=0.4)
        composite_penalty = (
            0.22 * n_pen + 0.22 * p_pen + 0.22 * k_pen + 0.22 * ph_pen + 0.12 * t_pen
        )
    else:
        composite_penalty = 0.25 * (n_pen + p_pen + k_pen + ph_pen)

    suitability_score = round(max(0.0, min(100.0, (1.0 - composite_penalty) * 100.0)), 2)

    # Classify match status
    if suitability_score >= 82.0:
        match_status = "Highly Suitable"
    elif suitability_score >= 65.0:
        match_status = "Moderately Suitable"
    else:
        match_status = "Marginal"

    # Generate diagnostic notes
    diagnostic_notes = []
    if ph_min <= ph <= ph_max:
        diagnostic_notes.append(f"Soil pH ({ph:.1f}) is in the optimal range ({ph_min:.1f} - {ph_max:.1f}).")
    elif ph < ph_min:
        diagnostic_notes.append(f"Soil pH ({ph:.1f}) is acidic for {crop_name}; target range is {ph_min:.1f} - {ph_max:.1f}.")
    else:
        diagnostic_notes.append(f"Soil pH ({ph:.1f}) is alkaline for {crop_name}; target range is {ph_min:.1f} - {ph_max:.1f}.")

    if n_min <= nitrogen <= n_max:
        diagnostic_notes.append(f"Nitrogen ({nitrogen:.0f} kg/ha) matches optimal requirement ({n_min:.0f}-{n_max:.0f} kg/ha).")
    elif nitrogen < n_min:
        diagnostic_notes.append(f"Nitrogen deficit of {n_min - nitrogen:.0f} kg/ha for target crop yield.")
    else:
        diagnostic_notes.append(f"Nitrogen surplus of {nitrogen - n_max:.0f} kg/ha above standard crop intake.")

    if p_min <= phosphorus <= p_max and k_min <= potassium <= k_max:
        diagnostic_notes.append(f"P & K nutrient balance is well-aligned with {crop_name} uptake.")

    return {
        "crop": crop_name,
        "suitability_score": suitability_score,
        "match_status": match_status,
        "optimal_ranges": PRECOMPUTED_OPTIMAL_RANGES[crop_name],
        "diagnostic_notes": diagnostic_notes,
    }


def recommend_top_crops(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    rainfall: Optional[float] = None,
    temperature: Optional[float] = None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Evaluates all crop profiles in the lookup table and returns top_k ranked candidates.
    Uses heapq.nlargest for O(N log k) top candidates extraction.
    """
    evaluations = [
        evaluate_crop_suitability(
            crop_name=crop_name,
            profile=profile,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            ph=ph,
            rainfall=rainfall,
            temperature=temperature,
        )
        for crop_name, profile in OPTIMAL_CROP_PROFILES.items()
    ]

    # Efficiently pick top_k by suitability_score
    top_candidates = heapq.nlargest(top_k, evaluations, key=lambda x: x["suitability_score"])

    # Assign ranks
    for rank, item in enumerate(top_candidates, start=1):
        item["rank"] = rank

    return top_candidates
