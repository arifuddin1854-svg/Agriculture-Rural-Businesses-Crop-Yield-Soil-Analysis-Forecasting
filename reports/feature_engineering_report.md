# 🧪 Day 7: Agronomic Feature Engineering Quality Audit Report

**Generated**: 2026-10-21 15:12:27  
**Scope**: Day 7 Milestone — Agronomic Interaction & Stoichiometric Feature Engineering  
**Input Master Dataset**: `data\processed\unified_agri_dataset.csv`  
**Total Records**: 225 rows  
**Feature Count Expansion**: 35 base features ➔ **63 total features** (+28 engineered)  
**Data Integrity**: 0 missing values across all engineered columns  

---

## 🧱 Newly Derived Agronomic Features Breakdown

| Feature Name | Category | Agronomic Formula / Description | Mean Value | Range [Min, Max] |
| :--- | :--- | :--- | :---: | :---: |
| `npk_total_kg_ha` | Soil Stoichiometry | Domain engineered metric | 317.008 | [199.29, 385.86] |
| `n_ratio` | Soil Stoichiometry | Domain engineered metric | 0.3599 | [0.1931, 0.5392] |
| `p_ratio` | Soil Stoichiometry | Domain engineered metric | 0.1412 | [0.0612, 0.2122] |
| `k_ratio` | Soil Stoichiometry | Domain engineered metric | 0.499 | [0.3385, 0.6471] |
| `n_to_p_ratio` | Soil Stoichiometry | Domain engineered metric | 2.9169 | [1.0608, 6.0051] |
| `n_to_k_ratio` | Soil Stoichiometry | Domain engineered metric | 0.7746 | [0.3091, 1.4558] |
| `p_to_k_ratio` | Soil Stoichiometry | Domain engineered metric | 0.2992 | [0.107, 0.5236] |
| `npk_imbalance_index` | Soil Stoichiometry | Domain engineered metric | 0.349 | [0.1885, 0.4971] |
| `soil_ph_deviation_from_neutral` | Soil Stoichiometry | Domain engineered metric | 0.5027 | [0.06, 1.71] |
| `soil_acidity_stress` | Hydrothermal Stress | Domain engineered metric | 0.1367 | [0.0, 1.21] |
| `soil_alkalinity_stress` | Hydrothermal Stress | Domain engineered metric | 0.0 | [0.0, 0.0] |
| `som_buffering_capacity` | Soil Stoichiometry | Domain engineered metric | 2.2566 | [1.1623, 3.3196] |
| `hydro_thermal_stress_index` | Hydrothermal Stress | Domain engineered metric | 18.6723 | [2.5356, 40.3354] |
| `rainfall_per_season_day` | Encoded Categorical | Domain engineered metric | 4.8822 | [0.5626, 10.1954] |
| `heat_stress_frequency` | Hydrothermal Stress | Domain engineered metric | 0.4014 | [0.0, 0.7974] |
| `drought_severity_index` | Hydrothermal Stress | Domain engineered metric | 0.395 | [0.0177, 5.1745] |
| `photothermal_ratio` | Soil Stoichiometry | Domain engineered metric | 88.0919 | [66.7132, 107.5845] |
| `atmospheric_aridity_index` | Soil Stoichiometry | Domain engineered metric | 16.4762 | [13.5036, 20.8125] |
| `crop_type_Cotton` | Encoded Categorical | Domain engineered metric | 0.1733 | [0.0, 1.0] |
| `crop_type_Maize` | Encoded Categorical | Domain engineered metric | 0.1956 | [0.0, 1.0] |
| `crop_type_Rice` | Encoded Categorical | Domain engineered metric | 0.2356 | [0.0, 1.0] |
| `crop_type_Soybean` | Encoded Categorical | Domain engineered metric | 0.1778 | [0.0, 1.0] |
| `crop_type_Wheat` | Encoded Categorical | Domain engineered metric | 0.2178 | [0.0, 1.0] |
| `soil_type_Alluvial` | Encoded Categorical | Domain engineered metric | 0.1333 | [0.0, 1.0] |
| `soil_type_Black Soil` | Encoded Categorical | Domain engineered metric | 0.1333 | [0.0, 1.0] |
| `soil_type_Clay Loam` | Encoded Categorical | Domain engineered metric | 0.4 | [0.0, 1.0] |
| `soil_type_Laterite` | Encoded Categorical | Domain engineered metric | 0.2667 | [0.0, 1.0] |
| `soil_type_Red Sandy Loam` | Encoded Categorical | Domain engineered metric | 0.0667 | [0.0, 1.0] |

---

## 🎯 Downstream Hand-off Readiness
- ✅ **Feature Store**: Saved to `data/processed/train_features.csv` and `data/processed/train_features.parquet`.
- ✅ **Spatial Integrity**: Retained `district_id` and `year` unencoded for spatial block cross-validation.
- ✅ **Ready for Day 8**: Advanced correlation ranking and Notebook 02 generation.