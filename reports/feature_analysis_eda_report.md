# 📈 Day 8: Agronomic Feature Analysis & EDA Report

**Generated**: 2026-10-21 15:14:17  
**Scope**: Day 8 Milestone — Feature Correlation Screening & Agronomic Response Curves  
**Records Analyzed**: 225 across 15 districts and 5 crops  

---

## 🔍 Key Findings & Feature Correlation Insights

| Rank | Feature | Pearson $r$ | Agronomic Meaning & Model Relevance |
| :---: | :--- | :---: | :--- |
| 1 | `crop_type_Maize` | **+0.653** | Strong positive driver for vegetative canopy biomass |
| 2 | `crop_type_Cotton` | **-0.606** | Critical thermal/moisture deficit penalty |
| 3 | `cold_stress_days` | **-0.473** | Critical thermal/moisture deficit penalty |
| 4 | `crop_type_Soybean` | **-0.416** | Critical thermal/moisture deficit penalty |
| 5 | `high_humidity_days` | **-0.217** | Nutrient stoichiometry balance indicator |
| 6 | `rainy_days_count` | **-0.215** | Nutrient stoichiometry balance indicator |
| 7 | `crop_type_Wheat` | **+0.206** | Nutrient stoichiometry balance indicator |
| 8 | `hydro_thermal_stress_index` | **-0.195** | Nutrient stoichiometry balance indicator |
| 9 | `mean_relative_humidity` | **-0.193** | Nutrient stoichiometry balance indicator |
| 10 | `photothermal_ratio` | **+0.190** | Nutrient stoichiometry balance indicator |
| 11 | `cumulative_rainfall_mm` | **-0.183** | Nutrient stoichiometry balance indicator |
| 12 | `extreme_temp_max` | **-0.182** | Nutrient stoichiometry balance indicator |
| 13 | `mean_daily_rainfall_mm` | **-0.166** | Nutrient stoichiometry balance indicator |
| 14 | `rainfall_per_season_day` | **-0.166** | Nutrient stoichiometry balance indicator |
| 15 | `heavy_rain_days_count` | **-0.164** | Nutrient stoichiometry balance indicator |

---

## 📊 Generated Visual Artifacts
1. `reports/figures/02_npk_stoichiometry_vs_yield.png`: Stoichiometric response curves.
2. `reports/figures/02_hydrothermal_stress_surface.png`: Joint VPD and moisture stress landscape.
3. `reports/figures/02_feature_correlation_ranking.png`: Top 15 ranked feature attributions.

---
✅ **Ready for Day 9**: Features selected for Spatial Block Group-KFold Model Training.