# 📈 Exploratory Data Analysis (EDA) Summary Report
**Author**: Syed Abdus Sabur (Member 1: Data Integration & Data Engineer)  
**Milestone**: Day 5 Deliverable  
**Master Dataset**: `data/processed/unified_agri_dataset.csv` (225 records, 35 features)  

---

## 🔍 Key Agronomic Findings

### 1. Target Yield Distribution Across Commodities
- **Maize** exhibits the highest average yield (**5.63 MT/Ha**), followed by **Wheat** (**4.39 MT/Ha**) and **Rice** (**4.12 MT/Ha**).
- **Cotton** (**2.11 MT/Ha**) and **Soybean** (**2.68 MT/Ha**) operate in lower volumetric ranges with higher coefficient of variation.

### 2. Soil Chemistry & Macronutrient Dynamics
- **Soil pH**: Optimal yield clustering occurs around neutral pH range (6.2 – 7.2). Both severe acidity (< 5.5) and high alkalinity (> 8.0) suppress nutrient availability.
- **Nitrogen (N) & Organic Matter (SOM %)**: Display strong positive correlations with vegetative biomass and grain yield.

### 3. Agro-Meteorological Drivers
- **Monsoon Rainfall**: Rice, Cotton, and Maize yields correlate directly with cumulative growing-season precipitation (102.4 mm to 1743.3 mm).
- **Heat Stress Shock ($T_{\max} \ge 35^\circ\text{C}$)**: Extended heat stress during flowering windows suppresses fertilization and grain weight.

### 4. Spatial Autocorrelation & Regional Variability
- High spatial clustering observed across adjacent districts, confirming the necessity of **Spatial GroupKFold cross-validation** (grouping by `district_id`) to prevent optimistic atmospheric leakage.

### 5. Multi-Year Climate Shocks
- Significant yield dip observed in **2021** due to regional precipitation deficit, followed by strong rebound in 2022–2023.

---

## 🖼️ Generated Visual Artifacts

1. `reports/figures/yield_distributions_by_crop.png`
2. `reports/figures/soil_chemistry_correlation_heatmap.png`
3. `reports/figures/weather_stress_vs_yield.png`
4. `reports/figures/district_spatial_yield_distribution.png`
5. `reports/figures/temporal_yield_trends_2019_2023.png`

---
✅ **Handoff Status**: Dataset, figures, and EDA notebook ready for Member 2 (Agronomic Feature Engineering).
