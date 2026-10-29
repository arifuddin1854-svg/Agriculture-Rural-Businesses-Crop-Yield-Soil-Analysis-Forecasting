# 🌦️ Phenological Weather Aggregation Quality Audit Report

**Generated**: 2026-10-10 10:07:18  
**Scope**: Member 1 (Day 3 Milestone) — Phenological Daily Weather Aggregation  
**Coverage**: 15 Districts, 5 Years (2019–2023)  

---

## 📊 Executive Summary of Key Phenological Metrics

| Crop Type | Records | Mean Rainfall (mm) | Mean GDD (°C-day) | Mean T_max (°C) | Heat Stress Days (Tmax ≥ 35°C) | Mean VPD (kPa) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cotton** | 39 | 1123.9 ± 258.5 | 3282.4 ± 274.5 | 33.7°C | 79.0 days | 1.37 kPa |
| **Maize** | 44 | 1042.2 ± 233.1 | 3315.6 ± 187.7 | 35.0°C | 100.2 days | 1.39 kPa |
| **Rice** | 53 | 1089.5 ± 220.5 | 3072.7 ± 215.2 | 33.7°C | 89.2 days | 1.28 kPa |
| **Soybean** | 40 | 996.1 ± 239.0 | 2618.4 ± 140.2 | 35.2°C | 89.7 days | 1.29 kPa |
| **Wheat** | 49 | 237.7 ± 105.7 | 2431.7 ± 400.2 | 25.6°C | 9.2 days | 1.27 kPa |

---

## 🔬 Agronomic Indicator Formulations

1. **Growing Degree Days (GDD)**:
   $$\text{Daily GDD} = \max\left(0, \frac{\min(T_{\max}, T_{\text{upper}}) + \max(T_{\min}, T_{\text{base}})}{2} - T_{\text{base}}\right)$$

2. **Vapor Pressure Deficit (VPD)** (FAO-56 Penman-Monteith Formulation):
   $$e_s = \frac{0.61078 e^{\frac{17.27 T_{\max}}{T_{\max} + 237.3}} + 0.61078 e^{\frac{17.27 T_{\min}}{T_{\min} + 237.3}}}{2}$$
   $$\text{VPD} = e_s \times \left(1 - \frac{\text{RH}}{100}\right)$$

3. **Heat Stress Days**: Count of days with $T_{\max} \ge 35^\circ\text{C}$ (or crop-specific threshold).
4. **Consecutive Dry Days**: Longest contiguous dry spell (precipitation $< 1.0\text{ mm}$).

---
✅ **Deliverable Integrity**: Processed dataset exported with zero missing values and verified agronomic ranges.