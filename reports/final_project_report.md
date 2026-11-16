# 🌾 AXLERO AGRI-TECH PLATFORM: COMPREHENSIVE PROJECT REPORT
## Project 3: Agriculture & Rural Businesses – Crop Yield & Soil Analysis Forecasting

**Author / Lead Data Integration & ML Engineer**: Syed Abdus Sabur  
**Organization**: Axlero Agri-Tech Initiative  
**Repository Branch**: `Syed-Abdus-Sabur`  
**Date**: November 2026  
**System Status**: Production-Ready | 20/20 System Health Checks Passing | 78/78 Unit Tests Passing  
**Champion Model**: Spatial Random Forest Regressor ($R^2 = 0.8033$, $\text{RMSE} = 0.5882\text{ MT/Ha}$)  

---

## 1. Executive Summary

Modern agriculture faces severe yield volatility driven by unpredictable climatic shifts, spatial soil heterogeneity, and unoptimized fertilizer application. Traditional agronomic management often relies on regional averages or historical intuition, leading to either nutrient depletion or chemical over-application, soil acidification, and depressed rural enterprise profit margins.

The **Axlero Crop Yield & Soil Analysis Forecasting Platform** is an enterprise-grade, end-to-end decision support system designed to bridge data science and rural agribusiness. By fusing multi-source telemetry—spanning five years of historical crop yields, laboratory soil chemistry (N-P-K, pH, organic matter, salinity), daily meteorology (temperature, precipitation, vapor pressure deficit, solar radiation), and dekadal satellite spectral vegetation indices (NDVI, NDRE, EVI, NDWI)—the platform delivers:

1. **Leak-Free Spatial Forecasting**: High-accuracy yield predictions ($R^2 = 0.8033$, $\text{MAPE} = 12.97\%$) evaluated using spatial block Group-KFold cross-validation to eliminate spatial autocorrelation leakage.
2. **Explainable AI (TreeSHAP)**: Factor disentanglement separating uncontrollable climate forces from farm-controllable soil interventions.
3. **Prescriptive Agronomic Advisory**: Automated counterfactual optimization prescribing exact amendments (e.g. Agricultural Limestone $\text{CaCO}_3$, synchronized Nitrogen top-dressing) with quantified dosage and yield upside ($+6.5\%\text{ to }+14.2\%$).
4. **Agri-Business Financial ROI & Portfolio Optimization**: Markowitz Modern Portfolio Theory (MPT) crop acreage optimization (Sharpe ratio $8.059$), prescriptive ROI modeling ($+154.4\%$ ROI for nitrogen interventions), and regional cooperative credit risk scoring.
5. **Full-Stack Operational Delivery**: A high-performance Python REST API ($<60\text{ ms}$ p95 latency) accompanied by a modern, reactive single-page dashboard featuring 1-click presets, dynamic nutrient badges, live auto-forecasting, covariate drift monitoring (PSI/KS tests), and executive PDF export.

---

## 2. System Context, Problem Statement & Objectives

### 2.1 The Core Problem
Organic farming businesses and agricultural trading platforms operate under high uncertainty:
* **Temporal Disparity**: Weather telemetry is recorded continuously at daily intervals, whereas crop harvests occur on annual or seasonal cycles.
* **Spatial Leakage**: Conventional $K$-fold cross-validation inflates model performance by allowing neighboring farms with identical weather to appear in both training and test sets.
* **Opaque Black-Box Models**: Pure deep learning or ensemble predictions lack agronomic interpretability, offering no clear path for farmers to rectify soil imbalances.
* **Economic Disconnect**: Yield predictions are rarely connected to input fertilizer costs, commodity pricing, and financial profit-and-loss margins.

### 2.2 System Objectives & Success Criteria
| Metric Category | Target Objective | Achieved Result | Evaluation Status |
| :--- | :--- | :--- | :---: |
| **Data Fusion** | Merge multi-modal datasets with distinct temporal and spatial granularities (daily weather, dekadal satellite, static soil, seasonal yields). | 225 clean records fused across 15 districts, 5 crops, and 5 years (35 base features, 57 engineered features, 0 missing records). | ✅ **Exceeded** |
| **Regression Accuracy** | Optimize Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE). | Spatial $R^2 = 0.8033$, Spatial $\text{RMSE} = 0.5882\text{ MT/Ha}$, Spatial $\text{MAPE} = 12.97\%$. | ✅ **Exceeded** |
| **Spatial Generalizability** | Prevent spatial autocorrelation leakage using geographic block validation. | 5-Fold District-wise Spatial Group-KFold CV and Leave-One-Year-Out (LOYO) cross-validation implemented. | ✅ **Exceeded** |
| **Feature Attribution** | Disentangle controllable soil factors from exogenous climate drivers using SHAP. | Full TreeSHAP global and local attribution isolating pH and N-P-K impact from solar and temperature drivers. | ✅ **Exceeded** |
| **Actionable Prescriptions** | Provide prescriptive counterfactual recommendations with exact dosage. | Automated advisor with specific chemical inputs, target metrics, and expected yield gains. | ✅ **Exceeded** |
| **Production Architecture** | REST API service, interactive dashboard, and automated test suite. | REST API ($<60\text{ ms}$ latency), responsive web dashboard, 78/78 passing tests, 20/20 health checks. | ✅ **Exceeded** |

---

## 3. Four-Week Development Lifecycle & Technical Execution

### Week 1: Geospatial & Weather Data Integration
* **Data Ingestion & Cleaning**: Ingested raw crop yield histories (225 records), static soil chemistry (15 agro-climatic districts), daily meteorology (27,390 rows), and dekadal satellite spectral readings (2,700 rows).
* **Validation & Quality Control**: Automated range validation, type casting, duplicate elimination, and physical boundary enforcement (e.g. soil pH $\in [3.5, 9.5]$, temperatures $\in [-10^\circ\text{C}, 55^\circ\text{C}]$).
* **Growing-Season Weather Aggregation**: Transformed daily time-series into agronomic growing-season indicators aligned with biological crop calendars (110–165 days):
  * Cumulative Rainfall ($P_{\text{cum}}$) and Rainfall Intensity
  * Growing Degree Days ($\text{GDD} = \sum \max(0, T_{\text{mean}} - 10)$)
  * Extreme Heat Days ($T_{\text{max}} \ge 35^\circ\text{C}$) during flowering and grain fill
  * Mean Atmospheric Vapor Pressure Deficit ($\text{VPD}$) and Diurnal Temperature Range
* **Geospatial Fusion**: Executed leak-free relational joins on `['district_id', 'year', 'crop_type']`, generating the `unified_agri_dataset` (225 rows, 35 columns).

### Week 2: Exploratory Data Analysis & Spatial-Temporal Feature Engineering
* **Bivariate Agronomic Trends**: Identified non-linear, concave relationships between soil pH and yield, verifying Liebig’s Law of the Minimum (yield is dictated by the scarcest nutrient rather than total resources).
* **Domain Feature Formulation**:
  * **NPK Stoichiometric Balance Index**: Quantifies Euclidean deviation from the optimal $4:2:1$ ($N:P_2O_5:K_2O$) ratio:
    $$\text{NPK\_Imbalance} = \sqrt{\left(\frac{N}{N+P+K} - 0.571\right)^2 + \left(\frac{P}{N+P+K} - 0.286\right)^2 + \left(\frac{K}{N+P+K} - 0.143\right)^2}$$
  * **Soil Reaction Stress Index**: Measures distance from neutral $6.5\text{ pH}$ ($|\text{pH} - 6.5|$).
  * **Hydro-Thermal Stress Interaction**: Interaction index ($P_{\text{cum}} \times T_{\text{mean}}$) capturing compound drought-heat pressure.
  * **Atmospheric Evaporative Demand**: Saturated vs. actual vapor pressure deficit:
    $$\text{VPD} = e_s(T) \times \left(1 - \frac{\text{RH}}{100}\right)$$
* **Feature Matrix Pipeline**: One-hot categorical encoding for crop and soil types, resulting in a finalized 57-feature matrix stored in `data/processed/train_features.csv`.

### Week 3: Supervised Regression Modeling & Leak-Free Validation
* **The Spatial Leakage Challenge**: Standard random $K$-fold cross-validation artificially inflates model accuracy because spatial autocorrelation causes neighboring farms with identical weather regimes to leak across splits.
* **Leak-Free Validation Strategies**:
  1. **Spatial Block Group-KFold (5 Folds by District)**: Farms within the same agro-climatic district are strictly grouped together, ensuring test districts are completely unseen during training.
  2. **Leave-One-Year-Out (LOYO) Temporal CV**: Iteratively holds out individual calendar years (2019–2023) to evaluate temporal generalizability under extreme climate shocks (e.g. the 2021 severe heatwave anomaly).
* **Supervised Model Benchmark Comparison**:
  | Model Algorithm | Spatial $R^2$ | Spatial RMSE (MT/Ha) | Spatial MAE (MT/Ha) | Spatial MAPE (%) | Status |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Ridge Regression** | 0.6771 | 0.7535 | 0.6026 | 17.88% | Baseline |
  | **Random Forest Regressor** ⭐ | **0.8033** | **0.5882** | **0.4558** | **12.97%** | **Production Champion** |
  | **Gradient Boosting Regressor** | 0.7634 | 0.6461 | 0.5017 | 14.02% | Competitive |
  | **HistGradientBoosting** | 0.7717 | 0.6332 | 0.5087 | 14.65% | Competitive |
  | **Deep Learning MLP Benchmark** | 0.3063 | 1.1070 | 0.8870 | 28.50% | Overfitting on Tabular |

### Week 4: Explainability (SHAP), Prescriptive Advisor & Full-Stack Platform
* **TreeSHAP Global Factor Attribution**: Deployed `shap.TreeExplainer` on the champion Random Forest model to calculate exact Shapley values.
* **Factor Disentanglement**:
  * **Exogenous Macro-Climate**: Cumulative solar radiation, relative humidity, and heat stress accounted for $0.6538$ cumulative SHAP impact.
  * **Spatial & Botanical Baselines**: Crop species genetics and district baselines accounted for $1.0568$ cumulative SHAP impact.
  * **Controllable Soil Chemistry**: Soil pH, Nitrogen, Phosphorus, and Potassium accounted for $0.0890$ cumulative impact.
* **Prescriptive Counterfactual Engine (`src/advisor.py`)**: Built an optimization module that computes marginal yield derivatives with respect to controllable parameters, prescribing:
  * Agricultural Limestone ($\text{CaCO}_3$) for acidic soils ($pH < 5.8$): $+0.25\text{ to }+0.58\text{ MT/Ha}$ upside ($+6.5\%$).
  * Split Nitrogen Top-Dressing for nitrogen-deficient fields: $+0.42\text{ MT/Ha}$ upside ($+11.2\%$).
  * Organic Biochar & Green Manure for degraded soils ($SOM < 1.5\%$): $+0.28\text{ MT/Ha}$ upside.
* **Probabilistic Uncertainty Estimation**: Formulated $95\%$ Confidence Intervals ($\pm 0.68\text{ MT/Ha}$) based on spatial model residual variance distributions.

---

## 4. Advanced System Capabilities & Extensions

### 4.1 Remote Sensing & Dekadal Satellite Integration
* Ingested multi-spectral satellite telemetry (Sentinel-2 bands B02, B04, B05, B08, B11).
* Engineered dekadal vegetative indices:
  * **Normalized Difference Vegetation Index (NDVI)**: Green canopy vigor.
  * **Red Edge Chlorophyll Index (NDRE)**: Mid-to-late season nitrogen status.
  * **Enhanced Vegetation Index (EVI)**: Atmospheric and soil-background corrected biomass.
  * **Normalized Difference Water Index (NDWI)**: Crop canopy moisture content.
* Constructed canopy time-series trajectories ($\text{NDVI}_{\text{max}}$, $\text{NDVI}_{\text{auc}}$, $\text{NDVI}_{\text{slope}}$) for pre-harvest growth tracking.

### 4.2 CMIP6 Climate Resilience & Scenario Simulation
* Integrated IPCC Sixth Assessment Report (AR6) climate change projections:
  * **SSP2-4.5 (Middle of the Road)**: $+1.5^\circ\text{C}$ warming, $+35\%$ heat stress frequency.
  * **SSP5-8.5 (High Emissions)**: $+2.8^\circ\text{C}$ warming, $+75\%$ heat stress frequency.
* Projected regional yield impacts, revealing that unadapted Maize and Wheat suffer $-14.2\%$ to $-22.5\%$ yield reductions under SSP5-8.5, whereas drought-tolerant crops and optimized soil buffering reduce vulnerability by $60\%$.

### 4.3 Rural Agricultural Economics & Markowitz Portfolio Optimization
* **Farm Profit & Loss Modeling**: Integrates commodity market prices ($\$265/\text{MT}$ for Wheat, $\$340/\text{MT}$ for Rice, $\$210/\text{MT}$ for Maize, $\$480/\text{MT}$ for Soybean, $\$750/\text{MT}$ for Cotton) and operational cultivation costs.
* **Prescriptive Amendment Financial ROI**:
  | Prescriptive Intervention | Investment Cost ($/Ha) | Yield Gain (MT/Ha) | Revenue Upside ($/Ha) | Net Profit Gain ($/Ha) | Financial ROI (%) | Agronomic Rating |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **Synchronized Nitrogen Top-Dressing** | $43.75 | +0.42 | +$111.30 | +$67.55 | **+154.4%** | `EXCELLENT` |
  | **Apply Agricultural Limestone (CaCO3)** | $90.00 | +0.58 | +$153.70 | +$63.70 | **+70.8%** | `POSITIVE` |
  | **Biochar & Green Manure Incorporation** | $120.00 | +0.28 | +$74.20 | -$45.80 | **-38.2%** | `SUBOPTIMAL` (Multi-Year) |
* **Modern Portfolio Theory (Markowitz Frontier)**: Applied quadratic optimization to maximize regional cooperative Sharpe ratio ($8.059$):
  * **Optimal Acreage Distribution**: Soybean ($51.4\%$), Cotton ($33.6\%$), Wheat ($5.0\%$), Rice ($5.0\%$), Maize ($5.0\%$).
* **Regional Cooperative Credit Risk Scoring**: Evaluated 15 agricultural districts across mean profit, volatility, and coefficient of variation, classifying regional holdings into institutional credit ratings (`AA` High Quality vs. `A` Upper Medium Grade).

### 4.4 Covariate Drift & MLOps Governance
* Continuous distribution tracking on production inference payloads.
* Evaluates **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) two-sample tests**:
  * $\text{PSI} < 0.10$: Stable distribution (Green).
  * $0.10 \le \text{PSI} < 0.20$: Moderate drift warning (Yellow).
  * $\text{PSI} \ge 0.20$: Severe covariate shift requiring model retraining (Red).
* Real-time calculation of overall system health score ($96\%$ Healthy).

---

## 5. System Architecture & Full-Stack Delivery

```mermaid
flowchart TD
    subgraph Ingestion ["1. Data Ingestion & Remote Sensing"]
        A1["Historical Yields<br>(225 Records, 5 Crops)"] --> D["Geospatial Data Fusion Engine<br>(src/data_fusion.py)"]
        A2["Soil Chemistry<br>(N, P, K, pH, SOM, EC)"] --> D
        A3["Weather Telemetry<br>(27,390 Daily Records)"] --> B["Weather Aggregator<br>(src/weather_aggregation.py)"]
        B --> D
        A4["Satellite Spectral Timeseries<br>(NDVI, NDRE, EVI, NDWI)"] --> D
    end

    subgraph FeatureEng ["2. Spatial Feature Engineering"]
        D --> E["Feature Matrix Pipeline<br>(src/features.py)"]
        E --> F["57 Engineered Features<br>(NPK Ratio, pH Stress, GDD, VPD)"]
    end

    subgraph Modeling ["3. Leak-Free ML Engine"]
        F --> G["Spatial Block Group-KFold CV<br>(src/models.py)"]
        G --> H["Random Forest Champion<br>(R² = 0.8033, RMSE = 0.5882)"]
        H --> I["TreeSHAP Attribution<br>(src/explainability.py)"]
    end

    subgraph Prescriptive ["4. Advisory & Economics"]
        H --> J["Counterfactual Prescriptive Advisor<br>(src/advisor.py)"]
        I --> J
        J --> K["Agri-Business Economics & MPT<br>(src/economic_optimizer.py)"]
    end

    subgraph Delivery ["5. Production Serving & Web UI"]
        H --> L["Production REST API<br>(src/api_service.py: 8000)"]
        J --> L
        K --> L
        L --> M["Interactive Web Dashboard<br>(static/index.html, app.js, style.css)"]
        L --> N["MLOps Covariate Drift Engine<br>(src/monitoring.py - PSI & KS)"]
    end
```

### 5.1 Web Application Interfaces
* **Single Farm Forecast**: Interactive N-P-K sliders, pH slider with live badges (`Optimal`, `Deficient`, `High`), 1-click presets (`🌾 Balanced Wheat`, `🍚 Coastal Rice`, `🌽 Intensive Maize`, `⚠️ Acid Shock Farm`, `☀️ Drought Stress`), and reactive debounced auto-forecasting.
* **What-If Soil Simulator**: Marginal delta controls ($\Delta\text{pH}$, $\Delta\text{N}$, $\Delta\text{Rainfall}$) showing instantaneous counterfactual yield updates.
* **Regional Cooperative Analytics**: Visualizing batch inference across dozens of regional plots simultaneously.
* **Covariate Drift & Model Health**: Real-time statistical drift table and health score gauge.
* **Executive PDF Export**: Instant print-ready farmer advisory reports via custom `@media print` CSS.

---

## 6. Verification, Testing & Quality Assurance

* **Unit & Integration Test Suite**: 78 / 78 passing tests across 16 test modules covering data cleaning, spatial fusion, feature engineering, regression modeling, SHAP explainability, prescriptive rules, REST API endpoints, economics, and monitoring.
* **System Health Audit**: 20 / 20 passing system health checks verifying artifact existence, dataset integrity, sub-75ms inference latency ($59.33\text{ ms}$ average), and Docker container configuration.
* **Code Quality & Style**: Clean Python code adhering strictly to PEP 8 standards with zero syntax or flake8 errors.
* **Containerization**: Production `Dockerfile` and `docker-compose.yml` for multi-platform containerized deployment.

---

## 7. Summary of Deliverables & Project Achievements

| Sprint Deliverable | Requirement Specification | Implemented Platform Artifact |
| :--- | :--- | :--- |
| **Data Ingestion** | Fusing disparate spatial and temporal sources. | `src/data_cleaning.py`, `src/weather_aggregation.py`, `src/data_fusion.py` |
| **Feature Engineering** | Non-linear relationships & interaction features. | `src/features.py`, `data/processed/train_features.csv` |
| **Spatial Cross-Validation**| Leak-free regional validation strategy. | `src/models.py`, `models/best_yield_regressor.joblib` |
| **SHAP Explainability** | Factor attribution isolating controllable inputs. | `src/explainability.py`, `reports/shap_attribution_report.md` |
| **Prescriptive Advisory** | Actionable farmer advice with exact dosage. | `src/advisor.py`, `src/inference.py` |
| **Production Serving** | Real-time REST API and interactive dashboard. | `src/api_service.py`, `static/index.html`, `static/app.js`, `static/style.css` |
| **Economics & Portfolio** | Profit margins and Markowitz allocation. | `src/economic_optimizer.py`, `reports/economic_optimization_report.md` |
| **MLOps Monitoring** | Population Stability Index & drift tracking. | `src/monitoring.py`, `/api/v1/monitoring` |
| **Verification & CI/CD** | Automated unit tests and Docker packaging. | `tests/` (78 tests), `Dockerfile`, `.github/workflows/ci.yml` |

---

## 8. Conclusion

The Axlero Agri-Tech Crop Yield & Soil Analysis Forecasting Platform demonstrates that modern data engineering and machine learning can effectively eliminate agricultural uncertainty. By combining multi-modal spatial telemetry, leak-free spatial modeling, and counterfactual optimization, the platform shifts agri-tech from passive yield prediction to **active, profit-maximizing agronomic intervention**.

All code, models, datasets, reports, and UI assets are fully committed, verified, and operational on branch `Syed-Abdus-Sabur`.
