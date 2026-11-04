# Axlero Agri-Tech Platform - System Health & Integrity Audit Report

**Audit Status:** `HEALTHY`  
**Timestamp:** `2026-11-04T16:48:28.602339`  
**Duration:** `2.948s`  
**Passing Checks:** `20 / 20`  

## Category Audit Breakdown

### Model Checkpoints
| Verification Check | Status | Details |
|---|---|---|
| Production Regressor Artifact | PASS | **path**: models\best_yield_regressor.joblib<br>**model_type**: RandomForestRegressor<br>**n_features**: 57<br>**spatial_cv_r2**: 0.8033<br>**target**: crop_yield_mt_ha |
| Neural Regressor Artifact | PASS | **path**: models\neural_yield_regressor.joblib<br>**model_type**: MLPRegressor<br>**n_features**: 57<br>**spatial_cv_r2**: 0.3063 |

### Data Store & Feature Layers
| Verification Check | Status | Details |
|---|---|---|
| Raw Crop Yield Records | PASS | **path**: data\raw\historical_yields.csv<br>**rows**: 225<br>**columns**: 6<br>**missing_required_cols**: [] |
| Raw Soil Profiles | PASS | **path**: data\raw\soil_chemistry.csv<br>**rows**: 15<br>**columns**: 8<br>**missing_required_cols**: [] |
| Raw Weather Telemetry | PASS | **path**: data\raw\weather_daily.csv<br>**rows**: 5000<br>**columns**: 9<br>**missing_required_cols**: [] |
| Dekadal Satellite Timeseries | PASS | **path**: data\raw\satellite_spectral_timeseries.csv<br>**rows**: 2700<br>**columns**: 13<br>**missing_required_cols**: [] |
| Unified Agronomic Dataset | PASS | **path**: data\processed\unified_agri_dataset.csv<br>**rows**: 225<br>**columns**: 35<br>**missing_required_cols**: [] |
| Engineered ML Feature Matrix | PASS | **path**: data\processed\train_features.csv<br>**rows**: 225<br>**columns**: 63<br>**missing_required_cols**: [] |
| Canopy Vegetation Indices | PASS | **path**: data\processed\canopy_vegetation_indices.csv<br>**rows**: 225<br>**columns**: 14<br>**missing_required_cols**: [] |
| Multimodal Agri Dataset | PASS | **path**: data\processed\multimodal_agri_dataset.csv<br>**rows**: 225<br>**columns**: 50<br>**missing_required_cols**: [] |
| Regional Batch Predictions | PASS | **path**: data\processed\batch_predictions.csv<br>**rows**: 225<br>**columns**: 18<br>**missing_required_cols**: [] |
| Climate Scenario Projections | PASS | **path**: data\processed\climate_scenario_projections.csv<br>**rows**: 450<br>**columns**: 12<br>**missing_required_cols**: [] |

### Real-Time Inference SLA
| Verification Check | Status | Details |
|---|---|---|
| Single-Farm Advisory & Inference Latency (<75ms SLA) | PASS | **iterations**: 50<br>**mean_ms**: 31.42<br>**p95_ms**: 33.16<br>**min_ms**: 29.73<br>**max_ms**: 33.49<br>**sla_target_ms**: 75.0 |

### Web Service & UI Assets
| Verification Check | Status | Details |
|---|---|---|
| Dashboard HTML | PASS | **path**: static\index.html<br>**size_kb**: 30.43 |
| Dashboard Stylesheet | PASS | **path**: static\style.css<br>**size_kb**: 19.61 |
| Dashboard Interactive JS | PASS | **path**: static\app.js<br>**size_kb**: 26.26 |
| REST API Service Module | PASS | **path**: src\api_service.py<br>**size_kb**: 10.06 |
| Production Dockerfile | PASS | **path**: Dockerfile<br>**size_kb**: 1.72 |
| Docker Compose Configuration | PASS | **path**: docker-compose.yml<br>**size_kb**: 0.67 |
| CI/CD Workflow | PASS | **path**: .github\workflows\ci.yml<br>**size_kb**: 2.76 |

