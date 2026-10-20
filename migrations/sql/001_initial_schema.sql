-- ============================================================================
-- PostgreSQL Schema Migration: 001_initial_schema.sql
-- Project: Axlero Agri-Tech Crop Yield & Soil Analysis Forecasting System
-- Deliverable: PostgreSQL Schema for Users, Soil Logs, and Yield Predictions
-- ============================================================================

-- 1. Enable required extensions for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 2. Trigger function to auto-update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_timestamp_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Table: users
-- Purpose: Authentication, tenant farm profile, and role-based access
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    role VARCHAR(50) NOT NULL DEFAULT 'farmer',
    farm_name VARCHAR(255),
    district VARCHAR(100),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_superuser BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_district ON users(district);

DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();

-- ============================================================================
-- Table: soil_logs
-- Purpose: In-situ chemical & physical soil sample readings per farm/parcel
-- ============================================================================
CREATE TABLE IF NOT EXISTS soil_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    field_name VARCHAR(150) NOT NULL,
    district VARCHAR(100) NOT NULL,
    latitude NUMERIC(9, 6),
    longitude NUMERIC(9, 6),
    nitrogen NUMERIC(7, 2) NOT NULL,
    phosphorus NUMERIC(7, 2) NOT NULL,
    potassium NUMERIC(7, 2) NOT NULL,
    ph NUMERIC(4, 2) NOT NULL,
    organic_matter NUMERIC(5, 2),
    moisture NUMERIC(5, 2),
    soil_type VARCHAR(50),
    sampled_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_soil_ph_range CHECK (ph >= 0.0 AND ph <= 14.0),
    CONSTRAINT chk_nitrogen_non_negative CHECK (nitrogen >= 0.0),
    CONSTRAINT chk_phosphorus_non_negative CHECK (phosphorus >= 0.0),
    CONSTRAINT chk_potassium_non_negative CHECK (potassium >= 0.0)
);

CREATE INDEX IF NOT EXISTS idx_soil_logs_user_id ON soil_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_soil_logs_district ON soil_logs(district);
CREATE INDEX IF NOT EXISTS idx_soil_logs_sampled_at ON soil_logs(sampled_at DESC);
CREATE INDEX IF NOT EXISTS idx_soil_logs_location ON soil_logs(latitude, longitude);

DROP TRIGGER IF EXISTS trg_soil_logs_updated_at ON soil_logs;
CREATE TRIGGER trg_soil_logs_updated_at
    BEFORE UPDATE ON soil_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();

-- ============================================================================
-- Table: yield_predictions
-- Purpose: Model inference outputs, SHAP explainability & agronomic advisory
-- ============================================================================
CREATE TABLE IF NOT EXISTS yield_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    soil_log_id UUID REFERENCES soil_logs(id) ON DELETE SET NULL,
    crop VARCHAR(100) NOT NULL,
    season VARCHAR(50),
    model_version VARCHAR(50) NOT NULL DEFAULT 'v1.0.0',
    predicted_yield NUMERIC(8, 3) NOT NULL,
    confidence_score NUMERIC(5, 4),
    baseline_yield NUMERIC(8, 3),
    potential_gain NUMERIC(8, 3),
    recommendations JSONB,
    shap_values JSONB,
    input_features JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_yield_pred_user_id ON yield_predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_yield_pred_soil_log_id ON yield_predictions(soil_log_id);
CREATE INDEX IF NOT EXISTS idx_yield_pred_crop ON yield_predictions(crop);
CREATE INDEX IF NOT EXISTS idx_yield_pred_created_at ON yield_predictions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_yield_pred_user_crop ON yield_predictions(user_id, crop);

DROP TRIGGER IF EXISTS trg_yield_predictions_updated_at ON yield_predictions;
CREATE TRIGGER trg_yield_predictions_updated_at
    BEFORE UPDATE ON yield_predictions
    FOR EACH ROW
    EXECUTE FUNCTION update_timestamp_column();
