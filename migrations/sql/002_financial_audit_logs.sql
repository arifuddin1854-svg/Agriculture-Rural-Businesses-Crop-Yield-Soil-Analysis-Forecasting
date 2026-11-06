-- Migration: 002_financial_audit_logs
-- Engineer 3: Financial Audit Logs & Assessment Session History Persistence

CREATE TABLE IF NOT EXISTS financial_audit_logs (
    id CHAR(36) PRIMARY KEY,
    user_id CHAR(36) REFERENCES users(id) ON DELETE SET NULL,
    soil_log_id CHAR(36) REFERENCES soil_logs(id) ON DELETE SET NULL,
    yield_prediction_id CHAR(36) REFERENCES yield_predictions(id) ON DELETE SET NULL,
    crop_type VARCHAR(100) NOT NULL,
    farm_size_ha DOUBLE PRECISION NOT NULL,
    predicted_yield DOUBLE PRECISION NOT NULL,
    total_production_mt DOUBLE PRECISION NOT NULL,
    currency VARCHAR(10) NOT NULL DEFAULT 'INR',
    crop_price_per_mt DOUBLE PRECISION NOT NULL,
    gross_revenue DOUBLE PRECISION NOT NULL,
    total_operational_costs DOUBLE PRECISION NOT NULL,
    net_profit DOUBLE PRECISION NOT NULL,
    roi_percentage DOUBLE PRECISION NOT NULL,
    profit_margin_percentage DOUBLE PRECISION NOT NULL,
    break_even_yield DOUBLE PRECISION,
    break_even_price DOUBLE PRECISION,
    cost_breakdown JSONB,
    shap_diagnostic_feedback JSONB,
    prescriptive_amendments JSONB,
    assessment_notes VARCHAR(255),
    audit_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_financial_audit_logs_user_id ON financial_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS ix_financial_audit_logs_crop_type ON financial_audit_logs(crop_type);
CREATE INDEX IF NOT EXISTS ix_financial_audit_logs_soil_log_id ON financial_audit_logs(soil_log_id);
CREATE INDEX IF NOT EXISTS ix_financial_audit_logs_yield_prediction_id ON financial_audit_logs(yield_prediction_id);
CREATE INDEX IF NOT EXISTS ix_financial_audit_logs_created_at ON financial_audit_logs(created_at);

