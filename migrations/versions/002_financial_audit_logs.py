"""Create financial_audit_logs table for session history persistence

Revision ID: 002_financial_audit_logs
Revises: 001_initial_schema
Create Date: 2026-11-02 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_financial_audit_logs'
down_revision: Union[str, None] = '001_initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'financial_audit_logs',
        sa.Column('id', sa.CHAR(length=36), nullable=False),
        sa.Column('user_id', sa.CHAR(length=36), nullable=True),
        sa.Column('soil_log_id', sa.CHAR(length=36), nullable=True),
        sa.Column('yield_prediction_id', sa.CHAR(length=36), nullable=True),
        sa.Column('crop_type', sa.String(length=100), nullable=False),
        sa.Column('farm_size_ha', sa.Float(), nullable=False),
        sa.Column('predicted_yield', sa.Float(), nullable=False, comment='Estimated harvest yield MT/Ha'),
        sa.Column('total_production_mt', sa.Float(), nullable=False, comment='Total harvest volume MT'),
        sa.Column('currency', sa.String(length=10), nullable=False, server_default='INR'),
        sa.Column('crop_price_per_mt', sa.Float(), nullable=False),
        sa.Column('gross_revenue', sa.Float(), nullable=False),
        sa.Column('total_operational_costs', sa.Float(), nullable=False),
        sa.Column('net_profit', sa.Float(), nullable=False),
        sa.Column('roi_percentage', sa.Float(), nullable=False),
        sa.Column('profit_margin_percentage', sa.Float(), nullable=False),
        sa.Column('break_even_yield', sa.Float(), nullable=True),
        sa.Column('break_even_price', sa.Float(), nullable=True),
        sa.Column('cost_breakdown', sa.JSON(), nullable=True),
        sa.Column('shap_diagnostic_feedback', sa.JSON(), nullable=True),
        sa.Column('prescriptive_amendments', sa.JSON(), nullable=True),
        sa.Column('assessment_notes', sa.String(length=255), nullable=True),
        sa.Column('audit_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['soil_log_id'], ['soil_logs.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['yield_prediction_id'], ['yield_predictions.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_financial_audit_logs_user_id'), 'financial_audit_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_financial_audit_logs_crop_type'), 'financial_audit_logs', ['crop_type'], unique=False)
    op.create_index(op.f('ix_financial_audit_logs_soil_log_id'), 'financial_audit_logs', ['soil_log_id'], unique=False)
    op.create_index(op.f('ix_financial_audit_logs_yield_prediction_id'), 'financial_audit_logs', ['yield_prediction_id'], unique=False)
    op.create_index(op.f('ix_financial_audit_logs_created_at'), 'financial_audit_logs', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_financial_audit_logs_created_at'), table_name='financial_audit_logs')
    op.drop_index(op.f('ix_financial_audit_logs_yield_prediction_id'), table_name='financial_audit_logs')
    op.drop_index(op.f('ix_financial_audit_logs_soil_log_id'), table_name='financial_audit_logs')
    op.drop_index(op.f('ix_financial_audit_logs_crop_type'), table_name='financial_audit_logs')
    op.drop_index(op.f('ix_financial_audit_logs_user_id'), table_name='financial_audit_logs')
    op.drop_table('financial_audit_logs')

