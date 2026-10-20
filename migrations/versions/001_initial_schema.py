"""Initial schema for users, soil_logs, and yield_predictions

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-10-19 17:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create 'users' table
    op.create_table(
        'users',
        sa.Column('id', sa.CHAR(length=36), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('role', sa.String(length=50), nullable=False, server_default='farmer'),
        sa.Column('farm_name', sa.String(length=255), nullable=True),
        sa.Column('district', sa.String(length=100), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_district'), 'users', ['district'], unique=False)

    # 2. Create 'soil_logs' table
    op.create_table(
        'soil_logs',
        sa.Column('id', sa.CHAR(length=36), nullable=False),
        sa.Column('user_id', sa.CHAR(length=36), nullable=False),
        sa.Column('field_name', sa.String(length=150), nullable=False),
        sa.Column('district', sa.String(length=100), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('nitrogen', sa.Float(), nullable=False, comment='Soil Nitrogen in kg/ha'),
        sa.Column('phosphorus', sa.Float(), nullable=False, comment='Soil Phosphorus in kg/ha'),
        sa.Column('potassium', sa.Float(), nullable=False, comment='Soil Potassium in kg/ha'),
        sa.Column('ph', sa.Float(), nullable=False, comment='Soil pH level (0-14)'),
        sa.Column('organic_matter', sa.Float(), nullable=True, comment='Soil Organic Matter %'),
        sa.Column('moisture', sa.Float(), nullable=True, comment='Soil Moisture %'),
        sa.Column('soil_type', sa.String(length=50), nullable=True),
        sa.Column('sampled_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint('ph >= 0 AND ph <= 14', name='chk_soil_ph_range'),
        sa.CheckConstraint('nitrogen >= 0', name='chk_nitrogen_non_negative'),
        sa.CheckConstraint('phosphorus >= 0', name='chk_phosphorus_non_negative'),
        sa.CheckConstraint('potassium >= 0', name='chk_potassium_non_negative'),
    )
    op.create_index(op.f('ix_soil_logs_user_id'), 'soil_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_soil_logs_district'), 'soil_logs', ['district'], unique=False)
    op.create_index(op.f('ix_soil_logs_sampled_at'), 'soil_logs', ['sampled_at'], unique=False)

    # 3. Create 'yield_predictions' table
    op.create_table(
        'yield_predictions',
        sa.Column('id', sa.CHAR(length=36), nullable=False),
        sa.Column('user_id', sa.CHAR(length=36), nullable=False),
        sa.Column('soil_log_id', sa.CHAR(length=36), nullable=True),
        sa.Column('crop', sa.String(length=100), nullable=False),
        sa.Column('season', sa.String(length=50), nullable=True),
        sa.Column('model_version', sa.String(length=50), nullable=False, server_default='v1.0.0'),
        sa.Column('predicted_yield', sa.Float(), nullable=False, comment='Forecasted yield in Metric Tons / Hectare'),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('baseline_yield', sa.Float(), nullable=True),
        sa.Column('potential_gain', sa.Float(), nullable=True),
        sa.Column('recommendations', sa.JSON(), nullable=True),
        sa.Column('shap_values', sa.JSON(), nullable=True),
        sa.Column('input_features', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['soil_log_id'], ['soil_logs.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_yield_predictions_user_id'), 'yield_predictions', ['user_id'], unique=False)
    op.create_index(op.f('ix_yield_predictions_soil_log_id'), 'yield_predictions', ['soil_log_id'], unique=False)
    op.create_index(op.f('ix_yield_predictions_crop'), 'yield_predictions', ['crop'], unique=False)
    op.create_index(op.f('ix_yield_predictions_created_at'), 'yield_predictions', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_yield_predictions_created_at'), table_name='yield_predictions')
    op.drop_index(op.f('ix_yield_predictions_crop'), table_name='yield_predictions')
    op.drop_index(op.f('ix_yield_predictions_soil_log_id'), table_name='yield_predictions')
    op.drop_index(op.f('ix_yield_predictions_user_id'), table_name='yield_predictions')
    op.drop_table('yield_predictions')

    op.drop_index(op.f('ix_soil_logs_sampled_at'), table_name='soil_logs')
    op.drop_index(op.f('ix_soil_logs_district'), table_name='soil_logs')
    op.drop_index(op.f('ix_soil_logs_user_id'), table_name='soil_logs')
    op.drop_table('soil_logs')

    op.drop_index(op.f('ix_users_district'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
