# Axlero Agri-Tech Crop Yield & Soil Analysis Platform
# Production Multi-Stage / Lean Container Image
# Member 1: Systems & ML Infrastructure Engineer

FROM python:3.11-slim

LABEL maintainer="Syed Abdus Sabur <sabur@axlero-agritech.com>"
LABEL version="1.0.0"
LABEL description="Production Crop Yield Forecasting & Prescriptive Advisory API Service"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install minimal OS dependencies for geospatial & ML libraries and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy platform source code, models, static dashboard assets, and documentation
COPY src/ /app/src/
COPY static/ /app/static/
COPY models/ /app/models/
COPY data/ /app/data/
COPY reports/ /app/reports/
COPY run_batch_inference.py /app/
COPY run_data_pipeline.py /app/
COPY run_system_health_audit.py /app/
COPY run_master_platform.py /app/

# Create non-root user for security compliance
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose internal microservice port
EXPOSE 8000

# Container healthcheck against REST API health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Default runtime entrypoint: run the high-performance REST API & Dashboard server
CMD ["python", "-m", "src.api_service", "--port", "8000"]
