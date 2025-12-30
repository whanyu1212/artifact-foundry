# Machine Learning Model Serving Dockerfile
#
# Optimized for deploying ML models to production (GKE, ECS, Cloud Run, etc.)
# Demonstrates:
# - Efficient dependency installation for ML packages
# - Model file handling
# - Resource-aware serving with gunicorn/uvicorn
# - Health checks for k8s readiness/liveness probes
#
# Assumes:
# - app.py (FastAPI/Flask serving endpoint)
# - requirements.txt (ML dependencies: scikit-learn, tensorflow, torch, etc.)
# - models/ directory (trained model files) OR model loaded from cloud storage

# ============================================================================
# Stage 1: Builder - Install dependencies
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies for ML packages
# These are needed to build/compile certain Python ML libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
# Building wheels speeds up installation and allows caching
COPY requirements.txt .

# Build wheels for all dependencies
# This pre-compiles packages like numpy, scipy, scikit-learn
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels \
        -r requirements.txt

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app \
    MODEL_PATH=/app/models \
    PORT=8080

WORKDIR $APP_HOME

# Install runtime system dependencies only
# Many ML packages need these shared libraries at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # Linear algebra libraries (for numpy, scipy)
        libgomp1 \
        libopenblas0 \
        # For health checks
        curl && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python packages from builder
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application code
COPY ./app ./app
COPY ./models ./models

# Create non-root user
# Important for security in cloud environments
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser $APP_HOME
USER mluser

# Expose port
# GKE, Cloud Run, etc. will route traffic to this port
EXPOSE $PORT

# Health check
# Kubernetes uses this for readiness/liveness probes
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run with gunicorn for production
# 4 workers is a good default, adjust based on CPU cores
# --worker-class uvicorn.workers.UvicornWorker for async (FastAPI)
# Remove for Flask (use default sync workers)
CMD exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    app.main:app

# Alternative: Simple uvicorn for development/testing
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
