"""
Example ML Model Serving Application with FastAPI

This is a reference implementation showing how to structure an ML serving API.
Demonstrates:
- Model loading from local file or cloud storage
- Request validation with Pydantic
- Health checks for Kubernetes
- Error handling
- Logging for production monitoring

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8080

    # Or with gunicorn for production:
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app

Requirements:
    fastapi
    uvicorn[standard]
    gunicorn
    pydantic
    scikit-learn  # or tensorflow, torch, etc.
    numpy
"""

import os
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models")
MODEL_FILE = os.path.join(MODEL_PATH, "model.pkl")

# Global variable to hold loaded model
model = None


# ============================================================================
# Model Loading
# ============================================================================
def load_model():
    """
    Load ML model from file or cloud storage.

    In production, you might load from:
    - Google Cloud Storage (GCS): gs://bucket/model.pkl
    - AWS S3: s3://bucket/model.pkl
    - Azure Blob Storage
    - Model registry (MLflow, Vertex AI Model Registry)
    """
    global model

    try:
        # Example: Load from local file
        # Replace with your actual model loading logic
        import joblib
        model = joblib.load(MODEL_FILE)
        logger.info(f"Model loaded successfully from {MODEL_FILE}")

        # Example: Load from GCS (uncomment if using)
        # from google.cloud import storage
        # client = storage.Client()
        # bucket = client.bucket("your-bucket")
        # blob = bucket.blob("models/model.pkl")
        # blob.download_to_filename("/tmp/model.pkl")
        # model = joblib.load("/tmp/model.pkl")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


# ============================================================================
# Lifespan Management
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    Load model on startup, clean up on shutdown.
    """
    logger.info("Starting ML API...")
    load_model()
    logger.info("ML API ready to serve predictions")

    yield  # Application runs here

    logger.info("Shutting down ML API...")
    # Cleanup if needed


# ============================================================================
# FastAPI Application
# ============================================================================
app = FastAPI(
    title="ML Model API",
    description="Production ML model serving with FastAPI",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Request/Response Models
# ============================================================================
class PredictionRequest(BaseModel):
    """Input data for prediction."""
    features: List[float] = Field(
        ...,
        description="Feature vector for prediction",
        example=[5.1, 3.5, 1.4, 0.2]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }


class PredictionResponse(BaseModel):
    """Prediction output."""
    prediction: Any = Field(..., description="Model prediction")
    probability: List[float] | None = Field(
        None,
        description="Class probabilities (if applicable)"
    )
    model_version: str = Field(..., description="Model version")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "setosa",
                "probability": [0.9, 0.05, 0.05],
                "model_version": "1.0.0"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    instances: List[List[float]] = Field(
        ...,
        description="List of feature vectors",
        example=[[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]]
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/", tags=["Info"])
async def root():
    """API information endpoint."""
    return {
        "name": "ML Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch_predict"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Health check endpoint.
    Used by Kubernetes liveness and readiness probes.
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Make a single prediction.

    Args:
        request: PredictionRequest with features

    Returns:
        PredictionResponse with prediction and probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to numpy array
        features = np.array(request.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Get probabilities if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(features)[0].tolist()

        return {
            "prediction": prediction.item() if hasattr(prediction, 'item') else prediction,
            "probability": probability,
            "model_version": "1.0.0"
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make predictions for multiple instances.
    More efficient than calling /predict multiple times.

    Args:
        request: BatchPredictionRequest with multiple feature vectors

    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert input to numpy array
        features = np.array(request.instances)

        # Make predictions
        predictions = model.predict(features)

        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features).tolist()

        return {
            "predictions": [
                pred.item() if hasattr(pred, 'item') else pred
                for pred in predictions
            ],
            "probabilities": probabilities,
            "model_version": "1.0.0",
            "count": len(predictions)
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ============================================================================
# Example Usage (when running this file directly)
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    # For development only
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
