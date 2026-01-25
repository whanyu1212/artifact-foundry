"""
Basic ML Model Serving with FastAPI

Demonstrates the fundamental pattern for serving machine learning models via REST API.
This example shows startup model loading, request/response validation, and simple inference.

Key Concepts:
- Model loading at application startup (lifespan events)
- Pydantic models for request/response validation
- Type hints for API documentation
- Storing model in application state

Run:
    uvicorn basic_model_serving:app --reload

Test:
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d '{"features": [1.0, 2.0, 3.0]}'
"""

from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Mock ML model for demonstration
# In practice, this would be sklearn, torch, tensorflow, etc.
class SimpleModel:
    """
    Mock ML model for demonstration purposes.

    In production, replace with actual model loading:
    - Scikit-learn: joblib.load('model.pkl')
    - PyTorch: torch.load('model.pt')
    - TensorFlow: tf.keras.models.load_model('model.h5')
    """

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def predict(self, features: np.ndarray) -> float:
        """Simple linear prediction: w^T x"""
        return float(np.dot(features, self.weights))


# Request/Response schemas using Pydantic
# These provide automatic validation and API documentation
class PredictionRequest(BaseModel):
    """Input features for prediction."""

    features: List[float] = Field(
        ...,  # Required field
        description="Input features for the model",
        min_length=3,
        max_length=3,
        example=[1.0, 2.0, 3.0]
    )


class PredictionResponse(BaseModel):
    """Model prediction output."""

    prediction: float = Field(
        ...,
        description="Model prediction value",
        example=14.0
    )
    model_version: str = Field(
        ...,
        description="Version of the model that generated this prediction",
        example="1.0.0"
    )


# Application lifespan manager
# Handles startup and shutdown events (FastAPI 0.93+)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events.

    Startup: Load model into memory once
    Shutdown: Clean up resources (if needed)

    This is more efficient than loading the model on each request.
    """
    # Startup: Load model
    print("Loading model...")
    model = SimpleModel(weights=np.array([1.0, 2.0, 3.0]))

    # Store model in app state (accessible in all route handlers)
    app.state.model = model
    app.state.model_version = "1.0.0"

    print("Model loaded successfully!")

    yield  # Application runs here

    # Shutdown: Cleanup (if needed)
    print("Shutting down...")
    # e.g., close database connections, save metrics, etc.


# Create FastAPI application
app = FastAPI(
    title="ML Model Serving API",
    description="Basic example of serving ML models with FastAPI",
    version="1.0.0",
    lifespan=lifespan  # Register lifespan manager
)


@app.get("/")
def root() -> dict:
    """Root endpoint - API information."""
    return {
        "message": "ML Model Serving API",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check() -> dict:
    """
    Health check endpoint.

    Used by load balancers and orchestrators (Kubernetes, Docker Swarm)
    to determine if the service is healthy.
    """
    return {
        "status": "healthy",
        "model_version": app.state.model_version
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make a prediction using the loaded model.

    Args:
        request: Validated input features

    Returns:
        Prediction and model version

    Example:
        >>> # POST /predict
        >>> # {"features": [1.0, 2.0, 3.0]}
        >>> # Response: {"prediction": 14.0, "model_version": "1.0.0"}
    """
    # Convert input to numpy array
    features = np.array(request.features)

    # Get model from application state
    model = app.state.model

    # Run inference
    prediction = model.predict(features)

    # Return validated response
    return PredictionResponse(
        prediction=prediction,
        model_version=app.state.model_version
    )


if __name__ == "__main__":
    import uvicorn

    # Run the application
    # workers=1 for development (reloading)
    # workers=4 for production (CPU cores)
    uvicorn.run(
        "basic_model_serving:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (dev only)
    )
