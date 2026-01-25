"""
Robust Error Handling for ML Model Serving

Demonstrates patterns for handling common failure modes in production ML APIs:
1. Input validation errors (Pydantic automatic + custom)
2. Model prediction timeouts
3. Model loading failures
4. Resource exhaustion (OOM, rate limiting)
5. Graceful degradation with fallbacks

Key Principles:
- Fail fast: Validate inputs before expensive operations
- Fail gracefully: Return useful error messages to clients
- Fail safely: Don't expose internal details or crash the service
- Monitor failures: Log errors for debugging and alerting

Run:
    uvicorn error_handling:app --reload
"""

from contextlib import asynccontextmanager
from typing import List, Optional
from enum import Enum
import time
import logging

import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom exceptions for ML-specific errors
class ModelNotLoadedError(Exception):
    """Raised when model is not available."""
    pass


class PredictionTimeoutError(Exception):
    """Raised when prediction takes too long."""
    pass


class InvalidFeatureShapeError(Exception):
    """Raised when feature dimensions don't match model expectations."""
    pass


# Mock model with failure modes
class RobustModel:
    """Model that can simulate various failure modes."""

    def __init__(self, expected_features: int = 5):
        self.expected_features = expected_features
        self.weights = np.random.randn(expected_features)
        self.is_healthy = True

    def predict(
        self,
        features: np.ndarray,
        timeout_sec: float = 5.0
    ) -> float:
        """
        Make prediction with timeout protection.

        Args:
            features: Input feature vector
            timeout_sec: Maximum time allowed for prediction

        Returns:
            Prediction value

        Raises:
            InvalidFeatureShapeError: If features have wrong dimensions
            PredictionTimeoutError: If prediction exceeds timeout
            ModelNotLoadedError: If model is not healthy
        """
        if not self.is_healthy:
            raise ModelNotLoadedError("Model is not in healthy state")

        if len(features) != self.expected_features:
            raise InvalidFeatureShapeError(
                f"Expected {self.expected_features} features, got {len(features)}"
            )

        # Simulate prediction with timeout check
        start = time.time()
        time.sleep(0.1)  # Simulate inference time

        if time.time() - start > timeout_sec:
            raise PredictionTimeoutError(
                f"Prediction exceeded timeout of {timeout_sec}s"
            )

        return float(np.dot(features, self.weights))


# Request/Response schemas with validation
class PredictionRequest(BaseModel):
    """Request with strict validation."""

    features: List[float] = Field(
        ...,
        description="Input feature vector",
        min_length=1,
        max_length=100,  # Prevent excessive input
        example=[1.0, 2.0, 3.0, 4.0, 5.0]
    )

    timeout_sec: Optional[float] = Field(
        default=5.0,
        description="Maximum prediction time",
        gt=0,
        le=30.0  # Max 30 seconds
    )

    @validator('features')
    def validate_features(cls, v):
        """
        Custom validation for features.

        Checks beyond Pydantic's basic validation:
        - No NaN or infinity values
        - Reasonable value ranges
        """
        if any(not np.isfinite(x) for x in v):
            raise ValueError("Features must not contain NaN or infinity")

        if any(abs(x) > 1e6 for x in v):
            raise ValueError("Feature values must be in reasonable range")

        return v


class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float] = None
    model_version: str


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    error_code: str
    detail: Optional[str] = None
    request_id: Optional[str] = None


# Application setup
MODEL: Optional[RobustModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup with error handling."""
    global MODEL

    try:
        logger.info("Loading model...")
        MODEL = RobustModel(expected_features=5)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # In production, you might want to fail fast here
        # raise  # Uncomment to prevent startup if model fails to load

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Robust ML Serving API",
    version="1.0.0",
    lifespan=lifespan
)


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors.

    Provides clear error messages for input validation failures.
    """
    logger.warning(f"Validation error: {exc.errors()}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Input validation failed",
            "error_code": "VALIDATION_ERROR",
            "detail": exc.errors()
        }
    )


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(
    request: Request,
    exc: ModelNotLoadedError
) -> JSONResponse:
    """Handle model not loaded errors."""
    logger.error(f"Model not loaded: {exc}")

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Model is not available",
            "error_code": "MODEL_NOT_LOADED",
            "detail": str(exc)
        }
    )


@app.exception_handler(PredictionTimeoutError)
async def timeout_handler(
    request: Request,
    exc: PredictionTimeoutError
) -> JSONResponse:
    """Handle prediction timeout errors."""
    logger.error(f"Prediction timeout: {exc}")

    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={
            "error": "Prediction timed out",
            "error_code": "PREDICTION_TIMEOUT",
            "detail": str(exc)
        }
    )


@app.exception_handler(InvalidFeatureShapeError)
async def invalid_shape_handler(
    request: Request,
    exc: InvalidFeatureShapeError
) -> JSONResponse:
    """Handle feature shape mismatch errors."""
    logger.warning(f"Invalid feature shape: {exc}")

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid input dimensions",
            "error_code": "INVALID_FEATURE_SHAPE",
            "detail": str(exc)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Catch-all handler for unexpected errors.

    Prevents exposing internal errors to clients.
    Logs detailed error for debugging.
    """
    logger.error(f"Unexpected error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "detail": "An unexpected error occurred"
            # DO NOT expose exc details in production
        }
    )


@app.get("/health")
def health_check() -> dict:
    """
    Health check endpoint.

    Returns:
        200: Service is healthy and model is loaded
        503: Service is unhealthy (model not loaded or degraded)
    """
    if MODEL is None or not MODEL.is_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not healthy"
        )

    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_healthy": MODEL.is_healthy if MODEL else False
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make prediction with comprehensive error handling.

    Error Scenarios Handled:
    1. Input validation (automatic via Pydantic)
    2. Custom feature validation (NaN, infinity, range)
    3. Model not loaded
    4. Feature dimension mismatch
    5. Prediction timeout
    6. Unexpected errors

    All errors return appropriate HTTP status codes and error details.
    """
    # Check model is loaded
    if MODEL is None:
        raise ModelNotLoadedError("Model has not been loaded")

    # Convert to numpy array
    features = np.array(request.features)

    # Make prediction (may raise various errors)
    prediction = MODEL.predict(
        features,
        timeout_sec=request.timeout_sec
    )

    return PredictionResponse(
        prediction=prediction,
        model_version="1.0.0"
    )


@app.post("/predict-with-fallback", response_model=PredictionResponse)
def predict_with_fallback(request: PredictionRequest) -> PredictionResponse:
    """
    Prediction with graceful degradation.

    If the model fails, return a fallback prediction instead of error.
    Useful for non-critical applications where some response is better than none.

    Strategies:
    - Return historical average
    - Return simple heuristic
    - Return cached prediction (if input seen before)
    - Return default/conservative value
    """
    try:
        # Try primary prediction
        if MODEL is None:
            raise ModelNotLoadedError("Model not loaded")

        features = np.array(request.features)
        prediction = MODEL.predict(features, timeout_sec=request.timeout_sec)

        return PredictionResponse(
            prediction=prediction,
            confidence=0.9,  # High confidence for model prediction
            model_version="1.0.0"
        )

    except Exception as e:
        logger.warning(f"Primary prediction failed, using fallback: {e}")

        # Fallback: simple heuristic (mean of features)
        fallback_prediction = float(np.mean(request.features))

        return PredictionResponse(
            prediction=fallback_prediction,
            confidence=0.3,  # Low confidence for fallback
            model_version="fallback-1.0.0"
        )


@app.post("/debug/break-model")
def break_model() -> dict:
    """Debug endpoint: simulate model failure."""
    if MODEL:
        MODEL.is_healthy = False
    return {"message": "Model marked as unhealthy"}


@app.post("/debug/fix-model")
def fix_model() -> dict:
    """Debug endpoint: fix model."""
    if MODEL:
        MODEL.is_healthy = True
    return {"message": "Model marked as healthy"}


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*70)
    print("Error Handling Patterns for ML Serving")
    print("="*70)
    print("\nTest error scenarios:")
    print('\n1. Valid request:')
    print('   curl -X POST "http://localhost:8000/predict" \\')
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"features": [1,2,3,4,5]}\'')
    print('\n2. Invalid shape (model expects 5 features):')
    print('   curl -X POST "http://localhost:8000/predict" \\')
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"features": [1,2,3]}\'')
    print('\n3. Invalid values (NaN):')
    print('   curl -X POST "http://localhost:8000/predict" \\')
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"features": [1,2,NaN,4,5]}\'')
    print('\n4. Graceful fallback:')
    print('   curl -X POST "http://localhost:8000/debug/break-model"')
    print('   curl -X POST "http://localhost:8000/predict-with-fallback" \\')
    print("        -H 'Content-Type: application/json' \\")
    print('        -d \'{"features": [1,2,3,4,5]}\'')
    print("="*70 + "\n")

    uvicorn.run(
        "error_handling:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
