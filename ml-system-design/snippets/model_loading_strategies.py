"""
Model Loading Strategies for ML Serving

Compares different approaches to loading ML models in production APIs:
1. Startup Loading - Load once at application startup (recommended)
2. Lazy Loading - Load on first request, cache afterward
3. On-Demand Loading - Load and unload per request (memory constrained)

Trade-offs:
- Startup: Fast inference, high memory if model is large
- Lazy: Flexible, first request is slow
- On-Demand: Memory efficient, every request is slow

Run:
    uvicorn model_loading_strategies:app --reload
"""

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import List, Optional
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Mock model for demonstration
class MockModel:
    """Simulates a model with loading time and memory footprint."""

    def __init__(self, model_id: str, load_time_sec: float = 2.0):
        self.model_id = model_id
        print(f"Loading model {model_id}... (simulating {load_time_sec}s)")
        time.sleep(load_time_sec)  # Simulate expensive loading
        self.weights = np.random.randn(1000, 1000)  # Simulates memory usage
        print(f"Model {model_id} loaded!")

    def predict(self, features: List[float]) -> float:
        return float(np.sum(features) * 0.5)


# Request/Response schemas
class PredictionRequest(BaseModel):
    features: List[float] = Field(..., min_length=1, example=[1.0, 2.0, 3.0])


class PredictionResponse(BaseModel):
    prediction: float
    model_id: str
    loading_strategy: str
    inference_time_ms: float


# Global variable for startup-loaded model
STARTUP_MODEL: Optional[MockModel] = None


# Cache for lazy-loaded models (max 3 models in memory)
@lru_cache(maxsize=3)
def get_cached_model(model_id: str) -> MockModel:
    """
    Lazy loading with caching.

    First call: Loads model (slow)
    Subsequent calls: Returns cached model (fast)

    The @lru_cache decorator automatically manages the cache:
    - maxsize=3: Keep at most 3 models in memory
    - LRU eviction: Remove least recently used when cache is full
    """
    return MockModel(model_id, load_time_sec=1.0)


def load_on_demand(model_id: str) -> MockModel:
    """
    On-demand loading without caching.

    Every call loads a fresh model instance.
    Memory efficient but slow for every request.

    Use case: Multiple large models, infrequent requests
    """
    return MockModel(model_id, load_time_sec=1.0)


# Lifespan: Startup loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Strategy 1: Startup Loading

    Pros:
    - Fast inference (model already in memory)
    - Predictable memory usage
    - No surprise latency on first request

    Cons:
    - Slower application startup
    - Memory allocated even if model unused
    - Harder to manage multiple models
    """
    global STARTUP_MODEL

    print("=== STARTUP LOADING ===")
    STARTUP_MODEL = MockModel("startup-model", load_time_sec=2.0)

    yield

    print("Cleaning up startup model...")
    STARTUP_MODEL = None


app = FastAPI(
    title="Model Loading Strategies",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict/startup", response_model=PredictionResponse)
def predict_startup(request: PredictionRequest) -> PredictionResponse:
    """
    Strategy 1: Startup Loading

    Model is loaded once at application startup and stored globally.

    Best for:
    - Single model or few models
    - Models that fit in memory
    - Low-latency requirements
    - High request volume
    """
    if STARTUP_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    prediction = STARTUP_MODEL.predict(request.features)
    inference_time = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        prediction=prediction,
        model_id=STARTUP_MODEL.model_id,
        loading_strategy="startup",
        inference_time_ms=inference_time
    )


@app.post("/predict/lazy/{model_id}", response_model=PredictionResponse)
def predict_lazy(model_id: str, request: PredictionRequest) -> PredictionResponse:
    """
    Strategy 2: Lazy Loading with Caching

    Model is loaded on first request and cached (LRU cache).
    Subsequent requests use the cached model.

    Best for:
    - Multiple models (serve different models per request)
    - Models that fit in memory
    - Acceptable first-request latency
    - Different models have different usage patterns

    Note: First request will be slow (~1s + inference), rest will be fast.

    Example:
        First call to /predict/lazy/model-A: ~1000ms (loading + inference)
        Second call to /predict/lazy/model-A: ~1ms (cached)
        Call to /predict/lazy/model-B: ~1000ms (new model)
    """
    start = time.perf_counter()

    # get_cached_model will load if not in cache, otherwise return cached
    model = get_cached_model(model_id)

    prediction = model.predict(request.features)
    inference_time = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        prediction=prediction,
        model_id=model.model_id,
        loading_strategy="lazy-cached",
        inference_time_ms=inference_time
    )


@app.post("/predict/on-demand/{model_id}", response_model=PredictionResponse)
def predict_on_demand(model_id: str, request: PredictionRequest) -> PredictionResponse:
    """
    Strategy 3: On-Demand Loading (No Caching)

    Model is loaded fresh for every request and discarded after.

    Best for:
    - Very large models (can't keep in memory)
    - Infrequent requests per model
    - Memory-constrained environments
    - When freshness is critical (load latest version each time)

    Cons:
    - Every request is slow (~1s + inference)
    - Not suitable for real-time serving

    Example:
        Every call to /predict/on-demand/model-X: ~1000ms
    """
    start = time.perf_counter()

    # Load model fresh each time
    model = load_on_demand(model_id)

    prediction = model.predict(request.features)
    inference_time = (time.perf_counter() - start) * 1000

    # Model will be garbage collected after this function returns
    return PredictionResponse(
        prediction=prediction,
        model_id=model.model_id,
        loading_strategy="on-demand",
        inference_time_ms=inference_time
    )


@app.get("/cache-info")
def cache_info() -> dict:
    """
    Inspect the lazy loading cache.

    Shows cache statistics:
    - hits: Number of times model was found in cache
    - misses: Number of times model had to be loaded
    - currsize: Number of models currently in cache
    - maxsize: Maximum cache capacity
    """
    info = get_cached_model.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "current_size": info.currsize,
        "max_size": info.maxsize,
        "hit_rate": info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
    }


@app.post("/cache-clear")
def clear_cache() -> dict:
    """Clear the lazy loading cache (free memory)."""
    get_cached_model.cache_clear()
    return {"message": "Cache cleared", "cache_info": cache_info()}


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("Model Loading Strategies Demo")
    print("="*60)
    print("\nEndpoints:")
    print("  1. POST /predict/startup - Startup loading (fastest)")
    print("  2. POST /predict/lazy/{model_id} - Lazy loading (flexible)")
    print("  3. POST /predict/on-demand/{model_id} - On-demand (memory efficient)")
    print("  4. GET /cache-info - View lazy loading cache stats")
    print("\nTry:")
    print('  curl -X POST "http://localhost:8000/predict/startup" \\')
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"features\": [1.0, 2.0, 3.0]}'")
    print("="*60 + "\n")

    uvicorn.run(
        "model_loading_strategies:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
