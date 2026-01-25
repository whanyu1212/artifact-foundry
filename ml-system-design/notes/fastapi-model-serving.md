# FastAPI for ML Model Serving

## Overview

FastAPI is a modern, high-performance web framework ideal for serving ML models in production. It provides automatic validation, async support, and automatic API documentation.

## Key Concepts for Model Serving

### Model Loading Strategies

**Startup Loading (Recommended for Most Cases)**
- Load model once when application starts
- Store in application state or global variable
- Fast inference, consistent memory usage
- Best for models that fit in memory

**Lazy Loading**
- Load model on first request
- Use caching to avoid repeated loads
- Good for multiple models or large models
- Trade-off: first request is slow

**On-Demand Loading**
- Load model per request, unload after
- Memory efficient for rarely-used models
- High latency, not suitable for real-time serving

### Request/Response Patterns

**Synchronous Inference**
```python
@app.post("/predict")
def predict(features: FeatureInput) -> PredictionOutput:
    # Blocking call - simple, works for fast models
    return model.predict(features)
```

**Asynchronous Inference**
```python
@app.post("/predict")
async def predict(features: FeatureInput) -> PredictionOutput:
    # Non-blocking - better for I/O bound operations
    result = await run_in_threadpool(model.predict, features)
    return result
```

**Background Tasks**
```python
@app.post("/predict-async")
async def predict_async(features: FeatureInput, background_tasks: BackgroundTasks):
    # Return immediately, process in background
    task_id = generate_task_id()
    background_tasks.add_task(process_prediction, task_id, features)
    return {"task_id": task_id}
```

### Error Handling for ML Models

**Common Failure Modes**
- Invalid input shape or types
- Model prediction timeout
- Out-of-memory errors
- Model not loaded
- Feature preprocessing failures

**Strategies**
- Use Pydantic for input validation (fail fast)
- Set timeouts for prediction calls
- Graceful degradation (return default/fallback)
- Proper logging for debugging
- Health checks to verify model is loaded

### Model Versioning

**URL-Based Versioning**
```
/v1/predict
/v2/predict
```

**Header-Based Versioning**
```
X-Model-Version: 1.0.0
```

**Path Parameter Versioning**
```
/models/{model_id}/predict
```

## Production Considerations

### Performance
- Use `uvicorn` with multiple workers for CPU-bound models
- Consider model optimization (quantization, pruning)
- Batch predictions when possible
- Cache predictions for identical inputs (if applicable)

### Monitoring
- Track inference latency (p50, p95, p99)
- Monitor prediction distribution for drift
- Log request/response for debugging
- Health check endpoints (`/health`, `/ready`)

### Scalability
- Horizontal scaling: multiple instances behind load balancer
- Vertical scaling: larger instances for bigger models
- Model serving infrastructure (Kubernetes, cloud services)

## Common Patterns

### Feature Preprocessing in API

**In-API Preprocessing** (Recommended)
- Ensures training/serving consistency
- Single source of truth for transformations
- May duplicate preprocessing logic

**External Preprocessing**
- Client handles preprocessing
- Risk of training/serving skew
- Lighter API server

### Batch vs. Online Prediction

**Online (Real-time)**
- Single prediction per request
- Low latency requirement
- User-facing applications

**Batch**
- Multiple predictions per request
- Higher throughput
- Background jobs, analytics

### Model Metadata Endpoints

Expose model information for debugging and monitoring:
- Model version and timestamp
- Feature schema and types
- Performance characteristics
- Example inputs/outputs

## Best Practices

1. **Separate model logic from API logic** - Keep model code in separate modules
2. **Use dependency injection** - Pass model as dependency, easier testing
3. **Validate inputs strictly** - Pydantic models prevent bad data
4. **Set timeouts** - Prevent hanging requests
5. **Implement health checks** - Readiness vs. liveness probes
6. **Log predictions** - Sample for monitoring and debugging
7. **Version your models** - Track which version served which request
8. **Test thoroughly** - Unit tests for preprocessing, integration tests for endpoints

## Related Topics

- See [software-engineering/notes/fastapi-api-design.md](../../software-engineering/notes/fastapi-api-design.md) for general API design patterns
- See [productionization/] for deployment strategies (when created)
- See [ml-system-design/notes/ml-system-design-overview.md](ml-system-design-overview.md) for broader system design context
