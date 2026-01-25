"""
Batch Prediction Endpoints for ML Models

Demonstrates patterns for handling batch predictions efficiently:
1. Synchronous batch - Process all at once, return when done
2. Asynchronous batch - Return task ID immediately, process in background
3. Streaming batch - Stream results as they're computed

Trade-offs:
- Sync: Simple, blocking, good for small batches
- Async: Non-blocking, good for large batches or slow models
- Streaming: Progressive results, good UX for long-running jobs

Key Concepts:
- Batch processing improves throughput (vectorization, GPU utilization)
- Background tasks for async processing
- Task tracking and status endpoints

Run:
    uvicorn batch_prediction:app --reload
"""

from contextlib import asynccontextmanager
from typing import List, Dict, Optional
from uuid import uuid4
import time

import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# Mock model
class BatchModel:
    """Model that benefits from batch processing."""

    def predict_single(self, features: List[float]) -> float:
        """Single prediction - inefficient for multiple samples."""
        time.sleep(0.1)  # Simulate model inference time
        return float(np.sum(features) * 0.5)

    def predict_batch(self, batch_features: List[List[float]]) -> List[float]:
        """
        Batch prediction - much more efficient.

        In practice, this would use:
        - NumPy vectorization
        - GPU batch inference (PyTorch, TensorFlow)
        - Parallel processing

        Example efficiency gain:
        - Single: 10 predictions = 10 * 100ms = 1000ms
        - Batch: 10 predictions = 200ms (vectorized)
        """
        time.sleep(0.2)  # Simulates batch processing (faster than N singles)
        return [float(np.sum(features) * 0.5) for features in batch_features]


# In-memory task store (use Redis/DB in production)
TASKS: Dict[str, dict] = {}


# Schemas
class BatchPredictionRequest(BaseModel):
    samples: List[List[float]] = Field(
        ...,
        description="List of feature vectors to predict",
        min_length=1,
        example=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    num_samples: int
    processing_time_ms: float


class AsyncTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[float] = None  # 0.0 to 1.0
    result: Optional[List[float]] = None
    error: Optional[str] = None


# Application setup
MODEL: Optional[BatchModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    print("Loading batch model...")
    MODEL = BatchModel()
    yield
    print("Shutting down...")


app = FastAPI(
    title="Batch Prediction API",
    version="1.0.0",
    lifespan=lifespan
)


@app.post("/predict/sync", response_model=BatchPredictionResponse)
def predict_batch_sync(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Strategy 1: Synchronous Batch Prediction

    Process entire batch and return when done.

    Best for:
    - Small batches (< 1000 samples)
    - Fast models (< 5s total processing time)
    - Simple clients that can wait

    Limitations:
    - Blocks request thread until complete
    - May timeout for large batches
    - Client must wait for entire batch

    Example:
        POST /predict/sync
        {"samples": [[1.0, 2.0], [3.0, 4.0]]}
        => {"predictions": [1.5, 3.5], "num_samples": 2, ...}
    """
    start = time.perf_counter()

    # Batch processing is more efficient than N single predictions
    predictions = MODEL.predict_batch(request.samples)

    processing_time = (time.perf_counter() - start) * 1000

    return BatchPredictionResponse(
        predictions=predictions,
        num_samples=len(predictions),
        processing_time_ms=processing_time
    )


@app.post("/predict/async", response_model=AsyncTaskResponse)
def predict_batch_async(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
) -> AsyncTaskResponse:
    """
    Strategy 2: Asynchronous Batch Prediction

    Return task ID immediately, process in background.

    Best for:
    - Large batches (> 1000 samples)
    - Slow models (> 5s processing time)
    - Clients that can poll for results

    Flow:
    1. Client submits batch, receives task_id
    2. Server processes in background
    3. Client polls /tasks/{task_id} for status
    4. When complete, client retrieves results

    Example:
        POST /predict/async {"samples": [[1.0, 2.0], ...]}
        => {"task_id": "abc123", "status": "pending", ...}

        GET /tasks/abc123
        => {"status": "processing", "progress": 0.5, ...}

        GET /tasks/abc123 (later)
        => {"status": "completed", "result": [1.5, ...], ...}
    """
    # Generate unique task ID
    task_id = str(uuid4())

    # Initialize task tracking
    TASKS[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "result": None,
        "error": None,
        "num_samples": len(request.samples)
    }

    # Schedule background processing
    background_tasks.add_task(
        process_batch_async,
        task_id,
        request.samples
    )

    return AsyncTaskResponse(
        task_id=task_id,
        status="pending",
        message=f"Processing {len(request.samples)} samples in background"
    )


def process_batch_async(task_id: str, samples: List[List[float]]) -> None:
    """
    Background task for async batch processing.

    Updates task status and progress as it processes.
    In production, this would be a Celery task or similar.
    """
    try:
        # Update status to processing
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["progress"] = 0.1

        # Simulate progress tracking
        # In practice, process in chunks and update progress
        predictions = MODEL.predict_batch(samples)

        TASKS[task_id]["progress"] = 1.0

        # Store result
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["result"] = predictions

    except Exception as e:
        # Handle errors
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Check status of async batch task.

    Clients poll this endpoint to check progress and get results.

    Status values:
    - pending: Task queued, not started
    - processing: Task is running
    - completed: Task finished successfully
    - failed: Task encountered an error
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    task = TASKS[task_id]

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        result=task["result"],
        error=task["error"]
    )


@app.post("/predict/stream")
async def predict_batch_stream(request: BatchPredictionRequest):
    """
    Strategy 3: Streaming Batch Prediction

    Stream results back as they're computed.

    Best for:
    - Long-running batch jobs
    - UI that can show progressive results
    - Large batches where partial results are useful

    Returns:
        Server-Sent Events (SSE) stream of predictions

    Example:
        POST /predict/stream {"samples": [[1.0, 2.0], [3.0, 4.0]]}
        => (streaming response)
        data: {"index": 0, "prediction": 1.5}
        data: {"index": 1, "prediction": 3.5}
        data: {"done": true}

    Client example (JavaScript):
        const eventSource = new EventSource('/predict/stream');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data);
        };
    """
    async def generate():
        """Generator function that yields predictions one by one."""
        import json

        for idx, features in enumerate(request.samples):
            # Predict one sample at a time
            prediction = MODEL.predict_single(features)

            # Yield result as JSON
            yield f"data: {json.dumps({'index': idx, 'prediction': prediction})}\n\n"

        # Signal completion
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.get("/")
def root():
    """API documentation."""
    return {
        "message": "Batch Prediction API",
        "endpoints": {
            "sync": "POST /predict/sync - Synchronous batch (blocking)",
            "async": "POST /predict/async - Async batch (background task)",
            "stream": "POST /predict/stream - Streaming batch (progressive)",
            "status": "GET /tasks/{task_id} - Check async task status"
        },
        "comparison": {
            "sync": "Simple, blocks until done, good for small batches",
            "async": "Non-blocking, poll for results, good for large batches",
            "stream": "Progressive results, good UX, needs special client"
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("Batch Prediction Patterns")
    print("="*60)
    print("\nTest with:")
    print('  # Sync')
    print('  curl -X POST "http://localhost:8000/predict/sync" \\')
    print("       -H 'Content-Type: application/json' \\")
    print('       -d \'{"samples": [[1,2], [3,4], [5,6]]}\'')
    print('\n  # Async')
    print('  curl -X POST "http://localhost:8000/predict/async" \\')
    print("       -H 'Content-Type: application/json' \\")
    print('       -d \'{"samples": [[1,2], [3,4]]}\'')
    print('  curl "http://localhost:8000/tasks/{TASK_ID}"')
    print("="*60 + "\n")

    uvicorn.run(
        "batch_prediction:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
