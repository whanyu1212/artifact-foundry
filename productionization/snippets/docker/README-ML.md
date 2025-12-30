# Machine Learning Model Deployment Guide

Simple, production-ready setup for deploying ML models to Kubernetes (GKE) or other container platforms.

## Quick Start

### 1. Local Testing

```bash
# Build and run
docker build -f Dockerfile.ml -t ml-model:latest .
docker run -p 8080:8080 ml-model:latest

# Or use docker-compose
docker compose -f docker-compose.ml.yml up --build

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Health check
curl http://localhost:8080/health
```

### 2. Deploy to GKE

```bash
# Build and push to Google Container Registry
export PROJECT_ID=your-gcp-project
export IMAGE=gcr.io/$PROJECT_ID/ml-model:v1

docker build -f Dockerfile.ml -t $IMAGE .
docker push $IMAGE

# Deploy to GKE
kubectl create deployment ml-model --image=$IMAGE
kubectl expose deployment ml-model --type=LoadBalancer --port=80 --target-port=8080

# Check status
kubectl get pods
kubectl get service ml-model
```

## File Structure

```
your-ml-project/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI app (see app-ml-example.py)
├── models/
│   └── model.pkl            # Your trained model
├── Dockerfile.ml            # Production Dockerfile
├── docker-compose.ml.yml    # Local testing
├── requirements-ml.txt      # Python dependencies
└── .dockerignore           # Exclude unnecessary files
```

## Model Loading Options

### Option 1: Bundle Model in Image (Simplest)

```dockerfile
# In Dockerfile.ml
COPY ./models ./models
```

**Pros**: Simple, self-contained
**Cons**: Large image size, need to rebuild for model updates

### Option 2: Load from Cloud Storage (Recommended for GKE)

```python
# In app/main.py
from google.cloud import storage

def load_model():
    client = storage.Client()
    bucket = client.bucket("your-model-bucket")
    blob = bucket.blob("models/model.pkl")
    blob.download_to_filename("/tmp/model.pkl")
    model = joblib.load("/tmp/model.pkl")
    return model
```

**Pros**: Smaller images, update models without rebuild
**Cons**: Requires cloud credentials, startup latency

### Option 3: Init Container (Kubernetes Pattern)

```yaml
# k8s deployment
initContainers:
- name: model-loader
  image: google/cloud-sdk:alpine
  command: ["gsutil", "cp", "gs://bucket/model.pkl", "/models/"]
  volumeMounts:
  - name: model-volume
    mountPath: /models
```

**Pros**: Separates model loading from app, cacheable
**Cons**: More complex setup

## Production Considerations

### Resource Allocation

Adjust based on your model:

```yaml
# For scikit-learn models (small)
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"

# For deep learning models (large)
resources:
  requests:
    memory: "2Gi"
    cpu: "2000m"
  limits:
    memory: "4Gi"
    cpu: "4000m"

# For GPU inference
resources:
  limits:
    nvidia.com/gpu: 1
```

### Scaling

```bash
# Horizontal Pod Autoscaler
kubectl autoscale deployment ml-model \
  --cpu-percent=70 \
  --min=2 \
  --max=10

# Or with custom metrics (e.g., request latency)
```

### Health Checks

```yaml
# Kubernetes deployment
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Monitoring

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, make_asgi_app

# Metrics
predictions_total = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.post("/predict")
@prediction_latency.time()
async def predict(request: PredictionRequest):
    predictions_total.inc()
    # ... prediction logic

# Expose metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

## Common Issues

### Large Image Size

**Problem**: Image is 2GB+ due to ML dependencies

**Solutions**:
```dockerfile
# Use smaller base images
FROM python:3.11-slim  # Not python:3.11

# Multi-stage builds
FROM python:3.11 as builder
# ... install deps
FROM python:3.11-slim
COPY --from=builder ...

# CPU-only versions
tensorflow-cpu  # Instead of tensorflow (50% smaller)
torch --index-url https://download.pytorch.org/whl/cpu
```

### Slow Startup

**Problem**: Takes 60+ seconds to start

**Solutions**:
- Use lighter models (quantization, pruning)
- Load model lazily (on first request)
- Use model caching (Redis)
- Pre-load model in image build

### Memory Issues

**Problem**: OOMKilled in Kubernetes

**Solutions**:
```python
# Batch processing
@app.post("/batch_predict")
async def batch_predict(requests, batch_size=32):
    for i in range(0, len(requests), batch_size):
        batch = requests[i:i+batch_size]
        yield process_batch(batch)

# Model quantization
import torch
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## Deployment Checklist

Before deploying to production:

- [ ] Model loads successfully in container
- [ ] Health endpoint returns 200
- [ ] Prediction endpoint works with sample data
- [ ] Error handling for invalid inputs
- [ ] Logging configured (stdout/stderr)
- [ ] Resource limits set appropriately
- [ ] Health checks configured (liveness + readiness)
- [ ] Autoscaling configured
- [ ] Monitoring/metrics enabled
- [ ] Load testing completed
- [ ] Security: non-root user, minimal base image
- [ ] Secrets managed properly (not in image)

## Example: Complete GKE Deployment

```bash
# 1. Build and push
docker build -f Dockerfile.ml -t gcr.io/PROJECT/ml-model:v1 .
docker push gcr.io/PROJECT/ml-model:v1

# 2. Create deployment
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: gcr.io/PROJECT/ml-model:v1
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model
spec:
  type: LoadBalancer
  selector:
    app: ml-model
  ports:
  - port: 80
    targetPort: 8080
EOF

# 3. Wait for external IP
kubectl get service ml-model -w

# 4. Test
EXTERNAL_IP=$(kubectl get service ml-model -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
curl -X POST http://$EXTERNAL_IP/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Cost Optimization for GKE

```bash
# Use preemptible nodes for dev/staging
gcloud container node-pools create preemptible-pool \
  --preemptible \
  --cluster=your-cluster

# Use Cloud Run for low-traffic models (cheaper than GKE)
gcloud run deploy ml-model \
  --image gcr.io/PROJECT/ml-model:v1 \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10
```

## Related Documentation

- **Deep dive**: [/productionization/notes/docker-containerization.md](../notes/docker-containerization.md)
- **Quick reference**: [/interview-prep/notes/docker-cheatsheet.md](../../interview-prep/notes/docker-cheatsheet.md)

---

**Keep it simple!** Start with bundled models and scale up complexity only when needed.
