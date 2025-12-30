# MLOps with GitHub Actions

## Overview

**GitHub Actions** is a CI/CD platform that can automate significant parts of your MLOps pipeline. While not a complete MLOps solution on its own, it excels at orchestrating workflows, triggering cloud services, and automating repetitive tasks in the ML lifecycle.

**Key Insight**: GitHub Actions is best used as the **orchestration layer** that coordinates your ML tools, not as the ML platform itself.

---

## Table of Contents

1. [GitHub Actions Basics](#1-github-actions-basics)
2. [Two Approaches: GitHub-Only vs Cloud-Hybrid](#2-two-approaches-github-only-vs-cloud-hybrid)
3. [Testing & Validation](#3-testing--validation)
4. [Training: GitHub-Only Approach](#4-training-github-only-approach)
5. [Training: Cloud-Hybrid Approach](#5-training-cloud-hybrid-approach)
6. [Deployment: GitHub-Only Approach](#6-deployment-github-only-approach)
7. [Deployment: Cloud-Hybrid Approach](#7-deployment-cloud-hybrid-approach)
8. [Monitoring & Retraining](#8-monitoring--retraining)
9. [Best Practices](#9-best-practices)
10. [Complete Examples](#10-complete-examples)
11. [Limitations & When to Use Each Approach](#11-limitations--when-to-use-each-approach)

---

## 1. GitHub Actions Basics

### 1.1 Workflow Structure

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline                    # Workflow name

on:                                  # Triggers
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 2 * * 0'             # Weekly on Sunday 2am

env:                                 # Environment variables
  PYTHON_VERSION: '3.9'
  PROJECT_ID: my-gcp-project

jobs:
  job-name:
    runs-on: ubuntu-latest          # Runner OS
    steps:
      - uses: actions/checkout@v3   # Pre-built action
      - name: Custom step
        run: echo "Hello"           # Shell command
```

### 1.2 Key Concepts

**Workflows**: Automated processes defined in YAML files
**Jobs**: Set of steps that execute on the same runner
**Steps**: Individual tasks (actions or shell commands)
**Runners**: Virtual machines that execute jobs
**Secrets**: Encrypted environment variables (API keys, credentials)
**Artifacts**: Files persisted between jobs

### 1.3 Available Runners

| Runner | vCPUs | RAM | Storage | Cost | Use Case |
|--------|-------|-----|---------|------|----------|
| `ubuntu-latest` | 2 | 7GB | 14GB | Free* | Most ML tasks |
| `ubuntu-latest-4-cores` | 4 | 16GB | 14GB | $0.008/min | Larger models |
| `ubuntu-latest-8-cores` | 8 | 32GB | 14GB | $0.016/min | Heavy computation |
| Self-hosted | Custom | Custom | Custom | Your cost | GPU training |

*Free for public repos; 2,000 min/month for private repos on free tier

**Important Limits**:
- **Max job time**: 6 hours (360 minutes)
- **Max workflow time**: 72 hours
- **Max concurrent jobs**: 20 (free), 60 (Team), 180 (Enterprise)

---

## 2. Two Approaches: GitHub-Only vs Cloud-Hybrid

There are **two main ways** to use GitHub Actions for MLOps. Understanding the difference is crucial for choosing the right approach.

### 2.1 Approach Comparison

| Aspect | 🏠 GitHub-Only | ☁️ Cloud-Hybrid |
|--------|----------------|-----------------|
| **Where training runs** | GitHub Actions runners | Cloud platform (GCP, AWS, Azure) |
| **Compute resources** | 2-8 vCPUs, 7-32GB RAM, CPU only | Any (GPUs, TPUs, 100+ cores) |
| **Max training time** | 6 hours | Unlimited |
| **Cost model** | Free tier: 2000 min/month | Pay for cloud compute |
| **ML tooling** | Bring your own (MLflow, DVC) | Native cloud tools (Vertex AI, SageMaker) |
| **Deployment target** | Any (cloud, on-prem, edge) | Typically same cloud |
| **Best for** | Small models, prototypes, CPU training | Production, large models, GPU training |

### 2.2 Decision Tree

```
Do you need GPU/TPU training?
├─ YES → Use ☁️ Cloud-Hybrid
└─ NO
    └─ Does training take > 6 hours?
        ├─ YES → Use ☁️ Cloud-Hybrid
        └─ NO
            └─ Is dataset > 10GB?
                ├─ YES → Use ☁️ Cloud-Hybrid
                └─ NO
                    └─ Is this production-critical?
                        ├─ YES → Use ☁️ Cloud-Hybrid (for features like monitoring, feature store)
                        └─ NO → Use 🏠 GitHub-Only (simpler, faster to set up)
```

### 2.3 What Runs Where

#### 🏠 GitHub-Only Approach

**Everything runs on GitHub runners:**

```
┌─────────────────────────────────────────────┐
│          GitHub Actions Runner               │
│                                              │
│  1. Checkout code                            │
│  2. Download data (from GCS/S3)              │
│  3. Run training script                      │
│  4. Evaluate model                           │
│  5. Upload model artifact                    │
│  6. Deploy (trigger deployment)              │
│                                              │
│  Optional: Upload to cloud storage           │
└─────────────────────────────────────────────┘
```

**Characteristics:**
- ✅ Simple setup (no cloud authentication complexity)
- ✅ Everything in one place (easy to debug)
- ✅ Good for learning and prototyping
- ❌ Limited compute (no GPUs on free tier)
- ❌ 6-hour timeout
- ❌ Missing ML-specific features (feature store, drift detection)

#### ☁️ Cloud-Hybrid Approach

**GitHub Actions orchestrates, cloud executes:**

```
┌────────────────────────┐         ┌─────────────────────────────┐
│   GitHub Actions       │         │      Cloud Platform         │
│                        │         │                             │
│  1. Checkout code      │         │                             │
│  2. Build container    │────────→│  Container Registry         │
│  3. Trigger job        │────────→│  Training Service           │
│  4. Wait/Monitor       │←────────│    (Vertex AI/SageMaker)    │
│  5. Deploy             │────────→│  Serving Service            │
│                        │         │    (Cloud Run/Lambda)       │
└────────────────────────┘         └─────────────────────────────┘
```

**Characteristics:**
- ✅ Unlimited compute (GPUs, TPUs, distributed)
- ✅ No timeout (train for days)
- ✅ ML-specific features (feature store, monitoring, AutoML)
- ✅ Production-ready infrastructure
- ❌ More complex setup (cloud auth, permissions)
- ❌ Costs money (cloud compute)
- ❌ Harder to debug (logs in cloud)

### 2.4 Hybrid Example: What's Different?

**🏠 GitHub-Only:**
```yaml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Train model
        run: python train.py  # ← Runs ON GitHub runner
```

**☁️ Cloud-Hybrid:**
```yaml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Trigger cloud training
        run: |
          gcloud ai custom-jobs create \  # ← Triggers cloud
            --container-image-uri=gcr.io/my-image
          # GitHub runner just submits the job, then waits
```

**Key Difference**: In GitHub-Only, `python train.py` runs **on the GitHub runner**. In Cloud-Hybrid, GitHub Actions just **triggers** a training job that runs **on cloud infrastructure**.

### 2.5 Can You Mix Both?

**Yes!** A common pattern:

- **Testing**: 🏠 GitHub-Only (fast, cheap)
- **Training**: ☁️ Cloud-Hybrid (powerful, scalable)
- **Deployment**: ☁️ Cloud-Hybrid (production infrastructure)

```yaml
jobs:
  test:
    runs-on: ubuntu-latest  # 🏠 GitHub-Only
    steps:
      - run: pytest tests/

  train:
    runs-on: ubuntu-latest  # ☁️ Cloud-Hybrid
    steps:
      - run: gcloud ai custom-jobs create ...

  deploy:
    runs-on: ubuntu-latest  # ☁️ Cloud-Hybrid
    steps:
      - run: gcloud run deploy ...
```

### 2.6 Visual Indicators Used in This Guide

Throughout this document, you'll see these indicators:

- **🏠 GitHub-Only**: Everything runs on GitHub Actions runners
- **☁️ Cloud-Hybrid**: GitHub Actions triggers cloud services (GCP, AWS, Azure)
- **Both 🏠 & ☁️**: Works for both approaches

**Example**:
- "### 4.1 Scheduled Retraining (🏠 GitHub-Only)" = Training runs ON GitHub runners
- "### 5.1 Trigger Cloud Training (☁️)" = GitHub triggers Vertex AI, training runs in cloud
- "### 8.1 Monitor Performance (Both)" = Monitoring strategy works for both

---

## 3. Testing & Validation

**Note**: Testing typically runs 🏠 **fully on GitHub** for both approaches, since tests are fast and don't need GPUs.

### 3.1 Basic ML Pipeline (🏠 GitHub-Only)

```yaml
name: Basic ML Pipeline

on:
  push:
    branches: [main]

jobs:
  # Stage 1: Test code
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
          cache: 'pip'              # Cache dependencies

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run tests
        run: pytest tests/

  # Stage 2: Train model
  train:
    needs: test                     # Only run if tests pass
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python train.py

      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: models/model.pkl

  # Stage 3: Deploy model
  deploy:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Download model
        uses: actions/download-artifact@v3
        with:
          name: trained-model
          path: models/

      - name: Deploy to production
        run: ./scripts/deploy.sh
```

### 3.2 Conditional Workflows

```yaml
# Different workflows for different branches
name: Environment-Specific Pipeline

on:
  push:
    branches:
      - main
      - develop
      - 'feature/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      # Deploy to production (main branch only)
      - name: Deploy to production
        if: github.ref == 'refs/heads/main'
        run: ./deploy.sh production

      # Deploy to staging (develop branch)
      - name: Deploy to staging
        if: github.ref == 'refs/heads/develop'
        run: ./deploy.sh staging

      # Run tests only (feature branches)
      - name: Test model
        if: startsWith(github.ref, 'refs/heads/feature/')
        run: pytest tests/
```

### 3.3 Matrix Builds (Multiple Configurations)

```yaml
name: Multi-Config Testing

on: [push]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10']
        model-type: [logistic, xgboost, neural-net]

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Test ${{ matrix.model-type }} on ${{ matrix.os }}
        run: pytest tests/test_${{ matrix.model-type }}.py
```

### 3.4 Code Quality Tests

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install linting tools
        run: pip install black flake8 mypy pylint

      - name: Check formatting (black)
        run: black --check .

      - name: Lint (flake8)
        run: flake8 . --max-line-length=100

      - name: Type checking (mypy)
        run: mypy src/

      - name: Code quality (pylint)
        run: pylint src/ --fail-under=8.0
```

### 3.2 Data Validation

```yaml
name: Data Validation

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours

jobs:
  validate-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install great-expectations pandas

      - name: Download latest data
        run: |
          gsutil cp gs://my-bucket/data/latest.csv data/

      - name: Validate data schema
        run: |
          python << EOF
          import great_expectations as ge
          import pandas as pd

          df = pd.read_csv('data/latest.csv')
          ge_df = ge.from_pandas(df)

          # Schema validation
          assert ge_df.expect_table_columns_to_match_ordered_list([
              'user_id', 'feature1', 'feature2', 'label'
          ]).success, "Schema mismatch!"

          # Data quality checks
          assert ge_df.expect_column_values_to_not_be_null('label').success
          assert ge_df.expect_column_values_to_be_between('feature1', 0, 100).success
          assert ge_df.expect_column_values_to_be_in_set('label', [0, 1]).success

          print("✓ All data validations passed!")
          EOF

      - name: Check for data drift
        run: python scripts/check_drift.py

      - name: Notify on failure
        if: failure()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Data validation failed!'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 3.3 Model Testing

```yaml
name: Model Testing

on: [push]

jobs:
  test-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Unit tests
        run: pytest tests/unit -v

      - name: Integration tests
        run: pytest tests/integration -v

      - name: Model performance tests
        run: |
          python << EOF
          import joblib
          from sklearn.metrics import accuracy_score, roc_auc_score
          import pandas as pd

          # Load model and test data
          model = joblib.load('models/model.pkl')
          X_test = pd.read_csv('data/test_features.csv')
          y_test = pd.read_csv('data/test_labels.csv').values.ravel()

          # Make predictions
          y_pred = model.predict(X_test)
          y_proba = model.predict_proba(X_test)[:, 1]

          # Check performance thresholds
          accuracy = accuracy_score(y_test, y_pred)
          auc = roc_auc_score(y_test, y_proba)

          print(f"Accuracy: {accuracy:.3f}")
          print(f"AUC: {auc:.3f}")

          # Fail if below threshold
          assert accuracy >= 0.85, f"Accuracy {accuracy:.3f} below threshold 0.85"
          assert auc >= 0.80, f"AUC {auc:.3f} below threshold 0.80"

          print("✓ Model meets performance requirements!")
          EOF

      - name: Behavioral tests (invariance)
        run: |
          python << EOF
          import joblib
          import numpy as np

          model = joblib.load('models/model.pkl')

          # Test 1: Invariance to feature scaling
          x1 = np.array([[1, 2, 3]])
          x2 = np.array([[10, 20, 30]])  # 10x scaled
          pred1 = model.predict_proba(x1)[0, 1]
          pred2 = model.predict_proba(x2)[0, 1]

          # Predictions should be similar if model handles scaling
          print(f"Pred1: {pred1:.3f}, Pred2: {pred2:.3f}")

          # Test 2: Directional expectation
          # Increasing key feature should increase prediction
          x_low = np.array([[1, 0, 0]])
          x_high = np.array([[10, 0, 0]])
          assert model.predict_proba(x_high)[0, 1] > model.predict_proba(x_low)[0, 1]

          print("✓ Behavioral tests passed!")
          EOF
```

---

## 4. Training: GitHub-Only Approach 🏠

**When to use**: Small models, CPU training, < 6 hour training time, prototyping

In this approach, **training runs directly on GitHub Actions runners**. The model training script executes on the GitHub-provided virtual machine.

### 4.1 Scheduled Retraining (🏠 GitHub-Only)

```yaml
name: Weekly Retraining

on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday at 2 AM UTC
  workflow_dispatch:     # Manual trigger

jobs:
  retrain:
    runs-on: ubuntu-latest-4-cores  # Use larger runner

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Download training data
        run: |
          # Get data from last week
          END_DATE=$(date +%Y-%m-%d)
          START_DATE=$(date -d '7 days ago' +%Y-%m-%d)

          python scripts/fetch_data.py \
            --start-date $START_DATE \
            --end-date $END_DATE \
            --output data/training.csv

      - name: Train model
        run: |
          python train.py \
            --data data/training.csv \
            --output models/model_$(date +%Y%m%d).pkl

      - name: Evaluate model
        id: evaluate
        run: |
          METRICS=$(python evaluate.py --model models/model_$(date +%Y%m%d).pkl)
          echo "metrics=$METRICS" >> $GITHUB_OUTPUT

      - name: Upload model to cloud
        if: success()
        run: |
          gsutil cp models/model_$(date +%Y%m%d).pkl \
            gs://my-bucket/models/

      - name: Update model metadata
        run: |
          python << EOF
          import json
          from datetime import datetime

          metadata = {
              'model_version': datetime.now().strftime('%Y%m%d'),
              'metrics': ${{ steps.evaluate.outputs.metrics }},
              'training_date': datetime.now().isoformat(),
              'git_commit': '${{ github.sha }}'
          }

          with open('model_metadata.json', 'w') as f:
              json.dump(metadata, f, indent=2)
          EOF

      - name: Commit metadata
        run: |
          git config user.name "GitHub Actions Bot"
          git config user.email "actions@github.com"
          git add model_metadata.json
          git commit -m "Update model metadata - $(date +%Y%m%d)"
          git push
```

---

## 5. Training: Cloud-Hybrid Approach ☁️

**When to use**: Large models, GPU/TPU training, > 6 hour training time, production systems

In this approach, **GitHub Actions triggers cloud training jobs**. The actual training runs on cloud infrastructure (Vertex AI, SageMaker, etc.), while GitHub Actions orchestrates the workflow.

### 5.1 Trigger Cloud Training (GCP Vertex AI) ☁️

```yaml
name: Trigger GCP Training

on:
  push:
    branches: [main]
    paths:
      - 'src/**'
      - 'train.py'
      - 'requirements.txt'

jobs:
  trigger-training:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Build training container
        run: |
          gcloud builds submit \
            --tag gcr.io/${{ env.PROJECT_ID }}/trainer:${{ github.sha }} \
            --timeout 20m

      - name: Submit Vertex AI training job
        id: training
        run: |
          JOB_NAME="training-$(date +%Y%m%d-%H%M%S)"

          gcloud ai custom-jobs create \
            --region=us-central1 \
            --display-name=$JOB_NAME \
            --worker-pool-spec=machine-type=n1-highmem-8,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/${{ env.PROJECT_ID }}/trainer:${{ github.sha }} \
            --format="value(name)" > job_id.txt

          echo "job_id=$(cat job_id.txt)" >> $GITHUB_OUTPUT

      - name: Wait for training completion (optional)
        run: |
          gcloud ai custom-jobs stream-logs ${{ steps.training.outputs.job_id }} \
            --region=us-central1

      - name: Get training metrics
        run: |
          python scripts/get_vertex_metrics.py \
            --job-id ${{ steps.training.outputs.job_id }}
```

### 4.2 Hyperparameter Tuning (🏠 GitHub-Only)

```yaml
name: Hyperparameter Tuning

on:
  workflow_dispatch:
    inputs:
      max_trials:
        description: 'Maximum number of trials'
        required: true
        default: '20'

jobs:
  hyperparam-tuning:
    runs-on: ubuntu-latest-4-cores

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt optuna

      - name: Run hyperparameter tuning
        run: |
          python << EOF
          import optuna
          from sklearn.ensemble import RandomForestClassifier
          from sklearn.model_selection import cross_val_score
          import pandas as pd
          import joblib

          # Load data
          X = pd.read_csv('data/X_train.csv')
          y = pd.read_csv('data/y_train.csv').values.ravel()

          def objective(trial):
              params = {
                  'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                  'max_depth': trial.suggest_int('max_depth', 3, 20),
                  'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                  'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
              }

              model = RandomForestClassifier(**params, random_state=42)
              score = cross_val_score(model, X, y, cv=3, scoring='roc_auc').mean()
              return score

          # Run optimization
          study = optuna.create_study(direction='maximize')
          study.optimize(objective, n_trials=${{ github.event.inputs.max_trials }})

          print(f"Best params: {study.best_params}")
          print(f"Best score: {study.best_value:.4f}")

          # Train final model with best params
          best_model = RandomForestClassifier(**study.best_params, random_state=42)
          best_model.fit(X, y)
          joblib.dump(best_model, 'models/best_model.pkl')

          # Save results
          results_df = study.trials_dataframe()
          results_df.to_csv('hyperparam_results.csv', index=False)
          EOF

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: tuning-results
          path: |
            models/best_model.pkl
            hyperparam_results.csv
```

---

## 6. Deployment: Docker & Containers (Both Approaches)

Docker containerization is used in **both approaches** - you build containers on GitHub, then deploy them anywhere (GitHub-hosted, cloud, on-premise).

### 6.1 Docker Build & Push (Both 🏠 & ☁️)

```yaml
name: Build and Deploy

on:
  push:
    branches: [main]
    tags:
      - 'v*'

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to GCR
        uses: docker/login-action@v2
        with:
          registry: gcr.io
          username: _json_key
          password: ${{ secrets.GCP_SA_KEY }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: gcr.io/${{ env.PROJECT_ID }}/model-service
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=sha,prefix={{branch}}-

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 7. Deployment: Cloud-Hybrid Approach ☁️

After building containers, deploy them to cloud platforms for production serving.

### 7.1 Deploy to Cloud Run (GCP) ☁️

```yaml
name: Deploy to Cloud Run

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Deploy to Cloud Run
        id: deploy
        uses: google-github-actions/deploy-cloudrun@v1
        with:
          service: model-service
          region: us-central1
          image: gcr.io/${{ env.PROJECT_ID }}/model-service:${{ github.sha }}
          env_vars: |
            MODEL_VERSION=${{ github.sha }}
            LOG_LEVEL=INFO
          flags: |
            --min-instances=1
            --max-instances=10
            --memory=2Gi
            --cpu=2
            --timeout=300
            --concurrency=80

      - name: Show deployment URL
        run: echo "Deployed to ${{ steps.deploy.outputs.url }}"

      - name: Run smoke tests
        run: |
          URL="${{ steps.deploy.outputs.url }}"

          # Health check
          curl -f "$URL/health" || exit 1

          # Test prediction endpoint
          curl -X POST "$URL/predict" \
            -H "Content-Type: application/json" \
            -d '{"features": [1, 2, 3]}' \
            | jq -e '.prediction' || exit 1

          echo "✓ Smoke tests passed!"
```

### 7.2 Canary Deployment (Cloud Run) ☁️

```yaml
name: Canary Deployment

on:
  push:
    branches: [main]

jobs:
  canary-deploy:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Deploy canary (10% traffic)
        run: |
          # Deploy new revision without traffic
          gcloud run deploy model-service \
            --image gcr.io/${{ env.PROJECT_ID }}/model-service:${{ github.sha }} \
            --region us-central1 \
            --tag canary \
            --no-traffic

          # Route 10% traffic to canary
          gcloud run services update-traffic model-service \
            --region us-central1 \
            --to-revisions canary=10,stable=90

      - name: Monitor canary for 10 minutes
        run: |
          python << EOF
          import time
          import requests
          from google.cloud import monitoring_v3
          from datetime import datetime, timedelta

          def check_canary_metrics():
              client = monitoring_v3.MetricServiceClient()
              project_name = f"projects/${{ env.PROJECT_ID }}"

              # Check error rate
              now = time.time()
              interval = monitoring_v3.TimeInterval({
                  "end_time": {"seconds": int(now)},
                  "start_time": {"seconds": int(now - 600)},  # Last 10 min
              })

              results = client.list_time_series(
                  request={
                      "name": project_name,
                      "filter": 'metric.type="run.googleapis.com/request_count" '
                               'AND resource.label.service_name="model-service" '
                               'AND metric.label.revision_name="canary"',
                      "interval": interval,
                  }
              )

              # Check if error rate is acceptable
              # (Simplified - real implementation would be more robust)
              for result in results:
                  print(f"Canary metrics: {result}")

              return True  # Return False if metrics are bad

          # Monitor for 10 minutes
          for i in range(10):
              print(f"Checking canary health... ({i+1}/10)")
              if not check_canary_metrics():
                  print("❌ Canary metrics degraded! Rolling back...")
                  exit(1)
              time.sleep(60)

          print("✓ Canary looks healthy!")
          EOF

      - name: Promote canary to 100%
        if: success()
        run: |
          gcloud run services update-traffic model-service \
            --region us-central1 \
            --to-latest

          echo "✓ Canary promoted to 100% traffic!"

      - name: Rollback on failure
        if: failure()
        run: |
          gcloud run services update-traffic model-service \
            --region us-central1 \
            --to-revisions stable=100

          echo "❌ Rolled back to stable version"
```

### 7.3 Deploy to Vertex AI Endpoint (GCP) ☁️

```yaml
name: Deploy to Vertex AI

on:
  workflow_dispatch:
    inputs:
      model_path:
        description: 'GCS path to model artifact'
        required: true

jobs:
  deploy-vertex:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install google-cloud-aiplatform

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Upload and deploy model
        run: |
          python << EOF
          from google.cloud import aiplatform

          aiplatform.init(
              project='${{ env.PROJECT_ID }}',
              location='us-central1'
          )

          # Upload model
          model = aiplatform.Model.upload(
              display_name='my-model-${{ github.sha }}',
              artifact_uri='${{ github.event.inputs.model_path }}',
              serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest'
          )

          print(f"Model uploaded: {model.resource_name}")

          # Create or get endpoint
          endpoints = aiplatform.Endpoint.list(
              filter='display_name="my-model-endpoint"'
          )

          if endpoints:
              endpoint = endpoints[0]
              print(f"Using existing endpoint: {endpoint.resource_name}")
          else:
              endpoint = aiplatform.Endpoint.create(
                  display_name='my-model-endpoint'
              )
              print(f"Created endpoint: {endpoint.resource_name}")

          # Deploy with traffic split (canary: 10%)
          endpoint.deploy(
              model=model,
              deployed_model_display_name='v-${{ github.sha }}',
              machine_type='n1-standard-2',
              min_replica_count=1,
              max_replica_count=5,
              traffic_percentage=10,  # Canary
              traffic_split={
                  'v-${{ github.sha }}': 10,
                  'stable': 90
              }
          )

          print(f"✓ Deployed to endpoint: {endpoint.resource_name}")
          print(f"Endpoint URL: https://us-central1-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict")
          EOF
```

---

## 8. Monitoring & Retraining

**Note**: Monitoring can be done in both approaches, but cloud-hybrid gets more built-in features.

### 8.1 Monitor Model Performance (Both 🏠 & ☁️)

```yaml
name: Monitor Model Performance

on:
  schedule:
    - cron: '0 */4 * * *'  # Every 4 hours

jobs:
  monitor:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install pandas numpy scikit-learn evidently

      - name: Fetch production predictions
        run: |
          # Get predictions from last 24 hours
          python scripts/fetch_predictions.py \
            --hours 24 \
            --output predictions.csv

      - name: Fetch ground truth labels
        run: |
          # Get actual outcomes (with lag)
          python scripts/fetch_labels.py \
            --hours 48 \
            --lag 24 \
            --output labels.csv

      - name: Calculate performance metrics
        id: metrics
        run: |
          python << EOF
          import pandas as pd
          import numpy as np
          from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
          import json

          # Load data
          preds = pd.read_csv('predictions.csv')
          labels = pd.read_csv('labels.csv')

          # Merge on ID
          df = preds.merge(labels, on='id', how='inner')

          if len(df) == 0:
              print("No labeled data available yet")
              exit(0)

          # Calculate metrics
          accuracy = accuracy_score(df['true_label'], df['predicted_label'])
          auc = roc_auc_score(df['true_label'], df['prediction_proba'])
          precision = precision_score(df['true_label'], df['predicted_label'])
          recall = recall_score(df['true_label'], df['predicted_label'])

          metrics = {
              'accuracy': float(accuracy),
              'auc': float(auc),
              'precision': float(precision),
              'recall': float(recall),
              'num_samples': len(df)
          }

          print(f"Metrics: {json.dumps(metrics, indent=2)}")

          # Save to output
          with open('current_metrics.json', 'w') as f:
              json.dump(metrics, f, indent=2)

          # Check thresholds
          if accuracy < 0.80:
              print(f"⚠️ Accuracy {accuracy:.3f} below threshold 0.80")
              exit(1)
          if auc < 0.75:
              print(f"⚠️ AUC {auc:.3f} below threshold 0.75")
              exit(1)

          print("✓ Model performance is healthy")
          EOF

      - name: Check for data drift
        run: |
          python << EOF
          import pandas as pd
          from evidently.report import Report
          from evidently.metric_preset import DataDriftPreset

          # Load reference data (training data)
          reference = pd.read_csv('data/reference.csv')

          # Load current production data
          current = pd.read_csv('predictions.csv')

          # Generate drift report
          report = Report(metrics=[DataDriftPreset()])
          report.run(reference_data=reference, current_data=current)
          report.save_html('drift_report.html')

          # Check if drift detected
          drift_results = report.as_dict()
          dataset_drift = drift_results['metrics'][0]['result']['dataset_drift']

          if dataset_drift:
              print("⚠️ Data drift detected!")
              exit(1)
          else:
              print("✓ No significant data drift")
          EOF

      - name: Upload drift report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: drift-report
          path: drift_report.html

      - name: Trigger retraining if needed
        if: failure()
        uses: peter-evans/repository-dispatch@v2
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          event-type: trigger-retraining
          client-payload: '{"reason": "performance_degradation"}'

      - name: Send alert
        if: failure()
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              "text": "🚨 Model Performance Alert",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Model performance degradation detected*\n\nMetrics below threshold or data drift detected."
                  }
                }
              ]
            }
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 8.2 Auto-Retraining on Performance Degradation (Both 🏠 & ☁️)

```yaml
name: Auto Retrain

on:
  repository_dispatch:
    types: [trigger-retraining]

jobs:
  retrain:
    runs-on: ubuntu-latest-4-cores

    steps:
      - uses: actions/checkout@v3

      - name: Log retraining trigger
        run: |
          echo "Retraining triggered by: ${{ github.event.client_payload.reason }}"

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Fetch fresh training data
        run: |
          # Get data from last 30 days
          python scripts/fetch_training_data.py --days 30

      - name: Train new model
        run: |
          python train.py \
            --data data/training.csv \
            --output models/model_retrained_$(date +%Y%m%d).pkl

      - name: Evaluate new model
        id: evaluate
        run: |
          python evaluate.py \
            --model models/model_retrained_$(date +%Y%m%d).pkl \
            --output evaluation_results.json

          # Check if new model is better
          python << EOF
          import json

          with open('evaluation_results.json') as f:
              new_metrics = json.load(f)

          with open('current_metrics.json') as f:
              old_metrics = json.load(f)

          improvement = new_metrics['auc'] - old_metrics['auc']

          if improvement > 0.01:  # At least 1% improvement
              print(f"✓ New model improved AUC by {improvement:.3f}")
              print("deploy=true" >> $GITHUB_OUTPUT)
          else:
              print(f"⚠️ New model did not improve significantly")
              print("deploy=false" >> $GITHUB_OUTPUT)
          EOF

      - name: Deploy if improved
        if: steps.evaluate.outputs.deploy == 'true'
        run: |
          # Upload model
          gsutil cp models/model_retrained_$(date +%Y%m%d).pkl \
            gs://my-bucket/models/latest.pkl

          # Trigger deployment workflow
          gh workflow run deploy.yml \
            --ref main \
            -f model_path=gs://my-bucket/models/latest.pkl
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## 9. Best Practices

### 9.1 Security

**1. Never commit secrets**:
```yaml
# ❌ BAD
env:
  API_KEY: sk-1234567890abcdef

# ✅ GOOD
env:
  API_KEY: ${{ secrets.API_KEY }}
```

**2. Use least-privilege permissions**:
```yaml
permissions:
  contents: read      # Read repo
  pull-requests: write # Comment on PRs
  id-token: write     # OIDC authentication
```

**3. Use OIDC for cloud auth** (no long-lived keys):
```yaml
- uses: google-github-actions/auth@v1
  with:
    workload_identity_provider: 'projects/123/locations/global/workloadIdentityPools/my-pool/providers/my-provider'
    service_account: 'my-service-account@project.iam.gserviceaccount.com'
```

### 9.2 Cost Optimization

**1. Use caching**:
```yaml
- name: Cache pip dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

- name: Cache trained models
  uses: actions/cache@v3
  with:
    path: models/
    key: model-${{ hashFiles('train.py') }}-${{ hashFiles('data/**') }}
```

**2. Use concurrency limits** (cancel old runs):
```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

**3. Conditional jobs**:
```yaml
- name: Skip if no code changes
  if: |
    !contains(github.event.head_commit.message, '[skip ci]') &&
    (contains(github.event.head_commit.modified, 'src/') ||
     contains(github.event.head_commit.modified, 'train.py'))
```

### 9.3 Debugging

**1. Enable debug logging**:
```yaml
# In workflow file
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

**2. Use tmate for SSH debugging**:
```yaml
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
  timeout-minutes: 30
```

**3. Save logs as artifacts**:
```yaml
- name: Upload logs
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: logs
    path: |
      *.log
      logs/
```

### 9.4 Testing Workflows Locally

Use **act** to test workflows locally:

```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run workflow
act push  # Simulate push event

# Run specific job
act -j test

# With secrets
act -s GITHUB_TOKEN=xxx
```

---

## 10. Complete Examples

### 10.1 End-to-End ML Pipeline

```yaml
name: Complete ML Pipeline

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 2 * * 0'
  workflow_dispatch:

env:
  PYTHON_VERSION: '3.9'
  PROJECT_ID: my-gcp-project

jobs:
  # ============================================
  # STAGE 1: CODE QUALITY & TESTING
  # ============================================
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Lint with flake8
        run: flake8 src/ tests/

      - name: Format check with black
        run: black --check src/ tests/

      - name: Type check with mypy
        run: mypy src/

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run unit tests
        run: pytest tests/unit -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  # ============================================
  # STAGE 2: DATA VALIDATION
  # ============================================
  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install dependencies
        run: pip install great-expectations pandas

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Validate data
        run: python scripts/validate_data.py

  # ============================================
  # STAGE 3: TRAINING
  # ============================================
  train:
    needs: [code-quality, unit-tests, data-validation]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' || github.event_name == 'schedule'

    outputs:
      model-id: ${{ steps.training.outputs.model_id }}

    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Build training container
        run: |
          gcloud builds submit \
            --tag gcr.io/${{ env.PROJECT_ID }}/trainer:${{ github.sha }}

      - name: Submit Vertex AI training job
        id: training
        run: |
          JOB_ID=$(gcloud ai custom-jobs create \
            --region=us-central1 \
            --display-name=training-$(date +%Y%m%d-%H%M%S) \
            --worker-pool-spec=machine-type=n1-highmem-8,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/${{ env.PROJECT_ID }}/trainer:${{ github.sha }} \
            --format="value(name)")

          echo "model_id=$JOB_ID" >> $GITHUB_OUTPUT

  # ============================================
  # STAGE 4: EVALUATION
  # ============================================
  evaluate:
    needs: train
    runs-on: ubuntu-latest

    outputs:
      should-deploy: ${{ steps.check.outputs.deploy }}

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Download model
        run: |
          gsutil cp gs://my-bucket/models/latest.pkl models/

      - name: Evaluate model
        run: |
          python evaluate.py \
            --model models/latest.pkl \
            --test-data data/test.csv \
            --output metrics.json

      - name: Check deployment criteria
        id: check
        run: |
          python << EOF
          import json

          with open('metrics.json') as f:
              metrics = json.load(f)

          # Check thresholds
          should_deploy = (
              metrics['accuracy'] >= 0.85 and
              metrics['auc'] >= 0.80
          )

          print(f"deploy={str(should_deploy).lower()}" >> $GITHUB_OUTPUT)
          EOF

  # ============================================
  # STAGE 5: DEPLOYMENT
  # ============================================
  deploy:
    needs: evaluate
    if: needs.evaluate.outputs.should-deploy == 'true'
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Build serving container
        run: |
          docker build -t gcr.io/${{ env.PROJECT_ID }}/model-service:${{ github.sha }} .
          docker push gcr.io/${{ env.PROJECT_ID }}/model-service:${{ github.sha }}

      - name: Deploy to Cloud Run (canary)
        run: |
          gcloud run deploy model-service \
            --image gcr.io/${{ env.PROJECT_ID }}/model-service:${{ github.sha }} \
            --region us-central1 \
            --tag canary \
            --no-traffic

          gcloud run services update-traffic model-service \
            --region us-central1 \
            --to-revisions canary=10,stable=90

      - name: Run smoke tests
        run: python scripts/smoke_test.py

      - name: Promote to 100%
        if: success()
        run: |
          gcloud run services update-traffic model-service \
            --region us-central1 \
            --to-latest

      - name: Rollback on failure
        if: failure()
        run: |
          gcloud run services update-traffic model-service \
            --region us-central1 \
            --to-revisions stable=100

  # ============================================
  # STAGE 6: NOTIFICATION
  # ============================================
  notify:
    needs: [train, evaluate, deploy]
    if: always()
    runs-on: ubuntu-latest

    steps:
      - name: Send Slack notification
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              "text": "ML Pipeline ${{ job.status }}",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*ML Pipeline Result*\n\nStatus: ${{ job.status }}\nCommit: ${{ github.sha }}\nAuthor: ${{ github.actor }}"
                  }
                }
              ]
            }
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 10.2 PR-Based Model Review

```yaml
name: Model Review on PR

on:
  pull_request:
    paths:
      - 'src/**'
      - 'train.py'

jobs:
  train-and-compare:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Fetch all history

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      # Train model with PR changes
      - name: Train model (PR version)
        run: |
          python train.py --output models/model_pr.pkl
          python evaluate.py --model models/model_pr.pkl --output metrics_pr.json

      # Train model with main branch code
      - name: Checkout main branch
        run: |
          git checkout origin/main -- train.py src/

      - name: Train model (main version)
        run: |
          python train.py --output models/model_main.pkl
          python evaluate.py --model models/model_main.pkl --output metrics_main.json

      # Compare models
      - name: Compare models
        id: compare
        run: |
          python << EOF
          import json

          with open('metrics_pr.json') as f:
              pr_metrics = json.load(f)

          with open('metrics_main.json') as f:
              main_metrics = json.load(f)

          # Calculate improvements
          acc_diff = pr_metrics['accuracy'] - main_metrics['accuracy']
          auc_diff = pr_metrics['auc'] - main_metrics['auc']

          # Create comment
          comment = f"""
          ## 📊 Model Performance Comparison

          | Metric | Main | PR | Diff |
          |--------|------|-----|------|
          | Accuracy | {main_metrics['accuracy']:.4f} | {pr_metrics['accuracy']:.4f} | {acc_diff:+.4f} |
          | AUC | {main_metrics['auc']:.4f} | {pr_metrics['auc']:.4f} | {auc_diff:+.4f} |

          {'✅ Performance improved!' if auc_diff > 0 else '⚠️ Performance regression detected'}
          """

          # Save for comment
          with open('comment.txt', 'w') as f:
              f.write(comment)
          EOF

      - name: Comment on PR
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const comment = fs.readFileSync('comment.txt', 'utf8');

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

---

## 11. Limitations & When to Use Each Approach

### 11.1 GitHub Actions Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| **6-hour job timeout** | Can't train large models | Trigger cloud training |
| **No GPU runners** (free tier) | Slow deep learning | Use self-hosted or cloud |
| **Limited storage** (14GB) | Can't handle large datasets | Download data on-demand |
| **No ML-specific features** | Need external tools | Integrate MLflow, W&B |
| **Network egress costs** | Expensive data transfers | Cache datasets, use cloud |

### 11.2 When to Use Alternatives

**Use Vertex AI Pipelines** instead when:
- Complex DAGs with many dependencies
- Need data lineage and model provenance
- Require built-in ML features (feature store, monitoring)
- Running on GCP infrastructure

**Use Airflow/Prefect** instead when:
- Complex scheduling requirements
- Dynamic pipeline generation
- Need advanced monitoring and retries
- Cross-platform orchestration

**Use Jenkins** instead when:
- On-premise infrastructure
- Existing Jenkins setup
- Need complete control over execution environment

### 11.3 Hybrid Approach (Recommended)

```
GitHub Actions:
├─ Code testing
├─ Build containers
├─ Trigger cloud pipelines
└─ Deploy services

Cloud Platform (GCP/AWS):
├─ Heavy model training
├─ Feature store
├─ Model serving
└─ Monitoring

External Tools:
├─ MLflow (experiment tracking)
├─ Weights & Biases (visualization)
└─ Great Expectations (data validation)
```

---

## Key Takeaways

1. **GitHub Actions is great for orchestration**, not the entire ML platform
2. **Trigger cloud training** for heavy workloads (GPU, long-running)
3. **Automate testing** at every stage (code, data, model)
4. **Use caching** aggressively to reduce costs and time
5. **Implement gradual rollouts** (canary, blue-green) for safety
6. **Monitor in production** and trigger retraining automatically
7. **Integrate with cloud services** for ML-specific features
8. **Keep workflows modular** for reusability and maintenance

---

## Further Resources

### Official Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Actions for ML Blog Post](https://github.blog/2020-06-17-using-github-actions-for-mlops-data-science/)

### Example Repositories
- [MLOps with GitHub Actions Examples](https://github.com/machine-learning-apps/MLOps-demo)
- [Full Stack Deep Learning - MLOps](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs)

### Complementary Tools
- [DVC](https://dvc.org/) - Data version control
- [MLflow](https://mlflow.org/) - Experiment tracking
- [Weights & Biases](https://wandb.ai/) - Experiment visualization
- [Great Expectations](https://greatexpectations.io/) - Data validation

### Courses
- [Full Stack Deep Learning - MLOps](https://fullstackdeeplearning.com/)
- [MLOps Specialization (Coursera)](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)

---

**Last Updated**: December 2024
