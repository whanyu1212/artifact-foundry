# MLOps Cheatsheet - Interview Prep

**Quick reference for MLOps interviews - deployment, monitoring, and production ML**

---

## Table of Contents

1. [MLOps Lifecycle](#mlops-lifecycle)
2. [CI/CD for ML](#cicd-for-ml)
3. [Deployment Patterns](#deployment-patterns)
4. [Monitoring & Observability](#monitoring--observability)
5. [GitHub Actions Quick Reference](#github-actions-quick-reference)
6. [Production Checklist](#production-checklist)
7. [Interview Questions](#interview-questions)

---

## MLOps Lifecycle

### The ML Production Loop

```
Code → Build → Train → Evaluate → Deploy → Monitor → Retrain
  ↑                                                      ↓
  └──────────────── Feedback Loop ──────────────────────┘
```

### MLOps vs DevOps

| Aspect | DevOps | MLOps |
|--------|--------|-------|
| **Artifacts** | Code, binaries | Code, data, models |
| **Testing** | Unit, integration | + Data validation, model quality |
| **Deployment** | Code deployment | Model + data deployment |
| **Monitoring** | Uptime, latency | + Data drift, model performance |
| **Versioning** | Code (Git) | Code + data + models |
| **Rollback** | Previous code | Previous model version |

### MLOps Maturity Levels

**Level 0 - Manual**:
- Notebooks, manual steps
- No automation
- ❌ Not production-ready

**Level 1 - Automated Training**:
- Training pipeline automated
- Manual deployment
- ⚠️ Good for POCs

**Level 2 - Automated Deployment**:
- CI/CD for models
- Automated testing
- ✅ Production-ready

**Level 3 - Full MLOps**:
- Automated retraining
- Monitoring triggers retraining
- ✅✅ Enterprise-grade

---

## CI/CD for ML

### CI/CD Pipeline Stages

```
1. CODE COMMIT
   ↓
2. CI - Continuous Integration
   ├─ Lint code (flake8, black)
   ├─ Run unit tests (pytest)
   ├─ Data validation (Great Expectations)
   ├─ Model tests (accuracy > threshold)
   └─ Build container

3. CD - Continuous Deployment
   ├─ Deploy to staging
   ├─ Integration tests
   ├─ Canary deployment (10% traffic)
   ├─ Monitor metrics
   └─ Promote to production (100%)

4. CT - Continuous Training
   ├─ Scheduled retraining (weekly)
   ├─ Triggered by drift
   └─ Auto-deploy if better
```

### Testing Pyramid for ML

```
       /\
      /  \  Model Quality Tests (Slow, Few)
     /────\
    /      \  Integration Tests (Medium)
   /────────\
  / Unit Tests \ (Fast, Many)
 /──────────────\
```

**Unit Tests**:
- Feature engineering logic
- Data preprocessing functions
- Model inference code

**Integration Tests**:
- End-to-end pipeline
- API endpoints
- Database connections

**Model Quality Tests**:
- Accuracy on hold-out set
- Fairness metrics
- Inference latency

### Data Validation

**Great Expectations checklist**:
```python
# Schema validation
expect_table_columns_to_match_ordered_list()
expect_column_values_to_not_be_null('user_id')

# Data quality
expect_column_values_to_be_between('age', 0, 120)
expect_column_values_to_be_in_set('status', ['active', 'inactive'])

# Distribution
expect_column_mean_to_be_between('price', 10, 1000)
expect_column_unique_value_count_to_be_between('user_id', 1000, 100000)
```

---

## Deployment Patterns

### 1. Shadow Deployment

```
Production Traffic
    ↓
    ├──→ Old Model (serves users)
    └──→ New Model (logs predictions, NO serving)
```

**Use when**:
- Testing new model in production
- Want to compare predictions
- Zero risk to users

**Interview point**: "Shadow deploy to collect real-world predictions before switching traffic"

### 2. Canary Deployment

```
Production Traffic
    ↓
    ├─ 90% → Old Model
    └─ 10% → New Model (canary)
         ↓
    Monitor metrics for 1-2 days
         ↓
    If good → Increase to 100%
    If bad → Rollback to 0%
```

**Use when**:
- Gradual rollout
- Want to catch issues early
- Can monitor business metrics

**Interview point**: "Canary with 10% traffic, monitor for 24 hours, auto-rollback if error rate > 0.1%"

### 3. Blue-Green Deployment

```
Blue (old)  ──→ 100% traffic
Green (new) ──→ 0% traffic

[Switch]

Blue (old)  ──→ 0% traffic (keep running for rollback)
Green (new) ──→ 100% traffic
```

**Use when**:
- Need instant rollback
- Atomic switch
- Have capacity for 2x infrastructure

### 4. A/B Testing

```
Users randomly split
    ↓
    ├─ 50% → Model A
    └─ 50% → Model B
         ↓
    Compare business metrics
         ↓
    Winner gets 100%
```

**Use when**:
- Testing model impact on business metrics
- Need statistical significance
- Can run for 1-2 weeks

**Statistical significance**:
```
Sample size = (z * σ / MDE)²
Where:
  z = 1.96 (95% confidence)
  σ = standard deviation
  MDE = minimum detectable effect
```

---

## Monitoring & Observability

### What to Monitor

```
┌─────────────────────────────────────┐
│   MODEL PERFORMANCE                 │
│   - Accuracy, Precision, Recall     │
│   - Prediction distribution         │
│   - Business metrics (CTR, revenue) │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│   DATA QUALITY                      │
│   - Feature distributions           │
│   - Missing values                  │
│   - Schema changes                  │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│   SYSTEM HEALTH                     │
│   - Latency (P50, P95, P99)         │
│   - Throughput (QPS)                │
│   - Error rate                      │
│   - Resource usage (CPU, memory)    │
└─────────────────────────────────────┘
```

### Data Drift Types

**1. Covariate Drift (Feature Drift)**
- **P(X) changes**: Input distribution shifts
- **Example**: User demographics change (more mobile users)
- **Detection**: KS test, KL divergence, Wasserstein distance
- **Action**: Retrain with recent data

**2. Concept Drift (Label Drift)**
- **P(Y|X) changes**: Relationship between features and labels changes
- **Example**: User preferences evolve, economic conditions change
- **Detection**: Model performance degradation
- **Action**: Retrain or redesign features

**3. Label Drift**
- **P(Y) changes**: Output distribution changes
- **Example**: More fraud cases than usual
- **Detection**: Compare prediction distribution to training
- **Action**: Check if real change or data issue

### Drift Detection Methods

| Method | Use For | Threshold |
|--------|---------|-----------|
| **KS Test** | Continuous features | p-value < 0.05 |
| **Chi-square** | Categorical features | p-value < 0.05 |
| **KL Divergence** | Distribution distance | > 0.1 |
| **PSI** (Population Stability Index) | Feature stability | > 0.25 (significant drift) |
| **Model-based** | Complex drift | AUC > 0.6 (can distinguish old vs new) |

### Retraining Strategies

```
When to retrain?
├─ TIME-BASED
│  ├─ Daily (news, stock prices)
│  ├─ Weekly (user behavior)
│  └─ Monthly (slow-changing domains)
│
├─ PERFORMANCE-BASED
│  └─ When accuracy drops > 5%
│
└─ DRIFT-BASED
   └─ When drift detected (PSI > 0.25)
```

**Interview point**: "Monitor accuracy weekly, auto-trigger retraining if drops below threshold, validate before deployment"

---

## GitHub Actions Quick Reference

### GitHub-Only vs Cloud-Hybrid

| | 🏠 GitHub-Only | ☁️ Cloud-Hybrid |
|---|---------------|----------------|
| **Training runs on** | GitHub runners | Cloud (Vertex AI, SageMaker) |
| **Compute** | 2-8 vCPUs, CPU only | GPUs, TPUs, unlimited |
| **Max time** | 6 hours | Unlimited |
| **Cost** | Free tier: 2000 min/month | Cloud compute costs |
| **Best for** | Prototypes, small models | Production, large models |

### Quick Workflow Templates

**Test Pipeline**:
```yaml
name: ML Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: pytest tests/
      - run: python validate_data.py
```

**Train & Deploy (GitHub-Only)**:
```yaml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - run: python train.py  # Runs ON GitHub
      - run: python evaluate.py
      - if: accuracy > 0.85
        run: ./deploy.sh
```

**Train & Deploy (Cloud-Hybrid)**:
```yaml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - run: gcloud ai custom-jobs create ...  # Triggers cloud
      - run: gcloud ai custom-jobs wait ...
      - run: ./deploy.sh
```

**Scheduled Retraining**:
```yaml
on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday 2 AM
```

---

## Production Checklist

### Pre-Deployment

- [ ] **Model performance**: Accuracy > threshold on test set
- [ ] **Fairness**: Check for bias across demographics
- [ ] **Latency**: P95 < SLA (e.g., 100ms)
- [ ] **Data validation**: Schema matches, no missing values
- [ ] **Model size**: < memory limit, reasonable inference time
- [ ] **Backward compatibility**: Can handle old feature formats
- [ ] **Rollback plan**: Can revert to previous model quickly
- [ ] **A/B test plan**: Metrics, sample size, duration

### Post-Deployment

- [ ] **Monitoring dashboards**: Set up alerts
- [ ] **Canary metrics**: Error rate, latency normal
- [ ] **Business metrics**: CTR, revenue tracking
- [ ] **Data drift**: Feature distributions stable
- [ ] **Model drift**: Prediction distribution stable
- [ ] **Logging**: Predictions logged for debugging
- [ ] **Incident response**: On-call rotation, runbooks

### Model Card

Document for each model:
- **Purpose**: What problem does it solve?
- **Training data**: Size, source, date range
- **Performance**: Metrics on test set
- **Fairness**: Performance across demographics
- **Limitations**: Known failure modes
- **Intended use**: When to use, when NOT to use
- **Deployment date**: Version, who deployed

---

## Interview Questions

### 1. "How would you deploy a new ML model to production?"

**Answer template**:
```
1. VALIDATE
   - Test accuracy > threshold
   - Check latency < SLA
   - Validate data schema

2. SHADOW DEPLOY
   - Run alongside old model
   - Compare predictions for 1 week
   - No user impact

3. CANARY DEPLOY
   - Route 10% traffic to new model
   - Monitor for 24-48 hours
   - Check error rate, latency, business metrics

4. GRADUAL ROLLOUT
   - If canary succeeds: 25% → 50% → 100%
   - If issues: auto-rollback to old model

5. MONITOR
   - Set up alerts for drift, performance
   - Log predictions for debugging
```

**Follow-up**: "What if latency increases by 20%?"
- "Check if trade-off worth it (accuracy vs latency)"
- "Optimize model (quantization, pruning)"
- "Use model cascading (fast model for easy cases)"

### 2. "How do you know when to retrain your model?"

**Answer**:
```
THREE TRIGGERS:

1. TIME-BASED (Proactive)
   - Weekly/monthly retraining
   - Pros: Predictable, simple
   - Cons: May retrain unnecessarily

2. PERFORMANCE-BASED (Reactive)
   - When accuracy drops > 5%
   - Pros: Only retrain when needed
   - Cons: Reactive, users affected first

3. DRIFT-BASED (Proactive)
   - When feature drift detected (PSI > 0.25)
   - Pros: Catch issues before performance drops
   - Cons: Need drift monitoring infrastructure

RECOMMENDED: Combine all three
- Weekly retraining (baseline)
- + Drift monitoring (proactive)
- + Performance alerts (safety net)
```

### 3. "How would you debug a drop in model performance?"

**Debugging checklist**:
```
1. CHECK DATA
   ├─ Schema changed? (new/missing features)
   ├─ Distribution shifted? (drift)
   ├─ Data quality issues? (nulls, outliers)
   └─ Training vs serving skew?

2. CHECK MODEL
   ├─ Deployed correct version?
   ├─ Preprocessing logic same as training?
   ├─ Model file corrupted?
   └─ Resource constraints? (CPU/memory)

3. CHECK SYSTEM
   ├─ Upstream service down?
   ├─ Feature store lag?
   ├─ Caching issues?
   └─ Traffic pattern changed?

4. ANALYZE ERRORS
   ├─ Which segment degraded? (user type, region, time)
   ├─ Failure patterns? (specific features)
   └─ Compare predictions old vs new model
```

### 4. "How do you ensure reproducibility in ML?"

**Answer**:
```
VERSION EVERYTHING:

1. CODE
   - Git commit SHA
   - Requirements.txt with pinned versions

2. DATA
   - Data versioning (DVC, LakeFS)
   - Record: source, date, preprocessing steps
   - Use BigQuery snapshots (bq://table@timestamp)

3. MODEL
   - Model registry (MLflow, Vertex AI)
   - Save: hyperparameters, training config, metrics

4. ENVIRONMENT
   - Docker containers (pin base image version)
   - Infrastructure as code (Terraform)

5. RANDOM SEEDS
   - Set seeds for reproducibility
   - np.random.seed(42), torch.manual_seed(42)

EXAMPLE:
Training run ID: abc123
- Code: git SHA abc123def
- Data: gs://bucket/data@2024-01-01
- Model: model_v1.2.pkl
- Config: config.yaml (lr=0.01, batch=32)
- Environment: Dockerfile (python:3.9-slim)
```

### 5. "What's your CI/CD pipeline for ML?"

**Answer**:
```
CONTINUOUS INTEGRATION (On PR):
1. Lint code (flake8, black)
2. Run unit tests (pytest)
3. Data validation tests (Great Expectations)
4. Train on subset, check accuracy > baseline
5. Build Docker container

CONTINUOUS DEPLOYMENT (On merge to main):
1. Train model on full dataset (or trigger cloud training)
2. Evaluate on hold-out set
3. If accuracy > threshold:
   - Register model in model registry
   - Deploy to staging
   - Run integration tests
   - Shadow deploy to prod
   - Canary deploy (10%)
   - Monitor for 24h
   - Promote to 100% if metrics good

CONTINUOUS TRAINING (Weekly):
1. Scheduled trigger (cron)
2. Fetch latest data
3. Retrain model
4. Auto-deploy if better than current
```

### 6. "How do you handle model versioning?"

**Answer**:
```
VERSION SCHEME: semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking change (new features, schema change)
- MINOR: New functionality (backward compatible)
- PATCH: Bug fix, retrain

TRACK:
1. Model artifacts (pkl, pb files)
2. Metadata:
   - Training date, data version
   - Hyperparameters, metrics
   - Git commit SHA
   - Who trained it

TOOLS:
- MLflow Model Registry
- Vertex AI Model Registry
- DVC for data versioning

EXAMPLE:
model_v2.3.1
- Version: 2.3.1
- Trained: 2024-01-15
- Data: data_v5 (2024-01-01 to 2024-01-14)
- Accuracy: 0.87
- Git SHA: abc123
- Deployed to prod: 2024-01-16
```

---

## Quick Wins for Interviews

### Impressive Talking Points

1. **"Shadow deploy before canary"**: Zero risk testing in production
2. **"Monitor both model AND business metrics"**: Accuracy isn't everything
3. **"Set up auto-rollback triggers"**: Error rate > 0.1% → rollback
4. **"Use feature store for train/serve consistency"**: Avoid skew
5. **"Drift monitoring triggers retraining"**: Proactive not reactive
6. **"Version everything: code, data, models"**: Reproducibility
7. **"Multi-armed bandits for continuous learning"**: Exploration vs exploitation

### Technologies to Mention

**Orchestration**: Airflow, Kubeflow, Vertex AI Pipelines
**Experiment Tracking**: MLflow, Weights & Biases, Neptune
**Feature Store**: Feast, Tecton, Vertex AI Feature Store
**Monitoring**: Evidently, Seldon, Vertex AI Model Monitoring
**Data Validation**: Great Expectations, TensorFlow Data Validation
**CI/CD**: GitHub Actions, Jenkins, GitLab CI
**Model Registry**: MLflow, Vertex AI, SageMaker

### Metrics to Track

**Model Performance**:
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR
- Business metrics (CTR, revenue, retention)

**System Performance**:
- Latency: P50, P95, P99 (not average!)
- Throughput: QPS, RPM
- Error rate: 4xx, 5xx

**Data Quality**:
- Missing value rate
- Feature drift (PSI, KS test)
- Schema violations

**Cost**:
- $ per prediction
- Compute cost (training + serving)
- Storage cost

---

## Key Takeaways

1. **Test everything**: Code, data, model performance
2. **Deploy gradually**: Shadow → Canary → Full
3. **Monitor continuously**: Performance, drift, system health
4. **Version everything**: Code, data, models, config
5. **Automate retraining**: Time-based + drift-based + performance-based
6. **Have rollback plan**: One-click revert to previous model
7. **Think feedback loops**: How to improve over time

**Golden Rule**: *In production, reliability > accuracy*

---

**Last Updated**: December 2024
