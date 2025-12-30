# ML System Design on Google Cloud Platform (GCP)

## Overview

This guide maps ML system design concepts to **Google Cloud Platform (GCP)** services. It complements the general [ML System Design Overview](ml-system-design-overview.md) by providing concrete GCP implementations for each component of the ML lifecycle.

**Why GCP for ML?**
- **Unified platform**: Vertex AI provides end-to-end ML tools
- **BigQuery**: Powerful data warehouse with built-in ML (BQML)
- **AutoML**: Quick prototypes without deep ML expertise
- **TPUs**: Google's custom ML accelerators
- **Integration**: Tight integration between GCP services

---

## GCP ML Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                     Data Sources                            │
│  Cloud SQL, Firestore, Cloud Storage, External APIs        │
└──────────────────────┬─────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────────┐
│                  Data Processing                            │
│  BigQuery, Dataflow, Dataproc, Pub/Sub                     │
└──────────────────────┬─────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────────┐
│              Feature Engineering & Storage                  │
│  Vertex AI Feature Store, BigQuery                         │
└──────────────────────┬─────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────────┐
│                Model Development                            │
│  Vertex AI Training, Workbench, AutoML                     │
└──────────────────────┬─────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────────┐
│              Experiment Tracking                            │
│  Vertex AI Experiments, Vertex ML Metadata                 │
└──────────────────────┬─────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────────┐
│              Model Serving                                  │
│  Vertex AI Prediction, Cloud Run, Cloud Functions          │
└──────────────────────┬─────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────────┐
│              Monitoring & Operations                        │
│  Vertex AI Model Monitoring, Cloud Monitoring, Logging     │
└────────────────────────────────────────────────────────────┘
```

---

## 1. Data Storage & Management

### 1.1 Data Storage Options

| GCP Service | Use Case | When to Use | Cost Profile |
|-------------|----------|-------------|--------------|
| **Cloud Storage** | Data lake, raw data, model artifacts | Unstructured data, backups, data archives | Cheapest ($0.02/GB/month standard) |
| **BigQuery** | Data warehouse, analytics | Structured data, SQL queries, large-scale analytics | Pay per query ($5/TB scanned) |
| **Cloud SQL** | Transactional database | OLTP, relational data, < 64TB | Pay per instance ($0.0150/hour) |
| **Firestore** | NoSQL document DB | Mobile/web apps, real-time sync | Pay per read/write |
| **Bigtable** | Wide-column NoSQL | High-throughput, low-latency, > 1TB | Pay per node ($0.65/hour/node) |
| **Spanner** | Global relational DB | Strong consistency, global scale | Most expensive |

**Typical ML Data Architecture**:
```
Raw Data → Cloud Storage (data lake)
    ↓
ETL/Processing → Dataflow/Dataproc
    ↓
Analytics/Features → BigQuery (data warehouse)
    ↓
Online Features → Vertex AI Feature Store
```

### 1.2 BigQuery for ML

**BigQuery ML (BQML)** - Train models directly in SQL:

```sql
-- Create a classification model
CREATE OR REPLACE MODEL `project.dataset.model_name`
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['label']
) AS
SELECT
  feature1,
  feature2,
  label
FROM `project.dataset.training_data`;

-- Make predictions
SELECT
  predicted_label,
  predicted_label_probs
FROM ML.PREDICT(MODEL `project.dataset.model_name`,
  (SELECT feature1, feature2 FROM `project.dataset.test_data`));
```

**When to use BQML**:
- ✅ Quick prototyping
- ✅ Data already in BigQuery
- ✅ Standard models (linear, logistic, DNN, boosted trees)
- ❌ Complex custom models
- ❌ Need for custom training loops

**BigQuery Features**:
- **Partitioning**: Reduce costs by partitioning by date/timestamp
- **Clustering**: Improve query performance (group related data)
- **Materialized views**: Pre-compute aggregations
- **Scheduled queries**: Automate data pipelines

---

## 2. Data Processing

### 2.1 Batch Processing

**Dataflow (Apache Beam)**:
- Unified batch + stream processing
- Fully managed, auto-scaling
- Python and Java SDKs

```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Define pipeline
with beam.Pipeline(options=PipelineOptions()) as p:
    (p
     | 'Read' >> beam.io.ReadFromText('gs://bucket/input.txt')
     | 'Transform' >> beam.Map(lambda x: x.upper())
     | 'Write' >> beam.io.WriteToText('gs://bucket/output.txt'))
```

**When to use Dataflow**:
- Complex ETL pipelines
- Need for both batch and streaming
- Apache Beam compatibility required

**Dataproc (Managed Hadoop/Spark)**:
- Fully managed Spark and Hadoop clusters
- Good for existing Spark jobs
- Ephemeral clusters (spin up for job, shut down)

```bash
# Submit Spark job
gcloud dataproc jobs submit pyspark \
    gs://bucket/preprocess.py \
    --cluster=my-cluster \
    --region=us-central1
```

**When to use Dataproc**:
- Existing Spark/Hadoop code
- Need HDFS ecosystem tools
- Large-scale batch processing

### 2.2 Stream Processing

**Pub/Sub** - Event streaming:
- Publish-subscribe messaging
- At-least-once delivery
- Automatically scales

```python
from google.cloud import pubsub_v1

# Publish messages
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path('project-id', 'topic-name')
future = publisher.publish(topic_path, b'message data')

# Subscribe to messages
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path('project-id', 'subscription-name')
subscriber.subscribe(subscription_path, callback=callback_function)
```

**Dataflow for Streaming**:
- Process Pub/Sub messages in real-time
- Windowing, late data handling
- Exactly-once processing

**Architecture for Real-time Features**:
```
Events → Pub/Sub → Dataflow → Bigtable/Feature Store → Model Serving
```

---

## 3. Feature Engineering

### 3.1 Vertex AI Feature Store

**Purpose**: Centralized, low-latency feature serving

**Architecture**:
```
Offline Features (BigQuery/Dataflow)
    ↓
Feature Store
    ├─→ Online Serving (low latency, < 10ms)
    └─→ Offline Serving (training data, batch)
```

**Creating a Feature Store**:

```python
from google.cloud import aiplatform

# Create featurestore
featurestore = aiplatform.Featurestore.create(
    featurestore_id='user_features',
    online_store_fixed_node_count=1
)

# Create entity type (e.g., User)
entity_type = featurestore.create_entity_type(
    entity_type_id='users',
    description='User entity'
)

# Create features
entity_type.create_feature(
    feature_id='age',
    value_type='INT64'
)

# Ingest features from BigQuery
entity_type.ingest_from_bq(
    feature_ids=['age', 'country'],
    bq_source_uri='bq://project.dataset.table'
)
```

**Reading Features**:

```python
# Online serving (real-time prediction)
features = entity_type.read(
    entity_ids=['user_123'],
    feature_ids=['age', 'country']
)

# Offline serving (training)
entity_type.batch_serve_to_bq(
    bq_destination_output_uri='bq://project.dataset.training_data',
    read_instances_uri='bq://project.dataset.entity_ids'
)
```

**Benefits**:
- ✅ Point-in-time correctness (no temporal leakage)
- ✅ Online/offline consistency
- ✅ Feature versioning
- ✅ Low-latency serving (< 10ms)

### 3.2 Feature Engineering in BigQuery

**Common Patterns**:

```sql
-- Time-based features
SELECT
  user_id,
  EXTRACT(HOUR FROM timestamp) AS hour_of_day,
  EXTRACT(DAYOFWEEK FROM timestamp) AS day_of_week,
  DATE_DIFF(CURRENT_DATE(), user_signup_date, DAY) AS days_since_signup,

-- Aggregation features (30-day window)
  COUNT(*) OVER (
    PARTITION BY user_id
    ORDER BY timestamp
    RANGE BETWEEN INTERVAL 30 DAY PRECEDING AND CURRENT ROW
  ) AS purchases_last_30d,

-- Categorical encoding
  CASE
    WHEN category = 'A' THEN 1
    WHEN category = 'B' THEN 2
    ELSE 0
  END AS category_encoded

FROM transactions;
```

**BigQuery ML Feature Transformations**:
```sql
CREATE OR REPLACE MODEL `project.dataset.model`
TRANSFORM(
  ML.QUANTILE_BUCKETIZE(price, 10) OVER() AS price_bucket,
  ML.STANDARD_SCALER(age) OVER() AS age_normalized,
  ML.FEATURE_CROSS(STRUCT(city, product)) AS city_product
)
OPTIONS(model_type='LINEAR_REG') AS
SELECT * FROM training_data;
```

---

## 4. Model Development

### 4.1 Vertex AI Workbench

**Managed Jupyter notebooks** for experimentation:
- Pre-installed ML frameworks (TensorFlow, PyTorch, scikit-learn)
- Integration with Git, GCS, BigQuery
- Managed instances (auto-shutdown to save costs)

**Best Practices**:
- Use **user-managed notebooks** for full control
- Use **managed notebooks** for ease of use
- Enable idle shutdown to reduce costs
- Store notebooks in Git (not just in instance)

### 4.2 Vertex AI Training

**Custom Training Jobs**:

```python
from google.cloud import aiplatform

# Define training job
job = aiplatform.CustomTrainingJob(
    display_name='my-training-job',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training/tf-cpu.2-11:latest',
    requirements=['pandas', 'scikit-learn'],
    model_serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-11:latest'
)

# Run training
model = job.run(
    dataset=dataset,
    replica_count=1,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    base_output_dir='gs://bucket/output'
)
```

**Distributed Training**:

```python
# Multi-worker training
job.run(
    replica_count=4,  # 4 workers
    machine_type='n1-highmem-8',
    reduction_server_replica_count=1  # For parameter servers
)
```

**Hyperparameter Tuning**:

```python
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# Define hyperparameter spec
hpt_job = aiplatform.HyperparameterTuningJob(
    display_name='hpt-job',
    custom_job=job,
    metric_spec={'accuracy': 'maximize'},
    parameter_spec={
        'learning_rate': hpt.DoubleParameterSpec(min=0.001, max=0.1, scale='log'),
        'batch_size': hpt.DiscreteParameterSpec(values=[32, 64, 128])
    },
    max_trial_count=20,
    parallel_trial_count=5
)

hpt_job.run()
```

### 4.3 AutoML

**Quick model development without code**:

```python
# AutoML Tables (structured data)
dataset = aiplatform.TabularDataset.create(
    display_name='my-dataset',
    bq_source='bq://project.dataset.table'
)

job = aiplatform.AutoMLTabularTrainingJob(
    display_name='automl-training',
    optimization_prediction_type='classification',
    optimization_objective='maximize-au-prc'
)

model = job.run(
    dataset=dataset,
    target_column='label',
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    budget_milli_node_hours=1000  # Max training time
)
```

**When to use AutoML**:
- ✅ Quick prototyping
- ✅ No ML expertise needed
- ✅ Standard tasks (classification, regression, forecasting)
- ❌ Need custom architectures
- ❌ Budget constraints (can be expensive)

### 4.4 Vertex AI Experiments

**Track experiments**:

```python
from google.cloud import aiplatform

# Initialize experiment
aiplatform.init(
    project='my-project',
    location='us-central1',
    experiment='recommendation-model'
)

# Start a run
aiplatform.start_run('run-1')

# Log parameters
aiplatform.log_params({
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 10
})

# Log metrics
aiplatform.log_metrics({
    'accuracy': 0.95,
    'loss': 0.12
})

# End run
aiplatform.end_run()
```

---

## 5. Model Serving

### 5.1 Vertex AI Prediction

**Online Prediction** (real-time, low latency):

```python
# Deploy model
endpoint = model.deploy(
    deployed_model_display_name='my-model-v1',
    machine_type='n1-standard-4',
    min_replica_count=1,
    max_replica_count=10,  # Auto-scaling
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)

# Make prediction
predictions = endpoint.predict(instances=[
    {'feature1': 1.0, 'feature2': 2.0}
])
```

**Batch Prediction** (offline, bulk):

```python
batch_prediction_job = model.batch_predict(
    job_display_name='batch-predict',
    gcs_source='gs://bucket/input.jsonl',
    gcs_destination_prefix='gs://bucket/output/',
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1
)
```

**Model Versioning**:
```python
# Deploy new version to same endpoint
endpoint.deploy(
    model=new_model,
    deployed_model_display_name='my-model-v2',
    traffic_split={'0': 90, '1': 10}  # Canary: 90% v1, 10% v2
)
```

### 5.2 Cloud Run (Custom Serving)

**For custom inference logic**:

```python
# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

```dockerfile
# Dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

```bash
# Deploy to Cloud Run
gcloud run deploy model-service \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --min-instances 1 \
    --max-instances 10
```

**When to use Cloud Run**:
- ✅ Custom inference logic
- ✅ Non-standard frameworks
- ✅ Variable traffic (auto-scales to zero)
- ✅ Cost-effective for low traffic

### 5.3 Cloud Functions (Lightweight)

**For simple predictions**:

```python
# main.py
import functions_framework
import joblib

model = joblib.load('model.pkl')

@functions_framework.http
def predict(request):
    data = request.get_json()
    prediction = model.predict([data['features']])
    return {'prediction': prediction.tolist()}
```

```bash
# Deploy
gcloud functions deploy predict \
    --runtime python39 \
    --trigger-http \
    --allow-unauthenticated
```

**When to use Cloud Functions**:
- ✅ Extremely simple models
- ✅ Event-driven predictions
- ✅ Very low traffic
- ❌ Cold start latency concerns

### 5.4 Serving Comparison

| Service | Latency | Scalability | Cost | Use Case |
|---------|---------|-------------|------|----------|
| **Vertex AI Prediction** | Low (< 100ms) | High (auto-scale) | $$$ | Production ML, standard frameworks |
| **Cloud Run** | Medium (100-500ms) | High (auto-scale to 0) | $$ | Custom logic, variable traffic |
| **Cloud Functions** | High (cold start) | Medium | $ | Simple, event-driven |
| **GKE** | Configurable | Very high | $$$$ | Full control, complex deployments |

---

## 6. Monitoring & Operations

### 6.1 Vertex AI Model Monitoring

**Automatic monitoring** for deployed models:

```python
# Enable model monitoring
monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name='monitoring-job',
    endpoint=endpoint,
    logging_sampling_strategy=aiplatform.ModelMonitoringConfig.SamplingStrategy(
        random_sample_config={'sample_rate': 0.1}
    ),
    model_deployment_monitoring_objective_configs=[
        aiplatform.ModelDeploymentMonitoringObjectiveConfig(
            deployed_model_id=deployed_model.id,
            objective_config=aiplatform.ModelMonitoringObjectiveConfig(
                training_dataset=training_dataset,
                training_prediction_skew_detection_config=aiplatform.ModelMonitoringAlertConfig(
                    email_alert_config={'user_emails': ['user@example.com']}
                ),
                prediction_drift_detection_config=aiplatform.ModelMonitoringAlertConfig(
                    email_alert_config={'user_emails': ['user@example.com']}
                )
            )
        )
    ],
    alert_config=aiplatform.ModelMonitoringAlertConfig(
        email_alert_config={'user_emails': ['user@example.com']}
    )
)
```

**What it monitors**:
- **Training-serving skew**: Feature distribution differences
- **Prediction drift**: Output distribution changes over time
- **Input/output anomalies**: Outliers, missing values

### 6.2 Cloud Monitoring

**Custom metrics**:

```python
from google.cloud import monitoring_v3

client = monitoring_v3.MetricServiceClient()
project_name = f"projects/{project_id}"

# Write custom metric
series = monitoring_v3.TimeSeries()
series.metric.type = 'custom.googleapis.com/model/prediction_latency'
series.resource.type = 'global'

point = monitoring_v3.Point()
point.value.double_value = latency_ms
point.interval.end_time.seconds = int(time.time())

series.points = [point]
client.create_time_series(name=project_name, time_series=[series])
```

**Alerting**:
```bash
# Create alert policy
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="High prediction latency" \
    --condition-display-name="Latency > 500ms" \
    --condition-threshold-value=500 \
    --condition-threshold-duration=60s \
    --aggregation-alignment-period=60s
```

### 6.3 Cloud Logging

**Log predictions for debugging**:

```python
from google.cloud import logging

logging_client = logging.Client()
logger = logging_client.logger('model-predictions')

# Log prediction
logger.log_struct({
    'model_version': 'v1',
    'features': features,
    'prediction': prediction,
    'latency_ms': latency,
    'timestamp': timestamp
})
```

**Query logs**:
```bash
# View logs
gcloud logging read "resource.type=ml_job" --limit 10

# Create log-based metric
gcloud logging metrics create prediction_errors \
    --description="Count of prediction errors" \
    --log-filter='jsonPayload.error_type="PREDICTION_ERROR"'
```

---

## 7. MLOps on GCP

### 7.1 CI/CD for ML

**Cloud Build + Vertex AI Pipelines**:

```yaml
# cloudbuild.yaml
steps:
  # Run tests
  - name: 'python:3.9'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        pip install -r requirements.txt
        pytest tests/

  # Build training container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/trainer:$SHORT_SHA', '.']

  # Push container
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/trainer:$SHORT_SHA']

  # Submit training job
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud ai custom-jobs create \
          --region=us-central1 \
          --display-name=training-$SHORT_SHA \
          --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=gcr.io/$PROJECT_ID/trainer:$SHORT_SHA
```

### 7.2 Vertex AI Pipelines (Kubeflow Pipelines)

**Define ML pipeline**:

```python
from kfp.v2 import dsl
from kfp.v2.dsl import component, Input, Output, Dataset, Model

@component(base_image='python:3.9')
def preprocess_data(
    input_data: Input[Dataset],
    output_data: Output[Dataset]
):
    import pandas as pd
    df = pd.read_csv(input_data.path)
    # ... preprocessing logic
    df.to_csv(output_data.path, index=False)

@component(base_image='python:3.9', packages_to_install=['scikit-learn'])
def train_model(
    training_data: Input[Dataset],
    model: Output[Model],
    learning_rate: float
):
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    # ... training logic
    joblib.dump(clf, model.path)

@dsl.pipeline(name='training-pipeline')
def training_pipeline(learning_rate: float = 0.01):
    preprocess_task = preprocess_data(input_data=...)
    train_task = train_model(
        training_data=preprocess_task.outputs['output_data'],
        learning_rate=learning_rate
    )

# Compile and run
from kfp.v2 import compiler
from google.cloud import aiplatform

compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path='pipeline.json'
)

job = aiplatform.PipelineJob(
    display_name='training-pipeline',
    template_path='pipeline.json',
    parameter_values={'learning_rate': 0.01}
)
job.run()
```

**Benefits**:
- ✅ Reproducible pipelines
- ✅ Component reusability
- ✅ Version control
- ✅ Automated orchestration

### 7.3 Model Registry

**Vertex AI Model Registry**:

```python
# Register model
model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri='gs://bucket/model/',
    serving_container_image_uri='gcr.io/cloud-aiplatform/prediction/tf2-cpu.2-11:latest',
    labels={'version': 'v1', 'stage': 'production'}
)

# List models
models = aiplatform.Model.list(filter='labels.stage=production')

# Get model by version
model = aiplatform.Model(model_name='projects/123/locations/us-central1/models/456@1')
```

---

## 8. Cost Optimization

### 8.1 Cost Breakdown

**Major cost drivers**:
1. **Compute**: Training and serving VMs/GPUs
2. **Storage**: BigQuery, Cloud Storage
3. **Network**: Data transfer, API calls
4. **BigQuery queries**: Pay per TB scanned

### 8.2 Cost Optimization Strategies

**Training**:
- Use **Preemptible VMs** (up to 80% cheaper, can be interrupted)
- Enable **auto-shutdown** for Workbench notebooks
- Use **managed notebooks** instead of user-managed
- Right-size machine types (don't over-provision)

```python
# Use preemptible VMs for training
job.run(
    replica_count=4,
    machine_type='n1-standard-4',
    base_output_dir='gs://bucket/output',
    enable_web_access=False,  # Reduce costs
    restart_job_on_worker_restart=True  # Handle preemption
)
```

**Serving**:
- Use **Cloud Run** for variable traffic (scales to zero)
- Set appropriate **min/max replicas** for Vertex AI endpoints
- Use **batch prediction** instead of online when possible
- Cache frequent predictions

**Storage**:
- Use **Nearline/Coldline** storage for infrequent access
- Set **lifecycle policies** to auto-delete old data
- Partition BigQuery tables to reduce query costs
- Use **BigQuery caching** (free for repeated queries)

```sql
-- Optimize BigQuery costs with partitioning
CREATE TABLE `project.dataset.events`
PARTITION BY DATE(timestamp)
CLUSTER BY user_id
AS SELECT * FROM source_table;

-- Query with partition filter (much cheaper)
SELECT * FROM `project.dataset.events`
WHERE DATE(timestamp) = '2024-01-01';  -- Only scans 1 day
```

**Monitoring**:
- Use **Cloud Monitoring quota** (free tier: 50GB/month)
- Sample logs instead of logging everything
- Set log retention policies

### 8.3 Budget Alerts

```bash
# Set up budget alert
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="ML Project Budget" \
    --budget-amount=1000USD \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=90 \
    --threshold-rule=percent=100
```

---

## 9. Common Design Patterns on GCP

### 9.1 Recommendation System

```
User Events → Pub/Sub → Dataflow → BigQuery
                           ↓
                    Feature Store
                           ↓
            ┌──────────────┴──────────────┐
            ↓                              ↓
    Candidate Generation              Ranking Model
    (Batch, Vertex AI)               (Online, Vertex AI)
            ↓                              ↓
    Candidates in Bigtable ──────→ Cloud Run/Endpoint
                                          ↓
                                    Recommendations
```

**Implementation**:
1. **Data Collection**: Pub/Sub + Dataflow
2. **Feature Engineering**: BigQuery + Feature Store
3. **Candidate Generation**: Batch prediction (Vertex AI)
4. **Ranking**: Online prediction (Vertex AI endpoint)
5. **Serving**: Cloud Run with caching (Memorystore)

### 9.2 Fraud Detection

```
Transaction → Cloud Function → Vertex AI Endpoint → Firestore
    ↓                                  ↓
Pub/Sub ──→ Dataflow ──→ Real-time Features ──┘
```

**Implementation**:
1. **Streaming**: Pub/Sub for transaction events
2. **Real-time Features**: Dataflow + Bigtable
3. **Inference**: Vertex AI endpoint (< 50ms latency)
4. **Storage**: Firestore for decisions
5. **Monitoring**: Cloud Monitoring for drift detection

### 9.3 Document Classification

```
Documents → Cloud Storage → Vertex AI Batch Prediction
    ↓                              ↓
AutoML Vision API           BigQuery (results)
    ↓                              ↓
Classification             Cloud Functions (action)
```

**Implementation**:
1. **Upload**: Documents to Cloud Storage
2. **OCR**: Document AI for text extraction
3. **Classification**: AutoML NLP or Vertex AI
4. **Results**: Store in BigQuery
5. **Action**: Cloud Functions for downstream processing

---

## 10. Best Practices

### 10.1 Security

**IAM (Identity and Access Management)**:
- Use **service accounts** for applications
- Follow **least privilege** principle
- Enable **VPC Service Controls** for data protection

```bash
# Create service account for training
gcloud iam service-accounts create ml-training \
    --display-name="ML Training Service Account"

# Grant minimal permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:ml-training@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

**Data Protection**:
- Enable **Cloud KMS** for encryption
- Use **customer-managed encryption keys (CMEK)**
- Enable **audit logging** for compliance

### 10.2 Reproducibility

**Version everything**:
- Code: Git
- Data: Data versioning (DVC, or BigQuery snapshots)
- Models: Vertex AI Model Registry
- Pipelines: Vertex AI Pipelines
- Infrastructure: Terraform

```python
# Tag training runs with Git commit
aiplatform.start_run('training-run')
aiplatform.log_params({
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
    'data_snapshot': 'bq://project.dataset.table@20240101'
})
```

### 10.3 Testing

**Data validation**:
```python
import tensorflow_data_validation as tfdv

# Generate statistics
stats = tfdv.generate_statistics_from_csv('data.csv')

# Infer schema
schema = tfdv.infer_schema(stats)

# Validate new data
new_stats = tfdv.generate_statistics_from_csv('new_data.csv')
anomalies = tfdv.validate_statistics(new_stats, schema)
```

**Model testing**:
- Unit tests for preprocessing logic
- Integration tests for pipelines
- Performance benchmarks (latency, throughput)
- Model quality tests (accuracy thresholds)

---

## 11. Migration Patterns

### 11.1 AWS to GCP

| AWS Service | GCP Equivalent |
|-------------|----------------|
| SageMaker | Vertex AI |
| S3 | Cloud Storage |
| Redshift | BigQuery |
| Kinesis | Pub/Sub + Dataflow |
| Lambda | Cloud Functions |
| ECS/EKS | Cloud Run / GKE |
| DynamoDB | Firestore / Bigtable |
| EMR | Dataproc |

### 11.2 On-Premise to GCP

**Hybrid approach** with Anthos:
- Run Vertex AI Pipelines on-premise (Anthos)
- Gradually migrate data to Cloud Storage/BigQuery
- Use **Transfer Service** for large data migration
- Federated queries (BigQuery + on-premise databases)

---

## 12. Interview Talking Points

When discussing ML system design with GCP:

**Data Processing**:
- "We'd use BigQuery for our data warehouse with date partitioning to optimize query costs"
- "Pub/Sub + Dataflow for real-time feature computation"

**Training**:
- "Vertex AI Pipelines for reproducible training workflows"
- "Preemptible VMs to reduce training costs by 80%"

**Serving**:
- "Vertex AI endpoints with auto-scaling for production serving"
- "Cloud Run for custom inference logic that scales to zero"

**Monitoring**:
- "Vertex AI Model Monitoring for automatic drift detection"
- "Cloud Logging and Monitoring for custom metrics"

**Cost**:
- "BigQuery on-demand pricing vs flat-rate for predictable costs"
- "Batch predictions for offline use cases to reduce serving costs"

---

## Key Takeaways

1. **Vertex AI** is the unified platform for ML on GCP
2. **BigQuery** is powerful for both analytics and ML (BQML)
3. **Managed services** reduce operational overhead
4. **Cost optimization** is critical (preemptible VMs, auto-scaling, partitioning)
5. **Feature Store** ensures training-serving consistency
6. **Pipelines** provide reproducibility and automation
7. **Monitoring** catches drift and performance issues

---

## Further Resources

### Official Documentation
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [BigQuery ML Documentation](https://cloud.google.com/bigquery-ml/docs)
- [GCP Architecture Center](https://cloud.google.com/architecture)

### Learning Paths
- [Google Cloud Skills Boost - ML Engineer Learning Path](https://www.cloudskillsboost.google/)
- [Coursera - Machine Learning on GCP Specialization](https://www.coursera.org/specializations/machine-learning-tensorflow-gcp)

### Blogs & Case Studies
- [Google Cloud Blog - AI & ML](https://cloud.google.com/blog/products/ai-machine-learning)
- [Vertex AI Customer Stories](https://cloud.google.com/vertex-ai#section-10)

### Certifications
- **Professional Machine Learning Engineer** (recommended)
- **Professional Data Engineer** (for data pipelines)

---

**Last Updated**: December 2024
