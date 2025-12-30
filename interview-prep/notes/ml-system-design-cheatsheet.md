# ML System Design Cheatsheet - Interview Prep

**Quick reference for ML system design interviews - optimized for recall under pressure**

---

## Table of Contents

1. [The Framework](#the-framework)
2. [Problem Formulation](#problem-formulation)
3. [Component Quick Reference](#component-quick-reference)
4. [Common Architectures](#common-architectures)
5. [GCP Services Map](#gcp-services-map)
6. [Scaling & Optimization](#scaling--optimization)
7. [Interview Strategy](#interview-strategy)
8. [Common Questions](#common-questions)

---

## The Framework

### 8-Step ML System Design Process

```
1. Problem Formulation (5-10 min)
   ├─ Clarify business objective
   ├─ Define success metrics
   └─ Identify constraints (latency, scale, budget)

2. Data Pipeline
   ├─ Data sources
   ├─ Storage (lake vs warehouse)
   └─ Processing (batch vs stream)

3. Feature Engineering
   ├─ Feature types
   ├─ Feature store
   └─ Avoid data leakage

4. Model Development
   ├─ Model selection
   ├─ Training pipeline
   └─ Experiment tracking

5. Model Evaluation
   ├─ Offline metrics
   └─ Online A/B testing

6. Deployment & Serving
   ├─ Serving pattern (batch/online/streaming)
   ├─ Latency optimization
   └─ Deployment strategy (canary/blue-green)

7. Monitoring & Maintenance
   ├─ Performance metrics
   ├─ Data drift detection
   └─ Retraining strategy

8. Iteration & Improvement
   └─ Feedback loops
```

### Quick Decision Points

| Question | Answer Impacts |
|----------|----------------|
| Online or Offline? | Serving pattern, latency requirements |
| Scale (QPS/Users/Data)? | Infrastructure, costs, complexity |
| Latency requirements? | Model complexity, caching strategy |
| Explainability needed? | Model choice (linear vs neural net) |
| Cold start problem? | Fallback strategy, hybrid approach |

---

## Problem Formulation

### ML Problem Type Mapping

| Business Problem | ML Formulation | Output | Key Metric |
|-----------------|----------------|--------|------------|
| **Recommendations** | Ranking/Retrieval | Ordered list | NDCG, MAP |
| **Fraud Detection** | Binary Classification | Probability | Precision, Recall |
| **Search Ranking** | Learning to Rank | Relevance scores | NDCG, MRR |
| **Content Moderation** | Multi-class | Category labels | F1, Precision |
| **Demand Forecasting** | Time Series | Future values | MAPE, RMSE |
| **Ad Click Prediction** | Binary Classification | Click probability | AUC, Log Loss |
| **Spam Detection** | Binary Classification | Spam/Not Spam | Precision, F1 |

### Metrics Cheatsheet

**Classification**:
- **Imbalanced data**: Precision, Recall, F1, **PR-AUC** (better than ROC-AUC)
- **Balanced data**: Accuracy, ROC-AUC
- **Business cost**: Precision (minimize false positives) or Recall (minimize false negatives)

**Ranking**:
- **NDCG**: Discounted cumulative gain (position matters)
- **MAP**: Mean average precision (binary relevance)
- **MRR**: Mean reciprocal rank (first relevant result)

**Regression**:
- **MSE/RMSE**: Penalizes large errors
- **MAE**: Robust to outliers
- **MAPE**: Percentage error (scale-independent)

---

## Component Quick Reference

### Data Storage Decision Tree

```
How much data?
├─ < 1TB → Database (Cloud SQL, PostgreSQL)
└─ > 1TB
    └─ Structured?
        ├─ YES
        │   └─ Analytics queries? → Data Warehouse (BigQuery, Redshift)
        └─ NO → Data Lake (Cloud Storage, S3)

Need real-time access?
├─ YES
│   ├─ Low latency (< 10ms) → NoSQL (Bigtable, DynamoDB)
│   └─ Complex queries → Database + Cache (Redis, Memcached)
└─ NO → Data Lake or Warehouse
```

### Feature Store When?

✅ **Use Feature Store when**:
- Training/serving skew is a risk
- Features reused across models
- Need point-in-time correctness
- Online + offline serving

❌ **Skip Feature Store when**:
- Simple model, few features
- Prototype/POC
- All features computed at inference time

### Model Selection Quick Guide

| Model Type | Pros | Cons | Use When |
|-----------|------|------|----------|
| **Linear/Logistic** | Fast, interpretable | Low capacity | Need explainability, baseline |
| **Tree Ensembles (XGBoost)** | High accuracy, handles mixed types | Slow inference | Tabular data, offline |
| **Neural Networks** | Highest capacity | Black box, slow training | Images, text, complex patterns |
| **Two-Tower NN** | Scalable embeddings | Requires lots of data | Recommendations, search |
| **LightGBM** | Fast training, low memory | Less accurate than XGBoost | Large datasets, need speed |

---

## Common Architectures

### 1. Recommendation System

**Architecture**:
```
User Events → Feature Store
              ↓
     ┌────────┴────────┐
     ↓                 ↓
Candidate Gen      Ranking Model
(Offline, 1000s)   (Online, Top 100)
     ↓                 ↓
  ANN Index ────→  Re-ranking
                  (Diversity, Business Rules)
                       ↓
                 Final Results
```

**Key Points**:
- **Two-stage**: Candidate generation (recall) + Ranking (precision)
- **Candidate generation**: Collaborative filtering, content-based, popularity
- **Ranking**: Rich features, complex model (neural net, GBDT)
- **Cold start**: Fallback to popularity, content-based for new users/items
- **Exploration vs Exploitation**: Multi-armed bandits, epsilon-greedy

**Interview Talking Points**:
- "Two-stage to balance recall and precision while staying within latency budget"
- "Use ANN (Approximate Nearest Neighbors) for fast candidate retrieval"
- "Feature store for real-time user features (clicks last hour, etc.)"
- "A/B test with engagement metrics (CTR, dwell time, long-term retention)"

### 2. Search Ranking

**Pipeline**:
```
Query → Query Understanding → Retrieval → Ranking → Blending
        (Spell, Expand)      (Inverted   (L2R)    (Personalize)
                              Index)
```

**Components**:
1. **Query Understanding**: Spell correction, expansion, intent classification
2. **Retrieval**: Inverted index (Elasticsearch) or vector search (embeddings)
3. **Ranking**: Learning to Rank (LambdaMART, neural ranker)
4. **Features**: Query-doc relevance, doc quality, personalization

**Interview Talking Points**:
- "Hybrid retrieval: keyword (BM25) + semantic (embeddings)"
- "Learning to Rank with pairwise or listwise loss"
- "Cache popular queries (80% traffic from 20% queries)"
- "Personalization in re-ranking layer (avoid filter bubbles)"

### 3. Fraud Detection

**Architecture**:
```
Transaction → Rules Engine → ML Model → Manual Review
   ↓              ↓              ↓           ↓
Pub/Sub → Real-time Features → Risk Score → Action
          (last 10 min stats)  (0-1)       (Block/Flag)
```

**Key Points**:
- **Real-time**: Streaming features (Pub/Sub + Dataflow + Bigtable)
- **Class imbalance**: Use PR-AUC, precision-recall trade-off
- **Adversarial**: Fraudsters adapt, need frequent retraining
- **Hybrid**: Rules (block obvious) + ML (nuanced cases)

**Interview Talking Points**:
- "Three-tier: Rules (fast, high precision) → ML → Human review"
- "Graph features to detect fraud rings (connected accounts)"
- "Online learning to adapt to new fraud patterns"
- "Cost-based threshold (false positive = lost customer, false negative = lost money)"

### 4. Content Moderation

**Pipeline**:
```
Content → Pre-filters → Classifiers → Human Review → Action
          (Blocklist)   (Multi-task)  (Borderline)   (Remove/Warn)
```

**Challenges**:
- **Multi-modal**: Text, image, video
- **Evolving policy**: New harm types
- **Precision vs Recall**: Over-block vs under-block

**Interview Talking Points**:
- "Multi-task learning: hate speech, spam, NSFW in one model"
- "Active learning: prioritize uncertain examples for human review"
- "Ensemble: multiple models for high-stakes decisions"

---

## GCP Services Map

### Quick Service Lookup

| ML Lifecycle Stage | GCP Service | Use For |
|-------------------|-------------|---------|
| **Data Storage** | Cloud Storage | Data lake, raw data, model artifacts |
| | BigQuery | Data warehouse, analytics, BQML |
| | Bigtable | High-throughput, low-latency NoSQL |
| **Data Processing** | Dataflow | Batch & stream ETL (Apache Beam) |
| | Dataproc | Spark/Hadoop jobs |
| | Pub/Sub | Event streaming |
| **Feature Engineering** | Vertex AI Feature Store | Online/offline features |
| | BigQuery | Feature computation (SQL) |
| **Model Training** | Vertex AI Training | Custom training (GPU/TPU) |
| | Vertex AI AutoML | Quick prototypes, no code |
| | Vertex AI Workbench | Jupyter notebooks |
| **Experiment Tracking** | Vertex AI Experiments | Track runs, compare models |
| **Model Registry** | Vertex AI Model Registry | Version, deploy models |
| **Model Serving** | Vertex AI Prediction | Managed endpoints, auto-scale |
| | Cloud Run | Custom inference, scales to 0 |
| | Cloud Functions | Simple, event-driven |
| **Monitoring** | Vertex AI Model Monitoring | Drift, skew detection |
| | Cloud Monitoring | Custom metrics, alerts |
| **Orchestration** | Vertex AI Pipelines | ML workflows (Kubeflow) |
| | Cloud Composer | Airflow-based workflows |

### Architecture Patterns on GCP

**Batch Prediction**:
```
BigQuery → Vertex AI Batch Prediction → BigQuery
(Features)                              (Predictions)
```

**Online Prediction**:
```
Request → Cloud Run → Vertex AI Feature Store → Vertex AI Endpoint
                      (Real-time features)      (Model)
```

**Real-time Feature Pipeline**:
```
Pub/Sub → Dataflow → Bigtable/Feature Store → Model Serving
(Events)  (Process)  (Online features)
```

---

## Scaling & Optimization

### Latency Optimization

| Technique | Latency Gain | Trade-off |
|-----------|-------------|-----------|
| **Model quantization** | 2-4x faster | Slight accuracy drop |
| **Model pruning** | 2-3x faster | Accuracy drop |
| **Feature caching** | 10-100x faster | Stale features |
| **Request batching** | 5-10x throughput | Increased latency |
| **Model cascading** | 3-5x faster | Complexity |
| **GPU/TPU** | 10-100x faster | Cost increase |

### Model Cascading

```
Request → Fast Model (99% of traffic)
          ↓ (only uncertain)
          Complex Model (1% of traffic)
```

**Example**: Ad click prediction
- Fast: Logistic regression (< 1ms)
- Complex: Neural network (10ms) for uncertain cases

### Cost Optimization

**Training**:
- Use preemptible VMs (80% cheaper)
- Right-size instances
- Batch small experiments

**Serving**:
- Auto-scale (min/max replicas)
- Cloud Run for variable traffic (scales to 0)
- Batch prediction when possible
- Cache predictions for popular inputs

**Storage**:
- BigQuery partitioning (reduce query costs)
- Lifecycle policies (move to coldline)
- Compress data

---

## Interview Strategy

### Time Allocation (45 min interview)

```
0-5 min:   Clarify problem
5-10 min:  High-level design (draw diagram)
10-30 min: Deep dive (1-2 components)
30-40 min: Scaling, monitoring, trade-offs
40-45 min: Questions for interviewer
```

### The STAR Template

**S**cope: "Let me clarify the requirements..."
- Scale (users, QPS, data size)
- Latency (< 100ms? < 1s?)
- Success metrics (CTR? Revenue? Engagement?)
- Constraints (budget, team size, timeline)

**T**ech Design: "Here's the end-to-end architecture..."
```
Data → Features → Model → Serving → Monitoring
```

**A**lternatives: "We could also consider..."
- Discuss trade-offs
- Explain why you chose your approach

**R**isks: "Things to watch out for..."
- Data quality, drift, cold start, scaling, cost

### Communication Tips

✅ **Do**:
- Think out loud
- Draw diagrams
- Ask clarifying questions
- Discuss trade-offs
- Use specific numbers (not "fast" but "< 100ms")
- Mention actual technologies (BigQuery, not "data warehouse")

❌ **Don't**:
- Jump to solutions immediately
- Assume requirements
- Ignore constraints
- Over-engineer for prototypes
- Use vague terms
- Get stuck on one component

### Common Mistakes

1. **Not clarifying scope**: "Should this support 100 or 100M users?"
2. **No metrics**: Define success before designing
3. **Ignoring constraints**: Latency, budget, team size matter
4. **Too much detail too early**: Start high-level, then dive deep
5. **Not discussing trade-offs**: Every decision has pros/cons
6. **Forgetting monitoring**: Models degrade, need to catch it
7. **No feedback loop**: How do you improve the model?

---

## Common Questions

### System Design Questions

#### 1. Design a recommendation system for YouTube

**Clarify**:
- Scale: Billions of users, millions of videos
- Latency: < 200ms for recommendations
- Metric: Watch time, CTR

**High-level**:
```
User → Candidate Gen (1000s) → Ranking (100) → Re-ranking → Display (20)
```

**Key points**:
- Two-tower model for candidate generation
- Negative sampling for training
- Freshness vs relevance trade-off
- Cold start: trending, content-based

#### 2. Design a fraud detection system

**Clarify**:
- Real-time (< 100ms) or batch?
- Fraud rate (0.1%? 1%?)
- Cost: FP vs FN

**Architecture**:
```
Transaction → Rules → ML Model → Manual Review
   ↓
Real-time features (last 10 min activity)
```

**Key points**:
- Streaming features (Pub/Sub + Dataflow)
- Class imbalance (use SMOTE, class weights, PR-AUC)
- Graph features (fraud rings)
- Frequent retraining (adversarial)

#### 3. Design a search ranking system

**Pipeline**:
```
Query → Understanding → Retrieval → Ranking → Results
```

**Key points**:
- Hybrid retrieval (keyword + semantic)
- Learning to Rank (LambdaMART)
- Personalization in re-ranking
- Cache popular queries

#### 4. Design an ML system to detect spam

**Clarify**:
- Email vs SMS vs social media?
- Volume (QPS)?
- Latency tolerance?

**Approach**:
```
Message → Feature Extraction → Model → Score → Action
          (TF-IDF, metadata)   (LR/NN)  (0-1)   (Block/Flag)
```

**Key points**:
- Text features: TF-IDF, word embeddings
- Metadata: sender reputation, timing patterns
- Online learning for new spam patterns
- Precision vs recall (user tolerance for FP)

#### 5. Design a system to predict Uber ETA

**Clarify**:
- Accuracy requirement?
- Latency (< 100ms)?
- Coverage (all cities)?

**Features**:
- Route: distance, traffic, historical time
- Context: time of day, weather, events
- Driver: driving patterns

**Model**: XGBoost (fast inference, interpretable)

**Key points**:
- Real-time traffic data integration
- City-specific models vs global
- Cold start for new routes
- Continuous learning from actual trip times

### Follow-up Deep Dives

Be ready to dive deep on:
- "How would you handle data drift?"
- "How would you scale to 10x traffic?"
- "What if latency needs to be < 10ms?"
- "How would you debug if model performance drops?"
- "How would you retrain the model?"

---

## Quick Wins for Interviews

### Impressive Talking Points

1. **Point-in-time correctness**: "Use feature store to avoid temporal leakage"
2. **Two-stage architecture**: "Candidate generation for recall, ranking for precision"
3. **Hybrid approach**: "Combine collaborative filtering with content-based for cold start"
4. **Cost-based threshold**: "Optimize for business cost, not just accuracy"
5. **Shadow deployment**: "Test new model alongside old before switching traffic"
6. **Multi-armed bandits**: "Balance exploration and exploitation"
7. **Model cascading**: "Fast model for most requests, complex for uncertain"

### Metrics to Mention

- **Not just accuracy**: Precision, Recall, F1, AUC, NDCG, MAP
- **Business metrics**: Revenue, engagement, retention
- **Latency**: P50, P95, P99 (not just average)
- **Cost**: $ per prediction, $ per user

### Technologies to Name-drop

**GCP**: Vertex AI, BigQuery, Dataflow, Pub/Sub, Bigtable, Cloud Run
**AWS**: SageMaker, Redshift, Kinesis, DynamoDB, Lambda
**Open Source**: MLflow, Airflow, Spark, Kafka, Feast, TensorFlow Serving

---

## Key Takeaways

1. **Clarify first**: Scope, scale, metrics, constraints
2. **Start high-level**: End-to-end diagram before details
3. **Two-stage is common**: Fast recall + accurate ranking
4. **Metrics matter**: Business metrics > model metrics
5. **Monitoring is critical**: Models degrade over time
6. **Trade-offs everywhere**: Latency vs accuracy, cost vs performance
7. **Real numbers**: "< 100ms" not "fast", "1M QPS" not "a lot"
8. **Feedback loops**: How to improve the model over time

**Golden Rule**: *Design for the problem at hand, not the most complex solution you know.*

---

**Last Updated**: December 2024
