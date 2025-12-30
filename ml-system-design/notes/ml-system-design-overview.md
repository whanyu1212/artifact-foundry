# ML System Design: A Comprehensive Guide

## Overview

Machine Learning System Design is about building end-to-end ML solutions that are **reliable, scalable, and maintainable** in production. Unlike algorithm-focused ML interviews, ML system design focuses on the entire lifecycle: data pipelines, infrastructure, deployment, monitoring, and iteration.

**Key Difference**:
- **ML Algorithm Design**: How does gradient descent work? Implement a decision tree from scratch.
- **ML System Design**: How do you build a recommendation system for 100M users? How do you detect when your model degrades?

---

## The ML System Design Framework

Every ML system design problem can be broken down into these components:

```
1. Problem Formulation
   ↓
2. Data Pipeline
   ↓
3. Feature Engineering
   ↓
4. Model Development
   ↓
5. Model Evaluation
   ↓
6. Deployment & Serving
   ↓
7. Monitoring & Maintenance
   ↓
8. Iteration & Improvement
```

---

## 1. Problem Formulation

### 1.1 Clarifying the Problem

**Questions to Ask**:
- **Business objective**: What are we optimizing for? (revenue, engagement, safety)
- **Success metrics**: How do we measure success? (precision, recall, revenue lift, user satisfaction)
- **Constraints**: Latency requirements? Budget? Data availability?
- **Scale**: How many users? How much data? Request volume?

### 1.2 Framing as an ML Problem

**Common ML Problem Types**:

| Business Problem | ML Formulation | Output |
|-----------------|----------------|--------|
| Product recommendations | Ranking/Retrieval | Ordered list of items |
| Fraud detection | Binary classification | Fraud probability |
| Content moderation | Multi-class classification | Category labels |
| Ad click prediction | Binary classification | Click probability |
| Search ranking | Learning to rank | Relevance scores |
| Demand forecasting | Time series prediction | Future values |

### 1.3 Online vs Offline Requirements

**Online (Real-time)**:
- Low latency required (< 100ms typical)
- Serve predictions on-demand
- Need fast inference
- Example: Ad serving, recommendation APIs

**Offline (Batch)**:
- Can process in bulk (minutes to hours)
- Pre-compute predictions
- Can use complex models
- Example: Email campaign targeting, daily reports

**Hybrid**:
- Offline pre-computation + online ranking
- Example: YouTube recommendations (candidate generation offline, ranking online)

---

## 2. Data Pipeline

### 2.1 Data Sources

**Common Sources**:
- **User interactions**: Clicks, views, purchases, ratings
- **User profiles**: Demographics, preferences, history
- **Content features**: Item metadata, descriptions, images
- **Context**: Time, location, device, session data
- **External data**: Weather, market data, social signals

### 2.2 Data Collection

**Considerations**:
- **Logging infrastructure**: What events to track? (impression, click, conversion)
- **Privacy & compliance**: GDPR, data retention policies
- **Data quality**: Handle missing values, duplicates, outliers
- **Sampling**: Do you need all data or can you sample?

### 2.3 Data Storage

**Storage Options**:

| Type | Use Case | Examples | Trade-offs |
|------|----------|----------|-----------|
| **Data Lake** | Raw, unstructured data | S3, HDFS | Cheap, flexible, no schema |
| **Data Warehouse** | Structured, analytics | Snowflake, BigQuery, Redshift | Fast queries, expensive |
| **Feature Store** | ML features, online/offline | Feast, Tecton | Consistency, versioning |
| **Database** | Transactional data | PostgreSQL, MySQL | ACID guarantees, limited scale |
| **NoSQL** | High-scale, low-latency | Cassandra, DynamoDB | Scalable, eventual consistency |

### 2.4 Data Processing

**Batch Processing**:
- Daily/hourly ETL jobs
- Tools: Spark, Hadoop, Airflow
- Use case: Training data preparation, feature computation

**Stream Processing**:
- Real-time event processing
- Tools: Kafka, Flink, Spark Streaming
- Use case: Real-time features, online metrics

---

## 3. Feature Engineering

### 3.1 Feature Types

**Numerical Features**:
- Continuous: age, price, distance
- Discrete: count of purchases, number of views
- Transformations: log, polynomial, binning

**Categorical Features**:
- One-hot encoding (low cardinality)
- Embedding/hashing (high cardinality)
- Target encoding (with care for leakage)

**Text Features**:
- TF-IDF, word embeddings
- Pre-trained models (BERT, etc.)

**Temporal Features**:
- Time since event, day of week, seasonality
- Moving averages, trends

**Interaction Features**:
- Cross-products: user × item, location × time
- Domain-specific combinations

### 3.2 Feature Store

**Purpose**: Centralized repository for ML features

**Benefits**:
- **Consistency**: Same features for training and serving
- **Reusability**: Share features across models
- **Versioning**: Track feature definitions over time
- **Online/Offline**: Serve features in both environments

**Architecture**:
```
Batch Features (Offline) ──┐
                           ├──> Feature Store ──> Model Training
Stream Features (Online) ──┘                  └──> Model Serving
```

**Popular Tools**:
- **Feast** (open source)
- **Tecton** (commercial)
- **AWS SageMaker Feature Store**
- **Databricks Feature Store**

### 3.3 Feature Engineering Best Practices

**Avoid Data Leakage**:
- Don't use future information in features
- Be careful with target encoding
- Split data before feature engineering

**Handle Missing Values**:
- Imputation (mean, median, mode)
- Indicator variables for missingness
- Model-based imputation

**Feature Scaling**:
- Normalization (0-1 range)
- Standardization (zero mean, unit variance)
- Robust scaling (for outliers)

---

## 4. Model Development

### 4.1 Model Selection

**Considerations**:
- **Interpretability**: Do you need to explain predictions? (linear models, decision trees)
- **Latency**: Real-time constraints? (simpler models, optimized serving)
- **Accuracy**: Best possible performance? (ensembles, deep learning)
- **Training time**: How often to retrain? (incremental learning vs full retraining)
- **Data size**: Large data? (distributed training, sampling)

**Common Model Types**:

| Problem | Simple Baseline | Advanced Options |
|---------|----------------|------------------|
| Classification | Logistic Regression | XGBoost, Neural Networks |
| Ranking | Popularity, CF | LambdaMART, Two-tower NN |
| Time Series | Moving Average | ARIMA, LSTM, Prophet |
| NLP | Bag of Words + LR | BERT, GPT |
| Computer Vision | ResNet (transfer) | Vision Transformers, CLIP |

### 4.2 Training Pipeline

**Components**:
1. **Data preprocessing**: Clean, transform, split
2. **Feature computation**: Engineer features
3. **Model training**: Fit model on training data
4. **Hyperparameter tuning**: Grid search, Bayesian optimization
5. **Model validation**: Evaluate on validation set
6. **Model versioning**: Track models and experiments

**Training Strategies**:
- **Full retraining**: Retrain from scratch (most common)
- **Incremental learning**: Update existing model (online learning)
- **Transfer learning**: Fine-tune pre-trained model
- **Ensemble**: Combine multiple models

### 4.3 Experiment Tracking

**What to Track**:
- Model architecture and hyperparameters
- Training/validation metrics
- Dataset versions
- Code version (git commit)
- Training duration and cost

**Tools**:
- **MLflow**: Open source, full lifecycle
- **Weights & Biases**: Excellent visualization
- **TensorBoard**: TensorFlow native
- **Neptune.ai**: Collaboration focused

---

## 5. Model Evaluation

### 5.1 Offline Evaluation

**Metrics by Problem Type**:

**Classification**:
- Binary: Precision, Recall, F1, ROC-AUC, PR-AUC
- Multi-class: Accuracy, macro/micro F1, confusion matrix

**Ranking**:
- MAP (Mean Average Precision)
- NDCG (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)

**Regression**:
- MSE, RMSE, MAE
- R², MAPE (Mean Absolute Percentage Error)

**Choosing Metrics**:
- Align with business objectives
- Consider class imbalance (use PR-AUC over ROC-AUC)
- Multiple metrics (precision vs recall trade-off)

### 5.2 Online Evaluation (A/B Testing)

**Why Online Testing?**
- Offline metrics don't always correlate with business impact
- Real user behavior differs from historical data
- Catch unexpected issues in production

**A/B Test Design**:
1. **Randomization**: Split users into control/treatment
2. **Sample size**: Ensure statistical power
3. **Duration**: Run long enough (1-2 weeks typical)
4. **Metrics**: Primary (business goal) + guardrail (safety)

**Common Pitfalls**:
- **Network effects**: Users influence each other
- **Novelty effect**: Temporary boost from newness
- **Seasonality**: Day of week, holidays
- **Simpson's Paradox**: Segment-level reversals

### 5.3 Validation Strategies

**Cross-Validation**:
- K-fold CV for stable estimates
- Time-based splits for temporal data
- Stratified for imbalanced data

**Train/Validation/Test Split**:
- Training: 70%
- Validation: 15% (hyperparameter tuning)
- Test: 15% (final evaluation)

**Temporal Validation**:
- Critical for time-series problems
- Train on past, validate on future
- Prevents temporal leakage

---

## 6. Deployment & Serving

### 6.1 Serving Patterns

**Batch Prediction**:
- Pre-compute predictions offline
- Store in database/cache
- Serve from lookup
- **Pros**: Can use complex models, no latency constraints
- **Cons**: Not real-time, stale predictions
- **Use case**: Email campaigns, daily recommendations

**Online Prediction (Real-time)**:
- Compute prediction on request
- **Pros**: Fresh predictions, incorporates latest data
- **Cons**: Latency constraints, infrastructure complexity
- **Use case**: Ad serving, search ranking

**Streaming Prediction**:
- Process events as they arrive
- **Pros**: Low latency, real-time updates
- **Cons**: Complex infrastructure
- **Use case**: Fraud detection, real-time bidding

**Hybrid**:
- Batch candidate generation + online ranking
- **Pros**: Balance complexity and freshness
- **Cons**: More complex architecture
- **Use case**: Recommendation systems

### 6.2 Model Serving Infrastructure

**Serving Options**:

| Option | Use Case | Pros | Cons |
|--------|----------|------|------|
| **REST API** | Standard serving | Simple, language-agnostic | Network overhead |
| **gRPC** | High-performance | Fast, typed | More complex |
| **Embedded** | Edge/mobile | No network latency | Limited model size |
| **Serverless** | Variable load | Auto-scaling, pay-per-use | Cold start latency |

**Serving Frameworks**:
- **TensorFlow Serving**: TensorFlow models
- **TorchServe**: PyTorch models
- **MLflow Models**: Framework-agnostic
- **Seldon Core**: Kubernetes-native
- **KFServing/KServe**: Kubernetes, standardized

### 6.3 Scaling Considerations

**Latency Optimization**:
- Model quantization (reduce precision)
- Model pruning (remove weights)
- Batching requests
- Caching frequent predictions
- Hardware acceleration (GPU, TPU)

**Throughput Optimization**:
- Horizontal scaling (more replicas)
- Load balancing
- Asynchronous processing
- Request queuing

**Cost Optimization**:
- Right-size instances
- Auto-scaling policies
- Spot instances for batch jobs
- Model compression

### 6.4 Deployment Strategies

**Blue-Green Deployment**:
- Two identical environments
- Switch traffic atomically
- Easy rollback

**Canary Deployment**:
- Gradually increase traffic to new model
- Monitor metrics closely
- Roll back if issues detected

**Shadow Deployment**:
- New model runs alongside old
- Predictions logged but not served
- Compare predictions offline

---

## 7. Monitoring & Maintenance

### 7.1 What to Monitor

**Model Performance**:
- **Prediction accuracy**: Online metrics (precision, recall)
- **Business metrics**: Revenue, engagement, conversion
- **Prediction distribution**: Are predictions skewed?

**Data Quality**:
- **Feature distributions**: Detect drift
- **Missing values**: Sudden increase?
- **Schema changes**: New fields, type changes

**System Health**:
- **Latency**: P50, P95, P99
- **Throughput**: Requests per second
- **Error rate**: Failed predictions
- **Resource usage**: CPU, memory, GPU

### 7.2 Data Drift

**Types of Drift**:

**Covariate Drift (Feature Drift)**:
- Input distribution changes: P(X) changes
- Example: User demographics shift over time
- Detection: Compare feature distributions

**Concept Drift (Label Drift)**:
- Relationship changes: P(Y|X) changes
- Example: User preferences evolve
- Detection: Monitor model performance

**Label Drift**:
- Output distribution changes: P(Y) changes
- Example: More fraud cases than usual
- Detection: Compare prediction distributions

**Detection Methods**:
- Statistical tests (KS test, chi-square)
- Distance metrics (KL divergence, Wasserstein)
- Model-based (train model to distinguish old vs new)

### 7.3 Retraining Strategies

**When to Retrain?**:
- **Time-based**: Weekly, monthly (simple, predictable)
- **Performance-based**: When metrics degrade (reactive)
- **Drift-based**: When data distribution changes (proactive)

**How to Retrain?**:
- **Full retrain**: From scratch on all data
- **Incremental**: Update with new data only
- **Transfer learning**: Fine-tune existing model

**Continuous Training**:
- Automated retraining pipeline
- Triggered by schedule or metrics
- Automatic validation before deployment

### 7.4 Model Debugging

**Low Performance**:
- Check data quality (missing values, outliers)
- Analyze errors by segment (user type, time)
- Feature importance analysis
- Compare with baseline

**High Latency**:
- Profile inference time
- Check feature computation time
- Optimize model (quantization, pruning)
- Scale infrastructure

**Unexpected Predictions**:
- Examine input features
- Check for data preprocessing bugs
- Validate feature engineering logic
- Review model assumptions

---

## 8. MLOps Infrastructure

### 8.1 MLOps Components

**MLOps Stack**:
```
┌─────────────────────────────────────────┐
│         Orchestration (Airflow)          │
├─────────────────────────────────────────┤
│  Data        Feature      Experiment     │
│  Pipeline    Store        Tracking       │
│  (Spark)     (Feast)      (MLflow)       │
├─────────────────────────────────────────┤
│  Training    Model        Serving        │
│  (K8s)       Registry     (TF Serving)   │
├─────────────────────────────────────────┤
│  Monitoring  Alerting     Logging        │
│  (Prometheus)(PagerDuty)  (ELK)          │
└─────────────────────────────────────────┘
```

### 8.2 CI/CD for ML

**Continuous Integration**:
- Code testing (unit, integration)
- Data validation tests
- Model training tests
- Performance benchmarks

**Continuous Deployment**:
- Automated model deployment
- Canary releases
- Rollback mechanisms
- Environment consistency

**ML-Specific Considerations**:
- **Data versioning**: DVC, LakeFS
- **Model versioning**: MLflow, Git LFS
- **Reproducibility**: Pin dependencies, seed random state

### 8.3 Infrastructure as Code

**Benefits**:
- Version control infrastructure
- Reproducible environments
- Easy rollbacks
- Documentation

**Tools**:
- **Terraform**: Multi-cloud provisioning
- **CloudFormation**: AWS native
- **Pulumi**: Code-based (Python, TypeScript)

---

## 9. Common Design Patterns

### 9.1 Recommendation Systems

**Architecture**:
```
1. Candidate Generation (Offline)
   - Retrieve 100s of candidates per user
   - Fast, simple models (collaborative filtering, popularity)

2. Ranking (Online)
   - Rank candidates by predicted engagement
   - Complex models, rich features

3. Re-ranking (Business Logic)
   - Apply business rules (diversity, freshness)
   - Filter inappropriate content
```

**Challenges**:
- Cold start (new users/items)
- Scalability (millions of users × items)
- Exploration vs exploitation
- Filter bubbles

**Solutions**:
- Hybrid models (content + collaborative)
- Two-tower architecture (user/item embeddings)
- Multi-armed bandits
- Diversity injection

### 9.2 Search Ranking

**Pipeline**:
```
Query → Query Understanding → Retrieval → Ranking → Blending
```

**Query Understanding**:
- Spell correction
- Query expansion
- Intent classification

**Retrieval**:
- Inverted index (ElasticSearch)
- Vector search (semantic similarity)
- Hybrid (keyword + semantic)

**Ranking**:
- Learning to Rank (LambdaMART, LambdaRank)
- Features: relevance, quality, personalization
- Pointwise, pairwise, or listwise

### 9.3 Fraud Detection

**Characteristics**:
- Highly imbalanced (< 1% fraud)
- Adversarial (fraudsters adapt)
- Real-time requirements
- High cost of false negatives

**Approach**:
1. **Rule-based filters**: Block obvious fraud
2. **ML model**: Score remaining transactions
3. **Manual review**: High-risk cases
4. **Feedback loop**: Learn from manual reviews

**Techniques**:
- Anomaly detection (isolation forest)
- Graph analysis (fraud rings)
- Sequential models (transaction patterns)
- Ensemble methods (combine signals)

### 9.4 Content Moderation

**Challenges**:
- Multi-modal (text, image, video)
- Evolving policy (new types of harm)
- Precision vs recall trade-off
- Latency requirements

**Architecture**:
```
Content → Pre-filters → ML Classifier → Human Review → Action
            (Rules)      (High recall)    (Borderline)   (Remove/Flag)
```

**Techniques**:
- Multi-task learning (hate speech, spam, NSFW)
- Active learning (prioritize hard examples)
- Adversarial training (robust to attacks)

---

## 10. Case Studies & Examples

### Example 1: YouTube Recommendations

**Requirements**:
- Billions of videos, billions of users
- Real-time recommendations
- Maximize watch time

**System Design**:
1. **Candidate Generation**:
   - Deep neural network
   - Input: User history, context
   - Output: Top 100 candidates from millions

2. **Ranking**:
   - Another deep neural network
   - Rich features (video metadata, user features)
   - Output: Ranked list

3. **Serving**:
   - Candidate generation: Offline (approximate nearest neighbors)
   - Ranking: Online
   - Cache popular results

**Key Innovations**:
- Two-stage approach (efficiency)
- Deep learning for both stages
- Implicit feedback (watch time)

---

### Example 2: Uber ETA Prediction

**Requirements**:
- Predict arrival time
- Real-time (< 100ms)
- High accuracy (affects pricing, matching)

**System Design**:
1. **Features**:
   - Route characteristics (distance, traffic)
   - Historical trip data
   - Real-time conditions (weather, events)
   - Driver behavior

2. **Model**:
   - Gradient Boosted Trees (XGBoost)
   - Fast inference, interpretable

3. **Serving**:
   - Online prediction
   - Feature caching
   - Model versioning

4. **Monitoring**:
   - Actual vs predicted ETA
   - By city, time of day
   - Retrain when drift detected

---

## Interview Strategy

### Typical Interview Flow

1. **Clarify (5-10 min)**:
   - Understand business problem
   - Define success metrics
   - Clarify constraints (latency, scale)

2. **High-Level Design (10-15 min)**:
   - Sketch end-to-end system
   - Data → Features → Model → Serving
   - Identify major components

3. **Deep Dive (20-25 min)**:
   - Interviewer chooses area to explore
   - Model selection rationale
   - Feature engineering details
   - Scaling considerations
   - Monitoring strategy

4. **Wrap-Up (5 min)**:
   - Trade-offs and alternatives
   - Potential improvements
   - Questions for interviewer

### Communication Tips

- **Think out loud**: Explain your reasoning
- **Ask clarifying questions**: Don't make assumptions
- **Consider trade-offs**: No perfect solution
- **Be specific**: Use actual technologies and metrics
- **Stay high-level initially**: Don't dive into details too early
- **Know when to go deep**: Read interviewer cues

---

## Key Takeaways

1. **Start with the problem**: Understand business objectives before jumping to models
2. **Data is king**: Good data > fancy models
3. **Simple baselines**: Always start simple, iterate
4. **End-to-end thinking**: Model is just one piece of the system
5. **Monitoring is critical**: Models degrade over time
6. **Trade-offs everywhere**: Latency vs accuracy, cost vs performance
7. **Business impact**: Always tie back to business metrics

---

## Further Resources

### Books
- "Designing Machine Learning Systems" by Chip Huyen
- "Machine Learning Systems Design" by Valliappa Lakshmanan
- "Building Machine Learning Powered Applications" by Emmanuel Ameisen

### Courses
- Stanford CS 329S: Machine Learning Systems Design
- Full Stack Deep Learning

### Blogs
- Chip Huyen's blog (huyenchip.com)
- Eugene Yan's blog (eugeneyan.com)
- Netflix Tech Blog, Uber Engineering Blog

### Practice
- Real ML system design interviews
- Design common systems (recommendations, search, fraud detection)
- Read company engineering blogs

---

**Last Updated**: December 2024
