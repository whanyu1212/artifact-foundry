# Bagging (Bootstrap Aggregating) - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Bootstrap Sampling](#bootstrap-sampling)
3. [The Bagging Algorithm](#the-bagging-algorithm)
4. [Why Bagging Works](#why-bagging-works)
5. [Advantages and Limitations](#advantages-and-limitations)
6. [Hyperparameters](#hyperparameters)
7. [Practical Examples](#practical-examples)
8. [Comparison with Other Methods](#comparison-with-other-methods)

---

## Introduction

**Bagging** (Bootstrap Aggregating) is an ensemble learning technique that combines predictions from multiple models trained on different subsets of the training data to reduce variance and improve accuracy.

### Key Idea

Instead of training one model on the full dataset, bagging:
1. Creates multiple bootstrapped datasets (random sampling with replacement)
2. Trains a separate model on each bootstrapped dataset
3. Aggregates predictions by voting (classification) or averaging (regression)

### Core Principle

```
Variance Reduction through Averaging

If you have N independent models with variance σ²:
- Single model variance: σ²
- Average of N models variance: σ²/N

Bagging approximates this by creating "pseudo-independent" models
```

---

## Bootstrap Sampling

### What is Bootstrap Sampling?

**Bootstrap** = random sampling **with replacement** from the original dataset

```python
Original data: [1, 2, 3, 4, 5]

Bootstrap sample 1: [1, 1, 3, 5, 5]  # 1 and 5 repeated, 2 and 4 missing
Bootstrap sample 2: [2, 3, 3, 4, 5]  # 3 repeated, 1 missing
Bootstrap sample 3: [1, 2, 2, 4, 5]  # 2 repeated, 3 missing
```

### Key Properties

**1. Sample Size**: Each bootstrap sample has the **same size** as original dataset

**2. Sampling with Replacement**: Data points can appear multiple times

**3. Coverage**: On average, each bootstrap sample contains ~63.2% unique samples

**Mathematical proof:**
```
Probability a sample is NOT selected in one draw: (n-1)/n

Probability NOT selected in n draws: ((n-1)/n)^n

As n → ∞: ((n-1)/n)^n → 1/e ≈ 0.368

Therefore, probability of being selected: 1 - 1/e ≈ 0.632 (63.2%)
```

**4. Out-of-Bag (OOB) Samples**: ~36.8% of samples not in each bootstrap

- Can be used for validation (no need for separate validation set!)
- Each model is evaluated on its OOB samples
- OOB error ≈ cross-validation error

---

## The Bagging Algorithm

### Algorithm Overview

```
function BAGGING(data, n_estimators, base_model):
    models = []

    # 1. CREATE BOOTSTRAP SAMPLES AND TRAIN
    for i in 1 to n_estimators:
        # Bootstrap sampling
        bootstrap_sample = sample_with_replacement(data, size=len(data))

        # Train base model
        model_i = base_model.fit(bootstrap_sample)
        models.append(model_i)

    # 2. AGGREGATE PREDICTIONS
    function PREDICT(X):
        predictions = [model.predict(X) for model in models]

        # Classification: majority vote
        if classification:
            return mode(predictions)

        # Regression: average
        else:
            return mean(predictions)

    return models, PREDICT
```

### Step-by-Step Example

**Original Dataset:**
```python
X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]
```

**Step 1: Create Bootstrap Samples (n_estimators=3)**
```python
Bootstrap 1:
  Indices: [0, 0, 2, 3, 4]
  X: [[1], [1], [3], [4], [5]]
  y: [0, 0, 1, 1, 1]
  OOB: [1]  # index 1 not selected

Bootstrap 2:
  Indices: [1, 2, 2, 3, 4]
  X: [[2], [3], [3], [4], [5]]
  y: [0, 1, 1, 1, 1]
  OOB: [0]

Bootstrap 3:
  Indices: [0, 1, 2, 3, 3]
  X: [[1], [2], [3], [4], [4]]
  y: [0, 0, 1, 1, 1]
  OOB: [4]
```

**Step 2: Train Base Models**
```python
Tree1 = DecisionTree.fit(Bootstrap1)
Tree2 = DecisionTree.fit(Bootstrap2)
Tree3 = DecisionTree.fit(Bootstrap3)
```

**Step 3: Make Predictions**
```python
X_test = [[2.5]]

Tree1.predict(X_test) = 1
Tree2.predict(X_test) = 1
Tree3.predict(X_test) = 0

# Majority vote
Final prediction = mode([1, 1, 0]) = 1
```

---

## Why Bagging Works

### 1. Variance Reduction

**High Variance Models** (e.g., deep decision trees):
- Very sensitive to training data
- Small changes → different models
- **Overfit easily**

**Bagging Solution:**
- Average multiple high-variance models
- Reduces overall variance
- More stable predictions

### Mathematical Intuition

If we have N **independent** models with:
- Individual variance: σ²
- Average variance: σ²/N

```
Var(Average) = Var((f₁ + f₂ + ... + fₙ)/N)
             = (1/N²) × Var(f₁ + f₂ + ... + fₙ)
             = (1/N²) × N × σ²  # if independent
             = σ²/N
```

**Reality**: Bagged models are **not independent** (trained on overlapping data)
- But still achieves significant variance reduction
- Typical reduction: 10-30% depending on correlation

### 2. Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Error

Bagging:
  ✓ Reduces variance (averaging effect)
  ✗ Bias unchanged (same model capacity)
  → Best for high-variance, low-bias models
```

**This is why decision trees are perfect for bagging:**
- Deep trees: low bias, high variance
- Bagging: reduces variance, keeps low bias
- Result: optimal tradeoff

### 3. Out-of-Bag Validation

Each model's OOB samples provide free validation:

```python
# For each sample in training set:
oob_predictions = []
for i, sample in enumerate(X_train):
    # Find models that didn't see this sample
    models_without_sample = [m for m in models if i in m.oob_indices]

    # Average their predictions
    oob_pred = mean([m.predict(sample) for m in models_without_sample])
    oob_predictions.append(oob_pred)

oob_score = accuracy(y_train, oob_predictions)
```

**OOB score ≈ Cross-validation score** (empirically proven)
- No need for separate validation set
- Computationally efficient

---

## Advantages and Limitations

### Advantages

**1. Reduces Overfitting**
- Variance reduction through averaging
- Especially effective for high-variance models

**2. Robust and Stable**
- Less sensitive to outliers and noise
- Small data changes → similar ensemble performance

**3. Parallel Training**
- Each model trains independently
- Easily parallelizable → faster training

**4. Out-of-Bag Evaluation**
- Free validation set (~36.8% of data per model)
- No need for cross-validation

**5. Works with Any Base Model**
- Decision trees (most common)
- Neural networks
- SVMs
- Any model with high variance

**6. Handles High-Dimensional Data**
- No dimensionality curse like KNN
- Can handle many features

### Limitations

**1. Doesn't Reduce Bias**
- Only reduces variance
- If base model has high bias → ensemble also has high bias
- **Not suitable for simple models** (e.g., shallow trees)

**2. Less Interpretable**
- Single tree: easy to visualize
- Ensemble of 100 trees: black box

**3. Computational Cost**
- Training N models takes N× time (but parallelizable)
- Prediction requires all N models

**4. Memory Intensive**
- Must store all N models
- Can be prohibitive for very large datasets

**5. Diminishing Returns**
- Adding more estimators improves performance
- But gains decrease: 10 → 50 helps a lot, 500 → 1000 helps little

---

## Hyperparameters

### 1. n_estimators

**Controls**: Number of models in the ensemble

```python
# Too few: high variance remains
bagging = BaggingClassifier(n_estimators=5)

# Good balance
bagging = BaggingClassifier(n_estimators=100)

# More doesn't hurt (diminishing returns)
bagging = BaggingClassifier(n_estimators=500)
```

**Effect:**
- **More estimators** → lower variance, better performance
- **Diminishing returns** after ~100-200 for most problems
- **No overfitting** from too many estimators (unlike depth)

**Typical values**: 50-500

**Rule of thumb**: Start with 100, increase until OOB error plateaus

### 2. max_samples

**Controls**: Number/fraction of samples in each bootstrap

```python
# Use 80% of data per bootstrap
bagging = BaggingClassifier(max_samples=0.8)

# Use 500 samples per bootstrap
bagging = BaggingClassifier(max_samples=500)

# Use all data (default, recommended)
bagging = BaggingClassifier(max_samples=1.0)
```

**Effect:**
- **Smaller samples** → more diversity, but weaker individual models
- **Larger samples** → stronger models, but less diversity

**Default**: 1.0 (same size as training set) - usually optimal

### 3. max_features

**Controls**: Number/fraction of features to consider per model

```python
# Use all features (default for bagging)
bagging = BaggingClassifier(max_features=1.0)

# Use subset of features (random forest approach)
bagging = BaggingClassifier(max_features=0.8)
```

**Effect:**
- **All features**: Standard bagging
- **Subset features**: More diversity (moves toward Random Forest)

**Default**: 1.0 for bagging, sqrt(n) for Random Forest

### 4. bootstrap

**Controls**: Whether to use bootstrap sampling

```python
# Standard bagging (with replacement)
bagging = BaggingClassifier(bootstrap=True)

# Pasting (without replacement)
bagging = BaggingClassifier(bootstrap=False)
```

**Effect:**
- **True**: Bootstrap sampling (default, recommended)
- **False**: "Pasting" - samples without replacement (less common)

### 5. oob_score

**Controls**: Whether to calculate out-of-bag score

```python
bagging = BaggingClassifier(oob_score=True)
bagging.fit(X_train, y_train)
print(f"OOB Score: {bagging.oob_score_}")
```

**Use case**: Quick validation without cross-validation

### Base Model Hyperparameters

**For Decision Trees (most common base model):**

```python
from sklearn.tree import DecisionTreeClassifier

base_tree = DecisionTreeClassifier(
    max_depth=None,        # Deep trees (high variance)
    min_samples_split=2,   # Allow aggressive splitting
    min_samples_leaf=1     # No leaf constraints
)

bagging = BaggingClassifier(
    estimator=base_tree,
    n_estimators=100,
    oob_score=True
)
```

**Key principle**: Use high-variance base models for best results

---

## Practical Examples

### Classification Example

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Dataset
X = np.array([
    [5.1, 3.5], [4.9, 3.0], [4.7, 3.2],  # Class 0
    [7.0, 3.2], [6.4, 3.2], [6.9, 3.1],  # Class 1
])
y = np.array([0, 0, 0, 1, 1, 1])

# Base model: deep decision tree (high variance)
base_tree = DecisionTreeClassifier(
    max_depth=None,
    min_samples_split=2
)

# Bagging ensemble
bagging = BaggingClassifier(
    estimator=base_tree,
    n_estimators=100,
    max_samples=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=42
)

# Train
bagging.fit(X, y)

# Out-of-bag score
print(f"OOB Score: {bagging.oob_score_:.3f}")

# Predict
X_test = np.array([[5.0, 3.3], [6.5, 3.1]])
predictions = bagging.predict(X_test)
print(f"Predictions: {predictions}")  # [0, 1]

# Prediction probabilities
proba = bagging.predict_proba(X_test)
print(f"Probabilities:\n{proba}")
```

### Regression Example

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# House prices based on size
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 300000, 400000, 500000, 600000])

# Base model
base_tree = DecisionTreeRegressor(
    max_depth=None,
    min_samples_split=2
)

# Bagging ensemble
bagging = BaggingRegressor(
    estimator=base_tree,
    n_estimators=100,
    bootstrap=True,
    oob_score=True,
    random_state=42
)

# Train
bagging.fit(X, y)

# OOB score (R² score)
print(f"OOB R² Score: {bagging.oob_score_:.3f}")

# Predict
X_test = np.array([[1800], [2200]])
predictions = bagging.predict(X_test)
print(f"Predictions: {predictions}")
```

### Visualizing Variance Reduction

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import learning_curve

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Single tree vs Bagging
single_tree = DecisionTreeClassifier(max_depth=None)
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100
)

# Learning curves
train_sizes = np.linspace(0.1, 1.0, 10)

for model, label in [(single_tree, "Single Tree"), (bagging, "Bagging")]:
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=5
    )

    plt.plot(train_sizes_abs, val_scores.mean(axis=1), label=label)

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Single Tree vs Bagging: Variance Reduction")
plt.show()
```

**Expected result**: Bagging has higher and more stable validation accuracy

---

## Comparison with Other Methods

### Bagging vs Single Model

| Aspect | Single Model | Bagging |
|--------|-------------|---------|
| **Variance** | High (especially deep trees) | Low (averaging effect) |
| **Bias** | Low (deep trees) | Same as base model |
| **Interpretability** | High (can visualize) | Low (ensemble) |
| **Training time** | Fast | Slow (N× models) |
| **Prediction time** | Fast | Slow (N predictions) |
| **Robustness** | Low (sensitive to data) | High (stable) |

### Bagging vs Boosting

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| **Training** | Parallel (independent) | Sequential (dependent) |
| **Focus** | Reduce variance | Reduce bias |
| **Best for** | High-variance models | High-bias models |
| **Weights** | Equal weight models | Weighted models |
| **Overfitting** | Resistant | Can overfit if not careful |
| **Example** | Random Forest | AdaBoost, Gradient Boosting |

### Bagging vs Random Forest

| Aspect | Bagging | Random Forest |
|--------|---------|---------------|
| **Feature sampling** | Uses all features | Random subset per split |
| **Diversity** | From bootstrap only | Bootstrap + feature sampling |
| **Performance** | Good | Better (more diversity) |
| **Correlation** | Higher between trees | Lower between trees |

**Random Forest = Bagging + Random Feature Selection**

---

## Common Use Cases

### 1. Reducing Overfitting in Decision Trees

**Problem**: Deep decision trees overfit training data

**Solution**: Bagging of deep trees
```python
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100
)
```

### 2. Model Variance Estimation

**Use OOB samples** to estimate prediction uncertainty:
```python
# Get predictions from each estimator
predictions = np.array([
    estimator.predict(X_test)
    for estimator in bagging.estimators_
])

# Variance across predictions
prediction_variance = predictions.var(axis=0)
# High variance → uncertain prediction
```

### 3. Quick Model Validation

**OOB score** provides validation without holdout set:
```python
bagging = BaggingClassifier(oob_score=True)
bagging.fit(X_train, y_train)

# OOB score ≈ cross-validation score
print(bagging.oob_score_)
```

---

## Key Takeaways

1. **Bagging = Bootstrap Aggregating**
   - Create multiple datasets via bootstrap sampling
   - Train model on each
   - Average predictions (regression) or vote (classification)

2. **Reduces Variance, Not Bias**
   - Works best with high-variance, low-bias models
   - Deep decision trees are ideal candidates

3. **Bootstrap Properties**
   - Sample with replacement
   - ~63.2% unique samples per bootstrap
   - ~36.8% out-of-bag for validation

4. **Parallel Training**
   - Models are independent
   - Easily parallelizable

5. **Out-of-Bag Evaluation**
   - Free validation set
   - OOB error ≈ cross-validation error

6. **Foundation for Random Forest**
   - Random Forest adds feature randomness to bagging
   - Even better performance through increased diversity

7. **Hyperparameter Tuning**
   - `n_estimators`: Start with 100, more is better
   - Base model: Use high-variance configuration
   - `max_samples`: 1.0 (default) usually optimal

8. **Trade-offs**
   - ✓ Better accuracy, robustness, stability
   - ✗ Less interpretability, more computation

---

## Further Reading

- **Original Paper**: Breiman, L. (1996) "Bagging Predictors"
- **Bootstrap Method**: Efron, B. (1979) "Bootstrap Methods: Another Look at the Jackknife"
- **Random Forests**: Extension with feature randomness
- **Out-of-Bag Error**: Unbiased error estimation
- **Ensemble Theory**: Bias-variance decomposition for ensembles
