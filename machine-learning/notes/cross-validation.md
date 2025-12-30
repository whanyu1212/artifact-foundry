# Cross-Validation - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Why Cross-Validation?](#why-cross-validation)
3. [Cross-Validation Techniques](#cross-validation-techniques)
4. [K-Fold Cross-Validation](#k-fold-cross-validation)
5. [Stratified K-Fold](#stratified-k-fold)
6. [Leave-One-Out Cross-Validation](#leave-one-out-cross-validation)
7. [Time Series Cross-Validation](#time-series-cross-validation)
8. [Nested Cross-Validation](#nested-cross-validation)
9. [Common Pitfalls](#common-pitfalls)
10. [Implementation Examples](#implementation-examples)

---

## Introduction

**Cross-validation** is a resampling technique used to evaluate machine learning models on a limited data sample. It systematically partitions data into training and validation sets to estimate model performance and generalization ability.

### Key Characteristics
- **Reduces overfitting**: Better estimates generalization than single train-test split
- **Efficient data usage**: Every sample used for both training and validation
- **Variance reduction**: Averaging multiple splits provides more stable estimates
- **Model selection**: Helps choose hyperparameters and compare models

### The Fundamental Problem

```
Single train-test split:
├─ Fast but high variance
├─ Performance depends on which samples end up in test set
└─ Wastes data (test set never used for training)

Cross-validation:
├─ More computation but lower variance
├─ Every sample used for validation exactly once
└─ Better use of limited data
```

---

## Why Cross-Validation?

### The Bias-Variance Trade-off in Evaluation

**Single Train-Test Split Issues:**

```python
# Example: Unstable evaluation
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

print(f"Scores: {scores}")
# Output: [0.82, 0.91, 0.78, 0.88, 0.85, 0.79, 0.90, 0.83, 0.87, 0.81]
# High variance! Which score should we trust?
```

**Cross-Validation Stabilizes Estimates:**

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X, y, cv=5)
print(f"CV Scores: {scores}")
# Output: [0.84, 0.86, 0.83, 0.85, 0.84]
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
# Output: Mean: 0.844 ± 0.010
# More stable and trustworthy!
```

### When to Use Cross-Validation

**Use CV when:**
- Dataset is small or medium-sized (< 100K samples)
- You need reliable performance estimates
- Comparing multiple models or hyperparameters
- Every data point is valuable

**Skip CV when:**
- Dataset is very large (> 1M samples) - simple split is sufficient
- Computational resources are very limited
- Working with time series (use time-aware splits instead)

---

## Cross-Validation Techniques

### Overview

| Method | Folds | Best For | Pros | Cons |
|--------|-------|----------|------|------|
| **K-Fold** | K (typically 5-10) | General purpose | Balanced bias-variance | Ignores data order |
| **Stratified K-Fold** | K | Classification | Preserves class distribution | Only for classification |
| **Leave-One-Out (LOO)** | N (samples) | Small datasets | Maximum training data | Computationally expensive |
| **Leave-P-Out (LPO)** | C(N, P) | Very small datasets | Thorough testing | Extremely expensive |
| **Time Series Split** | Variable | Time series | Respects temporal order | Fewer training samples early on |
| **Group K-Fold** | K | Grouped data | Prevents data leakage | Requires group labels |

---

## K-Fold Cross-Validation

### Algorithm

```
K-Fold Cross-Validation:

1. Shuffle the dataset randomly
2. Split dataset into K equal-sized folds
3. For each fold i (i = 1 to K):
   - Use fold i as validation set
   - Use remaining K-1 folds as training set
   - Train model on training set
   - Evaluate on validation set
   - Store validation score
4. Return mean and std of K validation scores
```

### Visual Representation

```
K=5 Cross-Validation (5-Fold):

Iteration 1: [Test][Train][Train][Train][Train]
Iteration 2: [Train][Test][Train][Train][Train]
Iteration 3: [Train][Train][Test][Train][Train]
Iteration 4: [Train][Train][Train][Test][Train]
Iteration 5: [Train][Train][Train][Train][Test]

Result: 5 scores → average for final estimate
```

### Mathematics

**Expected validation score:**
```
CV_score = (1/K) × Σ(score_i)  for i = 1 to K
```

**Standard error:**
```
SE = std(scores) / √K
```

**Confidence interval (95%):**
```
CI = CV_score ± 1.96 × SE
```

### Implementation

```python
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Create K-Fold splitter
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train and evaluate
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)
    scores.append(score)
    print(f"Fold {fold}: {score:.3f}")

print(f"\nMean CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### Choosing K

**Common values:**
- **K=5**: Good balance, standard choice
- **K=10**: More computation, slightly lower bias
- **K=N (LOO)**: Maximum training data, high variance

**Trade-offs:**

```
K Value      | Training Size | Bias  | Variance | Computation
-------------|---------------|-------|----------|-------------
K=2          | 50%           | High  | Low      | Fast
K=5          | 80%           | Medium| Medium   | Standard
K=10         | 90%           | Low   | High     | Slow
K=N (LOO)    | ~100%         | V.Low | V.High   | Very Slow
```

**Rule of thumb:**
- Small datasets (< 1000): K=10
- Medium datasets (1K-100K): K=5
- Large datasets (> 100K): K=3 or simple split

---

## Stratified K-Fold

### Why Stratification?

**Problem with regular K-Fold for classification:**

```python
# Imbalanced dataset: 90% class 0, 10% class 1
y = [0]*900 + [1]*100

# Regular K-Fold might create imbalanced folds:
Fold 1: 95% class 0, 5% class 1   ← Underrepresented
Fold 2: 88% class 0, 12% class 1  ← Overrepresented
Fold 3: 91% class 0, 9% class 1   ← Close to true distribution
...
# Inconsistent class distribution across folds!
```

**Stratified K-Fold ensures proportional class distribution:**

```python
# All folds maintain ~90% class 0, ~10% class 1
Fold 1: 90% class 0, 10% class 1  ✓
Fold 2: 90% class 0, 10% class 1  ✓
Fold 3: 90% class 0, 10% class 1  ✓
...
```

### Algorithm

```
Stratified K-Fold:

1. For each class c:
   - Shuffle samples of class c
   - Split class c samples into K equal parts
2. Construct folds by combining corresponding parts from each class
3. Proceed with standard K-Fold CV
```

### Implementation

```python
from sklearn.model_selection import StratifiedKFold

# Stratified K-Fold maintains class proportions
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    y_train, y_val = y[train_idx], y[val_idx]

    # Check class distribution
    train_dist = np.bincount(y_train) / len(y_train)
    val_dist = np.bincount(y_val) / len(y_val)

    print(f"Fold {fold}:")
    print(f"  Train: {train_dist}")
    print(f"  Val:   {val_dist}")
```

### When to Use

**Use Stratified K-Fold for:**
- **Classification tasks** (always recommended)
- **Imbalanced datasets** (critical)
- **Small datasets** where each class has few samples
- **Multi-class problems** with varying class sizes

**Don't use for:**
- Regression tasks (no classes to stratify)
- Time series (order matters)

---

## Leave-One-Out Cross-Validation

### Overview

**LOO-CV** is an extreme case of K-Fold where K = N (number of samples).

### Algorithm

```
LOO Cross-Validation:

For each sample i in dataset:
    1. Use sample i as test set (1 sample)
    2. Use all other samples as training set (N-1 samples)
    3. Train model and evaluate on sample i
    4. Store prediction

Final score = metric computed on all N predictions
```

### Visual Representation

```
N=5 samples:

Iteration 1: [Test][Train][Train][Train][Train]
Iteration 2: [Train][Test][Train][Train][Train]
Iteration 3: [Train][Train][Test][Train][Train]
Iteration 4: [Train][Train][Train][Test][Train]
Iteration 5: [Train][Train][Train][Train][Test]

Total: N training runs, each on N-1 samples
```

### Mathematics

**LOO Error:**
```
LOO_error = (1/N) × Σ L(y_i, ŷ_i)  for i = 1 to N

where:
- y_i = true label of sample i
- ŷ_i = prediction when sample i is held out
- L = loss function
```

**Bias and Variance:**
```
Bias: Very low (trains on N-1 samples, nearly full dataset)
Variance: High (N models are highly correlated)
```

### Implementation

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier

loo = LeaveOneOut()
scores = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

print(f"LOO Score: {np.mean(scores):.3f}")
print(f"Number of fits: {len(scores)}")  # Equal to N
```

### Pros and Cons

**Advantages:**
- **Minimum bias**: Each model trains on N-1 samples
- **Deterministic**: No randomness in splits
- **Maximum data usage**: Every sample used for training (N-1)/N times

**Disadvantages:**
- **Computationally expensive**: Requires N model fits
- **High variance**: Individual predictions are highly correlated
- **Not suitable for large datasets**: Impractical when N > 10,000

### When to Use

**Use LOO-CV when:**
- Dataset is very small (N < 100)
- Every training sample is precious
- Model training is fast

**Avoid when:**
- Dataset is medium or large (use 5-fold or 10-fold instead)
- Training is slow (LOO will be N times slower than single fit)

---

## Time Series Cross-Validation

### Why Time Series is Different

**Problem:** Traditional K-Fold shuffles data, which **breaks temporal dependencies** and causes **data leakage**.

```python
# WRONG: Regular K-Fold on time series
# This lets the model "see the future"!
[2020][2021][2022][2023][2024]
  ↓
Fold 1: [Test][Train][Train][Train][Train]  ← Training on future data!
```

**Solution:** Time Series Split maintains temporal order.

```python
# CORRECT: Time Series Split
# Model always predicts the future based on past
[2020][2021][2022][2023][2024]
  ↓
Fold 1: [Train]|[Test][----][----][----]
Fold 2: [Train][Train]|[Test][----][----]
Fold 3: [Train][Train][Train]|[Test][----]
Fold 4: [Train][Train][Train][Train]|[Test]
```

### Algorithm

```
Time Series Split:

Given N samples and K splits:

For split i (i = 1 to K):
    train_size = (N / (K+1)) × (i+1)
    test_size = N / (K+1)

    Train: samples[0 : train_size]
    Test:  samples[train_size : train_size + test_size]
```

### Visual Representation

```
TimeSeriesSplit with 5 folds:

Fold 1: [Train                    ]|[Test ]
Fold 2: [Train                         ]|[Test ]
Fold 3: [Train                              ]|[Test ]
Fold 4: [Train                                   ]|[Test ]
Fold 5: [Train                                        ]|[Test ]

Note: Training set grows, test set size stays constant
```

### Implementation

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    print(f"Fold {fold}:")
    print(f"  Train: {train_idx[0]:4d} to {train_idx[-1]:4d}  (N={len(train_idx)})")
    print(f"  Test:  {test_idx[0]:4d} to {test_idx[-1]:4d}  (N={len(test_idx)})")

    # Train on past, test on future
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  Score: {score:.3f}\n")
```

### Expanding vs. Sliding Window

**Expanding Window (default):**
```
Uses all historical data for training
Fold 1: [Train            ]|[Test]
Fold 2: [Train                 ]|[Test]
Fold 3: [Train                      ]|[Test]
```

**Sliding Window (fixed size):**
```python
# Custom implementation with fixed window
window_size = 1000

for i in range(n_splits):
    train_end = initial_train_size + i * step_size
    train_start = train_end - window_size
    test_end = train_end + test_size

    train_idx = range(train_start, train_end)
    test_idx = range(train_end, test_end)
```

```
Uses only recent data (useful if old data is less relevant)
Fold 1: [----][Train      ]|[Test]
Fold 2: [----][----][Train]|[Test]
Fold 3: [----][----][----][Train]|[Test]
```

### Important Considerations

**1. Feature Engineering with Time:**
```python
# WRONG: Using future information
X['rolling_mean_7d'] = y.rolling(7).mean()  # Includes future values!

# CORRECT: Only use past information
X['rolling_mean_7d'] = y.shift(1).rolling(7).mean()  # Lag by 1 period
```

**2. Avoiding Data Leakage:**
- Never normalize/scale using test set statistics
- Compute all statistics only on training set
- Refit scalers for each fold

```python
from sklearn.preprocessing import StandardScaler

for train_idx, test_idx in tscv.split(X):
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])

    # Apply same transformation to test data
    X_test_scaled = scaler.transform(X[test_idx])  # Note: transform, not fit_transform!
```

**3. Walk-Forward Validation:**
```python
# Alternative: retrain model at each time step
predictions = []

for t in range(train_size, len(X)):
    # Train on all data up to time t
    X_train = X[:t]
    y_train = y[:t]

    model.fit(X_train, y_train)

    # Predict next time step
    pred = model.predict(X[t:t+1])
    predictions.append(pred[0])
```

---

## Nested Cross-Validation

### The Problem: Hyperparameter Tuning with CV

**Naive approach (WRONG - data leakage):**
```python
# This leaks information from test set into model selection!
best_score = 0
for depth in [3, 5, 10]:
    scores = cross_val_score(DecisionTreeClassifier(max_depth=depth), X, y, cv=5)
    if scores.mean() > best_score:
        best_depth = depth
        best_score = scores.mean()

# best_score is OPTIMISTICALLY BIASED!
# We selected hyperparameters using the same data we're evaluating on
```

### Solution: Nested Cross-Validation

**Two levels of CV:**
1. **Outer loop**: Estimates generalization performance (unbiased)
2. **Inner loop**: Selects hyperparameters (biased, but doesn't matter)

### Algorithm

```
Nested CV:

Outer CV (K_outer folds):
    For each outer fold i:
        Split data into outer_train and outer_test

        Inner CV (K_inner folds) on outer_train:
            For each outer fold j:
                Split outer_train into inner_train and inner_val
                For each hyperparameter configuration:
                    Train model on inner_train
                    Evaluate on inner_val
            Select best hyperparameters based on inner CV

        Train model with best hyperparameters on outer_train
        Evaluate on outer_test
        Store outer score

Return mean of outer scores (unbiased estimate!)
```

### Visual Representation

```
Nested CV (Outer=3, Inner=3):

OUTER FOLD 1:
  Outer Train: [====================]  Outer Test: [====]
    Inner CV on Outer Train:
      Inner Fold 1: [Train][Train][Test]
      Inner Fold 2: [Train][Test][Train]
      Inner Fold 3: [Test][Train][Train]
    → Select best params → Train on full Outer Train → Eval on Outer Test

OUTER FOLD 2:
  Outer Train: [====================]  Outer Test: [====]
    Inner CV on Outer Train:
      [Same inner CV process]
    → Select best params → Train on full Outer Train → Eval on Outer Test

OUTER FOLD 3:
  Outer Train: [====================]  Outer Test: [====]
    Inner CV on Outer Train:
      [Same inner CV process]
    → Select best params → Train on full Outer Train → Eval on Outer Test

Final Score: Average of 3 outer test scores (UNBIASED!)
```

### Implementation

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inner CV: hyperparameter tuning
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
clf = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=param_grid,
    cv=inner_cv,
    scoring='accuracy'
)

# Outer CV: performance estimation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
nested_scores = cross_val_score(clf, X, y, cv=outer_cv, scoring='accuracy')

print(f"Nested CV Scores: {nested_scores}")
print(f"Mean: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
```

### Nested CV vs. Regular CV

```python
# Regular CV (OPTIMISTIC BIAS):
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X, y)
print(f"Regular CV: {grid_search.best_score_:.3f}")  # Biased!

# Nested CV (UNBIASED):
nested_scores = cross_val_score(grid_search, X, y, cv=5)
print(f"Nested CV: {nested_scores.mean():.3f}")  # True performance!

# Output example:
# Regular CV: 0.875  ← Optimistic (saw test data during selection)
# Nested CV:  0.841  ← Realistic (test data never used for selection)
```

### When to Use

**Use Nested CV when:**
- Reporting model performance in papers or to stakeholders
- Comparing different algorithms fairly
- Dataset is small-to-medium (< 10K samples)

**Skip Nested CV when:**
- Dataset is very large (simple split is sufficient)
- Only tuning hyperparameters (use regular GridSearchCV)
- Computational resources are limited (nested CV is K_outer × K_inner times slower)

**Important:** After nested CV, retrain the final model on the **entire dataset** with hyperparameters selected by inner CV on the full data.

---

## Common Pitfalls

### 1. Data Leakage

**Problem:** Information from validation set leaks into training.

```python
# WRONG: Scaling before splitting
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses statistics from ALL data!

kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    # Model has seen validation set statistics!
```

```python
# CORRECT: Scale within each fold
kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Only training data
    X_val_scaled = scaler.transform(X_val)  # Apply to validation
```

### 2. Not Stratifying for Classification

**Problem:** Imbalanced folds lead to unreliable estimates.

```python
# WRONG: Regular K-Fold for classification
kf = KFold(n_splits=5)  # Might create imbalanced folds

# CORRECT: Stratified K-Fold
skf = StratifiedKFold(n_splits=5)  # Maintains class proportions
```

### 3. Using Wrong CV for Time Series

**Problem:** Shuffling time series data causes data leakage.

```python
# WRONG: K-Fold with shuffle=True for time series
kf = KFold(n_splits=5, shuffle=True)  # Model sees future!

# CORRECT: Time Series Split
tscv = TimeSeriesSplit(n_splits=5)  # Respects temporal order
```

### 4. Choosing K Too Small or Too Large

**Problem:**
- **K too small (e.g., 2)**: High bias, each model trains on only 50% of data
- **K too large (e.g., N)**: High variance, expensive computation

```python
# Too small
kf = KFold(n_splits=2)  # High bias

# Too large
loo = LeaveOneOut()  # High variance, expensive

# Just right
kf = KFold(n_splits=5)  # Balanced
```

### 5. Reporting Training Score

**Problem:** Reporting performance on training folds instead of validation.

```python
# WRONG: Evaluating on training set
for train_idx, val_idx in kf.split(X):
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)  # Training score! Optimistic!

# CORRECT: Evaluating on validation set
for train_idx, val_idx in kf.split(X):
    clf.fit(X_train, y_train)
    score = clf.score(X_val, y_val)  # Validation score ✓
```

### 6. Nested CV Without Final Retrain

**Problem:** After nested CV, forgetting to train final model on full data.

```python
# After nested CV determines best hyperparameters:
best_params = {'max_depth': 5, 'min_samples_split': 10}

# MUST retrain on entire dataset for production use:
final_model = DecisionTreeClassifier(**best_params)
final_model.fit(X, y)  # Train on ALL data
```

---

## Implementation Examples

### Example 1: Basic Cross-Validation Comparison

```python
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    StratifiedKFold,
    ShuffleSplit
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

# Load data
X, y = load_breast_cancer(return_X_y=True)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# 1. Simple K-Fold
kf_scores = cross_val_score(clf, X, y, cv=5)
print(f"K-Fold (5):          {kf_scores.mean():.3f} ± {kf_scores.std():.3f}")

# 2. Stratified K-Fold (recommended for classification)
skf_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5))
print(f"Stratified K-Fold:   {skf_scores.mean():.3f} ± {skf_scores.std():.3f}")

# 3. Leave-One-Out (expensive!)
loo_scores = cross_val_score(clf, X, y, cv=len(X))
print(f"LOO:                 {loo_scores.mean():.3f} ± {loo_scores.std():.3f}")

# 4. Shuffle Split (random subsampling)
ss_scores = cross_val_score(clf, X, y, cv=ShuffleSplit(n_splits=10, test_size=0.2))
print(f"Shuffle Split:       {ss_scores.mean():.3f} ± {ss_scores.std():.3f}")
```

### Example 2: Cross-Validation with Custom Scoring

```python
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# Define multiple scorers
scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

from sklearn.model_selection import cross_validate

results = cross_validate(
    clf, X, y,
    cv=StratifiedKFold(n_splits=5),
    scoring=scoring,
    return_train_score=True
)

print("Validation Scores:")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    scores = results[f'test_{metric}']
    print(f"  {metric:10s}: {scores.mean():.3f} ± {scores.std():.3f}")
```

### Example 3: Hyperparameter Tuning with Nested CV

```python
from sklearn.model_selection import GridSearchCV
import numpy as np

# Hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Inner CV for hyperparameter selection
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
clf = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=inner_cv,
    scoring='f1_macro',
    n_jobs=-1
)

# Outer CV for performance estimation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run nested CV
nested_scores = []
best_params_per_fold = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner CV: find best hyperparameters
    clf.fit(X_train, y_train)
    best_params_per_fold.append(clf.best_params_)

    # Outer evaluation
    score = clf.score(X_test, y_test)
    nested_scores.append(score)
    print(f"Fold {fold}: {score:.3f} | Best params: {clf.best_params_}")

print(f"\nNested CV Score: {np.mean(nested_scores):.3f} ± {np.std(nested_scores):.3f}")

# Final model: retrain on all data with most common best params
final_clf = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=inner_cv,
    scoring='f1_macro'
)
final_clf.fit(X, y)
print(f"Final model best params: {final_clf.best_params_}")
```

### Example 4: Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np

# Create time series data
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
X = np.random.randn(1000, 5)  # 5 features
y = np.cumsum(np.random.randn(1000))  # Random walk target

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)

scores = []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Fold {fold}:")
    print(f"  Train: {dates[train_idx[0]]} to {dates[train_idx[-1]]} (n={len(train_idx)})")
    print(f"  Test:  {dates[test_idx[0]]} to {dates[test_idx[-1]]} (n={len(test_idx)})")

    # Train model
    from sklearn.linear_model import Ridge
    model = Ridge()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
    print(f"  R² Score: {score:.3f}\n")

print(f"Mean Time Series CV Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### Example 5: Custom Cross-Validation Strategy

```python
from sklearn.model_selection import BaseCrossValidator

class CustomStratifiedSplit(BaseCrossValidator):
    """
    Custom CV: Stratified split with minimum samples per class in test set
    """
    def __init__(self, n_splits=5, min_samples_per_class=10):
        self.n_splits = n_splits
        self.min_samples_per_class = min_samples_per_class

    def split(self, X, y, groups=None):
        n_samples = len(y)
        indices = np.arange(n_samples)

        # Group by class
        unique_classes = np.unique(y)
        class_indices = {c: indices[y == c] for c in unique_classes}

        for fold in range(self.n_splits):
            train_idx = []
            test_idx = []

            for c in unique_classes:
                c_indices = class_indices[c]
                np.random.shuffle(c_indices)

                # Take at least min_samples_per_class for test
                n_test = max(self.min_samples_per_class, len(c_indices) // self.n_splits)

                fold_test = c_indices[fold*n_test:(fold+1)*n_test]
                fold_train = np.setdiff1d(c_indices, fold_test)

                train_idx.extend(fold_train)
                test_idx.extend(fold_test)

            yield np.array(train_idx), np.array(test_idx)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Use custom CV
custom_cv = CustomStratifiedSplit(n_splits=5, min_samples_per_class=20)
scores = cross_val_score(clf, X, y, cv=custom_cv)
print(f"Custom CV Score: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Key Takeaways

1. **Cross-validation provides more reliable performance estimates** than a single train-test split by reducing variance through averaging.

2. **Choose the right CV strategy:**
   - **General classification**: Stratified K-Fold (K=5 or 10)
   - **Regression**: K-Fold (K=5 or 10)
   - **Time series**: TimeSeriesSplit (never shuffle!)
   - **Small datasets**: Leave-One-Out or K=10
   - **Hyperparameter tuning**: Nested CV for unbiased estimates

3. **Stratification is critical for classification** to ensure each fold has representative class distributions, especially with imbalanced data.

4. **Avoid data leakage:**
   - Fit preprocessing (scaling, imputation) **only on training folds**
   - Apply transformations to validation folds
   - Never use validation data for any training decisions

5. **K=5 is the standard choice** providing good bias-variance trade-off for most problems.

6. **Time series requires special handling** to respect temporal order and prevent future information leakage.

7. **Nested CV is essential** for unbiased performance estimates when comparing models with hyperparameter tuning.

8. **Computation trade-offs:**
   - K-Fold: K model fits
   - LOO: N model fits (expensive!)
   - Nested CV: K_outer × K_inner model fits (very expensive!)

9. **Always report mean ± std** to communicate both expected performance and variability.

10. **After CV, retrain on full dataset** for the final production model.

---

## Further Reading

- **Original K-Fold Paper**: Stone, M. (1974) "Cross-Validatory Choice and Assessment of Statistical Predictions"
- **Stratification**: Kohavi, R. (1995) "A Study of Cross-Validation and Bootstrap for Accuracy Estimation"
- **Time Series CV**: Bergmeir, C. & Benítez, J.M. (2012) "On the Use of Cross-validation for Time Series Prediction"
- **Nested CV**: Varma, S. & Simon, R. (2006) "Bias in Error Estimation When Using Cross-validation for Model Selection"
- **Scikit-learn Documentation**: [Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
