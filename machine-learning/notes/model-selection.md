# Model Selection and Cross-Validation

A comprehensive guide to evaluating, selecting, and tuning machine learning models through proper validation strategies.

## Table of Contents

1. [Why Model Selection Matters](#why-model-selection-matters)
2. [Train-Validation-Test Split](#train-validation-test-split)
3. [Cross-Validation Techniques](#cross-validation-techniques)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Model Selection Strategies](#model-selection-strategies)
6. [Common Pitfalls](#common-pitfalls)
7. [Best Practices](#best-practices)

---

## Why Model Selection Matters

### The Fundamental Problem

**Goal**: Select the model that will perform best on **unseen data**, not just training data.

**Challenge**: Models that fit training data perfectly often fail on new data (overfitting).

### Three Questions

1. **Which algorithm?** (Linear regression vs Random Forest vs SVM)
2. **Which hyperparameters?** (Learning rate, regularization strength, tree depth)
3. **How well will it generalize?** (Performance on new data)

**Model selection** answers these questions through systematic evaluation.

---

## Train-Validation-Test Split

### The Gold Standard: Three-Way Split

**Purpose of each set:**

1. **Training Set** (60-70%):
   - Used to fit model parameters (weights, coefficients)
   - Model "learns" from this data

2. **Validation Set** (15-20%):
   - Used to tune hyperparameters
   - Select between different models
   - Monitor overfitting during training

3. **Test Set** (15-20%):
   - **Final evaluation only**
   - Estimate generalization performance
   - Should be touched **only once** at the very end

### Why Three Sets?

**Two sets aren't enough:**
- Training + Test: Can't tune hyperparameters without leaking test information
- Using test set for model selection → overfitting to test set

**Proper workflow:**
```
1. Split data → Train | Validation | Test
2. For each model/hyperparameter combination:
   - Train on training set
   - Evaluate on validation set
3. Select best model based on validation performance
4. Final evaluation on test set (once!)
5. Report test set performance
```

### Implementation

```python
from sklearn.model_selection import train_test_split

# First split: separate test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratify for classification
)

# Second split: train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
)

# Result: 60% train, 20% validation, 20% test
```

### When to Stratify

**Classification**: Always use `stratify=y` to maintain class distribution
**Regression**: Not applicable (continuous targets)
**Imbalanced data**: Critical for maintaining minority class representation

---

## Cross-Validation Techniques

Cross-validation provides more robust performance estimates by using multiple train/validation splits.

### 1. K-Fold Cross-Validation

**Most common** validation strategy.

**Algorithm:**
1. Split data into K equal-sized folds
2. For each fold i = 1 to K:
   - Use fold i as validation
   - Use remaining K-1 folds as training
   - Train model and evaluate
3. Average performance across all K folds

**Typical K values:**
- K = 5: Fast, reasonable variance
- K = 10: More accurate, slower
- K = n (LOOCV): See below

**Advantages:**
- Every example used for both training and validation
- More reliable than single train/val split
- Reduces variance in performance estimate

**Disadvantages:**
- K times slower than single split
- Still has variance (different random splits give different results)

**Formula for performance estimate:**
$$
\text{CV Score} = \frac{1}{K} \sum_{i=1}^{K} \text{Score}_i
$$

**Standard error:**
$$
\text{SE} = \frac{\sigma}{\sqrt{K}}
$$
where $\sigma$ is standard deviation of fold scores.

**Implementation:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model, X, y,
    cv=5,  # 5-fold CV
    scoring='accuracy'  # or 'neg_mean_squared_error', etc.
)

print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 2. Stratified K-Fold

**For classification only**: Maintains class distribution in each fold.

**Why needed:**
- Regular K-Fold might create folds with very different class distributions
- Especially problematic for imbalanced datasets
- Each fold should be representative of the overall dataset

**Example problem:**
- Dataset: 90% class 0, 10% class 1
- Random fold might have 95% class 0, 5% class 1
- Another fold might have 85% class 0, 15% class 1
- Inconsistent evaluation

**Stratified K-Fold solution:**
- Each fold has approximately 90% class 0, 10% class 1
- Consistent, fair evaluation

**Implementation:**
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Train and evaluate
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
```

**Default for classification:**
- `cross_val_score` uses StratifiedKFold by default for classifiers
- Use regular KFold for regression

### 3. Leave-One-Out Cross-Validation (LOOCV)

**Extreme case**: K = n (number of samples)

**Algorithm:**
- For each sample:
  - Use that sample as validation (size 1)
  - Use all other n-1 samples as training
  - Train and evaluate
- Average across all n iterations

**Properties:**
- **Deterministic**: No randomness
- **Nearly unbiased**: Uses maximum training data (n-1)
- **High variance**: Each fold differs by only one sample
- **Very expensive**: n iterations

**Formula:**
$$
\text{LOOCV Score} = \frac{1}{n} \sum_{i=1}^{n} L(y_i, \hat{f}^{(-i)}(x_i))
$$
where $\hat{f}^{(-i)}$ is model trained on all data except sample i.

**When to use:**
- Small datasets (n < 100)
- When maximum training data is critical
- Computational cost acceptable

**When NOT to use:**
- Large datasets (too slow)
- When variance in estimate is a concern

**Implementation:**
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
```

### 4. Repeated K-Fold Cross-Validation

**Idea**: Run K-Fold CV multiple times with different random splits.

**Algorithm:**
1. For each repetition r = 1 to R:
   - Perform K-Fold CV with different random seed
   - Get K scores
2. Average across all R × K scores

**Advantages:**
- Reduces variance from random fold selection
- More stable performance estimate
- Better understanding of model variability

**Typical usage:**
- 5-Fold CV repeated 10 times = 50 train/val splits
- More reliable than single 5-Fold CV

**Implementation:**
```python
from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rkf)

print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")
```

### 5. Time Series Cross-Validation

**Special case**: When data has temporal order.

**Problem with regular K-Fold:**
- Randomly shuffling time series violates temporal structure
- Training on future data, testing on past = data leakage
- Unrealistic performance estimates

**Solution: Time Series Split (Forward Chaining)**

**Algorithm:**
```
Fold 1: Train [1:100]    → Test [101:200]
Fold 2: Train [1:200]    → Test [201:300]
Fold 3: Train [1:300]    → Test [301:400]
...
```

Each fold:
- Training set grows (all data up to that point)
- Test set is the next time period
- **Never train on future data**

**Implementation:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train on past, test on future
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
```

**Variants:**

**Fixed-size training window:**
```
Fold 1: Train [1:100]    → Test [101:200]
Fold 2: Train [101:200]  → Test [201:300]
Fold 3: Train [201:300]  → Test [301:400]
```
Use when older data becomes less relevant.

**Gap between train and test:**
```
Fold 1: Train [1:100]    → Gap [101:110] → Test [111:200]
```
Use when there's a prediction horizon (e.g., predict next month).

### 6. Group K-Fold Cross-Validation

**Use case**: Data has natural groups that should not be split.

**Examples:**
- Medical data: Multiple samples from same patient
- Image data: Multiple crops from same image
- Panel data: Multiple time points from same entity

**Problem:**
- Regular K-Fold might put same patient in both train and test
- Causes data leakage and overoptimistic performance

**Solution:**
- Ensure all samples from a group are in the same fold
- Groups either all in training or all in validation

**Implementation:**
```python
from sklearn.model_selection import GroupKFold

# groups: array indicating group membership
groups = np.array([1, 1, 1, 2, 2, 3, 3, 3, 4, 4])

gkf = GroupKFold(n_splits=3)

for train_idx, val_idx in gkf.split(X, y, groups=groups):
    # Groups are not split across train/val
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
```

### Choosing Cross-Validation Strategy

| Data Type | Best CV Strategy | Why |
|-----------|-----------------|-----|
| **Classification (balanced)** | Stratified K-Fold (K=5 or 10) | Maintains class distribution |
| **Classification (imbalanced)** | Stratified K-Fold | Critical for minority class |
| **Regression** | K-Fold (K=5 or 10) | Standard approach |
| **Time series** | Time Series Split | Respects temporal order |
| **Grouped data** | Group K-Fold | Prevents data leakage |
| **Small dataset (n<100)** | LOOCV or Repeated K-Fold | Maximize training data |
| **Large dataset** | 5-Fold or train/val/test | Faster computation |

---

## Hyperparameter Tuning

**Hyperparameters**: Parameters set before training (not learned from data).

**Examples:**
- Learning rate, regularization strength (λ)
- Number of trees, max depth (Random Forest)
- Number of neighbors (KNN)
- Kernel parameters (SVM)

### 1. Grid Search

**Idea**: Exhaustively try all combinations of hyperparameters.

**Algorithm:**
1. Define grid of hyperparameter values
2. For each combination:
   - Train model with those hyperparameters
   - Evaluate via cross-validation
3. Select combination with best CV score

**Example:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")

# Best model
best_model = grid_search.best_estimator_
```

**Number of fits:**
- 4 C values × 4 gamma values × 2 kernels = 32 combinations
- 5-fold CV → 32 × 5 = 160 model fits

**Advantages:**
- Guaranteed to find best combination in grid
- Thorough

**Disadvantages:**
- Exponential in number of hyperparameters
- Wastes computation on poor regions
- Need to specify grid manually

### 2. Random Search

**Idea**: Try random combinations for a fixed budget.

**Algorithm:**
1. Define distributions for each hyperparameter
2. For each iteration (up to budget):
   - Sample random hyperparameter values
   - Evaluate via cross-validation
3. Select best combination found

**Example:**
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

param_distributions = {
    'C': loguniform(0.01, 100),  # Log-uniform distribution
    'gamma': loguniform(0.0001, 1),
    'kernel': ['rbf', 'linear']
}

random_search = RandomizedSearchCV(
    SVC(),
    param_distributions,
    n_iter=50,  # Number of random samples
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

**Advantages:**
- More efficient than grid search
- Better exploration of continuous spaces
- Easy to add iterations (increase budget)

**Disadvantages:**
- Might miss optimal combination
- Random (different runs give different results)

**Grid vs Random:**

For the same budget, random search often works better:
- Grid search: 10 values × 10 values = 100 combinations
- Random search: 100 random samples exploring entire space

### 3. Bayesian Optimization

**Idea**: Use previous evaluations to guide search toward promising regions.

**How it works:**
1. Build a probabilistic model of objective function
2. Use model to select next hyperparameters to try
3. Update model with result
4. Repeat

**Advantages:**
- More sample-efficient than random search
- Explicitly balances exploration vs exploitation
- Works well for expensive objective functions

**Disadvantages:**
- More complex to set up
- Requires additional library (e.g., Optuna, Hyperopt)

**Example (using Optuna):**
```python
import optuna

def objective(trial):
    params = {
        'C': trial.suggest_loguniform('C', 0.01, 100),
        'gamma': trial.suggest_loguniform('gamma', 0.0001, 1),
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear'])
    }

    model = SVC(**params)
    return cross_val_score(model, X_train, y_train, cv=5).mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
```

### Nested Cross-Validation

**Problem**: Using same data for hyperparameter tuning and performance estimation → optimistic bias.

**Solution**: Nested CV (cross-validation within cross-validation)

**Structure:**
```
Outer Loop (performance estimation):
  For each outer fold:
    Inner Loop (hyperparameter tuning):
      For each inner fold:
        Train and evaluate with different hyperparameters
      Select best hyperparameters
    Train model with best hyperparameters on outer training set
    Evaluate on outer test fold
Average performance across outer folds
```

**Example:**
```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Inner CV: hyperparameter tuning
param_grid = {'C': [0.1, 1, 10]}
inner_cv = GridSearchCV(SVC(), param_grid, cv=3)

# Outer CV: performance estimation
outer_scores = cross_val_score(inner_cv, X, y, cv=5)

print(f"Nested CV score: {outer_scores.mean():.3f} (+/- {outer_scores.std():.3f})")
```

**When to use:**
- Final performance estimate for publication
- Comparing fundamentally different approaches
- When unbiased estimate is critical

**When NOT to use:**
- Computationally expensive (outer_folds × inner_folds × param_combinations)
- Typical workflow: Use regular CV for tuning, then report test set performance

---

## Model Selection Strategies

### 1. Comparing Models

**Goal**: Choose between different algorithms (e.g., Logistic Regression vs Random Forest vs SVM)

**Procedure:**
1. For each model:
   - Tune hyperparameters via inner CV
   - Evaluate via outer CV
2. Select model with best CV performance
3. Retrain on all training data
4. Final evaluation on held-out test set

**Statistical Testing:**

Use paired t-test on CV fold scores:
```python
from scipy.stats import ttest_rel

scores_model_a = cross_val_score(model_a, X, y, cv=5)
scores_model_b = cross_val_score(model_b, X, y, cv=5)

t_stat, p_value = ttest_rel(scores_model_a, scores_model_b)

if p_value < 0.05:
    print("Significant difference between models")
```

### 2. Learning Curves

**Purpose**: Diagnose if model would benefit from more data.

**Method**: Plot performance vs training set size

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5
)
```

**Interpretation:**
- **High bias** (underfitting): Both train and val scores low, converging
  - Solution: More complex model, more features

- **High variance** (overfitting): Large gap between train and val scores
  - Solution: More data, regularization, simpler model

- **Good fit**: Small gap, both scores high

### 3. Validation Curves

**Purpose**: Understand effect of a single hyperparameter.

**Method**: Plot performance vs hyperparameter value

```python
from sklearn.model_selection import validation_curve

param_range = [0.001, 0.01, 0.1, 1, 10, 100]

train_scores, val_scores = validation_curve(
    model, X, y,
    param_name="alpha",
    param_range=param_range,
    cv=5
)
```

**Interpretation:**
- Underfitting region: Both scores low
- Sweet spot: Val score highest
- Overfitting region: Train high, val decreasing

---

## Common Pitfalls

### 1. Data Leakage

**Definition**: Training data contains information about test data.

**Common sources:**

**Preprocessing before split:**
```python
# WRONG
X_scaled = scaler.fit_transform(X)  # Leakage! Test info in scaling
X_train, X_test = train_test_split(X_scaled, ...)

# CORRECT
X_train, X_test = train_test_split(X, ...)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use train statistics
```

**Feature selection before split:**
```python
# WRONG
selected_features = select_k_best(X, y, k=10)
X_train, X_test = train_test_split(X[selected_features], ...)

# CORRECT
X_train, X_test = train_test_split(X, ...)
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
```

**Time series leakage:**
```python
# WRONG for time series
X_train, X_test = train_test_split(X, ...)  # Random split

# CORRECT
split_point = int(0.8 * len(X))
X_train, X_test = X[:split_point], X[split_point:]
```

### 2. Multiple Testing

**Problem**: Trying many models on same test set → overfitting to test set.

**Solution**: Use test set only once for final evaluation.

### 3. Ignoring Computation Cost

**Problem**: Grid search with too many parameters, large dataset → weeks of computation.

**Solutions:**
- Start with coarse grid, then refine
- Use random search for initial exploration
- Use Bayesian optimization
- Use smaller subset for initial tuning

### 4. Wrong Metric

**Problem**: Optimizing accuracy on imbalanced dataset.

**Solution**: Choose metric appropriate for problem (see evaluation metrics notes).

---

## Best Practices

### 1. Standard Workflow

```python
# 1. Initial train/test split (stratified for classification)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Hyperparameter tuning via CV on training data
param_grid = {...}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train_full, y_train_full)

# 3. Get best model
best_model = grid_search.best_estimator_

# 4. Final evaluation on test set (ONCE!)
test_score = best_model.score(X_test, y_test)
print(f"Test score: {test_score:.3f}")
```

### 2. Choosing K in K-Fold CV

**General recommendations:**
- Small datasets (n < 1000): K = 10 or LOOCV
- Medium datasets: K = 5 or 10
- Large datasets: K = 3 or 5 (faster)

**Trade-offs:**
- Larger K: Lower bias, higher variance, slower
- Smaller K: Higher bias, lower variance, faster

### 3. Reporting Results

**What to report:**
- CV score: mean ± std
- Test score: single number
- Hyperparameters used
- Number of CV folds
- Metric used

**Example:**
```
Model: Random Forest
Hyperparameters: {n_estimators: 100, max_depth: 10}
5-Fold CV Score: 0.87 ± 0.02
Test Score: 0.85
Metric: Accuracy
```

### 4. Reproducibility

Always set random seeds:
```python
# Data splitting
train_test_split(..., random_state=42)

# Cross-validation
KFold(shuffle=True, random_state=42)

# Model training
RandomForestClassifier(random_state=42)
```

---

## Summary

### Key Takeaways

1. **Always use separate test set** - Touch it only once
2. **Use cross-validation** for reliable performance estimates
3. **Stratify for classification** - Especially with imbalanced data
4. **Respect temporal order** for time series
5. **Avoid data leakage** - Fit on training data only
6. **Choose appropriate CV strategy** based on data characteristics
7. **Report both CV and test scores** with standard deviations

### Quick Decision Guide

**How to split data?**
- Standard: 60% train, 20% val, 20% test
- Or: 80% train (with CV), 20% test

**Which CV method?**
- Classification → Stratified K-Fold (K=5)
- Regression → K-Fold (K=5)
- Time series → Time Series Split
- Grouped data → Group K-Fold
- Small data → LOOCV

**How to tune hyperparameters?**
- Few hyperparameters → Grid Search
- Many hyperparameters → Random Search
- Expensive evaluation → Bayesian Optimization

**How to compare models?**
- Same CV folds for all models
- Statistical testing on fold scores
- Final evaluation on test set

### The Golden Rule

**Fit on training data, evaluate on validation data, report on test data.**

Never leak information from validation/test into training!
