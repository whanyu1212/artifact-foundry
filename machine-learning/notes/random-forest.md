# Random Forest - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Algorithm Overview](#algorithm-overview)
3. [Key Differences from Bagging](#key-differences-from-bagging)
4. [Why Random Forest Works](#why-random-forest-works)
5. [Feature Importance](#feature-importance)
6. [Hyperparameters](#hyperparameters)
7. [Advantages and Limitations](#advantages-and-limitations)
8. [Practical Examples](#practical-examples)
9. [Out-of-Bag Error](#out-of-bag-error)

---

## Introduction

**Random Forest** is an ensemble learning method that builds multiple decision trees with two sources of randomness:
1. **Bootstrap sampling** (like bagging)
2. **Random feature selection** at each split (unique to Random Forest)

### Key Innovation

**Bagging** creates diversity through bootstrap sampling alone.

**Random Forest** adds a second layer of randomness: at each split, only consider a random subset of features.

```
Bagging:
  Tree 1: Bootstrap sample 1 → considers ALL features at each split
  Tree 2: Bootstrap sample 2 → considers ALL features at each split
  ...

Random Forest:
  Tree 1: Bootstrap sample 1 → considers √p features at each split
  Tree 2: Bootstrap sample 2 → considers different √p features at each split
  ...
  where p = total number of features
```

### Why the Name?

**Random** = Random feature selection at each split
**Forest** = Ensemble of decision trees

---

## Algorithm Overview

### Random Forest Algorithm

```
function RANDOM_FOREST(data, n_estimators, max_features):
    trees = []

    # BUILD TREES
    for i in 1 to n_estimators:
        # 1. Bootstrap sampling (same as bagging)
        bootstrap_sample = sample_with_replacement(data, size=len(data))

        # 2. Build tree with random feature selection
        tree = BUILD_RF_TREE(bootstrap_sample, max_features)
        trees.append(tree)

    # AGGREGATE PREDICTIONS
    function PREDICT(X):
        predictions = [tree.predict(X) for tree in trees]

        # Classification: majority vote
        if classification:
            return mode(predictions)
        # Regression: average
        else:
            return mean(predictions)

    return trees, PREDICT
```

### Building a Random Forest Tree

```
function BUILD_RF_TREE(data, max_features, depth=0):
    # BASE CASES (same as regular tree)
    if stopping_criteria_met():
        return LEAF(predicted_value)

    # RANDOM FEATURE SELECTION (different from regular tree)
    all_features = [0, 1, 2, ..., p-1]
    candidate_features = random_sample(all_features, size=max_features)

    # Find best split among CANDIDATE features only
    best_split = find_best_split(data, candidate_features)

    if no_good_split_found():
        return LEAF(predicted_value)

    # Recursive split
    left_data = data where feature <= threshold
    right_data = data where feature > threshold

    left_child = BUILD_RF_TREE(left_data, max_features, depth+1)
    right_child = BUILD_RF_TREE(right_data, max_features, depth+1)

    return INTERNAL_NODE(feature, threshold, left_child, right_child)
```

### Step-by-Step Example

**Dataset with 4 features:**
```python
X = np.array([
    [1.0, 2.0, 3.0, 4.0],  # sample 0
    [1.5, 2.5, 3.5, 4.5],  # sample 1
    [2.0, 3.0, 4.0, 5.0],  # sample 2
    [2.5, 3.5, 4.5, 5.5],  # sample 3
])
y = np.array([0, 0, 1, 1])

max_features = 2  # √4 = 2
```

**Tree 1:**
```
Bootstrap: [0, 0, 2, 3]  # Samples 0, 2, 3

Root split:
  All features: [0, 1, 2, 3]
  Random subset: [1, 3]  ← Only consider features 1 and 3
  Best split: feature 1 <= 2.75

Left child split:
  Random subset: [0, 2]  ← Different random subset
  Best split: feature 0 <= 1.25

Right child split:
  Random subset: [2, 3]  ← Different random subset
  Best split: feature 3 <= 5.0
```

**Tree 2:**
```
Bootstrap: [1, 2, 2, 3]  # Different bootstrap sample

Root split:
  Random subset: [0, 2]  ← Different from Tree 1
  Best split: feature 2 <= 4.25
  ...
```

**Key point**: Each split sees a different random subset of features!

---

## Key Differences from Bagging

### Bagging (Standard)

```python
# At EVERY split in EVERY tree
available_features = [0, 1, 2, 3, 4, 5]  # ALL features
best_split = find_best_split(data, available_features)
```

**Result**: Trees are highly correlated
- Same strong features dominate
- Similar tree structures
- Limited diversity

### Random Forest

```python
# At EVERY split in EVERY tree
available_features = random_sample([0, 1, 2, 3, 4, 5], size=√6 ≈ 2)
# Example: [1, 4] or [0, 5] or [2, 3], etc.
best_split = find_best_split(data, available_features)
```

**Result**: Trees are decorrelated
- Different features used in different trees
- More diverse tree structures
- Better ensemble performance

### Comparison Table

| Aspect | Bagging | Random Forest |
|--------|---------|---------------|
| **Bootstrap sampling** | ✓ Yes | ✓ Yes |
| **Feature randomness** | ✗ No | ✓ Yes |
| **Features per split** | All features | Random subset |
| **Tree correlation** | Higher | Lower |
| **Diversity** | Moderate | High |
| **Performance** | Good | Better |
| **Best base model** | Any high-variance | Decision trees |

---

## Why Random Forest Works

### 1. Decorrelation of Trees

**Problem with Bagging:**
```python
# Assume one feature is very strong predictor
Features: [age, income, credit_score, years_employed]
# credit_score is best predictor

# In bagging, ALL trees will likely:
Tree 1: Root split on credit_score
Tree 2: Root split on credit_score
Tree 3: Root split on credit_score
...
# Trees are highly similar (correlated)
```

**Random Forest Solution:**
```python
# Force trees to consider different features

Tree 1: Random features = [age, years_employed]
        → Must split on age or years_employed (not credit_score!)

Tree 2: Random features = [income, credit_score]
        → Can split on credit_score

Tree 3: Random features = [age, income]
        → Must find best among age/income
...
# Trees are more diverse (decorrelated)
```

### 2. Mathematical Intuition

**Variance of Average:**
```
For N correlated random variables with:
  - Variance: σ²
  - Correlation: ρ

Var(Average) = ρσ² + (1-ρ)σ²/N

As N → ∞:
  Var(Average) → ρσ²  (correlated term remains)

Key insight:
  - Bagging: High ρ → limited variance reduction
  - Random Forest: Low ρ → better variance reduction
```

**Example:**
```python
# Bagging with highly correlated trees (ρ=0.8)
Variance = 0.8 × σ² + 0.2 × σ²/100 = 0.802σ²
# Only ~20% variance reduction

# Random Forest with decorrelated trees (ρ=0.3)
Variance = 0.3 × σ² + 0.7 × σ²/100 = 0.307σ²
# ~69% variance reduction!
```

### 3. The Bias-Variance-Covariance Decomposition

For an ensemble of M trees:
```
MSE(Ensemble) = Bias² + (1/M)Variance + ((M-1)/M)Covariance

Where:
  - Bias: Same as individual trees
  - Variance: Reduced by 1/M (more trees help)
  - Covariance: Reduced by decorrelation (random features help)
```

**Random Forest tackles both:**
- More trees → reduce variance term
- Random features → reduce covariance term

---

## Feature Importance

One of the most valuable outputs of Random Forest!

### 1. Mean Decrease in Impurity (MDI)

**Idea**: How much does each feature reduce impurity across all trees?

```python
# For each tree
for each split using feature j:
    importance[j] += information_gain × n_samples_at_node

# Average across all trees
importance[j] /= n_estimators

# Normalize to sum to 1
importance /= sum(importance)
```

**Example:**
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

# Feature importances
importances = rf.feature_importances_
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.3f}")

# Output:
# Feature 0: 0.125
# Feature 1: 0.453  ← Most important
# Feature 2: 0.287
# Feature 3: 0.135
```

### 2. Permutation Importance

**Idea**: How much does performance drop when we shuffle a feature?

```python
# 1. Get baseline OOB score
baseline_score = oob_score(rf, X, y)

# 2. For each feature
for j in features:
    # Shuffle feature j
    X_permuted = X.copy()
    X_permuted[:, j] = np.random.permutation(X_permuted[:, j])

    # Get new OOB score
    permuted_score = oob_score(rf, X_permuted, y)

    # Importance = drop in performance
    importance[j] = baseline_score - permuted_score
```

**More reliable** than MDI for:
- Correlated features
- High-cardinality categorical features

### 3. Visualizing Feature Importance

```python
import matplotlib.pyplot as plt

# Get importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), indices)
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()
```

---

## Hyperparameters

### 1. n_estimators

**Controls**: Number of trees in the forest

```python
# Too few: variance not fully reduced
rf = RandomForestClassifier(n_estimators=10)

# Good balance
rf = RandomForestClassifier(n_estimators=100)

# More is better (diminishing returns)
rf = RandomForestClassifier(n_estimators=500)
```

**Effect:**
- More trees → better performance (up to a point)
- No overfitting from too many trees
- Diminishing returns after ~100-300

**Typical values**: 100-500

**Rule of thumb**: Start with 100, monitor OOB error

### 2. max_features

**Controls**: Number of features to consider at each split

**This is THE key hyperparameter that distinguishes Random Forest from bagging!**

```python
# Classification (default: sqrt)
rf = RandomForestClassifier(max_features='sqrt')  # √p features

# Regression (default: 1/3)
rf = RandomForestRegressor(max_features=0.33)  # p/3 features

# All features (becomes bagging)
rf = RandomForestClassifier(max_features=None)  # All p features

# Custom
rf = RandomForestClassifier(max_features=5)  # Exactly 5 features
rf = RandomForestClassifier(max_features=0.5)  # 50% of features
```

**Effect:**
- **Smaller max_features**: More diversity, less correlation, but weaker individual trees
- **Larger max_features**: Stronger trees, but more correlation

**Defaults (good starting points):**
- Classification: `sqrt(p)`
- Regression: `p/3`

**Why different defaults?**
- Classification: Usually many weak features → need fewer per split
- Regression: Usually fewer strong features → need more per split

### 3. max_depth

**Controls**: Maximum depth of each tree

```python
# Shallow trees (faster, may underfit)
rf = RandomForestClassifier(max_depth=5)

# Deep trees (default: None, usually best)
rf = RandomForestClassifier(max_depth=None)
```

**Effect:**
- **None** (unlimited): Usually best for Random Forest (ensemble handles overfitting)
- **Limited depth**: Faster training, less memory, but may underfit

**Typical values**: None (let trees grow fully)

### 4. min_samples_split & min_samples_leaf

**Controls**: Minimum samples to split / required in leaf

```python
rf = RandomForestClassifier(
    min_samples_split=2,   # Default: split as much as possible
    min_samples_leaf=1     # Default: no constraint
)

# More conservative (less overfitting per tree)
rf = RandomForestClassifier(
    min_samples_split=10,
    min_samples_leaf=5
)
```

**Effect:**
- Higher values → simpler trees, faster training
- Usually keep defaults for Random Forest (ensemble handles overfitting)

### 5. bootstrap

**Controls**: Whether to use bootstrap sampling

```python
# Standard Random Forest (recommended)
rf = RandomForestClassifier(bootstrap=True)

# "Random Patches" (all samples, random features)
rf = RandomForestClassifier(bootstrap=False)
```

**Default**: True (use bootstrap)

### 6. oob_score

**Controls**: Calculate out-of-bag error

```python
rf = RandomForestClassifier(oob_score=True)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.3f}")
```

**Use case**: Quick validation without holdout set

### 7. n_jobs

**Controls**: Number of parallel jobs

```python
# Use all CPU cores
rf = RandomForestClassifier(n_jobs=-1)

# Use 4 cores
rf = RandomForestClassifier(n_jobs=4)
```

**Effect**: Faster training (trees train in parallel)

### Hyperparameter Tuning Strategy

```python
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'n_estimators': [100, 200, 500, 1000],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(oob_score=True)

search = RandomizedSearchCV(
    rf,
    param_distributions,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
```

---

## Advantages and Limitations

### Advantages

**1. Excellent Performance**
- Often best "out-of-the-box" algorithm
- Competitive with gradient boosting
- Works well without much tuning

**2. Handles Non-Linearity**
- Captures complex interactions
- No feature engineering needed

**3. Robust to Overfitting**
- Can use unlimited tree depth
- Ensemble naturally regularizes

**4. Feature Importance**
- Automatic feature selection
- Interpretable importance scores

**5. Handles Missing Values**
- Can use surrogate splits (in some implementations)
- Robust to missing data

**6. No Feature Scaling Required**
- Tree-based → invariant to monotonic transformations
- No need to normalize/standardize

**7. Parallel Training**
- Trees train independently
- Efficient use of multi-core CPUs

**8. Out-of-Bag Evaluation**
- Free validation set (~36.8% of data)
- No need for cross-validation

**9. Versatile**
- Classification and regression
- Multi-output problems
- Probability estimates

### Limitations

**1. Less Interpretable than Single Tree**
- Can't easily visualize ensemble
- Black box (but has feature importance)

**2. Larger Model Size**
- Must store all trees
- Can be memory-intensive for very large forests

**3. Slower Prediction**
- Must query all trees
- Slower than single tree (but parallelizable)

**4. Not Great for Extrapolation**
- Predictions limited to range of training data
- Can't extrapolate beyond observed values

**5. Biased with Imbalanced Data**
- Tends to favor majority class
- Needs class balancing strategies

**6. Not Optimal for Very High-Dimensional Sparse Data**
- Text data, genomics sometimes better with linear models
- Many irrelevant features can hurt

---

## Practical Examples

### Classification Example

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Generate dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Classifier
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

# Train
rf.fit(X_train, y_train)

# OOB Score
print(f"OOB Score: {rf.oob_score_:.3f}")

# Test accuracy
test_accuracy = rf.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.3f}")

# Feature importance
importances = rf.feature_importances_
top_features = np.argsort(importances)[::-1][:5]
print(f"Top 5 features: {top_features}")

# Predictions
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)

print(f"Predictions: {y_pred[:5]}")
print(f"Probabilities:\n{y_proba[:5]}")
```

### Regression Example

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Generate dataset
X, y = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=10,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=100,
    max_features=0.33,  # 1/3 for regression
    max_depth=None,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

# Train
rf.fit(X_train, y_train)

# OOB Score (R²)
print(f"OOB R² Score: {rf.oob_score_:.3f}")

# Test performance
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.2f}")
print(f"Test R²: {r2:.3f}")

# Feature importance
importances = rf.feature_importances_
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.3f}")
```

### Feature Importance Analysis

```python
import matplotlib.pyplot as plt
import pandas as pd

# Fit Random Forest
rf = RandomForestClassifier(n_estimators=100, oob_score=True)
rf.fit(X_train, y_train)

# Get feature importances
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'feature': [f'Feature {i}' for i in indices],
    'importance': importances[indices],
    'std': std[indices]
})

# Plot
plt.figure(figsize=(10, 6))
plt.title("Feature Importances with Error Bars")
plt.bar(
    range(X.shape[1]),
    importances[indices],
    yerr=std[indices],
    alpha=0.7
)
plt.xticks(range(X.shape[1]), [f'F{i}' for i in indices])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

print(feature_importance_df.head(10))
```

### Comparing to Single Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

# Single deep tree
tree = DecisionTreeClassifier(max_depth=None, random_state=42)
tree.fit(X_train, y_train)
tree_accuracy = tree.score(X_test, y_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_accuracy = rf.score(X_test, y_test)

print(f"Single Tree Accuracy: {tree_accuracy:.3f}")
print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
print(f"Improvement: {(rf_accuracy - tree_accuracy)*100:.1f}%")
```

**Expected**: Random Forest significantly outperforms single tree

---

## Out-of-Bag Error

### What is OOB Error?

For each training sample:
1. ~36.8% of trees didn't see it during training (out-of-bag)
2. Use those trees to predict that sample
3. Compare prediction to true label
4. Aggregate errors across all samples

**Result**: Unbiased error estimate (similar to cross-validation)

### OOB Score Calculation

```python
# Pseudocode
for each sample i in training set:
    # Find trees that didn't use sample i
    oob_trees = [tree for tree in forest if i not in tree.bootstrap_indices]

    # Aggregate predictions from OOB trees
    oob_predictions[i] = aggregate([tree.predict(X[i]) for tree in oob_trees])

# Calculate accuracy
oob_score = accuracy(y_train, oob_predictions)
```

### Using OOB Score

```python
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)

# OOB score (on training data, but unbiased)
print(f"OOB Score: {rf.oob_score_:.3f}")

# OOB predictions
print(f"OOB Predictions: {rf.oob_decision_function_}")

# Compare to test score
test_score = rf.score(X_test, y_test)
print(f"Test Score: {test_score:.3f}")
print(f"Difference: {abs(rf.oob_score_ - test_score):.3f}")
```

**Expected**: OOB score ≈ test score (within a few percent)

### OOB for Hyperparameter Tuning

```python
# Find optimal n_estimators using OOB error
oob_errors = []
n_estimators_range = range(10, 201, 10)

for n in n_estimators_range:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True)
    rf.fit(X_train, y_train)
    oob_errors.append(1 - rf.oob_score_)

plt.plot(n_estimators_range, oob_errors)
plt.xlabel("Number of Trees")
plt.ylabel("OOB Error")
plt.title("OOB Error vs Number of Trees")
plt.show()

# Find where error plateaus
optimal_n = n_estimators_range[np.argmin(oob_errors)]
print(f"Optimal n_estimators: {optimal_n}")
```

---

## Comparison with Other Algorithms

| Algorithm | Accuracy | Speed | Interpretability | Overfitting Risk |
|-----------|----------|-------|------------------|------------------|
| **Single Tree** | Low | Fast | High | High |
| **Bagging** | Medium | Medium | Low | Low |
| **Random Forest** | High | Medium | Low | Very Low |
| **Gradient Boosting** | Very High | Slow | Very Low | Medium |
| **Linear Models** | Low-Medium | Very Fast | High | Low |
| **Neural Networks** | High | Slow | Very Low | High |

---

## Key Takeaways

1. **Random Forest = Bagging + Random Feature Selection**
   - Bootstrap sampling creates diverse datasets
   - Random features create diverse trees
   - Combination decorrelates trees → better ensemble

2. **Two Sources of Randomness**
   - Bootstrap sampling (which samples)
   - Random features (which features per split)

3. **Decorrelation is Key**
   - Lower tree correlation → better variance reduction
   - `max_features` controls decorrelation strength

4. **Hyperparameters**
   - `n_estimators`: 100-500 (more is better)
   - `max_features`: √p (classification), p/3 (regression)
   - `max_depth`: None (let trees grow)
   - Other params: usually keep defaults

5. **Feature Importance**
   - Automatic feature selection
   - MDI or permutation importance
   - One of Random Forest's biggest advantages

6. **Out-of-Bag Evaluation**
   - Free validation set (~36.8% of data per tree)
   - OOB error ≈ cross-validation error
   - No need for separate validation split

7. **Best Practices**
   - Use deep trees (max_depth=None)
   - Parallelize training (n_jobs=-1)
   - Monitor OOB error for tuning
   - Start with default hyperparameters

8. **When to Use**
   - ✓ Tabular data with mixed feature types
   - ✓ Non-linear relationships
   - ✓ Need feature importance
   - ✓ Want robust "out-of-the-box" performance
   - ✗ Need interpretability
   - ✗ Very high-dimensional sparse data
   - ✗ Need extrapolation beyond training range

---

## Further Reading

- **Original Paper**: Breiman, L. (2001) "Random Forests"
- **Feature Importance**: Strobl et al. (2007) "Bias in random forest variable importance measures"
- **Theory**: "The Elements of Statistical Learning" - Chapter 15
- **Extremely Randomized Trees**: Geurts et al. (2006) - Even more randomness
- **Proximity Measures**: Using Random Forests for clustering and outlier detection
