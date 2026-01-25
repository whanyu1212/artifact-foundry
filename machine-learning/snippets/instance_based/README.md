# Instance-Based Learning - From Scratch Implementations

This folder contains educational implementations of instance-based (lazy) learning algorithms that make predictions by comparing new instances to stored training examples.

## Contents

### K-Nearest Neighbors (KNN)

1. **[knn.py](knn.py)** - K-Nearest Neighbors Classifier and Regressor
   - Classification: Majority voting among K nearest neighbors
   - Regression: Average of K nearest neighbors
   - Multiple distance metrics: Euclidean, Manhattan, Minkowski, Cosine
   - Weighted voting option (inverse distance weighting)
   - Training: $O(1)$ (just store data), Prediction: $O(nd)$

## Mathematical Overview

### K-Nearest Neighbors Algorithm

**Classification**:
```
ŷ = argmax_y Σᵢ∈Nₖ(x) I(yᵢ = y)
```

Where:
- $N_k(x)$ = set of K nearest neighbors to test point x
- $I(\cdot)$ = indicator function (1 if true, 0 otherwise)

**Regression**:
```
ŷ = (1/K) Σᵢ∈Nₖ(x) yᵢ
```

**Weighted Voting** (classification):
```
ŷ = argmax_y Σᵢ∈Nₖ(x) wᵢ · I(yᵢ = y)
```

Where $w_i = 1 / d(x, x_i)$ for inverse distance weighting.

### Distance Metrics

**Euclidean Distance** (L2 norm):
```
d(x, x') = √(Σᵢ (xᵢ - x'ᵢ)²)
```

**Manhattan Distance** (L1 norm):
```
d(x, x') = Σᵢ |xᵢ - x'ᵢ|
```

**Minkowski Distance** (generalized Lp norm):
```
d(x, x') = (Σᵢ |xᵢ - x'ᵢ|ᵖ)^(1/p)
```
- $p = 1$: Manhattan
- $p = 2$: Euclidean
- $p → ∞$: Chebyshev (maximum difference)

**Cosine Distance** (angle-based):
```
d(x, x') = 1 - (x · x') / (||x|| ||x'||)
```

## Quick Comparison

### Distance Metrics

| Metric | Best For | Sensitive to Scale | Computational Cost |
|--------|----------|-------------------|-------------------|
| **Euclidean** | General purpose, continuous features | Yes | Medium |
| **Manhattan** | Grid-like paths, high dimensions | Yes | Low |
| **Minkowski** | Tunable between L1 and L2 | Yes | Medium-High |
| **Cosine** | Text, sparse data, direction matters | No | Medium |

**Rule of thumb**:
- Use Euclidean for general continuous data (default)
- Use Manhattan for high-dimensional data or grid-like spaces
- Use Cosine for text/sparse data or when magnitude doesn't matter

### KNN Characteristics

| Aspect | Characteristic | Implication |
|--------|---------------|-------------|
| **Training Time** | $O(1)$ | Instant training (lazy learning) |
| **Prediction Time** | $O(nd)$ | Slow for large datasets |
| **Memory** | $O(nd)$ | Stores all training data |
| **Decision Boundary** | Non-linear, flexible | Can fit complex patterns |
| **Interpretability** | High | Easy to explain individual predictions |
| **Parametric** | No | Non-parametric (no assumptions about distribution) |

## Choosing K

### Effect of K

- **Small K** (e.g., K=1, K=3):
  - Low bias, high variance
  - Sensitive to noise and outliers
  - Complex decision boundaries
  - Risk of overfitting

- **Large K** (e.g., K=50, K=100):
  - High bias, low variance
  - Robust to noise
  - Smooth decision boundaries
  - Risk of underfitting

**Optimal K**:
- Use cross-validation to select best K
- Typical range: K ∈ {1, 3, 5, 7, 9, ..., √n}
- Start with K = √n as a heuristic
- Always use odd K for binary classification (avoid ties)

## When to Use KNN

### Best For:
- **Small to medium datasets** (n < 10,000)
- **Low to medium dimensionality** (d < 20)
- **Non-linear decision boundaries**
- **No training time constraint** (lazy learning advantage)
- **Interpretability needed** (can show nearest neighbors)
- **Irregular decision boundaries** that don't fit parametric models

### Avoid When:
- **Large datasets** (n > 100,000): Too slow at prediction time
  - Use approximate nearest neighbors (KD-Tree, Ball Tree, LSH)
  - Or switch to parametric models
- **High dimensionality** (d > 50): Curse of dimensionality
  - Distance becomes meaningless (all points equally far)
  - Use dimensionality reduction (PCA, feature selection) first
- **Real-time prediction required**: Prediction is O(nd)
- **Imbalanced classes**: Majority class dominates
  - Use weighted KNN or SMOTE for balancing

## Implementation Notes

### Efficient Nearest Neighbor Search

**Brute Force** (our implementation):
- Compute distance to all training points
- Time: $O(nd)$ per prediction
- Works for any distance metric
- Best for small datasets or low dimensions

**KD-Tree** (axis-aligned splits):
- Preprocessing: $O(n \log n)$
- Query: $O(\log n)$ average, $O(n)$ worst case
- Only works for Euclidean and Manhattan
- Breaks down in high dimensions (d > 20)

**Ball Tree** (hypersphere hierarchy):
- Preprocessing: $O(n \log n)$
- Query: $O(\log n)$ average
- Works with any distance metric
- Better for high dimensions than KD-Tree

**Locality Sensitive Hashing (LSH)**:
- Approximate nearest neighbors
- Sublinear query time
- Good for very high dimensions

### Weighted Voting

**Inverse Distance Weighting**:
```python
w_i = 1 / d(x, x_i)  # if d > 0
w_i = 1.0            # if d = 0 (exact match)
```

**Advantages**:
- Closer neighbors have more influence
- More robust to choice of K
- Better boundary estimation

**When to use**:
- When proximity matters more than just "being in top K"
- When K is large (reduces impact of distant neighbors)

### Feature Scaling

**CRITICAL**: KNN is sensitive to feature scales!

Always standardize features before using KNN:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why**: Features with larger scales dominate distance calculations.

Example:
- Feature 1: Age (0-100)
- Feature 2: Income (0-1,000,000)
- Without scaling, income dominates distance!

## Example Usage

### KNN Classification

```python
from knn import KNNClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and scale data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
knn = KNNClassifier(n_neighbors=5, metric='euclidean', weights='distance')
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)  # class probabilities

# Accuracy
print(f"Accuracy: {knn.score(X_test, y_test):.4f}")
```

### KNN Regression

```python
from knn import KNNRegressor
from sklearn.datasets import load_diabetes

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
knn = KNNRegressor(n_neighbors=5, weights='uniform')
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# R² score
print(f"R² Score: {knn.score(X_test, y_test):.4f}")
```

### Choosing K with Cross-Validation

```python
from sklearn.model_selection import cross_val_score
import numpy as np

# Test different K values
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_scores = []

for k in k_values:
    knn = KNNClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

# Best K
best_k = k_values[np.argmax(cv_scores)]
print(f"Best K: {best_k}")
```

## Curse of Dimensionality

**Problem**: In high dimensions, all points become roughly equidistant.

**Why it happens**:
- Volume of hypersphere concentrates in outer shell
- Ratio of nearest to farthest neighbor distance → 1 as d → ∞
- Need exponentially more data to maintain same density

**Solutions**:
1. **Dimensionality reduction**: PCA, feature selection
2. **Distance weighting**: Emphasize features with high variance
3. **Use approximate methods**: LSH for high-dimensional data
4. **Switch to parametric models**: Logistic regression, SVM with RBF kernel

**Rule of thumb**: KNN works best when d < 20

## Comparison with Other Algorithms

### KNN vs Naive Bayes

| Aspect | KNN | Naive Bayes |
|--------|-----|-------------|
| **Assumptions** | None (non-parametric) | Feature independence, distribution |
| **Training** | O(1) | O(nd) |
| **Prediction** | O(nd) | O(Kd) |
| **Decision Boundary** | Non-linear, complex | Linear or quadratic |
| **Interpretability** | Show neighbors | Show probabilities |

**Use KNN when**: No assumptions about data distribution, need complex boundaries
**Use Naive Bayes when**: Features are independent, need fast prediction, small data

### KNN vs Decision Trees

| Aspect | KNN | Decision Trees |
|--------|-----|----------------|
| **Training** | O(1) | O(nd log n) |
| **Prediction** | O(nd) | O(log n) |
| **Interpretability** | Neighbors | Rules |
| **Feature Scaling** | Required | Not required |

**Use KNN when**: Instance-based reasoning makes sense, small dataset
**Use Decision Trees when**: Need fast prediction, don't want to scale features

## Extensions

### Weighted KNN by Feature Importance
Learn feature weights to emphasize important dimensions:
```python
d(x, x') = √(Σᵢ wᵢ(xᵢ - x'ᵢ)²)
```

### Adaptive K
Choose different K for different regions of feature space based on local density.

### Locally Weighted Regression
Fit a local linear model using nearby points instead of simple averaging.

## Further Reading

See the notes in `machine-learning/notes/`:
- [knn.md](../../notes/knn.md) - Comprehensive guide to K-Nearest Neighbors

## Running Examples

The implementation has a `__main__` block with demonstrations:

```bash
python knn.py
```

Requirements:
- `numpy` - Core numerical operations
- `scikit-learn` - For datasets and comparisons (examples only)
- `rich` - For formatted terminal output
