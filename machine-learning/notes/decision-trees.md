# Decision Trees - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [CART Algorithm](#cart-algorithm)
3. [Classification vs Regression](#classification-vs-regression)
4. [Implementation Architecture](#implementation-architecture)
5. [Splitting Criteria](#splitting-criteria)
6. [Hyperparameters](#hyperparameters)
7. [Practical Examples](#practical-examples)

---

## Introduction

**Decision Trees** are supervised learning models that recursively partition the feature space into regions and make predictions based on the majority class (classification) or average value (regression) in each region.

### Key Characteristics
- **Non-parametric**: No assumptions about data distribution
- **Interpretable**: Easy to visualize and understand
- **Non-linear**: Can capture complex decision boundaries
- **Greedy**: Makes locally optimal splits at each step

### Pros and Cons

**Advantages:**
- Simple to understand and interpret
- Requires little data preprocessing
- Handles both numerical and categorical data
- Can capture non-linear relationships
- Built-in feature selection

**Disadvantages:**
- Prone to overfitting (especially deep trees)
- Unstable (small data changes can change tree structure)
- Greedy algorithm (not globally optimal)
- Biased toward features with more levels

---

## CART Algorithm

**CART** = Classification And Regression Trees

### Algorithm Overview

```
function BUILD_TREE(data, depth):
    # BASE CASES (create leaf)
    if stopping_criteria_met():
        return LEAF(predicted_value)

    # RECURSIVE CASE
    best_split = find_best_split(data)

    if no_good_split_found():
        return LEAF(predicted_value)

    left_data = data where feature <= threshold
    right_data = data where feature > threshold

    left_child = BUILD_TREE(left_data, depth+1)
    right_child = BUILD_TREE(right_data, depth+1)

    return INTERNAL_NODE(feature, threshold, left_child, right_child)
```

### Stopping Criteria

The tree stops growing when ANY of these conditions are met:

1. **Max depth reached**: `depth >= max_depth`
2. **Too few samples**: `n_samples < min_samples_split`
3. **Pure node**: All targets are identical
4. **No beneficial split**: Best split has gain/reduction ≤ 0
5. **Min leaf constraint**: Split would create leaf with < `min_samples_leaf` samples

### Finding Best Split

```python
best_gain = -1
best_feature = None
best_threshold = None

for each feature:
    for each unique value as threshold:
        split data into left (<=) and right (>)

        if min_samples_leaf constraint violated:
            skip this split

        calculate split quality (information gain / variance reduction)

        if quality > best_gain:
            update best split

return best_feature, best_threshold, best_gain
```

**Time Complexity**: O(n_features × n_samples × log(n_samples))
- Dominated by sorting unique values for each feature

---

## Classification vs Regression

| Aspect | Classification | Regression |
|--------|---------------|------------|
| **Target variable** | Categorical (discrete classes) | Continuous (numerical values) |
| **Impurity measure** | Gini impurity or Entropy | Variance, MSE, or MAE |
| **Split criterion** | Information Gain | Variance Reduction |
| **Leaf prediction** | Mode (most common class) | Mean (MSE/Variance) or Median (MAE) |
| **Example** | Spam/Not Spam | House price prediction |

### Classification Metrics

#### Gini Impurity
```
Gini = 1 - Σ(p_i²)

where p_i = proportion of class i
```

- Range: [0, 1 - 1/n_classes]
- 0 = pure node (all same class)
- Maximum when all classes equally distributed
- **Faster** to compute than entropy (no logarithm)

**Example:**
```python
# Node with [0, 0, 1, 1, 1]
p_0 = 2/5 = 0.4
p_1 = 3/5 = 0.6
Gini = 1 - (0.4² + 0.6²) = 1 - (0.16 + 0.36) = 0.48
```

#### Entropy
```
Entropy = -Σ(p_i × log₂(p_i))

where p_i = proportion of class i
```

- Range: [0, log₂(n_classes)]
- 0 = pure node
- Maximum when all classes equally distributed
- Based on information theory

**Example:**
```python
# Same node: [0, 0, 1, 1, 1]
Entropy = -(0.4 × log₂(0.4) + 0.6 × log₂(0.6))
        = -(0.4 × -1.32 + 0.6 × -0.74)
        = 0.97
```

#### Information Gain
```
IG = Impurity(parent) - Weighted_Impurity(children)

where:
Weighted_Impurity = (n_left/n_total) × Impurity(left) +
                    (n_right/n_total) × Impurity(right)
```

**Maximize** information gain to find best split.

### Regression Metrics

#### Variance
```
Var(y) = (1/n) × Σ(y_i - mean(y))²
```

- Measures spread of values
- 0 = all values identical

#### Mean Squared Error (MSE)
```
MSE = (1/n) × Σ(y_i - mean(y))²
```

- **Equivalent to variance** when predicting mean
- Optimal prediction: **mean**
- Sensitive to outliers (squared term)

#### Mean Absolute Error (MAE)
```
MAE = (1/n) × Σ|y_i - median(y)|
```

- Optimal prediction: **median**
- **Robust to outliers** (no squaring)
- Slower to compute (requires sorting for median)

#### Variance Reduction
```
VR = Variance(parent) - Weighted_Variance(children)
```

**Maximize** variance reduction to find best split.

---

## Implementation Architecture

Our implementation is split into modular components:

### File Structure
```
machine-learning/snippets/
├── tree_node.py              # Node class (data structure)
├── base_tree.py              # BaseDecisionTree (shared logic)
├── decision_tree_classifier.py  # Classification implementation
├── decision_tree_regressor.py   # Regression implementation
├── tree_metrics.py           # Splitting criteria (already exists)
└── __init__.py               # Public API
```

### Class Hierarchy

```
Node
  └─ Represents individual tree nodes (leaf or internal)

BaseDecisionTree
  ├─ fit(): Main training entry point
  ├─ _grow_tree(): Recursive tree building (CART algorithm)
  ├─ _find_best_split(): Greedy split search
  ├─ predict(): Traverse tree for predictions
  └─ _traverse_tree(): Recursive prediction for single sample

DecisionTreeClassifier(BaseDecisionTree)
  ├─ _calculate_impurity(): Uses Gini or Entropy
  ├─ _calculate_leaf_value(): Returns mode (most common class)
  └─ _calculate_split_gain(): Uses information_gain()

DecisionTreeRegressor(BaseDecisionTree)
  ├─ _calculate_impurity(): Uses MSE, MAE, or Variance
  ├─ _calculate_leaf_value(): Returns mean or median
  └─ _calculate_split_gain(): Uses variance_reduction()
```

### How TreeMetrics is Used

```python
# In DecisionTreeClassifier
def _calculate_split_gain(self, y_parent, y_left, y_right):
    return TreeMetrics.information_gain(
        parent=y_parent.tolist(),
        left=y_left.tolist(),
        right=y_right.tolist(),
        mode=self.criterion  # 'gini' or 'entropy'
    )

# In DecisionTreeRegressor
def _calculate_split_gain(self, y_parent, y_left, y_right):
    return TreeMetrics.variance_reduction(
        parent=y_parent.tolist(),
        left=y_left.tolist(),
        right=y_right.tolist()
    )
```

---

## Splitting Criteria

### Why Use Information Gain?

Information gain measures how much a split reduces uncertainty:
- **High gain**: Split creates pure children (good!)
- **Low/zero gain**: Split doesn't separate classes well (bad)

### Example Split Evaluation

```python
# Before split
parent = [0, 0, 1, 1, 1, 1, 1]  # 2 class-0, 5 class-1
Gini(parent) = 1 - (2/7)² - (5/7)² = 0.408

# Split on feature X <= 3
left =  [0, 0, 1]       # 2 class-0, 1 class-1
right = [1, 1, 1, 1]    # 0 class-0, 4 class-1

Gini(left) = 1 - (2/3)² - (1/3)² = 0.444
Gini(right) = 1 - (0/4)² - (4/4)² = 0.0  # Pure!

# Weighted children impurity
weighted = (3/7) × 0.444 + (4/7) × 0.0 = 0.190

# Information gain
IG = 0.408 - 0.190 = 0.218  ✓ Good split!
```

### Gini vs Entropy

Both give similar results in practice:

```python
# Node: [0, 0, 0, 1, 1]
Gini    = 1 - (0.6² + 0.4²) = 0.48
Entropy = -(0.6×log₂(0.6) + 0.4×log₂(0.4)) = 0.97

# After normalization, they rank splits similarly
```

**When to choose:**
- **Gini**: Faster (default choice)
- **Entropy**: Slightly more balanced trees, information-theoretic interpretation

### MSE vs MAE for Regression

```python
# Data without outlier: [1, 2, 3, 4, 5]
MSE = 2.0
MAE = 1.2

# Data with outlier: [1, 2, 3, 4, 100]
MSE = 1558.0   # 779× increase!
MAE = 18.4     # 15× increase

# MAE is much more robust to outliers
```

**When to choose:**
- **MSE/Variance**: Standard choice, smooth optimization
- **MAE**: When data has outliers or you want robustness

---

## Hyperparameters

### 1. max_depth

**Controls**: Maximum depth of the tree (root is depth 0)

```python
# Shallow tree (underfit)
tree = DecisionTreeClassifier(max_depth=1)  # Decision stump

# Deep tree (overfit)
tree = DecisionTreeClassifier(max_depth=20)

# Balanced (needs tuning)
tree = DecisionTreeClassifier(max_depth=5)
```

**Effect:**
- **Low depth**: Simple model, may underfit
- **High depth**: Complex model, likely overfit
- **None**: Grow until other stopping criteria met (usually overfits)

**Typical values**: 3-10 for most problems

### 2. min_samples_split

**Controls**: Minimum samples required to attempt a split

```python
tree = DecisionTreeClassifier(min_samples_split=10)
# Won't split nodes with < 10 samples
```

**Effect:**
- **Higher values**: Fewer splits, simpler tree
- **Lower values**: More splits, more complex tree

**Default**: 2 (split as much as possible)

### 3. min_samples_leaf

**Controls**: Minimum samples required in a leaf node

```python
tree = DecisionTreeClassifier(min_samples_leaf=5)
# Each leaf must have at least 5 samples
```

**Effect:**
- **Higher values**: Smoother predictions, less overfitting
- Prevents creating leaves with very few samples

**Default**: 1 (no constraint)

### Tuning Strategy

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use cross-validation to find best combination
grid_search = GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5
)
grid_search.fit(X_train, y_train)
```

---

## Practical Examples

### Classification Example

```python
from decision_tree_classifier import DecisionTreeClassifier
import numpy as np

# Iris-like dataset
X = np.array([
    [5.1, 3.5],  # Class 0
    [4.9, 3.0],  # Class 0
    [7.0, 3.2],  # Class 1
    [6.4, 3.2],  # Class 1
    [5.9, 3.0],  # Class 1
])
y = np.array([0, 0, 1, 1, 1])

# Train classifier
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    min_samples_split=2
)
clf.fit(X, y)

# Make predictions
X_test = np.array([[5.0, 3.3], [6.5, 3.1]])
predictions = clf.predict(X_test)
print(predictions)  # [0, 1]
```

### Regression Example

```python
from decision_tree_regressor import DecisionTreeRegressor
import numpy as np

# House prices based on size
X = np.array([[1000], [1500], [2000], [2500], [3000]])
y = np.array([200000, 300000, 400000, 500000, 600000])

# Train regressor
reg = DecisionTreeRegressor(
    criterion='mse',
    max_depth=5
)
reg.fit(X, y)

# Predict
X_test = np.array([[1800], [2200]])
predictions = reg.predict(X_test)
print(predictions)  # [350000, 450000] (approximate)
```

### Visualizing the Tree

```python
def print_tree(node, depth=0):
    """Simple tree visualization"""
    indent = "  " * depth

    if node.is_leaf():
        print(f"{indent}Leaf: predict {node.value}")
    else:
        print(f"{indent}Node: X[{node.feature_idx}] <= {node.threshold:.2f}")
        print(f"{indent}├─ Left:")
        print_tree(node.left, depth + 1)
        print(f"{indent}└─ Right:")
        print_tree(node.right, depth + 1)

print_tree(clf.root)
```

Output:
```
Node: X[0] <= 6.00
├─ Left:
  Leaf: predict 0
└─ Right:
  Leaf: predict 1
```

---

## Common Pitfalls and Solutions

### 1. Overfitting

**Problem**: Tree perfectly fits training data but fails on test data

**Solutions:**
- Reduce `max_depth`
- Increase `min_samples_split` and `min_samples_leaf`
- Use ensemble methods (Random Forests, Gradient Boosting)
- Prune the tree after training

### 2. Imbalanced Classes

**Problem**: Tree biased toward majority class

**Solutions:**
- Use class weights
- Oversample minority class / undersample majority
- Use stratified splits
- Try different metrics (some are less sensitive)

### 3. High Cardinality Features

**Problem**: Features with many unique values get unfairly selected

**Solutions:**
- Bin continuous features
- Use regularization (min_samples_leaf)
- Consider other algorithms (Random Forests average over trees)

### 4. Categorical Features

**Problem**: Decision trees expect numerical input

**Solutions:**
- One-hot encoding for nominal features
- Ordinal encoding for ordinal features
- Target encoding (use with caution)

---

## Comparison with Other Algorithms

| Algorithm | Pros vs Decision Tree | Cons vs Decision Tree |
|-----------|----------------------|----------------------|
| **Linear Regression** | More stable, faster | Can't capture non-linearity |
| **Logistic Regression** | Probabilistic outputs | Linear decision boundary only |
| **KNN** | No training time | Slow prediction, memory intensive |
| **SVM** | Better generalization | Less interpretable, slower training |
| **Neural Networks** | More flexible | Black box, needs more data |
| **Random Forest** | More robust, accurate | Less interpretable, slower |

---

## Key Takeaways

1. **Decision trees partition feature space** recursively using greedy splits

2. **Classification uses information gain**, regression uses variance reduction

3. **Greedy algorithm** = locally optimal, not globally optimal

4. **Easily overfits** without proper hyperparameter tuning

5. **Interpretable** = can visualize decision logic

6. **Foundation for ensembles**: Random Forests, Gradient Boosting, XGBoost

7. **Use TreeMetrics class** to calculate:
   - `information_gain()` for classification
   - `variance_reduction()` / `mse_reduction()` for regression

8. **Hyperparameters prevent overfitting**:
   - `max_depth`: Most important
   - `min_samples_split`: Controls granularity
   - `min_samples_leaf`: Smooths predictions

---

## Further Reading

- **CART Paper**: Breiman et al. (1984) "Classification and Regression Trees"
- **Information Theory**: Shannon (1948) "A Mathematical Theory of Communication"
- **Ensemble Methods**: Random Forests, Gradient Boosting
- **Pruning**: Post-pruning with cost-complexity pruning
- **Extensions**: Extremely Randomized Trees, Isolation Forests
