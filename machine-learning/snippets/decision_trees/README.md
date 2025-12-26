# Decision Trees - CART Implementation

Educational implementation of Classification and Regression Trees from scratch.

## 📁 Files

```
tree_node.py                   # Node data structure
tree_metrics.py                # Splitting criteria (Gini, Entropy, MSE, MAE)
base_tree.py                   # Core CART algorithm
decision_tree_classifier.py    # Classification tree
decision_tree_regressor.py     # Regression tree
examples_decision_tree.py      # Usage examples
```

## 🏗️ Architecture

```
                    ┌─────────────────────────┐
                    │      TreeMetrics        │
                    │  (Gini, Entropy, MSE)   │
                    └───────────┬─────────────┘
                                │
                            uses│
                                │
                    ┌───────────▼─────────────┐
                    │   BaseDecisionTree      │
                    │   (CART Algorithm)      │
                    │  • _grow_tree()         │
                    │  • _find_best_split()   │
                    │  • predict()            │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
          ┌─────────▼─────────┐   ┌────────▼────────┐
          │  DecisionTree     │   │  DecisionTree   │
          │   Classifier      │   │   Regressor     │
          │                   │   │                 │
          │ • Gini/Entropy    │   │ • MSE/MAE       │
          │ • Predict mode    │   │ • Predict mean  │
          └─────────┬─────────┘   └────────┬────────┘
                    │                      │
                    └──────────┬───────────┘
                               │
                        builds │
                               │
                         ┌─────▼─────┐
                         │    Node   │
                         └───────────┘
```

## 🚀 Quick Start

### Classification

```python
from snippets.decision_trees import DecisionTreeClassifier
import numpy as np

X_train = np.array([[2, 3], [1, 1], [6, 6], [7, 5]])
y_train = np.array([0, 0, 1, 1])

clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
clf.fit(X_train, y_train)

predictions = clf.predict(np.array([[1, 2], [6, 5]]))
print(predictions)  # [0, 1]
```

### Regression

```python
from snippets.decision_trees import DecisionTreeRegressor
import numpy as np

X_train = np.array([[1], [2], [3], [8], [9], [10]])
y_train = np.array([5.0, 5.1, 4.9, 10.2, 9.8, 10.1])

reg = DecisionTreeRegressor(criterion='mse', max_depth=5)
reg.fit(X_train, y_train)

predictions = reg.predict(np.array([[2], [9]]))
print(predictions)  # [5.0, 10.03]
```

## 🔍 How It Works

**CART Algorithm:**
1. Start with all data at root
2. Try all possible splits (each feature × each threshold)
3. Pick split with highest information gain (classification) or variance reduction (regression)
4. Recursively split left and right children
5. Stop when max_depth reached, too few samples, or no good split
6. Leaf nodes predict: most common class (classification) or mean/median (regression)

**Key Idea:** TreeMetrics calculates split quality at each step.

## 🌳 Understanding Recursive Tree Building

The tree is built using **recursion** - the `_grow_tree()` method calls itself to build left and right subtrees. Here's how it works step by step:

### Recursive Flow

```
_grow_tree(all_data, depth=0)
│
├─ Check stopping criteria (max depth, min samples, pure node)
├─ If should stop → create LEAF node, return ✅
│
├─ Find best split across all features and thresholds
├─ If no good split → create LEAF node, return ✅
│
└─ Otherwise:
   ├─ Split data into left (≤ threshold) and right (> threshold)
   ├─ left_child = _grow_tree(left_data, depth+1)   ← RECURSE
   ├─ right_child = _grow_tree(right_data, depth+1) ← RECURSE
   └─ Return INTERNAL node with children ✅
```

### Visual Example: Building a Classification Tree

Let's walk through building a tree for this toy dataset:

```python
# Dataset: predict if person will buy (0=No, 1=Yes)
# Features: [Age, Income]
X = [[25, 30k], [30, 40k], [35, 50k], [40, 60k], [45, 70k]]
y = [0,         0,         1,         1,         1        ]
```

**Step-by-step tree growth:**

```
CALL 1: _grow_tree(all 5 samples, depth=0)
┌─────────────────────────────────────────────┐
│ Root: 5 samples, [0,0,1,1,1]                │
│ Best split: Age ≤ 32 (separates well)       │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Split into:  RECURSE      RECURSE
                 LEFT         RIGHT
   Left:  [25,30] → [0,0]
   Right: [35,40,45] → [1,1,1]


CALL 2: _grow_tree(left: 2 samples [0,0], depth=1)
┌─────────────────────────────────────────────┐
│ Left child: 2 samples, all class 0          │
│ PURE NODE! → Create LEAF predicting 0       │
└─────────────────────────────────────────────┘
                    │
                    └─→ RETURN Node(value=0) ✅


CALL 3: _grow_tree(right: 3 samples [1,1,1], depth=1)
┌─────────────────────────────────────────────┐
│ Right child: 3 samples, all class 1         │
│ PURE NODE! → Create LEAF predicting 1       │
└─────────────────────────────────────────────┘
                    │
                    └─→ RETURN Node(value=1) ✅


BACK TO CALL 1:
┌─────────────────────────────────────────────┐
│ Both children returned                      │
│ Create INTERNAL node:                       │
│   - Split: Age ≤ 32                         │
│   - Left: Node(value=0)                     │
│   - Right: Node(value=1)                    │
└─────────────────────────────────────────────┘
                    │
                    └─→ RETURN Node(split, left, right) ✅
```

**Final Tree:**

```
            [Root: Age ≤ 32?]
                 /    \
                /      \
         [Yes] /        \ [No]
              /          \
          Predict 0    Predict 1
         (2 samples)  (3 samples)
```

### Code Trace Through base_tree.py

Here's what happens in the code:

```python
# Initial call
self.root = self._grow_tree(X, y, depth=0)

def _grow_tree(X, y, depth):
    # 1. Check stopping criteria
    if depth >= max_depth:           # ← Stop condition
        return Node(value=leaf_value) # ← BASE CASE (recursion stops)

    if n_samples < min_samples_split:
        return Node(value=leaf_value) # ← BASE CASE

    if all_same_class(y):            # Pure node
        return Node(value=leaf_value) # ← BASE CASE

    # 2. Find best split
    best_feature, best_threshold, gain = _find_best_split(X, y)

    if gain <= 0:                     # No good split
        return Node(value=leaf_value) # ← BASE CASE

    # 3. Split data
    left_mask = X[:, best_feature] <= best_threshold
    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[~left_mask], y[~left_mask]

    # 4. RECURSIVE CALLS - Build subtrees
    left_child = self._grow_tree(X_left, y_left, depth+1)   # ← RECURSION
    right_child = self._grow_tree(X_right, y_right, depth+1) # ← RECURSION

    # 5. Return internal node
    return Node(
        feature_idx=best_feature,
        threshold=best_threshold,
        left=left_child,      # ← Children already built!
        right=right_child     # ← Children already built!
    )
```

### Key Insights

1. **Base Cases Stop Recursion**: Leaf nodes are created when stopping criteria are met (max depth, pure node, etc.)

2. **Depth Increases Each Level**: `depth+1` ensures the tree eventually hits `max_depth` and stops

3. **Data Gets Smaller**: Each recursive call works with a subset of the parent's data (split by threshold)

4. **Bottom-Up Construction**: Recursion builds leaves first (deepest calls), then works back up to build parent nodes

5. **Each Call Returns a Node**: Either a leaf (base case) or an internal node with children (recursive case)

### Deeper Tree Example

For a tree with `max_depth=2`, the recursion goes deeper:

```
Depth 0: _grow_tree(100 samples)
         ├─ Depth 1: _grow_tree(60 samples, left)
         │           ├─ Depth 2: _grow_tree(40 samples) → LEAF ✅
         │           └─ Depth 2: _grow_tree(20 samples) → LEAF ✅
         │
         └─ Depth 1: _grow_tree(40 samples, right)
                     ├─ Depth 2: _grow_tree(25 samples) → LEAF ✅
                     └─ Depth 2: _grow_tree(15 samples) → LEAF ✅
```

The recursion naturally creates the tree structure by:
- **Going deep** (recursive calls)
- **Hitting base cases** (creating leaves)
- **Returning back up** (building internal nodes)

## ⚙️ Hyperparameters

- **criterion**: `'gini'` or `'entropy'` (classification), `'mse'` or `'mae'` (regression)
- **max_depth**: Tree depth limit (3-10 typically good, `None` = unlimited ⚠️ overfits)
- **min_samples_split**: Min samples to split a node (default: 2)
- **min_samples_leaf**: Min samples in leaf (default: 1)

## 🆚 Comparison with scikit-learn

Our educational implementation vs `sklearn.tree.DecisionTree*`:

| Feature | Our Implementation | scikit-learn |
|---------|-------------------|--------------|
| **Purpose** | Educational, learn CART | Production-ready |
| **Code clarity** | ✅ Clean, well-commented | Optimized Cython |
| **Speed** | Slow (pure Python) | ⚡ Fast (C backend) |
| **Pruning** | ❌ None | ✅ Cost-complexity pruning |
| **Missing values** | ❌ Not supported | ✅ Supported |
| **Categorical features** | ❌ Must encode manually | ✅ Built-in support |
| **Parallelization** | ❌ Single-threaded | ✅ Multi-threaded |
| **API compatibility** | Similar `fit`/`predict` | Standard sklearn API |
| **Good for** | Understanding how it works | Real applications |

**scikit-learn Documentation:**
- [DecisionTreeClassifier API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [DecisionTreeRegressor API Reference](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Decision Trees User Guide](https://scikit-learn.org/stable/modules/tree.html)

## 🧪 Run Examples

```bash
# From machine-learning/snippets/ directory
cd machine-learning/snippets
python -m decision_trees.examples_decision_tree
```

Includes: classification, regression, overfitting demo, outlier robustness.

## 📖 Learn More

- **[../../notes/decision-trees.md](../../notes/decision-trees.md)** - Complete theory guide
- **[../../notes/tree-metrics.md](../../notes/tree-metrics.md)** - Metric explanations

Topics: CART algorithm, splitting criteria, hyperparameter tuning, common pitfalls.

---

**Next:** Build Tree Ensemble Models on top of this!
