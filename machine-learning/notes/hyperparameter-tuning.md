# Hyperparameter Tuning - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Hyperparameters vs Parameters](#hyperparameters-vs-parameters)
3. [Search Strategies](#search-strategies)
4. [Grid Search](#grid-search)
5. [Random Search](#random-search)
6. [Bayesian Optimization](#bayesian-optimization)
7. [Optuna Deep Dive](#optuna-deep-dive)
8. [Pruning Strategies](#pruning-strategies)
9. [Best Practices](#best-practices)
10. [Implementation Examples](#implementation-examples)

---

## Introduction

**Hyperparameter tuning** is the process of finding optimal configuration values for machine learning algorithms that are set before training begins. These values significantly impact model performance but cannot be learned from data.

### Why Hyperparameter Tuning Matters

```python
# Same algorithm, different hyperparameters
model_default = DecisionTreeClassifier()  # Accuracy: 0.85
model_tuned = DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=10,
    min_samples_leaf=4
)  # Accuracy: 0.93

# 8% improvement just from tuning!
```

### Key Characteristics
- **Not learned from data**: Set before training starts
- **Algorithm-specific**: Each model has different hyperparameters
- **Impact performance**: Can make 5-20% difference in accuracy
- **Require search**: No closed-form solution, must try different values

---

## Hyperparameters vs Parameters

### Parameters (Learned from Data)
```python
# Linear Regression: y = mx + b
# Parameters: m (slope), b (intercept)
# Learned during training via gradient descent
```

**Characteristics:**
- Learned automatically during training
- Optimized by the algorithm itself
- Examples: neural network weights, linear regression coefficients, decision tree split points

### Hyperparameters (Set Before Training)
```python
# Neural Network hyperparameters
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Hyperparameter
    learning_rate=0.001,            # Hyperparameter
    max_iter=1000                   # Hyperparameter
)
# The actual weights are parameters (learned during fit)
```

**Characteristics:**
- Must be set before training
- Control learning process and model capacity
- Require external search/tuning
- Examples: learning rate, tree depth, number of neighbors in KNN

### Common Hyperparameters by Algorithm

| Algorithm | Key Hyperparameters |
|-----------|-------------------|
| **Decision Trees** | max_depth, min_samples_split, min_samples_leaf, criterion |
| **Random Forest** | n_estimators, max_depth, max_features, min_samples_split |
| **SVM** | C (regularization), kernel, gamma, degree |
| **Neural Networks** | learning_rate, hidden_layers, activation, dropout, batch_size |
| **KNN** | n_neighbors, weights, metric, p (for Minkowski) |
| **Gradient Boosting** | learning_rate, n_estimators, max_depth, subsample |
| **Logistic Regression** | C, penalty, solver, max_iter |

---

## Search Strategies

### Overview

| Strategy | How It Works | Pros | Cons | Best For |
|----------|-------------|------|------|----------|
| **Manual** | Try values by intuition | Fast for experts | Inconsistent, misses optima | Quick experiments |
| **Grid Search** | Exhaustive over grid | Finds best in grid | Exponentially expensive | Small spaces (2-3 params) |
| **Random Search** | Random sampling | Better than grid | Wastes some trials | Medium spaces (4-6 params) |
| **Bayesian** | Learn from past trials | Most efficient | Complex, some overhead | Large spaces, expensive models |
| **Evolutionary** | Genetic algorithms | Good for complex spaces | Many iterations needed | Discrete/categorical spaces |
| **Gradient-based** | Use gradients | Very efficient | Requires differentiability | Neural networks only |

### The Curse of Dimensionality

```
Grid Search with different # of hyperparameters:

2 params × 10 values each = 100 trials
3 params × 10 values each = 1,000 trials
4 params × 10 values each = 10,000 trials
5 params × 10 values each = 100,000 trials ← Impractical!

Random Search with 100 trials:
- 2 params: 10 samples per dimension
- 5 params: 2.5 samples per dimension (on average)
- Still explores all dimensions!
```

---

## Grid Search

### Algorithm

```
Grid Search:

1. Define hyperparameter space as discrete grid:
   - max_depth: [3, 5, 7, 10]
   - min_samples_split: [2, 5, 10]

2. Generate all combinations (Cartesian product):
   - (3, 2), (3, 5), (3, 10)
   - (5, 2), (5, 5), (5, 10)
   - (7, 2), (7, 5), (7, 10)
   - (10, 2), (10, 5), (10, 10)
   Total: 4 × 3 = 12 combinations

3. Evaluate each combination with cross-validation

4. Return best configuration
```

### Implementation

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Total combinations: 4 × 3 × 3 × 2 = 72

# Create grid search
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                # 5-fold CV
    scoring='accuracy',
    n_jobs=-1,          # Parallel execution
    verbose=1
)

# This will train 72 × 5 = 360 models!
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")
```

### Pros and Cons

**Advantages:**
- ✓ **Exhaustive**: Guaranteed to find best in grid
- ✓ **Reproducible**: Same grid = same result
- ✓ **Parallel**: Easy to distribute
- ✓ **Simple**: Easy to understand and implement

**Disadvantages:**
- ✗ **Exponential cost**: O(n^k) where k = # hyperparameters
- ✗ **Discrete**: Misses values between grid points
- ✗ **Inefficient**: Wastes time on bad regions
- ✗ **Impractical**: > 4 hyperparameters becomes infeasible

### When to Use

**Use Grid Search when:**
- Few hyperparameters (≤ 3)
- Small search space
- You know good value ranges
- Computation is cheap
- Need exhaustive search

**Example:** Tuning SVM with 2-3 hyperparameters on small dataset.

---

## Random Search

### Algorithm

```
Random Search:

1. Define hyperparameter space (continuous or discrete):
   - max_depth: uniform(1, 20)
   - min_samples_split: uniform(2, 20)
   - learning_rate: log-uniform(0.001, 0.1)

2. Randomly sample N configurations from distributions

3. Evaluate each configuration with cross-validation

4. Return best configuration found
```

### Why Random > Grid?

**Key insight:** Not all hyperparameters matter equally!

```
Scenario: 2 hyperparameters, 1 important, 1 unimportant

Grid Search (9 trials):
  Important param: 3 unique values
  Unimportant param: 3 unique values
  Result: Wastes 6 trials on unimportant dimension

Random Search (9 trials):
  Important param: ~9 unique values
  Unimportant param: ~9 unique values
  Result: More coverage of important dimension!
```

**Visualization:**
```
Grid (9 trials):             Random (9 trials):

  ×  ×  ×                      ×     ×
  ×  ×  ×                   ×      ×
  ×  ×  ×                        ×  ×  ×  ×

Important →                   Important →
```

### Implementation

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define distributions
param_distributions = {
    'max_depth': randint(1, 21),           # Discrete: integers 1-20
    'min_samples_split': randint(2, 21),   # Discrete: integers 2-20
    'min_samples_leaf': randint(1, 11),    # Discrete: integers 1-10
    'max_features': uniform(0.1, 0.9),     # Continuous: 0.1 to 1.0
}

# Create random search
random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=100,         # Number of random samples
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# This will train 100 × 5 = 500 models
random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
```

### Pros and Cons

**Advantages:**
- ✓ **Efficient**: Better coverage per trial than grid
- ✓ **Scalable**: Linear cost with # trials
- ✓ **Continuous**: Can sample any value in range
- ✓ **Simple**: Easy to implement and parallelize

**Disadvantages:**
- ✗ **Random**: No learning from past trials
- ✗ **Wasteful**: Samples bad regions repeatedly
- ✗ **Suboptimal**: Unlikely to find true optimum
- ✗ **No guidance**: Doesn't exploit promising regions

### When to Use

**Use Random Search when:**
- Medium # of hyperparameters (3-6)
- Large search space
- Don't know which params matter most
- Grid search is too expensive
- Good baseline before Bayesian methods

---

## Bayesian Optimization

### Core Idea

**Learn from previous trials to guide future searches.**

```
Iteration 1: Try random configuration
  → Score: 0.75

Iteration 2: Model suggests promising region based on #1
  → Score: 0.82 (better!)

Iteration 3: Model exploits good region + explores uncertainty
  → Score: 0.85 (even better!)

Iteration N: Converges to optimal region
  → Score: 0.91
```

### Algorithm Overview

```
Bayesian Optimization:

1. Build probabilistic model (surrogate) of objective function
   - Input: Hyperparameters
   - Output: Performance (predicted)

2. Use model to estimate:
   - Expected performance (mean)
   - Uncertainty (variance)

3. Acquisition function balances:
   - Exploitation: Sample where performance is high
   - Exploration: Sample where uncertainty is high

4. Evaluate real objective at suggested point

5. Update model with new observation

6. Repeat steps 2-5
```

### Surrogate Models

Common choices:

1. **Gaussian Processes (GP)**
   - Smooth, continuous functions
   - Provides uncertainty estimates
   - Expensive for many trials (> 500)

2. **Tree-structured Parzen Estimator (TPE)** ⭐
   - Used by Optuna
   - Efficient for high dimensions
   - Handles discrete/categorical params
   - Scales to thousands of trials

3. **Random Forest**
   - Used by SMAC
   - Robust to noise
   - Good for discrete spaces

### Acquisition Functions

Decide where to sample next:

1. **Expected Improvement (EI)**
   ```
   EI(x) = E[max(0, f(x) - f(x_best))]
   ```
   - Expected improvement over current best
   - Balances exploration/exploitation

2. **Probability of Improvement (PI)**
   ```
   PI(x) = P(f(x) > f(x_best))
   ```
   - Probability of beating current best
   - More exploitative than EI

3. **Upper Confidence Bound (UCB)**
   ```
   UCB(x) = μ(x) + κ × σ(x)
   ```
   - Mean + confidence interval
   - κ controls exploration vs exploitation

### Exploration vs Exploitation

```
Exploration: Try new regions (high uncertainty)
  ← Might find better optimum
  ← Wastes trials if region is bad

Exploitation: Focus on known good regions
  ← Improves current best
  ← Might miss global optimum

Bayesian Optimization: Balances both!
```

**Example:**
```
Search space: [0, 10]
Current best: x=5, score=0.8

High uncertainty at x=1 (unexplored)
Low uncertainty at x=5 (well-explored)

Acquisition function suggests x=2.5:
  - Near unexplored region (exploration)
  - Close enough to good region (exploitation)
```

### Efficiency Comparison

```python
# Example: Tune 6 hyperparameters

Grid Search:
  10 values per param = 10^6 = 1,000,000 trials ← Impossible!

Random Search:
  100 trials = Limited coverage
  Best score: 0.87

Bayesian Optimization (Optuna):
  50 trials = Intelligent search
  Best score: 0.91 ← Better with fewer trials!
```

---

## Optuna Deep Dive

### What Makes Optuna Special?

**Optuna** is a modern hyperparameter optimization framework that excels through:

1. **TPE Algorithm (Tree-structured Parzen Estimator)**
2. **Automatic pruning** of unpromising trials
3. **Define-by-run** API (dynamic search spaces)
4. **Distributed optimization** support
5. **Built-in visualization** and analysis tools

### TPE Algorithm Explained

**Traditional Bayesian Optimization:**
```
Model: P(performance | hyperparameters)
"Given hyperparameters, what's the predicted performance?"
```

**TPE (Optuna's approach):**
```
Model: P(hyperparameters | performance)
"Given good/bad performance, what hyperparameters are likely?"

Split trials into:
- l(x): Distribution of hyperparameters that gave GOOD results
- g(x): Distribution of hyperparameters that gave BAD results

Choose next x that maximizes: l(x) / g(x)
```

**Why this is clever:**
- Easier to model (densities are simpler)
- Efficient for high dimensions
- Handles categorical variables naturally
- Scales better than Gaussian Processes

### TPE Visual Example

```
After 20 trials:

Bad Results (bottom 70%):         Good Results (top 30%):
max_depth: [3,4,5,6,7,...]       max_depth: [7,8,9,10]
learning_rate: [0.01,0.001,...]  learning_rate: [0.05,0.1]

TPE builds two distributions:
  g(x) = "bad" hyperparameters distribution
  l(x) = "good" hyperparameters distribution

Next trial samples from where l(x)/g(x) is high:
  → max_depth = 8-10 (common in good trials)
  → learning_rate = 0.05-0.1 (common in good trials)
```

### Optuna Architecture

```
Study
  ├─ Sampler (TPE, Random, Grid)
  │    └─ Suggests hyperparameters
  │
  ├─ Pruner (Median, Hyperband)
  │    └─ Stops unpromising trials early
  │
  ├─ Storage (SQLite, MySQL, Redis)
  │    └─ Persists trials (enables distributed optimization)
  │
  └─ Trials
       ├─ Trial 0: params={...}, value=0.75, state=COMPLETE
       ├─ Trial 1: params={...}, value=0.82, state=COMPLETE
       ├─ Trial 2: params={...}, state=PRUNED
       └─ Trial 3: params={...}, value=0.91, state=COMPLETE
```

### Basic Optuna Workflow

```python
import optuna

# 1. Define objective function
def objective(trial):
    # Suggest hyperparameters
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    # Train model
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=42
    )

    # Evaluate with cross-validation
    scores = cross_val_score(clf, X, y, cv=5)

    # Return metric to optimize
    return scores.mean()

# 2. Create study
study = optuna.create_study(
    direction='maximize',           # or 'minimize'
    sampler=optuna.samplers.TPESampler(seed=42)
)

# 3. Optimize
study.optimize(objective, n_trials=100)

# 4. Get results
print(f"Best value: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")
```

### Hyperparameter Suggestion Methods

```python
def objective(trial):
    # Integer (discrete)
    n_estimators = trial.suggest_int('n_estimators', 10, 200, step=10)

    # Float (continuous)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)

    # Categorical
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

    # Conditional hyperparameters
    if optimizer == 'sgd':
        momentum = trial.suggest_float('momentum', 0.0, 0.99)
    else:
        momentum = None

    # Discrete uniform
    dropout = trial.suggest_float('dropout', 0.0, 0.5, step=0.1)

    return train_and_evaluate(...)
```

### Optuna vs Sklearn GridSearchCV

```python
# Sklearn GridSearchCV: Static grid
param_grid = {
    'max_depth': [3, 5, 7, 10],      # Must define all values upfront
    'min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(clf, param_grid, cv=5)
grid.fit(X, y)

# Optuna: Dynamic, conditional hyperparameters
def objective(trial):
    max_depth = trial.suggest_int('max_depth', 2, 20)  # Sample from range

    if max_depth > 10:
        # Only use min_samples_leaf for deep trees
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 4, 10)
    else:
        min_samples_leaf = 1

    # Hyperparameters can depend on each other!
    min_samples_split = trial.suggest_int('min_samples_split', 2, min_samples_leaf * 3)

    return evaluate_model(max_depth, min_samples_split, min_samples_leaf)
```

---

## Pruning Strategies

### What is Pruning?

**Pruning** stops unpromising trials early to save computation.

```
Traditional approach:
Trial 1: [████████████████████] 100% → Score: 0.75 (10 min)
Trial 2: [████████████████████] 100% → Score: 0.82 (10 min)
Trial 3: [████████████████████] 100% → Score: 0.68 (10 min) ← Waste!

With pruning:
Trial 1: [████████████████████] 100% → Score: 0.75 (10 min)
Trial 2: [████████████████████] 100% → Score: 0.82 (10 min)
Trial 3: [████░░░░░░░░░░░░░░░░] 20%  → PRUNED! (2 min) ← Save 8 min!
  └─ After 20%, score is 0.50 (much worse than best)
```

### When Pruning Helps

**Best for:**
- Neural networks (evaluate after each epoch)
- Iterative algorithms (boosting, early iterations)
- Long-running models (deep learning)

**Not useful for:**
- Single-shot models (linear regression, one-pass training)
- Very fast models (< 1 second to train)

### Median Pruner

**Idea:** Prune if trial is worse than median at the same step.

```python
import optuna
from optuna.pruners import MedianPruner

# Prune if performance is below median
pruner = MedianPruner(
    n_startup_trials=5,    # Don't prune first 5 trials (need baseline)
    n_warmup_steps=3,      # Don't prune before step 3
    interval_steps=1       # Check every step
)

study = optuna.create_study(direction='maximize', pruner=pruner)

def objective(trial):
    model = RandomForestClassifier(n_estimators=100)

    # Train incrementally (add trees one by one)
    for n_trees in range(10, 101, 10):
        model.set_params(n_estimators=n_trees)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)

        # Report intermediate value
        trial.report(score, step=n_trees // 10)

        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score
```

**Example:**
```
Step 1: Trial scores [0.70, 0.72, 0.68, 0.71, 0.69]
        Median = 0.70
        New trial = 0.65 → PRUNE! (below median)

Step 2: Trial scores [0.75, 0.77, 0.73, 0.76]
        Median = 0.755
        New trial = 0.78 → CONTINUE (above median)
```

### Hyperband Pruner

**Idea:** Successive halving with multiple brackets.

```
Hyperband allocates budget adaptively:

Bracket 1: Start with 27 trials, small budget each
  └─ Keep best 9, increase budget
     └─ Keep best 3, increase budget
        └─ Keep best 1, full budget

Bracket 2: Start with 9 trials, medium budget each
  └─ Keep best 3, increase budget
     └─ Keep best 1, full budget

Bracket 3: Start with 3 trials, large budget each
  └─ Keep best 1, full budget
```

```python
from optuna.pruners import HyperbandPruner

pruner = HyperbandPruner(
    min_resource=1,      # Minimum epochs/steps
    max_resource=100,    # Maximum epochs/steps
    reduction_factor=3   # Keep top 1/3 at each stage
)

study = optuna.create_study(direction='maximize', pruner=pruner)
```

### Pruning Example: Neural Network

```python
def objective_with_pruning(trial):
    # Suggest hyperparameters
    n_layers = trial.suggest_int('n_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)

    model = build_model(n_layers, lr)

    # Train with pruning
    for epoch in range(50):
        train_loss = train_one_epoch(model)
        val_score = evaluate(model, X_val, y_val)

        # Report intermediate result
        trial.report(val_score, epoch)

        # Check if should stop early
        if trial.should_prune():
            print(f"Trial pruned at epoch {epoch}")
            raise optuna.TrialPruned()

    return val_score
```

### Pruning Benefits

```python
# Without pruning: 100 trials × 50 epochs = 5000 total epochs
# With pruning: 100 trials start, ~60 pruned early
#   - 60 trials × 10 epochs (avg before pruning) = 600 epochs
#   - 40 trials × 50 epochs (complete) = 2000 epochs
#   - Total: 2600 epochs (48% savings!)
```

---

## Best Practices

### 1. Always Use Cross-Validation

```python
# BAD: Single train-test split
def objective_bad(trial):
    params = {...}
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)  # Optimistic!

# GOOD: Cross-validation
def objective_good(trial):
    params = {...}
    cv = StratifiedKFold(n_splits=5)
    scores = cross_val_score(model, X, y, cv=cv)
    return scores.mean()  # More reliable
```

### 2. Use Nested CV for Reporting

```python
# Outer CV: True performance estimate
# Inner CV: Hyperparameter tuning

outer_cv = StratifiedKFold(n_splits=5)
outer_scores = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Inner optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)

    # Evaluate best on outer test
    best_model = build_model(**study.best_params)
    best_model.fit(X_train, y_train)
    outer_scores.append(best_model.score(X_test, y_test))

print(f"True performance: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")
```

### 3. Scale Data Properly

```python
def objective(trial):
    params = trial.suggest_...

    # CORRECT: Scale within CV
    cv = StratifiedKFold(n_splits=5)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_val_scaled = scaler.transform(X[val_idx])

        model.fit(X_train_scaled, y[train_idx])
        scores.append(model.score(X_val_scaled, y[val_idx]))

    return np.mean(scores)
```

### 4. Set Reasonable Ranges

```python
# BAD: Too wide ranges
max_depth = trial.suggest_int('max_depth', 1, 1000)  # Unlikely need > 50

# GOOD: Based on domain knowledge
max_depth = trial.suggest_int('max_depth', 2, 20)     # Reasonable for most data

# GOOD: Log scale for learning rates
lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)  # Covers orders of magnitude
```

### 5. Start Simple, Then Expand

```python
# Phase 1: Coarse search (fast)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Phase 2: Fine-tune around best region
best_depth = study.best_params['max_depth']
refined_ranges = {
    'max_depth': [best_depth - 2, best_depth + 2],  # Narrow range
    'min_samples_split': [2, 5, 10, 15, 20]
}
# Continue with refined search...
```

### 6. Monitor and Visualize

```python
import optuna.visualization as vis

# Optimization history
vis.plot_optimization_history(study)

# Parameter importances
vis.plot_param_importances(study)

# Parallel coordinate plot
vis.plot_parallel_coordinate(study)

# Contour plot (2D slice)
vis.plot_contour(study, params=['max_depth', 'min_samples_split'])
```

### 7. Use Callbacks

```python
# Early stopping if good enough
class EarlyStoppingCallback:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, study, trial):
        if study.best_value > self.threshold:
            study.stop()

study = optuna.create_study(direction='maximize')
study.optimize(
    objective,
    n_trials=1000,
    callbacks=[EarlyStoppingCallback(threshold=0.95)]
)
```

### 8. Save and Resume Studies

```python
# Create study with persistent storage
study = optuna.create_study(
    study_name='my_study',
    storage='sqlite:///optuna.db',  # Persistent database
    load_if_exists=True              # Resume if exists
)

study.optimize(objective, n_trials=50)

# Later, resume the same study
study = optuna.load_study(
    study_name='my_study',
    storage='sqlite:///optuna.db'
)

study.optimize(objective, n_trials=50)  # Continue from where we left off
```

---

## Implementation Examples

### Example 1: Simple Optuna Tuning

```python
import optuna
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_breast_cancer(return_X_y=True)

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy'])
    }

    clf = DecisionTreeClassifier(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

    return scores.mean()

# Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best accuracy: {study.best_value:.3f}")
print(f"Best params: {study.best_params}")
```

### Example 2: Multi-Objective Optimization

```python
def multi_objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    }

    clf = DecisionTreeClassifier(**params, random_state=42)

    # Objective 1: Accuracy
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    accuracy = scores.mean()

    # Objective 2: Model size (smaller is better)
    clf.fit(X, y)
    n_nodes = clf.tree_.node_count

    return accuracy, -n_nodes  # Maximize accuracy, minimize nodes

# Multi-objective study
study = optuna.create_study(directions=['maximize', 'maximize'])
study.optimize(multi_objective, n_trials=100)

# Get Pareto front
print(f"Number of Pareto-optimal solutions: {len(study.best_trials)}")
for trial in study.best_trials:
    print(f"  Accuracy: {trial.values[0]:.3f}, Nodes: {-trial.values[1]:.0f}")
```

### Example 3: Conditional Hyperparameters

```python
def objective_conditional(trial):
    classifier_name = trial.suggest_categorical('classifier', ['RF', 'SVM', 'NN'])

    if classifier_name == 'RF':
        n_estimators = trial.suggest_int('rf_n_estimators', 10, 200)
        max_depth = trial.suggest_int('rf_max_depth', 2, 20)
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    elif classifier_name == 'SVM':
        C = trial.suggest_float('svm_C', 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical('svm_kernel', ['linear', 'rbf', 'poly'])
        classifier = SVC(C=C, kernel=kernel, random_state=42)

    else:  # Neural Network
        n_layers = trial.suggest_int('nn_n_layers', 1, 3)
        n_units = trial.suggest_int('nn_n_units', 32, 256)
        layers = tuple([n_units] * n_layers)
        classifier = MLPClassifier(hidden_layer_sizes=layers, random_state=42)

    scores = cross_val_score(classifier, X, y, cv=5)
    return scores.mean()
```

### Example 4: With Pruning

```python
import optuna
from sklearn.ensemble import RandomForestClassifier

def objective_with_pruning(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    }

    # Train incrementally, report at each step
    for n_estimators in range(10, 101, 10):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            **params,
            random_state=42
        )

        scores = cross_val_score(clf, X, y, cv=3)  # Fewer folds for speed
        intermediate_value = scores.mean()

        # Report current performance
        trial.report(intermediate_value, step=n_estimators // 10)

        # Check if should prune
        if trial.should_prune():
            raise optuna.TrialPruned()

    return intermediate_value

# Use MedianPruner
study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
)

study.optimize(objective_with_pruning, n_trials=50)
print(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
```

---

## Key Takeaways

1. **Hyperparameters control learning**, parameters are learned. Both are crucial but require different optimization strategies.

2. **Grid Search is exhaustive but expensive**. Only practical for ≤ 3 hyperparameters with small ranges.

3. **Random Search beats Grid Search** because it samples more unique values for important hyperparameters.

4. **Bayesian Optimization is most efficient**. Learns from past trials to focus search on promising regions.

5. **Optuna uses TPE algorithm**:
   - Models P(hyperparameters | performance) instead of P(performance | hyperparameters)
   - More efficient than Gaussian Processes
   - Handles high dimensions and categorical variables well

6. **Pruning saves computation**:
   - Stops unpromising trials early
   - Most beneficial for iterative/neural models
   - Median Pruner: Simple, based on median performance
   - Hyperband: Sophisticated, successive halving

7. **Always use CV for evaluation**:
   - Single split is unreliable
   - Nested CV for unbiased performance estimates

8. **Start coarse, then refine**:
   - Initial wide search (20-50 trials)
   - Narrow around promising regions
   - Increase trials for final tuning

9. **Optuna advantages over Sklearn**:
   - Dynamic, conditional hyperparameters
   - Pruning support
   - Better for large search spaces
   - Distributed optimization
   - Advanced visualizations

10. **Trade-offs matter**:
    - More trials = better results but more time
    - More CV folds = more reliable but slower
    - Deeper search = risk of overfitting to validation set
    - Use nested CV to avoid optimistic bias

---

## Further Reading

- **Random Search Paper**: Bergstra & Bengio (2012) "Random Search for Hyper-Parameter Optimization"
- **Bayesian Optimization Survey**: Shahriari et al. (2016) "Taking the Human Out of the Loop"
- **TPE Algorithm**: Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization"
- **Hyperband**: Li et al. (2017) "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"
- **Optuna Paper**: Akiba et al. (2019) "Optuna: A Next-generation Hyperparameter Optimization Framework"
- **Optuna Documentation**: https://optuna.readthedocs.io/
- **AutoML Book**: Hutter et al. (2019) "Automated Machine Learning: Methods, Systems, Challenges"
