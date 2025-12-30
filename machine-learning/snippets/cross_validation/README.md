# Cross-Validation Examples

Practical implementations and demonstrations of cross-validation techniques for machine learning.

## Contents

### 1. `basic_cv_examples.py`
Fundamental cross-validation demonstrations:
- **K-Fold vs Stratified K-Fold**: Shows importance of stratification for classification
- **Data Leakage Prevention**: Demonstrates correct preprocessing within CV folds
- **Time Series CV**: How to handle temporal data without future leakage
- **Leave-One-Out CV**: When and why to use LOO

**Run:**
```bash
python basic_cv_examples.py
```

**Key Concepts:**
- Always use `StratifiedKFold` for classification
- Fit preprocessing (scaling, imputation) only on training folds
- Never shuffle time series data
- LOO is for very small datasets (N < 100)

---

### 2. `optuna_cv_tuning.py`
Hyperparameter optimization using Optuna with cross-validation:
- **Basic Optuna Workflow**: Minimal example (10 trials)
- **Optuna vs Default**: Shows benefit of tuning
- **Nested CV**: Unbiased performance estimation
- **Search Strategy Comparison**: Bayesian (Optuna) vs Random
- **Ensemble Tuning**: Random Forest hyperparameter optimization

**Run:**
```bash
python optuna_cv_tuning.py
```

**Key Concepts:**
- Optuna uses Bayesian optimization (smarter than grid search)
- Minimal iterations (10-20) still provide significant improvements
- Nested CV prevents optimistic bias in hyperparameter tuning
- Optuna converges faster than random search

**Requirements:**
```bash
pip install optuna
```

---

### 3. `cv_comparison.py`
Comprehensive comparison of CV strategies:
- **Performance Comparison**: Evaluate different CV methods
- **Fold Structure Visualization**: ASCII art showing train/test splits
- **Bias-Variance Trade-off**: How K affects estimates
- **Scenario Recommendations**: Which CV to use when
- **Common Mistakes**: Pitfalls to avoid

**Run:**
```bash
python cv_comparison.py
```

**Key Concepts:**
- K=5 is the standard choice for most problems
- Higher K = lower bias but higher variance
- Choose CV method based on dataset size and problem type
- Avoid data leakage by preprocessing within folds

---

## Quick Start

Install dependencies:
```bash
pip install scikit-learn numpy rich optuna
```

Run all examples:
```bash
# Basic CV concepts
python basic_cv_examples.py

# Hyperparameter tuning with Optuna
python optuna_cv_tuning.py

# Compare different CV strategies
python cv_comparison.py
```

---

## Cross-Validation Decision Tree

```
Is your data temporal (time series)?
├─ YES → Use TimeSeriesSplit
└─ NO  → Is it classification or regression?
          ├─ Classification → Is data imbalanced?
          │                   ├─ YES → StratifiedKFold (K=5)
          │                   └─ NO  → StratifiedKFold (K=5)  # Still recommended
          └─ Regression → How large is dataset?
                          ├─ N < 1000    → K-Fold (K=10)
                          ├─ N < 100     → LeaveOneOut or K-Fold (K=10)
                          └─ N > 100,000 → K-Fold (K=3) or simple split
```

---

## Key Takeaways

### 1. Choose the Right CV Method
- **Default**: Stratified K-Fold (K=5) for classification
- **Time series**: TimeSeriesSplit (never shuffle!)
- **Small datasets**: K=10 or Leave-One-Out
- **Large datasets**: K=3 or simple train-test split

### 2. Avoid Data Leakage
```python
# WRONG: Scaling before split
scaler.fit(X)  # Sees ALL data including test!
X_scaled = scaler.transform(X)
cross_val_score(model, X_scaled, y, cv=5)

# CORRECT: Scale within each fold
for train_idx, test_idx in cv.split(X, y):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])  # Only train data
    X_test = scaler.transform(X[test_idx])        # Apply to test
```

### 3. Use Nested CV for Hyperparameter Tuning
```python
# Regular CV: Optimistically biased
grid_search.fit(X, y)  # best_score_ is OPTIMISTIC

# Nested CV: Unbiased estimate
outer_scores = cross_val_score(grid_search, X, y, cv=5)  # REALISTIC
```

### 4. Stratification Matters
- Always use `StratifiedKFold` for classification
- Especially important for imbalanced datasets
- Ensures each fold has representative class distribution

### 5. K=5 is the Sweet Spot
- Good bias-variance balance
- Computationally efficient
- Standard in ML literature
- Use K=10 for small datasets, K=3 for large datasets

---

## Hyperparameter Tuning Workflow

```python
# Step 1: Define objective function
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    }
    model = DecisionTreeClassifier(**params)
    cv = StratifiedKFold(n_splits=5)
    return cross_val_score(model, X, y, cv=cv).mean()

# Step 2: Optimize with Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Step 3: Get best parameters
best_params = study.best_params
print(f"Best CV score: {study.best_value:.3f}")

# Step 4: Train final model on ALL data
final_model = DecisionTreeClassifier(**best_params)
final_model.fit(X, y)
```

---

## Computational Cost

| Method | # Fits | Relative Cost | Use Case |
|--------|--------|---------------|----------|
| K-Fold (K=5) | 5 | 1× (baseline) | Standard choice |
| K-Fold (K=10) | 10 | 2× | Small datasets |
| LOO (N=1000) | 1000 | 200× | Very small datasets only |
| Nested CV (5×3) | 15 | 3× | Hyperparameter tuning |
| Optuna (50 trials) × CV (5) | 250 | 50× | Serious optimization |

---

## Further Reading

- **Scikit-learn CV Guide**: https://scikit-learn.org/stable/modules/cross_validation.html
- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Time Series CV**: Bergmeir & Benítez (2012) "On the Use of Cross-validation for Time Series Predictor Evaluation"
- **Nested CV**: Varma & Simon (2006) "Bias in error estimation when using cross-validation for model selection"

---

## Notes

All examples use:
- **Rich library** for formatted terminal output (tables, panels, colors)
- **Minimal iterations** for fast demonstration (increase for production)
- **Educational comments** explaining concepts, not syntax
- **Real datasets** from scikit-learn

These are learning materials - production code should:
- Use more CV folds (K=5-10)
- Run more Optuna trials (50-200)
- Add proper logging and error handling
- Consider computational budget and time constraints
