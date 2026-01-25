# Linear Models - From Scratch Implementations

This folder contains educational implementations of linear models from scratch, covering regression and classification with regularization.

## Contents

### Regression Models

1. **[linear_regression.py](linear_regression.py)** - Linear Regression
   - Normal equations (closed-form): `O(d³)` complexity
   - Gradient descent (iterative): `O(knd)` for `k` iterations
   - Model: `ŷ = Xw`
   - Loss: MSE (Mean Squared Error)

2. **[ridge_regression.py](ridge_regression.py)** - Ridge Regression (L2)
   - Adds L2 penalty: `λ||w||²`
   - Closed-form: `w = (X^T X + nλI)^-1 X^T y`
   - Shrinks all weights, handles multicollinearity
   - Always invertible (even when `X^T X` is singular)

3. **[lasso_regression.py](lasso_regression.py)** - Lasso Regression (L1)
   - Adds L1 penalty: `λ||w||₁`
   - No closed-form (coordinate descent required)
   - Produces sparse solutions (many weights exactly zero)
   - Automatic feature selection

4. **[elastic_net.py](elastic_net.py)** - Elastic Net (L1 + L2)
   - Combines L1 and L2: `λ[α||w||₁ + (1-α)||w||²]`
   - Sparse like Lasso, stable like Ridge
   - Handles correlated features via grouping effect
   - Best for high-dimensional data with correlations

### Classification Models

5. **[logistic_regression.py](logistic_regression.py)** - Logistic Regression
   - Binary classification using sigmoid function
   - Model: `P(y=1|x) = σ(w^T x)`
   - Loss: Binary cross-entropy (log-loss)
   - Supports L2 regularization

## Mathematical Overview

### Linear Regression
```
Model:    ŷ = w^T x
Loss:     L(w) = (1/n)||y - Xw||²
Solution: w = (X^T X)^-1 X^T y  (normal equations)
          OR gradient descent
```

### Ridge Regression (L2)
```
Loss:     L(w) = (1/n)||y - Xw||² + λ||w||²
Solution: w = (X^T X + nλI)^-1 X^T y
Effect:   Shrinks all weights toward zero
```

### Lasso Regression (L1)
```
Loss:     L(w) = (1/n)||y - Xw||² + λ||w||₁
Solution: Coordinate descent (no closed form)
Effect:   Sets many weights exactly to zero (sparse)
```

### Elastic Net (L1 + L2)
```
Loss:     L(w) = (1/n)||y - Xw||² + λ[α||w||₁ + (1-α)||w||²]
Solution: Coordinate descent with L2 modification
Effect:   Sparse + grouped selection of correlated features
```

### Logistic Regression
```
Model:    P(y=1|x) = 1 / (1 + exp(-w^T x))
Loss:     L(w) = -mean[y log(p) + (1-y) log(1-p)]
Solution: Gradient descent (convex optimization)
```

## Quick Comparison

| Model | Penalty | Sparsity | Closed-Form | Use Case |
|-------|---------|----------|-------------|----------|
| **Linear Regression** | None | No | Yes (normal eqs) | Baseline, simple relationships |
| **Ridge (L2)** | `λ||w||²` | No | Yes | Correlated features, multicollinearity |
| **Lasso (L1)** | `λ||w||₁` | Yes | No | Feature selection, high-dimensional |
| **Elastic Net** | L1 + L2 | Yes | No | Correlated + high-dimensional |
| **Logistic** | Optional L2 | No | No | Binary classification |

## When to Use What?

### Linear Regression
- ✓ Baseline model for regression
- ✓ Interpretability is important
- ✓ Simple linear relationships
- ✗ High-dimensional data (use Ridge/Lasso)
- ✗ Multicollinearity (use Ridge)

### Ridge Regression
- ✓ Many features, most are relevant
- ✓ Features are correlated
- ✓ Need smooth, stable coefficients
- ✓ Fast computation (closed-form)
- ✗ Want sparse models (use Lasso)
- ✗ Want feature selection (use Lasso)

### Lasso Regression
- ✓ High-dimensional data with many irrelevant features
- ✓ Want automatic feature selection
- ✓ Interpretability is critical
- ✗ Features are highly correlated (use Elastic Net)
- ✗ Need stable solutions (use Ridge/Elastic Net)

### Elastic Net
- ✓ High-dimensional data
- ✓ Features are correlated
- ✓ Want feature selection AND stability
- ✓ Safest choice when unsure
- ✗ Need fastest computation (use Ridge)

### Logistic Regression
- ✓ Binary classification
- ✓ Need probability estimates
- ✓ Linear decision boundary
- ✓ Interpretability important
- ✗ Non-linear boundaries (use trees, kernels, neural nets)

## Key Implementation Details

### Standardization
**Critical for regularized models!** Always standardize features before Ridge/Lasso/Elastic Net:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Regularization penalties treat all features equally, so different scales lead to unfair penalization.

### Bias Term Handling
All implementations handle the intercept (bias) term correctly:
- **Centered data**: Mean-subtracted to avoid regularizing bias
- **Separate intercept**: Computed after fitting weights
- **Standard practice**: Never regularize the bias term

### Convergence
- **Normal equations**: Exact solution in one step (`O(d³)`)
- **Gradient descent**: Iterative, check convergence via loss change
- **Coordinate descent** (Lasso/Elastic Net): Efficient for sparse solutions

## Example Usage

### Linear Regression
```python
from linear_regression import LinearRegression

# Method 1: Normal equations (exact)
lr = LinearRegression(method="normal_equations")
lr.fit(X_train, y_train)

# Method 2: Gradient descent
lr = LinearRegression(method="gradient_descent", learning_rate=0.01)
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)
r2 = lr.score(X_test, y_test)
```

### Ridge Regression
```python
from ridge_regression import RidgeRegression

# Tune alpha via cross-validation
for alpha in [0.1, 1.0, 10.0]:
    ridge = RidgeRegression(alpha=alpha)
    ridge.fit(X_train, y_train)
    print(f"α={alpha}: R²={ridge.score(X_test, y_test):.4f}")
```

### Lasso Regression
```python
from lasso_regression import LassoRegression

lasso = LassoRegression(alpha=1.0, max_iterations=2000)
lasso.fit(X_train, y_train)

print(f"Selected {lasso.n_nonzero_weights_} / {X_train.shape[1]} features")
print(f"Sparsity: {(1 - lasso.n_nonzero_weights_/X_train.shape[1])*100:.1f}%")
```

### Elastic Net
```python
from elastic_net import ElasticNet

# l1_ratio controls L1/L2 mix: 1=Lasso, 0=Ridge, 0.5=balanced
enet = ElasticNet(alpha=1.0, l1_ratio=0.5)
enet.fit(X_train, y_train)
```

### Logistic Regression
```python
from logistic_regression import LogisticRegression

lr = LogisticRegression(learning_rate=0.1, alpha=0.01)  # alpha for L2 reg
lr.fit(X_train, y_train)

# Probabilities
probas = lr.predict_proba(X_test)

# Class labels
y_pred = lr.predict(X_test, threshold=0.5)
```

## Hyperparameter Tuning

### Regularization Strength (α/lambda)
Use **cross-validation** to select optimal regularization:

```python
from sklearn.model_selection import cross_val_score

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alphas:
    model = RidgeRegression(alpha=alpha)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"α={alpha}: Mean R²={scores.mean():.4f} (±{scores.std():.4f})")
```

**Typical ranges**:
- Ridge/Lasso: `[0.01, 0.1, 1, 10, 100]`
- Elastic Net `l1_ratio`: `[0.1, 0.3, 0.5, 0.7, 0.9]`

### Learning Rate (Gradient Descent)
- Too large → diverges or oscillates
- Too small → slow convergence
- Typical range: `[0.001, 0.01, 0.1]`
- Use adaptive methods (Adam) for robustness

## Educational Notes

### Why Implement from Scratch?
These implementations prioritize **understanding** over performance:
- Explicit mathematical operations (no magic)
- Extensive comments explaining concepts
- Direct translation from formulas to code
- Multiple solution methods shown (closed-form vs iterative)

### Differences from Scikit-learn
Production libraries (scikit-learn) have:
- Better numerical stability (e.g., SVD decomposition)
- Optimized algorithms (LARS for Lasso, warm starts)
- More robust convergence checks
- Extensive input validation
- Parallel processing

**Use these implementations to learn, scikit-learn for production.**

### Key Concepts Demonstrated
1. **Closed-form vs iterative solutions** (normal equations vs gradient descent)
2. **Regularization** (bias-variance tradeoff, preventing overfitting)
3. **Sparsity** (L1 penalty creates exact zeros)
4. **Convex optimization** (guarantees global minimum)
5. **Feature scaling** (why it matters for regularization)
6. **Coordinate descent** (efficient for sparse solutions)

## Further Reading

See the notes in `machine-learning/notes/`:
- [linear-regression.md](../../notes/linear-regression.md) - Linear regression fundamentals
- [regularization.md](../../notes/regularization.md) - Ridge, Lasso, Elastic Net theory
- [logistic-regression.md](../../notes/logistic-regression.md) - Classification and GLMs

## Running the Examples

Each file has a `__main__` block with demonstrations:

```bash
# Run individual examples
python linear_regression.py
python ridge_regression.py
python lasso_regression.py
python elastic_net.py
python logistic_regression.py
```

Requirements:
- `numpy` - Core numerical operations
- `scikit-learn` - For generating synthetic data and comparisons
