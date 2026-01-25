# Support Vector Machines - From Scratch Implementation

This folder contains educational implementations of Support Vector Machines for binary classification.

## Contents

### SVM Classifier

1. **[svm.py](svm.py)** - Linear and Kernel SVM Classifier
   - Linear SVM: Maximum margin hyperplane
   - Soft margin: Allows misclassifications (regularization parameter C)
   - Kernel trick: RBF, Polynomial, Linear kernels
   - Training: Gradient descent on primal formulation (simplified)
   - Note: Production implementations use SMO algorithm for dual formulation

## Mathematical Overview

### Linear SVM

**Hyperplane**: $\mathbf{w}^T \mathbf{x} + b = 0$

**Decision function**: $f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)$

**Margin**: Distance from hyperplane to nearest point = $\frac{1}{\|\mathbf{w}\|}$

### Hard Margin (Linearly Separable)

Maximize margin subject to correct classification:

$$
\begin{align}
\min_{\mathbf{w}, b} \quad & \frac{1}{2}\|\mathbf{w}\|^2 \\
\text{s.t.} \quad & y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad \forall i
\end{align}
$$

### Soft Margin (Non-Separable)

Allow violations using slack variables $\xi_i \geq 0$:

$$
\begin{align}
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad & \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i \\
\text{s.t.} \quad & y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i \\
& \xi_i \geq 0
\end{align}
$$

**Regularization parameter C**:
- Large C: Fewer violations, risk overfitting (hard margin)
- Small C: More violations, larger margin (soft margin)

### Hinge Loss Formulation

Equivalent unconstrained formulation:

$$
\min_{\mathbf{w}, b} \quad \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))
$$

**Hinge loss**: $L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})$

### The Kernel Trick

Map to high-dimensional space $\phi(\mathbf{x})$ implicitly:

**Kernel function**: $K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')$

**Decision function**: $f(\mathbf{x}) = \text{sign}\left(\sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)$

### Common Kernels

**Linear**:
$$K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'$$

**Polynomial** (degree d):
$$K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T \mathbf{x}' + c)^d$$

**RBF (Gaussian)**:
$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$
- $\gamma = \frac{1}{2\sigma^2}$ controls width
- Large $\gamma$: narrow RBF, complex boundary (risk overfitting)
- Small $\gamma$: wide RBF, smooth boundary (risk underfitting)

## Implementation Notes

### Primal vs Dual Formulation

**Our Implementation**: Simplified primal formulation with gradient descent
- Direct optimization of weights $\mathbf{w}$ and bias $b$
- Uses hinge loss + L2 regularization
- Simpler to understand and implement
- Less efficient than dual formulation

**Production SVM**: Dual formulation with SMO algorithm
- Optimizes Lagrange multipliers $\alpha_i$
- Identifies support vectors automatically
- More efficient for kernel methods
- Used in LIBSVM and scikit-learn

**Why simplified**: Full dual formulation with quadratic programming (QP) solver is complex for educational purposes. Our implementation demonstrates core SVM concepts.

### Gradient Descent on Hinge Loss

**Objective**:
$$
J(\mathbf{w}, b) = \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))
$$

**Subgradient**:
- If $y_i(\mathbf{w}^T \mathbf{x}_i + b) < 1$ (violation):
  - $\frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} - C y_i \mathbf{x}_i$
  - $\frac{\partial J}{\partial b} = -C y_i$

- If $y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1$ (correct):
  - $\frac{\partial J}{\partial \mathbf{w}} = \mathbf{w}$
  - $\frac{\partial J}{\partial b} = 0$

### Kernel Methods

For kernel SVM, we maintain dual representation:
- Store support vectors and their coefficients
- Prediction uses kernel evaluations with support vectors
- No explicit feature mapping needed

## Hyperparameters

### Regularization Parameter C

| C Value | Effect | Use When |
|---------|--------|----------|
| **Large** (100, 1000) | Hard margin, few violations | Data is cleanly separable, low noise |
| **Small** (0.01, 0.1) | Soft margin, larger margin | Data is noisy, overlapping classes |
| **Medium** (1.0) | Balanced | General purpose (good starting point) |

### RBF Kernel Parameter γ (gamma)

| γ Value | Effect | Use When |
|---------|--------|----------|
| **Large** (10, 100) | Narrow RBF, complex boundary | Need to fit intricate patterns |
| **Small** (0.001, 0.01) | Wide RBF, smooth boundary | Want regularization, simple patterns |
| **Auto** (1/n_features) | Data-dependent default | Starting point for tuning |

**Grid Search**: Typically tune C and γ together
- C ∈ {0.1, 1, 10, 100}
- γ ∈ {0.001, 0.01, 0.1, 1}

## When to Use SVM

### Best For:
- **High-dimensional data** (d large, even d > n)
- **Clear margin of separation** between classes
- **Small to medium datasets** (n < 10,000)
- **Non-linear boundaries** (with RBF kernel)
- **Text classification** (high-dimensional, sparse)
- **Image recognition** (with appropriate kernels)

### Avoid When:
- **Very large datasets** (n > 100,000): Too slow
  - Use Linear SVM with SGD or simpler models
- **Noisy data** with overlapping classes
  - Probabilistic models may be better
- **Need probability estimates**
  - Use calibrated Logistic Regression instead
- **Interpretability crucial**
  - Linear models or decision trees better

## Comparison with Other Classifiers

| Aspect | SVM | Logistic Regression | KNN |
|--------|-----|---------------------|-----|
| **Training Time** | O(n²) - O(n³) | O(nd) | O(1) |
| **Prediction Time** | O(n_SV · d) | O(d) | O(nd) |
| **Decision Boundary** | Linear or non-linear | Linear (or polynomial) | Non-linear |
| **High Dimensions** | Excellent | Good | Poor (curse) |
| **Interpretability** | Low (kernel) | High | Medium |
| **Probabilistic** | No (needs calibration) | Yes | Yes |
| **Feature Scaling** | Required | Recommended | Required |

## Example Usage

### Linear SVM

```python
from svm import SVM
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate data
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, random_state=42)
y = 2 * y - 1  # Convert to {-1, +1}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# CRITICAL: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train linear SVM
svm = SVM(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# Predictions
y_pred = svm.predict(X_test)
print(f"Accuracy: {svm.score(X_test, y_test):.4f}")
```

### Kernel SVM (RBF)

```python
# Train RBF kernel SVM
svm_rbf = SVM(kernel='rbf', C=1.0, gamma=0.1)
svm_rbf.fit(X_train, y_train)

y_pred_rbf = svm_rbf.predict(X_test)
print(f"RBF Accuracy: {svm_rbf.score(X_test, y_test):.4f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Grid search over C and gamma
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# Note: For grid search with sklearn, you'd need to wrap our SVM
# or use sklearn's SVC directly for production use
```

## Multi-Class SVM

SVM is inherently binary. For multi-class:

### One-vs-Rest (OvR)
Train K binary classifiers (one per class):
- Class k vs. all others
- Predict class with highest decision function value

### One-vs-One (OvO)
Train K(K-1)/2 classifiers (one per pair):
- Each classifier votes
- Predict class with most votes

**Scikit-learn default**: One-vs-One

## Practical Tips

1. **Always scale features**: Use StandardScaler (CRITICAL for SVM)
2. **Start with Linear SVM**: Try linear kernel first (fast, simple baseline)
3. **Then try RBF**: Most versatile for non-linear data
4. **Grid search for C and γ**: Use cross-validation
   - C ∈ {0.1, 1, 10, 100}
   - γ ∈ {0.001, 0.01, 0.1, 1}
5. **Check class balance**: Use class_weight='balanced' for imbalanced data
6. **For large n**: Use LinearSVC with SGD (scikit-learn) for scalability
7. **Need probabilities**: Use CalibratedClassifierCV wrapper

## Support Vectors

**Definition**: Training points where $\alpha_i > 0$
- Lie on or inside the margin
- Determine the decision boundary
- Typically small fraction of data (sparse solution)

**Our implementation**: Simplified version doesn't explicitly compute support vectors, but demonstrates the concept through margin-based training.

**Production SVM**: Automatically identifies support vectors through dual formulation.

## Limitations

1. **Slow training**: O(n²) to O(n³) for quadratic programming
2. **Memory intensive**: Kernel matrix is O(n²)
3. **No probability estimates**: Requires calibration (Platt scaling)
4. **Hyperparameter tuning**: Expensive grid search
5. **Choice of kernel**: Requires domain knowledge
6. **Sensitive to scaling**: Must normalize features

## Extensions

### Support Vector Regression (SVR)
Use ε-insensitive loss for regression:
$$L_\epsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \epsilon)$$

### One-Class SVM
Outlier detection by fitting hypersphere around data.

### Nu-SVM
Alternative formulation with ν parameter controlling support vector fraction.

## Further Reading

See the notes in `machine-learning/notes/`:
- [svm.md](../../notes/svm.md) - Comprehensive guide to Support Vector Machines

## Running Examples

The implementation has a `__main__` block with demonstrations:

```bash
python svm.py
```

Requirements:
- `numpy` - Core numerical operations
- `scikit-learn` - For datasets and comparisons (examples only)
- `rich` - For formatted terminal output
- `matplotlib` - For decision boundary visualization
