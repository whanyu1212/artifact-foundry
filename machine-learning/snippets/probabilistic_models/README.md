# Probabilistic Models - From Scratch Implementations

This folder contains educational implementations of probabilistic classification models based on Bayes' theorem and generative modeling.

## Contents

### Naive Bayes Classifiers

1. **[gaussian_naive_bayes.py](gaussian_naive_bayes.py)** - Gaussian Naive Bayes
   - For continuous features (assumes Gaussian distribution)
   - Independence assumption: $P(\mathbf{x} | y) = \prod_i P(x_i | y)$
   - Fast training and prediction: $O(nd)$
   - Works well even when independence assumption is violated

### Discriminant Analysis

2. **[lda.py](lda.py)** - Linear Discriminant Analysis
   - Assumes multivariate Gaussian with shared covariance
   - Linear decision boundaries
   - Also supports dimensionality reduction (supervised PCA)
   - Projects to at most $(K-1)$ dimensions

## Mathematical Overview

### Naive Bayes

**Bayes' Theorem**:
```
P(y|X) = P(X|y)P(y) / P(X)
```

**Naive Independence Assumption**:
```
P(X|y) = ∏ᵢ P(xᵢ|y)
```

**Classification**:
```
ŷ = argmax_y [log P(y) + Σᵢ log P(xᵢ|y)]
```

**Gaussian Naive Bayes**:
- Each feature follows Gaussian: $P(x_i | y) \sim \mathcal{N}(\mu_{iy}, \sigma^2_{iy})$
- Estimate mean and variance for each feature-class pair
- Fast: $O(nd)$ training, $O(Kd)$ prediction

### Linear Discriminant Analysis (LDA)

**Multivariate Gaussian**:
```
P(X|y) ~ N(μ_y, Σ)
```

**Shared Covariance**: All classes have same $\Sigma$

**Discriminant Function** (linear in $\mathbf{x}$):
```
δ_y(x) = x^T Σ^(-1) μ_y - ½ μ_y^T Σ^(-1) μ_y + log P(y)
```

**Decision Boundary**: Hyperplane where $\delta_i(x) = \delta_j(x)$

**Projection**: Maximize $J(w) = w^T S_B w / w^T S_W w$
- $S_B$ = between-class scatter
- $S_W$ = within-class scatter
- Solution: eigenvectors of $S_W^{-1} S_B$

## Quick Comparison

### Naive Bayes vs LDA

| Aspect | Naive Bayes | LDA |
|--------|-------------|-----|
| **Covariance Structure** | Diagonal (independence) | Full covariance matrix |
| **Decision Boundary** | Quadratic (even with shared variance) | Linear |
| **Parameters** | $2Kd$ (means + variances) | $Kd + \frac{d(d+1)}{2}$ |
| **Assumptions** | Feature independence | Gaussian with shared Σ |
| **Training Complexity** | $O(nd)$ | $O(nd^2 + d^3)$ |
| **Data Required** | Less | More |

**Rule of thumb**:
- Use Naive Bayes when features are independent or $d$ is very large
- Use LDA when features are correlated and you have enough data

### Generative vs Discriminative

| Model Type | Examples | What They Model | Pros | Cons |
|------------|----------|-----------------|------|------|
| **Generative** | Naive Bayes, LDA, QDA | $P(X\|y)$ and $P(y)$ | - Works with small data<br>- Can generate samples<br>- Handles missing features | - Stronger assumptions<br>- Less accurate with large data |
| **Discriminative** | Logistic Regression, SVM | $P(y\|X)$ directly | - More accurate with large data<br>- Fewer assumptions | - Needs more data<br>- Cannot generate samples |

## Naive Bayes Variants (Not Yet Implemented)

### Multinomial Naive Bayes
For discrete count data (e.g., word counts in text):
```python
P(xᵢ|y) = pᵢy  # probability of feature i in class y
pᵢy = (count(xᵢ, y) + α) / (total_count(y) + αd)  # with Laplace smoothing
```

**Primary use**: Text classification

### Bernoulli Naive Bayes
For binary features (presence/absence):
```python
P(xᵢ|y) = pᵢy^xᵢ (1 - pᵢy)^(1-xᵢ)
```

**Use case**: Short text classification with word presence features

## When to Use What?

### Gaussian Naive Bayes

**✓ Use when:**
- Features are continuous and approximately Gaussian
- Features are (approximately) independent
- Small to medium datasets
- Need very fast training and prediction
- High-dimensional data
- Baseline model

**✗ Avoid when:**
- Features are discrete counts (use Multinomial NB)
- Features are binary (use Bernoulli NB)
- Features are highly correlated (use LDA or Logistic Regression)
- Need best accuracy with large data (use discriminative models)

**Examples**: Sensor data, medical measurements, iris classification

### Linear Discriminant Analysis (LDA)

**✓ Use when:**
- Features are continuous and approximately Gaussian
- Classes have similar covariance structures
- Need linear decision boundaries
- Want dimensionality reduction with class information
- Interpretability matters (can visualize class distributions)
- Small to medium datasets

**✗ Avoid when:**
- Non-linear decision boundaries needed (use QDA or kernel methods)
- Classes have very different covariance structures (use QDA)
- Features are non-Gaussian (use logistic regression)
- Very high dimensional data ($d > n$) (use Naive Bayes or regularization)

**Examples**: Biometrics, pattern recognition, preprocessing for classification

## Implementation Notes

### Numerical Stability

All implementations use **log probabilities** to avoid underflow:
```python
log P(y|X) = log P(y) + Σ log P(xᵢ|y)
```

**Softmax with log-sum-exp trick** for probabilities:
```python
log_probs_max = max(log_probs)
probs = exp(log_probs - log_probs_max)
probs = probs / sum(probs)
```

### Variance Smoothing (Gaussian NB)

Add small constant to variances to prevent division by zero:
```python
var = estimated_var + var_smoothing  # var_smoothing ~ 1e-9
```

### Covariance Inversion (LDA)

Use **pseudo-inverse** for numerical stability:
```python
cov_inv = np.linalg.pinv(covariance)  # handles near-singular matrices
```

## Example Usage

### Gaussian Naive Bayes

```python
from gaussian_naive_bayes import GaussianNaiveBayes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
gnb = GaussianNaiveBayes()
gnb.fit(X_train, y_train)

# Predictions
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)  # class probabilities

# Accuracy
print(f"Accuracy: {gnb.score(X_test, y_test):.4f}")
```

### Linear Discriminant Analysis

```python
from lda import LinearDiscriminantAnalysis

# Classification
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)

# Dimensionality reduction (project to 2D)
lda_2d = LinearDiscriminantAnalysis(n_components=2)
X_projected = lda_2d.fit_transform(X_train, y_train)

# Can visualize X_projected in 2D scatter plot
```

## Probability Calibration

Naive Bayes often produces **poorly calibrated probabilities** (too extreme):
- Reason: Independence assumption is violated
- Effect: Overconfident predictions (probabilities close to 0 or 1)
- **But**: Class rankings are still correct (classification still works)

**If calibrated probabilities are needed**:
- Use Platt scaling (fit logistic regression on NB outputs)
- Use isotonic regression
- Use beta calibration

## Extensions

### Regularized Discriminant Analysis (RDA)

For high-dimensional or small-sample settings, regularize covariance:
```python
Σ_k(λ, γ) = (1-γ)[(1-λ)Σ_k + λΣ] + γσ²I
```
- $\lambda \in [0,1]$: interpolate between QDA ($\lambda=0$) and LDA ($\lambda=1$)
- $\gamma \in [0,1]$: shrink toward diagonal (Ridge-like)

### Quadratic Discriminant Analysis (QDA)

Allow class-specific covariances → quadratic decision boundaries:
```python
# Each class has own covariance Σ_k
δ_y(x) = -½ log|Σ_y| - ½(x - μ_y)^T Σ_y^(-1) (x - μ_y) + log P(y)
```

**Trade-off**: More flexible but requires more data ($K$ times more parameters).

## Further Reading

See the notes in `machine-learning/notes/`:
- [naive-bayes.md](../../notes/naive-bayes.md) - Comprehensive guide to Naive Bayes variants
- [discriminant-analysis.md](../../notes/discriminant-analysis.md) - LDA, QDA, and comparisons

## Running Examples

Each implementation has a `__main__` block with demonstrations:

```bash
python gaussian_naive_bayes.py
python lda.py
```

Requirements:
- `numpy` - Core numerical operations
- `scikit-learn` - For datasets and comparisons (examples only)
- `matplotlib` - For visualizations (LDA projection)
