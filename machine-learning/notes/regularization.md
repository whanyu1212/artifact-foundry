# Regularization for Linear Models

## Overview

**Regularization** is a technique to prevent overfitting by adding a penalty term to the loss function that discourages complex models (large weights). Instead of just minimizing prediction error, regularized models balance **fitting the training data** with **keeping the model simple**.

**Why Regularization?**
- **Prevents overfitting**: Especially when $d \approx n$ or $d > n$ (high-dimensional data)
- **Handles multicollinearity**: When features are highly correlated
- **Improves generalization**: Better performance on unseen data
- **Feature selection**: Some methods (Lasso) automatically select important features
- **Numerical stability**: Makes $\mathbf{X}^T\mathbf{X}$ invertible even when singular

**Three Main Types**:
1. **Ridge Regression (L2 regularization)**: Shrinks all coefficients, good for correlated features
2. **Lasso Regression (L1 regularization)**: Sets some coefficients to zero, performs feature selection
3. **Elastic Net**: Combines L1 and L2, balances their strengths

---

## The Regularization Framework

### Standard Linear Regression Loss

$$
\mathcal{L}_{\text{OLS}}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2
$$

**Problem**: Minimizing only prediction error can lead to very large weights, especially with:
- Many features relative to data points ($d$ close to $n$)
- Correlated features (multicollinearity)
- Noisy data

**Solution**: Add a penalty for model complexity.

### Regularized Loss (General Form)

$$
\mathcal{L}_{\text{reg}}(\mathbf{w}) = \underbrace{\frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2}_{\text{Data fitting term}} + \underbrace{\lambda \cdot R(\mathbf{w})}_{\text{Regularization penalty}}
$$

Where:
- $R(\mathbf{w})$ = regularization function (measures model complexity)
- $\lambda \geq 0$ = regularization strength (hyperparameter)
  - $\lambda = 0$: No regularization (standard linear regression)
  - $\lambda \to \infty$: Weights forced to zero (underfitting)

**The Bias-Variance Tradeoff**:
- Small $\lambda$: Low bias, high variance (may overfit)
- Large $\lambda$: High bias, low variance (may underfit)
- Optimal $\lambda$: Minimizes total error via cross-validation

---

## Ridge Regression (L2 Regularization)

### Mathematical Formulation

**Loss function**:

$$
\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_2^2
$$

Where the L2 penalty is:

$$
\|\mathbf{w}\|_2^2 = \sum_{j=1}^{d} w_j^2 = \mathbf{w}^T \mathbf{w}
$$

**Matrix form**:

$$
\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \mathbf{w}^T \mathbf{w}
$$

**Note**: Typically we **do not penalize the bias term** $w_0$. In practice, center the data (subtract mean) to eliminate the bias term, or exclude $w_0$ from the penalty.

### Closed-Form Solution

Ridge regression has an analytical solution (like OLS).

**Derivation**:

$$
\nabla_{\mathbf{w}} \mathcal{L}_{\text{Ridge}} = -\frac{2}{n} \mathbf{X}^T (\mathbf{y} - \mathbf{X}\mathbf{w}) + 2\lambda \mathbf{w} = 0
$$

Solving for $\mathbf{w}$:

$$
\mathbf{X}^T \mathbf{X} \mathbf{w} + n\lambda \mathbf{w} = \mathbf{X}^T \mathbf{y}
$$

$$
\boxed{\mathbf{w}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + n\lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}}
$$

Where $\mathbf{I}$ is the $(d+1) \times (d+1)$ identity matrix.

**Key Insight**: Adding $n\lambda \mathbf{I}$ ensures $\mathbf{X}^T\mathbf{X} + n\lambda \mathbf{I}$ is always **invertible**, even when $\mathbf{X}^T\mathbf{X}$ is singular (e.g., when $d > n$).

### How Ridge Works

**Geometric Interpretation**:
- OLS finds weights that minimize squared error
- Ridge adds a spherical constraint: $\|\mathbf{w}\|_2^2 \leq t$ for some budget $t$
- Solution is where error contours touch the sphere

**Effect on Weights**:
- **Shrinks** all coefficients toward zero (but never exactly zero)
- Larger $\lambda$ → stronger shrinkage
- Coefficients of correlated features are shrunk together (shared responsibility)

**Eigenvalue Perspective**:
Let $\mathbf{X}^T\mathbf{X} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T$ (eigendecomposition).

OLS weights:
$$
\mathbf{w}_{\text{OLS}} = \mathbf{V}\mathbf{\Lambda}^{-1}\mathbf{V}^T\mathbf{X}^T\mathbf{y}
$$

Ridge weights:
$$
\mathbf{w}_{\text{Ridge}} = \mathbf{V}(\mathbf{\Lambda} + n\lambda \mathbf{I})^{-1}\mathbf{V}^T\mathbf{X}^T\mathbf{y}
$$

Small eigenvalues (near-collinear features) are boosted by $n\lambda$, preventing instability.

### Advantages
- **Handles multicollinearity**: Stabilizes solution when features are correlated
- **Always has a solution**: Even when $d > n$ or $\mathbf{X}^T\mathbf{X}$ is singular
- **Closed-form solution**: Fast to compute
- **Smooth optimization**: Differentiable everywhere (gradient descent works well)

### Disadvantages
- **No feature selection**: All features remain (weights shrink but never zero)
- **Less interpretable**: All features contribute, hard to identify important ones
- **Requires tuning $\lambda$**: Must use cross-validation

---

## Lasso Regression (L1 Regularization)

### Mathematical Formulation

**Loss function**:

$$
\mathcal{L}_{\text{Lasso}}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_1
$$

Where the L1 penalty is:

$$
\|\mathbf{w}\|_1 = \sum_{j=1}^{d} |w_j|
$$

**Key Difference from Ridge**: Uses absolute values instead of squares.

### No Closed-Form Solution

**Problem**: The L1 penalty is **not differentiable** at $w_j = 0$.

**Gradient** (for $w_j \neq 0$):

$$
\frac{\partial \mathcal{L}_{\text{Lasso}}}{\partial w_j} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i) x_{ij} + \lambda \cdot \text{sign}(w_j)
$$

Where $\text{sign}(w_j) = +1$ if $w_j > 0$, $-1$ if $w_j < 0$.

**Solution Methods**:
1. **Coordinate Descent**: Optimize one weight at a time (holding others fixed)
2. **Proximal Gradient Descent**: Use subgradient methods
3. **LARS (Least Angle Regression)**: Efficient path algorithm
4. **Iterative Soft Thresholding**: Apply soft-thresholding operator

Most libraries (scikit-learn) use **coordinate descent** for efficiency.

### How Lasso Works

**Geometric Interpretation**:
- L1 constraint is a diamond (in 2D) or hypercube (in higher dimensions)
- Error contours often touch the diamond at **corners** (where some $w_j = 0$)
- This causes **sparse solutions** (many weights exactly zero)

**Effect on Weights**:
- **Shrinks** coefficients toward zero
- **Sets many coefficients exactly to zero** (feature selection)
- Only the most important features remain
- Larger $\lambda$ → more coefficients set to zero

**Why L1 Leads to Sparsity**:
The L1 penalty has "sharp corners" along the axes. When error contours intersect these corners, some weights become exactly zero. L2 penalty (sphere) has no corners, so coefficients only shrink.

### Advantages
- **Automatic feature selection**: Sets irrelevant feature weights to zero
- **Interpretability**: Sparse models are easier to understand (fewer features)
- **Handles high-dimensional data**: Works even when $d > n$
- **Useful for datasets with many irrelevant features**

### Disadvantages
- **No closed-form solution**: Requires iterative optimization (slower than Ridge)
- **Unstable with correlated features**: Randomly picks one from a correlated group
- **Biased estimates**: Shrinks large coefficients too much
- **Non-differentiable**: Gradient descent requires special handling

---

## Elastic Net (L1 + L2 Regularization)

### Mathematical Formulation

**Loss function**:

$$
\mathcal{L}_{\text{ElasticNet}}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \left( \alpha \|\mathbf{w}\|_1 + \frac{1-\alpha}{2} \|\mathbf{w}\|_2^2 \right)
$$

Where:
- $\lambda \geq 0$ = overall regularization strength
- $\alpha \in [0, 1]$ = mixing parameter
  - $\alpha = 1$: Pure Lasso (L1 only)
  - $\alpha = 0$: Pure Ridge (L2 only)
  - $0 < \alpha < 1$: Combination of both

**Alternative Parameterization** (used in scikit-learn):

$$
\mathcal{L}_{\text{ElasticNet}}(\mathbf{w}) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \alpha \lambda \|\mathbf{w}\|_1 + \frac{1-\alpha}{2} \lambda \|\mathbf{w}\|_2^2
$$

### Why Elastic Net?

Elastic Net was designed to address limitations of both Ridge and Lasso:

**Ridge limitations**:
- No feature selection (all weights remain)
- Less interpretable

**Lasso limitations**:
- Unstable when features are correlated (picks one arbitrarily)
- When $d > n$, Lasso selects at most $n$ features (even if more are relevant)
- Tends to select only one feature from a group of correlated features

**Elastic Net benefits**:
- **Feature selection** like Lasso (L1 term)
- **Grouping effect** like Ridge (L2 term): Correlated features tend to have similar weights
- **Stable** with correlated features
- **Better prediction** in high-dimensional settings

### How Elastic Net Works

**Effect on Weights**:
- L1 term: Encourages sparsity (sets some weights to zero)
- L2 term: Shrinks all weights, groups correlated features

**Grouping Effect**: If features $x_j$ and $x_k$ are highly correlated, Elastic Net tends to assign them similar weights (both non-zero or both zero). Lasso would pick one randomly.

### Solution Method

Like Lasso, Elastic Net has no closed-form solution. Use:
- **Coordinate Descent** (most common)
- **Proximal Gradient Methods**

### Advantages
- **Combines best of Ridge and Lasso**
- **Feature selection** (sparsity from L1)
- **Handles correlated features** better than Lasso (grouping from L2)
- **Works well when $d > n$**
- **More stable** than Lasso

### Disadvantages
- **Two hyperparameters** to tune: $\lambda$ and $\alpha$ (more expensive cross-validation)
- **Slower** than Ridge (no closed-form solution)
- **More complex** to interpret than pure Ridge or Lasso

---

## Choosing Regularization Strength ($\lambda$)

**Goal**: Find $\lambda$ that minimizes prediction error on unseen data.

### Cross-Validation

**Standard approach**:
1. Define a grid of $\lambda$ values: `[0.001, 0.01, 0.1, 1, 10, 100]`
2. For each $\lambda$:
   - Perform K-Fold cross-validation
   - Compute average validation error
3. Select $\lambda$ with lowest validation error
4. Retrain on full training set with optimal $\lambda$

**Regularization Path**:
Many libraries (scikit-learn's `LassoCV`, `RidgeCV`) efficiently compute solutions for a range of $\lambda$ values simultaneously.

### Validation Curve

Plot training and validation error vs. $\lambda$:
- **Low $\lambda$**: Training error low, validation error high (overfitting)
- **High $\lambda$**: Both errors high (underfitting)
- **Optimal $\lambda$**: Minimizes validation error (sweet spot)

---

## Comparison: Ridge vs. Lasso vs. Elastic Net

| Aspect | Ridge (L2) | Lasso (L1) | Elastic Net (L1 + L2) |
|--------|------------|------------|------------------------|
| **Penalty** | $\sum w_j^2$ | $\sum \|w_j\|$ | $\alpha \sum \|w_j\| + (1-\alpha) \sum w_j^2$ |
| **Sparsity** | No (all weights remain) | Yes (many weights = 0) | Yes (some weights = 0) |
| **Feature Selection** | No | Yes | Yes |
| **Closed-Form** | Yes | No | No |
| **Correlated Features** | Keeps all, shrinks together | Picks one randomly | Groups them (similar weights) |
| **High-dimensional** ($d > n$) | Works but no selection | Selects $\leq n$ features | Works well, more stable |
| **Interpretability** | Low (all features) | High (sparse) | Medium (sparse but more features than Lasso) |
| **Speed** | Fastest (closed-form) | Slower (iterative) | Slower (iterative) |
| **Hyperparameters** | $\lambda$ | $\lambda$ | $\lambda$ and $\alpha$ |

**When to Use**:

- **Ridge**:
  - Many features, most are relevant
  - Features are correlated (multicollinearity)
  - Want smooth, stable coefficients
  - Need fast computation

- **Lasso**:
  - High-dimensional data with many irrelevant features
  - Want automatic feature selection
  - Interpretability is important
  - Features are not highly correlated

- **Elastic Net**:
  - High-dimensional data with correlated features
  - Want feature selection but more stable than Lasso
  - Correlated features are expected to be jointly relevant
  - Willing to tune two hyperparameters

**Rule of Thumb**:
1. Start with Ridge as baseline
2. Try Lasso if you suspect many irrelevant features
3. Use Elastic Net if Lasso is unstable or when features are correlated

---

## Standardization and Regularization

**Important**: Always **standardize features** before applying regularization.

### Why?

Regularization penalties treat all features equally (same $\lambda$ for all $w_j$). If features are on different scales:
- Large-scale features (e.g., area in sq ft: 1000-5000)
- Small-scale features (e.g., number of rooms: 1-10)

The penalty will unfairly shrink weights of large-scale features more.

### Standardization Formula

$$
x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

Where:
- $\mu_j$ = mean of feature $j$
- $\sigma_j$ = standard deviation of feature $j$

After standardization, all features have mean 0 and variance 1.

**Note**: Some libraries (scikit-learn) standardize internally, but it's good practice to do it explicitly.

---

## Regularization and Bias Term

**Convention**: Do **not** regularize the bias term $w_0$.

**Why?**
- Bias term doesn't affect model complexity (just shifts predictions)
- Penalizing bias would depend on arbitrary scaling of target $y$

**Implementation**:
1. **Center the data**: Subtract mean from features and target
   - After centering, optimal bias is $w_0 = 0$, so we can ignore it during optimization
2. **Exclude $w_0$ from penalty**: Only penalize $w_1, \ldots, w_d$

---

## Bayesian Interpretation

Regularization has an elegant Bayesian interpretation.

### Ridge Regression

Equivalent to **Maximum A Posteriori (MAP) estimation** with Gaussian prior on weights:

$$
w_j \sim \mathcal{N}(0, \frac{1}{2\lambda})
$$

Minimizing Ridge loss = maximizing posterior probability with Gaussian prior.

### Lasso Regression

Equivalent to **MAP estimation** with Laplace prior on weights:

$$
p(w_j) = \frac{\lambda}{2} e^{-\lambda |w_j|}
$$

Laplace distribution has sharp peak at zero, encouraging sparsity.

**Insight**: Regularization is equivalent to having prior beliefs about weights (they should be small or zero).

---

## Practical Tips

### 1. Start with Ridge
- Ridge is fast, stable, and often works well as a baseline
- If performance is acceptable, no need for more complex methods

### 2. Use Cross-Validation for $\lambda$
- Grid search with CV is standard
- Consider logarithmic grid: `[0.001, 0.01, 0.1, 1, 10, 100, 1000]`

### 3. Feature Engineering First
- Polynomial features and interactions increase $d$
- Regularization becomes even more important

### 4. Check Coefficients
- Large coefficients indicate potential overfitting (increase $\lambda$)
- Many zero coefficients (Lasso) suggest aggressive feature selection (decrease $\lambda$)

### 5. Regularization Path
- Visualize how coefficients change with $\lambda$ (regularization path plot)
- Helps understand feature importance and stability

### 6. Don't Regularize Test Data
- Fit regularization parameters ($\lambda$, $\alpha$) on training/validation data only
- Evaluate final model on held-out test set

---

## Summary

**Regularization** prevents overfitting by penalizing model complexity:

- **Ridge (L2)**: Shrinks all coefficients, handles multicollinearity, always has solution
- **Lasso (L1)**: Produces sparse models, automatic feature selection, interpretable
- **Elastic Net**: Combines L1 and L2, handles correlated features better than Lasso

**Key Takeaways**:
- Regularization is essential for high-dimensional data ($d$ close to or greater than $n$)
- $\lambda$ controls bias-variance tradeoff (tune via cross-validation)
- Always standardize features before regularization
- Ridge is fastest (closed-form), Lasso/Elastic Net require iterative optimization
- Elastic Net is often the safest choice for real-world data with correlated features

**Next Steps**:
- Implement Ridge, Lasso, Elastic Net from scratch
- Logistic regression (extend regularization to classification)
- Generalized Linear Models (other target distributions)
