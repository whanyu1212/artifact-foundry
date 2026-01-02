# Linear and Quadratic Discriminant Analysis

## Overview

**Discriminant Analysis** is a family of generative classification algorithms that model the distribution of features for each class as a **multivariate Gaussian**. Like Naive Bayes, these methods use Bayes' theorem but make different assumptions about the data.

**Two Main Variants:**
1. **Linear Discriminant Analysis (LDA)**: Assumes all classes share the same covariance matrix → **linear** decision boundaries
2. **Quadratic Discriminant Analysis (QDA)**: Allows different covariance matrices per class → **quadratic** decision boundaries

**Key Characteristics:**
- **Generative models**: Model $P(\mathbf{x} | y)$ using multivariate Gaussian distributions
- **Probabilistic**: Provide class probabilities and posterior distributions
- **Closed-form solution**: No iterative optimization required
- **Dimensionality reduction**: LDA can also be used to project data to lower dimensions

---

## Generative Classification Framework

### Bayes' Theorem for Classification

Given features $\mathbf{x}$, the posterior probability of class $k$ is:

$$
P(y = k | \mathbf{x}) = \frac{P(\mathbf{x} | y = k) P(y = k)}{P(\mathbf{x})} = \frac{P(\mathbf{x} | y = k) P(y = k)}{\sum_{j=1}^{K} P(\mathbf{x} | y = j) P(y = j)}
$$

**Classification rule**:

$$
\hat{y} = \arg\max_{k} P(y = k | \mathbf{x}) = \arg\max_{k} P(\mathbf{x} | y = k) P(y = k)
$$

### Multivariate Gaussian Assumption

Both LDA and QDA assume features follow a **multivariate Gaussian distribution** for each class:

$$
P(\mathbf{x} | y = k) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}_k|^{1/2}} \exp\left( -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) \right)
$$

Where:
- $\boldsymbol{\mu}_k \in \mathbb{R}^d$ = mean vector for class $k$
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{d \times d}$ = covariance matrix for class $k$
- $d$ = number of features

**Difference between LDA and QDA**:
- **LDA**: All classes share the same covariance matrix ($\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$ for all $k$)
- **QDA**: Each class has its own covariance matrix ($\boldsymbol{\Sigma}_k$ can differ)

---

## Linear Discriminant Analysis (LDA)

### Assumptions

1. **Multivariate Gaussian**: Features $\mathbf{x} | y = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma})$
2. **Shared covariance**: All classes have the same covariance matrix $\boldsymbol{\Sigma}$
3. **Different means**: Each class has its own mean vector $\boldsymbol{\mu}_k$

### Discriminant Function

Taking the log of the posterior (for numerical stability):

$$
\log P(y = k | \mathbf{x}) = \log P(\mathbf{x} | y = k) + \log P(y = k) - \log P(\mathbf{x})
$$

Substitute multivariate Gaussian with shared covariance:

$$
\log P(\mathbf{x} | y = k) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)
$$

Expand the quadratic term:

$$
-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) = -\frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x} + \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2}\boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k
$$

**Key insight**: The term $-\frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}^{-1} \mathbf{x}$ is **independent of class** $k$ (constant for all classes).

Drop constant terms (don't affect argmax):

$$
\delta_k(\mathbf{x}) = \mathbf{x}^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k - \frac{1}{2}\boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log P(y = k)
$$

This is the **LDA discriminant function** (linear in $\mathbf{x}$).

**Classification rule**:

$$
\hat{y} = \arg\max_{k} \delta_k(\mathbf{x})
$$

### Linear Decision Boundary

The discriminant function $\delta_k(\mathbf{x})$ is **linear in $\mathbf{x}$**:

$$
\delta_k(\mathbf{x}) = \mathbf{w}_k^T \mathbf{x} + w_{k0}
$$

Where:
- $\mathbf{w}_k = \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k$ (weight vector)
- $w_{k0} = -\frac{1}{2}\boldsymbol{\mu}_k^T \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_k + \log P(y = k)$ (bias)

**Decision boundary** between classes $i$ and $j$:

$$
\delta_i(\mathbf{x}) = \delta_j(\mathbf{x}) \quad \Rightarrow \quad \mathbf{w}^T \mathbf{x} + w_0 = 0
$$

This is a **hyperplane** in $\mathbb{R}^d$ (linear decision boundary).

### Parameter Estimation (Training)

Estimate parameters using **Maximum Likelihood Estimation** from training data:

**Class priors**:

$$
\hat{P}(y = k) = \frac{n_k}{n}
$$

Where $n_k$ = number of examples in class $k$, $n$ = total examples.

**Class means**:

$$
\hat{\boldsymbol{\mu}}_k = \frac{1}{n_k} \sum_{i: y_i = k} \mathbf{x}_i
$$

**Pooled covariance matrix** (shared across all classes):

$$
\hat{\boldsymbol{\Sigma}} = \frac{1}{n - K} \sum_{k=1}^{K} \sum_{i: y_i = k} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)^T
$$

Where $K$ = number of classes.

**Complexity**: $O(nd^2 + d^3)$ for training ($d^3$ for inverting covariance matrix).

---

## Quadratic Discriminant Analysis (QDA)

### Assumptions

1. **Multivariate Gaussian**: Features $\mathbf{x} | y = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$
2. **Class-specific covariance**: Each class $k$ has its own covariance matrix $\boldsymbol{\Sigma}_k$
3. **Different means**: Each class has its own mean vector $\boldsymbol{\mu}_k$

### Discriminant Function

Unlike LDA, the quadratic term $\mathbf{x}^T \boldsymbol{\Sigma}_k^{-1} \mathbf{x}$ **depends on class** $k$ (cannot be dropped).

$$
\delta_k(\mathbf{x}) = -\frac{1}{2}\log|\boldsymbol{\Sigma}_k| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) + \log P(y = k)
$$

Expanding:

$$
\delta_k(\mathbf{x}) = -\frac{1}{2}\mathbf{x}^T \boldsymbol{\Sigma}_k^{-1} \mathbf{x} + \mathbf{x}^T \boldsymbol{\Sigma}_k^{-1} \boldsymbol{\mu}_k - \frac{1}{2}\boldsymbol{\mu}_k^T \boldsymbol{\Sigma}_k^{-1} \boldsymbol{\mu}_k - \frac{1}{2}\log|\boldsymbol{\Sigma}_k| + \log P(y = k)
$$

This is **quadratic in $\mathbf{x}$** (due to $\mathbf{x}^T \boldsymbol{\Sigma}_k^{-1} \mathbf{x}$ term).

### Quadratic Decision Boundary

Decision boundary between classes $i$ and $j$:

$$
\delta_i(\mathbf{x}) = \delta_j(\mathbf{x})
$$

Results in a **quadratic equation** in $\mathbf{x}$ (ellipse, parabola, or hyperbola).

**More flexible** than LDA but requires more parameters.

### Parameter Estimation

**Class-specific covariance matrices**:

$$
\hat{\boldsymbol{\Sigma}}_k = \frac{1}{n_k - 1} \sum_{i: y_i = k} (\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_k)^T
$$

Each class has its own covariance estimate (unlike LDA's pooled covariance).

**Number of parameters**:
- **LDA**: $Kd + \frac{d(d+1)}{2}$ (shared covariance)
- **QDA**: $Kd + K\frac{d(d+1)}{2}$ (class-specific covariances)

QDA requires **more data** to reliably estimate $K$ covariance matrices.

---

## LDA vs QDA: When to Use Which?

| Aspect | LDA | QDA |
|--------|-----|-----|
| **Decision Boundary** | Linear (hyperplane) | Quadratic (curved) |
| **Covariance Assumption** | Shared across classes | Class-specific |
| **Parameters** | $Kd + \frac{d(d+1)}{2}$ | $Kd + K\frac{d(d+1)}{2}$ |
| **Bias-Variance Tradeoff** | Higher bias, lower variance | Lower bias, higher variance |
| **Sample Size Required** | Smaller | Larger |
| **Flexibility** | Less flexible | More flexible |
| **Risk of Overfitting** | Lower | Higher (especially for large $d$, small $n$) |

### Decision Guide

**Use LDA when**:
- Classes have similar covariance structures
- Small to medium training dataset
- Need simpler, more interpretable model
- Want to reduce dimensionality (LDA provides projection)
- High-dimensional data (LDA more stable)

**Use QDA when**:
- Classes have clearly different covariance structures (shapes)
- Large training dataset
- Decision boundary is clearly non-linear
- Willing to accept more complex model for better accuracy

**Rule of thumb**:
- If $n < d^2$: Use LDA (QDA may overfit)
- If covariances look similar: Use LDA (simpler is better)
- If QDA doesn't significantly improve over LDA: Stick with LDA (Occam's razor)

---

## Comparison with Other Classifiers

### LDA vs Logistic Regression

Both produce **linear decision boundaries**, but differ in approach:

| Aspect | LDA (Generative) | Logistic Regression (Discriminative) |
|--------|------------------|--------------------------------------|
| **Models** | $P(\mathbf{x} \| y)$ and $P(y)$ | $P(y \| \mathbf{x})$ directly |
| **Assumptions** | Gaussian features, shared covariance | Linear log-odds, no distribution assumption |
| **Parameters** | $Kd + \frac{d(d+1)}{2}$ | $d$ (or $d+1$ with bias) |
| **Training data needed** | Less (stronger assumptions) | More (fewer assumptions) |
| **Accuracy (large $n$)** | Lower | Higher |
| **Robustness** | Sensitive to Gaussian assumption | More robust to non-Gaussian data |

**When LDA wins**:
- Data is approximately Gaussian
- Small training dataset
- Classes are well-separated

**When Logistic Regression wins**:
- Large training dataset
- Features are non-Gaussian
- Need best predictive accuracy

### LDA vs Naive Bayes

Both are generative models but differ in covariance assumptions:

| Aspect | LDA | Naive Bayes (Gaussian) |
|--------|-----|------------------------|
| **Covariance** | Full covariance matrix $\boldsymbol{\Sigma}$ | Diagonal covariance (independence) |
| **Decision Boundary** | Linear | Quadratic (even with shared variance) |
| **Parameters** | $Kd + \frac{d(d+1)}{2}$ | $2Kd$ (means + variances) |
| **Flexibility** | Models feature correlations | Assumes feature independence |
| **Sample size needed** | More | Less |

**Connection**: Naive Bayes is a special case where covariance matrix is **diagonal** (features are independent).

---

## LDA for Dimensionality Reduction

Beyond classification, LDA can be used to **project data to lower dimensions** while preserving class separability.

### The Idea

Find directions (linear combinations of features) that:
1. **Maximize between-class variance**: Separate classes as much as possible
2. **Minimize within-class variance**: Keep classes compact

**Result**: Project $d$-dimensional data to $K-1$ dimensions (or fewer).

### Projection Formulation

**Between-class scatter matrix** $\mathbf{S}_B$:

$$
\mathbf{S}_B = \sum_{k=1}^{K} n_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T
$$

Where $\boldsymbol{\mu} = \frac{1}{n}\sum_i \mathbf{x}_i$ (overall mean).

**Within-class scatter matrix** $\mathbf{S}_W$:

$$
\mathbf{S}_W = \sum_{k=1}^{K} \sum_{i: y_i = k} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T
$$

**Objective**: Find projection direction $\mathbf{w}$ that maximizes:

$$
J(\mathbf{w}) = \frac{\mathbf{w}^T \mathbf{S}_B \mathbf{w}}{\mathbf{w}^T \mathbf{S}_W \mathbf{w}}
$$

**Solution**: Eigenvectors of $\mathbf{S}_W^{-1} \mathbf{S}_B$ (generalized eigenvalue problem).

**Top $m$ eigenvectors** give the best $m$-dimensional projection.

### LDA vs PCA for Dimensionality Reduction

| Aspect | LDA | PCA |
|--------|-----|-----|
| **Goal** | Maximize class separability | Maximize variance |
| **Supervised** | Yes (uses labels) | No (ignores labels) |
| **Max dimensions** | $K - 1$ | $d$ |
| **Use case** | Visualization, feature extraction for classification | General dimensionality reduction |

LDA finds directions that best separate classes; PCA finds directions of maximum variance (may not separate classes).

---

## Assumptions and Diagnostics

### Key Assumptions

1. **Multivariate Gaussian distribution**: Each class follows $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$
2. **Shared covariance (LDA only)**: All classes have same $\boldsymbol{\Sigma}$
3. **No perfect multicollinearity**: Features are not perfectly correlated

### Checking Assumptions

**Normality**:
- Univariate: Shapiro-Wilk test, Q-Q plots for each feature
- Multivariate: Mardia's test, chi-square Q-Q plot

**Homogeneity of covariance (LDA)**:
- Box's M test (sensitive to normality violations)
- Visually compare covariance matrices across classes

**Multicollinearity**:
- Compute correlation matrix
- Check condition number of $\boldsymbol{\Sigma}$ (should be < 30)

### Robustness

**LDA/QDA are sensitive to**:
- Outliers (Gaussian assumption)
- Non-Gaussian distributions
- Small sample sizes relative to dimensionality

**Solutions**:
- Robust variants: Use robust covariance estimators
- Transform features: Box-Cox, log transform to make more Gaussian
- Regularization: Shrink covariance estimates toward identity (Regularized Discriminant Analysis)

---

## Regularized Discriminant Analysis (RDA)

### Motivation

**Problem**: When $d$ is large or $n_k$ is small, covariance estimates are unreliable.

**Solution**: Regularize covariance matrix by shrinking toward simpler structure.

### RDA Formulation

Interpolate between LDA and QDA covariance estimates:

$$
\hat{\boldsymbol{\Sigma}}_k(\lambda) = (1 - \lambda) \hat{\boldsymbol{\Sigma}}_k + \lambda \hat{\boldsymbol{\Sigma}}
$$

Where:
- $\lambda \in [0, 1]$ controls regularization
- $\lambda = 0$: QDA (class-specific covariances)
- $\lambda = 1$: LDA (pooled covariance)

Additionally, shrink toward diagonal (Ridge-like regularization):

$$
\hat{\boldsymbol{\Sigma}}_k(\lambda, \gamma) = (1 - \gamma) \hat{\boldsymbol{\Sigma}}_k(\lambda) + \gamma \sigma^2 \mathbf{I}
$$

Where $\gamma \in [0, 1]$ and $\sigma^2 = \text{tr}(\hat{\boldsymbol{\Sigma}}_k) / d$.

**Hyperparameters** $\lambda$ and $\gamma$ tuned via cross-validation.

---

## Computational Complexity

### Training

**LDA**:
- Compute means: $O(nd)$
- Compute pooled covariance: $O(nd^2)$
- Invert covariance: $O(d^3)$
- **Total**: $O(nd^2 + d^3)$

**QDA**:
- Compute $K$ class-specific covariances: $O(nd^2)$
- Invert $K$ covariances: $O(Kd^3)$
- **Total**: $O(nd^2 + Kd^3)$

### Prediction

**LDA**: $O(Kd^2)$ (compute $\delta_k(\mathbf{x})$ for each class)
**QDA**: $O(Kd^2)$ (similar, but with class-specific $\boldsymbol{\Sigma}_k$)

### Scalability

- Both scale poorly with dimensionality $d$ (due to $d^2$ and $d^3$ terms)
- For high-dimensional data ($d > 1000$), consider:
  - Feature selection / dimensionality reduction first
  - Regularized variants (RDA)
  - Naive Bayes (assumes diagonal covariance, $O(nd)$ complexity)

---

## Practical Tips

1. **Feature scaling**: LDA/QDA are sensitive to scale (use StandardScaler)

2. **Check Gaussian assumption**: Plot histograms, Q-Q plots
   - If violated: Consider transformations or robust methods

3. **Check covariance homogeneity**: Compare covariance matrices visually
   - If similar: Use LDA (simpler)
   - If very different: Use QDA

4. **Regularization**: For high-dimensional or small-sample settings
   - Use RDA or shrinkage covariance estimators

5. **Compare with logistic regression**:
   - If LDA performs similarly: LDA is overfitting (use logistic regression)
   - If LDA much worse: Gaussian assumption violated

6. **Dimensionality reduction**: Use LDA projection for visualization
   - Project to 2D or 3D for plotting class separation

---

## Summary

**Discriminant Analysis** models class-conditional distributions as multivariate Gaussians:

- **LDA**: Shared covariance → linear decision boundaries
  - Simpler, fewer parameters, better for small datasets
  - Assumes classes have same shape (covariance), different locations (means)

- **QDA**: Class-specific covariances → quadratic decision boundaries
  - More flexible, more parameters, requires larger datasets
  - Allows classes to have different shapes and locations

**Key Takeaways**:
- Generative models: Learn $P(\mathbf{x} | y)$ using Gaussian distributions
- Closed-form solution: No iterative optimization (fast training)
- Interpretable: Can visualize class distributions and decision boundaries
- Sensitive to Gaussian assumption: Performance degrades with non-Gaussian data
- LDA also useful for dimensionality reduction (supervised alternative to PCA)

**When to use**:
- Data is approximately Gaussian
- Want probabilistic outputs
- Need dimensionality reduction with class information
- Baseline before trying more complex methods

**Next Steps**:
- Implement LDA and QDA from scratch
- Compare with Naive Bayes and Logistic Regression
- Explore regularized variants for high-dimensional data
- Use LDA for supervised dimensionality reduction
