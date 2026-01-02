# Support Vector Machines (SVM)

## Overview

**Support Vector Machines** are powerful supervised learning algorithms for classification and regression that find the optimal hyperplane maximizing the margin between classes. SVMs are particularly effective for high-dimensional data and can handle non-linear decision boundaries using the kernel trick.

**Key Characteristics:**
- **Maximum margin classifier**: Finds decision boundary with largest separation between classes
- **Support vectors**: Only boundary points matter (sparse solution)
- **Kernel trick**: Implicitly maps to high-dimensional space for non-linear boundaries
- **Convex optimization**: Global optimum guaranteed (quadratic programming)
- **Robust to outliers**: Hinge loss and regularization provide robustness

**Common Applications:**
- Text classification (spam detection, sentiment analysis)
- Image recognition (face detection, object classification)
- Bioinformatics (protein classification, gene expression)
- Handwriting recognition (digit classification)

---

## Linear SVM

### The Margin Concept

For linearly separable data, many hyperplanes can separate the classes. SVM chooses the one with **maximum margin**.

**Hyperplane equation**:
$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

where:
- $\mathbf{w}$ = normal vector (perpendicular to hyperplane)
- $b$ = bias (intercept)
- $\mathbf{x}$ = feature vector

**Decision function**:
$$
f(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b)
$$

**Margin**: Distance from hyperplane to nearest point of either class.

**Intuition**: Maximize margin → better generalization (more confident predictions).

### Hard Margin SVM (Linearly Separable)

For perfectly linearly separable data with labels $y_i \in \{-1, +1\}$:

**Constraints**: All points correctly classified with margin:
$$
y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i
$$

**Margin width**: $\frac{2}{\|\mathbf{w}\|}$

**Optimization problem** (maximize margin = minimize $\|\mathbf{w}\|$):
$$
\begin{align}
\min_{\mathbf{w}, b} \quad & \frac{1}{2}\|\mathbf{w}\|^2 \\
\text{s.t.} \quad & y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n
\end{align}
$$

**Why $\frac{1}{2}\|\mathbf{w}\|^2$?**: Makes derivative simple (quadratic program).

**Solution**: Use Lagrange multipliers → Quadratic Programming (QP).

### Soft Margin SVM (Non-Separable Data)

Real data is rarely perfectly separable. Allow some misclassifications using **slack variables** $\xi_i \geq 0$.

**Relaxed constraints**:
$$
y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i
$$

where:
- $\xi_i = 0$: Point correctly classified with margin
- $0 < \xi_i < 1$: Point inside margin but correctly classified
- $\xi_i \geq 1$: Point misclassified

**Optimization** (balance margin vs. violations):
$$
\begin{align}
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad & \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i \\
\text{s.t.} \quad & y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i \\
& \xi_i \geq 0, \quad i = 1, \ldots, n
\end{align}
$$

**Hyperparameter $C$** (regularization):
- **Large $C$**: Fewer violations allowed (risk overfitting, small margin)
- **Small $C$**: More violations allowed (larger margin, risk underfitting)
- **$C = \infty$**: Hard margin SVM

---

## Dual Formulation and Support Vectors

### The Dual Problem

Using Lagrange multipliers $\alpha_i \geq 0$, the **dual formulation** is:

$$
\begin{align}
\max_{\boldsymbol{\alpha}} \quad & \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \\
\text{s.t.} \quad & 0 \leq \alpha_i \leq C, \quad i = 1, \ldots, n \\
& \sum_{i=1}^{n} \alpha_i y_i = 0
\end{align}
$$

**Key insight**: Optimization depends only on **dot products** $\mathbf{x}_i^T \mathbf{x}_j$ (enables kernel trick).

### Support Vectors

**Optimal weights**:
$$
\mathbf{w} = \sum_{i=1}^{n} \alpha_i y_i \mathbf{x}_i
$$

**Support vectors**: Training points with $\alpha_i > 0$
- Lie on or inside the margin
- Determine the decision boundary
- Typically small fraction of training data (sparse solution)

**Decision function**:
$$
f(\mathbf{x}) = \text{sign}\left( \sum_{i \in SV} \alpha_i y_i \mathbf{x}_i^T \mathbf{x} + b \right)
$$

**Sparsity**: Only support vectors contribute to prediction (efficient).

---

## The Kernel Trick

### Non-Linear Decision Boundaries

For non-linearly separable data, map to higher-dimensional space where data becomes linearly separable.

**Feature map**: $\phi: \mathbb{R}^d \to \mathbb{R}^D$ (where $D \gg d$)

**Example** (polynomial features for $d=2$):
$$
\phi(\mathbf{x}) = [1, x_1, x_2, x_1^2, x_2^2, x_1 x_2]^T
$$

**Problem**: Explicit mapping to high $D$ is expensive.

### The Kernel Function

**Kernel trick**: Compute dot product in high-dimensional space **without** explicit mapping.

**Kernel function**:
$$
K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T \phi(\mathbf{x}')
$$

**Dual formulation with kernels**:
Replace $\mathbf{x}_i^T \mathbf{x}_j$ with $K(\mathbf{x}_i, \mathbf{x}_j)$:

$$
\max_{\boldsymbol{\alpha}} \quad \sum_{i=1}^{n} \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)
$$

**Decision function**:
$$
f(\mathbf{x}) = \text{sign}\left( \sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)
$$

**Power**: Implicitly work in infinite-dimensional space (e.g., RBF kernel).

### Common Kernel Functions

**1. Linear Kernel** (no transformation):
$$
K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T \mathbf{x}'
$$

**2. Polynomial Kernel**:
$$
K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T \mathbf{x}' + c)^d
$$
- $d$ = degree (hyperparameter)
- $c$ = constant term (usually 0 or 1)

**3. Radial Basis Function (RBF) / Gaussian Kernel**:
$$
K(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \|\mathbf{x} - \mathbf{x}'\|^2 \right)
$$
- $\gamma = \frac{1}{2\sigma^2}$ controls width
- Most popular kernel (very flexible)
- Corresponds to infinite-dimensional feature space

**4. Sigmoid Kernel**:
$$
K(\mathbf{x}, \mathbf{x}') = \tanh(\alpha \mathbf{x}^T \mathbf{x}' + c)
$$
- Mimics neural network
- Not always positive semi-definite (may not be valid kernel)

**Kernel Matrix** (Gram matrix):
$$
\mathbf{K}_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)
$$

**Valid kernel**: Must be **positive semi-definite** (all eigenvalues $\geq 0$).

---

## Hyperparameters

### Regularization Parameter $C$

Controls trade-off between margin size and training error.

- **Large $C$** (e.g., $C = 100$):
  - Hard margin (few violations)
  - Risk overfitting
  - Sensitive to outliers

- **Small $C$** (e.g., $C = 0.01$):
  - Soft margin (many violations allowed)
  - Larger margin
  - More robust, risk underfitting

**Tuning**: Use cross-validation to select optimal $C$.

### RBF Kernel Parameter $\gamma$

Controls influence of single training example.

- **Large $\gamma$** (e.g., $\gamma = 10$):
  - Narrow RBF (Gaussian)
  - Each example has small influence radius
  - High variance, low bias (risk overfitting)
  - Complex decision boundary

- **Small $\gamma$** (e.g., $\gamma = 0.01$):
  - Wide RBF
  - Each example influences large region
  - Low variance, high bias (risk underfitting)
  - Smooth decision boundary

**Tuning**: Grid search over $C$ and $\gamma$ using cross-validation.

**Typical grid**:
- $C \in \{0.01, 0.1, 1, 10, 100\}$
- $\gamma \in \{0.001, 0.01, 0.1, 1, 10\}$

---

## Multi-Class SVM

SVM is inherently binary. Two strategies for multi-class:

### One-vs-Rest (OvR)

Train $K$ binary classifiers (one per class):
- Class $k$: positive examples
- All other classes: negative examples

**Prediction**: Choose class with highest decision function value.

**Drawback**: Imbalanced classes (1 vs. $K-1$).

### One-vs-One (OvO)

Train $\frac{K(K-1)}{2}$ classifiers (one for each pair of classes).

**Prediction**: Each classifier votes; choose class with most votes.

**Advantage**: Each classifier sees balanced data.

**Drawback**: Many classifiers for large $K$.

**Scikit-learn default**: One-vs-One for multi-class SVM.

---

## SVM for Regression (SVR)

SVM can be adapted for regression: **Support Vector Regression (SVR)**.

### $\epsilon$-Insensitive Loss

Ignore errors smaller than $\epsilon$ (tube around predictions).

**Loss**:
$$
L_\epsilon(y, \hat{y}) = \begin{cases}
0 & \text{if } |y - \hat{y}| \leq \epsilon \\
|y - \hat{y}| - \epsilon & \text{otherwise}
\end{cases}
$$

**Optimization**:
$$
\begin{align}
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \quad & \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*) \\
\text{s.t.} \quad & y_i - (\mathbf{w}^T \mathbf{x}_i + b) \leq \epsilon + \xi_i \\
& (\mathbf{w}^T \mathbf{x}_i + b) - y_i \leq \epsilon + \xi_i^* \\
& \xi_i, \xi_i^* \geq 0
\end{align}
$$

**Support vectors**: Points outside the $\epsilon$-tube.

**Hyperparameters**:
- $C$: Regularization
- $\epsilon$: Width of tube (insensitivity)
- Kernel parameters (e.g., $\gamma$ for RBF)

---

## Computational Complexity

### Training

**Optimization**: Quadratic Programming (QP)
- **Time**: $O(n^2 d)$ to $O(n^3)$ (depends on solver)
- $n$ = number of training examples
- $d$ = number of features

**Scaling**:
- Slow for large datasets ($n > 10{,}000$)
- Use approximations (e.g., Linear SVM, SGD-based methods)

**Sequential Minimal Optimization (SMO)**:
- Efficient algorithm for solving SVM QP
- Breaks problem into 2-variable sub-problems
- Used in LIBSVM library

### Prediction

**Time**: $O(n_{SV} \cdot d)$
- $n_{SV}$ = number of support vectors
- Typically $n_{SV} \ll n$ (sparse solution)

**Faster than KNN** if $n_{SV}$ is small.

---

## Advantages of SVM

1. **Effective in high dimensions**: Works well even when $d > n$
2. **Memory efficient**: Only support vectors stored (sparse solution)
3. **Versatile**: Different kernels for various data types
4. **Robust to overfitting**: Regularization parameter $C$ controls complexity
5. **Global optimum**: Convex optimization (no local minima)
6. **Works with small datasets**: Effective with limited training data
7. **Margin maximization**: Theoretically motivated (statistical learning theory)

---

## Disadvantages of SVM

1. **Slow training for large $n$**: Quadratic to cubic complexity
2. **Memory intensive**: Kernel matrix is $O(n^2)$ (for non-linear kernels)
3. **Choice of kernel**: Requires domain knowledge or extensive tuning
4. **Hyperparameter tuning**: Must tune $C$, kernel parameters (grid search expensive)
5. **No probability estimates**: Decision function gives signed distance, not probability
   - Can calibrate using Platt scaling (extra step)
6. **Sensitive to feature scaling**: Like KNN, requires normalization
7. **Less interpretable**: Kernel SVM is a "black box"

---

## Comparison with Other Classifiers

| Aspect | SVM | Logistic Regression | Decision Trees | KNN |
|--------|-----|---------------------|----------------|-----|
| **Training Time** | $O(n^2)$ - $O(n^3)$ | $O(nd)$ (gradient descent) | $O(nd \log n)$ | $O(1)$ |
| **Prediction Time** | $O(n_{SV} d)$ | $O(d)$ | $O(\log n)$ | $O(nd)$ |
| **High Dimensions** | Excellent | Good | Poor | Poor (curse) |
| **Small Datasets** | Good | Good | Poor (overfits) | Good |
| **Interpretability** | Low (kernel) | High (coefficients) | High (rules) | Low (instance-based) |
| **Probabilistic** | No (needs calibration) | Yes | Yes | Yes (vote fraction) |
| **Kernel Trick** | Yes | No | No | Distance metrics |

---

## When to Use SVM

### Best For:
- **High-dimensional data** ($d$ large, even $d > n$)
- **Clear margin of separation** between classes
- **Small to medium datasets** ($n < 10{,}000$)
- **Non-linear boundaries** (with RBF or polynomial kernels)
- **Text classification** (high-dimensional, sparse features)
- **Image recognition** (with appropriate kernels)

### Avoid When:
- **Very large datasets** ($n > 100{,}000$): Too slow
  - Use Linear SVM with SGD or simpler models (Logistic Regression)
- **Noisy data with overlapping classes**: Probabilistic models may be better
- **Interpretability crucial**: Use Logistic Regression or Decision Trees
- **Probability estimates needed**: Use calibrated Logistic Regression

---

## Practical Tips

1. **Always scale features**: Use StandardScaler (critical for SVM)
2. **Start with Linear SVM**: Try linear kernel first (fast, simple)
3. **Then try RBF kernel**: Most versatile for non-linear data
4. **Grid search for $C$ and $\gamma$**: Use cross-validation
   - $C \in \{0.1, 1, 10, 100\}$
   - $\gamma \in \{0.001, 0.01, 0.1, 1\}$ (for RBF)
5. **Check class balance**: Use `class_weight='balanced'` for imbalanced data
6. **For large $n$**: Use Linear SVM with SGD (scikit-learn: `LinearSVC` or `SGDClassifier`)
7. **Probability calibration**: Use `CalibratedClassifierCV` if probabilities needed

---

## Summary

**Support Vector Machines** find the optimal hyperplane that maximizes the margin between classes:

- **Maximum margin**: Better generalization than arbitrary separating hyperplane
- **Support vectors**: Only boundary points determine decision boundary (sparse)
- **Soft margin**: Slack variables allow misclassifications (controlled by $C$)
- **Kernel trick**: Implicitly map to high-dimensional space for non-linear boundaries
- **RBF kernel**: Most popular (Gaussian, very flexible)
- **Effective for high dimensions**: Works well when $d > n$

**Key Takeaways**:
- Powerful for classification with clear margin
- Slow training for large datasets (use Linear SVM with SGD)
- Requires careful hyperparameter tuning ($C$, kernel parameters)
- Must scale features (distance-based in feature space)
- Excellent for text and image classification

**Next Steps**:
- Implement linear SVM from scratch (dual formulation with QP solver)
- Explore kernel functions and their effect on decision boundaries
- Learn about kernel design for specific domains (string kernels, graph kernels)
- Study online/scalable SVM variants (Pegasos, LASVM)
