# Logistic Regression and Generalized Linear Models

## Overview

**Logistic regression** is a fundamental algorithm for **binary classification** (predicting one of two classes). Despite its name, it's a classification method, not regression. The key idea: use a linear model to predict the **log-odds** of class membership, then transform to probabilities using the **sigmoid function**.

**Key Characteristics:**
- **Probabilistic output**: Predicts class probabilities, not just labels
- **Linear decision boundary**: Separates classes with a hyperplane
- **Convex loss function**: Guarantees global optimum (no local minima)
- **Interpretable**: Coefficients show how features affect log-odds
- **Extends to multi-class**: One-vs-Rest, multinomial logistic regression (softmax)

**Logistic regression is a special case of Generalized Linear Models (GLMs)**, which extend linear regression to non-Gaussian targets using **link functions**.

---

## From Linear to Logistic Regression

### The Problem with Linear Regression for Classification

For binary classification ($y \in \{0, 1\}$), we want to predict probabilities: $P(y=1 | \mathbf{x})$.

**Naive approach**: Use linear regression to predict probabilities directly:

$$
P(y=1 | \mathbf{x}) = \mathbf{w}^T \mathbf{x}
$$

**Problem**:
- Linear function outputs values in $(-\infty, \infty)$
- Probabilities must be in $[0, 1]$
- No guarantee that $0 \leq \mathbf{w}^T \mathbf{x} \leq 1$

**Solution**: Apply a **sigmoid function** to map $\mathbb{R} \to [0, 1]$.

---

## Sigmoid Function (Logistic Function)

### Definition

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Properties**:
- Domain: $z \in (-\infty, \infty)$
- Range: $\sigma(z) \in (0, 1)$
- $\sigma(0) = 0.5$ (decision boundary)
- $\sigma(z) \to 1$ as $z \to \infty$
- $\sigma(z) \to 0$ as $z \to -\infty$
- **Symmetric**: $\sigma(-z) = 1 - \sigma(z)$

**S-shaped curve**: Smoothly transitions from 0 to 1, with steepest slope at $z=0$.

### Derivative

$$
\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))
$$

**Proof**:

$$
\frac{d}{dz} \left( \frac{1}{1 + e^{-z}} \right) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) \cdot (1 - \sigma(z))
$$

**Useful property**: Derivative depends only on $\sigma(z)$ itself, simplifying gradient computation.

---

## Logistic Regression Model

### Binary Classification

For binary classification with classes $y \in \{0, 1\}$:

$$
P(y=1 | \mathbf{x}; \mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}}}
$$

$$
P(y=0 | \mathbf{x}; \mathbf{w}) = 1 - P(y=1 | \mathbf{x}; \mathbf{w}) = \sigma(-\mathbf{w}^T \mathbf{x})
$$

Where:
- $\mathbf{w}$ = weight vector (including bias $w_0$)
- $\mathbf{x}$ = feature vector (prepend 1 for bias)
- $\mathbf{w}^T \mathbf{x}$ = **logit** (log-odds)

**Decision rule**:

$$
\hat{y} = \begin{cases}
1 & \text{if } P(y=1 | \mathbf{x}) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

Equivalently:

$$
\hat{y} = \begin{cases}
1 & \text{if } \mathbf{w}^T \mathbf{x} \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

**Decision boundary**: $\mathbf{w}^T \mathbf{x} = 0$ (a hyperplane in feature space).

---

## Log-Odds (Logit)

### Odds

**Odds** of event $A$ occurring:

$$
\text{Odds}(A) = \frac{P(A)}{1 - P(A)} = \frac{P(A)}{P(\neg A)}
$$

**Example**: If $P(y=1) = 0.8$, then odds = $0.8 / 0.2 = 4$ (4-to-1 odds in favor).

**Range**: Odds $\in [0, \infty)$

### Log-Odds (Logit)

**Log-odds** (logit):

$$
\text{logit}(p) = \log \left( \frac{p}{1-p} \right)
$$

**Range**: $\text{logit}(p) \in (-\infty, \infty)$

**For logistic regression**:

$$
\log \left( \frac{P(y=1 | \mathbf{x})}{P(y=0 | \mathbf{x})} \right) = \mathbf{w}^T \mathbf{x}
$$

**Interpretation**: Logistic regression models the **log-odds as a linear function** of features.

### Inverting the Logit

Given log-odds $z = \mathbf{w}^T \mathbf{x}$, recover probability:

$$
p = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}} = \sigma(z)
$$

**This is the sigmoid function!** Sigmoid is the inverse of the logit.

---

## Loss Function: Log-Loss (Binary Cross-Entropy)

### Maximum Likelihood Estimation (MLE)

**Goal**: Find weights $\mathbf{w}$ that maximize the likelihood of observed data.

**Likelihood** for single example $(x_i, y_i)$:

$$
P(y_i | \mathbf{x}_i; \mathbf{w}) = \begin{cases}
\sigma(\mathbf{w}^T \mathbf{x}_i) & \text{if } y_i = 1 \\
1 - \sigma(\mathbf{w}^T \mathbf{x}_i) & \text{if } y_i = 0
\end{cases}
$$

**Compact form**:

$$
P(y_i | \mathbf{x}_i; \mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x}_i)^{y_i} \cdot (1 - \sigma(\mathbf{w}^T \mathbf{x}_i))^{1 - y_i}
$$

**Likelihood for all $n$ examples** (assuming independence):

$$
L(\mathbf{w}) = \prod_{i=1}^{n} \sigma(\mathbf{w}^T \mathbf{x}_i)^{y_i} \cdot (1 - \sigma(\mathbf{w}^T \mathbf{x}_i))^{1 - y_i}
$$

**Log-likelihood**:

$$
\log L(\mathbf{w}) = \sum_{i=1}^{n} \left[ y_i \log \sigma(\mathbf{w}^T \mathbf{x}_i) + (1 - y_i) \log (1 - \sigma(\mathbf{w}^T \mathbf{x}_i)) \right]
$$

**Maximize log-likelihood** = **Minimize negative log-likelihood**.

### Binary Cross-Entropy Loss

**Loss function** (negative log-likelihood divided by $n$):

$$
\mathcal{L}(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right]
$$

Where $\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i) = P(y=1 | \mathbf{x}_i; \mathbf{w})$.

**Also called**:
- **Log-loss**
- **Binary cross-entropy**
- **Logistic loss**

**Intuition**:
- If $y_i = 1$: Loss $= -\log \hat{p}_i$ (small if $\hat{p}_i \approx 1$, large if $\hat{p}_i \approx 0$)
- If $y_i = 0$: Loss $= -\log (1 - \hat{p}_i)$ (small if $\hat{p}_i \approx 0$, large if $\hat{p}_i \approx 1$)

**Convex**: Like linear regression, logistic regression has a convex loss (guaranteed global minimum).

---

## Optimization: Gradient Descent

### No Closed-Form Solution

Unlike linear regression, logistic regression has **no analytical solution**. We must use **iterative optimization** (gradient descent or variants).

### Gradient Computation

**Gradient of log-loss**:

$$
\nabla_{\mathbf{w}} \mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{p}_i) \mathbf{x}_i
$$

Where $\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$.

**Matrix form**:

$$
\nabla_{\mathbf{w}} \mathcal{L} = -\frac{1}{n} \mathbf{X}^T (\mathbf{y} - \hat{\mathbf{p}})
$$

Where $\hat{\mathbf{p}} = \sigma(\mathbf{X}\mathbf{w})$ (apply sigmoid element-wise).

**Derivation** (for single example):

$$
\frac{\partial \mathcal{L}}{\partial w_j} = \frac{\partial}{\partial w_j} \left[ -y \log \sigma(z) - (1-y) \log(1 - \sigma(z)) \right]
$$

Where $z = \mathbf{w}^T \mathbf{x}$.

$$
= -y \frac{1}{\sigma(z)} \sigma'(z) \frac{\partial z}{\partial w_j} - (1-y) \frac{-1}{1-\sigma(z)} \sigma'(z) \frac{\partial z}{\partial w_j}
$$

Using $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and $\frac{\partial z}{\partial w_j} = x_j$:

$$
= -y \frac{1}{\sigma(z)} \sigma(z)(1-\sigma(z)) x_j + (1-y) \frac{1}{1-\sigma(z)} \sigma(z)(1-\sigma(z)) x_j
$$

$$
= -y (1-\sigma(z)) x_j + (1-y) \sigma(z) x_j
$$

$$
= -y x_j + y \sigma(z) x_j + \sigma(z) x_j - y \sigma(z) x_j
$$

$$
= (\sigma(z) - y) x_j = (\hat{p} - y) x_j
$$

**Remarkably similar to linear regression gradient**, but with $\hat{p}$ (probability) instead of $\hat{y}$ (prediction).

### Gradient Descent Update Rule

**Batch gradient descent**:

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \nabla_{\mathbf{w}} \mathcal{L}
$$

$$
\boxed{\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \frac{\alpha}{n} \mathbf{X}^T (\mathbf{y} - \hat{\mathbf{p}})}
$$

**Stochastic gradient descent** (single example):

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \alpha (y_i - \hat{p}_i) \mathbf{x}_i
$$

**Convergence**:
- Convex loss guarantees convergence to global minimum
- Typically converges in 10-100 iterations for moderate-sized problems
- Use adaptive methods (Adam, RMSprop) for faster convergence

---

## Regularization for Logistic Regression

Like linear regression, logistic regression can overfit (especially with high-dimensional data or small datasets).

### Ridge (L2) Regularization

**Loss function**:

$$
\mathcal{L}_{\text{Ridge}}(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right] + \lambda \|\mathbf{w}\|_2^2
$$

**Gradient**:

$$
\nabla_{\mathbf{w}} \mathcal{L}_{\text{Ridge}} = -\frac{1}{n} \mathbf{X}^T (\mathbf{y} - \hat{\mathbf{p}}) + 2\lambda \mathbf{w}
$$

**Common in neural networks** (weight decay).

### Lasso (L1) Regularization

**Loss function**:

$$
\mathcal{L}_{\text{Lasso}}(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log \hat{p}_i + (1 - y_i) \log (1 - \hat{p}_i) \right] + \lambda \|\mathbf{w}\|_1
$$

**Encourages sparse models** (feature selection).

### Elastic Net

Combines L1 and L2 penalties (balances sparsity and grouping).

**Note**: Scikit-learn's `LogisticRegression` uses L2 regularization by default, controlled by parameter `C` (inverse of $\lambda$).

---

## Multi-Class Classification

Logistic regression extends to multi-class problems ($K > 2$ classes).

### One-vs-Rest (OvR) / One-vs-All (OvA)

**Strategy**: Train $K$ binary classifiers, one for each class.

**For class $k$**:
- Positive examples: $y = k$
- Negative examples: $y \neq k$

**Prediction**:

$$
\hat{y} = \arg\max_{k} P(y=k | \mathbf{x})
$$

Choose the class with the highest predicted probability.

**Advantages**:
- Simple to implement (just $K$ binary classifiers)
- Works with any binary classifier

**Disadvantages**:
- Probabilities don't sum to 1 (each classifier is independent)
- Imbalanced classes per classifier (one positive class vs. $K-1$ negative)

### Multinomial Logistic Regression (Softmax Regression)

**Better approach**: Model all $K$ classes jointly using **softmax function**.

**Model**: For each class $k$, learn a weight vector $\mathbf{w}_k$.

**Probability of class $k$**:

$$
P(y=k | \mathbf{x}; \mathbf{W}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x}}} = \text{softmax}(\mathbf{z})_k
$$

Where:
- $\mathbf{W} = [\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_K]$ (matrix of all class weights)
- $\mathbf{z} = \mathbf{W}^T \mathbf{x}$ (logits for all classes)
- Softmax normalizes logits to probabilities (sums to 1)

**Softmax function**:

$$
\text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Properties**:
- Outputs valid probabilities: $\sum_k P(y=k | \mathbf{x}) = 1$
- Reduces to sigmoid for $K=2$ (binary case)
- Differentiable (gradient descent works)

### Categorical Cross-Entropy Loss

**Loss function** (for multi-class):

$$
\mathcal{L}(\mathbf{W}) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}(y_i = k) \log P(y=k | \mathbf{x}_i; \mathbf{W})
$$

Where $\mathbb{1}(y_i = k)$ is 1 if example $i$ belongs to class $k$, else 0.

**One-hot encoding**: If $\mathbf{y}_i$ is one-hot vector (e.g., $[0, 1, 0]$ for class 2):

$$
\mathcal{L}(\mathbf{W}) = -\frac{1}{n} \sum_{i=1}^{n} \mathbf{y}_i^T \log \hat{\mathbf{p}}_i
$$

**Gradient** for class $k$:

$$
\nabla_{\mathbf{w}_k} \mathcal{L} = -\frac{1}{n} \mathbf{X}^T (\mathbf{y}_k - \hat{\mathbf{p}}_k)
$$

Where $\hat{\mathbf{p}}_k = [\hat{p}_{1k}, \hat{p}_{2k}, \ldots, \hat{p}_{nk}]^T$ are predicted probabilities for class $k$.

**Same form as binary case!** Softmax gradient has elegant structure.

---

## Generalized Linear Models (GLMs)

Logistic regression is a special case of **Generalized Linear Models**, which extend linear regression to targets with non-Gaussian distributions.

### Components of a GLM

1. **Random component**: Distribution of target $y$ (from exponential family)
   - Linear regression: Gaussian
   - Logistic regression: Bernoulli
   - Poisson regression: Poisson
   - Gamma regression: Gamma

2. **Systematic component**: Linear predictor
   $$\eta = \mathbf{w}^T \mathbf{x}$$

3. **Link function** $g$: Connects mean $\mu = \mathbb{E}[y]$ to linear predictor
   $$\eta = g(\mu)$$

   **Inverse link function** $g^{-1}$:
   $$\mu = g^{-1}(\eta)$$

### Examples

| Regression Type | Target Distribution | Link Function $g(\mu)$ | Inverse Link $g^{-1}(\eta)$ |
|-----------------|---------------------|------------------------|------------------------------|
| **Linear** | Gaussian | Identity: $g(\mu) = \mu$ | $\mu = \eta$ |
| **Logistic** | Bernoulli | Logit: $g(\mu) = \log\frac{\mu}{1-\mu}$ | Sigmoid: $\mu = \frac{1}{1+e^{-\eta}}$ |
| **Poisson** | Poisson (count data) | Log: $g(\mu) = \log \mu$ | Exponential: $\mu = e^{\eta}$ |
| **Gamma** | Gamma (continuous, positive) | Inverse: $g(\mu) = \frac{1}{\mu}$ | $\mu = \frac{1}{\eta}$ |

### Canonical Link Functions

Each distribution in the exponential family has a **canonical link** that simplifies computation (e.g., logit for Bernoulli, log for Poisson).

**Why GLMs?**
- Unified framework for regression, classification, count data, etc.
- Maximum likelihood estimation via iterative reweighted least squares (IRLS)
- Flexible modeling of various data types

---

## Model Evaluation for Classification

### Confusion Matrix

For binary classification:

|                     | **Predicted Positive** | **Predicted Negative** |
|---------------------|------------------------|------------------------|
| **Actual Positive** | True Positive (TP)     | False Negative (FN)    |
| **Actual Negative** | False Positive (FP)    | True Negative (TN)     |

### Key Metrics

**Accuracy**:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

Proportion of correct predictions. **Issue**: Misleading for imbalanced classes.

**Precision** (Positive Predictive Value):

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Of all predicted positives, how many are actually positive?

**Recall** (Sensitivity, True Positive Rate):

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

Of all actual positives, how many did we correctly predict?

**F1-Score** (Harmonic mean of precision and recall):

$$
F1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

Balances precision and recall (useful when classes are imbalanced).

**Specificity** (True Negative Rate):

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

### ROC Curve and AUC

**ROC (Receiver Operating Characteristic) curve**:
- Plot **True Positive Rate (Recall)** vs. **False Positive Rate** as threshold varies
- FPR = $\frac{FP}{FP + TN}$ (proportion of negatives incorrectly classified as positive)

**AUC (Area Under the ROC Curve)**:
- Measures overall model performance across all thresholds
- Range: $[0, 1]$ (0.5 = random, 1 = perfect)
- Interpretation: Probability that model ranks random positive example higher than random negative example

**Why ROC-AUC?**
- Threshold-independent evaluation
- Good for comparing models
- Works well for imbalanced datasets

---

## Assumptions and Diagnostics

### Assumptions

1. **Linear relationship between log-odds and features**:
   $$\text{logit}(P(y=1)) = \mathbf{w}^T \mathbf{x}$$
   If relationship is non-linear, use polynomial features or non-linear models.

2. **Independence of observations**:
   Examples are independently sampled.

3. **No perfect multicollinearity**:
   Features should not be perfectly correlated (use regularization if present).

4. **Large sample size**:
   MLE is asymptotically consistent. Small samples may lead to unreliable estimates.

### When Logistic Regression Fails

- **Non-linear decision boundary**: Use polynomial features, kernel methods, or tree-based models
- **Feature interactions**: Manually add interaction terms
- **Imbalanced classes**: Use class weights, resampling (SMOTE), or adjust decision threshold
- **Perfect separation**: When classes are perfectly separable, MLE diverges (weights $\to \infty$). Use regularization.

---

## Practical Considerations

### Class Imbalance

**Problem**: If 95% of examples are class 0, predicting all 0's gives 95% accuracy but is useless.

**Solutions**:
1. **Class weights**: Penalize misclassifications of minority class more
   - Scikit-learn: `class_weight='balanced'`
2. **Resampling**: Oversample minority class (SMOTE) or undersample majority class
3. **Adjust threshold**: Instead of 0.5, use threshold that balances precision/recall
4. **Use F1, ROC-AUC**: Metrics that handle imbalance better than accuracy

### Feature Scaling

**Important**: Gradient descent converges faster with standardized features.

Regularization (L1/L2) requires features on the same scale to penalize fairly.

### Interpreting Coefficients

**For feature $j$**:
- $w_j > 0$: Increasing $x_j$ increases log-odds of class 1 (makes class 1 more likely)
- $w_j < 0$: Increasing $x_j$ decreases log-odds of class 1 (makes class 0 more likely)
- $|w_j|$: Magnitude shows feature importance (if features are standardized)

**Odds ratio**: $e^{w_j}$ gives the multiplicative change in odds for one-unit increase in $x_j$.

### Probability Calibration

Predicted probabilities $\hat{p}$ may not be well-calibrated (e.g., model outputs 0.7 but true probability is 0.5).

**Calibration methods**:
- Platt scaling (fit sigmoid on validation set)
- Isotonic regression (fit monotonic function)

**Check calibration**: Plot predicted probabilities vs. true frequencies (calibration curve).

---

## Comparison with Other Classifiers

| Aspect | Logistic Regression | Linear SVM | Decision Trees | Neural Networks |
|--------|---------------------|------------|----------------|-----------------|
| **Decision Boundary** | Linear | Linear | Non-linear (axis-aligned) | Non-linear (arbitrary) |
| **Probabilistic Output** | Yes | No (can calibrate) | Yes (via leaf purity) | Yes (with softmax) |
| **Loss Function** | Log-loss | Hinge loss | Gini/Entropy | Cross-entropy |
| **Regularization** | L1/L2 | Built-in (margin) | Pruning | Dropout, L2 |
| **Interpretability** | High | Medium | High | Low |
| **Scalability** | High | High | Medium | Medium |
| **Feature Scaling** | Required | Required | Not required | Required |

**When to use logistic regression**:
- Linear decision boundary is appropriate
- Need probabilistic outputs
- Want interpretable model (coefficients)
- Baseline classifier before trying complex methods

---

## Summary

**Logistic regression** is a fundamental classification algorithm that models the log-odds as a linear function of features:

- **Sigmoid function**: Maps linear predictor to probabilities in $[0, 1]$
- **Log-loss**: Convex loss derived from maximum likelihood estimation
- **Gradient descent**: Iterative optimization (no closed-form solution)
- **Regularization**: L1/L2 penalties prevent overfitting
- **Multi-class**: One-vs-Rest or softmax regression
- **GLM framework**: Special case of generalized linear models with logit link

**Key Takeaways**:
- Despite name, it's for **classification**, not regression
- Outputs **probabilities**, not just labels
- **Linear decision boundary** (use polynomial features for non-linear boundaries)
- **Convex optimization** guarantees global optimum
- **Simple, interpretable, effective** baseline for classification

**Next Steps**:
- Implement logistic regression from scratch (gradient descent, regularization)
- Explore other GLMs (Poisson, Gamma regression)
- Study non-linear classifiers (SVM with kernels, tree-based methods, neural networks)
