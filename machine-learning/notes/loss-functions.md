# Loss Functions - Comprehensive Guide

Loss functions (also called cost functions or objective functions) are mathematical functions that quantify the discrepancy between predicted and actual values. They form the foundation of machine learning optimization—the choice of loss function directly determines what your model learns to optimize.

## Table of Contents

1. [Fundamental Concepts](#fundamental-concepts)
2. [Classification Loss Functions](#classification-loss-functions)
3. [Regression Loss Functions](#regression-loss-functions)
4. [Mathematical Properties](#mathematical-properties)
5. [Choosing the Right Loss Function](#choosing-the-right-loss-function)
6. [Implementation Considerations](#implementation-considerations)

---

## Fundamental Concepts

### What is a Loss Function?

A loss function $L(y, \hat{y})$ measures the "cost" of predicting $\hat{y}$ when the true value is $y$.

**Key Properties:**
- **Non-negative**: $L(y, \hat{y}) \geq 0$
- **Zero at perfect prediction**: $L(y, y) = 0$
- **Increases with error**: Larger errors → larger loss

**Optimization Goal:**
$$
\min_{\theta} \frac{1}{n} \sum_{i=1}^{n} L(y_i, f(x_i; \theta))
$$

Where:
- $\theta$ = model parameters
- $f(x; \theta)$ = model prediction function
- $n$ = number of samples

### Loss vs Cost vs Objective Function

**Terminology** (often used interchangeably, but subtle differences):

- **Loss function**: Error for a single example $L(y_i, \hat{y}_i)$
- **Cost function**: Average loss over dataset $J(\theta) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{y}_i)$
- **Objective function**: What we optimize (cost + regularization)

$$
J(\theta) = \underbrace{\frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{y}_i)}_{\text{empirical risk}} + \underbrace{\lambda R(\theta)}_{\text{regularization}}
$$

---

## Classification Loss Functions

Classification tasks predict discrete labels. Loss functions measure prediction quality differently for binary vs multi-class problems.

### 1. Log Loss (Binary Cross-Entropy)

**Use Case**: Binary classification with probabilistic outputs

**Formula:**
$$
L(y, \hat{p}) = -[y \log(\hat{p}) + (1-y) \log(1-\hat{p})]
$$

Where:
- $y \in \{0, 1\}$ = true label
- $\hat{p} \in [0, 1]$ = predicted probability of class 1

**Multi-class Extension (Categorical Cross-Entropy):**
$$
L(y, \hat{p}) = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)
$$

Where:
- $y_k \in \{0, 1\}$ = one-hot encoded true label
- $\hat{p}_k$ = predicted probability for class $k$
- $\sum_{k=1}^{K} \hat{p}_k = 1$

**Properties:**
- ✓ **Convex**: Guaranteed global optimum
- ✓ **Smooth**: Continuous gradients everywhere
- ✓ **Probabilistic**: Works with probability outputs (0 to 1)
- ✓ **Penalizes confidence**: Wrong confident predictions penalized heavily
- ✗ **Unbounded**: Can approach infinity for very wrong predictions

**Gradient (Binary):**
$$
\frac{\partial L}{\partial \hat{p}} = -\frac{y}{\hat{p}} + \frac{1-y}{1-\hat{p}} = \frac{\hat{p} - y}{\hat{p}(1-\hat{p})}
$$

**When Combined with Sigmoid:**

If $\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}$, the gradient simplifies beautifully:
$$
\frac{\partial L}{\partial z} = \hat{p} - y
$$

This is why logistic regression + log loss is the standard combination.

**When to Use:**
- Binary or multi-class classification
- Need probability estimates
- Standard choice for neural networks (with softmax)
- Balanced classes

**Limitations:**
- Sensitive to class imbalance (majority class dominates)
- Outliers can cause numerical instability (log of very small numbers)

---

### 2. Hinge Loss

**Use Case**: Binary classification for margin-based classifiers (SVM)

**Formula:**
$$
L(y, f(x)) = \max(0, 1 - y \cdot f(x))
$$

Where:
- $y \in \{-1, +1\}$ = true label
- $f(x) \in \mathbb{R}$ = raw model output (not probability)

**Interpretation:**
- If $y \cdot f(x) \geq 1$: Correct classification with margin → $L = 0$
- If $y \cdot f(x) < 1$: Penalize proportional to violation → $L = 1 - y \cdot f(x)$

**Variants:**

**Squared Hinge Loss:**
$$
L(y, f(x)) = \max(0, 1 - y \cdot f(x))^2
$$
- Smoother, differentiable everywhere
- Penalizes large margin violations more heavily

**Multi-class Hinge (Crammer-Singer):**
$$
L(y, f) = \max(0, \max_{k \neq y} [f_k - f_y + 1])
$$

**Properties:**
- ✓ **Margin-based**: Encourages confident correct predictions
- ✓ **Sparse**: Many examples have zero loss (support vectors)
- ✓ **Robust**: Less sensitive to outliers than log loss
- ✗ **Not differentiable**: At $y \cdot f(x) = 1$ (but sub-gradient exists)
- ✗ **Not probabilistic**: Outputs aren't calibrated probabilities

**Sub-gradient:**
$$
\frac{\partial L}{\partial f(x)} = \begin{cases}
0 & \text{if } y \cdot f(x) > 1 \\
-y & \text{if } y \cdot f(x) < 1 \\
\text{undefined} & \text{if } y \cdot f(x) = 1
\end{cases}
$$

In practice, use $-y$ when $y \cdot f(x) \leq 1$.

**When to Use:**
- SVMs (linear classifiers)
- Want maximum margin separation
- Don't need probability estimates
- Robust to outliers

**Limitations:**
- Not probabilistic (can't get calibrated probabilities)
- Less interpretable than log loss

---

### 3. Focal Loss

**Use Case**: Binary/multi-class classification with severe class imbalance

**Formula (Binary):**
$$
L(y, \hat{p}) = -\alpha_y (1 - \hat{p}_y)^{\gamma} \log(\hat{p}_y)
$$

Where:
- $\hat{p}_y$ = predicted probability for true class
- $\gamma \geq 0$ = focusing parameter (typically 2)
- $\alpha_y$ = class weight for balancing

**Interpretation:**

Standard cross-entropy with a modulating factor $(1 - \hat{p}_y)^{\gamma}$:
- **Easy examples** ($\hat{p}_y$ close to 1): $(1-\hat{p}_y)^{\gamma}$ is small → loss down-weighted
- **Hard examples** ($\hat{p}_y$ far from 1): $(1-\hat{p}_y)^{\gamma}$ is large → loss emphasized

**Effect of $\gamma$:**
- $\gamma = 0$: Reduces to standard cross-entropy
- $\gamma = 1$: Moderate focusing
- $\gamma = 2$: Strong focusing (common default)
- Higher $\gamma$: More extreme focusing on hard examples

**Properties:**
- ✓ **Addresses class imbalance**: Automatically focuses on hard examples
- ✓ **Reduces easy example impact**: Prevents overwhelming by majority class
- ✓ **Improves rare class detection**: Better performance on minority classes
- ✗ **Hyperparameter sensitive**: Need to tune $\gamma$ and $\alpha$

**When to Use:**
- Severe class imbalance (e.g., 1:1000)
- Object detection (RetinaNet uses this)
- Medical diagnosis (rare diseases)
- Fraud detection

**Comparison to Weighted Cross-Entropy:**
- Weighted CE: Applies fixed weight to each class
- Focal Loss: Dynamic weighting based on prediction confidence
- Focal Loss more effective for extreme imbalance

---

## Regression Loss Functions

Regression tasks predict continuous values. Different loss functions handle outliers and error distributions differently.

### 1. Mean Squared Error (MSE, L2 Loss)

**Use Case**: Standard regression loss, assumes Gaussian errors

**Formula:**
$$
L(y, \hat{y}) = (y - \hat{y})^2
$$

**Average over dataset:**
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Properties:**
- ✓ **Convex**: Unique global minimum
- ✓ **Smooth**: Differentiable everywhere
- ✓ **Penalizes large errors**: Quadratic penalty
- ✓ **Statistical interpretation**: Maximum likelihood for Gaussian noise
- ✗ **Sensitive to outliers**: Squared term heavily penalizes outliers
- ✗ **Units**: Squared units (harder to interpret)

**Gradient:**
$$
\frac{\partial L}{\partial \hat{y}} = -2(y - \hat{y})
$$

**When to Use:**
- Standard regression problems
- Errors roughly normally distributed
- Want to heavily penalize large errors
- No significant outliers

**Maximum Likelihood Interpretation:**

Assuming $y = f(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$:
$$
P(y | x, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f(x; \theta))^2}{2\sigma^2}\right)
$$

Maximizing likelihood ≡ Minimizing MSE.

---

### 2. Mean Absolute Error (MAE, L1 Loss)

**Use Case**: Regression robust to outliers

**Formula:**
$$
L(y, \hat{y}) = |y - \hat{y}|
$$

**Average over dataset:**
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**Properties:**
- ✓ **Robust to outliers**: Linear penalty (not quadratic)
- ✓ **Same units**: Directly interpretable
- ✓ **Statistical interpretation**: Maximum likelihood for Laplace noise
- ✗ **Not differentiable**: At $y = \hat{y}$ (but sub-gradient exists)
- ✗ **Slower convergence**: Constant gradient magnitude

**Sub-gradient:**
$$
\frac{\partial L}{\partial \hat{y}} = \begin{cases}
-1 & \text{if } y > \hat{y} \\
+1 & \text{if } y < \hat{y} \\
\text{undefined} & \text{if } y = \hat{y}
\end{cases}
$$

In practice, use 0 when $y = \hat{y}$.

**When to Use:**
- Dataset has outliers
- Want robust regression
- Interpretability important (same units as target)
- Median regression preferred over mean

**Comparison to MSE:**

| Aspect | MSE | MAE |
|--------|-----|-----|
| **Outlier sensitivity** | High (squared) | Low (linear) |
| **Convergence** | Fast (growing gradient) | Slower (constant gradient) |
| **Differentiability** | Everywhere | Everywhere except at 0 |
| **Target** | Mean prediction | Median prediction |
| **Noise assumption** | Gaussian | Laplace |

---

### 3. Huber Loss

**Use Case**: Best of both worlds—MSE for small errors, MAE for large errors

**Formula:**
$$
L_{\delta}(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

Where $\delta > 0$ is a threshold parameter.

**Interpretation:**
- **Small errors** ($|y - \hat{y}| \leq \delta$): Behaves like MSE (quadratic)
- **Large errors** ($|y - \hat{y}| > \delta$): Behaves like MAE (linear)

**Properties:**
- ✓ **Robust to outliers**: Linear penalty for large errors
- ✓ **Differentiable**: Everywhere (unlike MAE)
- ✓ **Fast convergence**: Near optimum (like MSE)
- ✗ **Hyperparameter**: Need to tune $\delta$

**Gradient:**
$$
\frac{\partial L_{\delta}}{\partial \hat{y}} = \begin{cases}
-(y - \hat{y}) & \text{if } |y - \hat{y}| \leq \delta \\
-\delta \cdot \text{sign}(y - \hat{y}) & \text{otherwise}
\end{cases}
$$

**Choosing $\delta$:**
- Common heuristic: Use 1.35 × MAD (median absolute deviation)
- Or tune as hyperparameter via cross-validation
- Smaller $\delta$: More like MAE (robust)
- Larger $\delta$: More like MSE (efficient)

**When to Use:**
- Dataset has some outliers (not too many)
- Want robustness without sacrificing too much efficiency
- Standard choice for robust regression

---

### 4. Quantile Loss (Pinball Loss)

**Use Case**: Quantile regression—predict specific quantiles, not just mean/median

**Formula:**
$$
L_{\tau}(y, \hat{y}) = \begin{cases}
\tau \cdot (y - \hat{y}) & \text{if } y \geq \hat{y} \\
(1 - \tau) \cdot (\hat{y} - y) & \text{if } y < \hat{y}
\end{cases}
$$

Or equivalently:
$$
L_{\tau}(y, \hat{y}) = (y - \hat{y}) \cdot (\tau - \mathbb{1}_{y < \hat{y}})
$$

Where:
- $\tau \in (0, 1)$ = target quantile
- $\mathbb{1}_{y < \hat{y}}$ = indicator function

**Special Cases:**
- $\tau = 0.5$: Median regression (equivalent to MAE)
- $\tau = 0.9$: 90th percentile regression
- $\tau = 0.1$: 10th percentile regression

**Interpretation:**

Asymmetric penalty:
- **Underpredict** ($\hat{y} < y$): Penalty = $\tau \cdot |y - \hat{y}|$
- **Overpredict** ($\hat{y} > y$): Penalty = $(1-\tau) \cdot |y - \hat{y}|$

For $\tau = 0.9$:
- Underprediction costs 9× more than overprediction
- Model learns to predict high to capture 90th percentile

**Properties:**
- ✓ **Predicts quantiles**: Not just mean/median
- ✓ **Uncertainty estimation**: Multiple quantiles → prediction intervals
- ✓ **Robust**: Similar to MAE
- ✗ **Not differentiable**: At $y = \hat{y}$

**Sub-gradient:**
$$
\frac{\partial L_{\tau}}{\partial \hat{y}} = \begin{cases}
-\tau & \text{if } y > \hat{y} \\
1 - \tau & \text{if } y < \hat{y}
\end{cases}
$$

**When to Use:**
- Need prediction intervals (not just point estimates)
- Asymmetric costs (e.g., understock vs overstock in inventory)
- Forecasting (predict range of outcomes)
- Risk-sensitive predictions

**Example Application:**

Demand forecasting:
- Predict $\tau = 0.9$ quantile to ensure 90% service level
- Underprediction (stockout) is costly → high $\tau$

---

### 5. Log-Cosh Loss

**Use Case**: Smooth approximation to MAE, combines benefits of MSE and MAE

**Formula:**
$$
L(y, \hat{y}) = \log(\cosh(y - \hat{y}))
$$

Where $\cosh(x) = \frac{e^x + e^{-x}}{2}$ (hyperbolic cosine).

**Approximation:**
- For small errors: $\approx \frac{1}{2}(y - \hat{y})^2$ (like MSE)
- For large errors: $\approx |y - \hat{y}| - \log(2)$ (like MAE)

**Properties:**
- ✓ **Smooth**: Twice differentiable everywhere
- ✓ **Robust**: Approximately linear for large errors
- ✓ **No hyperparameters**: Unlike Huber loss
- ✗ **Computational cost**: More expensive than MSE/MAE

**Gradient:**
$$
\frac{\partial L}{\partial \hat{y}} = -\tanh(y - \hat{y})
$$

Since $\tanh(x) \in [-1, 1]$, gradient is bounded (robust).

**When to Use:**
- Want smoothness of MSE with robustness of MAE
- Don't want to tune hyperparameters (vs Huber)
- Gradient-based optimization with outliers

---

## Mathematical Properties

Understanding these properties helps choose the right loss function.

### Convexity

**Definition**: A function $L$ is convex if for all $\hat{y}_1, \hat{y}_2$ and $\lambda \in [0,1]$:
$$
L(\lambda \hat{y}_1 + (1-\lambda) \hat{y}_2) \leq \lambda L(\hat{y}_1) + (1-\lambda) L(\hat{y}_2)
$$

**Why It Matters:**
- Convex loss → guaranteed global optimum
- Non-convex loss → may get stuck in local minima

**Convexity Status:**

| Loss Function | Convex? |
|---------------|---------|
| MSE | ✓ Yes |
| MAE | ✓ Yes |
| Huber | ✓ Yes |
| Log-Cosh | ✓ Yes |
| Quantile | ✓ Yes |
| Log Loss | ✓ Yes (for linear model) |
| Hinge | ✓ Yes |
| Focal | ✗ No (but used successfully) |

**Note**: Convexity depends on model. Log loss is convex for linear models but non-convex for neural networks.

---

### Differentiability

**Why It Matters:**
- Smooth gradients → efficient gradient descent
- Non-differentiable → need sub-gradients or smoothing

**Differentiability Status:**

| Loss Function | Differentiable? | Notes |
|---------------|-----------------|-------|
| MSE | ✓ Everywhere | Smooth |
| MAE | ✗ At $y = \hat{y}$ | Use sub-gradient |
| Huber | ✓ Everywhere | Smooth approximation to MAE |
| Log-Cosh | ✓ Everywhere | Twice differentiable |
| Quantile | ✗ At $y = \hat{y}$ | Use sub-gradient |
| Log Loss | ✓ Everywhere | Undefined at 0, 1 (numerical issues) |
| Hinge | ✗ At $y \cdot f(x) = 1$ | Use sub-gradient |

---

### Robustness to Outliers

**Measurement**: How much does loss grow with error magnitude?

**Growth Rates:**

| Loss Function | Growth Rate | Robustness |
|---------------|-------------|------------|
| MSE | $O(e^2)$ | ✗ Not robust |
| MAE | $O(e)$ | ✓ Robust |
| Huber | $O(e^2)$ small, $O(e)$ large | ✓ Moderately robust |
| Log-Cosh | $O(e)$ for large $e$ | ✓ Robust |
| Quantile | $O(e)$ | ✓ Robust |

**Visual Comparison** (error on x-axis, loss on y-axis):
- **MSE**: Parabola (grows rapidly)
- **MAE**: V-shape (linear growth)
- **Huber**: Parabola transitioning to linear
- **Log-Cosh**: Smooth transition from parabolic to linear

---

### Gradient Magnitude

**Why It Matters:** Affects optimization convergence speed

| Loss Function | Gradient Behavior |
|---------------|-------------------|
| MSE | Grows with error (fast convergence far from optimum) |
| MAE | Constant magnitude (slower convergence) |
| Huber | Growing (small errors), constant (large errors) |
| Log-Cosh | Bounded via $\tanh$ (stable) |

**Trade-off:**
- Large gradients far from optimum → Fast convergence
- But sensitive to outliers
- Constant gradients → Slower but robust

---

## Choosing the Right Loss Function

### Classification

**Decision Tree:**

```
Is it binary or multi-class?
│
├─ Binary Classification
│  │
│  ├─ Severe class imbalance? → Focal Loss
│  ├─ Need probabilities? → Log Loss (Cross-Entropy)
│  └─ Maximum margin (SVM)? → Hinge Loss
│
└─ Multi-class Classification
   │
   ├─ Severe class imbalance? → Focal Loss
   ├─ Standard case → Categorical Cross-Entropy
   └─ SVM approach → Multi-class Hinge
```

**Summary Table:**

| Scenario | Best Loss | Reason |
|----------|-----------|--------|
| Standard binary classification | Log Loss | Probabilistic, well-calibrated |
| Imbalanced binary (e.g., fraud) | Focal Loss | Focuses on hard examples |
| SVM / maximum margin | Hinge Loss | Encourages large margins |
| Multi-class | Categorical Cross-Entropy | Standard, works with softmax |
| Multi-class imbalanced | Focal Loss | Handles class imbalance |

---

### Regression

**Decision Tree:**

```
What's the error distribution?
│
├─ Gaussian errors, no outliers? → MSE
│  (Most common case)
│
├─ Has outliers?
│  │
│  ├─ Few outliers → Huber Loss
│  ├─ Many outliers → MAE
│  └─ Want smoothness → Log-Cosh
│
└─ Need quantiles / prediction intervals? → Quantile Loss
   (e.g., forecasting, inventory optimization)
```

**Summary Table:**

| Scenario | Best Loss | Reason |
|----------|-----------|--------|
| Standard regression | MSE | Fast convergence, well-understood |
| Dataset has outliers | Huber or MAE | Robust to outliers |
| Need prediction intervals | Quantile Loss | Predicts specific quantiles |
| Want smooth + robust | Log-Cosh | Combines MSE and MAE benefits |
| Asymmetric cost | Quantile Loss ($\tau \neq 0.5$) | Penalize errors differently |

---

### Practical Guidelines

**1. Start Simple:**
- Classification: Log Loss (Cross-Entropy)
- Regression: MSE

**2. Diagnose Issues:**
- Model performs poorly on minority class → Try Focal Loss
- Large errors dominate training → Try robust loss (MAE, Huber)
- Want uncertainty estimates → Try Quantile Loss

**3. Consider Context:**
- Outliers expected? → Robust loss
- Class imbalance? → Weighted or Focal Loss
- Need probabilities? → Log Loss
- Asymmetric costs? → Custom loss or Quantile

**4. Validate:**
- Always validate on held-out data
- Loss function ≠ evaluation metric
- Optimize for what you care about

---

## Implementation Considerations

### Numerical Stability

**Log Loss:**
Problem: $\log(0) = -\infty$ and $\log(1) = 0$ cause numerical issues.

**Solution**: Clip predictions to $[\epsilon, 1-\epsilon]$ where $\epsilon \approx 10^{-15}$:
```python
eps = 1e-15
p_clipped = np.clip(p, eps, 1 - eps)
loss = -np.mean(y * np.log(p_clipped) + (1 - y) * np.log(1 - p_clipped))
```

**Better Solution**: Use log-sum-exp trick or built-in numerically stable implementations (e.g., `log_softmax`).

---

### Regularization

Loss functions are often combined with regularization:
$$
J(\theta) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, \hat{y}_i) + \lambda R(\theta)
$$

**Common Regularizers:**
- **L2 (Ridge)**: $R(\theta) = \|\theta\|_2^2$ (penalize large weights)
- **L1 (Lasso)**: $R(\theta) = \|\theta\|_1$ (encourage sparsity)
- **Elastic Net**: $R(\theta) = \alpha \|\theta\|_1 + (1-\alpha) \|\theta\|_2^2$

---

### Custom Loss Functions

You can design custom loss functions for specific problems:

**Requirements:**
1. Non-negative: $L(y, \hat{y}) \geq 0$
2. Zero at perfect prediction: $L(y, y) = 0$
3. Differentiable (or has sub-gradient)

**Example: Asymmetric MSE**
```python
def asymmetric_mse(y_true, y_pred, overpredict_penalty=2.0):
    """
    Penalize overprediction more than underprediction.
    Useful for inventory: overstocking is more costly.
    """
    error = y_true - y_pred
    loss = np.where(error >= 0,  # underprediction
                    error**2,
                    overpredict_penalty * error**2)
    return np.mean(loss)
```

---

### Optimization Considerations

**Gradient Descent:**
- Smooth losses (MSE, Log Loss) converge faster
- Non-smooth losses (MAE, Hinge) may need smaller learning rates

**Second-Order Methods:**
- Requires Hessian (second derivative)
- Only works for twice-differentiable losses
- Faster convergence but more expensive per iteration

**Stochastic Gradient Descent:**
- All losses work with mini-batch SGD
- May need gradient clipping for unstable losses

---

## Quick Reference Table

### All Loss Functions at a Glance

| Loss | Type | Formula | Convex | Smooth | Robust | Use Case |
|------|------|---------|--------|--------|--------|----------|
| **Log Loss** | Classification | $-y\log\hat{p} - (1-y)\log(1-\hat{p})$ | ✓ | ✓ | ✗ | Standard classification |
| **Hinge** | Classification | $\max(0, 1-y \cdot f(x))$ | ✓ | ✗ | ✓ | SVM, max margin |
| **Focal** | Classification | $-(1-\hat{p})^{\gamma}\log\hat{p}$ | ✗ | ✓ | - | Imbalanced classes |
| **MSE** | Regression | $(y-\hat{y})^2$ | ✓ | ✓ | ✗ | Standard regression |
| **MAE** | Regression | $\|y-\hat{y}\|$ | ✓ | ✗ | ✓ | Robust regression |
| **Huber** | Regression | Quadratic/Linear | ✓ | ✓ | ✓ | Robust + smooth |
| **Log-Cosh** | Regression | $\log(\cosh(y-\hat{y}))$ | ✓ | ✓✓ | ✓ | Smooth + robust |
| **Quantile** | Regression | $(y-\hat{y})(\tau - \mathbb{1}_{y<\hat{y}})$ | ✓ | ✗ | ✓ | Quantile regression |

### Gradient Summary

| Loss | Gradient $\frac{\partial L}{\partial \hat{y}}$ |
|------|-----------------------------------------------|
| MSE | $-2(y - \hat{y})$ |
| MAE | $-\text{sign}(y - \hat{y})$ |
| Huber ($\delta$) | Quadratic if $\|y-\hat{y}\| \leq \delta$, else linear |
| Log-Cosh | $-\tanh(y - \hat{y})$ |
| Log Loss | $\frac{\hat{p} - y}{\hat{p}(1-\hat{p})}$ (or $\hat{p} - y$ with sigmoid) |
| Hinge | $-y$ if $y \cdot f(x) < 1$, else 0 |

---

## Common Mistakes to Avoid

### 1. Wrong Loss for the Task
- **Mistake**: Using MSE for classification
- **Fix**: Use Log Loss (Cross-Entropy) for classification

### 2. Ignoring Class Imbalance
- **Mistake**: Standard Cross-Entropy on 1:100 imbalanced dataset
- **Fix**: Use Focal Loss or weighted Cross-Entropy

### 3. Not Handling Outliers
- **Mistake**: Using MSE when dataset has outliers
- **Fix**: Use MAE, Huber, or Log-Cosh

### 4. Numerical Instability
- **Mistake**: $\log(0)$ in Log Loss implementation
- **Fix**: Clip predictions to $[\epsilon, 1-\epsilon]$

### 5. Mismatch with Evaluation Metric
- **Mistake**: Optimizing MSE but evaluating on MAE
- **Fix**: Align training loss with evaluation metric when possible

### 6. Wrong Probability Range
- **Mistake**: Using Log Loss with outputs not in $[0, 1]$
- **Fix**: Apply sigmoid or softmax to ensure valid probabilities

### 7. Not Scaling Features
- **Mistake**: Different feature scales affect loss magnitude
- **Fix**: Standardize features before training (especially for distance-based losses)

---

## Further Reading

### Key Papers

1. **Focal Loss** - Lin et al. (2017): "Focal Loss for Dense Object Detection"
   - Addresses class imbalance in object detection

2. **Huber Loss** - Huber (1964): "Robust Estimation of a Location Parameter"
   - Original robust regression paper

3. **Quantile Regression** - Koenker & Bassett (1978): "Regression Quantiles"
   - Foundation of quantile loss

### Books

- **ESL**: Hastie et al., "The Elements of Statistical Learning" (Chapter 10: Boosting and Additive Trees)
- **Deep Learning**: Goodfellow et al. (Chapter 6: Deep Feedforward Networks)

### Online Resources

- Scikit-learn documentation on loss functions
- PyTorch loss function documentation
- TensorFlow Keras losses

---

## Summary

**Key Takeaways:**

1. **Loss function choice matters**: Determines what your model optimizes
2. **No one-size-fits-all**: Different problems need different losses
3. **Start simple**: MSE for regression, Cross-Entropy for classification
4. **Diagnose issues**: Outliers → robust loss, Imbalance → Focal loss
5. **Mathematical properties**: Convexity, smoothness, robustness guide choice
6. **Implementation**: Watch for numerical stability, especially in Log Loss
7. **Validate**: Always check performance on held-out data

**General Guidelines:**

- **Classification**: Default to Log Loss, switch to Focal for imbalance
- **Regression**: Default to MSE, switch to Huber/MAE for outliers
- **Uncertainty**: Use Quantile Loss for prediction intervals
- **Custom problems**: Design loss aligned with business objective

The right loss function bridges the gap between mathematical optimization and real-world objectives.
