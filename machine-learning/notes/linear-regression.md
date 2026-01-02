# Linear Regression

## Overview

**Linear regression** is one of the most fundamental supervised learning algorithms, modeling the relationship between a dependent variable (target) and one or more independent variables (features) using a linear function. Despite its simplicity, linear regression forms the foundation for understanding more complex models and introduces key concepts in machine learning: loss functions, optimization, gradient descent, and the bias-variance tradeoff.

**Key Characteristics:**
- **Parametric model**: Assumes a specific functional form (linear relationship)
- **Continuous output**: Predicts real-valued targets
- **Interpretable**: Coefficients directly show feature importance and direction of influence
- **Closed-form solution**: Can be solved analytically (normal equations) or iteratively (gradient descent)

---

## Mathematical Formulation

### Simple Linear Regression (One Feature)

For a single feature $x$, the model predicts:

$$
\hat{y} = w_0 + w_1 x
$$

Where:
- $\hat{y}$ = predicted value
- $w_0$ = intercept (bias term)
- $w_1$ = slope (weight/coefficient)
- $x$ = input feature

### Multiple Linear Regression

For $d$ features, the model becomes:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_d x_d
$$

**Vector notation** (more compact):

$$
\hat{y} = \mathbf{w}^T \mathbf{x}
$$

Where:
- $\mathbf{w} = [w_0, w_1, \ldots, w_d]^T$ = weight vector (includes bias as $w_0$)
- $\mathbf{x} = [1, x_1, x_2, \ldots, x_d]^T$ = feature vector (prepend 1 for bias term)

**Matrix notation** (for $n$ training examples):

$$
\hat{\mathbf{y}} = \mathbf{X} \mathbf{w}
$$

Where:
- $\mathbf{X} \in \mathbb{R}^{n \times (d+1)}$ = design matrix (each row is one example)
- $\mathbf{w} \in \mathbb{R}^{(d+1)}$ = weight vector
- $\hat{\mathbf{y}} \in \mathbb{R}^n$ = predictions for all examples

---

## Loss Function: Mean Squared Error (MSE)

Linear regression minimizes the **Mean Squared Error** between predictions and true values:

$$
\text{MSE}(\mathbf{w}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^{n} (y_i - \mathbf{w}^T \mathbf{x}_i)^2
$$

**Matrix form**:

$$
\text{MSE}(\mathbf{w}) = \frac{1}{n} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 = \frac{1}{n} (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w})
$$

**Why MSE?**
- **Differentiable**: Smooth gradient for optimization
- **Convex**: Guarantees single global minimum (no local minima)
- **Penalizes large errors**: Squared term heavily penalizes outliers
- **Statistical interpretation**: Maximum Likelihood Estimation (MLE) under Gaussian noise assumption

**Alternative**: Mean Absolute Error (MAE) is more robust to outliers but not differentiable at zero.

---

## Solution Methods

### Method 1: Normal Equations (Closed-Form Solution)

The optimal weights that minimize MSE can be found by setting the gradient to zero:

$$
\nabla_{\mathbf{w}} \text{MSE} = 0
$$

**Derivation**:

$$
\begin{align}
\text{MSE}(\mathbf{w}) &= \frac{1}{n} (\mathbf{y} - \mathbf{X}\mathbf{w})^T (\mathbf{y} - \mathbf{X}\mathbf{w}) \\
&= \frac{1}{n} (\mathbf{y}^T\mathbf{y} - 2\mathbf{y}^T\mathbf{X}\mathbf{w} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w})
\end{align}
$$

Taking the gradient with respect to $\mathbf{w}$:

$$
\nabla_{\mathbf{w}} \text{MSE} = \frac{1}{n} (-2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\mathbf{w}) = 0
$$

Solving for $\mathbf{w}$:

$$
\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}
$$

$$
\boxed{\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}
$$

This is the **normal equation** (also called the **ordinary least squares** or OLS solution).

**Computational Complexity**: $O(d^3 + nd^2)$
- $\mathbf{X}^T\mathbf{X}$: $O(nd^2)$ (matrix multiplication)
- Inverting $(\mathbf{X}^T\mathbf{X})$: $O(d^3)$ (matrix inversion)

**Advantages**:
- Exact solution (no hyperparameters)
- One-step computation
- No iterative tuning required

**Disadvantages**:
- $O(d^3)$ becomes expensive for high-dimensional data ($d > 10{,}000$)
- Requires $\mathbf{X}^T\mathbf{X}$ to be invertible (non-singular)
- Fails when features are linearly dependent (multicollinearity)
- Not suitable for online learning or streaming data

**Practical Note**: Use **Cholesky decomposition** or **QR decomposition** instead of explicit matrix inversion for numerical stability.

---

### Method 2: Gradient Descent (Iterative Optimization)

Instead of solving analytically, we can iteratively update weights by moving in the direction that decreases the loss.

**Gradient of MSE**:

$$
\nabla_{\mathbf{w}} \text{MSE} = -\frac{2}{n} \mathbf{X}^T (\mathbf{y} - \mathbf{X}\mathbf{w})
$$

**Update rule** (iterative):

$$
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \nabla_{\mathbf{w}} \text{MSE}
$$

$$
\boxed{\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \frac{2\alpha}{n} \mathbf{X}^T (\mathbf{y} - \mathbf{X}\mathbf{w}^{(t)})}
$$

Where:
- $\alpha$ = learning rate (controls step size)
- $t$ = iteration number

**Variants**:

1. **Batch Gradient Descent**: Use all $n$ examples per update
   - Stable convergence but slow for large datasets

2. **Stochastic Gradient Descent (SGD)**: Use one random example per update
   - Fast updates but noisy convergence
   - Gradient for single example $i$: $\nabla_{\mathbf{w}} = -2(y_i - \mathbf{w}^T\mathbf{x}_i)\mathbf{x}_i$

3. **Mini-Batch Gradient Descent**: Use small random subset (e.g., 32-256 examples)
   - Balances speed and stability
   - Standard in deep learning

**Convergence**:
- Linear regression has a **convex** loss surface (bowl-shaped)
- Guaranteed to converge to global minimum with appropriate learning rate
- Typical convergence: $O(1/\epsilon)$ iterations to reach $\epsilon$ accuracy

**Advantages**:
- Scales well to large datasets ($n \gg d$)
- Works even when $\mathbf{X}^T\mathbf{X}$ is singular
- Supports online learning (update as new data arrives)
- Extends naturally to non-linear models (neural networks)

**Disadvantages**:
- Requires tuning learning rate $\alpha$
- Needs multiple iterations (slower than normal equations for small $d$)
- Can converge slowly near minimum (zig-zagging)

**Learning Rate Selection**:
- Too large: Diverges or oscillates
- Too small: Converges very slowly
- Common strategies: Learning rate decay, adaptive methods (Adam, RMSprop)

---

## Assumptions of Linear Regression

Linear regression makes several assumptions. Violations affect model performance and inference:

### 1. **Linearity**
The relationship between features and target is linear.

**Check**: Residual plots should show no patterns
**Violation**: Use polynomial features or non-linear models

### 2. **Independence**
Observations are independent (no autocorrelation in residuals).

**Check**: Durbin-Watson test for time series
**Violation**: Use time series models or add lagged features

### 3. **Homoscedasticity**
Constant variance of errors across all levels of features.

$$
\text{Var}(\epsilon_i) = \sigma^2 \quad \text{(constant for all } i)
$$

**Check**: Plot residuals vs. fitted values (should be random scatter)
**Violation**: Use weighted least squares or transform target (log, sqrt)

### 4. **Normality of Residuals**
Errors follow a normal distribution.

$$
\epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

**Check**: Q-Q plot, Shapiro-Wilk test
**Violation**: Affects confidence intervals but not predictions (for large $n$, CLT helps)

### 5. **No Multicollinearity**
Features are not highly correlated with each other.

**Check**: Variance Inflation Factor (VIF) < 10
**Violation**: Remove correlated features or use regularization (Ridge)

---

## Model Evaluation

### 1. **R² (Coefficient of Determination)**

Proportion of variance in $y$ explained by the model:

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

Where:
- $\text{SS}_{\text{res}}$ = residual sum of squares (unexplained variance)
- $\text{SS}_{\text{tot}}$ = total sum of squares (total variance)
- $\bar{y}$ = mean of target values

**Range**: $[0, 1]$ (higher is better)
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: Model no better than predicting the mean

**Issue**: Always increases when adding features (even irrelevant ones)

### 2. **Adjusted R²**

Penalizes adding features that don't improve fit:

$$
R^2_{\text{adj}} = 1 - \frac{(1 - R^2)(n - 1)}{n - d - 1}
$$

Where:
- $n$ = number of examples
- $d$ = number of features

Only increases if new feature improves model more than expected by chance.

### 3. **Mean Squared Error (MSE) / Root Mean Squared Error (RMSE)**

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

**Advantage**: Same units as target variable (for RMSE)

### 4. **Mean Absolute Error (MAE)**

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**Advantage**: More robust to outliers than MSE

---

## Extensions

### 1. **Polynomial Regression**

Capture non-linear relationships by adding polynomial features:

$$
\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \cdots + w_p x^p
$$

**Still linear in parameters** $\mathbf{w}$, so normal equations still apply.

**Feature transformation**: Create new features $[1, x, x^2, x^3, \ldots, x^p]$

**Caution**: High-degree polynomials can overfit. Use cross-validation to select degree $p$.

### 2. **Interaction Features**

Model interactions between features:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + w_3 (x_1 \cdot x_2)
$$

**Example**: Predicting house price with area and location. The effect of area may depend on location (interaction).

### 3. **Feature Scaling**

Gradient descent converges faster when features are on similar scales.

**Standardization** (zero mean, unit variance):

$$
x'_j = \frac{x_j - \mu_j}{\sigma_j}
$$

**Min-Max Normalization** (scale to $[0, 1]$):

$$
x'_j = \frac{x_j - \min(x_j)}{\max(x_j) - \min(x_j)}
$$

**Note**: Normal equations don't require scaling, but it improves numerical stability.

---

## Practical Considerations

### When to Use Linear Regression

**Best for**:
- Linear relationships between features and target
- Interpretability is important (coefficients show feature impact)
- Small to medium feature dimensionality ($d < n$)
- Baseline model before trying complex methods

**Not suitable for**:
- Highly non-linear relationships (use trees, neural networks, or kernels)
- Very high-dimensional data ($d \gg n$) without regularization
- Classification tasks (use logistic regression instead)

### Handling Overfitting

**Symptoms**:
- High $R^2$ on training data, poor on test data
- Very large weight magnitudes

**Solutions**:
1. **Regularization**: Ridge, Lasso, Elastic Net (see regularization notes)
2. **Feature selection**: Remove irrelevant features
3. **Cross-validation**: Validate generalization
4. **More data**: Collect additional training examples

### Handling Underfitting

**Symptoms**:
- Low $R^2$ on both training and test data
- Residual plots show clear patterns

**Solutions**:
1. **Add features**: Polynomial terms, interactions
2. **Remove constraints**: Reduce regularization strength
3. **Try non-linear models**: Decision trees, kernel methods

---

## Comparison: Normal Equations vs. Gradient Descent

| Aspect | Normal Equations | Gradient Descent |
|--------|------------------|------------------|
| **Speed** | Fast for small $d$ ($d < 10{,}000$) | Slow but scales to large $d$ |
| **Complexity** | $O(d^3 + nd^2)$ | $O(knd)$ for $k$ iterations |
| **Large datasets** | Slow (requires computing $\mathbf{X}^T\mathbf{X}$) | Fast (can use mini-batches) |
| **Singular $\mathbf{X}^T\mathbf{X}$** | Fails (non-invertible) | Still works |
| **Online learning** | Not supported | Supported (update per example) |
| **Hyperparameters** | None | Learning rate $\alpha$ |
| **Convergence** | Exact, one step | Iterative, approximate |

**Rule of thumb**:
- Use **normal equations** for $d < 10{,}000$ and when exact solution is needed
- Use **gradient descent** for $d \geq 10{,}000$, online learning, or when extending to non-linear models

---

## Connection to Other Models

### From Linear Regression to...

1. **Logistic Regression**: Apply sigmoid function to linear model for binary classification
2. **Generalized Linear Models (GLMs)**: Use different link functions and distributions
3. **Ridge/Lasso**: Add regularization penalties to prevent overfitting
4. **Neural Networks**: Stack multiple linear layers with non-linear activations
5. **Support Vector Regression (SVR)**: Use different loss function (epsilon-insensitive)

---

## Summary

**Linear regression** is the simplest supervised learning algorithm but teaches fundamental concepts:
- **Loss functions** (MSE) and optimization (gradient descent vs. closed-form)
- **Bias-variance tradeoff** (simple model, high bias, low variance)
- **Assumptions** that affect model validity
- **Evaluation metrics** ($R^2$, RMSE) for regression tasks

**Key Takeaways**:
- Closed-form solution exists but doesn't scale to large $d$
- Gradient descent is iterative but more flexible and scalable
- Assumptions matter for statistical inference but less for prediction
- Simple yet powerful baseline that's often underestimated
- Understanding linear regression is essential before moving to complex models

**Next Steps**:
- Regularization (Ridge, Lasso, Elastic Net) to handle overfitting
- Logistic regression for classification
- Generalized Linear Models (GLMs) for non-Gaussian targets
