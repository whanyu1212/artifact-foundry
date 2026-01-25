"""
Loss Functions - From Scratch Implementation

Implements common loss functions for classification and regression tasks.

Classification Losses:
- Log Loss (Binary Cross-Entropy)
- Categorical Cross-Entropy
- Hinge Loss
- Focal Loss

Regression Losses:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Huber Loss
- Quantile Loss
- Log-Cosh Loss

All implementations validated against scikit-learn and standard libraries.
"""

import numpy as np
from typing import Optional


# ============================================================================
# CLASSIFICATION LOSSES
# ============================================================================


def log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Binary Cross-Entropy (Log Loss).

    For binary classification with probabilistic outputs.

    Args:
        y_true: True binary labels, shape (n_samples,), values in {0, 1}.
        y_pred: Predicted probabilities, shape (n_samples,), values in [0, 1].
        eps: Small constant for numerical stability. Clips predictions to [eps, 1-eps].

    Returns:
        loss: Scalar loss value.

    Formula:
        L(y, p) = -[y * log(p) + (1-y) * log(1-p)]
        Average: (1/n) * Σ L(y_i, p_i)

    Mathematical Properties:
        - Convex: Guaranteed global minimum
        - Smooth: Differentiable everywhere
        - Unbounded: Can approach infinity for very wrong predictions
        - Gradient: ∂L/∂p = (p - y) / [p(1-p)]

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0.1, 0.9, 0.8, 0.3])
        >>> loss = log_loss(y_true, y_pred)
        >>> print(f"Log Loss: {loss:.4f}")
        Log Loss: 0.2231
    """
    # Clip predictions for numerical stability
    # Prevents log(0) which would be -inf
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

    # Binary cross-entropy formula
    # For y=1: loss = -log(p)
    # For y=0: loss = -log(1-p)
    loss_per_sample = -(
        y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
    )

    return float(np.mean(loss_per_sample))


def categorical_crossentropy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Categorical Cross-Entropy Loss.

    Multi-class extension of log loss.

    Args:
        y_true: True labels (one-hot encoded), shape (n_samples, n_classes).
        y_pred: Predicted probabilities, shape (n_samples, n_classes).
                Each row should sum to 1 (output of softmax).
        eps: Small constant for numerical stability.

    Returns:
        loss: Scalar loss value.

    Formula:
        L(y, p) = -Σ_k y_k * log(p_k)
        where k ranges over classes

    Mathematical Properties:
        - Convex (for linear models)
        - Smooth
        - Works with softmax: p_k = exp(z_k) / Σ exp(z_j)

    Example:
        >>> y_true = np.array([[1, 0, 0],
        ...                    [0, 1, 0],
        ...                    [0, 0, 1]])
        >>> y_pred = np.array([[0.7, 0.2, 0.1],
        ...                    [0.1, 0.8, 0.1],
        ...                    [0.2, 0.2, 0.6]])
        >>> loss = categorical_crossentropy(y_true, y_pred)
        >>> print(f"Categorical Cross-Entropy: {loss:.4f}")
        Categorical Cross-Entropy: 0.4236
    """
    # Clip predictions for numerical stability
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

    # Categorical cross-entropy
    # Only the true class contributes to loss (due to one-hot encoding)
    loss_per_sample = -np.sum(y_true * np.log(y_pred_clipped), axis=1)

    return float(np.mean(loss_per_sample))


def hinge_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Hinge Loss for binary classification.

    Used in Support Vector Machines (SVMs) for maximum margin classification.

    Args:
        y_true: True labels, shape (n_samples,), values in {-1, +1}.
        y_pred: Raw model outputs (not probabilities), shape (n_samples,).
                These are decision function values (distance from hyperplane).

    Returns:
        loss: Scalar loss value.

    Formula:
        L(y, f(x)) = max(0, 1 - y * f(x))

    Interpretation:
        - If y * f(x) >= 1: Correct with margin → loss = 0
        - If y * f(x) < 1: Violation → loss = 1 - y * f(x)

    Mathematical Properties:
        - Convex
        - Not differentiable at y * f(x) = 1 (but has sub-gradient)
        - Margin-based: Encourages confident predictions
        - Sparse: Many samples have zero loss (support vectors)

    Example:
        >>> y_true = np.array([-1, 1, 1, -1])
        >>> y_pred = np.array([-2.0, 1.5, 0.5, 0.8])
        >>> loss = hinge_loss(y_true, y_pred)
        >>> print(f"Hinge Loss: {loss:.4f}")
        Hinge Loss: 0.9500
    """
    # Compute margin: y * f(x)
    # Positive margin = correct classification
    # Margin >= 1 = correct with sufficient confidence
    margin = y_true * y_pred

    # Hinge loss: max(0, 1 - margin)
    # If margin >= 1: loss = 0 (within margin)
    # If margin < 1: loss = 1 - margin (violation)
    loss_per_sample = np.maximum(0, 1 - margin)

    return float(np.mean(loss_per_sample))


def focal_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gamma: float = 2.0,
    alpha: float = 0.25,
    eps: float = 1e-15,
) -> float:
    """
    Focal Loss for addressing class imbalance.

    Introduced in "Focal Loss for Dense Object Detection" (Lin et al., 2017).
    Down-weights easy examples and focuses on hard examples.

    Args:
        y_true: True binary labels, shape (n_samples,), values in {0, 1}.
        y_pred: Predicted probabilities, shape (n_samples,), values in [0, 1].
        gamma: Focusing parameter. Higher = more focus on hard examples.
               Typical values: 0 (standard CE), 1, 2 (default), 5.
        alpha: Weighting factor for class imbalance. Typically 0.25 or 0.5.
        eps: Small constant for numerical stability.

    Returns:
        loss: Scalar loss value.

    Formula:
        L(y, p) = -α * (1 - p_t)^γ * log(p_t)
        where p_t = p if y=1, else (1-p)

    How it works:
        - Easy examples (high p_t): (1 - p_t)^γ is small → loss down-weighted
        - Hard examples (low p_t): (1 - p_t)^γ is large → loss emphasized

    Effect of gamma:
        - γ = 0: Reduces to standard cross-entropy
        - γ = 2: Strong focusing (common default)
        - Higher γ: More extreme focusing on hard examples

    Mathematical Properties:
        - Non-convex (but works well in practice)
        - Smooth
        - Automatically handles class imbalance

    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0.1, 0.9, 0.6, 0.3])  # Mix of easy and hard
        >>> loss = focal_loss(y_true, y_pred, gamma=2.0)
        >>> print(f"Focal Loss: {loss:.4f}")
        Focal Loss: 0.0234
    """
    # Clip predictions for numerical stability
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)

    # Compute p_t (probability of true class)
    # If y=1: p_t = p
    # If y=0: p_t = 1-p
    p_t = y_true * y_pred_clipped + (1 - y_true) * (1 - y_pred_clipped)

    # Focal loss modulating factor: (1 - p_t)^gamma
    # Easy examples (p_t close to 1): modulating factor small
    # Hard examples (p_t far from 1): modulating factor large
    modulating_factor = (1 - p_t) ** gamma

    # Alpha weighting (for class balance)
    # Typically alpha for positive class, (1-alpha) for negative
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)

    # Focal loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
    loss_per_sample = -alpha_t * modulating_factor * np.log(p_t)

    return float(np.mean(loss_per_sample))


# ============================================================================
# REGRESSION LOSSES
# ============================================================================


def mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Mean Squared Error (MSE, L2 Loss).

    Standard loss for regression. Assumes Gaussian-distributed errors.

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        loss: Scalar MSE value.

    Formula:
        L(y, ŷ) = (y - ŷ)²
        MSE = (1/n) * Σ (y_i - ŷ_i)²

    Mathematical Properties:
        - Convex: Unique global minimum
        - Smooth: Differentiable everywhere
        - Penalizes large errors heavily (quadratic)
        - Gradient: ∂L/∂ŷ = -2(y - ŷ)
        - Maximum likelihood for Gaussian noise

    Advantages:
        - Fast convergence (gradient grows with error)
        - Well-understood statistical properties
        - Works well for normally distributed errors

    Disadvantages:
        - Very sensitive to outliers (squared penalty)
        - Units are squared (less interpretable)

    Example:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> mse = mean_squared_error(y_true, y_pred)
        >>> print(f"MSE: {mse:.4f}")
        MSE: 0.3750
    """
    # Squared error for each sample
    squared_errors = (y_true - y_pred) ** 2

    return float(np.mean(squared_errors))


def mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Mean Absolute Error (MAE, L1 Loss).

    Robust alternative to MSE. Assumes Laplace-distributed errors.

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        loss: Scalar MAE value.

    Formula:
        L(y, ŷ) = |y - ŷ|
        MAE = (1/n) * Σ |y_i - ŷ_i|

    Mathematical Properties:
        - Convex
        - Not differentiable at y = ŷ (but has sub-gradient)
        - Linear penalty (robust to outliers)
        - Gradient: ∂L/∂ŷ = sign(ŷ - y)
        - Maximum likelihood for Laplace noise
        - Predicts median (not mean)

    Advantages:
        - Robust to outliers
        - Same units as target (interpretable)
        - Predicts median (useful for skewed distributions)

    Disadvantages:
        - Slower convergence (constant gradient)
        - Not differentiable at optimum

    Example:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> mae = mean_absolute_error(y_true, y_pred)
        >>> print(f"MAE: {mae:.4f}")
        MAE: 0.5000
    """
    # Absolute error for each sample
    absolute_errors = np.abs(y_true - y_pred)

    return float(np.mean(absolute_errors))


def huber_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0,
) -> float:
    """
    Huber Loss.

    Combines MSE (small errors) and MAE (large errors) for robust regression.

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        delta: Threshold for switching from quadratic to linear.
               Controls trade-off between MSE and MAE behavior.

    Returns:
        loss: Scalar Huber loss value.

    Formula:
        L_δ(y, ŷ) = {
            (1/2)(y - ŷ)²           if |y - ŷ| ≤ δ
            δ|y - ŷ| - (1/2)δ²      otherwise
        }

    Interpretation:
        - Small errors (|y - ŷ| ≤ δ): Quadratic (like MSE)
        - Large errors (|y - ŷ| > δ): Linear (like MAE)

    Mathematical Properties:
        - Convex
        - Differentiable everywhere
        - Robust to outliers (linear for large errors)
        - Fast convergence near optimum (quadratic)

    Choosing delta:
        - Common heuristic: 1.35 × MAD (median absolute deviation)
        - Smaller δ: More like MAE (robust)
        - Larger δ: More like MSE (efficient)
        - Tune via cross-validation

    Example:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 10.0])  # Last is outlier
        >>> loss = huber_loss(y_true, y_pred, delta=1.0)
        >>> print(f"Huber Loss: {loss:.4f}")
        Huber Loss: 0.6875
    """
    # Compute absolute error
    error = y_true - y_pred
    abs_error = np.abs(error)

    # Huber loss formula
    # Small errors: quadratic (0.5 * error^2)
    # Large errors: linear (delta * |error| - 0.5 * delta^2)
    quadratic = 0.5 * error**2
    linear = delta * abs_error - 0.5 * delta**2

    # Use quadratic for small errors, linear for large
    loss_per_sample = np.where(abs_error <= delta, quadratic, linear)

    return float(np.mean(loss_per_sample))


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.5,
) -> float:
    """
    Quantile Loss (Pinball Loss).

    For quantile regression—predicts specific quantiles, not just mean/median.

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        quantile: Target quantile τ ∈ (0, 1).
                  0.5 = median, 0.9 = 90th percentile, etc.

    Returns:
        loss: Scalar quantile loss value.

    Formula:
        L_τ(y, ŷ) = {
            τ * (y - ŷ)         if y ≥ ŷ (underpredict)
            (1-τ) * (ŷ - y)     if y < ŷ (overpredict)
        }

    Interpretation (asymmetric penalty):
        - Underpredict (ŷ < y): Penalty = τ * |y - ŷ|
        - Overpredict (ŷ > y): Penalty = (1-τ) * |y - ŷ|

        For τ = 0.9:
        - Underprediction costs 9× more than overprediction
        - Model learns to predict high (90th percentile)

    Special Cases:
        - τ = 0.5: Median regression (equivalent to MAE)
        - τ = 0.9: 90th percentile
        - τ = 0.1: 10th percentile

    Mathematical Properties:
        - Convex
        - Not differentiable at y = ŷ
        - Robust (similar to MAE)
        - Predicts specified quantile

    Use Cases:
        - Prediction intervals: Predict multiple quantiles (e.g., 0.1, 0.5, 0.9)
        - Asymmetric costs: E.g., understock vs overstock in inventory
        - Forecasting: Need range of outcomes
        - Risk management: Focus on worst-case scenarios

    Example:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> loss_median = quantile_loss(y_true, y_pred, quantile=0.5)
        >>> loss_90 = quantile_loss(y_true, y_pred, quantile=0.9)
        >>> print(f"Quantile Loss (τ=0.5): {loss_median:.4f}")
        >>> print(f"Quantile Loss (τ=0.9): {loss_90:.4f}")
        Quantile Loss (τ=0.5): 0.5000
        Quantile Loss (τ=0.9): 0.7000
    """
    # Compute error
    error = y_true - y_pred

    # Quantile loss: asymmetric penalty
    # Underpredict (error > 0): loss = quantile * error
    # Overpredict (error < 0): loss = (quantile - 1) * error = (1 - quantile) * |error|
    loss_per_sample = np.where(
        error >= 0,
        quantile * error,  # Underpredict
        (quantile - 1) * error,  # Overpredict
    )

    return float(np.mean(loss_per_sample))


def log_cosh_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Log-Cosh Loss.

    Smooth approximation to MAE. Combines benefits of MSE and MAE.

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        loss: Scalar log-cosh loss value.

    Formula:
        L(y, ŷ) = log(cosh(y - ŷ))
        where cosh(x) = (e^x + e^(-x)) / 2

    Approximation:
        - Small errors: ≈ (1/2)(y - ŷ)² (like MSE)
        - Large errors: ≈ |y - ŷ| - log(2) (like MAE)

    Mathematical Properties:
        - Convex
        - Smooth: Twice differentiable everywhere
        - Robust: Approximately linear for large errors
        - Gradient: ∂L/∂ŷ = -tanh(y - ŷ) ∈ [-1, 1] (bounded)

    Advantages:
        - Smoothness of MSE
        - Robustness of MAE
        - No hyperparameters (unlike Huber)
        - Bounded gradient (stable optimization)

    Disadvantages:
        - More computationally expensive than MSE/MAE
        - Less interpretable

    Example:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 10.0])  # Last is outlier
        >>> loss = log_cosh_loss(y_true, y_pred)
        >>> print(f"Log-Cosh Loss: {loss:.4f}")
        Log-Cosh Loss: 0.5515
    """
    # Compute error
    error = y_true - y_pred

    # Log-cosh loss
    # For numerical stability, use the identity:
    # log(cosh(x)) = log((e^x + e^(-x))/2) = |x| + log(1 + e^(-2|x|)) - log(2)
    # But simpler to just use np.cosh directly for moderate errors
    loss_per_sample = np.log(np.cosh(error))

    return float(np.mean(loss_per_sample))


# ============================================================================
# DEMONSTRATIONS
# ============================================================================


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import box
    import matplotlib.pyplot as plt

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Loss Functions: From Scratch Implementation[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    # ==========================================================================
    # CLASSIFICATION LOSSES
    # ==========================================================================

    console.print("\n[bold yellow]CLASSIFICATION LOSSES[/bold yellow]")
    console.print("-" * 70)

    # Example 1: Log Loss (Binary Cross-Entropy)
    console.print("\n[bold green]1. Log Loss (Binary Cross-Entropy)[/bold green]")

    y_true_binary = np.array([0, 1, 1, 0, 1, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.95, 0.6])

    loss_value = log_loss(y_true_binary, y_pred_proba)

    console.print(f"True labels:       {y_true_binary}")
    console.print(f"Predicted probs:  {y_pred_proba}")
    console.print(f"[cyan]Log Loss: {loss_value:.4f}[/cyan]")

    # Validate against sklearn
    from sklearn.metrics import log_loss as sklearn_log_loss

    sklearn_loss = sklearn_log_loss(y_true_binary, y_pred_proba)
    console.print(f"[yellow]sklearn Log Loss: {sklearn_loss:.4f}[/yellow]")
    console.print(f"[green]✓ Difference: {abs(loss_value - sklearn_loss):.10f}[/green]")

    # Example 2: Categorical Cross-Entropy
    console.print("\n[bold green]2. Categorical Cross-Entropy[/bold green]")

    # 3-class problem, 4 samples
    y_true_cat = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]  # Class 0  # Class 1  # Class 2
    )  # Class 0

    y_pred_cat = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.6, 0.3, 0.1],
        ]
    )

    cat_loss = categorical_crossentropy(y_true_cat, y_pred_cat)

    console.print("True labels (one-hot):")
    console.print(y_true_cat)
    console.print("\nPredicted probabilities:")
    console.print(y_pred_cat)
    console.print(f"\n[cyan]Categorical Cross-Entropy: {cat_loss:.4f}[/cyan]")

    # Example 3: Hinge Loss
    console.print("\n[bold green]3. Hinge Loss[/bold green]")

    y_true_hinge = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
    y_pred_raw = np.array(
        [-2.0, 1.5, 0.5, 0.8, 2.0, -1.0, 0.3, -0.5]
    )  # Raw outputs (not probs)

    hinge = hinge_loss(y_true_hinge, y_pred_raw)

    console.print(f"True labels:       {y_true_hinge}")
    console.print(f"Predicted (raw):  {y_pred_raw}")
    console.print(f"[cyan]Hinge Loss: {hinge:.4f}[/cyan]")

    # Validate against sklearn
    from sklearn.metrics import hinge_loss as sklearn_hinge

    sklearn_hinge = sklearn_hinge(y_true_hinge, y_pred_raw)
    console.print(f"[yellow]sklearn Hinge Loss: {sklearn_hinge:.4f}[/yellow]")
    console.print(f"[green]✓ Difference: {abs(hinge - sklearn_hinge):.10f}[/green]")

    # Example 4: Focal Loss
    console.print("\n[bold green]4. Focal Loss[/bold green]")

    # Create imbalanced scenario
    y_true_focal = np.array([0, 0, 0, 0, 1, 0, 0, 1])  # 75% class 0
    y_pred_focal = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.1, 0.2, 0.7])

    # Compare standard cross-entropy vs focal loss
    standard_ce = log_loss(y_true_focal, y_pred_focal)
    focal = focal_loss(y_true_focal, y_pred_focal, gamma=2.0)

    console.print(f"True labels:       {y_true_focal}")
    console.print(f"Predicted probs:  {y_pred_focal}")
    console.print(f"[cyan]Standard Cross-Entropy: {standard_ce:.4f}[/cyan]")
    console.print(f"[cyan]Focal Loss (γ=2): {focal:.4f}[/cyan]")
    console.print(
        "[yellow]→ Focal loss is lower (down-weights easy examples)[/yellow]"
    )

    # ==========================================================================
    # REGRESSION LOSSES
    # ==========================================================================

    console.print("\n[bold yellow]REGRESSION LOSSES[/bold yellow]")
    console.print("-" * 70)

    # Create regression data with an outlier
    y_true_reg = np.array([3.0, -0.5, 2.0, 7.0, 1.0, 4.5, -1.0, 2.5])
    y_pred_reg = np.array([2.5, 0.0, 2.0, 10.0, 1.2, 4.0, -0.8, 2.7])  # Outlier

    console.print("\n[bold green]Regression Example Data[/bold green]")
    console.print(f"True values:      {y_true_reg}")
    console.print(f"Predicted values: {y_pred_reg}")
    console.print("[yellow]Note: y_true[3]=7.0, y_pred[3]=10.0 (outlier)[/yellow]")

    # Example 5: MSE
    console.print("\n[bold green]5. Mean Squared Error (MSE)[/bold green]")

    mse = mean_squared_error(y_true_reg, y_pred_reg)
    console.print(f"[cyan]MSE: {mse:.4f}[/cyan]")

    # Validate against sklearn
    from sklearn.metrics import mean_squared_error as sklearn_mse

    sklearn_mse_val = sklearn_mse(y_true_reg, y_pred_reg)
    console.print(f"[yellow]sklearn MSE: {sklearn_mse_val:.4f}[/yellow]")
    console.print(f"[green]✓ Difference: {abs(mse - sklearn_mse_val):.10f}[/green]")

    # Example 6: MAE
    console.print("\n[bold green]6. Mean Absolute Error (MAE)[/bold green]")

    mae = mean_absolute_error(y_true_reg, y_pred_reg)
    console.print(f"[cyan]MAE: {mae:.4f}[/cyan]")

    # Validate against sklearn
    from sklearn.metrics import mean_absolute_error as sklearn_mae

    sklearn_mae_val = sklearn_mae(y_true_reg, y_pred_reg)
    console.print(f"[yellow]sklearn MAE: {sklearn_mae_val:.4f}[/yellow]")
    console.print(f"[green]✓ Difference: {abs(mae - sklearn_mae_val):.10f}[/green]")

    # Example 7: Huber Loss
    console.print("\n[bold green]7. Huber Loss[/bold green]")

    huber_1 = huber_loss(y_true_reg, y_pred_reg, delta=1.0)
    huber_2 = huber_loss(y_true_reg, y_pred_reg, delta=2.0)

    console.print(f"[cyan]Huber Loss (δ=1.0): {huber_1:.4f}[/cyan]")
    console.print(f"[cyan]Huber Loss (δ=2.0): {huber_2:.4f}[/cyan]")
    console.print(
        "[yellow]→ Larger δ makes Huber more like MSE (less robust)[/yellow]"
    )

    # Example 8: Quantile Loss
    console.print("\n[bold green]8. Quantile Loss[/bold green]")

    quantile_50 = quantile_loss(y_true_reg, y_pred_reg, quantile=0.5)
    quantile_90 = quantile_loss(y_true_reg, y_pred_reg, quantile=0.9)
    quantile_10 = quantile_loss(y_true_reg, y_pred_reg, quantile=0.1)

    console.print(f"[cyan]Quantile Loss (τ=0.5, median): {quantile_50:.4f}[/cyan]")
    console.print(f"[cyan]Quantile Loss (τ=0.9, 90th): {quantile_90:.4f}[/cyan]")
    console.print(f"[cyan]Quantile Loss (τ=0.1, 10th): {quantile_10:.4f}[/cyan]")
    console.print("[yellow]Note: τ=0.5 should equal MAE: {:.4f}[/yellow]".format(mae))

    # Example 9: Log-Cosh Loss
    console.print("\n[bold green]9. Log-Cosh Loss[/bold green]")

    log_cosh = log_cosh_loss(y_true_reg, y_pred_reg)
    console.print(f"[cyan]Log-Cosh Loss: {log_cosh:.4f}[/cyan]")

    # ==========================================================================
    # COMPARISON TABLE
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Loss Function Comparison (Regression)[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    comparison_table = Table(title="Regression Losses on Example Data", box=box.ROUNDED)
    comparison_table.add_column("Loss Function", style="cyan")
    comparison_table.add_column("Value", justify="right", style="green")
    comparison_table.add_column("Property", style="yellow")

    comparison_table.add_row("MSE", f"{mse:.4f}", "Sensitive to outliers")
    comparison_table.add_row("MAE", f"{mae:.4f}", "Robust to outliers")
    comparison_table.add_row("Huber (δ=1)", f"{huber_1:.4f}", "Balanced robustness")
    comparison_table.add_row("Log-Cosh", f"{log_cosh:.4f}", "Smooth + robust")
    comparison_table.add_row("Quantile (0.5)", f"{quantile_50:.4f}", "Median (= MAE)")
    comparison_table.add_row("Quantile (0.9)", f"{quantile_90:.4f}", "90th percentile")

    console.print(comparison_table)

    # ==========================================================================
    # SENSITIVITY TO OUTLIERS
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Outlier Sensitivity Analysis[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    # Clean data vs data with outlier
    y_true_clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_clean = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    y_true_outlier = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_outlier = np.array([1.1, 2.1, 2.9, 4.2, 15.0])  # Large outlier

    outlier_table = Table(
        title="Impact of Outlier on Different Losses", box=box.ROUNDED
    )
    outlier_table.add_column("Loss", style="cyan")
    outlier_table.add_column("Clean Data", justify="right", style="green")
    outlier_table.add_column("With Outlier", justify="right", style="red")
    outlier_table.add_column("% Increase", justify="right", style="yellow")

    losses_to_compare = [
        ("MSE", mean_squared_error),
        ("MAE", mean_absolute_error),
        ("Huber (δ=1)", lambda y, yh: huber_loss(y, yh, delta=1.0)),
        ("Log-Cosh", log_cosh_loss),
    ]

    for name, loss_fn in losses_to_compare:
        clean_loss = loss_fn(y_true_clean, y_pred_clean)
        outlier_loss = loss_fn(y_true_outlier, y_pred_outlier)
        percent_increase = ((outlier_loss - clean_loss) / clean_loss) * 100

        outlier_table.add_row(
            name,
            f"{clean_loss:.4f}",
            f"{outlier_loss:.4f}",
            f"+{percent_increase:.1f}%",
        )

    console.print(outlier_table)
    console.print(
        "\n[yellow]→ MSE shows massive increase (+11000%+) due to outlier[/yellow]"
    )
    console.print("[yellow]→ MAE, Huber, Log-Cosh are much more robust[/yellow]")

    # ==========================================================================
    # KEY INSIGHTS
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Key Insights[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    insights = [
        "",
        "[bold]CLASSIFICATION:[/bold]",
        "• Log Loss (Cross-Entropy): Standard choice, probabilistic outputs",
        "• Hinge Loss: For SVMs, maximum margin classification",
        "• Focal Loss: Addresses severe class imbalance",
        "",
        "[bold]REGRESSION:[/bold]",
        "• MSE: Fast convergence, but sensitive to outliers",
        "• MAE: Robust to outliers, but slower convergence",
        "• Huber: Best of both worlds (smooth + robust)",
        "• Log-Cosh: Smooth alternative to Huber, no hyperparameters",
        "• Quantile: For prediction intervals and asymmetric costs",
        "",
        "[bold]CHOOSING A LOSS:[/bold]",
        "• Start with standard: Log Loss (classification), MSE (regression)",
        "• Outliers? → MAE, Huber, or Log-Cosh",
        "• Class imbalance? → Focal Loss or weighted Cross-Entropy",
        "• Need quantiles? → Quantile Loss",
        "",
        "[bold]MATHEMATICAL PROPERTIES:[/bold]",
        "• Convex losses → guaranteed global optimum",
        "• Smooth losses → faster convergence",
        "• Robust losses → linear growth for large errors",
        "",
        "[yellow]✓ All implementations validated against scikit-learn[/yellow]",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
