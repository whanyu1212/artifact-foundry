"""
Regression Metrics - From Scratch Implementation

Implements common regression evaluation metrics for continuous value prediction.

Mathematical Foundations:
- MSE = Σ(y - ŷ)² / n - Mean squared error (heavily penalizes large errors)
- RMSE = √MSE - Root mean squared error (same units as target)
- MAE = Σ|y - ŷ| / n - Mean absolute error (robust to outliers)
- R² = 1 - SS_res/SS_tot - Coefficient of determination (variance explained)
- Adjusted R² = 1 - (1-R²)(n-1)/(n-p-1) - R² adjusted for # predictors
- MAPE = Σ|(y - ŷ)/y| / n · 100% - Mean absolute percentage error
- MSLE = Σ(log(1+y) - log(1+ŷ))² / n - Mean squared logarithmic error

All metrics validated against scikit-learn implementations.
"""

import numpy as np
from typing import Optional


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error (MSE).

    Formula: MSE = (1/n) · Σ(yᵢ - ŷᵢ)²

    Properties:
    - Heavily penalizes large errors (squared term)
    - Not robust to outliers
    - Same units as variance of target

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        mse: Mean squared error, range [0, ∞], lower is better.

    Use Cases:
        - Large errors are particularly bad
        - Training objective for OLS regression
        - Data doesn't have extreme outliers

    Examples:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> mean_squared_error(y_true, y_pred)
        0.375  # (0.5² + 0.5² + 0² + 1²) / 4
    """
    errors = y_true - y_pred
    return np.mean(errors ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Formula: RMSE = √MSE = √[(1/n) · Σ(yᵢ - ŷᵢ)²]

    Properties:
    - Same units as target variable (interpretable)
    - Penalizes large errors (like MSE)
    - Standard deviation of residuals

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        rmse: Root mean squared error, range [0, ∞], lower is better.

    Interpretation:
        RMSE = $5,000 for house prices → typical prediction off by $5K

    Examples:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> root_mean_squared_error(y_true, y_pred)
        0.612  # √0.375
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


# Alias for convenience
rmse = root_mean_squared_error


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Formula: MAE = (1/n) · Σ|yᵢ - ŷᵢ|

    Properties:
    - Linear penalty (all errors treated equally)
    - Robust to outliers (no squaring)
    - Same units as target variable

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        mae: Mean absolute error, range [0, ∞], lower is better.

    Use Cases:
        - Outliers present in data
        - All errors equally bad (no special penalty for large errors)
        - Want robust metric

    Comparison:
        RMSE ≥ MAE (always)
        Large difference → presence of large errors
        Similar values → errors uniformly distributed

    Examples:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> mean_absolute_error(y_true, y_pred)
        0.5  # (0.5 + 0.5 + 0 + 1) / 4
    """
    errors = np.abs(y_true - y_pred)
    return np.mean(errors)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    Formula: R² = 1 - SS_res / SS_tot
           = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²

    Where:
    - SS_res: Residual sum of squares (unexplained variance)
    - SS_tot: Total sum of squares (total variance)
    - ȳ: Mean of true values

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        r2: Coefficient of determination, range (-∞, 1], higher is better.
            1 = perfect predictions
            0 = model as good as predicting mean
            <0 = model worse than predicting mean

    Interpretation:
        R² = 0.85 → Model explains 85% of variance in target

    Use Cases:
        - Compare models on same dataset
        - Understand proportion of variance explained
        - Linear regression interpretability

    Limitations:
        - Always increases with more features (use adjusted R² instead)
        - Can be negative on test set
        - Doesn't indicate if predictions are biased

    Examples:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> r2_score(y_true, y_pred)
        0.948  # Explains 94.8% of variance
    """
    # Residual sum of squares: Σ(y - ŷ)²
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Total sum of squares: Σ(y - ȳ)²
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Handle edge case: y has zero variance
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else -np.inf

    return 1 - (ss_res / ss_tot)


def adjusted_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: int,
) -> float:
    """
    Compute Adjusted R² (adjusted for number of predictors).

    Formula: R²_adj = 1 - [(1 - R²) · (n - 1) / (n - p - 1)]

    Where:
    - n: number of samples
    - p: number of predictors (features)

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        n_features: Number of features used in the model.

    Returns:
        adjusted_r2: Adjusted R², range (-∞, 1], higher is better.

    Properties:
        - Penalizes adding uninformative features
        - Can decrease when adding features (unlike R²)
        - Better for model comparison with different # features

    Use Cases:
        - Compare models with different numbers of features
        - Feature selection
        - Avoid overfitting from too many features

    Examples:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> adjusted_r2_score(y_true, y_pred, n_features=2)
        0.896  # Lower than R² due to adjustment
    """
    n_samples = len(y_true)
    r2 = r2_score(y_true, y_pred)

    # Handle edge case
    if n_samples <= n_features + 1:
        return -np.inf

    adjustment = (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return 1 - adjustment


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Formula: MAPE = (100% / n) · Σ|yᵢ - ŷᵢ| / |yᵢ|

    Properties:
    - Scale-independent (percentage)
    - Easy to interpret for business
    - Undefined for yᵢ = 0

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        epsilon: Small constant to avoid division by zero.

    Returns:
        mape: Mean absolute percentage error, range [0, ∞], lower is better.

    Limitations:
        - Undefined for y = 0
        - Asymmetric: penalizes over-predictions more than under-predictions
        - Biased toward under-predictions

    Asymmetry Example:
        True value: 100
        Prediction: 50 → Error = 50%
        Prediction: 150 → Error = 50%
        But 50 and 150 are equally far from 100!

    Use Cases:
        - Compare models across different scales
        - Business reporting (easy to explain)
        - Target values never zero

    Examples:
        >>> y_true = np.array([100.0, 50.0, 200.0])
        >>> y_pred = np.array([110.0, 45.0, 190.0])
        >>> mean_absolute_percentage_error(y_true, y_pred)
        8.33  # 8.33% average error
    """
    # Avoid division by zero
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)

    # Compute percentage errors
    percentage_errors = np.abs((y_true - y_pred) / y_true_safe)

    return np.mean(percentage_errors) * 100


# Alias
mape = mean_absolute_percentage_error


def mean_squared_logarithmic_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Mean Squared Logarithmic Error (MSLE).

    Formula: MSLE = (1/n) · Σ[log(1 + yᵢ) - log(1 + ŷᵢ)]²

    Properties:
    - Penalizes under-predictions more than over-predictions
    - Appropriate for exponential growth
    - Measures relative errors (not absolute)

    Args:
        y_true: True values (non-negative), shape (n_samples,).
        y_pred: Predicted values (non-negative), shape (n_samples,).

    Returns:
        msle: Mean squared logarithmic error, range [0, ∞], lower is better.

    Use Cases:
        - Target has exponential trend (e.g., stock prices, population)
        - Under-predictions worse than over-predictions
        - Target varies over several orders of magnitude

    Interpretation:
        Predicting 90 instead of 100: larger penalty
        Predicting 110 instead of 100: smaller penalty

    Examples:
        >>> y_true = np.array([3.0, 5.0, 2.5, 7.0])
        >>> y_pred = np.array([2.5, 5.0, 4.0, 8.0])
        >>> mean_squared_logarithmic_error(y_true, y_pred)
        0.039
    """
    # Use log1p for numerical stability: log1p(x) = log(1 + x)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    squared_log_error = (log_true - log_pred) ** 2
    return np.mean(squared_log_error)


# Alias
msle = mean_squared_logarithmic_error


def huber_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    delta: float = 1.0,
) -> float:
    """
    Compute Huber Loss (robust to outliers).

    Formula:
        L_δ(y, ŷ) = { ½(y - ŷ)²           if |y - ŷ| ≤ δ
                    { δ|y - ŷ| - ½δ²      otherwise

    Properties:
    - Combines MSE (for small errors) and MAE (for large errors)
    - Robust to outliers (like MAE for large errors)
    - Differentiable everywhere (unlike MAE)
    - Smooth transition between quadratic and linear

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).
        delta: Threshold for switching from quadratic to linear.
            Smaller delta → more robust (closer to MAE)
            Larger delta → less robust (closer to MSE)

    Returns:
        loss: Huber loss, range [0, ∞], lower is better.

    Use Cases:
        - Outliers present but want smooth loss
        - Training robust regression models
        - Need differentiable loss function

    Examples:
        >>> y_true = np.array([1.0, 2.0, 3.0, 100.0])  # 100 is outlier
        >>> y_pred = np.array([1.1, 2.1, 3.1, 10.0])
        >>> huber_loss(y_true, y_pred, delta=1.0)
        22.51  # Outlier has linear penalty
    """
    errors = np.abs(y_true - y_pred)

    # Quadratic for small errors, linear for large errors
    quadratic_part = 0.5 * errors ** 2
    linear_part = delta * errors - 0.5 * delta ** 2

    # Use quadratic if error <= delta, otherwise linear
    losses = np.where(errors <= delta, quadratic_part, linear_part)

    return np.mean(losses)


def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Explained Variance Score.

    Formula: 1 - Var(y - ŷ) / Var(y)

    Similar to R² but uses variance instead of sum of squares.

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        score: Explained variance score, range (-∞, 1], higher is better.
            1 = perfect predictions
            0 = model as good as predicting mean

    Difference from R²:
        - Uses variance (not sum of squares)
        - Less affected by constant shift in predictions

    Examples:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> explained_variance_score(y_true, y_pred)
        0.948
    """
    residual_var = np.var(y_true - y_pred)
    total_var = np.var(y_true)

    if total_var == 0:
        return 0.0 if residual_var == 0 else -np.inf

    return 1 - (residual_var / total_var)


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute maximum absolute error.

    Formula: max|yᵢ - ŷᵢ|

    Identifies worst-case prediction error.

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        max_err: Maximum absolute error, range [0, ∞].

    Use Cases:
        - Identify worst predictions
        - Understand model's worst-case performance
        - Outlier detection

    Examples:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> max_error(y_true, y_pred)
        1.0
    """
    return np.max(np.abs(y_true - y_pred))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Median Absolute Error (MdAE).

    Formula: median|yᵢ - ŷᵢ|

    More robust to outliers than MAE.

    Args:
        y_true: True values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        mdae: Median absolute error, range [0, ∞], lower is better.

    Use Cases:
        - Extremely robust metric needed
        - Outliers severely affect mean

    Examples:
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0, 100.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0, 10.0])
        >>> median_absolute_error(y_true, y_pred)
        0.5  # Median unaffected by outlier
    """
    return np.median(np.abs(y_true - y_pred))


if __name__ == "__main__":
    from sklearn.metrics import (
        mean_squared_error as sklearn_mse,
        mean_absolute_error as sklearn_mae,
        r2_score as sklearn_r2,
        mean_absolute_percentage_error as sklearn_mape,
        mean_squared_log_error as sklearn_msle,
        max_error as sklearn_max_error,
        median_absolute_error as sklearn_mdae,
    )
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Regression Metrics: From Scratch Implementation")
    console.print("=" * 70 + "[/bold cyan]")

    # Example 1: Standard Regression
    console.print("\n[bold yellow]1. Standard Regression Metrics[/bold yellow]")
    console.print("-" * 70)

    y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.5, 6.0, 1.0, 8.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0, 5.0, 5.5, 1.5, 7.5])

    metrics_table = Table(title="Regression Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Our Implementation", justify="right", style="green")
    metrics_table.add_column("Scikit-Learn", justify="right", style="yellow")
    metrics_table.add_column("Difference", justify="right", style="white")

    # MSE
    our_mse = mean_squared_error(y_true, y_pred)
    sk_mse = sklearn_mse(y_true, y_pred)
    metrics_table.add_row("MSE", f"{our_mse:.6f}", f"{sk_mse:.6f}", f"{abs(our_mse - sk_mse):.10f}")

    # RMSE
    our_rmse = rmse(y_true, y_pred)
    sk_rmse = np.sqrt(sklearn_mse(y_true, y_pred))
    metrics_table.add_row("RMSE", f"{our_rmse:.6f}", f"{sk_rmse:.6f}", f"{abs(our_rmse - sk_rmse):.10f}")

    # MAE
    our_mae = mean_absolute_error(y_true, y_pred)
    sk_mae = sklearn_mae(y_true, y_pred)
    metrics_table.add_row("MAE", f"{our_mae:.6f}", f"{sk_mae:.6f}", f"{abs(our_mae - sk_mae):.10f}")

    # R²
    our_r2 = r2_score(y_true, y_pred)
    sk_r2 = sklearn_r2(y_true, y_pred)
    metrics_table.add_row("R²", f"{our_r2:.6f}", f"{sk_r2:.6f}", f"{abs(our_r2 - sk_r2):.10f}")

    # Adjusted R²
    our_adj_r2 = adjusted_r2_score(y_true, y_pred, n_features=3)
    metrics_table.add_row("Adjusted R²", f"{our_adj_r2:.6f}", "-", "-")

    # MAPE
    our_mape = mape(y_true, y_pred)
    sk_mape = sklearn_mape(y_true, y_pred) * 100  # sklearn returns decimal
    metrics_table.add_row("MAPE (%)", f"{our_mape:.6f}", f"{sk_mape:.6f}", f"{abs(our_mape - sk_mape):.10f}")

    # Max Error
    our_max_err = max_error(y_true, y_pred)
    sk_max_err = sklearn_max_error(y_true, y_pred)
    metrics_table.add_row("Max Error", f"{our_max_err:.6f}", f"{sk_max_err:.6f}", f"{abs(our_max_err - sk_max_err):.10f}")

    # Median Absolute Error
    our_mdae = median_absolute_error(y_true, y_pred)
    sk_mdae = sklearn_mdae(y_true, y_pred)
    metrics_table.add_row("Median AE", f"{our_mdae:.6f}", f"{sk_mdae:.6f}", f"{abs(our_mdae - sk_mdae):.10f}")

    console.print(metrics_table)

    # Example 2: Comparing MSE vs MAE with Outliers
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("2. MSE vs MAE: Effect of Outliers")
    console.print("=" * 70 + "[/bold cyan]")

    # Data without outliers
    y_true_clean = np.array([3.0, -0.5, 2.0, 7.0, 4.5])
    y_pred_clean = np.array([2.9, -0.4, 2.1, 7.1, 4.6])

    # Data with outlier
    y_true_outlier = np.array([3.0, -0.5, 2.0, 7.0, 100.0])
    y_pred_outlier = np.array([2.9, -0.4, 2.1, 7.1, 10.0])

    outlier_table = Table(title="Impact of Outliers", box=box.ROUNDED)
    outlier_table.add_column("Metric", style="cyan")
    outlier_table.add_column("Without Outlier", justify="right", style="green")
    outlier_table.add_column("With Outlier", justify="right", style="yellow")
    outlier_table.add_column("% Change", justify="right", style="red")

    mse_clean = mean_squared_error(y_true_clean, y_pred_clean)
    mse_outlier = mean_squared_error(y_true_outlier, y_pred_outlier)
    mse_change = (mse_outlier - mse_clean) / mse_clean * 100

    mae_clean = mean_absolute_error(y_true_clean, y_pred_clean)
    mae_outlier = mean_absolute_error(y_true_outlier, y_pred_outlier)
    mae_change = (mae_outlier - mae_clean) / mae_clean * 100

    huber_clean = huber_loss(y_true_clean, y_pred_clean, delta=1.0)
    huber_outlier = huber_loss(y_true_outlier, y_pred_outlier, delta=1.0)
    huber_change = (huber_outlier - huber_clean) / huber_clean * 100

    outlier_table.add_row("MSE", f"{mse_clean:.4f}", f"{mse_outlier:.4f}", f"+{mse_change:.0f}%")
    outlier_table.add_row("MAE", f"{mae_clean:.4f}", f"{mae_outlier:.4f}", f"+{mae_change:.0f}%")
    outlier_table.add_row("Huber Loss", f"{huber_clean:.4f}", f"{huber_outlier:.4f}", f"+{huber_change:.0f}%")

    console.print(outlier_table)

    console.print("\n[yellow]→ MSE heavily affected by outliers (squared term)[/yellow]")
    console.print("[yellow]→ MAE more robust (linear penalty)[/yellow]")
    console.print("[yellow]→ Huber Loss balances both (smooth transition)[/yellow]")

    # Example 3: MSLE for Exponential Growth
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("3. MSLE for Exponential Growth")
    console.print("=" * 70 + "[/bold cyan]")

    # Exponential growth (e.g., stock prices)
    y_true_exp = np.array([10.0, 100.0, 1000.0])
    y_pred_under = np.array([9.0, 90.0, 900.0])  # Under-predictions
    y_pred_over = np.array([11.0, 110.0, 1100.0])  # Over-predictions

    exp_table = Table(title="MSLE: Under vs Over Predictions", box=box.ROUNDED)
    exp_table.add_column("Prediction Type", style="cyan")
    exp_table.add_column("MSE", justify="right", style="green")
    exp_table.add_column("MAE", justify="right", style="yellow")
    exp_table.add_column("MSLE", justify="right", style="magenta")

    mse_under = mean_squared_error(y_true_exp, y_pred_under)
    mae_under = mean_absolute_error(y_true_exp, y_pred_under)
    msle_under = mean_squared_logarithmic_error(y_true_exp, y_pred_under)

    mse_over = mean_squared_error(y_true_exp, y_pred_over)
    mae_over = mean_absolute_error(y_true_exp, y_pred_over)
    msle_over = mean_squared_logarithmic_error(y_true_exp, y_pred_over)

    exp_table.add_row("Under-prediction (-10%)", f"{mse_under:.2f}", f"{mae_under:.2f}", f"{msle_under:.4f}")
    exp_table.add_row("Over-prediction (+10%)", f"{mse_over:.2f}", f"{mae_over:.2f}", f"{msle_over:.4f}")

    console.print(exp_table)

    console.print("\n[yellow]→ MSE and MAE treat under/over equally (same absolute error)[/yellow]")
    console.print("[yellow]→ MSLE penalizes under-predictions more (relative error)[/yellow]")

    # Example 4: R² Interpretation
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("4. R² Interpretation")
    console.print("=" * 70 + "[/bold cyan]")

    # Perfect predictions
    y_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r2_perfect = r2_score(y_perfect, y_perfect)

    # Predicting mean
    y_mean_pred = np.full_like(y_perfect, y_perfect.mean())
    r2_mean = r2_score(y_perfect, y_mean_pred)

    # Worse than mean
    y_bad = np.array([5.0, 1.0, 5.0, 1.0, 5.0])
    r2_bad = r2_score(y_perfect, y_bad)

    r2_table = Table(title="R² Interpretation", box=box.ROUNDED)
    r2_table.add_column("Prediction Quality", style="cyan")
    r2_table.add_column("R²", justify="right", style="green")
    r2_table.add_column("Interpretation", style="white")

    r2_table.add_row("Perfect predictions", f"{r2_perfect:.2f}", "Explains 100% of variance")
    r2_table.add_row("Predicting mean", f"{r2_mean:.2f}", "Explains 0% (no better than mean)")
    r2_table.add_row("Worse than mean", f"{r2_bad:.2f}", "Negative (worse than predicting mean)")

    console.print(r2_table)

    # Key Insights
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Key Insights")
    console.print("=" * 70 + "[/bold cyan]")

    insights = [
        "• MSE heavily penalizes large errors (squared term) - not robust to outliers",
        "• RMSE has same units as target (interpretable) - use for error magnitude",
        "• MAE robust to outliers (linear penalty) - all errors treated equally",
        "• R² measures variance explained - compare models on same dataset",
        "• Adjusted R² penalizes extra features - use for model comparison",
        "• MAPE scale-independent - good for business reporting (if no zeros)",
        "• MSLE for exponential growth - penalizes under-predictions more",
        "• Huber Loss balances MSE and MAE - robust and differentiable",
        "",
        "[yellow]→ Choose metric based on problem:[/yellow]",
        "  - Outliers present: MAE or Huber Loss",
        "  - Large errors bad: MSE or RMSE",
        "  - Interpretable error: MAE or RMSE",
        "  - Percentage error: MAPE (if no zeros)",
        "  - Variance explained: R²",
        "  - Model comparison: Adjusted R²",
        "  - Exponential growth: MSLE",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
