"""
Linear Regression Implementation

Implements linear regression with two solution methods:
1. Normal Equations (closed-form): w = (X^T X)^-1 X^T y
2. Gradient Descent (iterative): w = w - α * ∇L

Mathematical Foundation:
- Model: ŷ = X @ w (linear combination of features)
- Loss: MSE = (1/n) * ||y - X@w||²
- Gradient: ∇L = -(2/n) * X^T @ (y - X@w)

The normal equations provide an exact solution but have O(d³) complexity.
Gradient descent is iterative but scales better to high dimensions.
"""

import numpy as np
from typing import Literal, Optional


class LinearRegression:
    """
    Linear Regression using normal equations or gradient descent.

    Linear regression models the relationship between features X and target y
    as a linear function: ŷ = X @ w, where w are the learned weights.

    Attributes:
        weights_ (np.ndarray): Learned weight vector including bias term.
            Shape: (n_features + 1,) where first element is bias.
        n_iterations_ (int): Number of iterations performed (gradient descent only).
        losses_ (list[float]): Loss history during training (gradient descent only).

    Mathematical Notes:
        Normal equations solve: X^T X w = X^T y directly
        Gradient descent iteratively updates: w := w - α * ∇L
        where ∇L = -(2/n) * X^T @ (y - ŷ)
    """

    def __init__(
        self,
        method: Literal["normal_equations", "gradient_descent"] = "normal_equations",
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> None:
        """
        Initialize Linear Regression.

        Args:
            method: Solution method to use.
                - "normal_equations": Closed-form solution (exact, one-step)
                - "gradient_descent": Iterative optimization
            learning_rate: Step size for gradient descent (α in update rule).
                Larger values converge faster but may overshoot minimum.
            max_iterations: Maximum iterations for gradient descent.
                Typical values: 1000-10000 depending on convergence.
            tolerance: Convergence threshold for gradient descent.
                Stop if change in loss < tolerance.
            verbose: If True, print progress during gradient descent.

        Notes:
            - Normal equations: Fast for small d (<10k features), requires invertible X^T X
            - Gradient descent: Scales to large d, works even with singular X^T X
        """
        self.method = method
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

        # Learned parameters (set during fit)
        self.weights_: Optional[np.ndarray] = None
        self.n_iterations_: int = 0
        self.losses_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit linear regression model.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).

        Returns:
            self: Fitted model instance.

        Raises:
            ValueError: If X and y have incompatible shapes.
            np.linalg.LinAlgError: If X^T X is singular (normal equations only).

        Notes:
            Automatically adds bias column (all ones) to X internally.
            After fitting, self.weights_[0] is bias, self.weights_[1:] are feature weights.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same n_samples: {X.shape[0]} vs {y.shape[0]}")

        # Add bias column (prepend column of ones)
        # Shape: (n_samples, n_features + 1)
        X_with_bias = self._add_bias(X)

        if self.method == "normal_equations":
            self.weights_ = self._fit_normal_equations(X_with_bias, y)
        elif self.method == "gradient_descent":
            self.weights_ = self._fit_gradient_descent(X_with_bias, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Args:
            X: Features to predict on, shape (n_samples, n_features).

        Returns:
            predictions: Predicted values, shape (n_samples,).

        Raises:
            RuntimeError: If model has not been fitted yet.

        Formula:
            ŷ = X @ w = w₀ + w₁x₁ + w₂x₂ + ... + wₐxₐ
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        X_with_bias = self._add_bias(X)
        return X_with_bias @ self.weights_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score (coefficient of determination).

        Args:
            X: Features, shape (n_samples, n_features).
            y: True targets, shape (n_samples,).

        Returns:
            r2_score: R² value in range (-∞, 1]. Higher is better.
                - R² = 1: Perfect predictions
                - R² = 0: Model no better than predicting mean
                - R² < 0: Model worse than predicting mean

        Formula:
            R² = 1 - (SS_res / SS_tot)
            where SS_res = Σ(y - ŷ)² (residual sum of squares)
                  SS_tot = Σ(y - ȳ)² (total sum of squares)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # Avoid division by zero if all y values are identical
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)

    def _fit_normal_equations(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve using normal equations: w = (X^T X)^-1 X^T y

        Args:
            X: Features with bias column, shape (n_samples, n_features+1).
            y: Targets, shape (n_samples,).

        Returns:
            weights: Optimal weight vector, shape (n_features+1,).

        Complexity: O(n*d² + d³) where n=n_samples, d=n_features
            - X^T @ X: O(n*d²)
            - Matrix inversion: O(d³)

        Notes:
            Uses np.linalg.lstsq for numerical stability (better than explicit inverse).
            lstsq handles singular matrices using SVD decomposition.
        """
        # Solve X^T X w = X^T y using least squares
        # More numerically stable than computing inverse explicitly
        weights, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        return weights

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve using batch gradient descent.

        Args:
            X: Features with bias column, shape (n_samples, n_features+1).
            y: Targets, shape (n_samples,).

        Returns:
            weights: Optimized weight vector, shape (n_features+1,).

        Algorithm:
            1. Initialize weights to zeros
            2. Repeat until convergence or max_iterations:
                a. Compute predictions: ŷ = X @ w
                b. Compute gradient: ∇L = -(2/n) * X^T @ (y - ŷ)
                c. Update weights: w := w - α * ∇L
                d. Check convergence: |loss_new - loss_old| < tolerance

        Notes:
            Gradient ∇L = -(2/n) * X^T @ (y - X@w) points in direction of steepest ascent.
            We subtract it (gradient descent) to minimize loss.
        """
        n_samples, n_features = X.shape

        # Initialize weights to zeros
        weights = np.zeros(n_features)

        # Track loss history for convergence monitoring
        prev_loss = float('inf')

        for iteration in range(self.max_iterations):
            # Compute predictions
            y_pred = X @ weights

            # Compute MSE loss: (1/n) * ||y - ŷ||²
            loss = np.mean((y - y_pred) ** 2)
            self.losses_.append(loss)

            # Check convergence: if loss change is tiny, stop early
            if abs(prev_loss - loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration}, loss={loss:.6f}")
                break

            # Compute gradient: ∇L = -(2/n) * X^T @ (y - ŷ)
            # Note: We can ignore the factor of 2 by adjusting learning rate
            gradient = -(2 / n_samples) * X.T @ (y - y_pred)

            # Update weights: w := w - α * ∇L
            weights -= self.learning_rate * gradient

            prev_loss = loss

            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iterations}, Loss: {loss:.6f}")

        self.n_iterations_ = iteration + 1
        return weights

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        """
        Add bias column (all ones) to feature matrix.

        Args:
            X: Original features, shape (n_samples, n_features).

        Returns:
            X_with_bias: Features with bias column, shape (n_samples, n_features+1).
                First column is all ones (for bias term w₀).

        Example:
            >>> X = np.array([[1, 2], [3, 4]])
            >>> _add_bias(X)
            array([[1., 1., 2.],
                   [1., 3., 4.]])
        """
        n_samples = X.shape[0]
        # Prepend column of ones: [1, x₁, x₂, ..., xₐ]
        return np.hstack([np.ones((n_samples, 1)), X])


if __name__ == "__main__":
    # Example usage and comparison of methods
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("=" * 70)
    print("Linear Regression: Normal Equations vs Gradient Descent")
    print("=" * 70)

    # Method 1: Normal Equations
    lr_normal = LinearRegression(method="normal_equations")
    lr_normal.fit(X_train, y_train)
    y_pred_normal = lr_normal.predict(X_test)

    print("\n1. Normal Equations (Closed-Form Solution)")
    print("-" * 70)
    print(f"Weights: {lr_normal.weights_}")
    print(f"R² Score: {lr_normal.score(X_test, y_test):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred_normal):.4f}")

    # Method 2: Gradient Descent
    lr_gd = LinearRegression(
        method="gradient_descent",
        learning_rate=0.01,
        max_iterations=1000,
        tolerance=1e-6,
        verbose=False
    )
    lr_gd.fit(X_train, y_train)
    y_pred_gd = lr_gd.predict(X_test)

    print("\n2. Gradient Descent (Iterative Optimization)")
    print("-" * 70)
    print(f"Weights: {lr_gd.weights_}")
    print(f"Converged in {lr_gd.n_iterations_} iterations")
    print(f"Final Loss: {lr_gd.losses_[-1]:.6f}")
    print(f"R² Score: {lr_gd.score(X_test, y_test):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred_gd):.4f}")

    # Compare weights (should be nearly identical)
    print("\n3. Comparison")
    print("-" * 70)
    print(f"Weight difference (L2 norm): {np.linalg.norm(lr_normal.weights_ - lr_gd.weights_):.6f}")
    print(f"Prediction difference (L2 norm): {np.linalg.norm(y_pred_normal - y_pred_gd):.6f}")

    print("\nKey Insights:")
    print("• Normal equations give exact solution in one step")
    print("• Gradient descent approximates solution iteratively")
    print("• Both methods converge to same weights (within numerical precision)")
    print("• Normal equations: O(d³) complexity, fails if X^T X singular")
    print("• Gradient descent: O(k*n*d) for k iterations, works even if X^T X singular")
