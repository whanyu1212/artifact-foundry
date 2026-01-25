"""
Ridge Regression (L2 Regularization)

Implements Ridge regression with closed-form solution and gradient descent.

Mathematical Foundation:
- Loss: L(w) = (1/n)||y - Xw||² + λ||w||²
- Closed-form: w = (X^T X + nλI)^-1 X^T y
- Gradient: ∇L = -(2/n)X^T(y - Xw) + 2λw

Ridge adds L2 penalty (squared weights) to prevent overfitting.
The penalty term makes X^T X + nλI always invertible, even when X^T X is singular.
"""

import numpy as np
from typing import Literal, Optional


class RidgeRegression:
    """
    Ridge Regression with L2 regularization.

    Ridge regression adds a penalty term λ||w||² to the standard linear regression
    loss. This shrinks all coefficients toward zero, preventing overfitting and
    handling multicollinearity.

    Attributes:
        weights_ (np.ndarray): Learned weight vector including bias term.
        n_iterations_ (int): Number of iterations (gradient descent only).
        losses_ (list[float]): Loss history (gradient descent only).

    Mathematical Notes:
        Ridge loss: L = MSE + λ * ||w||²
        Regularization makes solution stable even when features are correlated
        or d > n (more features than samples).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        method: Literal["closed_form", "gradient_descent"] = "closed_form",
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        fit_intercept: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Initialize Ridge Regression.

        Args:
            alpha: Regularization strength (λ). Must be >= 0.
                - α = 0: Equivalent to ordinary linear regression (no regularization)
                - α → ∞: Weights shrink to zero (extreme regularization)
                Typical values: 0.01, 0.1, 1, 10, 100
            method: Solution method.
                - "closed_form": Exact solution via (X^T X + nλI)^-1 X^T y
                - "gradient_descent": Iterative optimization
            learning_rate: Step size for gradient descent.
            max_iterations: Maximum iterations for gradient descent.
            tolerance: Convergence threshold for gradient descent.
            fit_intercept: If True, add bias term (recommended).
                Note: Bias term is NOT regularized (standard practice).
            verbose: If True, print training progress.

        Notes:
            Always standardize features before Ridge regression (regularization
            is scale-dependent). Scikit-learn does this automatically internally.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        self.alpha = alpha
        self.method = method
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        # Learned parameters
        self.weights_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iterations_: int = 0
        self.losses_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        """
        Fit Ridge regression model.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).

        Returns:
            self: Fitted model instance.

        Notes:
            If fit_intercept=True, data is centered (subtract mean) before fitting.
            Intercept is computed separately and NOT regularized.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same n_samples: {X.shape[0]} vs {y.shape[0]}")

        # Center data if fitting intercept
        # This removes the bias term from regularization
        if self.fit_intercept:
            X_mean = X.mean(axis=0)
            y_mean = y.mean()
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0

        # Fit model on centered data
        if self.method == "closed_form":
            self.weights_ = self._fit_closed_form(X_centered, y_centered)
        elif self.method == "gradient_descent":
            self.weights_ = self._fit_gradient_descent(X_centered, y_centered)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute intercept: intercept = y_mean - X_mean^T @ weights
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.weights_
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Ridge model.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            predictions: Predicted values, shape (n_samples,).

        Formula:
            ŷ = X @ w + intercept
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        return X @ self.weights_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.

        Args:
            X: Features, shape (n_samples, n_features).
            y: True targets, shape (n_samples,).

        Returns:
            r2_score: R² value. Higher is better.
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)

    def _fit_closed_form(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve Ridge regression using closed-form solution.

        Args:
            X: Centered features, shape (n_samples, n_features).
            y: Centered targets, shape (n_samples,).

        Returns:
            weights: Optimal weight vector, shape (n_features,).

        Formula:
            w = (X^T X + nλI)^-1 X^T y

        Key Insight:
            Adding nλI to X^T X ensures the matrix is always invertible,
            even when X^T X is singular (e.g., d > n or collinear features).

        Complexity: O(n*d² + d³)
        """
        n_samples, n_features = X.shape

        # Compute X^T X + nλI
        # nλI term makes matrix invertible (regularization effect)
        XTX = X.T @ X
        ridge_matrix = XTX + (n_samples * self.alpha) * np.eye(n_features)

        # Solve (X^T X + nλI) w = X^T y
        # Using lstsq is more numerically stable than explicit inverse
        weights, _, _, _ = np.linalg.lstsq(ridge_matrix, X.T @ y, rcond=None)

        return weights

    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve Ridge regression using gradient descent.

        Args:
            X: Centered features, shape (n_samples, n_features).
            y: Centered targets, shape (n_samples,).

        Returns:
            weights: Optimized weight vector, shape (n_features,).

        Gradient:
            ∇L = -(2/n) * X^T @ (y - X@w) + 2λw
               = -(2/n) * X^T @ (y - ŷ) + 2λw

        Update rule:
            w := w - α * ∇L
        """
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        prev_loss = float('inf')

        for iteration in range(self.max_iterations):
            # Predictions
            y_pred = X @ weights

            # Compute Ridge loss: MSE + λ * ||w||²
            mse = np.mean((y - y_pred) ** 2)
            l2_penalty = self.alpha * np.sum(weights ** 2)
            loss = mse + l2_penalty
            self.losses_.append(loss)

            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration}, loss={loss:.6f}")
                break

            # Compute gradient: ∇L = -(2/n)X^T(y - ŷ) + 2λw
            data_gradient = -(2 / n_samples) * X.T @ (y - y_pred)
            reg_gradient = 2 * self.alpha * weights
            gradient = data_gradient + reg_gradient

            # Update weights
            weights -= self.learning_rate * gradient

            prev_loss = loss

            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1}: Loss={loss:.6f}, MSE={mse:.6f}, L2={l2_penalty:.6f}")

        self.n_iterations_ = iteration + 1
        return weights


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    print("=" * 70)
    print("Ridge Regression: Demonstrating Regularization Effect")
    print("=" * 70)

    # Generate data with many features (high-dimensional)
    X, y = make_regression(
        n_samples=100,
        n_features=50,  # More features than typical
        n_informative=10,  # Only 10 are actually useful
        noise=20,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (important for Ridge!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compare different alpha values
    alphas = [0.0, 0.1, 1.0, 10.0, 100.0]

    print("\nComparing Ridge with different regularization strengths (α):")
    print("-" * 70)
    print(f"{'Alpha':<10} {'R² (Train)':<15} {'R² (Test)':<15} {'||w||² (Weight Norm)'}")
    print("-" * 70)

    for alpha in alphas:
        ridge = RidgeRegression(alpha=alpha, method="closed_form")
        ridge.fit(X_train_scaled, y_train)

        train_score = ridge.score(X_train_scaled, y_train)
        test_score = ridge.score(X_test_scaled, y_test)
        weight_norm = np.sum(ridge.weights_ ** 2)

        print(f"{alpha:<10.1f} {train_score:<15.4f} {test_score:<15.4f} {weight_norm:.4f}")

    print("\nKey Observations:")
    print("• α = 0: Standard linear regression (no regularization)")
    print("  - May overfit: High train R², lower test R²")
    print("  - Large weight magnitudes (high ||w||²)")
    print("\n• α > 0: Ridge regularization")
    print("  - Weights shrink as α increases")
    print("  - Trade-off: Lower train R² but potentially better test R²")
    print("  - Prevents overfitting by penalizing large weights")
    print("\n• α too large: Underfitting")
    print("  - Both train and test R² decrease")
    print("  - Weights shrink too much, model too simple")

    # Compare closed-form vs gradient descent
    print("\n" + "=" * 70)
    print("Comparing Solution Methods (α = 1.0)")
    print("=" * 70)

    ridge_cf = RidgeRegression(alpha=1.0, method="closed_form")
    ridge_cf.fit(X_train_scaled, y_train)

    ridge_gd = RidgeRegression(
        alpha=1.0,
        method="gradient_descent",
        learning_rate=0.01,
        max_iterations=2000,
        verbose=False
    )
    ridge_gd.fit(X_train_scaled, y_train)

    print(f"\nClosed-Form Solution:")
    print(f"  R² (Test): {ridge_cf.score(X_test_scaled, y_test):.4f}")
    print(f"  Weight norm: {np.linalg.norm(ridge_cf.weights_):.4f}")

    print(f"\nGradient Descent Solution:")
    print(f"  Iterations: {ridge_gd.n_iterations_}")
    print(f"  R² (Test): {ridge_gd.score(X_test_scaled, y_test):.4f}")
    print(f"  Weight norm: {np.linalg.norm(ridge_gd.weights_):.4f}")
    print(f"  Weight difference: {np.linalg.norm(ridge_cf.weights_ - ridge_gd.weights_):.6f}")

    print("\n• Both methods converge to essentially the same solution")
    print("• Closed-form: Exact, one-step, but O(d³) complexity")
    print("• Gradient descent: Iterative approximation, scales better to large d")
