"""
Lasso Regression (L1 Regularization)

Implements Lasso regression using coordinate descent.

Mathematical Foundation:
- Loss: L(w) = (1/n)||y - Xw||² + λ||w||₁
- L1 penalty: ||w||₁ = Σ|wⱼ|
- No closed-form solution (non-differentiable at w=0)
- Coordinate descent: Optimize one weight at a time

Lasso produces sparse solutions (many weights exactly zero), performing
automatic feature selection. Unlike Ridge, Lasso sets irrelevant features
to exactly zero rather than just shrinking them.
"""

import numpy as np
from typing import Optional


class LassoRegression:
    """
    Lasso Regression with L1 regularization.

    Lasso adds a penalty term λ||w||₁ to the linear regression loss.
    The L1 penalty encourages sparsity: many weights become exactly zero,
    effectively performing feature selection.

    Attributes:
        weights_ (np.ndarray): Learned weight vector (sparse, many zeros).
        intercept_ (float): Bias term (not regularized).
        n_iterations_ (int): Number of coordinate descent iterations.
        n_nonzero_weights_ (int): Number of non-zero weights (sparsity measure).

    Mathematical Notes:
        L1 penalty has sharp corners at zero, causing some weights to be
        exactly zero (unlike L2 which only shrinks toward zero).
        Solved using coordinate descent: update one weight at a time.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
        fit_intercept: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Initialize Lasso Regression.

        Args:
            alpha: Regularization strength (λ). Must be >= 0.
                - α = 0: Standard linear regression (no sparsity)
                - Larger α: More coefficients set to zero (sparser model)
                Typical values: 0.01, 0.1, 1, 10
            max_iterations: Maximum coordinate descent iterations.
                Typical values: 1000-10000. Each iteration updates all weights once.
            tolerance: Convergence threshold.
                Stop if max weight change < tolerance.
            fit_intercept: If True, fit bias term (not regularized).
            verbose: If True, print progress.

        Notes:
            Always standardize features before Lasso (L1 is scale-dependent).
            Coordinate descent is efficient for Lasso (O(knd) for k iterations).
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")

        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        # Learned parameters
        self.weights_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iterations_: int = 0
        self.n_nonzero_weights_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegression":
        """
        Fit Lasso regression using coordinate descent.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).

        Returns:
            self: Fitted model instance.

        Algorithm (Coordinate Descent):
            1. Initialize weights to zeros
            2. Repeat until convergence:
                For each feature j:
                    a. Compute partial residual: r = y - X@w (excluding wⱼ)
                    b. Compute optimal wⱼ using soft-thresholding
                    c. Update wⱼ
            3. Stop when max |Δwⱼ| < tolerance

        Soft-Thresholding Operator:
            wⱼ = sign(ρⱼ) * max(0, |ρⱼ| - λ) / ||xⱼ||²
            where ρⱼ = xⱼ^T @ (y - X_{-j}@w_{-j})
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same n_samples")

        # Center data if fitting intercept
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

        # Fit using coordinate descent
        self.weights_ = self._coordinate_descent(X_centered, y_centered)

        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.weights_
        else:
            self.intercept_ = 0.0

        # Count non-zero weights (sparsity metric)
        self.n_nonzero_weights_ = np.sum(np.abs(self.weights_) > 1e-10)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the Lasso model.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            predictions: Predicted values, shape (n_samples,).
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        return X @ self.weights_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)

    def _coordinate_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve Lasso using coordinate descent algorithm.

        Args:
            X: Centered features, shape (n_samples, n_features).
            y: Centered targets, shape (n_samples,).

        Returns:
            weights: Sparse weight vector, shape (n_features,).

        Coordinate Descent Algorithm:
            Cyclically optimize one coordinate (weight) at a time while
            holding others fixed. For Lasso, the optimal single-coordinate
            update has a closed form using soft-thresholding.

        Complexity: O(k*n*d) for k iterations
            Much faster than general gradient descent for sparse solutions.
        """
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)

        # Precompute column norms (for efficiency)
        # ||xⱼ||² = sum of squared values in column j
        X_norms_sq = np.sum(X ** 2, axis=0)

        # Avoid division by zero for zero columns
        X_norms_sq = np.where(X_norms_sq < 1e-10, 1.0, X_norms_sq)

        for iteration in range(self.max_iterations):
            max_weight_change = 0.0

            # Update each weight sequentially
            for j in range(n_features):
                # Compute residual excluding feature j
                # r = y - X @ w, but w[j] is excluded
                residual = y - X @ weights + X[:, j] * weights[j]

                # Compute correlation: ρⱼ = xⱼ^T @ r
                rho = X[:, j] @ residual

                # Soft-thresholding operator
                # This is the closed-form solution for minimizing Lasso w.r.t. wⱼ
                # wⱼ* = soft_threshold(ρⱼ / ||xⱼ||², λ)
                old_weight = weights[j]
                weights[j] = self._soft_threshold(rho / n_samples, self.alpha) / (X_norms_sq[j] / n_samples)

                # Track maximum weight change for convergence check
                weight_change = abs(weights[j] - old_weight)
                max_weight_change = max(max_weight_change, weight_change)

            # Check convergence: stop if weights barely changed
            if max_weight_change < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            if self.verbose and (iteration + 1) % 100 == 0:
                # Compute current loss for monitoring
                y_pred = X @ weights
                mse = np.mean((y - y_pred) ** 2)
                l1_penalty = self.alpha * np.sum(np.abs(weights))
                loss = mse + l1_penalty
                n_nonzero = np.sum(np.abs(weights) > 1e-10)
                print(f"Iteration {iteration + 1}: Loss={loss:.6f}, Non-zero weights={n_nonzero}")

        self.n_iterations_ = iteration + 1
        return weights

    @staticmethod
    def _soft_threshold(rho: float, lambda_val: float) -> float:
        """
        Soft-thresholding operator (proximal operator for L1 norm).

        Args:
            rho: Input value (correlation).
            lambda_val: Threshold parameter (regularization strength).

        Returns:
            Soft-thresholded value.

        Formula:
            soft_threshold(ρ, λ) = sign(ρ) * max(0, |ρ| - λ)
                                 = { ρ - λ   if ρ > λ
                                   { 0       if |ρ| ≤ λ
                                   { ρ + λ   if ρ < -λ

        Intuition:
            Shrinks ρ toward zero by amount λ. If |ρ| < λ, result is exactly zero.
            This is what creates sparsity in Lasso solutions.
        """
        if rho > lambda_val:
            return rho - lambda_val
        elif rho < -lambda_val:
            return rho + lambda_val
        else:
            return 0.0


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print("=" * 70)
    print("Lasso Regression: Demonstrating Sparsity and Feature Selection")
    print("=" * 70)

    # Generate data with many irrelevant features
    X, y, true_coef = make_regression(
        n_samples=100,
        n_features=50,
        n_informative=10,  # Only 10 features are relevant
        noise=20,
        coef=True,  # Return true coefficients
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (critical for Lasso!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nData Info:")
    print(f"  Total features: {X.shape[1]}")
    print(f"  Informative features: 10")
    print(f"  True non-zero coefficients: {np.sum(np.abs(true_coef) > 1e-10)}")

    # Compare different alpha values
    print("\n" + "=" * 70)
    print("Comparing Lasso with different regularization strengths (α):")
    print("-" * 70)
    print(f"{'Alpha':<10} {'R² (Train)':<12} {'R² (Test)':<12} {'Non-zero':<12} {'Sparsity %'}")
    print("-" * 70)

    alphas = [0.01, 0.1, 1.0, 5.0, 10.0]

    for alpha in alphas:
        lasso = LassoRegression(alpha=alpha, max_iterations=2000, verbose=False)
        lasso.fit(X_train_scaled, y_train)

        train_score = lasso.score(X_train_scaled, y_train)
        test_score = lasso.score(X_test_scaled, y_test)
        n_nonzero = lasso.n_nonzero_weights_
        sparsity_pct = (1 - n_nonzero / X.shape[1]) * 100

        print(f"{alpha:<10.2f} {train_score:<12.4f} {test_score:<12.4f} {n_nonzero:<12d} {sparsity_pct:.1f}%")

    print("\nKey Observations:")
    print("• Small α (0.01): Few features zeroed out, similar to Ridge")
    print("• Medium α (1.0): Many features set to zero (automatic feature selection)")
    print("• Large α (10.0): Most features zeroed, only strongest signals remain")
    print("• Trade-off: Sparsity vs. predictive performance")

    # Detailed analysis for α = 1.0
    print("\n" + "=" * 70)
    print("Detailed Analysis: α = 1.0")
    print("=" * 70)

    lasso = LassoRegression(alpha=1.0, max_iterations=2000, verbose=False)
    lasso.fit(X_train_scaled, y_train)

    print(f"\nModel Statistics:")
    print(f"  Converged in: {lasso.n_iterations_} iterations")
    print(f"  Non-zero weights: {lasso.n_nonzero_weights_} / {X.shape[1]}")
    print(f"  R² (Train): {lasso.score(X_train_scaled, y_train):.4f}")
    print(f"  R² (Test): {lasso.score(X_test_scaled, y_test):.4f}")

    # Show which features were selected
    selected_features = np.where(np.abs(lasso.weights_) > 1e-10)[0]
    print(f"\nSelected features (indices): {selected_features.tolist()[:15]}...")  # Show first 15

    # Compare weight magnitudes
    nonzero_weights = lasso.weights_[lasso.weights_ != 0]
    if len(nonzero_weights) > 0:
        print(f"\nNon-zero weight statistics:")
        print(f"  Min |w|: {np.min(np.abs(nonzero_weights)):.4f}")
        print(f"  Max |w|: {np.max(np.abs(nonzero_weights)):.4f}")
        print(f"  Mean |w|: {np.mean(np.abs(nonzero_weights)):.4f}")

    print("\n" + "=" * 70)
    print("Summary: Lasso vs Ridge")
    print("=" * 70)
    print("LASSO (L1):")
    print("  ✓ Produces sparse models (many weights exactly zero)")
    print("  ✓ Automatic feature selection")
    print("  ✓ Interpretable (fewer features)")
    print("  ✗ Unstable with correlated features (picks one arbitrarily)")
    print("  ✗ No closed-form solution (requires coordinate descent)")
    print("\nRIDGE (L2):")
    print("  ✓ Handles correlated features well (shrinks together)")
    print("  ✓ Closed-form solution (faster)")
    print("  ✓ More stable")
    print("  ✗ All features remain (no feature selection)")
    print("  ✗ Less interpretable")
    print("\n→ Use Lasso when you want sparse models and feature selection")
    print("→ Use Ridge when features are correlated or you want all features")
