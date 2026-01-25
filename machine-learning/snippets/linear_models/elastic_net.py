"""
Elastic Net Regression (L1 + L2 Regularization)

Implements Elastic Net using coordinate descent.

Mathematical Foundation:
- Loss: L(w) = (1/n)||y - Xw||² + λ[α||w||₁ + (1-α)/2 ||w||²]
- Combines L1 (Lasso) and L2 (Ridge) penalties
- α ∈ [0,1] controls the mix: α=1 is Lasso, α=0 is Ridge

Elastic Net addresses limitations of both Ridge and Lasso:
- L1 term: Provides sparsity (feature selection)
- L2 term: Handles correlated features (grouping effect)
Result: Stable feature selection even with correlated features.
"""

import numpy as np
from typing import Optional


class ElasticNet:
    """
    Elastic Net Regression with combined L1 and L2 regularization.

    Elastic Net adds both L1 (sparsity) and L2 (grouping) penalties,
    combining the best of Lasso and Ridge regression.

    Attributes:
        weights_ (np.ndarray): Learned sparse weight vector.
        intercept_ (float): Bias term (not regularized).
        n_iterations_ (int): Number of coordinate descent iterations.
        n_nonzero_weights_ (int): Number of non-zero weights.

    Mathematical Notes:
        The L2 term prevents the "one feature per group" limitation of Lasso.
        When features are correlated, Elastic Net tends to select them together
        (grouped selection) rather than arbitrarily picking one.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iterations: int = 1000,
        tolerance: float = 1e-4,
        fit_intercept: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Initialize Elastic Net.

        Args:
            alpha: Overall regularization strength (λ). Must be >= 0.
                Larger α → more regularization → sparser model.
            l1_ratio: Mixing parameter (α in formula). Must be in [0, 1].
                - l1_ratio = 1: Pure Lasso (L1 only)
                - l1_ratio = 0: Pure Ridge (L2 only)
                - 0 < l1_ratio < 1: Elastic Net (combination)
                Typical value: 0.5 (equal mix)
            max_iterations: Maximum coordinate descent iterations.
            tolerance: Convergence threshold.
            fit_intercept: If True, fit bias term (not regularized).
            verbose: If True, print training progress.

        Notes:
            Penalty = λ * [α||w||₁ + (1-α)/2 ||w||²]
            where λ = alpha, α = l1_ratio
            Always standardize features before using Elastic Net.
        """
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if not 0 <= l1_ratio <= 1:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        # Learned parameters
        self.weights_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.n_iterations_: int = 0
        self.n_nonzero_weights_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ElasticNet":
        """
        Fit Elastic Net using coordinate descent.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).

        Returns:
            self: Fitted model instance.

        Algorithm:
            Same as Lasso (coordinate descent) but with modified update rule
            that includes L2 penalty in addition to soft-thresholding.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same n_samples")

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

        # Count non-zero weights
        self.n_nonzero_weights_ = np.sum(np.abs(self.weights_) > 1e-10)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the Elastic Net model."""
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
        Solve Elastic Net using coordinate descent.

        Args:
            X: Centered features, shape (n_samples, n_features).
            y: Centered targets, shape (n_samples,).

        Returns:
            weights: Sparse weight vector, shape (n_features,).

        Coordinate Update for Elastic Net:
            The optimal update for coordinate j is:
            wⱼ = soft_threshold(ρⱼ / n, λ_L1) / (||xⱼ||²/n + λ_L2)

            where:
            - ρⱼ = xⱼ^T @ (y - X_{-j}@w_{-j})  [correlation with residual]
            - λ_L1 = α * l1_ratio  [L1 penalty strength]
            - λ_L2 = α * (1 - l1_ratio)  [L2 penalty strength]

        Key Difference from Lasso:
            L2 term appears in denominator, shrinking the update.
            This stabilizes solution and groups correlated features.
        """
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)

        # Precompute column norms
        X_norms_sq = np.sum(X ** 2, axis=0)
        X_norms_sq = np.where(X_norms_sq < 1e-10, 1.0, X_norms_sq)

        # Compute L1 and L2 penalty strengths
        lambda_l1 = self.alpha * self.l1_ratio
        lambda_l2 = self.alpha * (1 - self.l1_ratio)

        for iteration in range(self.max_iterations):
            max_weight_change = 0.0

            # Update each weight
            for j in range(n_features):
                # Compute residual excluding feature j
                residual = y - X @ weights + X[:, j] * weights[j]

                # Correlation with residual: ρⱼ = xⱼ^T @ r
                rho = X[:, j] @ residual

                # Soft-thresholding (L1 effect)
                numerator = self._soft_threshold(rho / n_samples, lambda_l1)

                # L2 penalty in denominator (shrinkage + grouping effect)
                denominator = X_norms_sq[j] / n_samples + lambda_l2

                old_weight = weights[j]
                weights[j] = numerator / denominator

                # Track convergence
                weight_change = abs(weights[j] - old_weight)
                max_weight_change = max(max_weight_change, weight_change)

            # Check convergence
            if max_weight_change < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break

            if self.verbose and (iteration + 1) % 100 == 0:
                y_pred = X @ weights
                mse = np.mean((y - y_pred) ** 2)
                l1_penalty = lambda_l1 * np.sum(np.abs(weights))
                l2_penalty = (lambda_l2 / 2) * np.sum(weights ** 2)
                loss = mse + l1_penalty + l2_penalty
                n_nonzero = np.sum(np.abs(weights) > 1e-10)
                print(f"Iter {iteration + 1}: Loss={loss:.6f}, Non-zero={n_nonzero}")

        self.n_iterations_ = iteration + 1
        return weights

    @staticmethod
    def _soft_threshold(rho: float, lambda_val: float) -> float:
        """Soft-thresholding operator (same as Lasso)."""
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
    import sys
    sys.path.append('..')  # To import Ridge and Lasso from same folder

    from ridge_regression import RidgeRegression
    from lasso_regression import LassoRegression

    print("=" * 70)
    print("Elastic Net: Best of Ridge and Lasso")
    print("=" * 70)

    # Generate data with correlated features
    # This highlights Elastic Net's advantage over Lasso
    np.random.seed(42)
    n_samples, n_features = 100, 50

    # Create features where some are highly correlated
    X = np.random.randn(n_samples, n_features)

    # Make features 0-4 correlated (group 1)
    for i in range(1, 5):
        X[:, i] = X[:, 0] + 0.1 * np.random.randn(n_samples)

    # Make features 10-14 correlated (group 2)
    for i in range(11, 15):
        X[:, i] = X[:, 10] + 0.1 * np.random.randn(n_samples)

    # True coefficients: Only groups 1 and 2 are relevant
    true_coef = np.zeros(n_features)
    true_coef[0:5] = 5.0  # Group 1 all have same coefficient
    true_coef[10:15] = -3.0  # Group 2 all have same coefficient

    # Generate target
    y = X @ true_coef + 2 * np.random.randn(n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nData Setup:")
    print(f"  Total features: {n_features}")
    print(f"  Correlated groups: 2 (features 0-4 and 10-14)")
    print(f"  Relevant features: 10 (in correlated groups)")
    print(f"  Irrelevant features: 40")

    # Compare Ridge, Lasso, and Elastic Net
    print("\n" + "=" * 70)
    print("Comparison: Ridge vs Lasso vs Elastic Net")
    print("=" * 70)

    models = {
        "Ridge (α=1.0)": RidgeRegression(alpha=1.0, method="closed_form"),
        "Lasso (α=1.0)": LassoRegression(alpha=1.0, max_iterations=2000),
        "Elastic Net (α=1.0, l1_ratio=0.5)": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iterations=2000),
        "Elastic Net (α=1.0, l1_ratio=0.7)": ElasticNet(alpha=1.0, l1_ratio=0.7, max_iterations=2000),
    }

    print(f"\n{'Model':<35} {'R² Train':<10} {'R² Test':<10} {'Non-zero':<10} {'Sparsity %'}")
    print("-" * 70)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)

        # Count non-zero weights
        if hasattr(model, 'n_nonzero_weights_'):
            n_nonzero = model.n_nonzero_weights_
        else:
            n_nonzero = np.sum(np.abs(model.weights_) > 1e-10)

        sparsity_pct = (1 - n_nonzero / n_features) * 100

        print(f"{name:<35} {train_score:<10.4f} {test_score:<10.4f} {n_nonzero:<10d} {sparsity_pct:.1f}%")

    # Detailed analysis: Which features did each model select?
    print("\n" + "=" * 70)
    print("Feature Selection Analysis: Correlated Groups")
    print("=" * 70)

    ridge = RidgeRegression(alpha=1.0, method="closed_form")
    ridge.fit(X_train_scaled, y_train)

    lasso = LassoRegression(alpha=1.0, max_iterations=2000)
    lasso.fit(X_train_scaled, y_train)

    enet = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iterations=2000)
    enet.fit(X_train_scaled, y_train)

    def count_selected_in_group(weights, indices):
        """Count how many features in a group have non-zero weights."""
        return np.sum(np.abs(weights[indices]) > 1e-10)

    group1_indices = list(range(0, 5))
    group2_indices = list(range(10, 15))

    print(f"\nGroup 1 (features 0-4, all relevant and correlated):")
    print(f"  Ridge:       {count_selected_in_group(ridge.weights_, group1_indices)}/5 selected")
    print(f"  Lasso:       {count_selected_in_group(lasso.weights_, group1_indices)}/5 selected")
    print(f"  Elastic Net: {count_selected_in_group(enet.weights_, group1_indices)}/5 selected")

    print(f"\nGroup 2 (features 10-14, all relevant and correlated):")
    print(f"  Ridge:       {count_selected_in_group(ridge.weights_, group2_indices)}/5 selected")
    print(f"  Lasso:       {count_selected_in_group(lasso.weights_, group2_indices)}/5 selected")
    print(f"  Elastic Net: {count_selected_in_group(enet.weights_, group2_indices)}/5 selected")

    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("RIDGE:")
    print("  • Keeps ALL features (no sparsity)")
    print("  • Shrinks correlated features together")
    print("  • Less interpretable due to many non-zero weights")

    print("\nLASSO:")
    print("  • Produces sparse models (feature selection)")
    print("  • UNSTABLE with correlated features:")
    print("    Randomly picks 1-2 features from each correlated group")
    print("  • May miss relevant features if they're correlated")

    print("\nELASTIC NET:")
    print("  • Combines sparsity (L1) and grouping (L2)")
    print("  • STABLE with correlated features:")
    print("    Tends to select correlated features together")
    print("  • Best of both worlds: sparse yet stable")
    print("  • Recommended when features may be correlated")

    print("\n→ Use Elastic Net when:")
    print("  - You want feature selection (like Lasso)")
    print("  - Features might be correlated (unlike Lasso, it handles this well)")
    print("  - You need stable, interpretable models")
