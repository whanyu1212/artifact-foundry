"""
Linear Discriminant Analysis (LDA)

Implements LDA for classification and dimensionality reduction.

Mathematical Foundation:
- Assumes multivariate Gaussian: P(X|y) ~ N(μ_y, Σ)
- Shared covariance: All classes have same Σ
- Discriminant: δ_y(x) = x^T Σ^(-1) μ_y - ½ μ_y^T Σ^(-1) μ_y + log P(y)
- Linear decision boundary: δ_i(x) = δ_j(x) is a hyperplane

Classification: ŷ = argmax_y δ_y(x)
Projection: Find directions maximizing between-class / within-class variance
"""

import numpy as np
from typing import Optional


class LinearDiscriminantAnalysis:
    """
    Linear Discriminant Analysis for classification and dimensionality reduction.

    LDA models each class as a multivariate Gaussian with class-specific mean
    but shared covariance matrix. This leads to linear decision boundaries.

    Attributes:
        means_ (np.ndarray): Class means μ_y, shape (n_classes, n_features).
        covariance_ (np.ndarray): Pooled covariance matrix Σ, shape (n_features, n_features).
        priors_ (np.ndarray): Class priors P(y), shape (n_classes,).
        classes_ (np.ndarray): Unique class labels.
        scalings_ (np.ndarray): Projection matrix for dimensionality reduction.

    Mathematical Notes:
        Discriminant function (linear in x):
        δ_y(x) = x^T Σ^(-1) μ_y - ½ μ_y^T Σ^(-1) μ_y + log P(y)

        Decision boundary between classes i and j:
        w^T x + w_0 = 0 where w = Σ^(-1)(μ_i - μ_j)
    """

    def __init__(self, n_components: Optional[int] = None) -> None:
        """
        Initialize Linear Discriminant Analysis.

        Args:
            n_components: Number of components for dimensionality reduction.
                If None, use all components (min(n_features, n_classes - 1)).
                For classification only, this parameter is ignored.

        Notes:
            LDA can project to at most (n_classes - 1) dimensions because
            there are only (n_classes - 1) independent discriminant directions.
        """
        self.n_components = n_components

        # Learned parameters (set during fit)
        self.means_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.scalings_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearDiscriminantAnalysis":
        """
        Fit Linear Discriminant Analysis model.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).

        Returns:
            self: Fitted model instance.

        Algorithm:
            1. Compute class priors: P(y) = n_y / n
            2. Compute class means: μ_y = mean of samples in class y
            3. Compute pooled covariance: Σ = Σ_y Σ_i (x_i - μ_y)(x_i - μ_y)^T / (n - K)
            4. Compute projection for dimensionality reduction (optional)
        """
        n_samples, n_features = X.shape

        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize parameter arrays
        self.means_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        # Compute class statistics
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]

            # Class prior: P(y=c) = n_c / n
            self.priors_[idx] = X_c.shape[0] / n_samples

            # Class mean: μ_c
            self.means_[idx, :] = X_c.mean(axis=0)

        # Compute pooled (shared) covariance matrix
        # Σ = (1 / (n - K)) Σ_y Σ_i (x_i - μ_y)(x_i - μ_y)^T
        self.covariance_ = np.zeros((n_features, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            # Center data: subtract class mean
            X_c_centered = X_c - self.means_[idx]
            # Add to covariance: X^T X
            self.covariance_ += X_c_centered.T @ X_c_centered

        # Divide by (n - K) for unbiased estimate
        self.covariance_ /= (n_samples - n_classes)

        # Compute projection for dimensionality reduction
        if self.n_components is not None:
            self._compute_projection(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Features to predict, shape (n_samples, n_features).

        Returns:
            predictions: Predicted class labels, shape (n_samples,).

        Formula:
            ŷ = argmax_y δ_y(x)
            where δ_y(x) = x^T Σ^(-1) μ_y - ½ μ_y^T Σ^(-1) μ_y + log P(y)
        """
        # Compute discriminant scores for all classes
        discriminants = self._discriminant_scores(X)

        # Return class with highest discriminant score
        return self.classes_[np.argmax(discriminants, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Features to predict, shape (n_samples, n_features).

        Returns:
            probabilities: Class probabilities, shape (n_samples, n_classes).

        Notes:
            Probabilities computed using softmax of discriminant scores.
        """
        discriminants = self._discriminant_scores(X)

        # Softmax with numerical stability (log-sum-exp trick)
        discriminants_max = np.max(discriminants, axis=1, keepdims=True)
        exp_discriminants = np.exp(discriminants - discriminants_max)
        probabilities = exp_discriminants / np.sum(exp_discriminants, axis=1, keepdims=True)

        return probabilities

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data to lower-dimensional space using LDA projection.

        Args:
            X: Features to transform, shape (n_samples, n_features).

        Returns:
            X_transformed: Projected features, shape (n_samples, n_components).

        Notes:
            Projects data onto discriminant directions that maximize
            class separability (between-class variance / within-class variance).
        """
        if self.scalings_ is None:
            raise RuntimeError("Model must be fitted with n_components specified")

        # Project: X_new = X @ scalings
        return X @ self.scalings_

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit LDA and transform data in one step.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).

        Returns:
            X_transformed: Projected features, shape (n_samples, n_components).
        """
        self.fit(X, y)
        return self.transform(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _discriminant_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute discriminant scores for all classes.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            scores: Discriminant δ_y(x), shape (n_samples, n_classes).

        Formula:
            δ_y(x) = x^T Σ^(-1) μ_y - ½ μ_y^T Σ^(-1) μ_y + log P(y)

        Implementation:
            For efficiency, precompute Σ^(-1) μ_y for all classes.
        """
        if self.means_ is None or self.covariance_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Invert covariance matrix (use pseudo-inverse for numerical stability)
        cov_inv = np.linalg.pinv(self.covariance_)

        # Precompute Σ^(-1) μ_y for all classes
        # Shape: (n_classes, n_features)
        means_transformed = self.means_ @ cov_inv.T

        # Initialize discriminant scores
        scores = np.zeros((n_samples, n_classes))

        for c in range(n_classes):
            # Linear term: x^T Σ^(-1) μ_y
            # Equivalent to: x @ (Σ^(-1) μ_y)
            linear_term = X @ means_transformed[c]

            # Constant term: -½ μ_y^T Σ^(-1) μ_y
            constant_term = -0.5 * self.means_[c] @ means_transformed[c]

            # Prior term: log P(y)
            prior_term = np.log(self.priors_[c])

            # Discriminant: δ_y(x) = linear + constant + prior
            scores[:, c] = linear_term + constant_term + prior_term

        return scores

    def _compute_projection(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Compute LDA projection for dimensionality reduction.

        Finds projection directions that maximize:
        J(w) = w^T S_B w / w^T S_W w

        where S_B = between-class scatter, S_W = within-class scatter.

        Solution: Eigenvectors of S_W^(-1) S_B.
        """
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Overall mean
        overall_mean = X.mean(axis=0)

        # Between-class scatter matrix S_B
        S_B = np.zeros((n_features, n_features))
        for idx, c in enumerate(self.classes_):
            n_c = np.sum(y == c)
            mean_diff = (self.means_[idx] - overall_mean).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # Within-class scatter matrix S_W (same as covariance)
        S_W = self.covariance_ * (n_samples - n_classes)

        # Solve generalized eigenvalue problem: S_B w = λ S_W w
        # Equivalent to: S_W^(-1) S_B w = λ w
        S_W_inv = np.linalg.pinv(S_W)
        eigvals, eigvecs = np.linalg.eig(S_W_inv @ S_B)

        # Sort eigenvectors by eigenvalues (descending)
        idx_sorted = np.argsort(eigvals)[::-1]
        eigvecs_sorted = eigvecs[:, idx_sorted].real

        # Maximum number of components is min(n_features, n_classes - 1)
        max_components = min(n_features, n_classes - 1)

        if self.n_components is None:
            n_components = max_components
        else:
            n_components = min(self.n_components, max_components)

        # Store top n_components eigenvectors as projection matrix
        self.scalings_ = eigvecs_sorted[:, :n_components]


if __name__ == "__main__":
    from sklearn.datasets import load_iris, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("Linear Discriminant Analysis: Classification & Projection")
    print("=" * 70)

    # Example 1: Classification on Iris dataset
    print("\n1. Classification (Iris Dataset)")
    print("-" * 70)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    y_pred = lda.predict(X_test)
    y_proba = lda.predict_proba(X_test)

    print(f"Training Accuracy: {lda.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {lda.score(X_test, y_test):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Example 2: Dimensionality Reduction
    print("\n" + "=" * 70)
    print("2. Dimensionality Reduction (Project to 2D)")
    print("-" * 70)

    lda_2d = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda_2d.fit_transform(X, y)

    print(f"Original dimensions: {X.shape[1]}")
    print(f"Projected dimensions: {X_lda.shape[1]}")
    print(f"Explained variance directions: {X_lda.shape[1]} out of {min(X.shape[1], len(np.unique(y)) - 1)} maximum")

    # Show class means in projected space
    print("\nClass means in 2D LDA space:")
    print(f"{'Class':<15} {'LD1':<10} {'LD2'}")
    print("-" * 35)
    for idx, class_name in enumerate(iris.target_names):
        X_class = X_lda[y == idx]
        mean_ld1 = X_class[:, 0].mean()
        mean_ld2 = X_class[:, 1].mean()
        print(f"{class_name:<15} {mean_ld1:<10.4f} {mean_ld2:.4f}")

    print("\n" + "=" * 70)
    print("3. Learned Parameters")
    print("-" * 70)
    print(f"Class priors: {lda.priors_}")
    print(f"\nShared covariance matrix shape: {lda.covariance_.shape}")
    print(f"Covariance determinant: {np.linalg.det(lda.covariance_):.4f}")

    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("• LDA assumes Gaussian distributions with shared covariance")
    print("• Produces linear decision boundaries (hyperplanes)")
    print("• Can project to at most (n_classes - 1) dimensions")
    print("• More stable than QDA for small datasets (fewer parameters)")
    print("• Works well when classes have similar covariance structure")
    print("\n→ Use for classification when Gaussian assumption holds")
    print("→ Use for supervised dimensionality reduction (better than PCA for classification)")
