"""
Gaussian Naive Bayes Classifier

Implements Gaussian Naive Bayes for continuous features.

Mathematical Foundation:
- Bayes' theorem: P(y|X) = P(X|y)P(y) / P(X)
- Naive independence: P(X|y) = ∏ P(xᵢ|y)
- Gaussian likelihood: P(xᵢ|y) ~ N(μᵢy, σ²ᵢy)

Classification: ŷ = argmax_y [log P(y) + Σ log P(xᵢ|y)]
"""

import numpy as np
from typing import Optional


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier for continuous features.

    Assumes each feature follows a Gaussian (normal) distribution within
    each class. Features are assumed conditionally independent given the class.

    Attributes:
        class_prior_ (np.ndarray): Prior probabilities P(y) for each class.
        classes_ (np.ndarray): Unique class labels.
        theta_ (np.ndarray): Mean μᵢy for each feature i and class y.
        var_ (np.ndarray): Variance σ²ᵢy for each feature i and class y.

    Mathematical Notes:
        For each class y and feature i, estimate Gaussian parameters:
        - μᵢy = mean of feature i in class y
        - σ²ᵢy = variance of feature i in class y

        Log posterior: log P(y|X) = log P(y) + Σᵢ log N(xᵢ; μᵢy, σ²ᵢy)
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        """
        Initialize Gaussian Naive Bayes.

        Args:
            var_smoothing: Small value added to variances for numerical stability.
                Prevents division by zero when a feature has zero variance.
                Typical values: 1e-9 to 1e-6.

        Notes:
            Variance smoothing is essential when features have very low or zero
            variance in some classes (e.g., constant feature values).
        """
        self.var_smoothing = var_smoothing

        # Learned parameters (set during fit)
        self.class_prior_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None  # Means
        self.var_: Optional[np.ndarray] = None  # Variances

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        """
        Fit Gaussian Naive Bayes model.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).

        Returns:
            self: Fitted model instance.

        Algorithm:
            1. Compute class priors: P(y) = n_y / n
            2. For each class y and feature i:
                a. Compute mean: μᵢy = mean of feature i in class y
                b. Compute variance: σ²ᵢy = variance of feature i in class y
        """
        n_samples, n_features = X.shape

        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Initialize parameter arrays
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        # Estimate parameters for each class
        for idx, c in enumerate(self.classes_):
            # Get samples belonging to class c
            X_c = X[y == c]

            # Class prior: P(y=c) = n_c / n
            self.class_prior_[idx] = X_c.shape[0] / n_samples

            # Mean for each feature: μᵢc = mean of feature i in class c
            self.theta_[idx, :] = X_c.mean(axis=0)

            # Variance for each feature: σ²ᵢc = var of feature i in class c
            # Add smoothing to prevent zero variance
            self.var_[idx, :] = X_c.var(axis=0) + self.var_smoothing

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Features to predict, shape (n_samples, n_features).

        Returns:
            predictions: Predicted class labels, shape (n_samples,).

        Formula:
            ŷ = argmax_y P(y|X) = argmax_y [P(y) ∏ᵢ P(xᵢ|y)]
        """
        # Get log posterior for each class
        log_posterior = self._log_posterior(X)

        # Return class with highest posterior
        return self.classes_[np.argmax(log_posterior, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Features to predict, shape (n_samples, n_features).

        Returns:
            probabilities: Class probabilities, shape (n_samples, n_classes).
                probabilities[i, k] = P(y=class_k | X[i])

        Notes:
            Probabilities are computed using softmax of log posteriors
            to ensure numerical stability and proper normalization.
        """
        log_posterior = self._log_posterior(X)

        # Convert log probabilities to probabilities using log-sum-exp trick
        # P(y|X) = exp(log P(y|X)) / Σ_y exp(log P(y|X))
        log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        exp_log_posterior = np.exp(log_posterior - log_posterior_max)
        probabilities = exp_log_posterior / np.sum(exp_log_posterior, axis=1, keepdims=True)

        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Features, shape (n_samples, n_features).
            y: True labels, shape (n_samples,).

        Returns:
            accuracy: Fraction of correct predictions, range [0, 1].
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _log_posterior(self, X: np.ndarray) -> np.ndarray:
        """
        Compute log posterior probabilities for each class.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            log_posterior: Log P(y|X) for each class, shape (n_samples, n_classes).

        Formula:
            log P(y|X) = log P(y) + Σᵢ log P(xᵢ|y)
            where P(xᵢ|y) = N(xᵢ; μᵢy, σ²ᵢy) (Gaussian PDF)

        Gaussian log PDF:
            log N(x; μ, σ²) = -½ log(2πσ²) - (x - μ)² / (2σ²)
        """
        if self.theta_ is None or self.var_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Initialize log posterior with log prior: log P(y)
        log_posterior = np.log(self.class_prior_)

        # For each class, compute log likelihood: Σᵢ log P(xᵢ|y)
        for c in range(n_classes):
            # Compute Gaussian log PDF for all features and samples
            # log P(xᵢ|y=c) = -½ log(2πσ²ᵢc) - (xᵢ - μᵢc)² / (2σ²ᵢc)

            # Shape: (n_features,) broadcast to (n_samples, n_features)
            mean_c = self.theta_[c, :]
            var_c = self.var_[c, :]

            # Compute log PDF components
            # Constant term: -½ log(2πσ²)
            log_pdf_constant = -0.5 * np.log(2 * np.pi * var_c)

            # Squared deviation term: -(x - μ)² / (2σ²)
            log_pdf_exp = -0.5 * ((X - mean_c) ** 2) / var_c

            # Sum log PDF: log P(xᵢ|y) for all features
            # Shape: (n_samples,)
            log_likelihood_c = np.sum(log_pdf_constant + log_pdf_exp, axis=1)

            # Add to log posterior: log P(y) + Σ log P(xᵢ|y)
            log_posterior[c] += log_likelihood_c

        # Transpose to shape (n_samples, n_classes)
        return log_posterior.T


if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler

    print("=" * 70)
    print("Gaussian Naive Bayes: Demonstration")
    print("=" * 70)

    # Example 1: Synthetic 2-class data
    print("\n1. Binary Classification (Synthetic Data)")
    print("-" * 70)

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Note: Gaussian NB doesn't require feature scaling (probabilities are scale-invariant)
    # but it can help with numerical stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gnb = GaussianNaiveBayes()
    gnb.fit(X_train_scaled, y_train)

    y_pred = gnb.predict(X_test_scaled)
    y_proba = gnb.predict_proba(X_test_scaled)

    print(f"Training Accuracy: {gnb.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {gnb.score(X_test_scaled, y_test):.4f}")

    print(f"\nPredicted probabilities (first 5 samples):")
    print(f"{'P(y=0)':<10} {'P(y=1)':<10} {'True':<8} {'Pred'}")
    print("-" * 40)
    for i in range(min(5, len(y_test))):
        print(f"{y_proba[i, 0]:<10.4f} {y_proba[i, 1]:<10.4f} {y_test[i]:<8} {y_pred[i]}")

    # Example 2: Iris dataset (multi-class)
    print("\n" + "=" * 70)
    print("2. Multi-Class Classification (Iris Dataset)")
    print("-" * 70)

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_iris, y_iris, test_size=0.3, random_state=42
    )

    gnb_iris = GaussianNaiveBayes()
    gnb_iris.fit(X_train, y_train)

    y_pred = gnb_iris.predict(X_test)

    print(f"Training Accuracy: {gnb_iris.score(X_train, y_train):.4f}")
    print(f"Test Accuracy: {gnb_iris.score(X_test, y_test):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Show learned parameters for one class
    print("\n" + "=" * 70)
    print("3. Learned Parameters (Class 0 - Setosa)")
    print("-" * 70)
    print(f"{'Feature':<20} {'Mean (μ)':<15} {'Variance (σ²)'}")
    print("-" * 50)
    for i, feature_name in enumerate(iris.feature_names):
        mean = gnb_iris.theta_[0, i]
        var = gnb_iris.var_[0, i]
        print(f"{feature_name:<20} {mean:<15.4f} {var:.4f}")

    print(f"\nClass prior P(y=0): {gnb_iris.class_prior_[0]:.4f}")

    print("\n" + "=" * 70)
    print("Key Insights")
    print("=" * 70)
    print("• Gaussian NB assumes features are conditionally independent")
    print("• Works well even when independence assumption is violated")
    print("• Fast training (just compute means and variances)")
    print("• Probability estimates may be poorly calibrated")
    print("• Works best with continuous, approximately Gaussian features")
    print("\n→ Excellent baseline for classification tasks")
    print("→ Particularly effective for small datasets and high dimensions")
