"""
Logistic Regression for Binary Classification

Implements binary logistic regression with gradient descent and L2 regularization.

Mathematical Foundation:
- Model: P(y=1|x) = σ(w^T x) where σ(z) = 1/(1 + e^(-z))
- Loss: Log-loss (binary cross-entropy)
  L(w) = -(1/n) Σ[y log(p) + (1-y) log(1-p)] + λ||w||²
- Gradient: ∇L = -(1/n) X^T (y - p) + 2λw
  where p = σ(Xw) are predicted probabilities

Logistic regression models log-odds as linear function of features,
then maps to probabilities using sigmoid function.
"""

import numpy as np
from typing import Optional, Literal


class LogisticRegression:
    """
    Binary Logistic Regression with L2 regularization.

    Logistic regression predicts probability of binary class membership using
    sigmoid function applied to linear combination of features.

    Attributes:
        weights_ (np.ndarray): Learned weight vector including bias.
        n_iterations_ (int): Number of gradient descent iterations performed.
        losses_ (list[float]): Training loss history.

    Mathematical Notes:
        - Sigmoid σ(z) maps R → (0,1), giving valid probabilities
        - Log-loss is convex, guaranteeing global optimum
        - Decision boundary is hyperplane where w^T x = 0
        - Coefficients show effect on log-odds, not probability directly
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        alpha: float = 0.0,
        fit_intercept: bool = True,
        verbose: bool = False
    ) -> None:
        """
        Initialize Logistic Regression.

        Args:
            learning_rate: Step size for gradient descent (α in update rule).
                Typical values: 0.001 - 0.1
            max_iterations: Maximum gradient descent iterations.
            tolerance: Convergence threshold (stop if loss change < tolerance).
            alpha: L2 regularization strength (λ).
                - α = 0: No regularization
                - α > 0: Ridge penalty to prevent overfitting
            fit_intercept: If True, add bias term (recommended).
            verbose: If True, print training progress.

        Notes:
            Gradient descent is required (no closed-form solution like linear regression).
            Consider using adaptive methods (Adam) for faster convergence.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        # Learned parameters
        self.weights_: Optional[np.ndarray] = None
        self.n_iterations_: int = 0
        self.losses_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit logistic regression using gradient descent.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels {0, 1}, shape (n_samples,).

        Returns:
            self: Fitted model instance.

        Raises:
            ValueError: If y contains values other than 0 and 1.

        Algorithm:
            1. Initialize weights to zeros
            2. Repeat until convergence:
                a. Compute probabilities: p = σ(Xw)
                b. Compute log-loss: L = -mean[y log(p) + (1-y) log(1-p)]
                c. Compute gradient: ∇L = -(1/n) X^T (y - p) + 2λw
                d. Update: w := w - α * ∇L
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same n_samples")

        # Check y is binary {0, 1}
        unique_classes = np.unique(y)
        if not np.array_equal(unique_classes, [0, 1]):
            raise ValueError(f"y must contain only 0 and 1, got {unique_classes}")

        # Add bias column if needed
        if self.fit_intercept:
            X_with_bias = self._add_bias(X)
        else:
            X_with_bias = X

        # Initialize weights
        n_features = X_with_bias.shape[1]
        self.weights_ = np.zeros(n_features)

        # Gradient descent optimization
        self.weights_ = self._gradient_descent(X_with_bias, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            probabilities: P(y=1|X) for each sample, shape (n_samples,).
                Values in range (0, 1).

        Formula:
            P(y=1|x) = σ(w^T x) = 1 / (1 + exp(-w^T x))
        """
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        if self.fit_intercept:
            X_with_bias = self._add_bias(X)
        else:
            X_with_bias = X

        # Compute logits: z = Xw
        logits = X_with_bias @ self.weights_

        # Apply sigmoid: σ(z)
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.

        Args:
            X: Features, shape (n_samples, n_features).
            threshold: Decision threshold (default 0.5).
                Classify as 1 if P(y=1|x) >= threshold, else 0.

        Returns:
            predictions: Binary labels {0, 1}, shape (n_samples,).

        Notes:
            Threshold can be adjusted for imbalanced classes or cost-sensitive scenarios.
            - Lower threshold: More samples predicted as class 1 (higher recall, lower precision)
            - Higher threshold: Fewer samples predicted as class 1 (lower recall, higher precision)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Features, shape (n_samples, n_features).
            y: True labels {0, 1}, shape (n_samples,).

        Returns:
            accuracy: Fraction of correct predictions, range [0, 1].

        Note:
            Accuracy can be misleading for imbalanced datasets.
            Consider using F1-score, ROC-AUC, or other metrics.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _gradient_descent(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Optimize weights using batch gradient descent.

        Args:
            X: Features with bias, shape (n_samples, n_features+1).
            y: Binary labels, shape (n_samples,).

        Returns:
            weights: Optimized weight vector, shape (n_features+1,).

        Loss:
            L(w) = -(1/n) Σ[yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)] + λ||w||²

        Gradient:
            ∇L = -(1/n) X^T (y - p) + 2λw
            where p = σ(Xw) are predicted probabilities

        This is remarkably similar to linear regression gradient,
        but with probabilities p instead of predictions ŷ.
        """
        n_samples = X.shape[0]
        weights = self.weights_.copy()
        prev_loss = float('inf')

        for iteration in range(self.max_iterations):
            # Compute probabilities: p = σ(Xw)
            logits = X @ weights
            probas = self._sigmoid(logits)

            # Compute binary cross-entropy loss
            # Clip probabilities to avoid log(0)
            probas_clipped = np.clip(probas, 1e-15, 1 - 1e-15)
            log_loss = -np.mean(y * np.log(probas_clipped) + (1 - y) * np.log(1 - probas_clipped))

            # Add L2 regularization term
            l2_penalty = self.alpha * np.sum(weights ** 2)
            loss = log_loss + l2_penalty
            self.losses_.append(loss)

            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}, loss={loss:.6f}")
                break

            # Compute gradient: ∇L = -(1/n) X^T (y - p) + 2λw
            data_gradient = -(1 / n_samples) * X.T @ (y - probas)
            reg_gradient = 2 * self.alpha * weights

            # Don't regularize bias term (first weight if fit_intercept=True)
            if self.fit_intercept:
                reg_gradient[0] = 0

            gradient = data_gradient + reg_gradient

            # Update weights: w := w - α * ∇L
            weights -= self.learning_rate * gradient

            prev_loss = loss

            if self.verbose and (iteration + 1) % 100 == 0:
                accuracy = np.mean((probas >= 0.5).astype(int) == y)
                print(f"Iter {iteration + 1}: Loss={loss:.6f}, Accuracy={accuracy:.4f}")

        self.n_iterations_ = iteration + 1
        return weights

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid (logistic) function.

        Args:
            z: Input values (logits), any shape.

        Returns:
            sigmoid: σ(z) = 1 / (1 + exp(-z)), same shape as z.
                Values in range (0, 1).

        Properties:
            - σ(0) = 0.5 (decision boundary)
            - σ(z) → 1 as z → ∞
            - σ(z) → 0 as z → -∞
            - σ(-z) = 1 - σ(z) (symmetry)
            - σ'(z) = σ(z)(1 - σ(z)) (convenient derivative)

        Numerical Stability:
            For large |z|, exp(-z) can overflow/underflow.
            Use stable formulation: σ(z) = exp(z) / (1 + exp(z)) for z < 0
        """
        # Numerically stable sigmoid
        # Avoid overflow: for z < 0, use exp(z)/(1 + exp(z))
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        """Add bias column to feature matrix."""
        n_samples = X.shape[0]
        return np.hstack([np.ones((n_samples, 1)), X])


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    print("=" * 70)
    print("Logistic Regression: Binary Classification")
    print("=" * 70)

    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features (helps convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    print("\n1. Training Logistic Regression")
    print("-" * 70)

    lr = LogisticRegression(
        learning_rate=0.1,
        max_iterations=2000,
        alpha=0.01,  # Small L2 regularization
        verbose=False
    )
    lr.fit(X_train_scaled, y_train)

    print(f"Converged in {lr.n_iterations_} iterations")
    print(f"Final loss: {lr.losses_[-1]:.6f}")
    print(f"Initial loss: {lr.losses_[0]:.6f}")
    print(f"Loss reduction: {lr.losses_[0] - lr.losses_[-1]:.6f}")

    # Predictions
    print("\n2. Model Evaluation")
    print("-" * 70)

    y_pred_train = lr.predict(X_train_scaled)
    y_pred_test = lr.predict(X_test_scaled)
    y_proba_test = lr.predict_proba(X_test_scaled)

    print(f"Training Accuracy: {lr.score(X_train_scaled, y_train):.4f}")
    print(f"Test Accuracy: {lr.score(X_test_scaled, y_test):.4f}")

    # Confusion matrix
    print("\n3. Confusion Matrix (Test Set)")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)
    print(f"\nTrue Negatives (TN): {cm[0, 0]}")
    print(f"False Positives (FP): {cm[0, 1]}")
    print(f"False Negatives (FN): {cm[1, 0]}")
    print(f"True Positives (TP): {cm[1, 1]}")

    # Classification metrics
    print("\n4. Detailed Metrics")
    print("-" * 70)
    print(classification_report(y_test, y_pred_test, target_names=['Class 0', 'Class 1']))

    # Probability calibration check
    print("\n5. Predicted Probabilities (first 10 test samples)")
    print("-" * 70)
    print(f"{'True Label':<12} {'Pred Label':<12} {'P(y=1)':<12} {'Correct'}")
    print("-" * 70)
    for i in range(min(10, len(y_test))):
        true_label = y_test[i]
        pred_label = y_pred_test[i]
        proba = y_proba_test[i]
        correct = "✓" if true_label == pred_label else "✗"
        print(f"{true_label:<12} {pred_label:<12} {proba:<12.4f} {correct}")

    # Compare regularization strengths
    print("\n" + "=" * 70)
    print("6. Effect of L2 Regularization")
    print("=" * 70)
    print(f"{'Alpha (λ)':<12} {'Train Acc':<12} {'Test Acc':<12} {'||w||² (norm)'}")
    print("-" * 70)

    for alpha in [0.0, 0.001, 0.01, 0.1, 1.0]:
        lr_reg = LogisticRegression(learning_rate=0.1, max_iterations=2000, alpha=alpha, verbose=False)
        lr_reg.fit(X_train_scaled, y_train)

        train_acc = lr_reg.score(X_train_scaled, y_train)
        test_acc = lr_reg.score(X_test_scaled, y_test)
        weight_norm = np.sum(lr_reg.weights_ ** 2)

        print(f"{alpha:<12.3f} {train_acc:<12.4f} {test_acc:<12.4f} {weight_norm:.4f}")

    print("\nKey Observations:")
    print("• α = 0: No regularization, may overfit (high ||w||²)")
    print("• α > 0: Regularization shrinks weights, prevents overfitting")
    print("• Trade-off: Regularization may slightly decrease training accuracy")
    print("  but often improves test accuracy (better generalization)")
