"""
K-Nearest Neighbors (KNN) Classifier and Regressor

Implements KNN for both classification and regression using various distance metrics.

Mathematical Foundation:
- Classification: ŷ = argmax_y Σᵢ∈Nₖ(x) I(yᵢ = y)
- Regression: ŷ = (1/K) Σᵢ∈Nₖ(x) yᵢ
- Weighted: Use inverse distance weighting for closer neighbors

Algorithm:
1. Store all training data (lazy learning)
2. For prediction, compute distance to all training points
3. Find K nearest neighbors
4. Classification: majority vote; Regression: average

Time Complexity:
- Training: O(1) (just store data)
- Prediction: O(nd) where n = training samples, d = features
"""

import numpy as np
from typing import Literal, Optional
from collections import Counter


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier.

    Classifies test samples based on majority voting among K nearest
    training samples. Supports multiple distance metrics and weighted voting.

    Attributes:
        n_neighbors (int): Number of neighbors to use for voting.
        metric (str): Distance metric ('euclidean', 'manhattan', 'minkowski', 'cosine').
        weights (str): Weight function ('uniform' or 'distance').
        p (float): Parameter for Minkowski metric (p=1: Manhattan, p=2: Euclidean).
        X_train_ (np.ndarray): Stored training features.
        y_train_ (np.ndarray): Stored training labels.
        classes_ (np.ndarray): Unique class labels.

    Notes:
        - Always scale features before using KNN (StandardScaler recommended)
        - Use odd K for binary classification to avoid ties
        - K=√n is a common heuristic starting point
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "manhattan", "minkowski", "cosine"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
        p: float = 2.0,
    ) -> None:
        """
        Initialize K-Nearest Neighbors Classifier.

        Args:
            n_neighbors: Number of neighbors to use for prediction.
                Smaller K → more complex boundary, higher variance.
                Larger K → smoother boundary, higher bias.
            metric: Distance metric to use.
                'euclidean': L2 norm, √(Σ(xᵢ - x'ᵢ)²)
                'manhattan': L1 norm, Σ|xᵢ - x'ᵢ|
                'minkowski': Generalized Lp norm, (Σ|xᵢ - x'ᵢ|ᵖ)^(1/p)
                'cosine': 1 - (x·x')/(||x|| ||x'||)
            weights: Weight function for voting.
                'uniform': All neighbors have equal weight (majority vote)
                'distance': Weight by inverse distance (closer = more influence)
            p: Parameter for Minkowski metric (only used if metric='minkowski').

        Raises:
            ValueError: If n_neighbors < 1 or p < 1.
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if p < 1:
            raise ValueError("p must be at least 1 for Minkowski distance")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.p = p

        # Learned parameters (set during fit)
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        Fit the K-Nearest Neighbors classifier (store training data).

        KNN is a lazy learner - it doesn't build a model during training,
        just stores the data for use during prediction.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training labels, shape (n_samples,).

        Returns:
            self: Fitted classifier instance.

        Notes:
            Training time is O(1) - just store the data.
            All computation happens during prediction.
        """
        # Store training data (lazy learning)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        # Store unique classes for prediction
        self.classes_ = np.unique(y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Features to predict, shape (n_samples, n_features).

        Returns:
            predictions: Predicted class labels, shape (n_samples,).

        Algorithm:
            For each test point:
            1. Compute distances to all training points
            2. Find K nearest neighbors
            3. Vote: majority class (or weighted by distance)

        Time Complexity: O(n_test * n_train * d)
        """
        if self.X_train_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=self.y_train_.dtype)

        # Predict each test sample
        for i in range(n_samples):
            # Find K nearest neighbors
            neighbor_indices, distances = self._find_k_nearest(X[i])

            # Get labels of K nearest neighbors
            neighbor_labels = self.y_train_[neighbor_indices]

            # Vote for class (uniform or distance-weighted)
            if self.weights == "uniform":
                # Majority vote: most common class
                predictions[i] = self._majority_vote(neighbor_labels)
            else:  # self.weights == "distance"
                # Weighted vote: closer neighbors have more influence
                predictions[i] = self._weighted_vote(neighbor_labels, distances)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Features to predict, shape (n_samples, n_features).

        Returns:
            probabilities: Class probabilities, shape (n_samples, n_classes).
                Each row sums to 1.0.

        Notes:
            Probabilities are estimated as:
            - Uniform: P(y|x) = (count of class y in neighbors) / K
            - Distance: P(y|x) = (sum of weights for class y) / (sum of all weights)
        """
        if self.X_train_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            neighbor_indices, distances = self._find_k_nearest(X[i])
            neighbor_labels = self.y_train_[neighbor_indices]

            if self.weights == "uniform":
                # Count votes for each class
                for j, cls in enumerate(self.classes_):
                    probabilities[i, j] = np.sum(neighbor_labels == cls) / self.n_neighbors
            else:  # self.weights == "distance"
                # Weight votes by inverse distance
                weights = self._compute_weights(distances)
                total_weight = np.sum(weights)

                for j, cls in enumerate(self.classes_):
                    class_mask = neighbor_labels == cls
                    probabilities[i, j] = np.sum(weights[class_mask]) / total_weight

        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Args:
            X: Test features, shape (n_samples, n_features).
            y: True labels, shape (n_samples,).

        Returns:
            accuracy: Fraction of correctly classified samples.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def _find_k_nearest(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Find K nearest neighbors to a test point.

        Args:
            x: Test sample, shape (n_features,).

        Returns:
            indices: Indices of K nearest neighbors in training set, shape (K,).
            distances: Distances to K nearest neighbors, shape (K,).

        Algorithm:
            1. Compute distances to all training points: O(nd)
            2. Partially sort to find K smallest: O(n log K) using argpartition
        """
        # Compute distances to all training points
        distances = self._compute_distances(x, self.X_train_)

        # Find indices of K smallest distances
        # argpartition is O(n) instead of O(n log n) for full sort
        if self.n_neighbors < len(distances):
            # Partition: elements before k are smaller than elements after k
            partition_indices = np.argpartition(distances, self.n_neighbors - 1)
            k_indices = partition_indices[: self.n_neighbors]

            # Sort the K nearest (for consistent tie-breaking and distance weighting)
            k_indices = k_indices[np.argsort(distances[k_indices])]
        else:
            # If K >= n_train, use all points
            k_indices = np.argsort(distances)

        k_distances = distances[k_indices]

        return k_indices, k_distances

    def _compute_distances(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Compute distances from a point to multiple points.

        Args:
            x: Single point, shape (n_features,).
            X: Multiple points, shape (n_samples, n_features).

        Returns:
            distances: Distances from x to each point in X, shape (n_samples,).
        """
        if self.metric == "euclidean":
            # Euclidean: L2 norm, √(Σ(xᵢ - x'ᵢ)²)
            return np.sqrt(np.sum((X - x) ** 2, axis=1))

        elif self.metric == "manhattan":
            # Manhattan: L1 norm, Σ|xᵢ - x'ᵢ|
            return np.sum(np.abs(X - x), axis=1)

        elif self.metric == "minkowski":
            # Minkowski: Generalized Lp norm, (Σ|xᵢ - x'ᵢ|ᵖ)^(1/p)
            return np.sum(np.abs(X - x) ** self.p, axis=1) ** (1 / self.p)

        elif self.metric == "cosine":
            # Cosine distance: 1 - cos(θ) = 1 - (x·x')/(||x|| ||x'||)
            # Range: [0, 2], where 0 = identical direction, 2 = opposite
            dot_products = X @ x
            x_norm = np.linalg.norm(x)
            X_norms = np.linalg.norm(X, axis=1)

            # Avoid division by zero
            denominator = X_norms * x_norm
            cosine_similarity = np.where(
                denominator > 0, dot_products / denominator, 0.0
            )

            # Convert similarity to distance
            return 1 - cosine_similarity

        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    @staticmethod
    def _majority_vote(labels: np.ndarray) -> int:
        """
        Return the most common label (majority vote).

        Args:
            labels: Labels of neighbors, shape (K,).

        Returns:
            majority_label: Most frequent label.

        Notes:
            Uses Counter for O(K) time complexity.
            In case of tie, returns the label that appears first.
        """
        # Counter.most_common(1) returns [(label, count)]
        return Counter(labels).most_common(1)[0][0]

    def _weighted_vote(self, labels: np.ndarray, distances: np.ndarray) -> int:
        """
        Return class with highest weighted vote (inverse distance weighting).

        Args:
            labels: Labels of neighbors, shape (K,).
            distances: Distances to neighbors, shape (K,).

        Returns:
            weighted_majority: Class with highest total weight.

        Algorithm:
            For each class, sum weights of neighbors belonging to that class.
            Weight = 1/distance (closer neighbors have more influence).
        """
        weights = self._compute_weights(distances)

        # Sum weights for each class
        class_weights = {}
        for label, weight in zip(labels, weights):
            class_weights[label] = class_weights.get(label, 0.0) + weight

        # Return class with highest total weight
        return max(class_weights, key=class_weights.get)

    @staticmethod
    def _compute_weights(distances: np.ndarray) -> np.ndarray:
        """
        Compute inverse distance weights.

        Args:
            distances: Distances to neighbors, shape (K,).

        Returns:
            weights: Inverse distance weights, shape (K,).

        Formula:
            w = 1 / distance (if distance > 0)
            w = 1.0 (if distance = 0, exact match)

        Notes:
            Exact matches (distance=0) get weight 1.0.
            Add small epsilon for numerical stability.
        """
        epsilon = 1e-10
        return 1.0 / (distances + epsilon)


class KNNRegressor:
    """
    K-Nearest Neighbors Regressor.

    Predicts continuous values by averaging (or weighted averaging) the
    target values of K nearest training samples.

    Attributes:
        n_neighbors (int): Number of neighbors to use for averaging.
        metric (str): Distance metric ('euclidean', 'manhattan', 'minkowski', 'cosine').
        weights (str): Weight function ('uniform' or 'distance').
        p (float): Parameter for Minkowski metric.
        X_train_ (np.ndarray): Stored training features.
        y_train_ (np.ndarray): Stored training targets.

    Notes:
        - Always scale features before using KNN
        - Use cross-validation to select optimal K
        - Weighted averaging often performs better than uniform
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "manhattan", "minkowski", "cosine"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
        p: float = 2.0,
    ) -> None:
        """
        Initialize K-Nearest Neighbors Regressor.

        Args:
            n_neighbors: Number of neighbors to use for prediction.
            metric: Distance metric to use.
            weights: Weight function for averaging.
                'uniform': Simple average of K neighbors
                'distance': Weighted average (closer = more weight)
            p: Parameter for Minkowski metric.
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if p < 1:
            raise ValueError("p must be at least 1 for Minkowski distance")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.p = p

        # Learned parameters
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        """
        Fit the K-Nearest Neighbors regressor (store training data).

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).

        Returns:
            self: Fitted regressor instance.
        """
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict continuous values for samples in X.

        Args:
            X: Features to predict, shape (n_samples, n_features).

        Returns:
            predictions: Predicted values, shape (n_samples,).

        Formula:
            Uniform: ŷ = (1/K) Σᵢ∈Nₖ(x) yᵢ
            Distance-weighted: ŷ = Σᵢ∈Nₖ(x) wᵢyᵢ / Σᵢ∈Nₖ(x) wᵢ
        """
        if self.X_train_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        n_samples = X.shape[0]
        predictions = np.empty(n_samples)

        # Reuse distance computation from classifier
        temp_classifier = KNNClassifier(
            n_neighbors=self.n_neighbors, metric=self.metric, p=self.p
        )
        temp_classifier.X_train_ = self.X_train_
        temp_classifier.y_train_ = self.y_train_

        for i in range(n_samples):
            neighbor_indices, distances = temp_classifier._find_k_nearest(X[i])
            neighbor_values = self.y_train_[neighbor_indices]

            if self.weights == "uniform":
                # Simple average
                predictions[i] = np.mean(neighbor_values)
            else:  # self.weights == "distance"
                # Weighted average
                weights = KNNClassifier._compute_weights(distances)
                predictions[i] = np.sum(weights * neighbor_values) / np.sum(weights)

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score (coefficient of determination).

        Args:
            X: Test features, shape (n_samples, n_features).
            y: True target values, shape (n_samples,).

        Returns:
            r2_score: R² = 1 - SS_res / SS_tot
                R² = 1: perfect predictions
                R² = 0: predicts mean
                R² < 0: worse than predicting mean

        Formula:
            R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
        """
        y_pred = self.predict(X)

        # Residual sum of squares: SS_res = Σ(y - ŷ)²
        ss_res = np.sum((y - y_pred) ** 2)

        # Total sum of squares: SS_tot = Σ(y - ȳ)²
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # R² = 1 - SS_res / SS_tot
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_diabetes, make_classification
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, mean_squared_error
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70)
    console.print("K-Nearest Neighbors: Classification & Regression")
    console.print("=" * 70 + "[/bold cyan]")

    # Example 1: Classification on Iris
    console.print("\n[bold yellow]1. KNN Classification (Iris Dataset)[/bold yellow]")
    console.print("-" * 70)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # CRITICAL: Scale features before KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN classifier
    knn = KNNClassifier(n_neighbors=5, metric="euclidean", weights="distance")
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)
    y_proba = knn.predict_proba(X_test_scaled)

    console.print(f"Training Accuracy: [green]{knn.score(X_train_scaled, y_train):.4f}[/green]")
    console.print(f"Test Accuracy: [green]{knn.score(X_test_scaled, y_test):.4f}[/green]")

    console.print("\n[bold]Classification Report:[/bold]")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Example 2: Compare Distance Metrics
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("2. Distance Metric Comparison")
    console.print("=" * 70 + "[/bold cyan]")

    metrics = ["euclidean", "manhattan", "cosine"]
    results_table = Table(title="Distance Metrics Performance", box=box.ROUNDED)
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Train Acc", justify="right", style="green")
    results_table.add_column("Test Acc", justify="right", style="yellow")

    for metric in metrics:
        knn_metric = KNNClassifier(n_neighbors=5, metric=metric, weights="distance")
        knn_metric.fit(X_train_scaled, y_train)

        train_acc = knn_metric.score(X_train_scaled, y_train)
        test_acc = knn_metric.score(X_test_scaled, y_test)

        results_table.add_row(metric.capitalize(), f"{train_acc:.4f}", f"{test_acc:.4f}")

    console.print(results_table)

    # Example 3: Effect of K
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("3. Effect of K (Number of Neighbors)")
    console.print("=" * 70 + "[/bold cyan]")

    k_values = [1, 3, 5, 7, 9, 15, 21]
    k_table = Table(title="K Selection Impact", box=box.ROUNDED)
    k_table.add_column("K", justify="center", style="cyan")
    k_table.add_column("CV Score (mean)", justify="right", style="green")
    k_table.add_column("CV Score (std)", justify="right", style="yellow")

    best_k = None
    best_score = 0.0

    for k in k_values:
        knn_k = KNNClassifier(n_neighbors=k, weights="distance")
        # Use sklearn's cross_val_score (our KNN implements fit/predict interface)
        scores = cross_val_score(knn_k, X_train_scaled, y_train, cv=5)

        mean_score = scores.mean()
        std_score = scores.std()

        k_table.add_row(str(k), f"{mean_score:.4f}", f"{std_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    console.print(k_table)
    console.print(f"\n[bold green]Best K: {best_k}[/bold green] (CV Score: {best_score:.4f})")

    # Example 4: Regression
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("4. KNN Regression (Diabetes Dataset)")
    console.print("=" * 70 + "[/bold cyan]")

    diabetes = load_diabetes()
    X_reg, y_reg = diabetes.data, diabetes.target

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )

    # Scale features
    scaler_reg = StandardScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)

    # Compare uniform vs distance weighting
    reg_table = Table(title="KNN Regression: Weighting Comparison", box=box.ROUNDED)
    reg_table.add_column("Weighting", style="cyan")
    reg_table.add_column("R² Score", justify="right", style="green")
    reg_table.add_column("RMSE", justify="right", style="yellow")

    for weight_type in ["uniform", "distance"]:
        knn_reg = KNNRegressor(n_neighbors=5, weights=weight_type)
        knn_reg.fit(X_train_reg_scaled, y_train_reg)

        y_pred_reg = knn_reg.predict(X_test_reg_scaled)
        r2 = knn_reg.score(X_test_reg_scaled, y_test_reg)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

        reg_table.add_row(weight_type.capitalize(), f"{r2:.4f}", f"{rmse:.2f}")

    console.print(reg_table)

    # Example 5: Compare with Scikit-Learn
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("5. Validation Against Scikit-Learn")
    console.print("=" * 70 + "[/bold cyan]")

    # Classification comparison
    our_knn = KNNClassifier(n_neighbors=5, metric="euclidean", weights="distance")
    our_knn.fit(X_train_scaled, y_train)

    sklearn_knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="distance")
    sklearn_knn.fit(X_train_scaled, y_train)

    our_acc = our_knn.score(X_test_scaled, y_test)
    sklearn_acc = sklearn_knn.score(X_test_scaled, y_test)

    comparison_table = Table(title="Implementation Comparison", box=box.ROUNDED)
    comparison_table.add_column("Implementation", style="cyan")
    comparison_table.add_column("Accuracy", justify="right", style="green")

    comparison_table.add_row("Our KNN", f"{our_acc:.4f}")
    comparison_table.add_row("Scikit-Learn", f"{sklearn_acc:.4f}")
    comparison_table.add_row("Difference", f"{abs(our_acc - sklearn_acc):.6f}")

    console.print(comparison_table)

    # Key insights
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Key Insights")
    console.print("=" * 70 + "[/bold cyan]")

    insights = [
        "• KNN is a lazy learner: O(1) training, O(nd) prediction",
        "• ALWAYS scale features before using KNN (distance-based)",
        "• Small K → complex boundary (high variance, low bias)",
        "• Large K → smooth boundary (low variance, high bias)",
        "• Distance weighting often performs better than uniform",
        "• Curse of dimensionality: performance degrades in high dimensions (d > 20)",
        "• Use cross-validation to select optimal K",
        "",
        "[yellow]→ Best for:[/yellow] Small-medium datasets, non-linear boundaries, interpretability",
        "[yellow]→ Avoid when:[/yellow] Large datasets (slow), high dimensions, real-time prediction",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
