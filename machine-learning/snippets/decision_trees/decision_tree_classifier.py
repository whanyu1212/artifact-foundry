"""
Decision Tree Classifier - Classification using CART algorithm
==============================================================
Uses information gain (Gini or Entropy) to find optimal splits.
Predicts the most common class in each leaf node.
"""

import numpy as np
from collections import Counter
from typing import Optional
from .base_tree import BaseDecisionTree
from .tree_metrics import TreeMetrics


class DecisionTreeClassifier(BaseDecisionTree):
    """
    Decision Tree for classification tasks.

    Splits using information gain:
        IG = Parent_Impurity - Weighted_Children_Impurity

    Impurity measures:
        - Gini: Faster, no logarithms, range [0, 0.5] for binary
        - Entropy: Information-theoretic, slightly more balanced trees

    Leaf prediction:
        - Mode (most common class in leaf)

    Args:
        criterion (str): 'gini' or 'entropy' (default: 'gini')
        max_depth (int): Maximum tree depth, None = unlimited (default: None)
        min_samples_split (int): Min samples to split node (default: 2)
        min_samples_leaf (int): Min samples in leaf node (default: 1)

    Attributes (set during fit):
        n_classes_ (int): Number of unique classes
        n_features_ (int): Number of features
        n_samples_ (int): Number of training samples
        root (Node): Root node of fitted tree

    Example:
        >>> clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> print(predictions)  # [0, 1, 1, 0, ...]
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        # Initialize parent class
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

        # Validate criterion
        if criterion not in ["gini", "entropy"]:
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got '{criterion}'"
            )

        self.criterion = criterion
        self.n_classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """
        Build classifier from training data.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Training labels (integers), shape (n_samples,)

        Returns:
            self: Fitted classifier (allows method chaining)

        Raises:
            ValueError: If y contains non-integer values
        """
        # Store number of classes
        self.n_classes_ = len(np.unique(y))

        # Call parent fit (builds tree)
        return super().fit(X, y)

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        Calculate impurity of a node using Gini or Entropy.

        Gini Impurity:
            Gini = 1 - Σ(p_i²)
            where p_i = proportion of class i

        Entropy:
            H = -Σ(p_i × log₂(p_i))
            where p_i = proportion of class i

        Args:
            y (np.ndarray): Class labels at this node

        Returns:
            float: Impurity value
                - Gini: [0, 1-1/n_classes], 0 = pure
                - Entropy: [0, log₂(n_classes)], 0 = pure

        Example:
            >>> y = np.array([0, 0, 1, 1, 1])  # 2 class-0, 3 class-1
            >>> # Gini = 1 - (0.4² + 0.6²) = 1 - 0.52 = 0.48
            >>> # Entropy = -(0.4×log₂(0.4) + 0.6×log₂(0.6)) ≈ 0.97
        """
        y_list = y.tolist()

        # Delegate to TreeMetrics utility for calculation
        if self.criterion == "gini":
            return TreeMetrics.gini(y_list)
        else:  # entropy
            return TreeMetrics.entropy(y_list)

    def _calculate_leaf_value(self, y: np.ndarray) -> int:
        """
        Calculate prediction for a leaf node (most common class).

        Uses mode: the class that appears most frequently.

        Args:
            y (np.ndarray): Class labels at this leaf

        Returns:
            int: Most common class label

        Example:
            >>> y = np.array([0, 0, 1, 0, 1])
            >>> # Class 0 appears 3 times, class 1 appears 2 times
            >>> # Returns 0
        """
        # Counter.most_common(1) returns list: [(label, count)]
        # Extract just the label with [0][0]
        return Counter(y.tolist()).most_common(1)[0][0]

    def _calculate_split_gain(
        self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """
        Calculate information gain from a split.

        Information Gain:
            IG = Impurity(parent) - Weighted_Impurity(children)

        where:
            Weighted_Impurity = (n_left/n_total) × Impurity(left) +
                                (n_right/n_total) × Impurity(right)

        Higher gain = better split (more reduction in impurity)

        Args:
            y_parent (np.ndarray): Labels before split
            y_left (np.ndarray): Labels in left child (feature <= threshold)
            y_right (np.ndarray): Labels in right child (feature > threshold)

        Returns:
            float: Information gain, range [0, parent_impurity]

        Example:
            >>> y_parent = np.array([0, 0, 1, 1, 1])  # Gini ≈ 0.48
            >>> y_left = np.array([0, 0])              # Gini = 0 (pure!)
            >>> y_right = np.array([1, 1, 1])          # Gini = 0 (pure!)
            >>> # IG = 0.48 - (2/5×0 + 3/5×0) = 0.48 (perfect split!)
        """
        return TreeMetrics.information_gain(
            parent=y_parent.tolist(),
            left=y_left.tolist(),
            right=y_right.tolist(),
            mode=self.criterion,  # 'gini' or 'entropy'
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Note: This is a simple implementation that returns 1.0 for the
        predicted class and 0.0 for others. A more sophisticated version
        would track class distributions in each leaf.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Probabilities, shape (n_samples, n_classes)

        Example:
            >>> probs = clf.predict_proba(X_test)
            >>> # probs[i, j] = probability sample i belongs to class j
        """
        predictions = self.predict(X)
        n_samples = X.shape[0]

        # Create probability matrix (one-hot encoding)
        proba = np.zeros((n_samples, self.n_classes_))
        for i, pred in enumerate(predictions):
            proba[i, pred] = 1.0

        return proba
