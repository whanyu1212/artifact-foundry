"""
Base Decision Tree Class
=========================================================
Implements shared functionality for both classification and regression trees.
Subclasses override methods for impurity calculation and leaf value computation.
"""

import numpy as np
from typing import Optional, Union
from .tree_node import Node


class BaseDecisionTree:
    """
    Base class implementing the CART (Classification And Regression Trees) algorithm.

    The CART algorithm:
        1. Start at root with all data
        2. Find best split (feature + threshold) using greedy search
        3. Split data: left (feature <= threshold), right (feature > threshold)
        4. Recursively build left and right subtrees
        5. Stop when stopping criteria met, create leaf node

    Stopping criteria:
        - Max depth reached
        - Too few samples to split (< min_samples_split)
        - Pure node (all targets identical)
        - No beneficial split found
        - Would create leaf with too few samples (< min_samples_leaf)

    Hyperparameters:
        max_depth: Maximum tree depth (None = unlimited, prone to overfitting)
        min_samples_split: Min samples required to split a node (default: 2)
        min_samples_leaf: Min samples required in leaf node (default: 1)

    Abstract methods (implemented by subclasses):
        _calculate_impurity(): Calculate node impurity (gini/entropy/mse/mae)
        _calculate_leaf_value(): Calculate prediction value (mode/mean/median)
        _calculate_split_gain(): Calculate split quality (info gain/var reduction)
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: Optional[Node] = None
        self.n_features_ = None  # Set during fit()
        self.n_samples_ = None  # Set during fit()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseDecisionTree":
        """
        Build decision tree from training data.

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Training targets, shape (n_samples,)

        Returns:
            self: Fitted tree (for method chaining)
        """
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]

        # Build tree recursively starting from root
        self.root = self._grow_tree(X, y, depth=0)
        # Returns self to enable method chaining
        # You can call predict() right after fit()
        # e.g., tree.fit(X, y).predict(X_test)
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively build tree using CART algorithm.

        This is the core tree-building logic:
            1. Check stopping criteria -> create leaf if met
            2. Find best split (feature + threshold)
            3. If no good split -> create leaf
            4. Otherwise -> split data and recurse on children

        Args:
            X (np.ndarray): Features at current node, shape (n_samples_node, n_features)
            y (np.ndarray): Targets at current node, shape (n_samples_node,)
            depth (int): Current depth in tree (root = 0)

        Returns:
            Node: Either internal node (with children) or leaf node
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Calculate current node's impurity
        current_impurity = self._calculate_impurity(y)

        # === STOPPING CRITERIA ===

        # Stop 1: Max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            leaf_value = self._calculate_leaf_value(y)
            return Node(
                value=leaf_value,
                impurity=current_impurity,
                n_samples=n_samples,
                depth=depth,
            )

        # Stop 2: Too few samples to split
        if n_samples < self.min_samples_split:
            leaf_value = self._calculate_leaf_value(y)
            return Node(
                value=leaf_value,
                impurity=current_impurity,
                n_samples=n_samples,
                depth=depth,
            )

        # Stop 3: Pure node (all targets identical)
        if n_labels == 1:
            leaf_value = self._calculate_leaf_value(y)
            return Node(
                value=leaf_value,
                impurity=0.0,  # Pure node has zero impurity
                n_samples=n_samples,
                depth=depth,
            )

        # === FIND BEST SPLIT ===
        best_feature, best_threshold, best_gain = self._find_best_split(X, y)

        # Stop 4: No beneficial split found
        if best_gain <= 0:
            leaf_value = self._calculate_leaf_value(y)
            return Node(
                value=leaf_value,
                impurity=current_impurity,
                n_samples=n_samples,
                depth=depth,
            )

        # === CREATE INTERNAL NODE AND RECURSE ===

        # Split data based on best feature and threshold
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        left_child = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        # Return internal node with split information
        return Node(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            impurity=current_impurity,
            n_samples=n_samples,
            depth=depth,
        )

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Find best feature and threshold to split on using greedy search.

        Algorithm (greedy, locally optimal):
            For each feature:
                For each unique value as potential threshold:
                    Split: left (feature <= threshold), right (feature > threshold)
                    Calculate split quality (information gain / variance reduction)
                    Track best split

        Args:
            X (np.ndarray): Feature matrix at current node
            y (np.ndarray): Target array at current node

        Returns:
            tuple: (best_feature_idx, best_threshold, best_gain)
                   Returns (None, None, -1) if no valid split found

        Time Complexity:
            O(n_features × n_samples × log(n_samples))
            - Dominated by unique value extraction (sorting)
            - For each feature, try O(n_samples) thresholds
            - Each split evaluation is O(n_samples)

        Note:
            This is a GREEDY algorithm - finds locally optimal split at each
            node, but doesn't guarantee globally optimal tree structure.
        """
        best_gain = -1
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        # Try each feature
        for feature_idx in range(n_features):
            # Get unique values for this feature as potential thresholds
            # np.unique() also sorts the values
            thresholds = np.unique(X[:, feature_idx])

            # Try each unique value as split threshold
            for threshold in thresholds:
                # Split data: left gets feature <= threshold
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                y_left = y[left_mask]
                y_right = y[right_mask]

                # Enforce min_samples_leaf constraint
                # Skip if either child would have too few samples
                if (
                    len(y_left) < self.min_samples_leaf
                    or len(y_right) < self.min_samples_leaf
                ):
                    continue

                # Calculate split quality using TreeMetrics
                # (information_gain for classification, variance_reduction for regression)
                gain = self._calculate_split_gain(y, y_left, y_right)

                # Track best split
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples by traversing tree.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Predictions, shape (n_samples,)

        Algorithm:
            For each sample:
                Start at root
                While not at leaf:
                    If feature[split_feature] <= threshold: go left
                    Else: go right
                Return leaf's prediction value
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> Union[int, float]:
        """
        Traverse tree to find prediction for a single sample.

        Args:
            x (np.ndarray): Single sample features, shape (n_features,)
            node (Node): Current node in traversal

        Returns:
            Union[int, float]: Prediction value from leaf node

        Algorithm (recursive):
            Base case: If at leaf -> return prediction value
            Recursive case:
                If x[split_feature] <= threshold -> recurse left
                Else -> recurse right
        """
        # Base case: reached leaf node
        if node.is_leaf():
            return node.value

        # Recursive case: traverse to appropriate child
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    # === ABSTRACT METHODS (must be implemented by subclasses) ===

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        Calculate impurity of a node.

        Subclass implementations:
            - Classification: Gini impurity or Entropy
            - Regression: Variance, MSE, or MAE
        """
        raise NotImplementedError("Subclass must implement _calculate_impurity()")

    def _calculate_leaf_value(self, y: np.ndarray) -> Union[int, float]:
        """
        Calculate prediction value for a leaf node.

        Subclass implementations:
            - Classification: Mode (most common class)
            - Regression: Mean (for MSE/Variance) or Median (for MAE)
        """
        raise NotImplementedError("Subclass must implement _calculate_leaf_value()")

    def _calculate_split_gain(
        self, y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """
        Calculate quality/gain of a split.

        Subclass implementations:
            - Classification: Information gain (using TreeMetrics)
            - Regression: Variance/MSE/MAE reduction (using TreeMetrics)
        """
        raise NotImplementedError("Subclass must implement _calculate_split_gain()")
