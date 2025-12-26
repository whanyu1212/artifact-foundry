"""
Decision Tree Regressor - Regression using CART algorithm
=========================================================
Uses variance/MSE/MAE reduction to find optimal splits.
Predicts mean (MSE) or median (MAE) of values in each leaf.
"""

import numpy as np
from typing import Optional
from .base_tree import BaseDecisionTree
from .tree_metrics import TreeMetrics


class DecisionTreeRegressor(BaseDecisionTree):
    """
    Decision Tree for regression tasks.

    Splits using variance/MSE/MAE reduction:
        Reduction = Parent_Impurity - Weighted_Children_Impurity

    Impurity measures:
        - MSE (Mean Squared Error): Most common, penalizes large errors
        - Variance: Equivalent to MSE for tree building
        - MAE (Mean Absolute Error): Robust to outliers

    Leaf prediction:
        - Mean (for MSE/Variance): Minimizes squared error
        - Median (for MAE): Minimizes absolute error

    Args:
        criterion (str): 'mse', 'mae', or 'variance' (default: 'mse')
        max_depth (int): Maximum tree depth, None = unlimited (default: None)
        min_samples_split (int): Min samples to split node (default: 2)
        min_samples_leaf (int): Min samples in leaf node (default: 1)

    Attributes (set during fit):
        n_features_ (int): Number of features
        n_samples_ (int): Number of training samples
        root (Node): Root node of fitted tree

    Example:
        >>> reg = DecisionTreeRegressor(criterion='mse', max_depth=5)
        >>> reg.fit(X_train, y_train)
        >>> predictions = reg.predict(X_test)
        >>> print(predictions)  # [123.4, 456.7, 234.5, ...]
    """

    def __init__(
        self,
        criterion: str = 'mse',
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1
    ):
        # Initialize parent class
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )

        # Validate criterion
        if criterion not in ['mse', 'mae', 'variance']:
            raise ValueError(
                f"criterion must be 'mse', 'mae', or 'variance', got '{criterion}'"
            )

        self.criterion = criterion

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        Calculate impurity of a node using MSE, MAE, or Variance.

        Mean Squared Error:
            MSE = (1/n) × Σ(y_i - mean(y))²

        Mean Absolute Error:
            MAE = (1/n) × Σ|y_i - median(y)|

        Variance:
            Var = (1/n) × Σ(y_i - mean(y))²
            (equivalent to MSE)

        Args:
            y (np.ndarray): Target values at this node

        Returns:
            float: Impurity value
                - Lower = more homogeneous (similar values)
                - 0 = all values identical (pure node)

        Example:
            >>> y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> # Mean = 3.0
            >>> # MSE = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5 = 2.0
        """
        y_list = y.tolist()

        if self.criterion == 'mse':
            return TreeMetrics.mse(y_list)
        elif self.criterion == 'mae':
            return TreeMetrics.mae(y_list)
        else:  # variance
            return TreeMetrics.variance(y_list)

    def _calculate_leaf_value(self, y: np.ndarray) -> float:
        """
        Calculate prediction for a leaf node.

        Prediction strategy:
            - MSE/Variance: Mean (minimizes squared error)
            - MAE: Median (minimizes absolute error)

        Theory:
            - The mean is the optimal constant prediction under MSE loss
            - The median is the optimal constant prediction under MAE loss

        Args:
            y (np.ndarray): Target values at this leaf

        Returns:
            float: Prediction value (mean or median)

        Example:
            >>> y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])  # Has outlier
            >>> # Mean = 22.0 (influenced by outlier)
            >>> # Median = 3.0 (robust to outlier)
        """
        if self.criterion == 'mae':
            # MAE uses median (more robust to outliers)
            return float(np.median(y))
        else:
            # MSE and Variance use mean
            return float(np.mean(y))

    def _calculate_split_gain(
        self,
        y_parent: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> float:
        """
        Calculate variance/MSE/MAE reduction from a split.

        Reduction:
            Reduction = Impurity(parent) - Weighted_Impurity(children)

        where:
            Weighted_Impurity = (n_left/n_total) × Impurity(left) +
                                (n_right/n_total) × Impurity(right)

        Higher reduction = better split (more reduction in variance/error)

        Args:
            y_parent (np.ndarray): Values before split
            y_left (np.ndarray): Values in left child (feature <= threshold)
            y_right (np.ndarray): Values in right child (feature > threshold)

        Returns:
            float: Impurity reduction, range [0, parent_impurity]

        Example:
            >>> y_parent = np.array([1, 2, 3, 9, 10])  # High variance
            >>> y_left = np.array([1, 2, 3])            # Low variance
            >>> y_right = np.array([9, 10])             # Low variance
            >>> # Good split! Separates low and high values
            >>> # Reduction will be high
        """
        y_parent_list = y_parent.tolist()
        y_left_list = y_left.tolist()
        y_right_list = y_right.tolist()

        if self.criterion == 'mse':
            return TreeMetrics.mse_reduction(
                y_parent_list, y_left_list, y_right_list
            )
        elif self.criterion == 'mae':
            return TreeMetrics.mae_reduction(
                y_parent_list, y_left_list, y_right_list
            )
        else:  # variance
            return TreeMetrics.variance_reduction(
                y_parent_list, y_left_list, y_right_list
            )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score (coefficient of determination).

        R² = 1 - (SS_res / SS_tot)

        where:
            SS_res = Σ(y_true - y_pred)²  (residual sum of squares)
            SS_tot = Σ(y_true - mean(y_true))²  (total sum of squares)

        Args:
            X (np.ndarray): Test features
            y (np.ndarray): True target values

        Returns:
            float: R² score
                - 1.0: Perfect predictions
                - 0.0: Predictions as good as predicting mean
                - < 0: Predictions worse than predicting mean

        Example:
            >>> r2 = reg.score(X_test, y_test)
            >>> print(f"R² score: {r2:.3f}")
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # Handle edge case: all y values are the same
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)
