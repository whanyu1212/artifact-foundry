import numpy as np
from collections import Counter
from typing import List, Union


class TreeMetrics:
    """
    Utility class for calculating metrics for CART models.
    Supports both classification (Gini, Entropy) and regression (MSE, MAE, Variance).
    """

    # ==================== CLASSIFICATION METRICS ====================

    @staticmethod
    def entropy(y: List[int]) -> float:
        """
        Calculates Shannon Entropy: H(S) = -sum(p * log2(p))
        Complexity: O(N) due to Counter iteration.

        Args:
            y (List[int]): List of class labels.

        Returns:
            float: Calculated entropy value.
        """
        # creates a dictionary of label counts
        # e.g., {0: 5, 1: 3}
        hist = Counter(y)
        total = len(y)
        return -sum((count / total) * np.log2(count / total) for count in hist.values())

    @staticmethod
    def gini(y: List[int]) -> float:
        """
        Calculates Gini Impurity: Gini = 1 - sum(p^2)
        Complexity: O(N), generally faster than entropy (no log).
        Args:
            y (List[int]): List of class labels.
        Returns:
            float: Calculated Gini impurity value.
        """
        hist = Counter(y)
        total = len(y)
        return 1 - sum((count / total) ** 2 for count in hist.values())

    @classmethod
    def information_gain(
        cls, parent: List[int], left: List[int], right: List[int], mode: str = "gini"
    ) -> float:
        """
        Calculates IG = Parent_Impurity - Weighted_Child_Impurity

        Args:
            parent (List[int]): Class labels of the parent node.
            left (List[int]): Class labels of the left child node.
            right (List[int]): Class labels of the right child node.
            mode (str, optional): Metric to use ("gini" or "entropy"). Defaults
        """
        # Select metric
        # Delegates to the appropriate static method
        metric_func = cls.entropy if mode == "entropy" else cls.gini

        # Weights
        w_left = len(left) / len(parent)
        w_right = len(right) / len(parent)

        # Formula
        gain = metric_func(parent) - (
            (w_left * metric_func(left)) + (w_right * metric_func(right))
        )
        return gain

    # ==================== REGRESSION METRICS ====================

    @staticmethod
    def variance(y: List[float]) -> float:
        """
        Calculates variance: Var(y) = (1/n) * sum((y_i - mean)^2)
        Used as impurity measure for regression trees.
        Complexity: O(N)

        Args:
            y (List[float]): List of continuous target values.

        Returns:
            float: Calculated variance value.
        """
        y_array = np.array(y)
        return np.var(y_array)

    @staticmethod
    def mse(y: List[float]) -> float:
        """
        Calculates Mean Squared Error: MSE = (1/n) * sum((y_i - mean)^2)
        Equivalent to variance when predicting the mean.
        Complexity: O(N)

        Args:
            y (List[float]): List of continuous target values.

        Returns:
            float: Calculated MSE value.
        """
        y_array = np.array(y)
        mean_y = np.mean(y_array)
        return np.mean((y_array - mean_y) ** 2)

    @staticmethod
    def mae(y: List[float]) -> float:
        """
        Calculates Mean Absolute Error: MAE = (1/n) * sum(|y_i - median|)
        More robust to outliers than MSE.
        Optimal prediction is the median, not mean.
        Complexity: O(N log N) due to median calculation.

        Args:
            y (List[float]): List of continuous target values.

        Returns:
            float: Calculated MAE value.
        """
        y_array = np.array(y)
        median_y = np.median(y_array)
        return np.mean(np.abs(y_array - median_y))

    @classmethod
    def variance_reduction(
        cls, parent: List[float], left: List[float], right: List[float]
    ) -> float:
        """
        Calculates variance reduction (analogous to information gain for regression).
        VarReduction = Var(parent) - weighted_avg(Var(left), Var(right))

        Args:
            parent (List[float]): Target values of the parent node.
            left (List[float]): Target values of the left child node.
            right (List[float]): Target values of the right child node.

        Returns:
            float: Variance reduction from the split.
        """
        # Weights
        w_left = len(left) / len(parent)
        w_right = len(right) / len(parent)

        # Formula
        reduction = cls.variance(parent) - (
            w_left * cls.variance(left) + w_right * cls.variance(right)
        )
        return reduction

    @classmethod
    def mse_reduction(
        cls, parent: List[float], left: List[float], right: List[float]
    ) -> float:
        """
        Calculates MSE reduction (equivalent to variance_reduction).
        MSE_Reduction = MSE(parent) - weighted_avg(MSE(left), MSE(right))

        Args:
            parent (List[float]): Target values of the parent node.
            left (List[float]): Target values of the left child node.
            right (List[float]): Target values of the right child node.

        Returns:
            float: MSE reduction from the split.
        """
        # Weights
        w_left = len(left) / len(parent)
        w_right = len(right) / len(parent)

        # Formula
        reduction = cls.mse(parent) - (
            w_left * cls.mse(left) + w_right * cls.mse(right)
        )
        return reduction

    @classmethod
    def mae_reduction(
        cls, parent: List[float], left: List[float], right: List[float]
    ) -> float:
        """
        Calculates MAE reduction (robust to outliers).
        MAE_Reduction = MAE(parent) - weighted_avg(MAE(left), MAE(right))

        Args:
            parent (List[float]): Target values of the parent node.
            left (List[float]): Target values of the left child node.
            right (List[float]): Target values of the right child node.

        Returns:
            float: MAE reduction from the split.
        """
        # Weights
        w_left = len(left) / len(parent)
        w_right = len(right) / len(parent)

        # Formula
        reduction = cls.mae(parent) - (
            w_left * cls.mae(left) + w_right * cls.mae(right)
        )
        return reduction


if __name__ == "__main__":
    print("=" * 60)
    print("CLASSIFICATION METRICS EXAMPLE")
    print("=" * 60)

    # Classification example: before split
    y_parent = [0, 0, 1, 1, 1]

    # After split
    y_left = [0, 0]
    y_right = [1, 1, 1]

    print("\nParent node:", y_parent)
    print("Left child:", y_left)
    print("Right child:", y_right)
    print()

    print("Entropy of parent:", TreeMetrics.entropy(y_parent))
    print("Gini of parent:", TreeMetrics.gini(y_parent))
    print(
        "Information Gain (Gini):",
        TreeMetrics.information_gain(y_parent, y_left, y_right, mode="gini"),
    )
    print(
        "Information Gain (Entropy):",
        TreeMetrics.information_gain(y_parent, y_left, y_right, mode="entropy"),
    )

    print("\n" + "=" * 60)
    print("REGRESSION METRICS EXAMPLE")
    print("=" * 60)

    # Regression example: continuous target values
    # Parent has mixed values (high variance)
    y_reg_parent = [1.0, 2.0, 3.0, 9.0, 10.0]

    # Good split: separates low and high values
    y_reg_left = [1.0, 2.0, 3.0]
    y_reg_right = [9.0, 10.0]

    print("\nParent node:", y_reg_parent)
    print("Left child:", y_reg_left)
    print("Right child:", y_reg_right)
    print()

    print("Variance of parent:", TreeMetrics.variance(y_reg_parent))
    print("MSE of parent:", TreeMetrics.mse(y_reg_parent))
    print("MAE of parent:", TreeMetrics.mae(y_reg_parent))
    print()

    print(
        "Variance Reduction:",
        TreeMetrics.variance_reduction(y_reg_parent, y_reg_left, y_reg_right),
    )
    print(
        "MSE Reduction:",
        TreeMetrics.mse_reduction(y_reg_parent, y_reg_left, y_reg_right),
    )
    print(
        "MAE Reduction:",
        TreeMetrics.mae_reduction(y_reg_parent, y_reg_left, y_reg_right),
    )

    # Example with outliers to show MAE robustness
    print("\n" + "=" * 60)
    print("OUTLIER ROBUSTNESS COMPARISON")
    print("=" * 60)

    y_with_outlier = [1.0, 2.0, 3.0, 4.0, 100.0]  # 100.0 is outlier
    y_without_outlier = [1.0, 2.0, 3.0, 4.0, 5.0]

    print("\nWithout outlier:", y_without_outlier)
    print("With outlier:", y_with_outlier)
    print()

    print("MSE (without outlier):", TreeMetrics.mse(y_without_outlier))
    print("MSE (with outlier):", TreeMetrics.mse(y_with_outlier))
    print(
        "MSE increase:",
        f"{(TreeMetrics.mse(y_with_outlier) / TreeMetrics.mse(y_without_outlier) - 1) * 100:.1f}%",
    )
    print()

    print("MAE (without outlier):", TreeMetrics.mae(y_without_outlier))
    print("MAE (with outlier):", TreeMetrics.mae(y_with_outlier))
    print(
        "MAE increase:",
        f"{(TreeMetrics.mae(y_with_outlier) / TreeMetrics.mae(y_without_outlier) - 1) * 100:.1f}%",
    )
    print("\nNote: MAE is much more robust to outliers than MSE!")
