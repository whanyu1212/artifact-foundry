"""
Random Forest - Ensemble Method with Bootstrap and Feature Randomness
=====================================================================
Extends bagging by adding feature randomness at each split,
further decorrelating trees and improving generalization.

Theory:
    Random Forest = Bagging + Feature Randomness

    Key Innovations over Bagging:
        1. Bootstrap sampling (like bagging)
        2. Random feature subset at EACH split
        3. Fully grown trees (no pruning)

    At each node split:
        - Select random subset of m features (m = sqrt(n_features) by default)
        - Find best split using only these m features
        - Different random subset at each node

Benefits:
    - Further reduces correlation between trees
    - Reduces variance without increasing bias
    - More robust to noisy features
    - Feature importance can be computed from OOB error

Mathematics:
    Feature Subset Size (for classification):
        m = sqrt(n_features)  # default
        m = log2(n_features)  # alternative
        m = n_features        # equivalent to bagging

    Variance Reduction:
        As correlation ρ between trees decreases:
        Var(ensemble) = ρσ² + (1-ρ)σ²/n
        Random features reduce ρ, thus reducing variance

    Prediction (same as bagging):
        Classification: ŷ = mode([h₁(x), ..., h_m(x)])
        Regression: ŷ = (1/m) × Σ h_i(x)

Reference:
    Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
"""

import numpy as np
from typing import List, Optional, Union
import copy


class RandomForestClassifier:
    """
    Random Forest for classification tasks.

    Trains ensemble of decision trees with bootstrap sampling and
    random feature selection at each split.

    Args:
        n_estimators (int): Number of trees in forest (default: 100)
        max_depth (int): Maximum tree depth, None = unlimited (default: None)
        min_samples_split (int): Min samples to split node (default: 2)
        min_samples_leaf (int): Min samples in leaf (default: 1)
        max_features (Union[int, float, str]): Number of features per split:
            - int: Use exactly this many features
            - float: Use this fraction of features
            - 'sqrt': Use sqrt(n_features) (default, best for classification)
            - 'log2': Use log2(n_features)
            - None: Use all features (equivalent to bagging)
        criterion (str): Split criterion, 'gini' or 'entropy' (default: 'gini')
        random_state (int): Random seed for reproducibility (default: None)
        oob_score (bool): Whether to compute out-of-bag score (default: False)

    Attributes (set during fit):
        estimators_ (List): Fitted decision tree classifiers
        n_features_ (int): Number of features
        n_classes_ (int): Number of unique classes
        classes_ (np.ndarray): Unique class labels
        oob_score_ (float): Out-of-bag score (if oob_score=True)
        oob_decision_function_ (np.ndarray): OOB predictions

    Example:
        >>> rf = RandomForestClassifier(
        ...     n_estimators=100,
        ...     max_depth=10,
        ...     max_features='sqrt',
        ...     random_state=42
        ... )
        >>> rf.fit(X_train, y_train)
        >>> predictions = rf.predict(X_test)
        >>> print(f"Accuracy: {rf.score(X_test, y_test):.3f}")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[int, float, str, None] = 'sqrt',
        criterion: str = 'gini',
        random_state: Optional[int] = None,
        oob_score: bool = False,
    ):
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")

        if criterion not in ['gini', 'entropy']:
            raise ValueError(f"criterion must be 'gini' or 'entropy', got '{criterion}'")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.random_state = random_state
        self.oob_score = oob_score

        # Set during fit
        self.estimators_: List = []
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.oob_score_ = None
        self.oob_decision_function_ = None

    def _calculate_max_features(self, n_features: int) -> int:
        """
        Calculate number of features to consider at each split.

        Defaults:
            - Classification: sqrt(n_features)
            - Regression: n_features / 3

        Args:
            n_features (int): Total number of features

        Returns:
            int: Number of features to consider at each split

        Example:
            >>> # For 16 features with max_features='sqrt'
            >>> # Returns 4 features
            >>> # For 16 features with max_features=0.5
            >>> # Returns 8 features
        """
        if self.max_features is None:
            return n_features
        elif self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            if self.max_features < 1 or self.max_features > n_features:
                raise ValueError(
                    f"max_features must be in [1, {n_features}], got {self.max_features}"
                )
            return self.max_features
        elif isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1.0:
                raise ValueError(
                    f"max_features as float must be in (0, 1], got {self.max_features}"
                )
            return max(1, int(self.max_features * n_features))
        else:
            raise ValueError(
                f"Invalid max_features: {self.max_features}. "
                "Use int, float, 'sqrt', 'log2', or None"
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """
        Build random forest from training data.

        Algorithm:
            1. Calculate max_features from total features
            2. For each tree i = 1 to n_estimators:
                a. Create bootstrap sample B_i
                b. Create RandomDecisionTree with feature sampling
                c. Fit tree on B_i
                d. Store tree
            3. If oob_score=True, compute out-of-bag predictions

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Training labels, shape (n_samples,)

        Returns:
            self: Fitted forest

        Time Complexity:
            O(n_estimators × n_samples × log(n_samples) × max_features)

        Example:
            >>> rf.fit(X_train, y_train)
            >>> print(f"Trained {len(rf.estimators_)} trees")
        """
        # Import here to avoid circular dependency
        try:
            from ..decision_trees import DecisionTreeClassifier
        except ImportError:
            from decision_trees import DecisionTreeClassifier

        # Store metadata
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Calculate number of features per split
        max_features_per_split = self._calculate_max_features(self.n_features_)

        n_samples = X.shape[0]

        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)

        # Track OOB predictions if requested
        if self.oob_score:
            oob_predictions = np.zeros((n_samples, self.n_classes_))
            oob_counts = np.zeros(n_samples)

        # Train estimators
        self.estimators_ = []

        for i in range(self.n_estimators):
            # Create bootstrap sample (sample WITH replacement)
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Create decision tree with feature randomness
            # We create a custom tree wrapper that samples features at each split
            tree = _RandomFeatureDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=max_features_per_split,
                random_state=rng.randint(0, 2**31 - 1),  # Different seed per tree
            )

            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(tree)

            # Track OOB samples
            if self.oob_score:
                # Find out-of-bag samples
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[indices] = False
                oob_indices = np.where(oob_mask)[0]

                if len(oob_indices) > 0:
                    # Predict on OOB samples
                    oob_preds = tree.predict(X[oob_indices])

                    # Accumulate predictions
                    for idx, pred in zip(oob_indices, oob_preds):
                        oob_predictions[idx, pred] += 1
                        oob_counts[idx] += 1

        # Compute OOB score if requested
        if self.oob_score:
            oob_mask = oob_counts > 0

            if not np.any(oob_mask):
                raise ValueError(
                    "No out-of-bag samples found. "
                    "Try increasing n_estimators."
                )

            # Get OOB predictions via majority vote
            oob_decision = np.argmax(oob_predictions[oob_mask], axis=1)

            # Calculate OOB accuracy
            self.oob_score_ = np.mean(oob_decision == y[oob_mask])
            self.oob_decision_function_ = oob_predictions

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using majority voting.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,)

        Example:
            >>> predictions = rf.predict(X_test)
        """
        if not self.estimators_:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Get vote probabilities and convert to predictions
        proba = self.predict_proba(X)

        # Return class with highest vote count
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities as fraction of votes.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, n_classes)

        Example:
            >>> proba = rf.predict_proba(X_test)
            >>> # proba[i, j] = fraction of trees predicting class j for sample i
        """
        if not self.estimators_:
            raise ValueError("Model not fitted yet. Call fit() first.")

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))

        # Get predictions from all estimators
        all_predictions = np.array([est.predict(X) for est in self.estimators_])

        # Count votes for each class
        for i in range(n_samples):
            sample_preds = all_predictions[:, i]
            for class_idx in range(self.n_classes_):
                proba[i, class_idx] = np.sum(sample_preds == self.classes_[class_idx])

        # Normalize to probabilities
        proba /= self.n_estimators

        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate classification accuracy.

        Args:
            X (np.ndarray): Test features
            y (np.ndarray): True labels

        Returns:
            float: Accuracy score in [0, 1]

        Example:
            >>> accuracy = rf.score(X_test, y_test)
            >>> print(f"Accuracy: {accuracy:.1%}")
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class _RandomFeatureDecisionTree:
    """
    Decision tree that samples random features at each split.

    This is an internal wrapper around DecisionTreeClassifier that
    implements feature randomness by modifying the split-finding logic.

    The key modification: at each node, select a random subset of features
    before finding the best split.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        max_features: int = None,
        random_state: Optional[int] = None,
    ):
        try:
            from ..decision_trees import DecisionTreeClassifier
        except ImportError:
            from decision_trees import DecisionTreeClassifier

        # Create base tree - we'll override its split method
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
        )

        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RandomFeatureDecisionTree":
        """
        Fit tree with random feature selection at each split.

        We override the base tree's _find_best_split method temporarily
        to implement feature randomness.
        """
        # Store original method
        original_find_best_split = self.tree._find_best_split

        # Create wrapper that samples features
        def _find_best_split_with_random_features(X_node, y_node):
            n_features = X_node.shape[1]

            # Select random subset of features
            if self.max_features and self.max_features < n_features:
                feature_indices = self.rng.choice(
                    n_features,
                    size=self.max_features,
                    replace=False
                )
                feature_indices = np.sort(feature_indices)
            else:
                feature_indices = np.arange(n_features)

            # Find best split using only selected features
            best_gain = -1
            best_feature = None
            best_threshold = None

            for feature_idx in feature_indices:
                # Get unique values for this feature
                thresholds = np.unique(X_node[:, feature_idx])

                for threshold in thresholds:
                    # Split data
                    left_mask = X_node[:, feature_idx] <= threshold
                    right_mask = ~left_mask

                    y_left = y_node[left_mask]
                    y_right = y_node[right_mask]

                    # Check min_samples_leaf
                    if (len(y_left) < self.tree.min_samples_leaf or
                        len(y_right) < self.tree.min_samples_leaf):
                        continue

                    # Calculate split gain
                    gain = self.tree._calculate_split_gain(y_node, y_left, y_right)

                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold

            return best_feature, best_threshold, best_gain

        # Temporarily replace method
        self.tree._find_best_split = _find_best_split_with_random_features

        # Fit tree
        self.tree.fit(X, y)

        # Restore original method
        self.tree._find_best_split = original_find_best_split

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the fitted tree."""
        return self.tree.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the fitted tree."""
        return self.tree.predict_proba(X)
