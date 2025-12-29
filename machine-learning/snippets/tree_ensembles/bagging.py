"""
Bagging (Bootstrap Aggregating) - Ensemble Learning Method
===========================================================
Reduces variance by training multiple models on bootstrap samples
and aggregating their predictions through voting or averaging.

Theory:
    Bagging = Bootstrap + Aggregating

    1. Bootstrap: Randomly sample n_samples WITH replacement
    2. Train: Fit one model on each bootstrap sample
    3. Aggregate: Combine predictions via majority vote (classification)
                  or averaging (regression)

Key Benefits:
    - Reduces variance without increasing bias
    - Effective for high-variance models (e.g., deep decision trees)
    - Can be parallelized (each model trains independently)
    - Out-of-bag (OOB) samples provide built-in validation

Mathematics:
    Given dataset D with n samples:

    Bootstrap Sample B_i:
        - Draw n samples from D with replacement
        - ~63.2% unique samples (1 - (1-1/n)^n → 1 - 1/e)
        - ~36.8% out-of-bag (OOB) samples

    Classification Prediction:
        ŷ = mode([h₁(x), h₂(x), ..., h_m(x)])
        where h_i is the i-th model

    Regression Prediction:
        ŷ = (1/m) × Σ h_i(x)

Reference:
    Breiman, L. (1996). Bagging Predictors. Machine Learning, 24(2), 123-140.
"""

import numpy as np
from typing import List, Optional, Any


class BaggingClassifier:
    """
    Bootstrap Aggregating for classification tasks.

    Trains multiple base estimators on random subsets (with replacement)
    of the training data and aggregates predictions via majority voting.

    Args:
        base_estimator: Base classifier to train (must have fit/predict)
            Default: None (user must provide estimator)
        n_estimators (int): Number of base estimators to train (default: 10)
        max_samples (float): Fraction of samples to draw for each bootstrap
            Range: (0, 1], default: 1.0 (sample n_samples with replacement)
        random_state (int): Random seed for reproducibility (default: None)
        oob_score (bool): Whether to compute out-of-bag score (default: False)

    Attributes (set during fit):
        estimators_ (List): Fitted base estimators
        n_features_ (int): Number of features
        n_classes_ (int): Number of unique classes
        classes_ (np.ndarray): Unique class labels
        oob_score_ (float): Out-of-bag score (if oob_score=True)
        oob_decision_function_ (np.ndarray): OOB predictions for each sample

    Example:
        >>> from decision_trees import DecisionTreeClassifier
        >>> base_tree = DecisionTreeClassifier(max_depth=5)
        >>> bag = BaggingClassifier(base_estimator=base_tree, n_estimators=10)
        >>> bag.fit(X_train, y_train)
        >>> predictions = bag.predict(X_test)
        >>> print(f"Accuracy: {np.mean(predictions == y_test):.3f}")
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        random_state: Optional[int] = None,
        oob_score: bool = False,
    ):
        if base_estimator is None:
            raise ValueError("base_estimator cannot be None. Provide a classifier.")

        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")

        if not 0 < max_samples <= 1.0:
            raise ValueError(f"max_samples must be in (0, 1], got {max_samples}")

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score

        # Set during fit
        self.estimators_: List[Any] = []
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.oob_score_ = None
        self.oob_decision_function_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggingClassifier":
        """
        Build bagging ensemble from training data.

        Algorithm:
            1. For i = 1 to n_estimators:
                a. Create bootstrap sample B_i (sample with replacement)
                b. Train estimator h_i on B_i
                c. Store h_i
            2. If oob_score=True, compute out-of-bag predictions

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Training labels, shape (n_samples,)

        Returns:
            self: Fitted ensemble

        Time Complexity:
            O(n_estimators × T(n))
            where T(n) is the time to train one base estimator

        Example:
            >>> bag.fit(X_train, y_train)
            >>> print(f"Trained {len(bag.estimators_)} estimators")
        """
        # Store metadata
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        n_samples = X.shape[0]
        n_bootstrap = int(n_samples * self.max_samples)

        # Initialize random number generator
        rng = np.random.RandomState(self.random_state)

        # Track which samples are OOB for each estimator
        if self.oob_score:
            oob_predictions = np.zeros((n_samples, self.n_classes_))
            oob_counts = np.zeros(n_samples)

        # Train estimators
        self.estimators_ = []

        for i in range(self.n_estimators):
            # Create bootstrap sample (sample WITH replacement)
            # This is the key difference from random subsampling
            indices = rng.choice(n_samples, size=n_bootstrap, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Clone and train estimator
            # We need to create a new instance to avoid overwriting
            import copy
            estimator = copy.deepcopy(self.base_estimator)
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(estimator)

            # Track OOB samples (samples NOT in bootstrap sample)
            if self.oob_score:
                # Find out-of-bag samples (not selected in bootstrap)
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[indices] = False
                oob_indices = np.where(oob_mask)[0]

                if len(oob_indices) > 0:
                    # Predict on OOB samples
                    oob_preds = estimator.predict(X[oob_indices])

                    # Accumulate predictions (for voting)
                    for idx, pred in zip(oob_indices, oob_preds):
                        oob_predictions[idx, pred] += 1
                        oob_counts[idx] += 1

        # Compute OOB score if requested
        if self.oob_score:
            # Only use samples that were OOB at least once
            oob_mask = oob_counts > 0

            if not np.any(oob_mask):
                raise ValueError(
                    "No out-of-bag samples found. "
                    "Try increasing n_estimators or max_samples."
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

        For each sample:
            1. Get predictions from all n_estimators
            2. Count votes for each class
            3. Return class with most votes (argmax)

        Ties are broken by choosing the smallest class index (argmax behavior).
        This ensures consistency with predict_proba() and sklearn.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,)

        Example:
            >>> predictions = bag.predict(X_test)
            >>> # If 10 estimators: 6 predict class 1, 4 predict class 0
            >>> # Final prediction: class 1 (majority vote)
        """
        if not self.estimators_:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Get vote probabilities and convert to predictions
        proba = self.predict_proba(X)

        # Return class with highest vote count (argmax handles ties consistently)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities as fraction of votes.

        For each sample:
            P(class_i | x) = (# estimators predicting class_i) / n_estimators

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, n_classes)

        Example:
            >>> proba = bag.predict_proba(X_test)
            >>> # proba[0] = [0.3, 0.7] means:
            >>> # 30% of estimators predicted class 0
            >>> # 70% of estimators predicted class 1
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
                # Count how many estimators predicted this class
                proba[i, class_idx] = np.sum(sample_preds == self.classes_[class_idx])

        # Normalize to probabilities
        proba /= self.n_estimators

        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate classification accuracy.

        Accuracy = (# correct predictions) / (# total predictions)

        Args:
            X (np.ndarray): Test features, shape (n_samples, n_features)
            y (np.ndarray): True labels, shape (n_samples,)

        Returns:
            float: Accuracy score in [0, 1]

        Example:
            >>> accuracy = bag.score(X_test, y_test)
            >>> print(f"Accuracy: {accuracy:.1%}")
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
