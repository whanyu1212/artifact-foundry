"""
Gradient Boosting - Sequential Ensemble with Gradient Descent in Function Space
================================================================================
Builds ensemble by iteratively fitting new models to the negative gradient
of a loss function, combining them with a learning rate for regularization.

Theory:
    Gradient Boosting performs gradient descent in function space:

    F₀(x) = initial prediction (e.g., mean, log-odds)

    For m = 1 to M (n_estimators):
        1. Compute pseudo-residuals (negative gradient of loss):
           r_i = -∂L(y_i, F(x_i))/∂F(x_i)

        2. Fit weak learner h_m to residuals:
           h_m = argmin Σ(r_i - h(x_i))²

        3. Update ensemble:
           F_m(x) = F_{m-1}(x) + ν × h_m(x)
           where ν is the learning rate

    Final prediction: F_M(x)

Key Differences from Bagging/RF:
    - Sequential (not parallel) - each tree corrects previous mistakes
    - Fits residuals (gradients), not original targets
    - Learning rate controls contribution of each tree
    - Typically uses shallow trees (stumps or depth 3-5)
    - More prone to overfitting without regularization

Loss Functions:
    Binary Classification (log loss):
        L(y, F) = log(1 + exp(-2yF))  where y ∈ {-1, +1}
        Gradient: -y / (1 + exp(yF))

    Multiclass (multinomial deviance):
        Uses one-vs-all or softmax formulation

    Regression (MSE):
        L(y, F) = (y - F)²
        Gradient: -(y - F)  (simple residual)

Regularization:
    - Learning rate ν: smaller = slower learning, better generalization
    - Max depth: shallow trees = less overfitting
    - Subsampling: train each tree on random subset
    - Early stopping: stop when validation error stops improving

Reference:
    Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
    Annals of Statistics, 29(5), 1189-1232.
"""

import numpy as np
from typing import List, Optional
import copy


class GradientBoostingClassifier:
    """
    Gradient Boosting for classification using log loss (deviance).

    Builds ensemble sequentially, with each tree fitting the negative
    gradient of the loss function.

    Args:
        n_estimators (int): Number of boosting stages (trees) (default: 100)
        learning_rate (float): Shrinks contribution of each tree (default: 0.1)
            Smaller values require more estimators but may generalize better
        max_depth (int): Maximum depth of individual trees (default: 3)
            Shallow trees (3-5) work best for boosting
        min_samples_split (int): Min samples to split node (default: 2)
        min_samples_leaf (int): Min samples in leaf (default: 1)
        random_state (int): Random seed (default: None)

    Attributes (set during fit):
        estimators_ (List): Fitted regression trees (one per stage)
        learning_rate (float): Learning rate used
        n_features_ (int): Number of features
        n_classes_ (int): Number of unique classes
        classes_ (np.ndarray): Unique class labels
        init_prediction_ (float): Initial prediction (for binary: log-odds)

    Example:
        >>> gb = GradientBoostingClassifier(
        ...     n_estimators=100,
        ...     learning_rate=0.1,
        ...     max_depth=3,
        ...     random_state=42
        ... )
        >>> gb.fit(X_train, y_train)
        >>> predictions = gb.predict(X_test)
        >>> print(f"Accuracy: {gb.score(X_test, y_test):.3f}")

    Note:
        This implementation focuses on educational clarity over performance.
        For production, use sklearn.ensemble.GradientBoostingClassifier or XGBoost.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
    ):
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")

        if not 0 < learning_rate <= 1:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")

        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        # Set during fit
        self.estimators_: List = []
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        self.init_prediction_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingClassifier":
        """
        Build gradient boosting classifier.

        Algorithm (Binary Classification with Log Loss):
            1. Initialize F₀ = log(p/(1-p)) where p = mean(y)
            2. For m = 1 to n_estimators:
                a. Compute pseudo-residuals: r_i = y_i - p_i
                   where p_i = sigmoid(F_{m-1}(x_i))
                b. Fit regression tree h_m to residuals
                c. Update: F_m = F_{m-1} + learning_rate × h_m
            3. Final prediction: sigmoid(F_M(x))

        Args:
            X (np.ndarray): Training features, shape (n_samples, n_features)
            y (np.ndarray): Training labels, shape (n_samples,)
                Should contain class labels (will be converted to 0/1 for binary)

        Returns:
            self: Fitted classifier

        Time Complexity:
            O(n_estimators × n_samples × n_features × max_depth)

        Example:
            >>> gb.fit(X_train, y_train)
            >>> print(f"Trained {len(gb.estimators_)} trees")
        """
        try:
            from ..decision_trees import DecisionTreeRegressor
        except ImportError:
            from decision_trees import DecisionTreeRegressor

        # Store metadata
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Currently only support binary classification
        if self.n_classes_ != 2:
            raise NotImplementedError(
                f"Only binary classification supported, got {self.n_classes_} classes. "
                "Multiclass requires one-vs-all or softmax approach."
            )

        # Convert labels to 0/1 for binary classification
        # This ensures we work with 0/1 regardless of original labels
        y_binary = (y == self.classes_[1]).astype(float)

        n_samples = X.shape[0]

        # Initialize F₀ with log-odds: log(p/(1-p))
        # This is the optimal constant prediction for log loss
        p_init = np.mean(y_binary)
        # Clip to avoid log(0) or log(inf)
        p_init = np.clip(p_init, 1e-15, 1 - 1e-15)
        self.init_prediction_ = np.log(p_init / (1 - p_init))

        # Current predictions in log-odds space (raw scores, not probabilities)
        F = np.full(n_samples, self.init_prediction_)

        # Train estimators sequentially
        self.estimators_ = []

        for m in range(self.n_estimators):
            # Convert log-odds to probabilities using sigmoid
            # p = 1 / (1 + exp(-F))
            probabilities = self._sigmoid(F)

            # Compute pseudo-residuals (negative gradient of log loss)
            # For log loss: -∂L/∂F = y - p
            residuals = y_binary - probabilities

            # Fit regression tree to residuals
            # Tree predicts the residual (gradient direction)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion='mse',
            )
            tree.fit(X, residuals)
            self.estimators_.append(tree)

            # Update predictions: F_m = F_{m-1} + ν × h_m
            # Learning rate ν shrinks each tree's contribution
            update = self.learning_rate * tree.predict(X)
            F += update

        return self

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function: σ(x) = 1 / (1 + exp(-x))

        Numerically stable implementation using:
            σ(x) = exp(x) / (1 + exp(x))     if x >= 0
            σ(x) = 1 / (1 + exp(-x))         if x < 0

        Args:
            x (np.ndarray): Input values (raw scores)

        Returns:
            np.ndarray: Probabilities in [0, 1]

        Example:
            >>> _sigmoid(0.0)  # Returns 0.5
            >>> _sigmoid(5.0)  # Returns ~0.993
            >>> _sigmoid(-5.0) # Returns ~0.007
        """
        # Clip to prevent overflow in exp
        x = np.clip(x, -500, 500)

        # Use numerically stable computation
        positive = x >= 0
        result = np.zeros_like(x, dtype=float)

        # For positive x: exp(-x) is small, no overflow
        result[positive] = 1 / (1 + np.exp(-x[positive]))

        # For negative x: exp(x) is small, no overflow
        exp_x = np.exp(x[~positive])
        result[~positive] = exp_x / (1 + exp_x)

        return result

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Predict raw scores (log-odds) before applying sigmoid.

        F(x) = F₀ + ν × Σ h_m(x)

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            np.ndarray: Raw scores (log-odds)

        Note:
            This is used internally. For probabilities, use predict_proba().
        """
        if not self.estimators_:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Start with initial prediction
        F = np.full(X.shape[0], self.init_prediction_)

        # Add contribution from each tree
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)

        return F

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        For binary classification:
            P(class=1 | x) = sigmoid(F(x))
            P(class=0 | x) = 1 - sigmoid(F(x))

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Class probabilities, shape (n_samples, 2)
                Column 0: P(class=classes_[0])
                Column 1: P(class=classes_[1])

        Example:
            >>> proba = gb.predict_proba(X_test)
            >>> # proba[i, 1] = probability sample i belongs to positive class
        """
        # Get raw scores (log-odds)
        F = self._predict_raw(X)

        # Convert to probabilities
        p1 = self._sigmoid(F)
        p0 = 1 - p1

        # Return as (n_samples, 2) array
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X (np.ndarray): Feature matrix, shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted class labels, shape (n_samples,)

        Example:
            >>> predictions = gb.predict(X_test)
        """
        proba = self.predict_proba(X)

        # Predict class with highest probability
        class_indices = np.argmax(proba, axis=1)

        return self.classes_[class_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate classification accuracy.

        Args:
            X (np.ndarray): Test features
            y (np.ndarray): True labels

        Returns:
            float: Accuracy score in [0, 1]

        Example:
            >>> accuracy = gb.score(X_test, y_test)
            >>> print(f"Accuracy: {accuracy:.1%}")
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def staged_predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Predict class probabilities at each boosting stage.

        Useful for:
            - Visualizing how predictions evolve
            - Early stopping (find optimal n_estimators)
            - Understanding learning dynamics

        Args:
            X (np.ndarray): Feature matrix

        Returns:
            List[np.ndarray]: List of probability arrays, one per stage

        Example:
            >>> staged_probas = gb.staged_predict_proba(X_val)
            >>> # staged_probas[i] = predictions using first i+1 trees
        """
        if not self.estimators_:
            raise ValueError("Model not fitted yet. Call fit() first.")

        n_samples = X.shape[0]
        F = np.full(n_samples, self.init_prediction_)

        staged_probas = []

        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
            p1 = self._sigmoid(F)
            p0 = 1 - p1
            staged_probas.append(np.column_stack([p0, p1]))

        return staged_probas

    def staged_score(self, X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Calculate accuracy at each boosting stage.

        Useful for early stopping and analyzing overfitting.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True labels

        Returns:
            List[float]: Accuracy scores, one per stage

        Example:
            >>> train_scores = gb.staged_score(X_train, y_train)
            >>> val_scores = gb.staged_score(X_val, y_val)
            >>> # Plot to see if model is overfitting
        """
        staged_probas = self.staged_predict_proba(X)
        scores = []

        for proba in staged_probas:
            class_indices = np.argmax(proba, axis=1)
            predictions = self.classes_[class_indices]
            scores.append(np.mean(predictions == y))

        return scores
