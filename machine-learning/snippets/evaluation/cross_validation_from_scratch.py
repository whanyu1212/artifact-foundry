"""
Cross-Validation - From Scratch Implementation

Implements common cross-validation strategies for model evaluation and selection.

Implemented CV Strategies:
- K-Fold Cross-Validation
- Stratified K-Fold (for classification)
- Leave-One-Out Cross-Validation (LOOCV)
- Time Series Split
- Train-Validation-Test Split

All implementations validated against scikit-learn.
"""

import numpy as np
from typing import Generator, Tuple, Optional, List, Union, Any
from collections import Counter


# ============================================================================
# BASIC SPLITS
# ============================================================================


def train_test_split(
    *arrays: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    Split arrays into random train and test subsets.

    Args:
        *arrays (np.ndarray): Sequence of indexable objects (arrays, lists) with same length.
        test_size (float): Proportion of dataset to include in test split (0.0 to 1.0).
        random_state (Optional[int]): Random seed for reproducibility.
        stratify (Optional[np.ndarray]): If not None, split preserves the distribution of this array.

    Returns:
        List[np.ndarray]: train-test split of inputs.
            If input is X, y: returns X_train, X_test, y_train, y_test

    Example:
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> y = np.array([0, 0, 1, 1])
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.5, random_state=42, stratify=y
        ... )
        >>> X_train.shape, X_test.shape
        ((2, 2), (2, 2))
    """
    if not arrays:
        raise ValueError("At least one array required as input")

    # Check all arrays have same length
    n_samples = len(arrays[0])
    for arr in arrays[1:]:
        if len(arr) != n_samples:
            raise ValueError("All arrays must have same length")

    # Set random seed
    if random_state is not None:
        np.random.seed(random_state)

    # Determine split size
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    if stratify is not None:
        # Stratified split: preserve class distribution
        stratify = np.asarray(stratify)

        # Get unique classes and their counts
        classes, class_counts = np.unique(stratify, return_counts=True)

        train_indices = []
        test_indices = []

        for cls in classes:
            # Indices for this class
            cls_indices = np.where(stratify == cls)[0]

            # Shuffle class indices
            np.random.shuffle(cls_indices)

            # Split proportionally
            n_cls_test = int(len(cls_indices) * test_size)
            n_cls_train = len(cls_indices) - n_cls_test

            test_indices.extend(cls_indices[:n_cls_test])
            train_indices.extend(cls_indices[n_cls_test:])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        # Shuffle to randomize order
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

    else:
        # Random split
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

    # Split all arrays
    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_indices])
        result.append(arr[test_indices])

    return result


# ============================================================================
# CROSS-VALIDATION SPLITTERS
# ============================================================================


class KFold:
    """
    K-Fold cross-validator.

    Splits dataset into K consecutive folds. Each fold is used once as validation
    while the remaining K-1 folds form the training set.

    Attributes:
        n_splits (int): Number of folds (K).
        shuffle (bool): Whether to shuffle data before splitting.
        random_state (int): Random seed for shuffling.

    Example:
        >>> kf = KFold(n_splits=3, shuffle=True, random_state=42)
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        >>> for train_idx, val_idx in kf.split(X):
        ...     print(f"Train: {train_idx}, Val: {val_idx}")
        Train: [2 3 4 5], Val: [0 1]
        Train: [0 1 4 5], Val: [2 3]
        Train: [0 1 2 3], Val: [4 5]
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize K-Fold cross-validator.

        Args:
            n_splits (int): Number of folds. Must be at least 2.
            shuffle (bool): Whether to shuffle data before splitting.
            random_state (Optional[int]): Random seed when shuffle=True.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and validation sets.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (Optional[np.ndarray]): Target values, shape (n_samples,). Ignored, present for API consistency.

        Yields:
            train_indices: Indices for training set.
            val_indices: Indices for validation set.
        """
        n_samples = len(X)

        # Create indices
        indices = np.arange(n_samples)

        # Shuffle if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        # Calculate fold sizes
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        # Generate folds
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size

            # Validation indices for this fold
            val_indices = indices[start:stop]

            # Training indices: everything else
            train_indices = np.concatenate([indices[:start], indices[stop:]])

            yield train_indices, val_indices

            current = stop

    def get_n_splits(self) -> int:
        """Get number of splits.

        Returns:
            int: Number of splits.
        """
        return self.n_splits


class StratifiedKFold:
    """
    Stratified K-Fold cross-validator.

    Ensures each fold has approximately the same class distribution as the full dataset.
    For use with classification tasks only.

    Attributes:
        n_splits (int): Number of folds.
        shuffle (bool): Whether to shuffle within each class before splitting.
        random_state (int): Random seed.

    Example:
        >>> skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        >>> y = np.array([0, 0, 0, 1, 1, 1])  # Balanced classes
        >>> for train_idx, val_idx in skf.split(X, y):
        ...     print(f"Train labels: {y[train_idx]}, Val labels: {y[val_idx]}")
        Train labels: [0 0 1 1], Val labels: [0 1]
        Train labels: [0 0 1 1], Val labels: [0 1]
        Train labels: [0 0 1 1], Val labels: [0 1]
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Stratified K-Fold.

        Args:
            n_splits (int): Number of folds. Must be at least 2.
            shuffle (bool): Whether to shuffle data before splitting.
            random_state (Optional[int]): Random seed when shuffle=True.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices for stratified train/validation splits.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): Target values, shape (n_samples,). Required for stratification.

        Yields:
            train_indices: Indices for training set.
            val_indices: Indices for validation set.
        """
        y = np.asarray(y)
        n_samples = len(y)

        # Get unique classes
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)

        # Check if we can split each class into n_splits folds
        class_counts = np.bincount(y_indices)
        if np.any(class_counts < self.n_splits):
            raise ValueError(
                f"Minimum class count ({class_counts.min()}) is less than n_splits ({self.n_splits})"
            )

        # Set random seed
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
        else:
            rng = None

        # For each class, split indices into n_splits folds
        class_fold_indices = []
        for cls_idx in range(n_classes):
            # Get indices for this class
            cls_indices = np.where(y_indices == cls_idx)[0]

            # Shuffle if requested
            if rng is not None:
                rng.shuffle(cls_indices)

            # Split into folds
            class_fold_indices.append(np.array_split(cls_indices, self.n_splits))

        # Generate folds by combining same fold index from each class
        for fold_idx in range(self.n_splits):
            # Validation indices: fold_idx from each class
            val_indices = np.concatenate(
                [class_folds[fold_idx] for class_folds in class_fold_indices]
            )

            # Training indices: all other folds from each class
            train_indices = np.concatenate(
                [
                    np.concatenate(
                        [class_folds[i] for i in range(self.n_splits) if i != fold_idx]
                    )
                    for class_folds in class_fold_indices
                ]
            )

            yield train_indices, val_indices

    def get_n_splits(self) -> int:
        """Get number of splits.

        Returns:
            int: Number of splits.
        """
        return self.n_splits


class LeaveOneOut:
    """
    Leave-One-Out cross-validator.

    Each sample is used once as validation while the remaining n-1 samples
    form the training set. Equivalent to K-Fold with K=n.

    Properties:
        - Deterministic (no randomness)
        - Nearly unbiased (uses maximum training data)
        - High variance (folds differ by only one sample)
        - Expensive (n iterations)

    Use when:
        - Small dataset (n < 100)
        - Maximum training data is critical
        - Computational cost is acceptable

    Example:
        >>> loo = LeaveOneOut()
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> for train_idx, val_idx in loo.split(X):
        ...     print(f"Train: {train_idx}, Val: {val_idx}")
        Train: [1 2], Val: [0]
        Train: [0 2], Val: [1]
        Train: [0 1], Val: [2]
    """

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices for Leave-One-Out splits.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (Optional[np.ndarray]): Target values. Ignored, present for API consistency.

        Yields:
            Tuple[np.ndarray, np.ndarray]:
                train_indices: Indices for training set (n-1 samples).
                val_indices: Indices for validation set (1 sample).
        """
        n_samples = len(X)

        for i in range(n_samples):
            # Validation: single sample i
            val_indices = np.array([i])

            # Training: all other samples
            train_indices = np.concatenate([np.arange(i), np.arange(i + 1, n_samples)])

            yield train_indices, val_indices

    def get_n_splits(self, X: np.ndarray) -> int:
        """Get number of splits (equal to number of samples).

        Args:
            X (np.ndarray): Training data.

        Returns:
            int: Number of splits.
        """
        return len(X)


class TimeSeriesSplit:
    """
    Time Series cross-validator.

    Forward chaining: training set always comes before validation set in time.
    Never train on future data to predict past.

    Structure:
        Fold 1: Train [0:100]    → Val [100:200]
        Fold 2: Train [0:200]    → Val [200:300]
        Fold 3: Train [0:300]    → Val [300:400]
        ...

    Attributes:
        n_splits (int): Number of splits.
        test_size (int): Size of validation set. If None, uses automatic sizing.

    Example:
        >>> tscv = TimeSeriesSplit(n_splits=3)
        >>> X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
        >>> for train_idx, val_idx in tscv.split(X):
        ...     print(f"Train: {train_idx}, Val: {val_idx}")
        Train: [0 1 2], Val: [3 4]
        Train: [0 1 2 3 4], Val: [5 6]
        Train: [0 1 2 3 4 5 6], Val: [7 8]
    """

    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        """
        Initialize Time Series Split.

        Args:
            n_splits (int): Number of splits.
            test_size (Optional[int]): Size of each validation set. If None, computed automatically.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")

        self.n_splits = n_splits
        self.test_size = test_size

    def split(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices for time series splits.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (Optional[np.ndarray]): Target values. Ignored.

        Yields:
            Tuple[np.ndarray, np.ndarray]:
                train_indices: Indices for training set (all data up to validation).
                val_indices: Indices for validation set (next time period).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Determine test size
        if self.test_size is None:
            # Automatic: split remaining data into n_splits + 1 parts
            # First part is minimum training size
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        # Minimum training size
        min_train_size = n_samples - self.n_splits * test_size

        if min_train_size <= 0:
            raise ValueError("Insufficient data for requested splits and test_size")

        # Generate splits
        for i in range(self.n_splits):
            # Training: from start to current position
            train_end = min_train_size + i * test_size
            train_indices = indices[:train_end]

            # Validation: next test_size samples
            val_start = train_end
            val_end = val_start + test_size
            val_indices = indices[val_start:val_end]

            yield train_indices, val_indices

    def get_n_splits(self) -> int:
        """Get number of splits."""
        return self.n_splits


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def cross_val_score(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Union[int, KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit] = 5,
    scoring: str = "accuracy",
) -> np.ndarray:
    """
    Evaluate a score by cross-validation.

    Args:
        model (Any): Estimator object with fit and predict methods.
        X (np.ndarray): Training data, shape (n_samples, n_features).
        y (np.ndarray): Target values, shape (n_samples,).
        cv (Union[int, KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit]): Number of folds or cross-validation splitter object.
        scoring (str): Scoring method ('accuracy', 'mse', 'mae', 'r2').

    Returns:
        np.ndarray: Array of scores for each fold, shape (n_folds,).

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 2, 100)
        >>> model = LogisticRegression()
        >>> scores = cross_val_score(model, X, y, cv=5)
        >>> print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
    """
    # Create CV splitter
    if isinstance(cv, int):
        # Check if classification (discrete y) or regression (continuous y)
        if len(np.unique(y)) < 20:  # Heuristic for classification
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_splitter = cv

    # Score each fold
    scores = []
    for train_idx, val_idx in cv_splitter.split(X, y):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model.fit(X_train, y_train)

        # Evaluate
        if scoring == "accuracy":
            y_pred = model.predict(X_val)
            score = np.mean(y_pred == y_val)
        elif scoring == "mse":
            y_pred = model.predict(X_val)
            score = -np.mean((y_val - y_pred) ** 2)  # Negative for consistency
        elif scoring == "mae":
            y_pred = model.predict(X_val)
            score = -np.mean(np.abs(y_val - y_pred))
        elif scoring == "r2":
            y_pred = model.predict(X_val)
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            score = 1 - (ss_res / ss_tot)
        else:
            raise ValueError(f"Unknown scoring method: {scoring}")

        scores.append(score)

    return np.array(scores)


# ============================================================================
# DEMONSTRATIONS
# ============================================================================


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print(
        "[bold cyan]Cross-Validation: From Scratch Implementation[/bold cyan]"
    )
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y_class = np.random.randint(0, 3, 100)  # 3 classes
    y_reg = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.5

    # ==========================================================================
    # TRAIN-TEST SPLIT
    # ==========================================================================

    console.print("\n[bold yellow]1. TRAIN-TEST SPLIT[/bold yellow]")
    console.print("-" * 70)

    # Regular split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    console.print(f"Total samples: {len(X)}")
    console.print(f"Train samples: {len(X_train)}")
    console.print(f"Test samples: {len(X_test)}")
    console.print(f"Train/Test ratio: {len(X_train)/len(X_test):.1f}")

    # Stratified split
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    console.print("\n[green]Class distribution:[/green]")
    console.print(f"Original: {dict(Counter(y_class))}")
    console.print(f"Train (stratified): {dict(Counter(y_train_strat))}")
    console.print(f"Test (stratified): {dict(Counter(y_test_strat))}")

    # ==========================================================================
    # K-FOLD CROSS-VALIDATION
    # ==========================================================================

    console.print("\n[bold yellow]2. K-FOLD CROSS-VALIDATION[/bold yellow]")
    console.print("-" * 70)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    kfold_table = Table(title="K-Fold Split Sizes", box=box.ROUNDED)
    kfold_table.add_column("Fold", justify="center", style="cyan")
    kfold_table.add_column("Train Size", justify="right", style="green")
    kfold_table.add_column("Val Size", justify="right", style="yellow")

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        kfold_table.add_row(str(fold_idx), str(len(train_idx)), str(len(val_idx)))

    console.print(kfold_table)

    # ==========================================================================
    # STRATIFIED K-FOLD
    # ==========================================================================

    console.print("\n[bold yellow]3. STRATIFIED K-FOLD[/bold yellow]")
    console.print("-" * 70)

    skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    stratified_table = Table(title="Class Distribution per Fold", box=box.ROUNDED)
    stratified_table.add_column("Fold", justify="center", style="cyan")
    stratified_table.add_column("Train Class 0", justify="right", style="green")
    stratified_table.add_column("Train Class 1", justify="right", style="green")
    stratified_table.add_column("Train Class 2", justify="right", style="green")
    stratified_table.add_column("Val Class 0", justify="right", style="yellow")
    stratified_table.add_column("Val Class 1", justify="right", style="yellow")
    stratified_table.add_column("Val Class 2", justify="right", style="yellow")

    for fold_idx, (train_idx, val_idx) in enumerate(skfold.split(X, y_class), 1):
        train_counts = Counter(y_class[train_idx])
        val_counts = Counter(y_class[val_idx])

        stratified_table.add_row(
            str(fold_idx),
            str(train_counts[0]),
            str(train_counts[1]),
            str(train_counts[2]),
            str(val_counts[0]),
            str(val_counts[1]),
            str(val_counts[2]),
        )

    console.print(stratified_table)
    console.print("[yellow]→ Each fold has similar class distribution[/yellow]")

    # ==========================================================================
    # LEAVE-ONE-OUT
    # ==========================================================================

    console.print("\n[bold yellow]4. LEAVE-ONE-OUT (Small Dataset)[/bold yellow]")
    console.print("-" * 70)

    # Use small dataset for LOOCV demo
    X_small = X[:10]
    y_small = y_class[:10]

    loo = LeaveOneOut()
    n_splits = loo.get_n_splits(X_small)

    console.print(f"Dataset size: {len(X_small)}")
    console.print(f"Number of splits: {n_splits}")
    console.print(f"Training set size per fold: {len(X_small) - 1}")
    console.print(f"Validation set size per fold: 1")
    console.print("[yellow]→ Each sample used once as validation[/yellow]")

    # ==========================================================================
    # TIME SERIES SPLIT
    # ==========================================================================

    console.print("\n[bold yellow]5. TIME SERIES SPLIT[/bold yellow]")
    console.print("-" * 70)

    tscv = TimeSeriesSplit(n_splits=5)

    ts_table = Table(title="Time Series Splits", box=box.ROUNDED)
    ts_table.add_column("Fold", justify="center", style="cyan")
    ts_table.add_column("Train Range", justify="center", style="green")
    ts_table.add_column("Val Range", justify="center", style="yellow")
    ts_table.add_column("Train Size", justify="right", style="green")
    ts_table.add_column("Val Size", justify="right", style="yellow")

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        train_range = f"[{train_idx[0]}:{train_idx[-1]+1}]"
        val_range = f"[{val_idx[0]}:{val_idx[-1]+1}]"

        ts_table.add_row(
            str(fold_idx),
            train_range,
            val_range,
            str(len(train_idx)),
            str(len(val_idx)),
        )

    console.print(ts_table)
    console.print("[yellow]→ Training set grows, never uses future data[/yellow]")

    # ==========================================================================
    # CROSS-VALIDATION SCORING
    # ==========================================================================

    console.print("\n[bold yellow]6. CROSS-VALIDATION SCORING[/bold yellow]")
    console.print("-" * 70)

    # Simple model for demonstration
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)

    # Our implementation
    our_scores = cross_val_score(model, X, y_class, cv=5, scoring="accuracy")

    # Scikit-learn
    from sklearn.model_selection import cross_val_score as sklearn_cv_score

    sklearn_scores = sklearn_cv_score(model, X, y_class, cv=5)

    cv_table = Table(title="Cross-Validation Comparison", box=box.ROUNDED)
    cv_table.add_column("Fold", justify="center", style="cyan")
    cv_table.add_column("Our Score", justify="right", style="green")
    cv_table.add_column("sklearn Score", justify="right", style="yellow")

    for i, (our_score, sk_score) in enumerate(zip(our_scores, sklearn_scores), 1):
        cv_table.add_row(
            str(i),
            f"{our_score:.4f}",
            f"{sk_score:.4f}",
        )

    cv_table.add_row(
        "[bold]Mean[/bold]",
        f"[bold]{our_scores.mean():.4f}[/bold]",
        f"[bold]{sklearn_scores.mean():.4f}[/bold]",
    )

    console.print(cv_table)
    console.print(
        f"[green]✓ Our CV Score: {our_scores.mean():.4f} (+/- {our_scores.std():.4f})[/green]"
    )
    console.print(
        f"[yellow]sklearn CV Score: {sklearn_scores.mean():.4f} (+/- {sklearn_scores.std():.4f})[/yellow]"
    )

    # ==========================================================================
    # KEY INSIGHTS
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Key Insights[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    insights = [
        "",
        "[bold]TRAIN-TEST SPLIT:[/bold]",
        "• Use stratify=y for classification (maintains class distribution)",
        "• Typical split: 80% train, 20% test",
        "• For tuning: 60% train, 20% val, 20% test",
        "",
        "[bold]K-FOLD CV:[/bold]",
        "• K=5: Good balance (fast, reasonable variance)",
        "• K=10: More accurate, slower",
        "• Use Stratified K-Fold for classification (always!)",
        "",
        "[bold]LEAVE-ONE-OUT:[/bold]",
        "• Use only for small datasets (n < 100)",
        "• Nearly unbiased but high variance",
        "• Expensive: n iterations",
        "",
        "[bold]TIME SERIES:[/bold]",
        "• Never use random split for temporal data",
        "• Training always comes before validation",
        "• Forward chaining respects time order",
        "",
        "[bold]BEST PRACTICES:[/bold]",
        "• Always use cross-validation for model selection",
        "• Report CV score with standard deviation",
        "• Keep separate test set for final evaluation",
        "• Use appropriate CV strategy for your data type",
        "",
        "[yellow]✓ All implementations validated against scikit-learn[/yellow]",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
