"""
Cross-Validation - Using scikit-learn API

Demonstrates how to use scikit-learn's cross-validation tools for model evaluation
and selection. Compare with cross_validation_from_scratch.py to understand
what sklearn does internally.

Key sklearn.model_selection tools:
- train_test_split: Split data into train/test sets
- KFold, StratifiedKFold: K-fold cross-validation
- LeaveOneOut: LOOCV
- TimeSeriesSplit: Time series CV
- cross_val_score: Evaluate model with CV
- cross_validate: More detailed CV with multiple metrics
- GridSearchCV, RandomizedSearchCV: Hyperparameter tuning with CV
"""

import numpy as np
from typing import Dict, List, Any, Union, Optional
from collections import Counter

# sklearn cross-validation tools
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
)


# ============================================================================
# BASIC SPLITS
# ============================================================================


def demo_train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> Dict[str, Any]:
    """
    Demonstrate train_test_split with various options.

    Args:
        X (np.ndarray): Features array, shape (n_samples, n_features).
        y (np.ndarray): Target array, shape (n_samples,).
        test_size (float): Proportion for test set.
        random_state (int): Random seed.

    Returns:
        Dict[str, Any]: Dictionary with split information and results.

    Example:
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 2, 100)
        >>> results = demo_train_test_split(X, y)
        >>> results['train_size'], results['test_size']
        (80, 20)
    """
    results = {}

    # Basic split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results["basic"] = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_labels": dict(Counter(y_train)),
        "test_labels": dict(Counter(y_test)),
    }

    # Stratified split (for classification)
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results["stratified"] = {
        "train_size": len(X_train_strat),
        "test_size": len(X_test_strat),
        "train_labels": dict(Counter(y_train_strat)),
        "test_labels": dict(Counter(y_test_strat)),
    }

    # Train-val-test split (nested splits)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp
    )  # 0.25 * 0.8 = 0.2

    results["train_val_test"] = {
        "train_size": len(X_train),  # 60%
        "val_size": len(X_val),  # 20%
        "test_size": len(X_test),  # 20%
    }

    return results


# ============================================================================
# CROSS-VALIDATION SPLITTERS
# ============================================================================


def demo_kfold_strategies(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5
) -> Dict[str, Any]:
    """
    Demonstrate different K-Fold strategies.

    Args:
        X (np.ndarray): Features array.
        y (np.ndarray): Target array.
        n_splits (int): Number of folds.

    Returns:
        Dict[str, Any]: Dictionary with fold information for each strategy.

    Example:
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 3, 100)
        >>> results = demo_kfold_strategies(X, y)
        >>> len(results['kfold']['folds'])
        5
    """
    results = {}

    # Standard K-Fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    kfold_info = []
    for train_idx, val_idx in kfold.split(X):
        kfold_info.append(
            {
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
        )
    results["kfold"] = {"folds": kfold_info, "n_splits": kfold.get_n_splits()}

    # Stratified K-Fold (for classification)
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    skfold_info = []
    for train_idx, val_idx in skfold.split(X, y):
        skfold_info.append(
            {
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_dist": dict(Counter(y[train_idx])),
                "val_dist": dict(Counter(y[val_idx])),
            }
        )
    results["stratified_kfold"] = {
        "folds": skfold_info,
        "n_splits": skfold.get_n_splits(),
    }

    # Leave-One-Out (on small subset for demo)
    X_small = X[:20]
    y_small = y[:20]
    loo = LeaveOneOut()
    results["leave_one_out"] = {
        "n_splits": loo.get_n_splits(X_small),
        "train_size_per_fold": len(X_small) - 1,
        "val_size_per_fold": 1,
    }

    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    tscv_info = []
    for train_idx, val_idx in tscv.split(X):
        tscv_info.append(
            {
                "train_range": (train_idx[0], train_idx[-1] + 1),
                "val_range": (val_idx[0], val_idx[-1] + 1),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
        )
    results["time_series"] = {"folds": tscv_info, "n_splits": tscv.get_n_splits()}

    return results


# ============================================================================
# CROSS-VALIDATION SCORING
# ============================================================================


def demo_cross_val_score(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    cv: Union[int, KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit] = 5,
) -> Dict[str, Any]:
    """
    Demonstrate cross_val_score with different metrics.

    Args:
        X (np.ndarray): Features array.
        y (np.ndarray): Target array.
        model (Any): Estimator with fit/predict methods.
        cv (Union[int, KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit]): Number of CV folds or splitter object.

    Returns:
        Dict[str, Any]: Dictionary with scores for different metrics.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 2, 100)
        >>> model = LogisticRegression()
        >>> results = demo_cross_val_score(X, y, model)
        >>> 'accuracy' in results
        True
    """
    results = {}

    # Classification metrics
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"]

    for metric in metrics:
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            results[metric] = {
                "scores": scores,
                "mean": scores.mean(),
                "std": scores.std(),
            }
        except Exception as e:
            results[metric] = {"error": str(e)}

    return results


def demo_cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    cv: Union[int, KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit] = 5,
) -> Dict[str, Any]:
    """
    Demonstrate cross_validate for multiple metrics and timing.

    cross_validate is more flexible than cross_val_score:
    - Multiple metrics simultaneously
    - Returns train scores (if requested)
    - Includes timing information
    - Returns fitted estimators (if requested)

    Args:
        X (np.ndarray): Features array.
        y (np.ndarray): Target array.
        model (Any): Estimator.
        cv (Union[int, KFold, StratifiedKFold, LeaveOneOut, TimeSeriesSplit]): Number of CV folds or splitter object.

    Returns:
        Dict[str, Any]: Dictionary with detailed CV results.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 2, 100)
        >>> model = LogisticRegression()
        >>> results = demo_cross_validate(X, y, model)
        >>> 'test_accuracy' in results
        True
    """
    # Multiple scoring metrics
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }

    # Get detailed CV results
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,  # Include train scores
        return_estimator=False,  # Don't return fitted models (saves memory)
        n_jobs=-1,  # Parallel execution
    )

    # Format results
    results = {
        "fit_time": {
            "mean": cv_results["fit_time"].mean(),
            "std": cv_results["fit_time"].std(),
        },
        "score_time": {
            "mean": cv_results["score_time"].mean(),
            "std": cv_results["score_time"].std(),
        },
    }

    # Add train and test scores for each metric
    for metric in scoring.keys():
        results[f"train_{metric}"] = {
            "mean": cv_results[f"train_{metric}"].mean(),
            "std": cv_results[f"train_{metric}"].std(),
        }
        results[f"test_{metric}"] = {
            "mean": cv_results[f"test_{metric}"].mean(),
            "std": cv_results[f"test_{metric}"].std(),
        }

    return results


# ============================================================================
# HYPERPARAMETER TUNING WITH CV
# ============================================================================


def demo_grid_search_cv(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Demonstrate GridSearchCV for hyperparameter tuning.

    GridSearchCV:
    - Exhaustive search over specified parameter grid
    - Uses cross-validation to evaluate each combination
    - Returns best parameters and best score
    - Refits on full dataset with best parameters

    Args:
        X (np.ndarray): Features array.
        y (np.ndarray): Target array.

    Returns:
        Dict[str, Any]: Dictionary with grid search results.

    Example:
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 2, 100)
        >>> results = demo_grid_search_cv(X, y)
        >>> 'best_params' in results
        True
    """
    from sklearn.svm import SVC

    # Define parameter grid
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }

    # Create GridSearchCV
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1,
    )

    # Fit (searches all combinations)
    grid_search.fit(X, y)

    # Extract results
    results = {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "n_combinations": len(grid_search.cv_results_["params"]),
        "best_estimator": str(grid_search.best_estimator_),
    }

    # Top 3 combinations
    cv_results = grid_search.cv_results_
    sorted_idx = np.argsort(cv_results["rank_test_score"])[:3]

    results["top_3"] = [
        {
            "params": cv_results["params"][i],
            "mean_test_score": cv_results["mean_test_score"][i],
            "std_test_score": cv_results["std_test_score"][i],
        }
        for i in sorted_idx
    ]

    return results


def demo_randomized_search_cv(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Demonstrate RandomizedSearchCV for efficient hyperparameter tuning.

    RandomizedSearchCV:
    - Samples random combinations from parameter distributions
    - More efficient than GridSearch for large parameter spaces
    - Can specify continuous distributions (e.g., uniform, log-uniform)
    - Set n_iter to control budget

    Args:
        X (np.ndarray): Features array.
        y (np.ndarray): Target array.

    Returns:
        Dict[str, Any]: Dictionary with randomized search results.

    Example:
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 2, 100)
        >>> results = demo_randomized_search_cv(X, y)
        >>> 'best_params' in results
        True
    """
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import randint, uniform

    # Define parameter distributions
    param_distributions = {
        "n_estimators": randint(50, 200),  # Random integers
        "max_depth": randint(3, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": uniform(0.1, 0.9),  # Random floats
    }

    # Create RandomizedSearchCV
    random_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions,
        n_iter=20,  # Number of parameter combinations to try
        cv=5,
        scoring="accuracy",
        random_state=42,
        n_jobs=-1,
    )

    # Fit
    random_search.fit(X, y)

    # Extract results
    results = {
        "best_params": random_search.best_params_,
        "best_score": random_search.best_score_,
        "n_iterations": random_search.n_iter,
        "best_estimator": str(random_search.best_estimator_),
    }

    # Top 3 combinations
    cv_results = random_search.cv_results_
    sorted_idx = np.argsort(cv_results["rank_test_score"])[:3]

    results["top_3"] = [
        {
            "params": cv_results["params"][i],
            "mean_test_score": cv_results["mean_test_score"][i],
            "std_test_score": cv_results["std_test_score"][i],
        }
        for i in sorted_idx
    ]

    return results


# ============================================================================
# DEMONSTRATIONS
# ============================================================================


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Cross-Validation: Using scikit-learn API[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y_class = np.random.randint(0, 3, 200)  # 3 classes

    # ==========================================================================
    # TRAIN-TEST SPLIT
    # ==========================================================================

    console.print("\n[bold yellow]1. TRAIN-TEST SPLIT[/bold yellow]")
    console.print("-" * 70)

    split_results = demo_train_test_split(X, y_class)

    split_table = Table(title="Train-Test Split Comparison", box=box.ROUNDED)
    split_table.add_column("Split Type", style="cyan")
    split_table.add_column("Train Size", justify="right", style="green")
    split_table.add_column("Test Size", justify="right", style="yellow")
    split_table.add_column("Train Dist", style="green")
    split_table.add_column("Test Dist", style="yellow")

    for split_type, data in split_results.items():
        if split_type == "train_val_test":
            continue
        split_table.add_row(
            split_type.title(),
            str(data["train_size"]),
            str(data["test_size"]),
            str(data["train_labels"]),
            str(data["test_labels"]),
        )

    console.print(split_table)

    console.print("\n[green]Train-Val-Test Split:[/green]")
    tvt = split_results["train_val_test"]
    console.print(
        f"  Train: {tvt['train_size']} (60%) | Val: {tvt['val_size']} (20%) | Test: {tvt['test_size']} (20%)"
    )

    # ==========================================================================
    # K-FOLD STRATEGIES
    # ==========================================================================

    console.print("\n[bold yellow]2. CROSS-VALIDATION STRATEGIES[/bold yellow]")
    console.print("-" * 70)

    cv_strategies = demo_kfold_strategies(X, y_class)

    # K-Fold
    console.print("\n[green]K-Fold (5 splits):[/green]")
    kfold_table = Table(box=box.SIMPLE)
    kfold_table.add_column("Fold", justify="center", style="cyan")
    kfold_table.add_column("Train Size", justify="right", style="green")
    kfold_table.add_column("Val Size", justify="right", style="yellow")

    for i, fold in enumerate(cv_strategies["kfold"]["folds"], 1):
        kfold_table.add_row(str(i), str(fold["train_size"]), str(fold["val_size"]))

    console.print(kfold_table)

    # Stratified K-Fold
    console.print("\n[green]Stratified K-Fold (maintains class distribution):[/green]")
    strat_table = Table(box=box.SIMPLE)
    strat_table.add_column("Fold", justify="center", style="cyan")
    strat_table.add_column("Train Dist", style="green")
    strat_table.add_column("Val Dist", style="yellow")

    for i, fold in enumerate(cv_strategies["stratified_kfold"]["folds"], 1):
        strat_table.add_row(str(i), str(fold["train_dist"]), str(fold["val_dist"]))

    console.print(strat_table)

    # Time Series Split
    console.print("\n[green]Time Series Split (forward chaining):[/green]")
    ts_table = Table(box=box.SIMPLE)
    ts_table.add_column("Fold", justify="center", style="cyan")
    ts_table.add_column("Train Range", style="green")
    ts_table.add_column("Val Range", style="yellow")
    ts_table.add_column("Sizes", style="magenta")

    for i, fold in enumerate(cv_strategies["time_series"]["folds"], 1):
        train_range = f"[{fold['train_range'][0]}:{fold['train_range'][1]}]"
        val_range = f"[{fold['val_range'][0]}:{fold['val_range'][1]}]"
        sizes = f"{fold['train_size']} / {fold['val_size']}"
        ts_table.add_row(str(i), train_range, val_range, sizes)

    console.print(ts_table)

    # Leave-One-Out
    loo_info = cv_strategies["leave_one_out"]
    console.print(f"\n[green]Leave-One-Out:[/green]")
    console.print(
        f"  Splits: {loo_info['n_splits']} | Train per fold: {loo_info['train_size_per_fold']} | Val per fold: {loo_info['val_size_per_fold']}"
    )

    # ==========================================================================
    # CROSS-VALIDATION SCORING
    # ==========================================================================

    console.print("\n[bold yellow]3. CROSS-VALIDATION SCORING[/bold yellow]")
    console.print("-" * 70)

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=50, random_state=42)

    # cross_val_score
    console.print("\n[green]cross_val_score (single metric):[/green]")
    accuracy_scores = cross_val_score(model, X, y_class, cv=5, scoring="accuracy")
    console.print(
        f"  Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})"
    )
    console.print(f"  Per fold: {[f'{s:.4f}' for s in accuracy_scores]}")

    # cross_validate
    console.print("\n[green]cross_validate (multiple metrics + timing):[/green]")
    cv_results = demo_cross_validate(X, y_class, model)

    metrics_table = Table(title="Cross-Validate Results", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Train Score", justify="right", style="green")
    metrics_table.add_column("Test Score", justify="right", style="yellow")

    for metric in ["accuracy", "precision", "recall", "f1"]:
        train_score = f"{cv_results[f'train_{metric}']['mean']:.4f} ± {cv_results[f'train_{metric}']['std']:.4f}"
        test_score = f"{cv_results[f'test_{metric}']['mean']:.4f} ± {cv_results[f'test_{metric}']['std']:.4f}"
        metrics_table.add_row(metric.title(), train_score, test_score)

    console.print(metrics_table)
    console.print(
        f"\n  [dim]Fit time: {cv_results['fit_time']['mean']:.4f}s ± {cv_results['fit_time']['std']:.4f}s[/dim]"
    )

    # ==========================================================================
    # HYPERPARAMETER TUNING
    # ==========================================================================

    console.print("\n[bold yellow]4. HYPERPARAMETER TUNING WITH CV[/bold yellow]")
    console.print("-" * 70)

    # Use smaller dataset for faster demo
    X_small = X[:100]
    y_small = y_class[:100]

    # GridSearchCV
    console.print("\n[green]GridSearchCV (exhaustive search):[/green]")
    grid_results = demo_grid_search_cv(X_small, y_small)

    console.print(f"  Combinations tested: {grid_results['n_combinations']}")
    console.print(f"  Best score: {grid_results['best_score']:.4f}")
    console.print(f"  Best params: {grid_results['best_params']}")

    grid_table = Table(title="Top 3 Combinations", box=box.SIMPLE)
    grid_table.add_column("Rank", justify="center", style="cyan")
    grid_table.add_column("Parameters", style="green")
    grid_table.add_column("Score", justify="right", style="yellow")

    for i, result in enumerate(grid_results["top_3"], 1):
        grid_table.add_row(
            str(i),
            str(result["params"]),
            f"{result['mean_test_score']:.4f} ± {result['std_test_score']:.4f}",
        )

    console.print(grid_table)

    # RandomizedSearchCV
    console.print("\n[green]RandomizedSearchCV (efficient random sampling):[/green]")
    random_results = demo_randomized_search_cv(X_small, y_small)

    console.print(f"  Iterations: {random_results['n_iterations']}")
    console.print(f"  Best score: {random_results['best_score']:.4f}")
    console.print(f"  Best params: {random_results['best_params']}")

    random_table = Table(title="Top 3 Combinations", box=box.SIMPLE)
    random_table.add_column("Rank", justify="center", style="cyan")
    random_table.add_column("Parameters", style="green")
    random_table.add_column("Score", justify="right", style="yellow")

    for i, result in enumerate(random_results["top_3"], 1):
        random_table.add_row(
            str(i),
            str(result["params"]),
            f"{result['mean_test_score']:.4f} ± {result['std_test_score']:.4f}",
        )

    console.print(random_table)

    # ==========================================================================
    # KEY INSIGHTS
    # ==========================================================================

    console.print("\n[bold cyan]" + "=" * 70 + "[/bold cyan]")
    console.print("[bold cyan]Key sklearn.model_selection Tools[/bold cyan]")
    console.print("[bold cyan]" + "=" * 70 + "[/bold cyan]")

    insights = [
        "",
        "[bold]SPLITTING:[/bold]",
        "• train_test_split: Simple train/test or train/val/test split",
        "• Use stratify= for classification to maintain class balance",
        "",
        "[bold]CV SPLITTERS:[/bold]",
        "• KFold: Standard k-fold (use shuffle=True)",
        "• StratifiedKFold: Maintains class distribution (for classification)",
        "• TimeSeriesSplit: Forward chaining (never use future data)",
        "• LeaveOneOut: Exhaustive but expensive (small datasets only)",
        "",
        "[bold]SCORING:[/bold]",
        "• cross_val_score: Quick CV with single metric",
        "• cross_validate: More detailed (multiple metrics, timing, train scores)",
        "• Both support parallel execution (n_jobs=-1)",
        "",
        "[bold]HYPERPARAMETER TUNING:[/bold]",
        "• GridSearchCV: Exhaustive search (small parameter spaces)",
        "• RandomizedSearchCV: Random sampling (large parameter spaces)",
        "• Both automatically refit on full dataset with best params",
        "",
        "[bold]BEST PRACTICES:[/bold]",
        "• Always use StratifiedKFold for classification",
        "• Report mean ± std from CV",
        "• Use RandomizedSearchCV for initial exploration, GridSearchCV to refine",
        "• Keep separate held-out test set for final evaluation",
        "• Use n_jobs=-1 for faster execution on multi-core machines",
        "",
        "[yellow]💡 Compare with cross_validation_from_scratch.py to see what sklearn does internally![/yellow]",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
