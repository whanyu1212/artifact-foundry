"""
Basic Cross-Validation Examples

Demonstrates fundamental cross-validation techniques:
- K-Fold vs Stratified K-Fold
- Leave-One-Out Cross-Validation
- Time Series Split
- Data leakage pitfalls and how to avoid them

Educational focus: Understanding when and how to use each CV method.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box


console = Console()


def demonstrate_kfold_vs_stratified() -> None:
    """
    Compare K-Fold and Stratified K-Fold on imbalanced data.

    Shows why stratification matters for classification tasks,
    especially with class imbalance.
    """
    console.print("\n[bold cyan]K-Fold vs Stratified K-Fold on Imbalanced Data[/bold cyan]\n")

    # Create imbalanced dataset: 90% class 0, 10% class 1
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        random_state=42
    )

    console.print(f"Dataset: {len(y)} samples")
    console.print(f"Class distribution: {np.bincount(y)} ({np.bincount(y)/len(y)*100}%)\n")

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Regular K-Fold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    table_kf = Table(title="K-Fold (No Stratification)", box=box.ROUNDED)
    table_kf.add_column("Fold", style="cyan")
    table_kf.add_column("Train Class 0", justify="right")
    table_kf.add_column("Train Class 1", justify="right")
    table_kf.add_column("Test Class 0", justify="right")
    table_kf.add_column("Test Class 1", justify="right")
    table_kf.add_column("Score", justify="right", style="yellow")

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X[train_idx], y_train)
        score = clf.score(X[test_idx], y_test)

        train_counts = np.bincount(y_train, minlength=2)
        test_counts = np.bincount(y_test, minlength=2)

        table_kf.add_row(
            str(fold),
            f"{train_counts[0]} ({train_counts[0]/len(y_train)*100:.1f}%)",
            f"{train_counts[1]} ({train_counts[1]/len(y_train)*100:.1f}%)",
            f"{test_counts[0]} ({test_counts[0]/len(y_test)*100:.1f}%)",
            f"{test_counts[1]} ({test_counts[1]/len(y_test)*100:.1f}%)",
            f"{score:.3f}"
        )

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    table_skf = Table(title="Stratified K-Fold (With Stratification)", box=box.ROUNDED)
    table_skf.add_column("Fold", style="cyan")
    table_skf.add_column("Train Class 0", justify="right")
    table_skf.add_column("Train Class 1", justify="right")
    table_skf.add_column("Test Class 0", justify="right")
    table_skf.add_column("Test Class 1", justify="right")
    table_skf.add_column("Score", justify="right", style="yellow")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X[train_idx], y_train)
        score = clf.score(X[test_idx], y_test)

        train_counts = np.bincount(y_train, minlength=2)
        test_counts = np.bincount(y_test, minlength=2)

        table_skf.add_row(
            str(fold),
            f"{train_counts[0]} ({train_counts[0]/len(y_train)*100:.1f}%)",
            f"{train_counts[1]} ({train_counts[1]/len(y_train)*100:.1f}%)",
            f"{test_counts[0]} ({test_counts[0]/len(y_test)*100:.1f}%)",
            f"{test_counts[1]} ({test_counts[1]/len(y_test)*100:.1f}%)",
            f"{score:.3f}"
        )

    console.print(table_kf)
    console.print()
    console.print(table_skf)

    # Key takeaway
    console.print("\n[bold green]Key Observation:[/bold green]")
    console.print("• K-Fold: Class distribution varies across folds (some have 0-2 samples of class 1!)")
    console.print("• Stratified K-Fold: Maintains ~90%/10% split in every fold ✓")
    console.print("• [bold]Always use StratifiedKFold for classification![/bold]\n")


def demonstrate_data_leakage() -> None:
    """
    Show the impact of data leakage from improper preprocessing.

    Demonstrates WRONG vs CORRECT way to apply scaling with CV.
    Key lesson: Fit preprocessing only on training data, not all data.
    """
    console.print("\n[bold cyan]Data Leakage: Scaling Before vs After CV Split[/bold cyan]\n")

    X, y = load_breast_cancer(return_X_y=True)
    clf = LogisticRegression(max_iter=1000, random_state=42)

    # WRONG: Scale before splitting (data leakage)
    scaler_wrong = StandardScaler()
    X_scaled_wrong = scaler_wrong.fit_transform(X)  # Uses test set statistics!

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_wrong = cross_val_score(clf, X_scaled_wrong, y, cv=cv)

    # CORRECT: Scale within each fold
    scores_correct = []
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit scaler ONLY on training data
        scaler_correct = StandardScaler()
        X_train_scaled = scaler_correct.fit_transform(X_train)
        X_test_scaled = scaler_correct.transform(X_test)  # Apply, don't fit!

        clf.fit(X_train_scaled, y_train)
        scores_correct.append(clf.score(X_test_scaled, y_test))

    scores_correct = np.array(scores_correct)

    # Display results
    table = Table(title="Impact of Data Leakage", box=box.DOUBLE)
    table.add_column("Method", style="cyan")
    table.add_column("Mean Score", justify="right", style="yellow")
    table.add_column("Std Dev", justify="right")
    table.add_column("Status", justify="center")

    table.add_row(
        "Scale BEFORE split (WRONG)",
        f"{scores_wrong.mean():.4f}",
        f"{scores_wrong.std():.4f}",
        "[red]✗ Optimistic![/red]"
    )
    table.add_row(
        "Scale WITHIN split (CORRECT)",
        f"{scores_correct.mean():.4f}",
        f"{scores_correct.std():.4f}",
        "[green]✓ Realistic[/green]"
    )

    console.print(table)

    # Explanation
    console.print("\n[bold green]Why This Happens:[/bold green]")
    console.print("• WRONG method: Test set statistics leak into training (mean, std computed on ALL data)")
    console.print("• CORRECT method: Each fold's test set is truly unseen during preprocessing")
    console.print("• Difference may seem small, but it's a systematic bias")
    console.print("• [bold]Rule: Fit preprocessing ONLY on training data![/bold]\n")


def demonstrate_time_series_cv() -> None:
    """
    Demonstrate Time Series Cross-Validation.

    Shows why regular K-Fold is wrong for time series (data leakage from future).
    TimeSeriesSplit respects temporal order.
    """
    console.print("\n[bold cyan]Time Series Cross-Validation[/bold cyan]\n")

    # Create simple time series data (100 time steps)
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    # y is cumulative sum (has temporal dependency)
    y = np.cumsum(np.random.randn(n_samples)) + X[:, 0] * 2

    console.print(f"Dataset: {n_samples} time steps, 5 features\n")

    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=5)

    table = Table(title="Time Series Split (Respects Temporal Order)", box=box.ROUNDED)
    table.add_column("Fold", style="cyan", justify="center")
    table.add_column("Train Range", justify="center")
    table.add_column("Train Size", justify="right")
    table.add_column("Test Range", justify="center")
    table.add_column("Test Size", justify="right")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        table.add_row(
            str(fold),
            f"[{train_idx[0]:3d}:{train_idx[-1]:3d}]",
            str(len(train_idx)),
            f"[{test_idx[0]:3d}:{test_idx[-1]:3d}]",
            str(len(test_idx))
        )

    console.print(table)

    # Key visualization
    console.print("\n[bold green]Visual Representation:[/bold green]")
    console.print("Fold 1: [blue]Train.................[/blue]|[red]Test...[/red]")
    console.print("Fold 2: [blue]Train....................[/blue]|[red]Test...[/red]")
    console.print("Fold 3: [blue]Train.......................[/blue]|[red]Test...[/red]")
    console.print("Fold 4: [blue]Train..........................[/blue]|[red]Test...[/red]")
    console.print("Fold 5: [blue]Train..............................[/blue]|[red]Test...[/red]")

    console.print("\n[bold yellow]Key Properties:[/bold yellow]")
    console.print("• Training set grows with each fold (uses all past data)")
    console.print("• Test set always comes AFTER training set (no future leakage)")
    console.print("• Models are tested on progressively future time periods")
    console.print("• [bold]Never shuffle time series data![/bold]\n")


def demonstrate_loo_cv() -> None:
    """
    Demonstrate Leave-One-Out Cross-Validation.

    Shows when LOO is useful (small datasets) and when it's impractical (large datasets).
    Compares with K-Fold in terms of computation and results.
    """
    console.print("\n[bold cyan]Leave-One-Out Cross-Validation (LOO)[/bold cyan]\n")

    # Small dataset for LOO demonstration
    X, y = make_classification(
        n_samples=50,  # Small dataset
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )

    console.print(f"Dataset: {len(y)} samples (small dataset ideal for LOO)\n")

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)

    # LOO-CV: K = N
    loo = LeaveOneOut()
    scores_loo = cross_val_score(clf, X, y, cv=loo)

    # Compare with 5-Fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_kf = cross_val_score(clf, X, y, cv=kf)

    # Compare with 10-Fold
    kf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores_kf10 = cross_val_score(clf, X, y, cv=kf10)

    # Display results
    table = Table(title="CV Method Comparison", box=box.DOUBLE)
    table.add_column("Method", style="cyan")
    table.add_column("# Folds", justify="right")
    table.add_column("Train Size", justify="right")
    table.add_column("Test Size", justify="right")
    table.add_column("Mean Score", justify="right", style="yellow")
    table.add_column("Std Dev", justify="right")

    table.add_row(
        "5-Fold CV",
        "5",
        "40",
        "10",
        f"{scores_kf.mean():.4f}",
        f"{scores_kf.std():.4f}"
    )
    table.add_row(
        "10-Fold CV",
        "10",
        "45",
        "5",
        f"{scores_kf10.mean():.4f}",
        f"{scores_kf10.std():.4f}"
    )
    table.add_row(
        "LOO-CV",
        str(len(y)),
        "49",
        "1",
        f"{scores_loo.mean():.4f}",
        f"{scores_loo.std():.4f}"
    )

    console.print(table)

    console.print("\n[bold green]Trade-offs:[/bold green]")
    console.print("• LOO uses maximum training data (N-1 samples)")
    console.print("• LOO requires N model fits (expensive for large datasets)")
    console.print("• LOO has high variance (predictions are highly correlated)")
    console.print("• [bold]Use LOO only for very small datasets (N < 100)[/bold]")
    console.print("• For most cases, 5-Fold or 10-Fold is better\n")


def main() -> None:
    """Run all basic cross-validation demonstrations."""
    console.print(Panel.fit(
        "[bold white]Basic Cross-Validation Examples[/bold white]\n"
        "Demonstrates fundamental CV techniques and common pitfalls",
        border_style="bright_blue"
    ))

    demonstrate_kfold_vs_stratified()
    demonstrate_data_leakage()
    demonstrate_time_series_cv()
    demonstrate_loo_cv()

    console.print(Panel.fit(
        "[bold green]✓ All demonstrations completed![/bold green]\n"
        "Key takeaways:\n"
        "• Use StratifiedKFold for classification\n"
        "• Fit preprocessing only on training data\n"
        "• Use TimeSeriesSplit for temporal data\n"
        "• Choose K=5 or K=10 for most problems",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
