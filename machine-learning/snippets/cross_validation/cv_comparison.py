"""
Cross-Validation Strategy Comparison

Side-by-side comparison of different CV methods:
- Performance metrics across methods
- Computation time comparison
- Visual representations of fold structures
- Recommendations for different scenarios

Helps choose the right CV strategy for your problem.
"""

import numpy as np
import time
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    LeaveOneOut,
    TimeSeriesSplit,
    ShuffleSplit,
    cross_val_score,
)
from sklearn.tree import DecisionTreeClassifier
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box
from typing import Tuple, List, Dict


console = Console()


def compare_cv_methods_performance() -> None:
    """
    Compare different CV methods on the same dataset.

    Evaluates performance, variance, and computation time for:
    - K-Fold (K=5, 10)
    - Stratified K-Fold
    - Leave-One-Out
    - Shuffle Split
    """
    console.print("\n[bold cyan]CV Methods Performance Comparison[/bold cyan]\n")

    # Use smaller dataset for LOO to be practical
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )

    clf = DecisionTreeClassifier(max_depth=5, random_state=42)

    methods = {
        "K-Fold (K=5)": KFold(n_splits=5, shuffle=True, random_state=42),
        "K-Fold (K=10)": KFold(n_splits=10, shuffle=True, random_state=42),
        "Stratified K-Fold (K=5)": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "Stratified K-Fold (K=10)": StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        "Leave-One-Out": LeaveOneOut(),
        "Shuffle Split (10 iter)": ShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
    }

    results = []

    for name, cv_method in methods.items():
        start_time = time.time()

        # Run cross-validation
        if "Stratified" in name or "Shuffle" in name:
            scores = cross_val_score(clf, X, y, cv=cv_method)
        else:
            scores = cross_val_score(clf, X, y, cv=cv_method)

        elapsed_time = time.time() - start_time

        results.append({
            'name': name,
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'n_folds': len(scores),
            'time': elapsed_time
        })

    # Display results
    table = Table(title="Cross-Validation Methods Comparison", box=box.DOUBLE)
    table.add_column("Method", style="cyan")
    table.add_column("# Folds", justify="right")
    table.add_column("Mean Score", justify="right", style="yellow")
    table.add_column("Std Dev", justify="right")
    table.add_column("Time (s)", justify="right", style="green")

    for result in results:
        table.add_row(
            result['name'],
            str(result['n_folds']),
            f"{result['mean_score']:.4f}",
            f"{result['std_score']:.4f}",
            f"{result['time']:.3f}"
        )

    console.print(table)
    console.print()


def visualize_fold_structures() -> None:
    """
    Visualize how different CV methods split data.

    Shows ASCII art representations of train/test splits for each method.
    """
    console.print("\n[bold cyan]Fold Structure Visualization[/bold cyan]\n")

    n_samples = 30  # Small number for visualization
    X = np.arange(n_samples).reshape(-1, 1)
    y = np.random.randint(0, 2, n_samples)

    # 5-Fold
    console.print("[bold yellow]K-Fold (K=5):[/bold yellow]")
    kf = KFold(n_splits=5, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        viz = ["·"] * n_samples
        for idx in test_idx:
            viz[idx] = "█"
        console.print(f"Fold {fold}: {''.join(viz)}")
    console.print("        █ = Test,  · = Train\n")

    # Stratified K-Fold
    console.print("[bold yellow]Stratified K-Fold (K=5):[/bold yellow]")
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        viz = ["·"] * n_samples
        for idx in test_idx:
            viz[idx] = "█"
        console.print(f"Fold {fold}: {''.join(viz)}")
    console.print("        █ = Test,  · = Train (maintains class distribution)\n")

    # Time Series Split
    console.print("[bold yellow]Time Series Split (K=5):[/bold yellow]")
    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        viz = [" "] * n_samples
        for idx in train_idx:
            viz[idx] = "·"
        for idx in test_idx:
            viz[idx] = "█"
        console.print(f"Fold {fold}: {''.join(viz)}")
    console.print("        █ = Test,  · = Train,   = Unused (training grows)\n")


def compare_bias_variance_tradeoff() -> None:
    """
    Demonstrate bias-variance trade-off for different K values.

    Shows how choice of K affects:
    - Training set size (bias)
    - Variance of estimates
    - Computation cost
    """
    console.print("\n[bold cyan]Bias-Variance Trade-off: Choosing K[/bold cyan]\n")

    X, y = load_breast_cancer(return_X_y=True)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)

    k_values = [2, 3, 5, 10, 20]
    results = []

    for k in k_values:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        # Run CV multiple times to measure stability
        multiple_runs = []
        for seed in range(5):
            cv_run = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            scores = cross_val_score(clf, X, y, cv=cv_run)
            multiple_runs.append(scores.mean())

        multiple_runs = np.array(multiple_runs)
        train_size = X.shape[0] * (k - 1) / k

        results.append({
            'k': k,
            'train_size': int(train_size),
            'train_pct': (k - 1) / k * 100,
            'mean_score': multiple_runs.mean(),
            'run_variance': multiple_runs.std(),  # Variance across different runs
        })

    # Display results
    table = Table(title="Effect of K on Bias and Variance", box=box.ROUNDED)
    table.add_column("K", justify="right", style="cyan")
    table.add_column("Train Size", justify="right")
    table.add_column("Train %", justify="right")
    table.add_column("Mean Score", justify="right", style="yellow")
    table.add_column("Run Variance", justify="right", style="red")
    table.add_column("Bias", justify="center")

    for result in results:
        # Lower K = higher bias (less training data)
        bias_indicator = "High" if result['k'] <= 3 else "Medium" if result['k'] <= 5 else "Low"

        table.add_row(
            str(result['k']),
            str(result['train_size']),
            f"{result['train_pct']:.1f}%",
            f"{result['mean_score']:.4f}",
            f"{result['run_variance']:.4f}",
            bias_indicator
        )

    console.print(table)

    console.print("\n[bold green]Guidelines for Choosing K:[/bold green]")
    console.print("• [bold]K=5[/bold]: Standard choice, good bias-variance balance")
    console.print("• [bold]K=10[/bold]: Lower bias, slightly higher variance")
    console.print("• [bold]K=2[/bold]: Fast but high bias (small training set)")
    console.print("• [bold]K=N (LOO)[/bold]: Lowest bias but high variance & expensive")
    console.print("• For small datasets (< 1000): Use K=10")
    console.print("• For large datasets (> 100K): Use K=3 or simple split\n")


def scenario_recommendations() -> None:
    """
    Provide CV method recommendations for different scenarios.

    Practical guide for choosing the right CV strategy based on:
    - Dataset size
    - Problem type
    - Computational constraints
    - Class balance
    """
    console.print("\n[bold cyan]CV Method Selection Guide[/bold cyan]\n")

    scenarios = [
        {
            "Scenario": "Small balanced dataset\n(N < 1000)",
            "Recommended": "Stratified K-Fold\n(K=10)",
            "Why": "Maximizes training data,\nstratification provides stability"
        },
        {
            "Scenario": "Medium imbalanced dataset\n(1K-100K, imbalanced)",
            "Recommended": "Stratified K-Fold\n(K=5)",
            "Why": "Ensures each fold has\nrepresentative classes"
        },
        {
            "Scenario": "Large balanced dataset\n(N > 100K)",
            "Recommended": "K-Fold (K=3) or\nsimple train/test split",
            "Why": "Simple split is sufficient\nwith large data"
        },
        {
            "Scenario": "Time series data",
            "Recommended": "TimeSeriesSplit",
            "Why": "Respects temporal order,\navoids future leakage"
        },
        {
            "Scenario": "Very small dataset\n(N < 100)",
            "Recommended": "Leave-One-Out\nor K=N/5",
            "Why": "Maximizes training data\nfor each fold"
        },
        {
            "Scenario": "Hyperparameter tuning\n(reporting performance)",
            "Recommended": "Nested CV\n(outer K=5, inner K=3)",
            "Why": "Unbiased performance\nestimate"
        },
        {
            "Scenario": "Quick experimentation",
            "Recommended": "Shuffle Split\n(10 iterations)",
            "Why": "Fast, random subsampling,\ngood for rapid testing"
        },
    ]

    table = Table(title="Scenario-Based CV Recommendations", box=box.DOUBLE, width=120)
    table.add_column("Scenario", style="cyan", width=25)
    table.add_column("Recommended Method", style="yellow", width=30)
    table.add_column("Reasoning", style="green", width=40)

    for scenario in scenarios:
        table.add_row(
            scenario["Scenario"],
            scenario["Recommended"],
            scenario["Why"]
        )

    console.print(table)
    console.print()


def common_mistakes() -> None:
    """
    Highlight common cross-validation mistakes and how to avoid them.

    Educational section covering frequent pitfalls in CV usage.
    """
    console.print("\n[bold cyan]Common Cross-Validation Mistakes[/bold cyan]\n")

    mistakes = [
        {
            "Mistake": "Scaling before splitting",
            "Wrong": "scaler.fit(X_all)\nthen split",
            "Correct": "Split first,\nthen fit scaler on train",
            "Impact": "Optimistic bias\n(data leakage)"
        },
        {
            "Mistake": "Using K-Fold for time series",
            "Wrong": "KFold(shuffle=True)\nfor temporal data",
            "Correct": "TimeSeriesSplit",
            "Impact": "Future leakage\n(unrealistic evaluation)"
        },
        {
            "Mistake": "Not stratifying classification",
            "Wrong": "KFold for\nimbalanced classes",
            "Correct": "StratifiedKFold",
            "Impact": "Unstable folds\n(class imbalance)"
        },
        {
            "Mistake": "LOO on large datasets",
            "Wrong": "LeaveOneOut with\nN=10000 samples",
            "Correct": "K-Fold (K=5 or 10)",
            "Impact": "Extremely slow\n(10K model fits)"
        },
        {
            "Mistake": "Single CV for hyperparameter tuning",
            "Wrong": "GridSearchCV.best_score_\nas final performance",
            "Correct": "Nested CV for\nunbiased estimate",
            "Impact": "Optimistic estimate\n(selection bias)"
        },
        {
            "Mistake": "K too small (K=2)",
            "Wrong": "Only 50% data\nfor training",
            "Correct": "K=5 or K=10",
            "Impact": "High bias\n(underfits)"
        },
    ]

    table = Table(title="Don't Make These Mistakes!", box=box.DOUBLE_EDGE, width=120)
    table.add_column("Mistake", style="red", width=30)
    table.add_column("❌ Wrong", style="red", width=25)
    table.add_column("✓ Correct", style="green", width=25)
    table.add_column("Impact", style="yellow", width=25)

    for mistake in mistakes:
        table.add_row(
            mistake["Mistake"],
            mistake["Wrong"],
            mistake["Correct"],
            mistake["Impact"]
        )

    console.print(table)
    console.print()


def main() -> None:
    """Run all CV comparison demonstrations."""
    console.print(Panel.fit(
        "[bold white]Cross-Validation Strategy Comparison & Guide[/bold white]\n"
        "Comprehensive comparison and recommendations for choosing CV methods",
        border_style="bright_blue"
    ))

    compare_cv_methods_performance()
    visualize_fold_structures()
    compare_bias_variance_tradeoff()
    scenario_recommendations()
    common_mistakes()

    console.print(Panel.fit(
        "[bold green]✓ All comparisons completed![/bold green]\n"
        "Quick Reference:\n"
        "• Default choice: Stratified K-Fold (K=5) for classification\n"
        "• Time series: TimeSeriesSplit (never shuffle!)\n"
        "• Hyperparameter reporting: Nested CV\n"
        "• Small datasets: K=10 or LOO\n"
        "• Always fit preprocessing on training data only",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
