"""
Ensemble Methods Comparison - Our Implementations vs. Sklearn
==============================================================
Demonstrates usage of bagging, random forest, and gradient boosting,
comparing our from-scratch implementations with sklearn's production versions.

Purpose:
    - Show API usage for each ensemble method
    - Compare performance and predictions
    - Demonstrate key features (OOB scores, feature importance, etc.)
    - Serve as quick reference for production sklearn usage

Key Takeaways:
    1. Bagging: Reduces variance through bootstrap aggregation
    2. Random Forest: Adds feature randomness to further decorrelate trees
    3. Gradient Boosting: Sequential learning, fits residuals

When to Use Each:
    - Bagging: When you have a high-variance model (deep trees)
    - Random Forest: Default choice for most classification tasks
    - Gradient Boosting: When you need maximum accuracy (beware overfitting)
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Rich imports for better formatting
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.progress import track

console = Console()

# Import our implementations
try:
    # Try relative imports (when run as module)
    from .bagging import BaggingClassifier
    from .random_forest import RandomForestClassifier
    from .gradient_boosting import GradientBoostingClassifier
    from ..decision_trees import DecisionTreeClassifier
except ImportError:
    # Fall back to direct imports (when run as script)
    import sys
    from pathlib import Path

    # Add parent directory to path for decision_trees
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

    # Now import
    from bagging import BaggingClassifier
    from random_forest import RandomForestClassifier
    from gradient_boosting import GradientBoostingClassifier
    from decision_trees import DecisionTreeClassifier

# Import sklearn versions for comparison
from sklearn.ensemble import (
    BaggingClassifier as SklearnBagging,
    RandomForestClassifier as SklearnRF,
    GradientBoostingClassifier as SklearnGB,
)
from sklearn.tree import DecisionTreeClassifier as SklearnTree


def load_dataset():
    """
    Load and prepare dataset for comparison.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Load breast cancer dataset (binary classification)
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create dataset info table
    table = Table(title="Dataset Information", box=box.ROUNDED, show_header=False)
    table.add_column("Property", style="cyan", width=20)
    table.add_column("Value", style="magenta")

    table.add_row("Dataset", "Breast Cancer")
    table.add_row("Task", "Binary Classification")
    table.add_row("Training samples", str(X_train.shape[0]))
    table.add_row("Test samples", str(X_test.shape[0]))
    table.add_row("Features", str(X_train.shape[1]))
    table.add_row("Classes", str(list(np.unique(y))))

    console.print(table)
    console.print()

    return X_train, X_test, y_train, y_test


def compare_bagging(X_train, X_test, y_train, y_test):
    """
    Compare our Bagging implementation with sklearn.

    Key Parameters:
        - n_estimators: Number of base estimators
        - max_samples: Fraction of samples to draw
        - oob_score: Whether to compute out-of-bag score
    """
    console.print(Panel.fit(
        "[bold cyan]BAGGING CLASSIFIER COMPARISON[/bold cyan]",
        border_style="cyan"
    ))

    # Our implementation
    console.print("\n[bold yellow]Training Our Implementation...[/bold yellow]")
    base_tree = DecisionTreeClassifier(max_depth=10)
    our_bag = BaggingClassifier(
        base_estimator=base_tree,
        n_estimators=50,
        max_samples=1.0,
        random_state=42,
        oob_score=True,
    )
    our_bag.fit(X_train, y_train)

    our_pred = our_bag.predict(X_test)
    our_accuracy = accuracy_score(y_test, our_pred)

    # Sklearn implementation
    console.print("[bold yellow]Training Sklearn Implementation...[/bold yellow]")
    sk_tree = SklearnTree(max_depth=10, random_state=42)
    sk_bag = SklearnBagging(
        estimator=sk_tree,
        n_estimators=50,
        max_samples=1.0,
        random_state=42,
        oob_score=True,
    )
    sk_bag.fit(X_train, y_train)

    sk_pred = sk_bag.predict(X_test)
    sk_accuracy = accuracy_score(y_test, sk_pred)

    # Create comparison table
    table = Table(title="Bagging Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Our Impl.", justify="right", style="green")
    table.add_column("Sklearn", justify="right", style="blue")
    table.add_column("Difference", justify="right", style="magenta")

    acc_diff = abs(our_accuracy - sk_accuracy)
    oob_diff = abs(our_bag.oob_score_ - sk_bag.oob_score_)

    table.add_row(
        "Test Accuracy",
        f"{our_accuracy:.4f}",
        f"{sk_accuracy:.4f}",
        f"{acc_diff:.4f}"
    )
    table.add_row(
        "OOB Score",
        f"{our_bag.oob_score_:.4f}",
        f"{sk_bag.oob_score_:.4f}",
        f"{oob_diff:.4f}"
    )
    table.add_row(
        "Number of Estimators",
        str(len(our_bag.estimators_)),
        str(len(sk_bag.estimators_)),
        "0"
    )

    console.print(table)
    console.print()


def compare_random_forest(X_train, X_test, y_train, y_test):
    """
    Compare our Random Forest implementation with sklearn.

    Key Parameters:
        - n_estimators: Number of trees
        - max_features: Features to consider at each split ('sqrt', 'log2', int, float)
        - max_depth: Maximum tree depth
        - oob_score: Out-of-bag score
    """
    console.print(Panel.fit(
        "[bold cyan]RANDOM FOREST CLASSIFIER COMPARISON[/bold cyan]",
        border_style="cyan"
    ))

    # Our implementation
    console.print("\n[bold yellow]Training Our Implementation...[/bold yellow]")
    our_rf = RandomForestClassifier(
        n_estimators=100,
        max_features='sqrt',
        max_depth=10,
        random_state=42,
        oob_score=True,
    )
    our_rf.fit(X_train, y_train)

    our_pred = our_rf.predict(X_test)
    our_accuracy = accuracy_score(y_test, our_pred)
    our_proba = our_rf.predict_proba(X_test)

    # Sklearn implementation
    console.print("[bold yellow]Training Sklearn Implementation...[/bold yellow]")
    sk_rf = SklearnRF(
        n_estimators=100,
        max_features='sqrt',
        max_depth=10,
        random_state=42,
        oob_score=True,
    )
    sk_rf.fit(X_train, y_train)

    sk_pred = sk_rf.predict(X_test)
    sk_accuracy = accuracy_score(y_test, sk_pred)
    sk_proba = sk_rf.predict_proba(X_test)

    # Create comparison table
    table = Table(title="Random Forest Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Our Impl.", justify="right", style="green")
    table.add_column("Sklearn", justify="right", style="blue")
    table.add_column("Difference", justify="right", style="magenta")

    acc_diff = abs(our_accuracy - sk_accuracy)
    oob_diff = abs(our_rf.oob_score_ - sk_rf.oob_score_)
    prob_correlation = np.corrcoef(our_proba[:, 1], sk_proba[:, 1])[0, 1]

    n_features_sqrt = int(np.sqrt(X_train.shape[1]))

    table.add_row(
        "Test Accuracy",
        f"{our_accuracy:.4f}",
        f"{sk_accuracy:.4f}",
        f"{acc_diff:.4f}"
    )
    table.add_row(
        "OOB Score",
        f"{our_rf.oob_score_:.4f}",
        f"{sk_rf.oob_score_:.4f}",
        f"{oob_diff:.4f}"
    )
    table.add_row(
        "Number of Trees",
        str(len(our_rf.estimators_)),
        str(len(sk_rf.estimators_)),
        "0"
    )
    table.add_row(
        "Max Features per Split",
        str(our_rf._calculate_max_features(X_train.shape[1])),
        str(n_features_sqrt),
        "0"
    )
    table.add_row(
        "Probability Correlation",
        "-",
        "-",
        f"{prob_correlation:.4f}"
    )

    console.print(table)
    console.print()


def compare_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    Compare our Gradient Boosting implementation with sklearn.

    Key Parameters:
        - n_estimators: Number of boosting stages
        - learning_rate: Shrinks contribution of each tree
        - max_depth: Maximum tree depth (typically 3-5 for boosting)
    """
    console.print(Panel.fit(
        "[bold cyan]GRADIENT BOOSTING CLASSIFIER COMPARISON[/bold cyan]",
        border_style="cyan"
    ))

    # Our implementation
    console.print("\n[bold yellow]Training Our Implementation...[/bold yellow]")
    our_gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    our_gb.fit(X_train, y_train)

    our_pred = our_gb.predict(X_test)
    our_accuracy = accuracy_score(y_test, our_pred)
    our_proba = our_gb.predict_proba(X_test)

    # Sklearn implementation
    console.print("[bold yellow]Training Sklearn Implementation...[/bold yellow]")
    sk_gb = SklearnGB(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    sk_gb.fit(X_train, y_train)

    sk_pred = sk_gb.predict(X_test)
    sk_accuracy = accuracy_score(y_test, sk_pred)
    sk_proba = sk_gb.predict_proba(X_test)

    # Create comparison table
    table = Table(title="Gradient Boosting Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Our Impl.", justify="right", style="green")
    table.add_column("Sklearn", justify="right", style="blue")
    table.add_column("Difference", justify="right", style="magenta")

    acc_diff = abs(our_accuracy - sk_accuracy)
    prob_correlation = np.corrcoef(our_proba[:, 1], sk_proba[:, 1])[0, 1]

    table.add_row(
        "Test Accuracy",
        f"{our_accuracy:.4f}",
        f"{sk_accuracy:.4f}",
        f"{acc_diff:.4f}"
    )
    table.add_row(
        "Number of Trees",
        str(len(our_gb.estimators_)),
        str(len(sk_gb.estimators_)),
        "0"
    )
    table.add_row(
        "Learning Rate",
        f"{our_gb.learning_rate:.2f}",
        f"{sk_gb.learning_rate:.2f}",
        "0.00"
    )
    table.add_row(
        "Max Depth",
        str(our_gb.max_depth),
        str(sk_gb.max_depth),
        "0"
    )
    table.add_row(
        "Probability Correlation",
        "-",
        "-",
        f"{prob_correlation:.4f}"
    )

    console.print(table)
    console.print()


def demonstrate_key_features(X_train, X_test, y_train, y_test):
    """
    Demonstrate key features unique to each ensemble method.
    """
    console.print(Panel.fit(
        "[bold cyan]KEY FEATURES DEMONSTRATION[/bold cyan]",
        border_style="cyan"
    ))

    # Bagging: OOB Score as validation metric
    console.print("\n[bold green]Bagging: Out-of-Bag Scoring[/bold green]")
    console.print("[dim]OOB score provides validation without separate validation set[/dim]")

    base_tree = DecisionTreeClassifier(max_depth=8)
    bag = BaggingClassifier(
        base_estimator=base_tree,
        n_estimators=50,
        random_state=42,
        oob_score=True,
    )
    bag.fit(X_train, y_train)
    test_score = bag.score(X_test, y_test)

    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right", style="yellow")
    table.add_row("OOB Score", f"{bag.oob_score_:.4f}")
    table.add_row("Test Score", f"{test_score:.4f}")
    table.add_row("Difference", f"{abs(bag.oob_score_ - test_score):.4f}")
    console.print(table)

    # Random Forest: Feature randomness effect
    console.print("\n[bold green]Random Forest: Feature Randomness[/bold green]")
    console.print("[dim]Comparing different max_features settings[/dim]")

    table = Table(box=box.ROUNDED)
    table.add_column("max_features", style="cyan")
    table.add_column("Features/Split", justify="right", style="yellow")
    table.add_column("Accuracy", justify="right", style="green")

    for max_feat in ['sqrt', 'log2', None]:
        rf = RandomForestClassifier(
            n_estimators=50,
            max_features=max_feat,
            max_depth=8,
            random_state=42,
        )
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        n_features = rf._calculate_max_features(X_train.shape[1])
        table.add_row(
            str(max_feat),
            str(n_features),
            f"{score:.4f}"
        )

    console.print(table)

    # Gradient Boosting: Staged predictions (learning progression)
    console.print("\n[bold green]Gradient Boosting: Staged Predictions[/bold green]")
    console.print("[dim]Accuracy at different boosting stages[/dim]")

    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    gb.fit(X_train, y_train)

    staged_scores = gb.staged_score(X_test, y_test)

    # Show scores at key stages
    table = Table(box=box.ROUNDED)
    table.add_column("Stage (Trees)", justify="right", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")

    stages = [10, 25, 50, 75, 100]
    for stage in stages:
        score = staged_scores[stage - 1]
        table.add_row(str(stage), f"{score:.4f}")

    console.print(table)
    console.print()


def show_usage_guidelines():
    """
    Print practical guidelines for choosing ensemble methods.
    """
    console.print(Panel.fit(
        "[bold cyan]PRACTICAL USAGE GUIDELINES[/bold cyan]",
        border_style="cyan"
    ))

    # When to use each method
    console.print("\n[bold yellow]WHEN TO USE EACH METHOD:[/bold yellow]\n")

    bagging_use = """[green]✓[/green] You have high-variance model (e.g., deep decision trees)
[green]✓[/green] Need quick improvement over single model
[green]✓[/green] Want easy parallelization
[green]✓[/green] Dataset has high variance or noise
[dim]Example: Bagging deep trees to reduce overfitting[/dim]"""

    rf_use = """[green]✓[/green] Default choice for most classification/regression tasks
[green]✓[/green] Need robust, general-purpose ensemble
[green]✓[/green] Want built-in feature importance
[green]✓[/green] Have many features (feature randomness helps)
[dim]Example: Most real-world tabular data problems[/dim]"""

    gb_use = """[green]✓[/green] Need maximum predictive accuracy
[green]✓[/green] Willing to tune hyperparameters carefully
[green]✓[/green] Have enough data to prevent overfitting
[green]✓[/green] Can afford longer training time (sequential)
[dim]Example: Kaggle competitions, when accuracy is critical[/dim]"""

    console.print(Panel(bagging_use, title="[bold]Bagging[/bold]", border_style="blue"))
    console.print(Panel(rf_use, title="[bold]Random Forest[/bold]", border_style="green"))
    console.print(Panel(gb_use, title="[bold]Gradient Boosting[/bold]", border_style="magenta"))

    # Hyperparameters table
    console.print("\n[bold yellow]KEY HYPERPARAMETERS:[/bold yellow]\n")

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Method", style="cyan", width=18)
    table.add_column("Parameter", style="yellow", width=20)
    table.add_column("Recommended Range", style="green")

    # Bagging parameters
    table.add_row("Bagging", "n_estimators", "10-500 (diminishing returns)")
    table.add_row("", "max_samples", "0.5-1.0 (1.0 is standard)")

    # Random Forest parameters
    table.add_row("Random Forest", "n_estimators", "100-500 (more is better)")
    table.add_row("", "max_features", "'sqrt' or 'log2'")
    table.add_row("", "max_depth", "None or tune to prevent overfitting")

    # Gradient Boosting parameters
    table.add_row("Gradient Boosting", "n_estimators", "100-1000 (use early stopping)")
    table.add_row("", "learning_rate", "0.01-0.3 (lower = better)")
    table.add_row("", "max_depth", "3-5 (shallow trees)")
    table.add_row("", "subsample", "0.5-1.0 (stochastic GB)")

    console.print(table)

    # Production recommendations
    console.print("\n[bold yellow]PRODUCTION RECOMMENDATIONS:[/bold yellow]\n")

    prod_info = """For production, use optimized libraries:
  • [bold cyan]XGBoost[/bold cyan]: Highly optimized GB with regularization
  • [bold cyan]LightGBM[/bold cyan]: Fast GB with novel algorithms (GOSS, EFB)
  • [bold cyan]CatBoost[/bold cyan]: GB with built-in categorical feature handling
  • [bold cyan]sklearn.ensemble[/bold cyan]: Well-tested, good defaults

[dim italic]Our implementations are educational - they teach HOW these algorithms work,
but production libraries have years of optimization and edge-case handling.[/dim italic]"""

    console.print(Panel(prod_info, border_style="yellow"))
    console.print()


def main():
    """
    Run all comparisons and demonstrations.
    """
    # Main header
    console.print()
    console.print(Panel.fit(
        "[bold white]ENSEMBLE METHODS: FROM-SCRATCH vs SKLEARN COMPARISON[/bold white]\n"
        "[dim]Comparing Bagging, Random Forest, and Gradient Boosting implementations[/dim]",
        border_style="bold cyan",
        padding=(1, 2)
    ))
    console.print()

    # Load data
    X_train, X_test, y_train, y_test = load_dataset()

    # Compare each ensemble method
    compare_bagging(X_train, X_test, y_train, y_test)
    compare_random_forest(X_train, X_test, y_train, y_test)
    compare_gradient_boosting(X_train, X_test, y_train, y_test)

    # Demonstrate key features
    demonstrate_key_features(X_train, X_test, y_train, y_test)

    # Show usage guidelines
    show_usage_guidelines()

    # Completion message
    console.print(Panel.fit(
        "[bold green]✓ COMPARISON COMPLETE[/bold green]",
        border_style="green"
    ))
    console.print()


if __name__ == "__main__":
    main()
