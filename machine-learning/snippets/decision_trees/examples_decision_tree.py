"""
Decision Tree Examples - Usage demonstrations
=============================================
Shows how to use DecisionTreeClassifier and DecisionTreeRegressor.

HOW TO RUN:
    From project root:
        python -m machine-learning.snippets.decision_trees.examples_decision_tree

    Alternative (using absolute paths):
        python /Users/hanyuwu/Study/artifact-foundry/machine-learning/snippets/decision_trees/examples_decision_tree.py
        Note: This only works if dependencies (numpy, rich) are in your path.
"""

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich import box
from .decision_tree_classifier import DecisionTreeClassifier
from .decision_tree_regressor import DecisionTreeRegressor

# Initialize rich console
console = Console()


def print_separator(title: str):
    """Print a formatted section separator using rich."""
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")
    console.print()


def build_rich_tree(node, tree_obj=None, label="Root", depth=0):
    """
    Recursively build a rich Tree object for visualization.

    Args:
        node: Current node to build
        tree_obj: Rich Tree object (creates new if None)
        label: Label for this node
        depth: Current depth

    Returns:
        Rich Tree object
    """

    # Format value based on type
    def format_value(val):
        return f"{val:.2f}" if isinstance(val, float) else str(val)

    # Format node label
    if node.is_leaf():
        node_label = (
            f"[bold green]{label}[/bold green] → "
            f"[yellow]predict {format_value(node.value)}[/yellow] "
            f"[dim](samples={node.n_samples}, impurity={node.impurity:.4f})[/dim]"
        )
    else:
        node_label = (
            f"[bold blue]{label}[/bold blue] "
            f"[cyan]X[{node.feature_idx}] ≤ {node.threshold:.2f}[/cyan] "
            f"[dim](samples={node.n_samples}, impurity={node.impurity:.4f})[/dim]"
        )

    # Create tree or update existing label
    if tree_obj is None:
        tree_obj = Tree(node_label)
    else:
        tree_obj.label = node_label

    # Add children if not leaf
    if not node.is_leaf():
        if node.left:
            build_rich_tree(node.left, tree_obj.add("Left"), "Left", depth + 1)
        if node.right:
            build_rich_tree(node.right, tree_obj.add("Right"), "Right", depth + 1)

    return tree_obj


def example_classification():
    """Demonstrate classification tree."""
    print_separator("CLASSIFICATION EXAMPLE")

    # Create synthetic dataset (XOR-like problem)
    console.print("[bold]📊 Dataset[/bold]")
    X_train = np.array(
        [[2, 3], [1, 1], [2, 2], [3, 3], [5, 5], [5, 4], [6, 6], [7, 5], [1, 2], [2, 1]]
    )
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0])

    info_table = Table(show_header=False, box=box.SIMPLE)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="yellow")
    info_table.add_row("Features shape", str(X_train.shape))
    info_table.add_row(
        "Target distribution", str(dict(zip(*np.unique(y_train, return_counts=True))))
    )
    console.print(info_table)

    # Train with Gini criterion
    console.print(
        "\n[bold]🌳 Training DecisionTreeClassifier[/bold] [dim](criterion='gini')[/dim]"
    )
    clf_gini = DecisionTreeClassifier(criterion="gini", max_depth=3)
    clf_gini.fit(X_train, y_train)

    # Make predictions
    X_test = np.array([[1, 2], [6, 5], [3, 2]])
    predictions_gini = clf_gini.predict(X_test)

    console.print("\n[bold]🔮 Predictions[/bold]")
    pred_table = Table(show_header=True, box=box.ROUNDED)
    pred_table.add_column("Sample", style="cyan", justify="left")
    pred_table.add_column("Prediction", style="yellow", justify="center")
    for x, pred in zip(X_test, predictions_gini):
        sample_str = f"[{x[0]}, {x[1]}]"
        pred_table.add_row(sample_str, f"Class {pred}")
    console.print(pred_table)

    # Training accuracy
    train_pred = clf_gini.predict(X_train)
    train_acc = np.mean(train_pred == y_train)
    console.print(
        f"\n[bold]📈 Training accuracy:[/bold] [green]{train_acc:.2%}[/green]"
    )

    # Compare with Entropy
    console.print("\n[bold]🌳 Training with Entropy for comparison...[/bold]")
    clf_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf_entropy.fit(X_train, y_train)
    predictions_entropy = clf_entropy.predict(X_test)

    console.print("\n[bold]🔍 Criterion comparison[/bold]")
    comp_table = Table(show_header=True, box=box.ROUNDED)
    comp_table.add_column("Criterion", style="cyan")
    comp_table.add_column("Predictions", style="yellow")
    comp_table.add_row("Gini", str(predictions_gini.tolist()))
    comp_table.add_row("Entropy", str(predictions_entropy.tolist()))
    console.print(comp_table)

    # Show tree structure
    console.print("\n[bold]🌲 Tree Structure (Gini)[/bold]")
    tree_viz = build_rich_tree(clf_gini.root)
    console.print(tree_viz)


def example_regression():
    """Demonstrate regression tree."""
    print_separator("REGRESSION EXAMPLE")

    # Create synthetic dataset (step function)
    console.print("[bold]📊 Dataset[/bold]")
    X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    # Step function: y ≈ 5 for x <= 5, y ≈ 10 for x > 5
    y_train = np.array([5.1, 4.9, 5.2, 5.0, 4.8, 10.1, 9.9, 10.2, 10.0, 9.8])

    info_table = Table(show_header=False, box=box.SIMPLE)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="yellow")
    info_table.add_row("Features shape", str(X_train.shape))
    info_table.add_row("Target range", f"[{y_train.min():.2f}, {y_train.max():.2f}]")
    console.print(info_table)

    # Train with MSE criterion
    console.print(
        "\n[bold]🌳 Training DecisionTreeRegressor[/bold] [dim](criterion='mse')[/dim]"
    )
    reg_mse = DecisionTreeRegressor(criterion="mse", max_depth=3)
    reg_mse.fit(X_train, y_train)

    # Make predictions
    X_test = np.array([[3], [7], [11]])
    predictions_mse = reg_mse.predict(X_test)

    console.print("\n[bold]🔮 Predictions[/bold]")
    pred_table = Table(show_header=True, box=box.ROUNDED)
    pred_table.add_column("X", style="cyan", justify="right")
    pred_table.add_column("Prediction", style="yellow", justify="right")
    for x, pred in zip(X_test, predictions_mse):
        pred_table.add_row(f"{x[0]:.0f}", f"{pred:.2f}")
    console.print(pred_table)

    # R² score
    r2 = reg_mse.score(X_train, y_train)
    console.print(f"\n[bold]📈 Training R² score:[/bold] [green]{r2:.4f}[/green]")

    # Compare with MAE
    console.print("\n[bold]🌳 Training with MAE for comparison...[/bold]")
    reg_mae = DecisionTreeRegressor(criterion="mae", max_depth=3)
    reg_mae.fit(X_train, y_train)
    predictions_mae = reg_mae.predict(X_test)

    console.print("\n[bold]🔍 Criterion comparison[/bold]")
    comp_table = Table(show_header=True, box=box.ROUNDED)
    comp_table.add_column("Criterion", style="cyan")
    comp_table.add_column("Predictions", style="yellow")
    comp_table.add_row("MSE", str([f"{p:.2f}" for p in predictions_mse]))
    comp_table.add_row("MAE", str([f"{p:.2f}" for p in predictions_mae]))
    console.print(comp_table)

    # Show tree structure
    console.print("\n[bold]🌲 Tree Structure (MSE)[/bold]")
    tree_viz = build_rich_tree(reg_mse.root)
    console.print(tree_viz)


def example_overfitting():
    """Demonstrate overfitting with different max_depth values."""
    print_separator("OVERFITTING DEMONSTRATION")

    # Small dataset
    np.random.seed(42)
    X_train = np.random.rand(20, 2) * 10
    y_train = (X_train[:, 0] + X_train[:, 1] > 10).astype(int)

    console.print("[bold]📊 Training set:[/bold] 20 samples, 2 features")
    console.print(
        f"[dim]Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}[/dim]"
    )

    console.print("\n[bold]🔬 Training trees with different max_depth values[/bold]")

    results_table = Table(show_header=True, box=box.ROUNDED)
    results_table.add_column("max_depth", style="cyan", justify="center")
    results_table.add_column("Train Accuracy", style="yellow", justify="center")
    results_table.add_column("Tree Nodes (approx)", style="magenta", justify="center")

    for depth in [1, 2, 3, 5, 10, None]:
        clf = DecisionTreeClassifier(criterion="gini", max_depth=depth)
        clf.fit(X_train, y_train)
        train_acc = np.mean(clf.predict(X_train) == y_train)

        # Count nodes (rough estimate based on perfect binary tree)
        if depth is None:
            nodes = "many"
            depth_str = "unlimited"
        else:
            nodes = f"{2**(depth+1) - 1} max"
            depth_str = str(depth)

        # Color code accuracy
        if train_acc == 1.0:
            acc_str = f"[red]{train_acc:.1%}[/red]"
        elif train_acc > 0.9:
            acc_str = f"[yellow]{train_acc:.1%}[/yellow]"
        else:
            acc_str = f"[green]{train_acc:.1%}[/green]"

        results_table.add_row(depth_str, acc_str, nodes)

    console.print(results_table)

    warning_panel = Panel(
        "[yellow]⚠️  Warning:[/yellow] Higher accuracy on small dataset often means overfitting!\n"
        "Use validation set or cross-validation to tune hyperparameters.",
        title="[bold red]Overfitting Alert[/bold red]",
        border_style="red",
        box=box.ROUNDED,
    )
    console.print("\n")
    console.print(warning_panel)


def example_robustness_to_outliers():
    """Demonstrate MAE robustness to outliers."""
    print_separator("OUTLIER ROBUSTNESS")

    console.print("[bold]📊 Dataset with outlier[/bold]")

    # Normal data
    X_normal = np.array([[1], [2], [3], [4], [5]])
    y_normal = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Data with outlier
    X_outlier = np.array([[1], [2], [3], [4], [5]])
    y_outlier = np.array([10.0, 20.0, 30.0, 40.0, 500.0])  # 500.0 is outlier

    data_table = Table(show_header=True, box=box.SIMPLE)
    data_table.add_column("Dataset", style="cyan")
    data_table.add_column("Values", style="yellow")
    data_table.add_row("Normal data", str(y_normal.tolist()))
    data_table.add_row("With outlier", str(y_outlier.tolist()))
    console.print(data_table)

    # Train both criteria on outlier data
    reg_mse = DecisionTreeRegressor(criterion="mse", max_depth=2)
    reg_mae = DecisionTreeRegressor(criterion="mae", max_depth=2)

    reg_mse.fit(X_outlier, y_outlier)
    reg_mae.fit(X_outlier, y_outlier)

    # Test prediction
    X_test = np.array([[3]])
    pred_mse = reg_mse.predict(X_test)[0]
    pred_mae = reg_mae.predict(X_test)[0]

    console.print("\n[bold]🔮 Prediction for X = 3[/bold]")
    pred_table = Table(show_header=True, box=box.ROUNDED)
    pred_table.add_column("Method", style="cyan")
    pred_table.add_column("Prediction", style="yellow", justify="right")
    pred_table.add_row("MSE criterion", f"{pred_mse:.2f}")
    pred_table.add_row("MAE criterion", f"[green]{pred_mae:.2f}[/green]")
    pred_table.add_row("True value", "[bold]30.00[/bold]")
    console.print(pred_table)

    insight_panel = Panel(
        "[cyan]💡 Key Insight:[/cyan]\n\n"
        "• [green]MAE uses median[/green] → more robust to outliers\n"
        "• [yellow]MSE uses mean[/yellow] → influenced by extreme values",
        title="[bold]Understanding Robustness[/bold]",
        border_style="blue",
        box=box.ROUNDED,
    )
    console.print("\n")
    console.print(insight_panel)


def example_tree_visualization():
    """Show how to extract tree information."""
    print_separator("TREE INSPECTION")

    # Simple dataset
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([0, 0, 0, 1, 1, 1])

    # Train shallow tree
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    console.print("[bold]🔍 Tree Metadata[/bold]")
    meta_table = Table(show_header=False, box=box.SIMPLE)
    meta_table.add_column("Property", style="cyan")
    meta_table.add_column("Value", style="yellow")
    meta_table.add_row("Number of features", str(clf.n_features_))
    meta_table.add_row("Number of samples", str(clf.n_samples_))
    meta_table.add_row("Number of classes", str(clf.n_classes_))
    console.print(meta_table)

    console.print("\n[bold]📊 Root Node Info[/bold]")
    root = clf.root
    root_table = Table(show_header=False, box=box.SIMPLE)
    root_table.add_column("Property", style="cyan")
    root_table.add_column("Value", style="yellow")
    root_table.add_row("Split feature", f"X[{root.feature_idx}]")
    root_table.add_row("Split threshold", f"{root.threshold:.2f}")
    root_table.add_row("Node impurity", f"{root.impurity:.4f}")
    root_table.add_row("Samples at node", str(root.n_samples))
    root_table.add_row("Is leaf?", str(root.is_leaf()))
    console.print(root_table)

    if root.left.is_leaf():
        console.print(
            f"\n[dim]  Left child:[/dim]  [green]Leaf predicting class {root.left.value}[/green]"
        )
    if root.right.is_leaf():
        console.print(
            f"[dim]  Right child:[/dim] [green]Leaf predicting class {root.right.value}[/green]"
        )


def main():
    """Run all examples."""
    console.print()
    console.print("🌳 " * 35, style="bold green")
    console.print()
    title = Text("DECISION TREE EXAMPLES", style="bold cyan", justify="center")
    console.print(Panel(title, border_style="cyan", box=box.DOUBLE))
    console.print("🌳 " * 35, style="bold green")

    example_classification()
    example_regression()
    example_overfitting()
    example_robustness_to_outliers()
    example_tree_visualization()

    print_separator("KEY TAKEAWAYS")

    takeaways = [
        (
            "[bold cyan]1. Classification Trees[/bold cyan]",
            [
                "• Use Gini or Entropy to measure impurity",
                "• Predict most common class in each leaf",
                "• Gini is faster, Entropy slightly more balanced",
            ],
        ),
        (
            "[bold cyan]2. Regression Trees[/bold cyan]",
            [
                "• Use MSE/Variance or MAE to measure impurity",
                "• Predict mean (MSE) or median (MAE) in each leaf",
                "• MAE is more robust to outliers",
            ],
        ),
        (
            "[bold cyan]3. Hyperparameters[/bold cyan]",
            [
                "• max_depth: Most important, controls overfitting",
                "• min_samples_split: Prevents splitting small nodes",
                "• min_samples_leaf: Ensures minimum samples per leaf",
            ],
        ),
        (
            "[bold cyan]4. Overfitting[/bold cyan]",
            [
                "• Deep trees fit training data perfectly but generalize poorly",
                "• Use validation set to tune hyperparameters",
                "• Consider ensemble methods (Random Forest, etc.)",
            ],
        ),
        (
            "[bold cyan]5. Implementation[/bold cyan]",
            [
                "• CART algorithm uses greedy, recursive partitioning",
                "• TreeMetrics provides split quality calculations",
                "• Modular design separates concerns cleanly",
            ],
        ),
    ]

    for title, points in takeaways:
        console.print(f"\n{title}")
        for point in points:
            console.print(f"  {point}")

    console.print()
    footer_panel = Panel(
        "📚 See [cyan]machine-learning/notes/decision-trees.md[/cyan] for detailed notes",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(footer_panel)
    console.print()


if __name__ == "__main__":
    main()
