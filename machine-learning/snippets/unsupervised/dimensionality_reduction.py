"""
Dimensionality Reduction - From Scratch Implementation

Implements Principal Component Analysis (PCA) for dimensionality reduction.

Mathematical Foundation:
- PCA: Find orthogonal directions of maximum variance
- Solution: Eigenvectors of covariance matrix
- Via SVD: X = UΣV^T, principal components = columns of V

Algorithm:
1. Center data: X ← X - mean(X)
2. Compute SVD: X = UΣV^T
3. Principal components = V
4. Eigenvalues = σ²/n
5. Project: Z = XV_k (keep top k)

All implementations validated against scikit-learn.
"""

import numpy as np
from typing import Optional


class PCA:
    """
    Principal Component Analysis (PCA) via Singular Value Decomposition.

    Finds orthogonal directions of maximum variance for dimensionality reduction.

    Attributes:
        n_components (int): Number of components to keep.
        components_ (np.ndarray): Principal components, shape (n_components, n_features).
        explained_variance_ (np.ndarray): Variance explained by each component.
        explained_variance_ratio_ (np.ndarray): Fraction of variance explained.
        singular_values_ (np.ndarray): Singular values from SVD.
        mean_ (np.ndarray): Mean of training data, shape (n_features,).
        n_features_ (int): Number of features in training data.
        n_samples_ (int): Number of samples in training data.

    Mathematical Formulation:
        Maximize projected variance: max_w (1/n) Σ(w^T x_i)² s.t. ||w|| = 1

        Solution: Eigenvectors of covariance matrix C = (1/n)X^T X
        Via SVD: X = UΣV^T
            - Principal components = columns of V
            - Eigenvalues = σ²/n (where σ = singular values)

    Time Complexity: O(min(nd², n²d)) for SVD
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
    ) -> None:
        """
        Initialize PCA.

        Args:
            n_components: Number of components to keep.
                If None, keep all components.
                If int, keep first n_components.
                If float in (0, 1), select n_components such that
                explained variance ≥ n_components.
        """
        self.n_components = n_components

        # Learned parameters
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None
        self.n_samples_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Fit PCA on training data.

        Args:
            X: Training data, shape (n_samples, n_features).

        Returns:
            self: Fitted instance.

        Algorithm:
            1. Center data: X_centered = X - mean(X)
            2. Compute SVD: X_centered = U Σ V^T
            3. Principal components = rows of V^T = columns of V
            4. Explained variance = σ²/n
        """
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features

        # Step 1: Center data (CRITICAL for PCA)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Compute SVD
        # X_centered = U Σ V^T
        # U: left singular vectors, shape (n, n) or (n, min(n,d))
        # Σ: singular values, shape (min(n,d),)
        # V^T: right singular vectors, shape (d, d) or (min(n,d), d)
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Step 3: Principal components = rows of V^T
        # Vt has shape (min(n,d), d)
        # Each row is a principal component (direction in feature space)

        # Step 4: Explained variance
        # Variance along component i = σ_i² / n
        explained_variance = (s**2) / (n_samples - 1)

        # Total variance
        total_variance = np.sum(explained_variance)

        # Fraction of variance explained
        explained_variance_ratio = explained_variance / total_variance

        # Determine number of components to keep
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        elif isinstance(self.n_components, float):
            # Keep enough components to explain n_components fraction of variance
            cumsum = np.cumsum(explained_variance_ratio)
            n_components = np.searchsorted(cumsum, self.n_components) + 1
        else:
            n_components = min(self.n_components, min(n_samples, n_features))

        # Store results
        self.components_ = Vt[:n_components]  # Shape: (n_components, n_features)
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]
        self.singular_values_ = s[:n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Args:
            X: Data to transform, shape (n_samples, n_features).

        Returns:
            X_transformed: Projected data, shape (n_samples, n_components).

        Formula: Z = (X - mean) @ V_k^T
        """
        if self.components_ is None:
            raise RuntimeError("Must fit before transform")

        # Center using training mean
        X_centered = X - self.mean_

        # Project onto principal components
        # Z = X @ V_k^T (where V_k = first k principal components)
        # components_ has shape (n_components, n_features)
        # So components_.T has shape (n_features, n_components)
        X_transformed = X_centered @ self.components_.T

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and apply dimensionality reduction.

        Args:
            X: Training data, shape (n_samples, n_features).

        Returns:
            X_transformed: Projected data, shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.

        Args:
            X_transformed: Projected data, shape (n_samples, n_components).

        Returns:
            X_reconstructed: Data in original space, shape (n_samples, n_features).

        Formula: X_reconstructed = Z @ V_k + mean
        """
        if self.components_ is None:
            raise RuntimeError("Must fit before inverse_transform")

        # Project back to original space
        X_reconstructed = X_transformed @ self.components_ + self.mean_

        return X_reconstructed


if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA as SKLearnPCA
    from rich.console import Console
    from rich.table import Table
    from rich import box
    import matplotlib.pyplot as plt

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Principal Component Analysis: From Scratch Implementation")
    console.print("=" * 70 + "[/bold cyan]")

    # Example 1: PCA on Iris Dataset
    console.print("\n[bold yellow]1. PCA on Iris Dataset (4D → 2D)[/bold yellow]")
    console.print("-" * 70)

    # Load Iris data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # CRITICAL: Scale features before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Our PCA
    our_pca = PCA(n_components=2)
    X_pca_ours = our_pca.fit_transform(X_scaled)

    # Scikit-learn PCA
    sk_pca = SKLearnPCA(n_components=2)
    X_pca_sk = sk_pca.fit_transform(X_scaled)

    pca_table = Table(title="PCA Results", box=box.ROUNDED)
    pca_table.add_column("Metric", style="cyan")
    pca_table.add_column("Our Implementation", justify="right", style="green")
    pca_table.add_column("Scikit-Learn", justify="right", style="yellow")

    # Explained variance
    our_var = our_pca.explained_variance_ratio_
    sk_var = sk_pca.explained_variance_ratio_

    pca_table.add_row(
        "PC1 Variance",
        f"{our_var[0]:.4f}",
        f"{sk_var[0]:.4f}",
    )
    pca_table.add_row(
        "PC2 Variance",
        f"{our_var[1]:.4f}",
        f"{sk_var[1]:.4f}",
    )
    pca_table.add_row(
        "Total Variance",
        f"{np.sum(our_var):.4f}",
        f"{np.sum(sk_var):.4f}",
    )

    console.print(pca_table)

    console.print(
        f"\n[green]Original dimensions: {X.shape[1]}[/green]"
    )
    console.print(f"[green]Reduced dimensions: {X_pca_ours.shape[1]}[/green]")
    console.print(
        f"[green]Variance preserved: {np.sum(our_var):.1%}[/green]"
    )

    # Example 2: Choosing Number of Components
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("2. Choosing Number of Components (Variance Threshold)")
    console.print("=" * 70 + "[/bold cyan]")

    # Load digits dataset (64 dimensions)
    digits = load_digits()
    X_digits = digits.data
    X_digits_scaled = scaler.fit_transform(X_digits)

    # Fit PCA with all components
    pca_full = PCA()
    pca_full.fit(X_digits_scaled)

    # Cumulative variance explained
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

    var_table = Table(title="Variance Explained by Components", box=box.ROUNDED)
    var_table.add_column("Components", justify="center", style="cyan")
    var_table.add_column("Cumulative Variance", justify="right", style="green")

    thresholds = [10, 20, 30, 40, 50]
    for n_comp in thresholds:
        if n_comp < len(cumsum_var):
            var_table.add_row(str(n_comp), f"{cumsum_var[n_comp-1]:.2%}")

    console.print(var_table)

    # Find components for 95% variance
    n_95 = np.searchsorted(cumsum_var, 0.95) + 1
    console.print(f"\n[yellow]→ Need {n_95} components for 95% variance (out of {X_digits.shape[1]})[/yellow]")

    # Example 3: Reconstruction Error
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("3. Reconstruction Error")
    console.print("=" * 70 + "[/bold cyan]")

    recon_table = Table(title="Reconstruction Error vs Components", box=box.ROUNDED)
    recon_table.add_column("Components", justify="center", style="cyan")
    recon_table.add_column("MSE", justify="right", style="green")
    recon_table.add_column("Variance Retained", justify="right", style="yellow")

    n_components_list = [5, 10, 20, 30, 40]

    for n_comp in n_components_list:
        pca_n = PCA(n_components=n_comp)
        X_reduced = pca_n.fit_transform(X_digits_scaled)
        X_reconstructed = pca_n.inverse_transform(X_reduced)

        # Reconstruction error (MSE)
        mse = np.mean((X_digits_scaled - X_reconstructed) ** 2)

        # Variance retained
        var_retained = np.sum(pca_n.explained_variance_ratio_)

        recon_table.add_row(
            str(n_comp),
            f"{mse:.6f}",
            f"{var_retained:.2%}",
        )

    console.print(recon_table)

    console.print("\n[yellow]→ More components = lower reconstruction error[/yellow]")
    console.print("[yellow]→ Trade-off: dimensionality vs information loss[/yellow]")

    # Example 4: Scree Plot Data
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("4. Explained Variance (Scree Plot Data)")
    console.print("=" * 70 + "[/bold cyan]")

    scree_table = Table(title="Top 10 Components", box=box.ROUNDED)
    scree_table.add_column("Component", justify="center", style="cyan")
    scree_table.add_column("Variance", justify="right", style="green")
    scree_table.add_column("Cumulative", justify="right", style="yellow")

    for i in range(min(10, len(pca_full.explained_variance_ratio_))):
        scree_table.add_row(
            f"PC{i+1}",
            f"{pca_full.explained_variance_ratio_[i]:.4f}",
            f"{cumsum_var[i]:.4f}",
        )

    console.print(scree_table)

    # Example 5: Validation Against Scikit-Learn
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("5. Validation Against Scikit-Learn")
    console.print("=" * 70 + "[/bold cyan]")

    # Compare principal components
    our_pca_full = PCA(n_components=4)
    our_pca_full.fit(X_scaled)

    sk_pca_full = SKLearnPCA(n_components=4)
    sk_pca_full.fit(X_scaled)

    comparison_table = Table(title="Component Comparison", box=box.ROUNDED)
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Difference", justify="right", style="green")

    # Components might have different signs (arbitrary)
    # Compare absolute values
    comp_diff = np.mean(np.abs(np.abs(our_pca_full.components_) - np.abs(sk_pca_full.components_)))

    comparison_table.add_row("Avg Component Difference", f"{comp_diff:.10f}")
    comparison_table.add_row(
        "Variance Difference",
        f"{np.mean(np.abs(our_pca_full.explained_variance_ - sk_pca_full.explained_variance_)):.10f}",
    )

    console.print(comparison_table)

    # Key Insights
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Key Insights")
    console.print("=" * 70 + "[/bold cyan]")

    insights = [
        "• PCA finds directions of maximum variance (unsupervised)",
        "• Principal components are orthogonal (uncorrelated)",
        "• CRITICAL: Center data before PCA (subtract mean)",
        "• IMPORTANT: Scale features if different units/scales",
        "• Choose k components based on variance threshold (e.g., 95%)",
        "• Scree plot: Look for elbow in explained variance",
        "• Can reconstruct original data (with loss if k < d)",
        "• SVD is numerically stable method for computing PCA",
        "",
        "[yellow]→ Use Case: Dimensionality reduction for visualization[/yellow]",
        "[yellow]→ Use Case: Preprocessing for other ML algorithms[/yellow]",
        "[yellow]→ Use Case: Noise reduction and compression[/yellow]",
        "[yellow]→ Limitation: Only finds linear structure[/yellow]",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
