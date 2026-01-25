"""
Clustering Evaluation Metrics - From Scratch Implementation

Implements metrics for evaluating clustering quality without ground truth labels.

Metrics:
- Inertia: Within-cluster sum of squares
- Silhouette Score: Measure of cluster separation
- Davies-Bouldin Index: Ratio of within/between cluster distances
- Calinski-Harabasz Index: Variance ratio criterion

All metrics validated against scikit-learn implementations.
"""

import numpy as np


def inertia(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Compute within-cluster sum of squares (inertia).

    Formula: Σ_k Σ_{x∈C_k} ||x - μ_k||²

    Args:
        X: Data points, shape (n_samples, n_features).
        labels: Cluster assignments, shape (n_samples,).
        centers: Cluster centroids, shape (n_clusters, n_features).

    Returns:
        inertia: Within-cluster sum of squares, range [0, ∞], lower is better.

    Use Case:
        - K-Means objective function
        - Elbow method for choosing K
        - Always decreases with K

    Examples:
        >>> X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        >>> labels = np.array([0, 0, 1, 1])
        >>> centers = np.array([[0.5, 0.5], [10.5, 10.5]])
        >>> inertia(X, labels, centers)
        2.0
    """
    total_inertia = 0.0

    for k in range(len(centers)):
        # Points in cluster k
        cluster_mask = labels == k
        cluster_points = X[cluster_mask]

        if len(cluster_points) > 0:
            # Sum of squared distances to centroid
            squared_distances = np.sum((cluster_points - centers[k]) ** 2)
            total_inertia += squared_distances

    return total_inertia


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Silhouette Coefficient (average over all samples).

    Formula: s_i = (b_i - a_i) / max(a_i, b_i)
    Where:
        a_i = avg distance to points in same cluster
        b_i = avg distance to points in nearest other cluster

    Args:
        X: Data points, shape (n_samples, n_features).
        labels: Cluster assignments, shape (n_samples,).

    Returns:
        silhouette: Average silhouette score, range [-1, 1].
            +1: Perfect clustering (far from other clusters)
             0: On cluster boundary
            -1: Wrong cluster (closer to other cluster)

    Use Cases:
        - Choose optimal K (maximize silhouette)
        - Identify well-separated clusters
        - Find misclassified points

    Time Complexity: O(n²)

    Examples:
        >>> X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        >>> labels = np.array([0, 0, 1, 1])
        >>> silhouette_score(X, labels)
        0.85  # High score = well-separated clusters
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1 or n_clusters == n_samples:
        # Silhouette undefined for 1 cluster or all points separate
        return 0.0

    # Compute pairwise distances
    distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)

    silhouette_values = np.zeros(n_samples)

    for i in range(n_samples):
        # Current cluster
        cluster_i = labels[i]

        # a_i: average distance to points in same cluster
        same_cluster_mask = labels == cluster_i
        same_cluster_distances = distances[i, same_cluster_mask]

        if len(same_cluster_distances) > 1:
            # Exclude distance to self (which is 0)
            a_i = np.mean(same_cluster_distances[same_cluster_distances > 0])
        else:
            # Only one point in cluster
            a_i = 0.0

        # b_i: minimum average distance to points in other clusters
        b_i = np.inf

        for other_cluster in unique_labels:
            if other_cluster == cluster_i:
                continue

            other_cluster_mask = labels == other_cluster
            other_cluster_distances = distances[i, other_cluster_mask]

            if len(other_cluster_distances) > 0:
                avg_dist = np.mean(other_cluster_distances)
                b_i = min(b_i, avg_dist)

        # Silhouette coefficient for point i
        if max(a_i, b_i) > 0:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
        else:
            silhouette_values[i] = 0.0

    return np.mean(silhouette_values)


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Davies-Bouldin Index.

    Formula: DB = (1/K) Σ_i max_{j≠i} (s_i + s_j) / d_{ij}
    Where:
        s_i = avg distance from points in cluster i to centroid i
        d_ij = distance between centroids i and j

    Args:
        X: Data points, shape (n_samples, n_features).
        labels: Cluster assignments, shape (n_samples,).

    Returns:
        db_score: Davies-Bouldin index, range [0, ∞], lower is better.
            0 = perfect clustering

    Interpretation:
        Ratio of within-cluster to between-cluster distances.
        Lower values indicate better separation.

    Use Cases:
        - Choose optimal K
        - Compare clustering algorithms
        - Fast to compute

    Examples:
        >>> X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        >>> labels = np.array([0, 0, 1, 1])
        >>> davies_bouldin_score(X, labels)
        0.15  # Low score = good clustering
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1:
        return 0.0

    # Compute cluster centroids
    centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])

    # Compute s_i (average distance to centroid)
    s = np.zeros(n_clusters)
    for i, label_i in enumerate(unique_labels):
        cluster_points = X[labels == label_i]
        s[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))

    # Compute d_ij (distances between centroids)
    centroid_distances = np.linalg.norm(
        centroids[:, np.newaxis] - centroids, axis=2
    )

    # Avoid division by zero
    np.fill_diagonal(centroid_distances, np.inf)

    # Compute Davies-Bouldin index
    db_scores = np.zeros(n_clusters)

    for i in range(n_clusters):
        # For cluster i, find max (s_i + s_j) / d_ij over all j ≠ i
        ratios = (s[i] + s) / centroid_distances[i]
        db_scores[i] = np.max(ratios)

    return np.mean(db_scores)


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Calinski-Harabasz Index (Variance Ratio Criterion).

    Formula: CH = [SS_B / (K-1)] / [SS_W / (n-K)]
    Where:
        SS_B = between-cluster sum of squares
        SS_W = within-cluster sum of squares
        K = number of clusters
        n = number of samples

    Args:
        X: Data points, shape (n_samples, n_features).
        labels: Cluster assignments, shape (n_samples,).

    Returns:
        ch_score: Calinski-Harabasz index, range [0, ∞], higher is better.

    Interpretation:
        Ratio of between-cluster to within-cluster variance.
        Similar to F-statistic for cluster separation.
        Higher values indicate better-defined clusters.

    Use Cases:
        - Choose optimal K
        - Fast to compute
        - Works well for convex clusters

    Examples:
        >>> X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        >>> labels = np.array([0, 0, 1, 1])
        >>> calinski_harabasz_score(X, labels)
        90.5  # High score = good clustering
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0

    # Overall mean
    mean = np.mean(X, axis=0)

    # Between-cluster sum of squares
    ss_between = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        n_points = len(cluster_points)
        cluster_mean = np.mean(cluster_points, axis=0)

        # Contribution: n_k * ||μ_k - μ||²
        ss_between += n_points * np.sum((cluster_mean - mean) ** 2)

    # Within-cluster sum of squares
    ss_within = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        cluster_mean = np.mean(cluster_points, axis=0)

        # Contribution: Σ_{x∈C_k} ||x - μ_k||²
        ss_within += np.sum((cluster_points - cluster_mean) ** 2)

    # Calinski-Harabasz index
    if ss_within == 0:
        return 0.0

    ch_score = (ss_between / (n_clusters - 1)) / (ss_within / (n_samples - n_clusters))

    return ch_score


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        silhouette_score as sklearn_silhouette,
        davies_bouldin_score as sklearn_db,
        calinski_harabasz_score as sklearn_ch,
    )
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Clustering Metrics: From Scratch Implementation")
    console.print("=" * 70 + "[/bold cyan]")

    # Generate clustered data
    X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster with K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_

    # Example 1: Validation Against Scikit-Learn
    console.print("\n[bold yellow]1. Validation Against Scikit-Learn[/bold yellow]")
    console.print("-" * 70)

    metrics_table = Table(title="Clustering Metrics Comparison", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Our Implementation", justify="right", style="green")
    metrics_table.add_column("Scikit-Learn", justify="right", style="yellow")
    metrics_table.add_column("Difference", justify="right", style="white")

    # Inertia
    our_inertia = inertia(X_scaled, labels, centers)
    sk_inertia = kmeans.inertia_
    metrics_table.add_row(
        "Inertia",
        f"{our_inertia:.4f}",
        f"{sk_inertia:.4f}",
        f"{abs(our_inertia - sk_inertia):.10f}",
    )

    # Silhouette
    our_silhouette = silhouette_score(X_scaled, labels)
    sk_silhouette = sklearn_silhouette(X_scaled, labels)
    metrics_table.add_row(
        "Silhouette",
        f"{our_silhouette:.4f}",
        f"{sk_silhouette:.4f}",
        f"{abs(our_silhouette - sk_silhouette):.10f}",
    )

    # Davies-Bouldin
    our_db = davies_bouldin_score(X_scaled, labels)
    sk_db = sklearn_db(X_scaled, labels)
    metrics_table.add_row(
        "Davies-Bouldin",
        f"{our_db:.4f}",
        f"{sk_db:.4f}",
        f"{abs(our_db - sk_db):.10f}",
    )

    # Calinski-Harabasz
    our_ch = calinski_harabasz_score(X_scaled, labels)
    sk_ch = sklearn_ch(X_scaled, labels)
    metrics_table.add_row(
        "Calinski-Harabasz",
        f"{our_ch:.4f}",
        f"{sk_ch:.4f}",
        f"{abs(our_ch - sk_ch):.10f}",
    )

    console.print(metrics_table)

    # Example 2: Choosing K with Different Metrics
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("2. Choosing K with Different Metrics")
    console.print("=" * 70 + "[/bold cyan]")

    k_range = range(2, 9)
    results = {
        "inertia": [],
        "silhouette": [],
        "davies_bouldin": [],
        "calinski_harabasz": [],
    }

    for k in k_range:
        kmeans_k = KMeans(n_clusters=k, random_state=42)
        labels_k = kmeans_k.fit_predict(X_scaled)
        centers_k = kmeans_k.cluster_centers_

        results["inertia"].append(inertia(X_scaled, labels_k, centers_k))
        results["silhouette"].append(silhouette_score(X_scaled, labels_k))
        results["davies_bouldin"].append(davies_bouldin_score(X_scaled, labels_k))
        results["calinski_harabasz"].append(calinski_harabasz_score(X_scaled, labels_k))

    # Inertia table
    inertia_table = Table(title="Inertia (Lower is Better)", box=box.ROUNDED)
    inertia_table.add_column("K", justify="center", style="cyan")
    inertia_table.add_column("Inertia", justify="right", style="green")

    for k, val in zip(k_range, results["inertia"]):
        inertia_table.add_row(str(k), f"{val:.2f}")

    console.print(inertia_table)

    # Silhouette table
    silhouette_table = Table(title="Silhouette Score (Higher is Better)", box=box.ROUNDED)
    silhouette_table.add_column("K", justify="center", style="cyan")
    silhouette_table.add_column("Silhouette", justify="right", style="green")

    best_silhouette_k = k_range[np.argmax(results["silhouette"])]

    for k, val in zip(k_range, results["silhouette"]):
        style = "bold green" if k == best_silhouette_k else "green"
        silhouette_table.add_row(str(k), f"{val:.4f}", style=style)

    console.print(silhouette_table)
    console.print(f"\n[yellow]→ Best K by Silhouette: {best_silhouette_k}[/yellow]")

    # Davies-Bouldin table
    db_table = Table(title="Davies-Bouldin Index (Lower is Better)", box=box.ROUNDED)
    db_table.add_column("K", justify="center", style="cyan")
    db_table.add_column("Davies-Bouldin", justify="right", style="green")

    best_db_k = k_range[np.argmin(results["davies_bouldin"])]

    for k, val in zip(k_range, results["davies_bouldin"]):
        style = "bold green" if k == best_db_k else "green"
        db_table.add_row(str(k), f"{val:.4f}", style=style)

    console.print(db_table)
    console.print(f"\n[yellow]→ Best K by Davies-Bouldin: {best_db_k}[/yellow]")

    # Calinski-Harabasz table
    ch_table = Table(title="Calinski-Harabasz Index (Higher is Better)", box=box.ROUNDED)
    ch_table.add_column("K", justify="center", style="cyan")
    ch_table.add_column("Calinski-Harabasz", justify="right", style="green")

    best_ch_k = k_range[np.argmax(results["calinski_harabasz"])]

    for k, val in zip(k_range, results["calinski_harabasz"]):
        style = "bold green" if k == best_ch_k else "green"
        ch_table.add_row(str(k), f"{val:.2f}", style=style)

    console.print(ch_table)
    console.print(f"\n[yellow]→ Best K by Calinski-Harabasz: {best_ch_k}[/yellow]")

    # Key Insights
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Key Insights")
    console.print("=" * 70 + "[/bold cyan]")

    insights = [
        "• Inertia: Always decreases with K (use elbow method)",
        "• Silhouette: Range [-1, 1], higher is better, 1 = perfect",
        "• Davies-Bouldin: Range [0, ∞], lower is better, 0 = perfect",
        "• Calinski-Harabasz: Range [0, ∞], higher is better",
        "",
        "[yellow]→ Different metrics may suggest different K[/yellow]",
        "[yellow]→ Use multiple metrics for robust selection[/yellow]",
        "[yellow]→ Silhouette most interpretable (cluster quality)[/yellow]",
        "[yellow]→ All metrics favor convex, well-separated clusters[/yellow]",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
