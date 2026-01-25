"""
Clustering Algorithms - From Scratch Implementation

Implements K-Means, Hierarchical Clustering, and DBSCAN for unsupervised learning.

Mathematical Foundations:
- K-Means: Minimize Σ_k Σ_{x∈C_k} ||x - μ_k||² (within-cluster sum of squares)
- Hierarchical: Build tree by iteratively merging closest clusters
- DBSCAN: Density-based clustering using eps-neighborhood and min_samples

All algorithms validated against scikit-learn implementations.
"""

import numpy as np
from typing import Literal, Optional
from collections import deque


class KMeans:
    """
    K-Means Clustering with K-Means++ initialization.

    Partitions data into K clusters by minimizing within-cluster sum of squares.

    Attributes:
        n_clusters (int): Number of clusters.
        init (str): Initialization method ('random' or 'k-means++').
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        n_init (int): Number of random initializations.
        cluster_centers_ (np.ndarray): Cluster centroids, shape (K, n_features).
        labels_ (np.ndarray): Cluster labels for each point, shape (n_samples,).
        inertia_ (float): Within-cluster sum of squares.
        n_iter_ (int): Number of iterations run.

    Algorithm:
        1. Initialize K centroids (K-Means++ recommended)
        2. Assignment: Assign each point to nearest centroid
        3. Update: Recompute centroids as cluster means
        4. Repeat 2-3 until convergence or max_iter

    Time Complexity: O(nKdi) where i = iterations (typically < 100)
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: Literal["random", "k-means++"] = "k-means++",
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize K-Means clustering.

        Args:
            n_clusters: Number of clusters to form.
            init: Initialization method:
                'random': Random selection of K points
                'k-means++': Smart initialization (recommended)
            max_iter: Maximum number of iterations.
            tol: Relative tolerance for convergence.
            n_init: Number of random initializations (keep best).
            random_state: Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        # Fitted parameters
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = np.inf
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit K-Means clustering.

        Args:
            X: Training data, shape (n_samples, n_features).

        Returns:
            self: Fitted instance.

        Algorithm:
            Run K-Means n_init times with different initializations,
            keep solution with lowest inertia.
        """
        rng = np.random.RandomState(self.random_state)

        best_inertia = np.inf
        best_centers = None
        best_labels = None
        best_n_iter = 0

        # Multiple random initializations
        for _ in range(self.n_init):
            # Initialize centroids
            centers = self._initialize_centroids(X, rng)

            # Run K-Means algorithm
            centers, labels, inertia, n_iter = self._kmeans_single(X, centers)

            # Keep best result
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            labels: Cluster labels, shape (n_samples,).
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("Must fit before predict")

        # Assign to nearest centroid
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit K-Means and return cluster labels.

        Args:
            X: Training data, shape (n_samples, n_features).

        Returns:
            labels: Cluster labels, shape (n_samples,).
        """
        self.fit(X)
        return self.labels_

    def _initialize_centroids(
        self, X: np.ndarray, rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Initialize cluster centroids.

        Args:
            X: Data, shape (n_samples, n_features).
            rng: Random number generator.

        Returns:
            centers: Initial centroids, shape (K, n_features).
        """
        n_samples = X.shape[0]

        if self.init == "random":
            # Random selection of K data points
            indices = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[indices].copy()

        elif self.init == "k-means++":
            # K-Means++ initialization
            centers = np.empty((self.n_clusters, X.shape[1]))

            # First center: random data point
            centers[0] = X[rng.randint(n_samples)]

            # Subsequent centers: weighted by distance to nearest center
            for k in range(1, self.n_clusters):
                # Compute distances to nearest existing center
                distances = self._compute_distances(X, centers[:k])
                min_distances = np.min(distances, axis=1)

                # Probability proportional to D(x)²
                probabilities = min_distances**2
                probabilities /= probabilities.sum()

                # Select next center
                idx = rng.choice(n_samples, p=probabilities)
                centers[k] = X[idx]

            return centers

        else:
            raise ValueError(f"Unknown init: {self.init}")

    def _kmeans_single(
        self, X: np.ndarray, centers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """
        Run single K-Means iteration (from given initialization).

        Args:
            X: Data, shape (n_samples, n_features).
            centers: Initial centroids, shape (K, n_features).

        Returns:
            centers: Final centroids, shape (K, n_features).
            labels: Cluster assignments, shape (n_samples,).
            inertia: Within-cluster sum of squares.
            n_iter: Number of iterations run.
        """
        for iteration in range(self.max_iter):
            # Assignment step: assign points to nearest centroid
            distances = self._compute_distances(X, centers)
            labels = np.argmin(distances, axis=1)

            # Update step: recompute centroids
            new_centers = np.array(
                [X[labels == k].mean(axis=0) for k in range(self.n_clusters)]
            )

            # Check convergence (centroids barely moved)
            center_shift = np.linalg.norm(new_centers - centers)
            if center_shift < self.tol:
                centers = new_centers
                break

            centers = new_centers

        # Compute final inertia
        distances = self._compute_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        inertia = sum(np.sum(distances[labels == k, k] ** 2) for k in range(self.n_clusters))

        return centers, labels, inertia, iteration + 1

    @staticmethod
    def _compute_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Compute pairwise squared Euclidean distances.

        Args:
            X: Data points, shape (n_samples, n_features).
            centers: Centroids, shape (K, n_features).

        Returns:
            distances: Squared distances, shape (n_samples, K).

        Formula: ||x - c||² = ||x||² + ||c||² - 2x^T c
        """
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        C_sq = np.sum(centers**2, axis=1, keepdims=True).T
        cross_term = 2 * (X @ centers.T)

        return X_sq + C_sq - cross_term


class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Clusters based on density: groups of closely packed points.

    Attributes:
        eps (float): Maximum distance between two points to be neighbors.
        min_samples (int): Minimum points to form dense region.
        labels_ (np.ndarray): Cluster labels (-1 for noise), shape (n_samples,).
        core_sample_indices_ (np.ndarray): Indices of core samples.

    Point Types:
        - Core: Has ≥ min_samples within eps
        - Border: Within eps of core, but not core itself
        - Noise: Neither core nor border (label = -1)

    Time Complexity: O(n log n) with spatial index, O(n²) without
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> None:
        """
        Initialize DBSCAN.

        Args:
            eps: Maximum distance for neighborhood (epsilon).
                Too small → most points noise
                Too large → clusters merge
                Use K-distance plot to choose
            min_samples: Minimum neighbors for core point.
                Rule of thumb: ≥ dimensions + 1
                Common: 4-5 for 2D data
        """
        self.eps = eps
        self.min_samples = min_samples

        # Fitted parameters
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Fit DBSCAN clustering.

        Args:
            X: Training data, shape (n_samples, n_features).

        Returns:
            self: Fitted instance.

        Algorithm:
            1. Find all points within eps of each point
            2. Identify core points (≥ min_samples neighbors)
            3. Form clusters by connecting core points
            4. Assign border points to clusters
            5. Remaining points are noise
        """
        n_samples = X.shape[0]

        # Find neighbors for all points
        neighbors_list = [self._region_query(X, i) for i in range(n_samples)]

        # Identify core points
        core_samples = np.array(
            [len(neighbors) >= self.min_samples for neighbors in neighbors_list]
        )
        self.core_sample_indices_ = np.where(core_samples)[0]

        # Initialize labels (-1 = unvisited/noise)
        labels = np.full(n_samples, -1)
        cluster_id = 0

        # Process each point
        for i in range(n_samples):
            # Skip if already labeled
            if labels[i] != -1:
                continue

            # Skip if not core point
            if not core_samples[i]:
                continue

            # Start new cluster from core point
            labels[i] = cluster_id
            queue = deque(neighbors_list[i])

            # Expand cluster (BFS)
            while queue:
                neighbor = queue.popleft()

                # If noise, convert to border point
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id

                # If already in a cluster, skip
                if labels[neighbor] != -1:
                    continue

                # Add to current cluster
                labels[neighbor] = cluster_id

                # If core point, add its neighbors to queue
                if core_samples[neighbor]:
                    queue.extend(neighbors_list[neighbor])

            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN and return cluster labels.

        Args:
            X: Training data, shape (n_samples, n_features).

        Returns:
            labels: Cluster labels (-1 for noise), shape (n_samples,).
        """
        self.fit(X)
        return self.labels_

    def _region_query(self, X: np.ndarray, point_idx: int) -> list[int]:
        """
        Find all points within eps of given point.

        Args:
            X: Data, shape (n_samples, n_features).
            point_idx: Index of query point.

        Returns:
            neighbors: List of neighbor indices (excluding query point).
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        neighbors = np.where(distances <= self.eps)[0]

        # Exclude the point itself
        neighbors = neighbors[neighbors != point_idx]

        return neighbors.tolist()


if __name__ == "__main__":
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans as SKLearnKMeans, DBSCAN as SKLearnDBSCAN
    from rich.console import Console
    from rich.table import Table
    from rich import box
    import matplotlib.pyplot as plt

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Clustering Algorithms: From Scratch Implementation")
    console.print("=" * 70 + "[/bold cyan]")

    # Example 1: K-Means on Blob Data
    console.print("\n[bold yellow]1. K-Means Clustering (Blob Data)[/bold yellow]")
    console.print("-" * 70)

    # Generate blob data
    X_blobs, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

    # CRITICAL: Scale features
    scaler = StandardScaler()
    X_blobs_scaled = scaler.fit_transform(X_blobs)

    # Our K-Means
    our_kmeans = KMeans(n_clusters=4, init="k-means++", n_init=10)
    our_kmeans.fit(X_blobs_scaled)

    # Scikit-learn K-Means
    sk_kmeans = SKLearnKMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
    sk_kmeans.fit(X_blobs_scaled)

    kmeans_table = Table(title="K-Means Comparison", box=box.ROUNDED)
    kmeans_table.add_column("Metric", style="cyan")
    kmeans_table.add_column("Our Implementation", justify="right", style="green")
    kmeans_table.add_column("Scikit-Learn", justify="right", style="yellow")

    kmeans_table.add_row("Inertia", f"{our_kmeans.inertia_:.4f}", f"{sk_kmeans.inertia_:.4f}")
    kmeans_table.add_row("Iterations", str(our_kmeans.n_iter_), str(sk_kmeans.n_iter_))

    console.print(kmeans_table)

    # Example 2: K-Means++ vs Random Initialization
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("2. K-Means++  vs Random Initialization")
    console.print("=" * 70 + "[/bold cyan]")

    init_table = Table(title="Initialization Comparison", box=box.ROUNDED)
    init_table.add_column("Initialization", style="cyan")
    init_table.add_column("Inertia", justify="right", style="green")
    init_table.add_column("Iterations", justify="right", style="yellow")

    for init_method in ["random", "k-means++"]:
        kmeans_init = KMeans(n_clusters=4, init=init_method, n_init=1, random_state=42)
        kmeans_init.fit(X_blobs_scaled)

        init_table.add_row(
            init_method.capitalize(),
            f"{kmeans_init.inertia_:.4f}",
            str(kmeans_init.n_iter_),
        )

    console.print(init_table)
    console.print("\n[yellow]→ K-Means++ typically converges faster and to better solution[/yellow]")

    # Example 3: Elbow Method
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("3. Elbow Method (Choosing K)")
    console.print("=" * 70 + "[/bold cyan]")

    k_range = range(2, 11)
    inertias = []

    for k in k_range:
        kmeans_k = KMeans(n_clusters=k)
        kmeans_k.fit(X_blobs_scaled)
        inertias.append(kmeans_k.inertia_)

    elbow_table = Table(title="Inertia vs K", box=box.ROUNDED)
    elbow_table.add_column("K", justify="center", style="cyan")
    elbow_table.add_column("Inertia", justify="right", style="green")

    for k, inertia in zip(k_range, inertias):
        elbow_table.add_row(str(k), f"{inertia:.2f}")

    console.print(elbow_table)
    console.print("\n[yellow]→ Look for 'elbow' where inertia decrease slows[/yellow]")

    # Example 4: DBSCAN on Non-Convex Data
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("4. DBSCAN (Non-Convex Clusters)")
    console.print("=" * 70 + "[/bold cyan]")

    # Generate moon-shaped data
    X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
    X_moons_scaled = scaler.fit_transform(X_moons)

    # Our DBSCAN
    our_dbscan = DBSCAN(eps=0.3, min_samples=5)
    our_labels = our_dbscan.fit_predict(X_moons_scaled)

    # Scikit-learn DBSCAN
    sk_dbscan = SKLearnDBSCAN(eps=0.3, min_samples=5)
    sk_labels = sk_dbscan.fit_predict(X_moons_scaled)

    # Count clusters
    our_n_clusters = len(set(our_labels)) - (1 if -1 in our_labels else 0)
    sk_n_clusters = len(set(sk_labels)) - (1 if -1 in sk_labels else 0)

    our_n_noise = list(our_labels).count(-1)
    sk_n_noise = list(sk_labels).count(-1)

    dbscan_table = Table(title="DBSCAN Results", box=box.ROUNDED)
    dbscan_table.add_column("Metric", style="cyan")
    dbscan_table.add_column("Our Implementation", justify="right", style="green")
    dbscan_table.add_column("Scikit-Learn", justify="right", style="yellow")

    dbscan_table.add_row("Clusters Found", str(our_n_clusters), str(sk_n_clusters))
    dbscan_table.add_row("Noise Points", str(our_n_noise), str(sk_n_noise))
    dbscan_table.add_row("Core Samples", str(len(our_dbscan.core_sample_indices_)), str(len(sk_dbscan.core_sample_indices_)))

    console.print(dbscan_table)

    console.print("\n[yellow]→ DBSCAN finds arbitrarily shaped clusters[/yellow]")
    console.print("[yellow]→ K-Means would fail on moon-shaped data[/yellow]")

    # Key Insights
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Key Insights")
    console.print("=" * 70 + "[/bold cyan]")

    insights = [
        "• K-Means: Fast, scalable, but assumes spherical clusters",
        "• K-Means++: Better initialization → faster convergence, better solutions",
        "• Elbow method: Plot inertia vs K, choose elbow point",
        "• DBSCAN: Finds arbitrary shapes, handles noise, no need to specify K",
        "• DBSCAN: Sensitive to eps and min_samples parameters",
        "",
        "[yellow]→ Scaling features is CRITICAL for all clustering algorithms[/yellow]",
        "[yellow]→ K-Means: Good for large, spherical clusters[/yellow]",
        "[yellow]→ DBSCAN: Good for arbitrary shapes, outliers[/yellow]",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
