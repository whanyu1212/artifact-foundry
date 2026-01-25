# Unsupervised Learning - From Scratch Implementations

This folder contains educational implementations of unsupervised learning algorithms for clustering and dimensionality reduction.

## Contents

### Clustering Algorithms

1. **[clustering.py](clustering.py)** - Clustering Algorithms
   - K-Means (Lloyd's algorithm with K-Means++ initialization)
   - Hierarchical Clustering (Agglomerative with multiple linkages)
   - DBSCAN (Density-Based Spatial Clustering)

### Dimensionality Reduction

2. **[dimensionality_reduction.py](dimensionality_reduction.py)** - Dimensionality Reduction
   - PCA (Principal Component Analysis via SVD)
   - Explained variance analysis

### Evaluation Metrics

3. **[clustering_metrics.py](clustering_metrics.py)** - Clustering Evaluation
   - Inertia (Within-cluster sum of squares)
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Index

## Quick Reference

### Clustering Algorithms

| Algorithm | Time Complexity | Clusters | K Required | Handles Noise | Cluster Shape |
|-----------|----------------|----------|------------|---------------|---------------|
| **K-Means** | O(nKdi) | Spherical | Yes | No | Convex |
| **Hierarchical** | O(n² log n) | Any | No | No | Any |
| **DBSCAN** | O(n log n) | Arbitrary | No | Yes | Non-convex |

### Dimensionality Reduction

| Method | Type | Time Complexity | Preserves | Use Case |
|--------|------|----------------|-----------|----------|
| **PCA** | Linear | O(nd² + d³) | Global variance | Compression, preprocessing |

### Evaluation Metrics

| Metric | Range | Best Value | Requires Labels | Notes |
|--------|-------|------------|-----------------|-------|
| **Inertia** | [0, ∞] | Lower | No | Always decreases with K |
| **Silhouette** | [-1, 1] | Higher | No | 1 = perfect, 0 = boundary, -1 = wrong cluster |
| **Davies-Bouldin** | [0, ∞] | Lower | No | Ratio of within/between cluster distances |
| **Calinski-Harabasz** | [0, ∞] | Higher | No | Variance ratio criterion |

## Mathematical Foundations

### K-Means

**Objective**: Minimize within-cluster sum of squares
$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$

**Algorithm**:
1. Initialize K centroids (K-Means++ for better results)
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Recompute centroids as cluster means
4. Repeat 2-3 until convergence

**K-Means++ Initialization**:
- First centroid: random data point
- Subsequent centroids: probability ∝ D(x)² (distance to nearest existing centroid)
- Proven to be O(log K)-competitive with optimal

### Hierarchical Clustering

**Agglomerative** (Bottom-up):
1. Start with n clusters (one per point)
2. Merge closest clusters
3. Update distance matrix
4. Repeat until one cluster

**Linkage Criteria**:
- **Single**: min distance between any two points
- **Complete**: max distance between any two points
- **Average**: average distance between all pairs
- **Ward**: minimize within-cluster variance

### DBSCAN

**Parameters**:
- **eps (ε)**: Neighborhood radius
- **min_samples**: Minimum points for core point

**Point Types**:
- **Core**: Has ≥ min_samples within eps
- **Border**: Within eps of core, but not core
- **Noise**: Neither core nor border

**Algorithm**:
1. For each unvisited point:
2. If core point: start new cluster, expand
3. If border: add to existing cluster
4. If noise: mark as outlier (-1)

### PCA

**Objective**: Find directions of maximum variance

**Steps**:
1. Center data: X ← X - mean(X)
2. Compute SVD: X = UΣV^T
3. Principal components = columns of V
4. Eigenvalues = singular values squared / n
5. Project: Z = XV_k (keep top k components)

**Variance Explained**:
$$
\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}
$$

## Implementation Notes

### K-Means

**Initialization**:
- Random initialization can lead to poor local minima
- K-Means++ significantly improves convergence
- Multiple restarts recommended (default: 10)

**Convergence**:
- Checks if centroids change by less than tolerance
- Or maximum iterations reached
- Typically converges in <100 iterations

**Inertia**:
- Objective function value
- Monotonically decreasing
- Use for elbow method (choose K)

### Hierarchical Clustering

**Distance Matrix**:
- Stores pairwise distances
- Memory: O(n²)
- Limits scalability to ~10,000 points

**Dendrogram**:
- Tree showing merge hierarchy
- Cut at height h to get flat clustering
- Visualize with scipy or matplotlib

**Linkage**:
- Ward most commonly used
- Complete for compact clusters
- Single for chaining effect
- Average balances both

### DBSCAN

**Parameter Selection**:

**eps**:
- Plot K-distance graph (distance to K-th nearest neighbor)
- Look for elbow point
- Too small → most points noise
- Too large → clusters merge

**min_samples**:
- Rule of thumb: ≥ dimensions + 1
- Common: 4-5 for 2D data
- Higher → stricter density requirement

**Spatial Index**:
- KD-Tree or Ball Tree for fast neighbor queries
- O(log n) query time vs O(n) brute force
- Breaks down in high dimensions (d > 20)

### PCA

**Centering**:
- CRITICAL: Must center data (subtract mean)
- Affects direction of principal components

**Scaling**:
- Standardize features if different scales
- PCA sensitive to feature magnitude

**Choosing k**:
- Variance threshold: 90-95% explained
- Scree plot: look for elbow
- Cross-validation on downstream task

**Whitening**:
- Decorrelate and normalize variance
- Z = XV_k Λ_k^(-1/2)
- Useful for some ML algorithms

## When to Use Which Algorithm

### Clustering

**Use K-Means when**:
- ✓ Large dataset (fast, scalable)
- ✓ Spherical clusters expected
- ✓ Know approximate K
- ✓ Speed important

**Use Hierarchical when**:
- ✓ Small dataset (n < 10,000)
- ✓ Want dendrogram visualization
- ✓ Don't know K (explore hierarchy)
- ✓ Need deterministic results

**Use DBSCAN when**:
- ✓ Arbitrary shaped clusters
- ✓ Dataset has noise/outliers
- ✓ Don't know K
- ✓ Clusters have similar density

### Dimensionality Reduction

**Use PCA when**:
- ✓ Need linear transformation
- ✓ Want to reduce noise
- ✓ Preprocessing for other algorithms
- ✓ Compression needed
- ✓ Understand variance structure

## Common Pitfalls

### Clustering

1. **Forgot to scale features**
   - K-Means sensitive to scale
   - Always use StandardScaler

2. **Wrong K in K-Means**
   - Use elbow method or silhouette
   - Try multiple values

3. **DBSCAN parameters**
   - eps too small → all noise
   - eps too large → one cluster
   - Use K-distance plot

4. **Curse of dimensionality**
   - Distances become meaningless in high-D
   - Use PCA first (reduce to 10-50 dimensions)

5. **Evaluating with wrong metric**
   - Silhouette favors convex clusters
   - Not appropriate for DBSCAN
   - Use appropriate metric for algorithm

### PCA

1. **Forgot to center data**
   - PCA requires zero-mean data
   - Subtract mean before decomposition

2. **Didn't scale features**
   - Features with large variance dominate
   - Use StandardScaler

3. **Over-interpreting PCs**
   - Principal components are linear combinations
   - Hard to interpret individual features

4. **Using too few components**
   - Check explained variance
   - 90-95% is common threshold

## Example Usage

### K-Means Clustering

```python
from clustering import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, random_state=42)

# Scale features (CRITICAL for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10)
kmeans.fit(X_scaled)

# Predictions
labels = kmeans.predict(X_scaled)
print(f"Inertia: {kmeans.inertia_:.2f}")
```

### Hierarchical Clustering

```python
from clustering import AgglomerativeClustering

# Fit hierarchical clustering
hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = hc.fit_predict(X_scaled)

# View dendrogram (requires scipy)
from scipy.cluster.hierarchy import dendrogram
dendrogram(hc.linkage_matrix_)
```

### DBSCAN

```python
from clustering import DBSCAN

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# -1 = noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
```

### PCA

```python
from dimensionality_reduction import PCA

# Fit PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print(f"Variance explained: {pca.explained_variance_ratio_}")
print(f"Total variance: {sum(pca.explained_variance_ratio_):.2%}")
```

### Choosing K with Elbow Method

```python
from clustering_metrics import inertia
import matplotlib.pyplot as plt

k_range = range(2, 11)
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
```

### Choosing K with Silhouette

```python
from clustering_metrics import silhouette_score

k_range = range(2, 11)
silhouettes = []

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouettes.append(score)

# Best K = highest silhouette
best_k = k_range[np.argmax(silhouettes)]
print(f"Best K: {best_k}")
```

## Validation Against Scikit-Learn

All implementations validated against scikit-learn:

```python
from sklearn.cluster import KMeans as SKLearnKMeans
from clustering import KMeans

# Our implementation
our_kmeans = KMeans(n_clusters=3)
our_labels = our_kmeans.fit_predict(X)

# Scikit-learn
sk_kmeans = SKLearnKMeans(n_clusters=3)
sk_labels = sk_kmeans.fit_predict(X)

# Compare inertia
print(f"Our inertia: {our_kmeans.inertia_:.4f}")
print(f"SK inertia: {sk_kmeans.inertia_:.4f}")
```

## Visualizations

The implementations include examples for:

1. **K-Means**: Cluster assignments, centroids, elbow plot
2. **Hierarchical**: Dendrogram
3. **DBSCAN**: Clusters with noise points highlighted
4. **PCA**: Scree plot, 2D projection, explained variance

## Performance Tips

### K-Means

- Use K-Means++ initialization (default)
- Multiple random restarts (n_init=10)
- Mini-batch K-Means for large data
- Consider K-Medoids for outliers

### Hierarchical

- Limit to n < 10,000 (memory O(n²))
- Use approximate methods for larger data
- Ward linkage most common

### DBSCAN

- Use spatial index (KD-Tree) for fast neighbors
- Tune eps with K-distance plot
- Consider HDBSCAN for varying density

### PCA

- Use randomized PCA for large data
- Incremental PCA for out-of-core
- Sparse PCA for interpretability

## Further Reading

See the notes in `machine-learning/notes/`:
- [unsupervised-learning.md](../../notes/unsupervised-learning.md) - Comprehensive guide

## Running Examples

Each implementation has a `__main__` block with demonstrations:

```bash
python clustering.py
python dimensionality_reduction.py
python clustering_metrics.py
```

Requirements:
- `numpy` - Core numerical operations
- `scikit-learn` - For validation and comparisons (examples only)
- `matplotlib` - For visualizations
- `rich` - For formatted terminal output
