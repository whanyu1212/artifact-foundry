# Unsupervised Learning

## Overview

**Unsupervised learning** discovers patterns and structure in data without labeled examples. The algorithm finds hidden patterns, groupings, or representations based solely on input features.

**Key Difference from Supervised Learning**:
- **Supervised**: Learn $f: X \rightarrow Y$ from labeled pairs $(x, y)$
- **Unsupervised**: Find structure in $X$ without labels $Y$

**Main Tasks**:
1. **Clustering**: Group similar data points together
2. **Dimensionality Reduction**: Find lower-dimensional representations
3. **Density Estimation**: Model probability distribution of data
4. **Anomaly Detection**: Identify outliers or unusual patterns

**Applications**:
- Customer segmentation (marketing)
- Image compression (PCA)
- Visualization (t-SNE, UMAP)
- Feature engineering (dimensionality reduction)
- Anomaly detection (fraud, network intrusion)
- Recommendation systems (collaborative filtering)

---

## Clustering

**Goal**: Partition data into groups (clusters) where:
- **High intra-cluster similarity**: Points within cluster are similar
- **Low inter-cluster similarity**: Points in different clusters are dissimilar

**Types of Clustering**:
- **Partition-based**: K-Means, K-Medoids
- **Hierarchical**: Agglomerative, Divisive
- **Density-based**: DBSCAN, OPTICS
- **Model-based**: Gaussian Mixture Models (GMM)

---

### K-Means Clustering

**Algorithm**: Partition data into K clusters by iteratively:
1. Assign points to nearest centroid
2. Update centroids as cluster means
3. Repeat until convergence

**Mathematical Formulation**:

Minimize within-cluster sum of squares (inertia):
$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$

Where:
- $K$: number of clusters
- $C_k$: set of points in cluster $k$
- $\mu_k$: centroid of cluster $k$

**Algorithm Steps**:

1. **Initialize**: Randomly select K centroids $\mu_1, ..., \mu_K$
2. **Assignment Step**: Assign each point to nearest centroid
   $$
   c_i = \arg\min_k \|x_i - \mu_k\|^2
   $$
3. **Update Step**: Recompute centroids as mean of assigned points
   $$
   \mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
   $$
4. **Repeat** steps 2-3 until convergence (centroids don't change)

**Convergence**:
- Always converges (objective decreases monotonically)
- May converge to local minimum (not global)
- Typically converges in <100 iterations

**Initialization Methods**:

1. **Random**: Randomly select K data points as initial centroids
   - Simple but can lead to poor results

2. **K-Means++**: Smart initialization (reduces bad local minima)
   - Select first centroid randomly
   - Select subsequent centroids with probability proportional to $D(x)^2$
   - $D(x)$ = distance to nearest existing centroid
   - Proven to be $O(\log K)$-competitive with optimal

3. **Multiple Random Restarts**: Run K-Means multiple times, keep best
   - Common: 10-100 restarts

**Time Complexity**:
- Per iteration: $O(nKd)$ where $n$ = samples, $K$ = clusters, $d$ = dimensions
- Total: $O(nKdi)$ where $i$ = iterations
- Typical: $O(nKd)$ as $i$ is usually small

**Advantages**:
- ✓ Simple and fast
- ✓ Scales to large datasets
- ✓ Works well with spherical clusters
- ✓ Easy to interpret

**Limitations**:
- ✗ Requires specifying K (number of clusters)
- ✗ Sensitive to initialization (use K-Means++)
- ✗ Sensitive to outliers (use K-Medoids instead)
- ✗ Assumes spherical clusters of similar size
- ✗ Assumes Euclidean distance (isotropic variance)
- ✗ Cannot handle non-convex clusters

**Choosing K**:

1. **Elbow Method**: Plot inertia vs K, look for "elbow"
   - Inertia always decreases with K
   - Elbow = point where decrease slows

2. **Silhouette Score**: Measure cluster quality
   - Range: [-1, 1], higher is better
   - Optimal K has highest average silhouette

3. **Domain Knowledge**: Business requirements may dictate K

**Variants**:
- **K-Medoids (PAM)**: Uses actual data points as centroids (robust to outliers)
- **Mini-Batch K-Means**: Uses random samples per iteration (faster for large data)
- **Fuzzy C-Means**: Soft assignments (points belong to multiple clusters)

---

### Hierarchical Clustering

**Idea**: Build hierarchy of clusters (dendrogram) showing nested groupings.

**Two Approaches**:

1. **Agglomerative (Bottom-Up)**:
   - Start: Each point is its own cluster
   - Iteratively merge closest clusters
   - Stop: All points in one cluster

2. **Divisive (Top-Down)**:
   - Start: All points in one cluster
   - Iteratively split clusters
   - Stop: Each point is its own cluster

**Agglomerative Clustering Algorithm**:

1. Start with $n$ clusters (one per point)
2. Compute distance matrix between all clusters
3. Merge two closest clusters
4. Update distance matrix
5. Repeat until one cluster remains

**Linkage Criteria** (how to measure distance between clusters):

1. **Single Linkage** (Minimum):
   $$
   d(C_i, C_j) = \min_{x \in C_i, x' \in C_j} d(x, x')
   $$
   - Minimum distance between any two points
   - Can create long, "stringy" clusters (chaining effect)

2. **Complete Linkage** (Maximum):
   $$
   d(C_i, C_j) = \max_{x \in C_i, x' \in C_j} d(x, x')
   $$
   - Maximum distance between any two points
   - Tends to create compact, spherical clusters

3. **Average Linkage** (UPGMA):
   $$
   d(C_i, C_j) = \frac{1}{|C_i| |C_j|} \sum_{x \in C_i} \sum_{x' \in C_j} d(x, x')
   $$
   - Average distance between all pairs
   - Balances single and complete linkage

4. **Ward's Method** (Minimum Variance):
   - Merge clusters minimizing within-cluster variance
   - Tends to create equal-sized clusters
   - Most commonly used

**Dendrogram**:
- Tree diagram showing cluster hierarchy
- Height = distance at which clusters merge
- **Cut dendrogram** at height $h$ to get flat clustering

**Time Complexity**:
- Naive: $O(n^3)$
- Optimized: $O(n^2 \log n)$
- Space: $O(n^2)$ for distance matrix

**Advantages**:
- ✓ No need to specify K upfront
- ✓ Produces dendrogram (interpretable hierarchy)
- ✓ Deterministic (no random initialization)
- ✓ Works with any distance metric

**Limitations**:
- ✗ Slow: $O(n^2 \log n)$ or worse
- ✗ High memory: $O(n^2)$ distance matrix
- ✗ Cannot undo merges (greedy algorithm)
- ✗ Sensitive to noise and outliers

**When to Use**:
- Small to medium datasets ($n < 10,000$)
- Want hierarchy of clusters
- Don't know K in advance
- Need deterministic results

---

### DBSCAN (Density-Based Spatial Clustering)

**Idea**: Clusters are dense regions separated by sparse regions.

**Key Concepts**:

1. **Eps (ε)**: Neighborhood radius
2. **MinPts**: Minimum points to form dense region

**Point Classifications**:

1. **Core Point**: Has ≥ MinPts within Eps radius
2. **Border Point**: Within Eps of core point, but not core itself
3. **Noise/Outlier**: Neither core nor border

**Algorithm**:

1. For each unvisited point $p$:
   - Mark $p$ as visited
   - Find neighbors within Eps: $N(p) = \{x : d(p, x) \leq \text{Eps}\}$
   - If $|N(p)| < \text{MinPts}$: mark as noise (for now)
   - Else: Create new cluster and expand:
     - Add $p$ to cluster
     - For each neighbor $q \in N(p)$:
       - If $q$ not visited: visit and add neighbors to cluster
       - If $q$ not in any cluster: add to current cluster

**Parameters**:

- **Eps**: Distance threshold
  - Too small: Most points are noise
  - Too large: Clusters merge
  - Heuristic: K-distance plot, look for elbow

- **MinPts**: Minimum cluster size
  - Rule of thumb: MinPts ≥ dimensions + 1
  - Common: MinPts = 4 for 2D data

**Time Complexity**:
- Naive: $O(n^2)$
- With spatial index (KD-tree): $O(n \log n)$

**Advantages**:
- ✓ Doesn't require K
- ✓ Finds arbitrarily shaped clusters
- ✓ Robust to outliers (marks as noise)
- ✓ Only two parameters (Eps, MinPts)

**Limitations**:
- ✗ Sensitive to parameter choice
- ✗ Struggles with varying density clusters
- ✗ High-dimensional data (curse of dimensionality)
- ✗ Border points can vary with visit order

**When to Use**:
- Clusters have arbitrary shapes
- Dataset has noise/outliers
- Clusters have similar density
- Don't know K in advance

---

### Gaussian Mixture Models (GMM)

**Idea**: Data is generated from mixture of K Gaussian distributions.

**Probabilistic Model**:
$$
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)
$$

Where:
- $\pi_k$: Mixing coefficient (probability of cluster $k$), $\sum \pi_k = 1$
- $\mu_k$: Mean of Gaussian $k$
- $\Sigma_k$: Covariance matrix of Gaussian $k$

**Soft Clustering**: Each point has probability of belonging to each cluster
$$
\gamma_{ik} = p(z_i = k | x_i) = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
$$

**Expectation-Maximization (EM) Algorithm**:

**E-Step** (Expectation): Compute responsibilities (soft assignments)
$$
\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
$$

**M-Step** (Maximization): Update parameters

$$
N_k = \sum_{i=1}^{n} \gamma_{ik}
$$

$$
\pi_k = \frac{N_k}{n}
$$

$$
\mu_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} x_i
$$

$$
\Sigma_k = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T
$$

**Repeat E and M steps until convergence**

**Convergence**: Increases log-likelihood monotonically
$$
\log p(X | \theta) = \sum_{i=1}^{n} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)
$$

**Covariance Types**:

1. **Full**: Each cluster has full covariance matrix $\Sigma_k$ (most flexible)
2. **Tied**: All clusters share same covariance $\Sigma$
3. **Diagonal**: Diagonal covariance (axes-aligned ellipses)
4. **Spherical**: $\Sigma_k = \sigma_k^2 I$ (reduces to K-Means when $\sigma_k = \sigma$)

**Advantages**:
- ✓ Soft clustering (probabilistic assignments)
- ✓ Flexible cluster shapes (elliptical)
- ✓ Provides density estimation
- ✓ Principled probabilistic framework

**Limitations**:
- ✗ Requires K (number of components)
- ✗ Sensitive to initialization
- ✗ Can converge to local maxima
- ✗ May overfit with full covariance

**GMM vs K-Means**:
- K-Means = special case of GMM (spherical, equal variance)
- GMM more flexible (elliptical clusters)
- GMM provides soft assignments
- K-Means faster and simpler

**Model Selection** (choosing K):
- **BIC (Bayesian Information Criterion)**: Penalizes complexity
- **AIC (Akaike Information Criterion)**: Less penalty than BIC
- **Cross-validation likelihood**

---

## Dimensionality Reduction

**Goal**: Find lower-dimensional representation preserving important structure.

**Why Reduce Dimensions**:
- **Visualization**: Plot high-D data in 2D/3D
- **Curse of dimensionality**: Many algorithms fail in high dimensions
- **Noise reduction**: Remove uninformative dimensions
- **Speed**: Faster training with fewer features
- **Storage**: Compress data

**Types**:
1. **Linear**: PCA, SVD, LDA (for supervised)
2. **Non-linear**: t-SNE, UMAP, Autoencoders

---

### Principal Component Analysis (PCA)

**Idea**: Find orthogonal directions of maximum variance.

**Goal**: Project $d$-dimensional data to $k$ dimensions ($k < d$) while preserving maximum variance.

**Mathematical Formulation**:

**Objective**: Find directions $w$ maximizing projected variance
$$
\max_w \frac{1}{n} \sum_{i=1}^{n} (w^T x_i)^2 \quad \text{s.t.} \quad \|w\| = 1
$$

**Solution**: Eigenvectors of covariance matrix

**Covariance Matrix**:
$$
C = \frac{1}{n} X^T X
$$

Where $X$ is centered: $X = X_{orig} - \bar{x}$

**Principal Components**: Eigenvectors of $C$, ordered by eigenvalue magnitude
$$
Cv_i = \lambda_i v_i
$$

- $v_i$: $i$-th principal component (direction)
- $\lambda_i$: variance along $i$-th component
- $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_d$

**Dimensionality Reduction**:

Keep top $k$ components:
$$
Z = XW_k
$$

Where $W_k = [v_1, v_2, ..., v_k]$ (first $k$ eigenvectors)

**Variance Explained**:
$$
\text{Variance Explained by k components} = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}
$$

**Algorithm**:

1. **Center data**: $X \leftarrow X - \bar{x}$
2. **Compute covariance**: $C = \frac{1}{n} X^T X$
3. **Eigendecomposition**: $C = V \Lambda V^T$
4. **Sort** eigenvectors by eigenvalue
5. **Project**: $Z = XW_k$ (keep top $k$ components)

**Choosing k** (number of components):

1. **Variance threshold**: Keep components explaining 90-95% variance
2. **Scree plot**: Plot eigenvalues, look for elbow
3. **Cross-validation**: Performance on downstream task

**Computational Methods**:

1. **Eigendecomposition of Covariance**: $O(d^3)$
   - Used when $d < n$

2. **SVD (Singular Value Decomposition)**: $O(\min(nd^2, n^2d))$
   - $X = U \Sigma V^T$
   - Principal components = columns of $V$
   - More numerically stable

3. **Randomized PCA**: $O(nkd)$
   - Approximate, much faster for large data
   - Good for $k \ll d$

**Whitening**:

Transform to zero mean, unit variance, uncorrelated:
$$
Z_{white} = X W_k \Lambda_k^{-1/2}
$$

Where $\Lambda_k$ = diagonal matrix of top $k$ eigenvalues

**Time Complexity**:
- Covariance: $O(nd^2)$
- Eigendecomposition: $O(d^3)$
- Total: $O(nd^2 + d^3)$

**Advantages**:
- ✓ Unsupervised (no labels needed)
- ✓ Finds global structure
- ✓ Unique solution (deterministic)
- ✓ Fast and scalable
- ✓ Reduces noise

**Limitations**:
- ✗ Linear only (cannot capture non-linear structure)
- ✗ Assumes high variance = important
- ✗ Sensitive to scaling (must standardize features)
- ✗ Interpretability reduced (PCs are linear combinations)

**When to Use**:
- Need linear dimensionality reduction
- Want to remove noise
- Compress data
- Preprocessing for other algorithms
- Understand variance structure

---

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Goal**: Visualize high-dimensional data in 2D/3D while preserving local structure.

**Idea**:
- In high-D: Model pairwise similarities using Gaussian
- In low-D: Model pairwise similarities using t-distribution
- Minimize KL divergence between distributions

**High-Dimensional Similarities** (Gaussian):
$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}
$$

Symmetrize:
$$
p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}
$$

**Low-Dimensional Similarities** (t-distribution):
$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
$$

**Objective**: Minimize KL divergence
$$
KL(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

**Why t-distribution?**
- Heavy tails (solves "crowding problem")
- Allows dissimilar points to be far apart in low-D
- Gaussian would compress all distances

**Parameters**:

1. **Perplexity**: Roughly number of neighbors to consider
   - Range: 5-50 (common: 30)
   - Low perplexity: Focus on local structure
   - High perplexity: Focus on global structure

2. **Learning rate**: Step size for gradient descent
   - Common: 100-1000

3. **Iterations**: Number of optimization steps
   - Common: 1000-5000

**Time Complexity**:
- Naive: $O(n^2)$ per iteration
- Barnes-Hut approximation: $O(n \log n)$

**Advantages**:
- ✓ Excellent for visualization
- ✓ Preserves local structure (clusters)
- ✓ Handles non-linear manifolds

**Limitations**:
- ✗ Slow for large datasets
- ✗ Non-deterministic (random initialization)
- ✗ Distances in low-D not meaningful
- ✗ Cannot embed new points (no projection)
- ✗ Sensitive to hyperparameters

**Best Practices**:
- Try different perplexities
- Run multiple times (different random seeds)
- Use PCA for initial reduction (e.g., 50D → 2D via PCA → t-SNE)
- Don't interpret distances between clusters

**When to Use**:
- Visualization only (not feature extraction)
- Small to medium datasets
- Explore cluster structure
- Understand local neighborhoods

---

## Clustering Evaluation Metrics

**Challenge**: No ground truth labels in unsupervised learning.

**Two Types**:
1. **Internal Metrics**: Use only data and clustering (no labels)
2. **External Metrics**: Compare to ground truth (if available)

---

### Internal Metrics

#### 1. Inertia (Within-Cluster Sum of Squares)

**Definition**: Sum of squared distances to nearest cluster center
$$
\text{Inertia} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$

**Range**: [0, ∞], lower is better

**Use Case**: K-Means objective, elbow method

**Limitations**:
- Always decreases with K
- No absolute interpretation
- Favors spherical clusters

---

#### 2. Silhouette Score

**Definition**: Measures how similar point is to its cluster vs other clusters.

For each point $i$:
$$
s_i = \frac{b_i - a_i}{\max(a_i, b_i)}
$$

Where:
- $a_i$ = average distance to points in same cluster
- $b_i$ = average distance to points in nearest other cluster

**Range**: [-1, 1]
- **1**: Perfect clustering (far from other clusters)
- **0**: On cluster boundary
- **-1**: Misclassified (closer to other cluster)

**Average Silhouette**: Mean over all points
$$
\text{Silhouette} = \frac{1}{n} \sum_{i=1}^{n} s_i
$$

**Use Case**:
- Choose optimal K (maximize silhouette)
- Identify well-separated clusters
- Find misclassified points

**Advantages**:
- ✓ Intuitive interpretation
- ✓ Works with any distance metric
- ✓ Identifies bad clusterings

**Limitations**:
- ✗ Expensive: $O(n^2)$
- ✗ Favors convex, dense clusters

---

#### 3. Davies-Bouldin Index

**Definition**: Average similarity between each cluster and its most similar cluster.

$$
DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i} \frac{s_i + s_j}{d_{ij}}
$$

Where:
- $s_i$ = average distance from points in cluster $i$ to centroid
- $d_{ij}$ = distance between centroids $i$ and $j$

**Range**: [0, ∞], lower is better
- 0 = perfect clustering

**Interpretation**: Ratio of within-cluster to between-cluster distances

**Advantages**:
- ✓ Fast to compute
- ✓ Intuitive (compactness vs separation)

**Limitations**:
- ✗ Assumes Euclidean distance
- ✗ Favors spherical clusters

---

#### 4. Calinski-Harabasz Index (Variance Ratio Criterion)

**Definition**: Ratio of between-cluster to within-cluster variance.

$$
CH = \frac{SS_B / (K - 1)}{SS_W / (n - K)}
$$

Where:
- $SS_B$ = between-cluster sum of squares
- $SS_W$ = within-cluster sum of squares

$$
SS_B = \sum_{k=1}^{K} |C_k| \|\mu_k - \mu\|^2
$$

$$
SS_W = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$

**Range**: [0, ∞], higher is better

**Interpretation**: F-statistic for cluster separation

**Advantages**:
- ✓ Fast to compute
- ✓ Higher score = better defined clusters

**Limitations**:
- ✗ Assumes convex clusters
- ✗ Can favor many small clusters

---

### External Metrics (with Ground Truth)

#### Adjusted Rand Index (ARI)

**Definition**: Similarity between two clusterings, adjusted for chance.

$$
ARI = \frac{\text{RI} - \text{Expected RI}}{\text{Max RI} - \text{Expected RI}}
$$

**Range**: [-1, 1]
- **1**: Perfect agreement
- **0**: Random labeling
- **<0**: Worse than random

**Use Case**: Compare clustering to ground truth labels

---

## Comparison of Methods

### Clustering Algorithms

| Algorithm | Time | Clusters | Cluster Shape | Handles Noise | K Required |
|-----------|------|----------|---------------|---------------|------------|
| **K-Means** | $O(nKd)$ | Spherical | Convex | No | Yes |
| **Hierarchical** | $O(n^2 \log n)$ | Any | Any | No | No (cut dendrogram) |
| **DBSCAN** | $O(n \log n)$ | Arbitrary | Non-convex | Yes | No |
| **GMM** | $O(nKd)$ | Elliptical | Convex | No | Yes |

### Dimensionality Reduction

| Method | Linear | Preserves | Time | Use Case |
|--------|--------|-----------|------|----------|
| **PCA** | Yes | Global variance | $O(nd^2 + d^3)$ | Compression, preprocessing |
| **t-SNE** | No | Local structure | $O(n^2)$ or $O(n \log n)$ | Visualization only |

---

## Practical Guidelines

### Clustering

**When to use K-Means**:
- Large dataset (millions of points)
- Spherical clusters expected
- Speed is important
- Have good estimate of K

**When to use Hierarchical**:
- Small dataset ($n < 10,000$)
- Want to explore different K
- Need dendrogram interpretation
- Deterministic results required

**When to use DBSCAN**:
- Arbitrary cluster shapes
- Dataset has noise/outliers
- Don't know K
- Clusters have similar density

**When to use GMM**:
- Need probabilistic assignments
- Elliptical clusters
- Want density estimation
- Soft clustering required

### Dimensionality Reduction

**When to use PCA**:
- Linear structure expected
- Need invertible transformation
- Preprocessing for ML
- Want to understand variance

**When to use t-SNE**:
- Visualization only
- Explore cluster structure
- Local relationships important
- Non-linear structure

---

## Common Pitfalls

1. **Not scaling features**: Clustering sensitive to scale
   - Always use StandardScaler or MinMaxScaler

2. **Wrong distance metric**: Euclidean not always appropriate
   - Cosine for text, Manhattan for high-D

3. **Too many dimensions**: Curse of dimensionality
   - Use PCA to reduce before clustering

4. **Evaluating with wrong metric**:
   - Silhouette favors convex clusters (bad for DBSCAN)
   - Use appropriate metric for algorithm

5. **Over-interpreting clusters**:
   - Algorithms will always find clusters (even in random data)
   - Validate with domain knowledge

6. **t-SNE misuse**:
   - Don't interpret distances between clusters
   - Don't use for feature extraction
   - Run multiple times with different perplexities

---

## Further Reading

- **K-Means**: Lloyd (1982), MacQueen (1967)
- **K-Means++**: Arthur & Vassilvitskii (2007)
- **Hierarchical**: Murtagh & Contreras (2012)
- **DBSCAN**: Ester et al. (1996)
- **GMM/EM**: Dempster et al. (1977), McLachlan & Krishnan (2008)
- **PCA**: Pearson (1901), Hotelling (1933), Jolliffe (2002)
- **t-SNE**: van der Maaten & Hinton (2008)
