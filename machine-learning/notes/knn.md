# K-Nearest Neighbors (KNN)

## Overview

**K-Nearest Neighbors (KNN)** is a simple, non-parametric, instance-based (lazy) learning algorithm used for both classification and regression. Instead of learning an explicit model during training, KNN stores all training examples and makes predictions based on the $K$ most similar instances in the training data.

**Key Characteristics:**
- **Instance-based learning**: Stores entire training dataset (no training phase)
- **Lazy learning**: Computation happens at prediction time, not training time
- **Non-parametric**: Makes no assumptions about data distribution
- **Distance-based**: Uses distance metrics to find nearest neighbors
- **Simple and intuitive**: Easy to understand and implement

**Common Use Cases:**
- Classification (image recognition, recommendation systems)
- Regression (house price prediction, stock forecasting)
- Anomaly detection (unusual instances have no nearby neighbors)
- Imputation (filling missing values based on similar examples)

---

## Algorithm Description

### KNN Classification

**Given:**
- Training set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$
- Test point $\mathbf{x}_{\text{query}}$
- Number of neighbors $K$
- Distance metric $d(\cdot, \cdot)$

**Prediction steps:**
1. **Compute distances**: Calculate distance from $\mathbf{x}_{\text{query}}$ to all training points
2. **Find $K$ nearest neighbors**: Select $K$ training points with smallest distances
3. **Majority vote**: Predict the most common class among the $K$ neighbors

$$
\hat{y} = \text{mode}\{y_{i_1}, y_{i_2}, \ldots, y_{i_K}\}
$$

where $i_1, i_2, \ldots, i_K$ are indices of the $K$ nearest neighbors.

**Weighted voting** (optional): Weight votes by inverse distance
$$
\hat{y} = \arg\max_{c} \sum_{i \in N_K(\mathbf{x})} w_i \cdot \mathbb{1}(y_i = c)
$$
where $w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$ or $w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)^2}$

### KNN Regression

For continuous targets, predict the **average** (or weighted average) of $K$ nearest neighbors:

$$
\hat{y} = \frac{1}{K} \sum_{i \in N_K(\mathbf{x})} y_i
$$

**Weighted average**:
$$
\hat{y} = \frac{\sum_{i \in N_K(\mathbf{x})} w_i \cdot y_i}{\sum_{i \in N_K(\mathbf{x})} w_i}
$$

where $w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}$ ($\epsilon$ prevents division by zero).

---

## Distance Metrics

The choice of distance metric significantly affects KNN performance.

### Euclidean Distance (L2 Norm)

Most common metric for continuous features:

$$
d_{\text{Euclidean}}(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{d} (x_j - x'_j)^2} = \|\mathbf{x} - \mathbf{x}'\|_2
$$

**Properties**:
- Sensitive to feature scales (requires normalization)
- Most intuitive ("straight-line" distance)
- Works well when all features are equally important

### Manhattan Distance (L1 Norm)

Sum of absolute differences (taxicab distance):

$$
d_{\text{Manhattan}}(\mathbf{x}, \mathbf{x}') = \sum_{j=1}^{d} |x_j - x'_j| = \|\mathbf{x} - \mathbf{x}'\|_1
$$

**Properties**:
- More robust to outliers than Euclidean
- Useful for high-dimensional data
- Appropriate when movement is restricted to grid (e.g., city blocks)

### Minkowski Distance (General Lp Norm)

Generalization of Euclidean and Manhattan:

$$
d_{\text{Minkowski}}(\mathbf{x}, \mathbf{x}') = \left( \sum_{j=1}^{d} |x_j - x'_j|^p \right)^{1/p}
$$

**Special cases**:
- $p = 1$: Manhattan distance
- $p = 2$: Euclidean distance
- $p \to \infty$: Chebyshev distance ($\max_j |x_j - x'_j|$)

### Cosine Similarity

Measures angle between vectors (not a distance, but a similarity):

$$
\text{similarity}(\mathbf{x}, \mathbf{x}') = \frac{\mathbf{x} \cdot \mathbf{x}'}{\|\mathbf{x}\| \|\mathbf{x}'\|}
$$

$$
d_{\text{cosine}}(\mathbf{x}, \mathbf{x}') = 1 - \text{similarity}(\mathbf{x}, \mathbf{x}')
$$

**Use case**: Text data, high-dimensional sparse features (e.g., TF-IDF vectors)

**Property**: Scale-invariant (only direction matters, not magnitude)

### Hamming Distance

For categorical or binary features:

$$
d_{\text{Hamming}}(\mathbf{x}, \mathbf{x}') = \sum_{j=1}^{d} \mathbb{1}(x_j \neq x'_j)
$$

**Use case**: DNA sequences, binary data, categorical features

### Mahalanobis Distance

Accounts for correlation between features:

$$
d_{\text{Mahalanobis}}(\mathbf{x}, \mathbf{x}') = \sqrt{(\mathbf{x} - \mathbf{x}')^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{x}')}
$$

where $\mathbf{\Sigma}$ is the covariance matrix.

**Property**: Automatically accounts for different feature scales and correlations

**Drawback**: Requires estimating covariance matrix (expensive for high dimensions)

---

## Choosing $K$ (Number of Neighbors)

### Effect of $K$

- **$K = 1$** (Nearest neighbor):
  - Low bias, high variance
  - Sensitive to noise and outliers
  - Decision boundary is complex (Voronoi tessellation)
  - Overfitting risk

- **Large $K$** (e.g., $K = n$):
  - High bias, low variance
  - Smooth decision boundary
  - Underfitting risk (predicts majority class everywhere)

- **Optimal $K$**: Balances bias and variance

### Selecting $K$

**1. Cross-Validation**:
   - Try different values of $K$ (e.g., $K \in \{1, 3, 5, 7, 9, 11, 15, 21, ...\}$)
   - Use cross-validation to estimate test error for each $K$
   - Choose $K$ with lowest validation error

**2. Rule of Thumb**:
   - $K = \sqrt{n}$ (where $n$ = number of training samples)
   - Use odd $K$ for binary classification (avoids ties)

**3. Validation Curve**:
   - Plot training and validation error vs. $K$
   - Look for $K$ where validation error is minimized

**Typical values**: $K \in \{3, 5, 7, 9, 11\}$ work well in practice.

---

## Computational Complexity

### Training

**Time**: $O(1)$ (just stores data)
**Space**: $O(nd)$ (store all $n$ examples with $d$ features)

**No training phase** — KNN is a "lazy learner"

### Prediction

**Naive approach**:
- **Time**: $O(nd)$ per prediction
  - Compute distance to all $n$ training points: $O(nd)$
  - Find $K$ smallest: $O(n \log K)$ (using heap)
  - **Total**: $O(nd + n \log K) \approx O(nd)$

**For $m$ test points**: $O(mnd)$

**Problem**: Slow for large datasets (must compare against all training points)

### Speeding Up KNN

**1. KD-Tree** (K-Dimensional Tree):
   - Preprocessing: $O(nd \log n)$
   - Query: $O(\log n)$ average case (degrades to $O(n)$ in high dimensions)
   - Works well for $d < 20$

**2. Ball Tree**:
   - Similar to KD-Tree but uses hyperspheres instead of hyperplanes
   - Better for high dimensions than KD-Tree
   - Query: $O(\log n)$ average case

**3. Locality-Sensitive Hashing (LSH)**:
   - Approximate nearest neighbors
   - Sublinear query time: $O(n^\rho)$ where $\rho < 1$
   - Trade-off: Speed vs. accuracy

**4. Approximate KNN**:
   - ANNOY, FAISS, HNSW libraries
   - Much faster for large-scale, high-dimensional data

---

## Curse of Dimensionality

As dimensionality $d$ increases, KNN performance degrades.

### The Problem

**High-dimensional space is sparse**:
- In high dimensions, all points are roughly equidistant
- "Nearest" neighbors are not actually near
- Volume concentrates in corners of hypercube

**Example**: Unit hypercube $[0, 1]^d$
- Volume of sphere inscribed in hypercube: $V_{\text{sphere}} / V_{\text{cube}} \to 0$ as $d \to \infty$
- Most data lies near the boundary, not interior

**Consequence**: Need exponentially more data to maintain density as $d$ increases.

**Rule**: KNN works well for $d \lesssim 10$. For $d > 20$, consider dimensionality reduction or other methods.

### Mitigating the Curse

1. **Feature selection**: Remove irrelevant features
2. **Dimensionality reduction**: Use PCA, LDA, or autoencoders
3. **Distance metric learning**: Learn a metric that emphasizes important features
4. **Use different algorithms**: Tree-based methods, neural networks handle high $d$ better

---

## Feature Scaling

**Critical for KNN**: Distance metrics are scale-dependent.

### Why Scaling Matters

If features have different scales:
- Feature with large range dominates distance calculation
- Feature with small range is ignored

**Example**:
- Age: 20-80 years (range: 60)
- Salary: \$30k-\$200k (range: 170,000)

Euclidean distance will be dominated by salary.

### Scaling Methods

**1. Standardization (Z-score normalization)**:
$$
x'_j = \frac{x_j - \mu_j}{\sigma_j}
$$
Result: Mean 0, variance 1 for each feature

**2. Min-Max Scaling**:
$$
x'_j = \frac{x_j - \min(x_j)}{\max(x_j) - \min(x_j)}
$$
Result: Features scaled to $[0, 1]$

**3. Robust Scaling** (for outliers):
$$
x'_j = \frac{x_j - \text{median}(x_j)}{\text{IQR}(x_j)}
$$

**Best practice**: Always scale features before using KNN.

---

## Decision Boundary

### Visualization

For $K = 1$:
- Decision boundary forms **Voronoi tessellation**
- Each training point has a region where it's the closest neighbor
- Highly irregular, non-smooth boundary

For larger $K$:
- Decision boundary becomes smoother
- Less sensitive to individual points

### Properties

- **Non-linear**: Can model complex decision boundaries (even with linear features)
- **Local**: Decision depends only on nearby training points
- **Piece-wise constant** (for classification): Regions of constant prediction

---

## Advantages of KNN

1. **Simple and intuitive**: Easy to understand and implement
2. **No training time**: Just store the data
3. **Non-parametric**: No assumptions about data distribution
4. **Naturally handles multi-class**: Extends trivially to $K > 2$ classes
5. **Flexible decision boundaries**: Can model complex, non-linear relationships
6. **Works for both classification and regression**
7. **Effective with sufficient data**: Given enough examples, can approximate any function

---

## Disadvantages of KNN

1. **Slow prediction**: Must compute distances to all training points ($O(nd)$ per query)
2. **Memory-intensive**: Must store entire training dataset
3. **Curse of dimensionality**: Poor performance in high dimensions ($d > 20$)
4. **Sensitive to irrelevant features**: All features equally weighted in distance
5. **Requires feature scaling**: Performance depends on proper normalization
6. **Imbalanced data**: Majority class can dominate predictions
7. **No interpretability**: Cannot inspect "model" (it's just the data)
8. **Sensitive to $K$ choice**: Performance varies with hyperparameter

---

## Variants and Extensions

### Weighted KNN

Weight neighbors by inverse distance:
$$
w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}
$$

**Benefit**: Closer neighbors have more influence (reduces effect of choice of $K$)

### Ball Tree / KD-Tree KNN

Use spatial data structures for faster nearest neighbor search.

**Trade-off**: Preprocessing time vs. faster queries

### Radius Neighbors

Instead of fixed $K$, use all neighbors within radius $r$:
$$
N_r(\mathbf{x}) = \{i : d(\mathbf{x}, \mathbf{x}_i) \leq r\}
$$

**Use case**: When density varies (adaptive neighborhood size)

### Locally Weighted Regression (LWR)

Fit a local linear model weighted by distance to query point.

**Benefit**: Captures local trends better than simple averaging

### Distance Metric Learning

Learn a distance metric from data (e.g., Mahalanobis distance with learned covariance).

**Methods**: Large Margin Nearest Neighbor (LMNN), Neighborhood Components Analysis (NCA)

---

## Handling Imbalanced Data

**Problem**: If class 1 has 90% of data, KNN will almost always predict class 1.

### Solutions

**1. Class Weighting**:
Weight votes by inverse class frequency:
$$
w_i = \frac{1}{\text{frequency}(y_i)}
$$

**2. Distance Weighting**:
Weight by inverse distance (already discussed)

**3. SMOTE (Synthetic Minority Over-sampling)**:
Generate synthetic examples of minority class

**4. Adjust $K$**:
Use smaller $K$ to reduce majority class dominance

**5. Use different metric**:
Choose metric that better separates classes

---

## KNN for Anomaly Detection

KNN can detect outliers/anomalies:

**Idea**: Anomalies have large distance to their $K$ nearest neighbors.

**Anomaly score**:
$$
\text{score}(\mathbf{x}) = \frac{1}{K} \sum_{i \in N_K(\mathbf{x})} d(\mathbf{x}, \mathbf{x}_i)
$$

**Threshold**: Flag as anomaly if score $>$ threshold.

**Advantage**: Non-parametric, no assumption about anomaly distribution.

---

## Comparison with Other Algorithms

### KNN vs. Naive Bayes

| Aspect | KNN | Naive Bayes |
|--------|-----|-------------|
| **Model** | Instance-based (no model) | Probabilistic (Gaussian, etc.) |
| **Training** | $O(1)$ (just store) | $O(nd)$ (compute statistics) |
| **Prediction** | $O(nd)$ (slow) | $O(Kd)$ (fast) |
| **Assumptions** | None | Feature independence, distribution |
| **Decision Boundary** | Non-linear, flexible | Quadratic (Gaussian NB) |
| **High dimensions** | Poor (curse of dimensionality) | Works well (independence helps) |

### KNN vs. Decision Trees

| Aspect | KNN | Decision Trees |
|--------|-----|----------------|
| **Training** | None | $O(nd \log n)$ |
| **Prediction** | $O(nd)$ | $O(\log n)$ (tree depth) |
| **Interpretability** | Low (no model) | High (rules) |
| **Feature scaling** | Required | Not required |
| **Decision boundary** | Smooth, local | Axis-aligned rectangles |
| **Overfitting** | Controlled by $K$ | Controlled by depth/pruning |

### KNN vs. SVM

| Aspect | KNN | SVM |
|--------|-----|-----|
| **Training** | None | $O(n^2)$ to $O(n^3)$ |
| **Prediction** | $O(nd)$ | $O(n_{\text{SV}} \cdot d)$ (fewer support vectors) |
| **Memory** | $O(nd)$ (all data) | $O(n_{\text{SV}} \cdot d)$ (support vectors only) |
| **Kernel** | Distance metrics | Kernel trick (implicit high-dim) |
| **Margin** | No concept of margin | Maximizes margin |
| **Outliers** | Sensitive | Robust (hinge loss) |

---

## When to Use KNN

### Best For:
- **Small to medium datasets** ($n < 100{,}000$)
- **Low dimensions** ($d < 20$)
- **Non-linear relationships** (complex decision boundaries)
- **Baseline model** (quick prototype)
- **Few training examples per class** (stores all examples)
- **Streaming data** (can add new examples easily)

### Avoid When:
- **Large datasets** (slow prediction)
- **High dimensions** (curse of dimensionality)
- **Real-time predictions** (latency matters)
- **Interpretability required** (KNN is a "black box")
- **Memory constrained** (must store all training data)

---

## Practical Tips

1. **Always scale features**: Use StandardScaler or MinMaxScaler
2. **Try multiple distance metrics**: Euclidean, Manhattan, cosine
3. **Use cross-validation for $K$**: Don't guess, validate
4. **Remove irrelevant features**: Feature selection improves performance
5. **Use weighted KNN**: Reduces sensitivity to $K$ choice
6. **Consider approximate methods**: For large datasets, use KD-Tree or approximate KNN
7. **Odd $K$ for binary classification**: Avoids ties
8. **Check for imbalanced classes**: Weight classes if needed

---

## Summary

**K-Nearest Neighbors** is a simple, non-parametric, instance-based algorithm:

- **No training phase**: Stores all training data (lazy learning)
- **Prediction**: Find $K$ nearest neighbors, vote (classification) or average (regression)
- **Distance-based**: Choice of metric (Euclidean, Manhattan, etc.) matters
- **Hyperparameter $K$**: Controls bias-variance tradeoff
- **Curse of dimensionality**: Performance degrades in high dimensions
- **Feature scaling required**: Distance metrics are scale-dependent

**Key Takeaways**:
- Simple to implement and understand
- Effective for small datasets with low dimensions
- Slow at prediction time (must search all training points)
- Works well as a baseline before trying complex models
- Sensitive to choice of $K$ and distance metric (tune via cross-validation)

**Next Steps**:
- Implement KNN from scratch with multiple distance metrics
- Explore spatial data structures (KD-Tree, Ball Tree)
- Learn about distance metric learning (LMNN, NCA)
- Compare with parametric models (Logistic Regression, SVM)
