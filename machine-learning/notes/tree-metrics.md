# Decision Tree Metrics & Splitting Criteria

## Overview

Decision trees recursively partition the feature space by choosing splits that maximize some measure of **purity** or **information gain**. Different metrics capture different notions of impurity and are optimized for different scenarios.

---

## Classification Metrics

### 1. Entropy (Shannon Entropy)

**Definition**: Measure of uncertainty/disorder in a set

$$H(S) = -\sum_{i=1}^c p_i \log_2(p_i)$$

Where:
- $c$: number of classes
- $p_i$: proportion of samples belonging to class $i$

**Properties**:
- Range: $[0, \log_2(c)]$
- $H(S) = 0$ when all samples belong to one class (perfectly pure)
- $H(S)$ is maximum when all classes are equally likely (maximum impurity)
- **Symmetric**: treats all errors equally

**Computational Complexity**: $O(n)$ where $n$ is number of samples (due to Counter + summation)

**Example**:
```python
# [0, 0, 1, 1, 1] → p(0)=2/5, p(1)=3/5
# H = -(2/5)*log2(2/5) - (3/5)*log2(3/5)
# H ≈ 0.971 bits
```

**When to Use**:
- Want theoretically grounded measure (information theory)
- Prefer balanced trees
- Using C4.5 or ID3 algorithms
- Classification problems where all misclassification costs are equal

**Pros**:
- ✅ Theoretically grounded in information theory
- ✅ Symmetric treatment of classes
- ✅ Smooth, differentiable function

**Cons**:
- ❌ Computationally expensive (logarithm operation)
- ❌ Can lead to overfitting without pruning
- ❌ Biased toward attributes with many values

---

### 2. Gini Impurity

**Definition**: Probability of incorrectly classifying a randomly chosen element if labeled randomly according to distribution

$$\text{Gini}(S) = 1 - \sum_{i=1}^c p_i^2 = \sum_{i=1}^c p_i(1-p_i)$$

**Alternative Formulation**:

$$\text{Gini}(S) = \sum_{i \neq j} p_i p_j$$

(Probability of picking two samples of different classes)

**Properties**:
- Range: $[0, 1-\frac{1}{c}]$
- $\text{Gini}(S) = 0$ when perfectly pure
- $\text{Gini}(S) = 1-\frac{1}{c}$ when uniformly distributed
- **Symmetric**: treats all errors equally
- Reaches maximum $0.5$ for binary classification

**Computational Complexity**: $O(n)$ - **faster than entropy** (no logarithm)

**Example**:
```python
# [0, 0, 1, 1, 1] → p(0)=2/5, p(1)=3/5
# Gini = 1 - (2/5)² - (3/5)²
# Gini = 1 - 0.16 - 0.36 = 0.48
```

**When to Use**:
- Default choice for CART algorithm
- Need computational efficiency (large datasets)
- Using scikit-learn (default criterion)
- Want slightly simpler trees

**Pros**:
- ✅ Computationally efficient (no logarithm)
- ✅ Tends to isolate most frequent class in own branch
- ✅ Less likely to overfit than entropy
- ✅ Slightly favors larger partitions

**Cons**:
- ❌ Less theoretically grounded than entropy
- ❌ May not perform as well with imbalanced classes
- ❌ Biased toward attributes with many values

---

### 3. Misclassification Error (Classification Error)

**Definition**: Proportion of misclassified samples

$$E(S) = 1 - \max_i(p_i)$$

**Properties**:
- Range: $[0, 1-\frac{1}{c}]$
- $E(S) = 0$ when perfectly pure
- Less sensitive to changes in class probabilities

**Example**:
```python
# [0, 0, 1, 1, 1] → p(0)=2/5, p(1)=3/5
# E = 1 - max(2/5, 3/5) = 1 - 3/5 = 0.4
```

**When to Use**:
- Pruning decision trees (not growing)
- Final evaluation metric
- When computational simplicity is paramount

**Pros**:
- ✅ Extremely simple to compute
- ✅ Directly interpretable
- ✅ Computationally fastest

**Cons**:
- ❌ **Not recommended for tree growing** (too insensitive)
- ❌ Doesn't discriminate well between different distributions
- ❌ Can miss important patterns

**Why Not Used for Splitting**:
```python
# Example showing insensitivity:
# S1: [0,0,0,0,1,1,1,1,1,1]  → E=0.4, Gini=0.48, Entropy=0.97
# S2: [0,0,0,0,0,1,1,1,1,1]  → E=0.5, Gini=0.50, Entropy=1.00

# Split S2 → Left: [0,0,0,0,0], Right: [1,1,1,1,1]
# Misclassification error doesn't change much, but Gini/Entropy show clear improvement
```

---

### 4. Information Gain

**Definition**: Reduction in entropy (or impurity) after splitting

$$\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

Where:
- $S$: parent node samples
- $A$: attribute/feature to split on
- $S_v$: subset of $S$ where attribute $A$ has value $v$

**For Binary Splits** (CART):

$$\text{IG}(S, A, t) = I(S) - \left(\frac{|S_{\text{left}}|}{|S|} I(S_{\text{left}}) + \frac{|S_{\text{right}}|}{|S|} I(S_{\text{right}})\right)$$

Where $I$ can be Gini or Entropy, and $t$ is the threshold.

**Properties**:
- Range: $[0, H(S)]$ or $[0, \text{Gini}(S)]$
- Higher is better (more pure children)
- Used by ID3, C4.5, and CART algorithms

**Example**:
```python
# Parent: [0, 0, 1, 1, 1]  → H(parent) = 0.971
# Split into:
#   Left: [0, 0]          → H(left) = 0.0
#   Right: [1, 1, 1]      → H(right) = 0.0

# IG = 0.971 - (2/5 * 0.0 + 3/5 * 0.0) = 0.971  (perfect split!)
```

**When to Use**:
- Selecting best split at each node
- Comparing different features
- Growing decision trees

**Bias Issue**:
- Favors attributes with many distinct values
- E.g., "customer ID" would have maximum IG but zero generalization

---

### 5. Gain Ratio (C4.5)

**Definition**: Information Gain normalized by split information

$$\text{GainRatio}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}$$

Where **Split Information** measures how much info is needed to represent the split:

$$\text{SplitInfo}(S, A) = -\sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2\left(\frac{|S_v|}{|S|}\right)$$

**Purpose**: Penalize attributes with many values (addresses IG bias)

**Properties**:
- Normalizes Information Gain
- Reduces bias toward multi-valued attributes
- Range: $[0, 1]$

**Example**:
```python
# Attribute A splits into 3 branches: [10 samples], [15 samples], [5 samples]
# Total = 30 samples

# SplitInfo = -(10/30*log2(10/30) + 15/30*log2(15/30) + 5/30*log2(5/30))
# SplitInfo ≈ 1.486

# If IG = 0.5, then:
# GainRatio = 0.5 / 1.486 ≈ 0.336
```

**When to Use**:
- C4.5 algorithm
- Features with varying numbers of distinct values
- Avoiding overfitting to high-cardinality features

**Pros**:
- ✅ Corrects bias toward multi-valued attributes
- ✅ Better generalization
- ✅ Used in C4.5 (improvement over ID3)

**Cons**:
- ❌ Can be unstable when SplitInfo is small
- ❌ More complex to compute
- ❌ May undervalue useful multi-valued attributes

**Heuristic**: C4.5 uses a two-step process:
1. First, filter features with above-average Information Gain
2. Then, among those, choose the one with highest Gain Ratio

---

## Regression Metrics

### 6. Variance Reduction

**Definition**: Reduction in variance after splitting (for continuous targets)

$$\text{VarReduction}(S, A) = \text{Var}(S) - \left(\frac{|S_L|}{|S|}\text{Var}(S_L) + \frac{|S_R|}{|S|}\text{Var}(S_R)\right)$$

Where:

$$\text{Var}(S) = \frac{1}{|S|}\sum_{i=1}^{|S|} (y_i - \bar{y})^2$$

**Properties**:
- Analogous to Information Gain for regression
- Maximizing variance reduction = minimizing weighted variance of children
- Used in CART for regression

**Example**:
```python
# Parent: y = [1.0, 2.0, 3.0, 9.0, 10.0]  → Var ≈ 16.8
# Split:
#   Left: [1.0, 2.0, 3.0]                 → Var ≈ 0.67
#   Right: [9.0, 10.0]                    → Var = 0.25

# VarReduction = 16.8 - (3/5 * 0.67 + 2/5 * 0.25)
#              = 16.8 - 0.502 ≈ 16.3  (excellent split!)
```

**When to Use**:
- Regression trees
- Continuous target variables
- CART algorithm for regression

---

### 7. Mean Squared Error (MSE) Reduction

**Definition**: Reduction in MSE after splitting

$$\text{MSE}(S) = \frac{1}{|S|}\sum_{i=1}^{|S|} (y_i - \bar{y})^2$$

$$\Delta\text{MSE} = \text{MSE}(S) - \left(\frac{|S_L|}{|S|}\text{MSE}(S_L) + \frac{|S_R|}{|S|}\text{MSE}(S_R)\right)$$

**Note**: MSE = Variance (when predicting mean)

**Prediction at Leaf**: $\hat{y} = \frac{1}{|S|}\sum_{i \in S} y_i$ (mean of samples)

**When to Use**:
- Default criterion for regression trees
- scikit-learn `DecisionTreeRegressor(criterion='squared_error')`
- Want to minimize squared error

---

### 8. Mean Absolute Error (MAE) Reduction

**Definition**: Reduction in MAE after splitting

$$\text{MAE}(S) = \frac{1}{|S|}\sum_{i=1}^{|S|} |y_i - \tilde{y}|$$

$$\Delta\text{MAE} = \text{MAE}(S) - \left(\frac{|S_L|}{|S|}\text{MAE}(S_L) + \frac{|S_R|}{|S|}\text{MAE}(S_R)\right)$$

**Prediction at Leaf**: $\hat{y} = \text{median}(y_i \text{ for } i \in S)$

**When to Use**:
- scikit-learn `DecisionTreeRegressor(criterion='absolute_error')`
- Robust to outliers (MAE less sensitive than MSE)
- Want to minimize absolute error

**Pros**:
- ✅ Robust to outliers
- ✅ More interpretable (same units as target)

**Cons**:
- ❌ Not differentiable at zero (optimization challenges)
- ❌ Computationally more expensive (finding median)

---

## Advanced Splitting Criteria

### 9. Twoing Criterion

**Definition**: Designed to find splits that divide data into two super-classes

$$\text{Twoing} = \frac{|S_L| \cdot |S_R|}{|S|^2} \left[\sum_{i=1}^c |p_L(i) - p_R(i)|\right]^2$$

Where:
- $p_L(i)$: proportion of class $i$ in left child
- $p_R(i)$: proportion of class $i$ in right child

**Purpose**:
- Find splits that separate any two groups of classes
- Useful for multi-class problems
- Alternative to Gini/Entropy

**Properties**:
- Maximizes difference in class distributions between children
- Good for creating balanced trees
- Used in some CART implementations

**When to Use**:
- Multi-class problems with many classes
- Want to create super-classes
- Alternative to traditional impurity measures

---

### 10. Chi-Square Statistic (CHAID)

**Definition**: Statistical test measuring association between split and target

$$\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where:
- $O_{ij}$: observed frequency in cell $(i,j)$
- $E_{ij}$: expected frequency assuming independence
- $r$: number of rows (child nodes)
- $c$: number of columns (classes)

**Expected Frequency**:

$$E_{ij} = \frac{(\text{row}_i \text{ total}) \times (\text{col}_j \text{ total})}{\text{grand total}}$$

**Properties**:
- Tests statistical significance of split
- P-value determines if split is meaningful
- Can handle multi-way splits (not just binary)

**When to Use**:
- CHAID (Chi-squared Automatic Interaction Detection) algorithm
- Want statistically significant splits
- Need to explain tree decisions with p-values
- Categorical features

**Pros**:
- ✅ Statistical rigor (p-values)
- ✅ Naturally handles multi-way splits
- ✅ Can detect non-linear relationships

**Cons**:
- ❌ Requires categorical data (or binning)
- ❌ Sensitive to sample size
- ❌ More complex interpretation

---

## Comparison Table

### Classification Metrics

| Metric | Range | Best Value | Computational Cost | Bias | Use Case |
|--------|-------|------------|-------------------|------|----------|
| **Entropy** | $[0, \log_2(c)]$ | 0 | High (log) | Multi-valued attrs | C4.5, ID3 |
| **Gini** | $[0, 1-1/c]$ | 0 | Low | Multi-valued attrs | CART, sklearn default |
| **Misclassification** | $[0, 1-1/c]$ | 0 | Very Low | Multi-valued attrs | Pruning only |
| **Gain Ratio** | $[0, 1]$ | 1 | High | Reduced bias | C4.5 |

### Regression Metrics

| Metric | Prediction | Outlier Sensitivity | Computational Cost |
|--------|-----------|---------------------|-------------------|
| **MSE** | Mean | High | Low |
| **MAE** | Median | Low | Medium |
| **Variance** | Mean | High | Low |

---

## Gini vs Entropy: Empirical Comparison

**Theoretical Differences**:

| Aspect | Gini | Entropy |
|--------|------|---------|
| **Formula** | $1 - \sum p_i^2$ | $-\sum p_i \log_2(p_i)$ |
| **Speed** | Faster (no log) | Slower (log operation) |
| **Purity Sensitivity** | Less sensitive | More sensitive |
| **Tree Depth** | Slightly shallower | Slightly deeper |
| **Performance** | ~Same in practice | ~Same in practice |

**Example Comparison** (binary classification):

| $p(0)$ | $p(1)$ | Gini | Entropy |
|--------|--------|------|---------|
| 0.5 | 0.5 | 0.500 | 1.000 |
| 0.6 | 0.4 | 0.480 | 0.971 |
| 0.7 | 0.3 | 0.420 | 0.881 |
| 0.8 | 0.2 | 0.320 | 0.722 |
| 0.9 | 0.1 | 0.180 | 0.469 |
| 1.0 | 0.0 | 0.000 | 0.000 |

**Observations**:
- Both reach minimum (0) at purity
- Entropy is more "curved" (more sensitive to changes near 0.5)
- Gini is more linear
- **In practice**: Performance differences are minimal (< 2%)

**Recommendation**:
- Use **Gini** (default) unless you have specific reason for Entropy
- Use **Entropy** if you want theoretically grounded measure or are implementing C4.5

---

## Implementation Considerations

### Which Metric to Choose?

**For Classification**:
```
├─ Using sklearn? → Gini (default, fast)
├─ Implementing C4.5? → Entropy + Gain Ratio
├─ Need interpretability? → Entropy (information theory)
├─ Large dataset? → Gini (faster)
├─ Multi-class with many categories? → Twoing or Chi-square
└─ Pruning? → Misclassification error
```

**For Regression**:
```
├─ Standard case? → MSE (default)
├─ Outliers present? → MAE
├─ Need robustness? → MAE
└─ Want interpretability? → MAE (same units as target)
```

### Computational Complexity

**Per Split Evaluation**:

| Metric | Complexity | Notes |
|--------|-----------|-------|
| Gini | $O(n \cdot k)$ | $n$ samples, $k$ classes |
| Entropy | $O(n \cdot k)$ | + log overhead |
| MSE | $O(n)$ | Simple arithmetic |
| MAE | $O(n \log n)$ | Median finding |

**For Continuous Features** (finding best threshold):
- Sort values: $O(n \log n)$
- Evaluate all thresholds: $O(n \cdot \text{metric\_cost})$

---

## Additional Metrics Not Covered

**Metrics You Might Add**:

1. **Friedman's MSE**: Weighted MSE used in gradient boosting

   $$\text{FriedmanMSE} = \frac{|S_L| \cdot |S_R|}{|S|^2} (\bar{y}_L - \bar{y}_R)^2$$

2. **Poisson Deviance**: For count data regression

3. **Log Loss**: For probability predictions

4. **Hellinger Distance**: Alternative to Gini/Entropy

   $$H(p,q) = \frac{1}{\sqrt{2}}\sqrt{\sum_i (\sqrt{p_i} - \sqrt{q_i})^2}$$

---

## References

- Breiman, L., et al. (1984). *Classification and Regression Trees* (CART)
- Quinlan, J. R. (1986). *Induction of Decision Trees* (ID3)
- Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*
- [Scikit-learn Documentation: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- Hastie, T., et al. (2009). *The Elements of Statistical Learning*

---

## Summary Checklist

**Your Implementation Has**:
- ✅ Entropy
- ✅ Gini Impurity
- ✅ Information Gain

**Consider Adding**:
- ⬜ Gain Ratio (for C4.5)
- ⬜ Variance Reduction (for regression trees)
- ⬜ MSE Reduction (for regression trees)
- ⬜ MAE Reduction (for robust regression trees)
- ⬜ Misclassification Error (for pruning)
- ⬜ Twoing Criterion (for multi-class)
- ⬜ Chi-Square (for CHAID)

**Key Takeaways**:
1. **Gini vs Entropy**: Similar performance, Gini is faster
2. **Information Gain**: Use for splitting, but beware of bias toward multi-valued attributes
3. **Gain Ratio**: Addresses Information Gain's bias
4. **Regression**: MSE for standard case, MAE for outliers
5. **Pruning**: Use misclassification error, not Gini/Entropy
