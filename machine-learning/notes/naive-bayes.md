# Naive Bayes Classifiers

## Overview

**Naive Bayes** is a family of probabilistic classifiers based on **Bayes' theorem** with a strong (naive) independence assumption: all features are conditionally independent given the class label. Despite this unrealistic assumption, Naive Bayes often performs surprisingly well in practice, especially for text classification and high-dimensional data.

**Key Characteristics:**
- **Generative model**: Models the joint distribution $P(X, y)$, not just decision boundary
- **Probabilistic**: Provides class probabilities, not just predictions
- **Fast training and prediction**: Linear time complexity in features and samples
- **Works with small datasets**: Requires fewer training examples than discriminative models
- **Handles high dimensions well**: Effective even when $d \gg n$

**Common Variants:**
1. **Gaussian Naive Bayes**: For continuous features (assumes Gaussian distribution)
2. **Multinomial Naive Bayes**: For discrete count data (e.g., word counts in text)
3. **Bernoulli Naive Bayes**: For binary features (e.g., word presence/absence)

---

## Bayes' Theorem

### The Foundation

**Bayes' theorem** relates conditional probabilities:

$$
P(y | \mathbf{x}) = \frac{P(\mathbf{x} | y) \cdot P(y)}{P(\mathbf{x})}
$$

Where:
- $P(y | \mathbf{x})$ = **posterior**: Probability of class $y$ given features $\mathbf{x}$
- $P(\mathbf{x} | y)$ = **likelihood**: Probability of observing features given class $y$
- $P(y)$ = **prior**: Probability of class $y$ (before seeing features)
- $P(\mathbf{x})$ = **evidence**: Marginal probability of features (constant for all classes)

### Classification Decision

To classify a new example $\mathbf{x}$, choose the class with highest posterior:

$$
\hat{y} = \arg\max_{y} P(y | \mathbf{x}) = \arg\max_{y} P(\mathbf{x} | y) \cdot P(y)
$$

We can drop $P(\mathbf{x})$ since it's the same for all classes (doesn't affect argmax).

---

## The Naive Independence Assumption

### The Problem

Computing $P(\mathbf{x} | y)$ requires estimating the joint distribution over all $d$ features:

$$
P(\mathbf{x} | y) = P(x_1, x_2, \ldots, x_d | y)
$$

**Challenge**: With $d$ features, this requires exponentially many parameters (intractable).

### The Naive Solution

**Naive Bayes assumption**: Features are **conditionally independent** given the class:

$$
P(\mathbf{x} | y) = P(x_1, x_2, \ldots, x_d | y) = \prod_{i=1}^{d} P(x_i | y)
$$

This assumes that knowing the class, each feature is independent of all others.

**Implication**: Instead of estimating one complex joint distribution, estimate $d$ simple univariate distributions.

**Why "naive"?** This assumption is almost always false in practice (features are usually correlated), yet the classifier often works well anyway.

### Classification with Independence

$$
\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^{d} P(x_i | y)
$$

**Log probabilities** (for numerical stability):

$$
\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{i=1}^{d} \log P(x_i | y) \right]
$$

Avoids underflow from multiplying many small probabilities.

---

## Gaussian Naive Bayes

### Use Case

For **continuous features** that are approximately normally distributed.

**Examples**: Height, weight, temperature, sensor measurements

### Model

Assumes each feature $x_i$ given class $y$ follows a **Gaussian (normal) distribution**:

$$
P(x_i | y) = \frac{1}{\sqrt{2\pi\sigma_{iy}^2}} \exp\left( -\frac{(x_i - \mu_{iy})^2}{2\sigma_{iy}^2} \right)
$$

Where:
- $\mu_{iy}$ = mean of feature $i$ for class $y$
- $\sigma_{iy}^2$ = variance of feature $i$ for class $y$

### Training (Parameter Estimation)

For each class $y$ and feature $i$, estimate mean and variance from training data:

$$
\mu_{iy} = \frac{1}{n_y} \sum_{j: y_j = y} x_{ji}
$$

$$
\sigma_{iy}^2 = \frac{1}{n_y} \sum_{j: y_j = y} (x_{ji} - \mu_{iy})^2
$$

Where $n_y$ = number of training examples in class $y$.

### Prediction

$$
\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{i=1}^{d} \log P(x_i | y) \right]
$$

Substitute Gaussian PDF:

$$
\log P(x_i | y) = -\frac{1}{2}\log(2\pi\sigma_{iy}^2) - \frac{(x_i - \mu_{iy})^2}{2\sigma_{iy}^2}
$$

### Decision Boundary

Gaussian Naive Bayes has **quadratic** decision boundaries (due to squared terms in Gaussian).

**Special case**: If all classes have equal variance ($\sigma_{iy}^2$ same for all $y$), decision boundary becomes linear.

---

## Multinomial Naive Bayes

### Use Case

For **discrete count data** (non-negative integers representing frequencies).

**Primary application**: **Text classification** (document categorization, spam detection)
- Features are word counts: $x_i$ = number of times word $i$ appears in document

### Model

Assumes features follow a **multinomial distribution**:

$$
P(\mathbf{x} | y) = \frac{N!}{\prod_i x_i!} \prod_{i=1}^{d} p_{iy}^{x_i}
$$

Where:
- $N = \sum_i x_i$ = total count (e.g., total words in document)
- $p_{iy}$ = probability of feature $i$ occurring in class $y$
- $\sum_i p_{iy} = 1$ (probabilities sum to 1 for each class)

**Simplified** (for classification, constant terms cancel):

$$
P(\mathbf{x} | y) \propto \prod_{i=1}^{d} p_{iy}^{x_i}
$$

### Training (Parameter Estimation)

Estimate probability of each feature for each class:

$$
p_{iy} = \frac{\text{count}(x_i, y) + \alpha}{\text{total\_count}(y) + \alpha d}
$$

Where:
- $\text{count}(x_i, y)$ = total occurrences of feature $i$ in class $y$
- $\text{total\_count}(y)$ = total occurrences of all features in class $y$
- $\alpha$ = smoothing parameter (Laplace/additive smoothing)

**Laplace Smoothing** ($\alpha = 1$): Prevents zero probabilities when a feature never appears in a class.

### Prediction

$$
\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{i=1}^{d} x_i \log p_{iy} \right]
$$

Note: Feature counts $x_i$ are **multiplied** by log probabilities (differs from Gaussian NB).

### Text Classification Example

**Training**:
- Class "spam": 100 emails, word "free" appears 50 times total, 500 total words
  - $p_{\text{free, spam}} = \frac{50 + 1}{500 + |V|} \approx 0.1$ (where $|V|$ = vocabulary size)

**Prediction**:
- New email: "free money now" → feature vector counts occurrences
- Compute $\log P(\text{spam}) + \sum_{\text{words}} \text{count} \times \log p_{\text{word, spam}}$
- Compare to $\log P(\text{ham}) + \sum_{\text{words}} \text{count} \times \log p_{\text{word, ham}}$

---

## Bernoulli Naive Bayes

### Use Case

For **binary features** (presence/absence, 0/1).

**Example**: Text classification using **word presence** (not counts)
- $x_i = 1$ if word $i$ appears in document, $x_i = 0$ otherwise

### Model

Each feature follows a **Bernoulli distribution**:

$$
P(x_i | y) = p_{iy}^{x_i} (1 - p_{iy})^{1 - x_i} = \begin{cases}
p_{iy} & \text{if } x_i = 1 \\
1 - p_{iy} & \text{if } x_i = 0
\end{cases}
$$

Where $p_{iy}$ = probability that feature $i$ is present (1) in class $y$.

### Training

$$
p_{iy} = \frac{\text{count}(x_i = 1, y) + \alpha}{n_y + 2\alpha}
$$

Where:
- $\text{count}(x_i = 1, y)$ = number of examples in class $y$ where feature $i$ is 1
- $n_y$ = total examples in class $y$
- $\alpha$ = smoothing parameter

### Prediction

$$
\hat{y} = \arg\max_{y} \left[ \log P(y) + \sum_{i=1}^{d} \left( x_i \log p_{iy} + (1 - x_i) \log(1 - p_{iy}) \right) \right]
$$

### Difference from Multinomial NB

**Key distinction**:
- **Multinomial**: Uses word **counts** (how many times word appears)
- **Bernoulli**: Uses word **presence** (whether word appears at all)

Bernoulli NB explicitly models **absence** of features (contributes $(1 - p_{iy})$ term), while Multinomial NB treats absence as zero count (no contribution).

**When to use**:
- **Bernoulli**: Short texts where presence matters more than frequency
- **Multinomial**: Longer documents where word counts are informative

---

## Prior Probabilities

### Maximum Likelihood Estimation

Prior $P(y)$ is estimated from class frequencies in training data:

$$
P(y) = \frac{n_y}{n}
$$

Where:
- $n_y$ = number of training examples in class $y$
- $n$ = total number of training examples

### Class Imbalance

**Problem**: If classes are imbalanced (e.g., 95% class 0, 5% class 1), prior heavily biases toward majority class.

**Solutions**:
1. **Uniform prior**: Set $P(y) = \frac{1}{K}$ for all classes (ignores class distribution)
2. **Balanced prior**: Adjust priors to compensate for imbalance
3. **Resampling**: Oversample minority class or undersample majority class

---

## Laplace Smoothing (Additive Smoothing)

### The Zero Probability Problem

If a feature value never appears in a class during training:

$$
P(x_i | y) = 0 \quad \Rightarrow \quad P(\mathbf{x} | y) = 0
$$

**Issue**: One zero probability makes entire posterior zero, regardless of other features.

### Laplace Smoothing Solution

Add a small "pseudocount" $\alpha$ to all counts:

$$
P(x_i | y) = \frac{\text{count}(x_i, y) + \alpha}{\text{total\_count}(y) + \alpha d}
$$

**Effect**:
- $\alpha = 1$ (Laplace smoothing): Adds 1 to all counts
- $\alpha < 1$: Less smoothing (closer to MLE)
- $\alpha > 1$: More smoothing (stronger regularization)

**Typical choice**: $\alpha = 1$ (Laplace) or $\alpha$ tuned via cross-validation.

---

## Generative vs. Discriminative Models

### Generative Models (Naive Bayes)

Model the **joint distribution** $P(X, y) = P(X | y) P(y)$:
- Learn how data is **generated** for each class
- Can generate synthetic examples: sample from $P(X | y)$
- Require fewer training examples (stronger assumptions)

**Naive Bayes**: Models $P(X | y)$ (likelihood) and $P(y)$ (prior)

### Discriminative Models (Logistic Regression)

Model the **conditional distribution** $P(y | X)$ directly:
- Learn the decision boundary between classes
- Cannot generate synthetic data
- Typically more accurate with sufficient data (fewer assumptions)

**Logistic Regression**: Directly models $P(y = 1 | X)$ using sigmoid

### Comparison

| Aspect | Generative (Naive Bayes) | Discriminative (Logistic Regression) |
|--------|--------------------------|--------------------------------------|
| **Models** | $P(X \| y)$ and $P(y)$ | $P(y \| X)$ directly |
| **Assumptions** | Strong (feature independence) | Weaker (linear decision boundary) |
| **Data required** | Less (faster to train) | More (needs sufficient samples) |
| **Accuracy** | Lower with large data | Higher with large data |
| **Handles missing features** | Easy (marginalize out) | Difficult |
| **Interpretability** | Probabilistic (class distributions) | Coefficients (feature importance) |

**When Naive Bayes wins**:
- Small training datasets
- High-dimensional data ($d \gg n$)
- Features are approximately independent
- Need fast training and prediction

**When Logistic Regression wins**:
- Large training datasets
- Features are correlated
- Need best predictive accuracy

---

## Advantages of Naive Bayes

1. **Fast training and prediction**: Linear time $O(nd)$ (just counting)
2. **Works with small datasets**: Generative models need fewer examples
3. **Handles high dimensions**: Effective even when $d \gg n$
4. **Probabilistic output**: Returns class probabilities, not just labels
5. **No hyperparameter tuning**: Only smoothing parameter $\alpha$ (usually $\alpha = 1$)
6. **Handles missing data**: Can ignore missing features during prediction
7. **Online learning**: Can update model incrementally with new data
8. **Interpretable**: Can examine $P(x_i | y)$ to understand class characteristics

---

## Disadvantages of Naive Bayes

1. **Strong independence assumption**: Features are rarely truly independent
   - **Result**: Probability estimates can be poorly calibrated (overconfident)
   - **Note**: Class predictions often still accurate despite this

2. **Cannot learn feature interactions**: Treats all features independently
   - **Example**: Can't learn that "hot" + "summer" is common, "hot" + "winter" is rare

3. **Zero-frequency problem**: Requires smoothing to avoid zero probabilities

4. **Less accurate than discriminative models**: With sufficient data, logistic regression typically outperforms

5. **Sensitive to irrelevant features**: All features contribute equally to prediction

6. **Limited expressiveness**: Cannot capture complex decision boundaries

---

## Practical Considerations

### Feature Engineering

**Naive Bayes is sensitive to feature quality**:
- Remove highly correlated features (violates independence assumption)
- Remove irrelevant features (all features contribute to prediction)
- Feature scaling **not required** (probabilities are scale-invariant)

### Text Classification Tips

1. **Preprocessing**:
   - Lowercase all text
   - Remove stop words (common words like "the", "is")
   - Stemming or lemmatization (reduce words to root form)

2. **Feature representation**:
   - **Multinomial**: Use word counts (TF or TF-IDF)
   - **Bernoulli**: Use binary presence/absence

3. **Vocabulary size**:
   - Limit to top $k$ most frequent words (reduces dimensionality)
   - Remove very rare words (occur in < 5 documents)

### When to Use Which Variant

| Variant | Use When | Examples |
|---------|----------|----------|
| **Gaussian** | Continuous features, approximately normal | Sensor data, measurements, iris dataset |
| **Multinomial** | Discrete count data, frequencies | Document classification (word counts), rating prediction |
| **Bernoulli** | Binary features, presence/absence | Short text (word presence), spam detection, sentiment with binary features |

**Mixed features**:
- Convert continuous → discrete (binning)
- Use separate NB models per feature type (ensemble)
- Use Gaussian NB with feature transformations

---

## Probability Calibration

### The Problem

Naive Bayes often produces **poorly calibrated probabilities**:
- Posterior $P(y | \mathbf{x})$ may be too extreme (close to 0 or 1)
- **Example**: Predicts $P(\text{spam}) = 0.99$ when true probability is 0.7

**Reason**: Independence assumption is violated, leads to overconfident predictions.

### Why It Still Works

**Key insight**: Classification only needs **correct ranking**, not calibrated probabilities.
- If $P(y=1 | \mathbf{x}) > P(y=0 | \mathbf{x})$, predict class 1 (even if absolute values are wrong)

### Calibration Methods

If calibrated probabilities are needed:
1. **Platt Scaling**: Fit logistic regression on NB outputs
2. **Isotonic Regression**: Fit monotonic function to calibrate
3. **Beta Calibration**: Fit beta distribution to outputs

---

## Computational Complexity

### Training

**Time**: $O(nd)$
- One pass through data to compute counts/statistics
- $n$ = number of examples, $d$ = number of features

**Space**: $O(Kd)$
- Store parameters for each class-feature pair
- $K$ = number of classes

### Prediction

**Time**: $O(Kd)$
- Compute posterior for each of $K$ classes
- Each requires summing over $d$ features

**Very fast** compared to other classifiers (e.g., SVM, neural networks).

---

## Comparison with Other Classifiers

| Classifier | Training Time | Prediction Time | Assumptions | Accuracy (large data) |
|------------|---------------|-----------------|-------------|------------------------|
| **Naive Bayes** | Very Fast | Very Fast | Feature independence | Medium |
| **Logistic Regression** | Fast | Fast | Linear boundary | High |
| **Decision Trees** | Medium | Fast | None | Medium-High |
| **Random Forest** | Slow | Medium | None | High |
| **SVM** | Slow | Medium | Margin-based | High |
| **Neural Networks** | Very Slow | Fast | Universal approximation | Very High |

**Naive Bayes niche**: Fast training/prediction, small datasets, high dimensions, baseline model.

---

## Summary

**Naive Bayes** is a simple yet powerful probabilistic classifier based on Bayes' theorem with a naive independence assumption:

- **Three variants**: Gaussian (continuous), Multinomial (counts), Bernoulli (binary)
- **Generative model**: Models $P(X | y)$ and $P(y)$, not just decision boundary
- **Fast and scalable**: $O(nd)$ training, $O(Kd)$ prediction
- **Works with small data**: Fewer parameters to estimate than discriminative models
- **Independence assumption**: Rarely true, but classifier often works anyway

**Key Takeaways**:
- Simple to implement and understand
- Excellent baseline for classification tasks
- Particularly effective for text classification (spam detection, document categorization)
- Probability estimates may be poorly calibrated (but rankings are often correct)
- Consider logistic regression or other models for better accuracy with large datasets

**Next Steps**:
- Discriminant Analysis (LDA/QDA): Alternative generative models without independence assumption
- Feature selection and engineering for Naive Bayes
- Ensemble methods combining Naive Bayes with other classifiers
