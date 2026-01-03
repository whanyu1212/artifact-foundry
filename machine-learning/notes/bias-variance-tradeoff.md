# Bias-Variance Tradeoff

A fundamental concept in machine learning that explains the relationship between model complexity, training error, and generalization performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Bias and Variance Defined](#bias-and-variance-defined)
3. [The Tradeoff](#the-tradeoff)
4. [Mathematical Derivation](#mathematical-derivation)
5. [Recognizing High Bias vs High Variance](#recognizing-high-bias-vs-high-variance)
6. [Solutions](#solutions)

---

## Introduction

### The Core Problem

**Goal**: Build a model that performs well on **new, unseen data**.

**Challenge**: Training error alone doesn't tell us about generalization.

- Model with perfect training accuracy might fail on new data (overfitting)
- Model with poor training accuracy definitely fails on new data (underfitting)

**The bias-variance tradeoff** explains this phenomenon mathematically.

---

## Bias and Variance Defined

### Definitions

Consider:
- True function: $y = f(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$
- Trained model: $\hat{f}(x)$ (learned from training data)
- We train many models on different training sets

**Bias**: Average error from wrong assumptions

$$
\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)
$$

- How far is the **average** model from the truth?
- High bias = underfitting = systematic error
- Caused by oversimplified models

**Variance**: Error from sensitivity to training data

$$
\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]
$$

- How much does $\hat{f}$ change with different training sets?
- High variance = overfitting = sensitivity to noise
- Caused by overcomplicated models

**Irreducible Error**: Noise in the data

$$
\sigma^2 = \text{Var}[\epsilon]
$$

- Cannot be reduced by any model
- Inherent randomness in the problem

### Visual Intuition

Imagine training many models on different random samples:

**High Bias, Low Variance:**
- All models make similar predictions
- But all predictions are wrong (far from truth)
- Like archers who all miss in the same direction

**Low Bias, High Variance:**
- Models make very different predictions
- Average of predictions is close to truth
- Like archers who scatter around the target

**Low Bias, Low Variance (goal):**
- Models make similar predictions
- All predictions close to truth
- Like archers who all hit the bullseye

---

## The Tradeoff

### Why It's a Tradeoff

**Simple models** (e.g., linear regression):
- ✓ Low variance (consistent predictions)
- ✗ High bias (can't capture complexity)

**Complex models** (e.g., deep neural networks):
- ✓ Low bias (can fit complex patterns)
- ✗ High variance (sensitive to training data)

**You cannot minimize both simultaneously without more data or better features.**

### Model Complexity Spectrum

```
Underfitting ←──────────────────────→ Overfitting
High Bias                           High Variance
Low Variance                        Low Bias

Simple ←────── Model Complexity ─────→ Complex
```

**Examples by complexity:**
1. Constant prediction (mean)
2. Linear regression
3. Polynomial regression (degree 2)
4. Polynomial regression (degree 10)
5. Polynomial regression (degree 100)
6. K-NN (K=100)
7. K-NN (K=1)
8. Decision tree (depth=1)
9. Decision tree (depth=20, unpruned)

---

## Mathematical Derivation

### Decomposition of Expected Prediction Error

For a point $x$, the expected squared error is:

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
$$

**Derivation:**

Given $y = f(x) + \epsilon$ where $\mathbb{E}[\epsilon] = 0$, $\text{Var}[\epsilon] = \sigma^2$:

$$
\begin{align}
\mathbb{E}[(y - \hat{f}(x))^2]
&= \mathbb{E}[(f(x) + \epsilon - \hat{f}(x))^2] \\
&= \mathbb{E}[(f(x) - \hat{f}(x))^2 + 2\epsilon(f(x) - \hat{f}(x)) + \epsilon^2] \\
&= \mathbb{E}[(f(x) - \hat{f}(x))^2] + 2\mathbb{E}[\epsilon]\mathbb{E}[f(x) - \hat{f}(x)] + \mathbb{E}[\epsilon^2] \\
&= \mathbb{E}[(f(x) - \hat{f}(x))^2] + 0 + \sigma^2 \\
&= \mathbb{E}[(f(x) - \hat{f}(x))^2] + \sigma^2
\end{align}
$$

Now decompose the first term:

$$
\begin{align}
\mathbb{E}[(f(x) - \hat{f}(x))^2]
&= \mathbb{E}[(f(x) - \mathbb{E}[\hat{f}(x)] + \mathbb{E}[\hat{f}(x)] - \hat{f}(x))^2] \\
&= \mathbb{E}[(f(x) - \mathbb{E}[\hat{f}(x)])^2] + \mathbb{E}[(\mathbb{E}[\hat{f}(x)] - \hat{f}(x))^2] \\
&\quad + 2(f(x) - \mathbb{E}[\hat{f}(x)])\mathbb{E}[\mathbb{E}[\hat{f}(x)] - \hat{f}(x)] \\
&= (f(x) - \mathbb{E}[\hat{f}(x)])^2 + \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2] + 0 \\
&= \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)]
\end{align}
$$

Therefore:

$$
\boxed{\mathbb{E}[(y - \hat{f}(x))^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2}
$$

### Interpretation

**Total Error** has three components:

1. **Bias²**: Error from wrong model assumptions
   - Reducible by using more complex model
   - But increases variance

2. **Variance**: Error from sensitivity to training data
   - Reducible by using simpler model or more data
   - But increases bias

3. **Irreducible Error (σ²)**: Noise in the problem
   - Cannot be reduced
   - Fundamental limit on performance

**The tradeoff**: Reducing bias increases variance, and vice versa.

---

## Recognizing High Bias vs High Variance

### Symptoms

| Symptom | High Bias (Underfitting) | High Variance (Overfitting) |
|---------|-------------------------|----------------------------|
| **Training error** | High | Low |
| **Validation error** | High (similar to train) | High (much higher than train) |
| **Gap** | Small | Large |
| **Model complexity** | Too simple | Too complex |
| **More data** | Doesn't help much | Helps significantly |
| **More features** | Helps | Might make it worse |

### Learning Curves

**High Bias:**
```
Error
  │
  │ ▼─────────────── Training Error
  │ ▼─────────────── Validation Error
  │
  └─────────────────── Training Set Size
```
Both errors high and converging. More data doesn't help.

**High Variance:**
```
Error
  │      ▼────────── Validation Error
  │         ▼────
  │            ▼──
  │              ▼ Training Error
  └─────────────────── Training Set Size
```
Large gap. More data helps close the gap.

**Good Fit:**
```
Error
  │ ▼──────▼──────── Validation Error
  │  ▼────▼────────── Training Error
  │
  └─────────────────── Training Set Size
```
Both errors low, small gap.

### Validation Curves

Plot error vs model complexity:

```
Error
  │
  │  Validation Error
  │      ╱
  │     ╱
  │    ╱  ╲
  │   ╱    ╲
  │  ╱      ╲ Training Error
  │ ╱________╲______
  │
  └─────────────────── Model Complexity
      Simple        Complex

   ←Underfitting→ Sweet Spot ←Overfitting→
```

**Optimal complexity**: Where validation error is minimum.

---

## Solutions

### For High Bias (Underfitting)

**Problem**: Model too simple, can't capture patterns.

**Solutions:**

1. **Increase model complexity**
   - Linear → Polynomial
   - Shallow tree → Deeper tree
   - Smaller network → Larger network

2. **Add more features**
   - Feature engineering
   - Polynomial features
   - Interaction terms

3. **Reduce regularization**
   - Lower λ in Ridge/Lasso
   - Remove L2 penalty

4. **Train longer** (for iterative methods)
   - More epochs
   - Lower learning rate

### For High Variance (Overfitting)

**Problem**: Model too complex, memorizes training data.

**Solutions:**

1. **Get more training data**
   - Best solution if possible
   - Directly reduces variance
   - Data augmentation

2. **Simplify model**
   - Lower polynomial degree
   - Prune decision trees
   - Smaller neural network

3. **Regularization**
   - L2 (Ridge): $\lambda \sum w_i^2$
   - L1 (Lasso): $\lambda \sum |w_i|$
   - Elastic Net: combination
   - Dropout (neural networks)

4. **Reduce features**
   - Feature selection
   - PCA
   - Remove irrelevant features

5. **Early stopping**
   - Stop training when validation error increases
   - Common for neural networks

6. **Ensemble methods**
   - Bagging (Random Forest) reduces variance
   - Averaging multiple models

7. **Cross-validation**
   - Use for model selection
   - Prevents overfitting to validation set

### Algorithm-Specific Strategies

**Decision Trees:**
- Bias: Increase max_depth, reduce min_samples_leaf
- Variance: Prune, limit depth, increase min_samples_leaf

**K-Nearest Neighbors:**
- Bias: Decrease K (more local, more complex)
- Variance: Increase K (more global, smoother)

**Neural Networks:**
- Bias: More layers/neurons, train longer
- Variance: Dropout, L2 regularization, early stopping, more data

**Regression:**
- Bias: Higher degree polynomials, more features
- Variance: Regularization (Ridge/Lasso), feature selection

---

## The Optimal Balance

### How to Find It

1. **Start simple**
   - Begin with simple model (linear, shallow tree)
   - Establish baseline

2. **Check for bias**
   - If training error high → increase complexity
   - Add features, increase capacity

3. **Check for variance**
   - If val error >> train error → reduce complexity
   - Add regularization, get more data

4. **Use cross-validation**
   - Systematically try different complexities
   - Plot validation curves
   - Select model with lowest CV error

5. **Iterate**
   - Adjust based on learning curves
   - Monitor both train and validation error

### General Principles

**More data almost always helps:**
- Reduces variance directly
- Allows more complex models
- Doesn't increase bias

**Regularization is powerful:**
- Controls complexity without changing architecture
- Can tune continuously (unlike discrete choices)
- Often easier than getting more data

**There's no free lunch:**
- Every model has bias-variance tradeoff
- Goal is to minimize total error
- Perfect fit on training data is NOT the goal

---

## Practical Example

Consider polynomial regression:

| Degree | Bias | Variance | Total Error | Diagnosis |
|--------|------|----------|-------------|-----------|
| 1 | High | Low | High | Underfitting |
| 2 | Medium | Low | Medium | Getting better |
| 3 | Low | Medium | **Low** | **Sweet spot** |
| 10 | Very Low | High | Medium | Starting to overfit |
| 50 | Nearly Zero | Very High | Very High | Severe overfitting |

**Degree 3** minimizes total error (bias² + variance).

---

## Summary

### Key Takeaways

1. **Bias-variance tradeoff is fundamental**
   - All models face this tradeoff
   - Understanding it guides model selection

2. **Total error = Bias² + Variance + Irreducible Error**
   - Cannot eliminate all error
   - Goal is to balance bias and variance

3. **Diagnose using learning curves**
   - High bias: both errors high, converging
   - High variance: large gap between train and val error

4. **Different solutions for different problems**
   - High bias → increase complexity
   - High variance → regularize or get more data

5. **More data helps variance, not bias**
   - Best way to reduce variance
   - Won't fix fundamentally wrong model

6. **Regularization is your friend**
   - Powerful tool to control variance
   - Usually easier than getting more data

### The Golden Rule

**Always monitor both training and validation error.**

- Training error alone is misleading
- Validation error tells you about generalization
- The gap reveals the bias-variance balance

### Quick Diagnostic

```python
# Check bias-variance balance
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)

if train_score < 0.7 and val_score < 0.7:
    print("High bias - model too simple")
elif train_score > 0.9 and val_score < 0.7:
    print("High variance - model too complex")
else:
    print("Good balance")
```

Understanding the bias-variance tradeoff is essential for building models that generalize well to unseen data.
