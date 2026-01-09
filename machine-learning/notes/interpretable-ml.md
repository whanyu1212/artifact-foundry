# Explainable & Interpretable Machine Learning

## Overview

Explainable AI (XAI) and Interpretable Machine Learning aim to make the decisions of machine learning models understandable to humans. As models become more complex (e.g., deep neural networks, ensemble methods), they often become "black boxes," making it difficult to trust their predictions or diagnose failures.

**Interpretability** typically refers to the degree to which a human can understand the cause of a decision.
**Explainability** often refers to the techniques used to reveal the internal mechanics or decision drivers of a model.

## Key Concepts

### Intrinsic vs. Post-hoc

1.  **Intrinsic Interpretability**: The model itself is understandable by design.
    *   **Examples**: [Linear Regression](linear-regression.md), [Logistic Regression](logistic-regression.md), [Decision Trees](decision-trees.md) (shallow), Generalized Additive Models (GAMs).
    *   **Pros**: Exact explanation of how the model works.
    *   **Cons**: Often less predictive power than complex non-linear models.

2.  **Post-hoc Interpretability**: Methods applied after model training to explain predictions.
    *   **Examples**: SHAP, LIME, Permutation Importance, Partial Dependence Plots.
    *   **Pros**: Can be applied to any model (model-agnostic).
    *   **Cons**: Approximations; may not perfectly reflect the model's true logic.

### Global vs. Local

1.  **Global Interpretability**: Explains the model's behavior across the entire dataset. "What features are generally most important?"
    *   **Examples**: Feature importance weights, global SHAP values, Permutation Importance.

2.  **Local Interpretability**: Explains the model's prediction for a *single specific instance*. "Why was this specific loan application rejected?"
    *   **Examples**: LIME, Force Plots (SHAP).

## Feature Importance Techniques

### 1. Permutation Importance (Model-Agnostic)

Measures the increase in the model's prediction error after permuting the feature's values, which breaks the relationship between the feature and the true outcome.

*   **Mechanism**:
    1.  Calculate baseline metric (e.g., accuracy, MSE) on validation set.
    2.  Shuffle (permute) column $j$ in the validation set.
    3.  Recalculate metric.
    4.  Importance = (Error after permutation) - (Baseline Error).
*   **Pros**: Model-agnostic, generally reliable.
*   **Cons**: Can be misleading if features are highly correlated (creates unrealistic data combinations).

### 2. Impurity-Based Importance (Tree-Specific)

Also known as "Gini Importance" or "Mean Decrease in Impurity" for [Random Forests](random-forest.md). It counts the times a feature is used to split a node, weighted by the number of samples it splits and the decrease in impurity (e.g., Gini, Entropy).

*   **Pros**: Fast to calculate during training.
*   **Cons**: Biased towards high-cardinality features (features with many unique values).

### 3. SHAP (SHapley Additive exPlanations)

Based on game theory (Shapley values). It assigns each feature an importance value for a particular prediction.

*   **Equation**: The SHAP value is the average marginal contribution of a feature value across all possible coalitions.
*   **TreeSHAP**: A fast algorithm specifically for tree ensembles (XGBoost, LightGBM, Random Forest).
*   **Global Summary**: The absolute SHAP values can be averaged across data to show global importance.
*   **Pros**: Theoretically sound (additivity, consistency), covers local and global.
*   **Cons**: Computationally expensive (though TreeSHAP is faster).

### 4. Partial Dependence Plots (PDP)

Shows the marginal effect one or two features have on the predicted outcome.

*   **Mechanism**: Fixes the value of the feature(s) of interest and averages predictions over the marginal distribution of other features.
*   **Pros**: Intuitive visualization of relationship (linear, monotonic, complex).
*   **Cons**: Assumes independence between the feature of interest and other features.

### 5. LIME (Local Interpretable Model-agnostic Explanations)

Approximates a complex model locally with a simple, interpretable model (like a linear model) around the prediction of interest.

*   **Mechanism**:
    1.  Perturb the input sample to create a new dataset.
    2.  Get predictions from the black-box model.
    3.  Weight samples by proximity to original instance.
    4.  Train a weighted interpretable model (e.g., Lasso).

## Summary Table

| Method | Type | Scope | Best For | Caution |
| :--- | :--- | :--- | :--- | :--- |
| **Coefficients** | Intrinsic | Global | Linear Models | requires scaling; correlation issues |
| **Gini Importance** | Intrinsic | Global | Tree Models (fast) | Biased to high cardinality |
| **Permutation** | Post-hoc | Global | Any Model | Correlated features |
| **SHAP** | Post-hoc | Global/Local | High accuracy need | Slow (unless TreeSHAP) |
| **LIME** | Post-hoc | Local | Single prediction explanations | Stability issues |
| **PDP** | Post-hoc | Global | Visualizing relationships | Assumes independence |
