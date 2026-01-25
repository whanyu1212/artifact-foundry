# Machine Learning Evaluation Metrics - From Scratch Implementations

This folder contains educational implementations of common machine learning evaluation metrics for classification and regression tasks.

## Contents

### Classification Metrics

1. **[classification_metrics.py](classification_metrics.py)** - Binary and Multi-Class Classification Metrics
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-Score
   - Specificity, False Positive Rate
   - ROC-AUC, PR-AUC
   - Log Loss (Binary Cross-Entropy)
   - Matthews Correlation Coefficient (MCC)
   - Multi-class averaging (macro, micro, weighted)

### Regression Metrics

2. **[regression_metrics.py](regression_metrics.py)** - Continuous Value Prediction Metrics
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R² (Coefficient of Determination)
   - Adjusted R²
   - Mean Absolute Percentage Error (MAPE)
   - Mean Squared Logarithmic Error (MSLE)
   - Huber Loss

## Quick Reference

### Classification Metrics

| Metric | Formula | Range | Best Value | Use Case |
|--------|---------|-------|------------|----------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | [0,1] | 1 | Balanced classes |
| **Precision** | TP/(TP+FP) | [0,1] | 1 | Minimize false positives |
| **Recall** | TP/(TP+FN) | [0,1] | 1 | Minimize false negatives |
| **F1-Score** | 2·P·R/(P+R) | [0,1] | 1 | Balance P and R |
| **ROC-AUC** | Area under ROC curve | [0,1] | 1 | Threshold-independent, balanced |
| **PR-AUC** | Area under PR curve | [0,1] | 1 | Threshold-independent, imbalanced |
| **Log Loss** | -Σ[y log(p) + (1-y)log(1-p)] | [0,∞] | 0 | Probability calibration |
| **MCC** | (TP·TN-FP·FN)/√(...) | [-1,1] | 1 | Imbalanced classes |

### Regression Metrics

| Metric | Formula | Range | Best Value | Use Case |
|--------|---------|-------|------------|----------|
| **MSE** | Σ(y-ŷ)²/n | [0,∞] | 0 | Penalize large errors |
| **RMSE** | √MSE | [0,∞] | 0 | Interpretable error (same units) |
| **MAE** | Σ\|y-ŷ\|/n | [0,∞] | 0 | Robust to outliers |
| **R²** | 1 - SS_res/SS_tot | (-∞,1] | 1 | Variance explained |
| **Adj. R²** | 1 - (1-R²)(n-1)/(n-p-1) | (-∞,1] | 1 | Model comparison |
| **MAPE** | Σ\|(y-ŷ)/y\|/n · 100% | [0,∞] | 0 | Percentage error |
| **MSLE** | Σ(log(1+y)-log(1+ŷ))²/n | [0,∞] | 0 | Exponential growth |
| **Huber** | Quadratic (small) + Linear (large) | [0,∞] | 0 | Robust, differentiable |

## Mathematical Foundations

### Binary Classification Confusion Matrix

```
                Predicted
              Negative  Positive
Actual Neg       TN       FP
       Pos       FN       TP
```

**Key Relationships**:
- Precision = TP / (TP + FP) - "What fraction of positive predictions are correct?"
- Recall = TP / (TP + FN) - "What fraction of actual positives did we catch?"
- Specificity = TN / (TN + FP) - "What fraction of actual negatives did we catch?"
- F1 = 2 · (Precision · Recall) / (Precision + Recall) - Harmonic mean

### Precision-Recall Trade-off

**Fundamental tension**: Improving one typically hurts the other
- ↑ Threshold → ↑ Precision, ↓ Recall (fewer, more confident predictions)
- ↓ Threshold → ↓ Precision, ↑ Recall (more predictions, more false alarms)

### ROC Curve

Plots True Positive Rate (Recall) vs False Positive Rate at different thresholds:
- **TPR** = TP / (TP + FN) = Recall = Sensitivity
- **FPR** = FP / (FP + TN) = 1 - Specificity

**ROC-AUC** = Area under ROC curve
- 1.0 = Perfect classifier
- 0.5 = Random guessing
- <0.5 = Worse than random (inverted predictions)

### R² Interpretation

$$R^2 = 1 - \frac{SS_{residual}}{SS_{total}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**Meaning**:
- Proportion of variance in y explained by model
- R² = 0.85 → Model explains 85% of variance
- R² = 0 → Model no better than predicting mean
- R² < 0 → Model worse than predicting mean (possible on test set)

## When to Use Which Metric

### Classification

**Balanced Classes**:
- Primary: Accuracy, F1-Score
- Threshold-independent: ROC-AUC
- Probabilities: Log Loss

**Imbalanced Classes** (e.g., 95% negative, 5% positive):
- ❌ **Avoid**: Accuracy, ROC-AUC
- ✅ **Use**: Precision, Recall, F1-Score, PR-AUC, MCC

**Specific Objectives**:
- **Minimize False Positives** (e.g., spam filtering): Maximize Precision
- **Minimize False Negatives** (e.g., cancer detection): Maximize Recall
- **Balance both**: Maximize F1-Score
- **Need probabilities**: Minimize Log Loss

### Regression

**Outliers Present**:
- ❌ **Avoid**: MSE, RMSE
- ✅ **Use**: MAE, Huber Loss

**Large Errors Particularly Bad**:
- ✅ **Use**: MSE, RMSE

**Interpretable Error**:
- Absolute error magnitude: MAE, RMSE
- Percentage error: MAPE
- Variance explained: R²

**Model Comparison**:
- Same # features: R²
- Different # features: Adjusted R²

**Exponential Growth** (e.g., stock prices):
- ✅ **Use**: MSLE

## Implementation Notes

### Numerical Stability

All implementations include numerical stability considerations:

**Classification**:
- **Log Loss**: Clip probabilities to [ε, 1-ε] to avoid log(0)
- **ROC/PR AUC**: Use trapezoidal rule for integration
- **MCC**: Handle edge cases where denominator is zero

**Regression**:
- **R²**: Handle case where y has zero variance
- **MAPE**: Skip samples where y = 0 (undefined)
- **MSLE**: Use log1p for numerical stability

### Multi-Class Extensions

**Averaging Strategies**:

1. **Macro-Average**: Compute metric per class, then average
   - Equal weight to each class
   - Use when all classes equally important

2. **Micro-Average**: Aggregate TP/FP/FN globally, then compute metric
   - Weight by class frequency
   - Use for imbalanced multi-class

3. **Weighted Average**: Weight metric by class support
   - Accounts for class imbalance
   - Similar to micro for some metrics

## Common Pitfalls

### Classification

1. **Using accuracy on imbalanced data**
   - Example: 99% negative, 1% positive → always predicting negative gives 99% accuracy!
   - Solution: Use Precision, Recall, F1, or MCC

2. **Using ROC-AUC on highly imbalanced data**
   - ROC-AUC can be misleading (still high even with poor precision)
   - Solution: Use PR-AUC instead

3. **Optimizing wrong metric**
   - Maximizing accuracy when false negatives are costly
   - Solution: Align metric with business objective

4. **Ignoring probability calibration**
   - Model outputs confident but uncalibrated probabilities
   - Solution: Use calibration methods (Platt scaling, isotonic regression)

### Regression

1. **Using MSE with outliers**
   - Outliers dominate loss due to squaring
   - Solution: Use MAE or Huber Loss

2. **Using MAPE with zeros**
   - Division by zero undefined
   - Solution: Use MSLE or filter zeros

3. **Comparing R² across datasets**
   - R² is dataset-dependent
   - Solution: Only compare on same dataset

4. **Using R² for non-linear models without considering adjusted R²**
   - R² always increases with more features
   - Solution: Use Adjusted R² for model comparison

## Example Usage

### Classification Metrics

```python
from classification_metrics import (
    accuracy, precision, recall, f1_score,
    confusion_matrix, roc_auc_score
)
import numpy as np

# Binary classification predictions
y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1])
y_prob = np.array([0.9, 0.1, 0.8, 0.6, 0.2, 0.85, 0.15, 0.55])

# Compute metrics
print(f"Accuracy: {accuracy(y_true, y_pred):.4f}")
print(f"Precision: {precision(y_true, y_pred):.4f}")
print(f"Recall: {recall(y_true, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"\nConfusion Matrix:\n{cm}")
```

### Regression Metrics

```python
from regression_metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, rmse
)

# Continuous predictions
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

# Compute metrics
print(f"MSE: {mean_squared_error(y_true, y_pred):.4f}")
print(f"RMSE: {rmse(y_true, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
print(f"R²: {r2_score(y_true, y_pred):.4f}")
```

### Multi-Class Classification

```python
from classification_metrics import precision_multiclass

# Multi-class predictions (3 classes)
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2])

# Different averaging strategies
macro_prec = precision_multiclass(y_true, y_pred, average='macro')
micro_prec = precision_multiclass(y_true, y_pred, average='micro')
weighted_prec = precision_multiclass(y_true, y_pred, average='weighted')

print(f"Macro Precision: {macro_prec:.4f}")
print(f"Micro Precision: {micro_prec:.4f}")
print(f"Weighted Precision: {weighted_prec:.4f}")
```

## Validation Against Scikit-Learn

All implementations are validated against scikit-learn's metrics:

```python
from sklearn.metrics import accuracy_score, precision_score
import numpy as np

# Our implementation
from classification_metrics import accuracy, precision

y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0])

# Compare
our_acc = accuracy(y_true, y_pred)
sklearn_acc = accuracy_score(y_true, y_pred)
print(f"Difference: {abs(our_acc - sklearn_acc):.10f}")  # Should be ~0
```

## Visualizations

The implementations include examples of common visualizations:

1. **Confusion Matrix Heatmap**
2. **ROC Curve** (TPR vs FPR)
3. **Precision-Recall Curve**
4. **Threshold Analysis** (metrics at different thresholds)
5. **Residual Plots** (for regression)

## Class Imbalance Handling

Example metrics for imbalanced dataset (95% negative, 5% positive):

```python
# Simulated imbalanced data
np.random.seed(42)
y_true = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
y_pred = model.predict(X)  # hypothetical model

# Appropriate metrics
print(f"Precision: {precision(y_true, y_pred):.4f}")
print(f"Recall: {recall(y_true, y_pred):.4f}")
print(f"F1: {f1_score(y_true, y_pred):.4f}")
print(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")
print(f"PR-AUC: {pr_auc_score(y_true, y_prob):.4f}")

# Misleading metrics
print(f"Accuracy: {accuracy(y_true, y_pred):.4f}")  # Can be high but useless
```

## Further Reading

See the notes in `machine-learning/notes/`:
- [ml-metrics.md](../../notes/ml-metrics.md) - Comprehensive guide to evaluation metrics

## Running Examples

Each implementation has a `__main__` block with demonstrations:

```bash
python classification_metrics.py
python regression_metrics.py
```

Requirements:
- `numpy` - Core numerical operations
- `scikit-learn` - For validation and comparisons (examples only)
- `matplotlib` - For visualizations
- `rich` - For formatted terminal output
