# Machine Learning Evaluation Metrics

## Overview

**Evaluation metrics** quantify model performance and guide model selection. Different tasks require different metrics, and choosing the right metric is critical for successful machine learning projects.

**Key Principles:**
- **No single best metric**: Different metrics emphasize different aspects of performance
- **Match metric to business objective**: Optimize what matters for your use case
- **Consider class imbalance**: Some metrics mislead with imbalanced data
- **Use multiple metrics**: Get complete picture of model performance
- **Understand trade-offs**: Improving one metric may hurt another

**Main Categories:**
- **Classification metrics**: For discrete class predictions
- **Regression metrics**: For continuous value predictions
- **Ranking metrics**: For recommendation systems (not covered here)
- **Clustering metrics**: For unsupervised learning (not covered here)

---

## Classification Metrics

### Confusion Matrix

The **confusion matrix** is the foundation for binary classification metrics.

**Structure** (for binary classification):

```
                 Predicted
               Negative  Positive
Actual Negative    TN       FP
       Positive    FN       TP
```

Where:
- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II error)

**Multi-class**: NxN matrix where N = number of classes

**Interpretation**:
- Diagonal elements: correct predictions
- Off-diagonal: errors (which classes confused with which)

---

### Accuracy

**Definition**: Fraction of correct predictions.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**Range**: [0, 1], higher is better

**When to use**:
- ✓ Balanced classes (roughly equal class sizes)
- ✓ All errors equally costly
- ✓ Simple baseline metric

**When NOT to use**:
- ✗ Imbalanced classes (e.g., 95% negative, 5% positive)
  - Model predicting all negative gets 95% accuracy!
- ✗ Asymmetric costs (false positive vs false negative)
- ✗ Need probability calibration

**Example**:
```
Spam detection: 990 non-spam, 10 spam
Model predicts all non-spam → 99% accuracy (terrible model!)
```

**Paradox**: High accuracy doesn't guarantee good model with imbalanced data.

---

### Precision

**Definition**: Of all positive predictions, what fraction are correct?

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**Alternative names**: Positive Predictive Value (PPV)

**Range**: [0, 1], higher is better

**Interpretation**:
- High precision → Few false positives
- "When model says positive, how often is it right?"

**When to use**:
- ✓ False positives are costly
- ✓ Want to be confident in positive predictions
- ✓ Limited resources to act on predictions

**Examples**:
- **Email spam filter**: Don't want to mark legitimate email as spam
  - FP = important email lost
  - High precision needed
- **Medical screening**: Want positive test to be reliable
  - FP = unnecessary treatment, anxiety
- **Fraud detection (alerts)**: Limited investigator time
  - FP = wasted investigation effort

**Trade-off**: Can achieve perfect precision by predicting positive rarely (but low recall).

---

### Recall

**Definition**: Of all actual positives, what fraction did we catch?

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**Alternative names**: Sensitivity, True Positive Rate (TPR), Hit Rate

**Range**: [0, 1], higher is better

**Interpretation**:
- High recall → Few false negatives
- "Of all actual positives, how many did we find?"

**When to use**:
- ✓ False negatives are costly
- ✓ Want to catch as many positives as possible
- ✓ Missing a positive is worse than false alarm

**Examples**:
- **Cancer screening**: Must catch all cases
  - FN = missed cancer diagnosis (deadly)
  - High recall needed
- **Fraud detection (blocking)**: Must catch all fraud
  - FN = fraudulent transaction goes through
- **Search engine**: Want to retrieve all relevant documents
  - FN = relevant result not shown

**Trade-off**: Can achieve perfect recall by predicting everything as positive (but low precision).

---

### Precision-Recall Trade-off

**Fundamental tension**: Improving one often hurts the other.

**Why the trade-off**:
- **Increase threshold** for positive prediction:
  - More confident predictions → Higher precision
  - But miss some positives → Lower recall

- **Decrease threshold**:
  - Catch more positives → Higher recall
  - But more false alarms → Lower precision

**Precision-Recall Curve**:
- Plot precision (y-axis) vs recall (x-axis) at different thresholds
- Shows full trade-off spectrum
- **Area Under Curve (PR-AUC)**: Single number summarizing performance
  - Range: [0, 1], higher is better
  - Better than ROC-AUC for imbalanced data

**Visual**:
```
Precision
    ^
  1 |\___
    |    \___
    |        \___
    |            \___
  0 |________________\___>
    0                    1  Recall
```

**Typical shape**: Precision decreases as recall increases.

---

### F1-Score

**Definition**: Harmonic mean of precision and recall.

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}
$$

**Range**: [0, 1], higher is better

**Why harmonic mean**:
- Balances precision and recall
- Punishes extreme values (low precision OR low recall → low F1)
- If precision=100%, recall=10% → F1=18% (not 55% with arithmetic mean)

**When to use**:
- ✓ Need balance between precision and recall
- ✓ Imbalanced classes (better than accuracy)
- ✓ No clear preference between FP and FN costs

**Generalization - F-beta Score**:
$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

Where:
- $\beta > 1$: Emphasize recall (e.g., F2 = 2x weight on recall)
- $\beta < 1$: Emphasize precision (e.g., F0.5 = 2x weight on precision)
- $\beta = 1$: F1-score (balanced)

**Example**:
- **Cancer detection**: Use F2 (recall more important)
- **Spam filtering**: Use F0.5 (precision more important)

---

### Specificity

**Definition**: Of all actual negatives, what fraction did we correctly identify?

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

**Alternative names**: True Negative Rate (TNR), Selectivity

**Range**: [0, 1], higher is better

**Interpretation**:
- High specificity → Few false positives
- "How good are we at identifying negatives?"

**Complement**:
$$
\text{False Positive Rate (FPR)} = 1 - \text{Specificity} = \frac{FP}{TN + FP}
$$

**When to use**:
- When correctly identifying negatives is important
- Complements recall (TPR) in ROC analysis

---

### ROC Curve and AUC

**ROC (Receiver Operating Characteristic) Curve**:
- Plots True Positive Rate (Recall) vs False Positive Rate
- Shows performance across all classification thresholds

**Axes**:
- **Y-axis**: TPR (Recall) = TP / (TP + FN)
- **X-axis**: FPR = FP / (FP + TN)

**Visual**:
```
TPR (Recall)
    ^
  1 |     Perfect
    |    /
    |   / Good
    |  /
    | /  Random
  0 |/_____________>
    0             1  FPR
```

**Interpretation**:
- **Diagonal line** (y=x): Random classifier
- **Above diagonal**: Better than random
- **Perfect classifier**: (0, 1) top-left corner (100% TPR, 0% FPR)

**AUC-ROC (Area Under ROC Curve)**:
$$
\text{AUC} \in [0, 1]
$$

**Interpretation**:
- **1.0**: Perfect classifier
- **0.9-1.0**: Excellent
- **0.8-0.9**: Good
- **0.7-0.8**: Fair
- **0.6-0.7**: Poor
- **0.5**: Random guessing
- **< 0.5**: Worse than random (predictions inverted)

**Probabilistic interpretation**:
AUC = Probability that model ranks a random positive example higher than a random negative example

**When to use AUC-ROC**:
- ✓ Threshold-independent metric
- ✓ Balanced classes
- ✓ Care about ranking (not just classification)

**When NOT to use**:
- ✗ Highly imbalanced data
  - High AUC even with poor precision
  - Use PR-AUC instead

**Example**:
- Imbalanced (1% positive):
  - Model A: Precision=10%, Recall=90%, AUC=0.95
  - Model B: Precision=50%, Recall=50%, AUC=0.75
  - AUC prefers A, but B more useful (higher precision)

---

### Log Loss (Cross-Entropy Loss)

**Definition**: Measures quality of predicted probabilities.

$$
\text{Log Loss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

Where:
- $y_i \in \{0, 1\}$: true label
- $p_i \in [0, 1]$: predicted probability of class 1

**Range**: [0, ∞], lower is better

**Interpretation**:
- Heavily penalizes confident wrong predictions
- $p=0.01$ for true positive → loss ≈ 4.6
- $p=0.99$ for true negative → loss ≈ 4.6

**When to use**:
- ✓ Need calibrated probabilities (not just class predictions)
- ✓ Probabilistic predictions matter
- ✓ Training objective for logistic regression, neural networks

**Advantages**:
- Differentiable (good for gradient descent)
- Encourages probability calibration
- Punishes overconfidence

**Multi-class generalization**:
$$
\text{Categorical Cross-Entropy} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{ic} \log(p_{ic})
$$

---

### Matthews Correlation Coefficient (MCC)

**Definition**: Correlation between predicted and actual classes.

$$
MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
$$

**Range**: [-1, 1]
- **+1**: Perfect prediction
- **0**: Random prediction
- **-1**: Perfect disagreement (inverted predictions)

**Advantages**:
- Balanced metric even with imbalanced classes
- Accounts for all four confusion matrix quadrants
- More informative than F1 for imbalanced data

**When to use**:
- ✓ Imbalanced classes
- ✓ Need single metric accounting for all errors
- ✓ Binary classification

**Why better than accuracy for imbalanced data**:
- Balances true/false, positive/negative rates
- Doesn't inflate with class imbalance

---

### Multi-Class Metrics

For K > 2 classes, extend binary metrics:

#### Averaging Strategies

**1. Macro-Average**:
- Compute metric for each class independently
- Average across classes (equal weight per class)

$$
\text{Macro-Avg} = \frac{1}{K} \sum_{k=1}^{K} \text{Metric}_k
$$

**Use when**: All classes equally important (e.g., species classification)

**2. Micro-Average**:
- Aggregate TP, FP, FN across all classes
- Compute metric from aggregated counts

$$
\text{Micro-Avg} = \text{Metric}(\sum TP, \sum FP, \sum FN)
$$

**Use when**: Classes weighted by frequency (e.g., document classification)

**3. Weighted Average**:
- Weight each class by its support (number of true instances)

$$
\text{Weighted-Avg} = \sum_{k=1}^{K} w_k \cdot \text{Metric}_k, \quad w_k = \frac{n_k}{n}
$$

**Use when**: Classes have different importance based on frequency

#### Example

Dataset: 100 samples, 3 classes (80 class 1, 15 class 2, 5 class 3)

| Class | Precision |
|-------|-----------|
| 1     | 0.90      |
| 2     | 0.70      |
| 3     | 0.50      |

- **Macro-Avg**: (0.90 + 0.70 + 0.50) / 3 = **0.70**
- **Weighted-Avg**: (0.90×80 + 0.70×15 + 0.50×5) / 100 = **0.865**

**Interpretation**:
- Macro: treats all classes equally
- Weighted: dominated by majority class performance

---

### Class Imbalance Strategies

**Problem**: Accuracy misleading with imbalanced data.

**Solutions**:

1. **Use appropriate metrics**:
   - Precision, Recall, F1, PR-AUC, MCC
   - Avoid: Accuracy, ROC-AUC

2. **Resampling**:
   - **Oversample** minority class (SMOTE, ADASYN)
   - **Undersample** majority class
   - Hybrid approaches

3. **Class weights**:
   - Penalize errors on minority class more
   - `class_weight='balanced'` in scikit-learn

4. **Threshold tuning**:
   - Lower threshold for minority class
   - Optimize for F1 or other relevant metric

5. **Ensemble methods**:
   - Balanced bagging
   - EasyEnsemble

---

## Regression Metrics

### Mean Squared Error (MSE)

**Definition**: Average of squared errors.

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Range**: [0, ∞], lower is better

**Properties**:
- **Heavily penalizes large errors** (squared term)
- **Not robust to outliers** (outliers have large impact)
- **Same units as variance** of target variable

**When to use**:
- ✓ Large errors are particularly bad
- ✓ Training objective for many models (OLS regression)
- ✓ Data doesn't have extreme outliers

**When NOT to use**:
- ✗ Outliers present (consider MAE or Huber loss)
- ✗ Units hard to interpret (use RMSE)

---

### Root Mean Squared Error (RMSE)

**Definition**: Square root of MSE.

$$
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

**Range**: [0, ∞], lower is better

**Properties**:
- **Same units as target variable** (interpretable)
- Penalizes large errors (like MSE)
- Standard deviation of residuals

**When to use**:
- ✓ Want interpretable error magnitude
- ✓ Large errors are particularly bad
- ✓ Most common regression metric

**Interpretation**:
- RMSE = $5,000 for house prices → typical prediction off by $5K

---

### Mean Absolute Error (MAE)

**Definition**: Average of absolute errors.

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

**Range**: [0, ∞], lower is better

**Properties**:
- **Linear penalty** (treats all errors equally)
- **Robust to outliers** (no squaring)
- Same units as target variable

**When to use**:
- ✓ Outliers present
- ✓ All errors equally bad (no special penalty for large errors)
- ✓ Want robust metric

**MAE vs RMSE**:
- RMSE ≥ MAE (always)
- Large difference → presence of large errors
- Similar values → errors uniformly distributed

---

### R² (Coefficient of Determination)

**Definition**: Proportion of variance explained by the model.

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
$$

Where:
- $SS_{res}$: Residual sum of squares
- $SS_{tot}$: Total sum of squares
- $\bar{y}$: Mean of target variable

**Range**: (-∞, 1], higher is better
- **1**: Perfect predictions
- **0**: Model as good as predicting mean
- **< 0**: Model worse than predicting mean

**Interpretation**:
- R² = 0.85 → Model explains 85% of variance in target

**When to use**:
- ✓ Compare models on same dataset
- ✓ Understand proportion of variance explained
- ✓ Linear regression interpretability

**Limitations**:
- Always increases with more features (even if useless)
  - Use adjusted R² for model comparison
- Can be negative on test set
- Doesn't indicate if predictions are biased

---

### Adjusted R²

**Definition**: R² adjusted for number of predictors.

$$
R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
$$

Where:
- $n$: number of samples
- $p$: number of predictors (features)

**Range**: (-∞, 1], higher is better

**Properties**:
- Penalizes adding uninformative features
- Can decrease when adding features (unlike R²)
- Better for model comparison

**When to use**:
- ✓ Compare models with different numbers of features
- ✓ Feature selection
- ✓ Avoid overfitting from too many features

**Interpretation**:
- If R²_adj decreases when adding feature → feature not useful

---

### Mean Absolute Percentage Error (MAPE)

**Definition**: Average of absolute percentage errors.

$$
MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

**Range**: [0, ∞], lower is better

**Properties**:
- **Scale-independent** (percentage)
- Easy to interpret

**When to use**:
- ✓ Compare models across different scales
- ✓ Business reporting (easy to explain)

**Limitations**:
- **Undefined for $y_i = 0$**
- **Asymmetric**: penalizes over-predictions more than under-predictions
- **Biased** toward under-predictions

**Example of asymmetry**:
- True value: 100
- Prediction: 50 → Error = 50%
- Prediction: 150 → Error = 50%
- But 50 and 150 are equally far from 100!

**Alternative**: Symmetric MAPE (sMAPE)

---

### Mean Squared Logarithmic Error (MSLE)

**Definition**: MSE on log-transformed values.

$$
MSLE = \frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2
$$

**Range**: [0, ∞], lower is better

**Properties**:
- Penalizes under-predictions more than over-predictions
- Appropriate for exponential growth
- Relative errors (not absolute)

**When to use**:
- ✓ Target has exponential trend (e.g., stock prices, population)
- ✓ Under-predictions worse than over-predictions
- ✓ Target varies over several orders of magnitude

**Interpretation**:
- Predicting 90 instead of 100: larger penalty
- Predicting 110 instead of 100: smaller penalty

---

### Huber Loss

**Definition**: Combines MSE (for small errors) and MAE (for large errors).

$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

**Parameter**: $\delta$ (threshold for switching from quadratic to linear)

**Properties**:
- **Robust to outliers** (like MAE for large errors)
- **Differentiable everywhere** (unlike MAE)
- Smooth transition between quadratic and linear

**When to use**:
- ✓ Outliers present but want smooth loss
- ✓ Training robust regression models
- ✓ Need differentiable loss function

---

## Metric Selection Guide

### Classification Decision Tree

```
Are classes balanced?
├─ Yes → Use Accuracy, F1, ROC-AUC
└─ No (Imbalanced)
   ├─ False Positives costly? → Optimize Precision
   ├─ False Negatives costly? → Optimize Recall
   ├─ Need balance? → Use F1, MCC, PR-AUC
   └─ Need probabilities? → Use Log Loss

Need threshold-independent metric?
├─ Yes → Use AUC-ROC (balanced) or PR-AUC (imbalanced)
└─ No → Use threshold-dependent metrics
```

### Regression Decision Tree

```
Outliers present?
├─ Yes → Use MAE or Huber Loss
└─ No → Use MSE or RMSE

Need interpretable scale?
├─ Absolute error → MAE or RMSE
├─ Percentage error → MAPE
└─ Variance explained → R²

Comparing models?
├─ Same # features → R²
└─ Different # features → Adjusted R²

Exponential growth?
└─ Use MSLE
```

---

## Common Scenarios

### Binary Classification

| Scenario | Primary Metric | Secondary Metrics |
|----------|---------------|-------------------|
| **Spam detection** | Precision (don't miss real emails) | F0.5, Accuracy |
| **Cancer screening** | Recall (catch all cases) | F2, PR-AUC |
| **Fraud detection** | PR-AUC, F1 | Precision, Recall |
| **Credit scoring** | Log Loss (need probabilities) | ROC-AUC, F1 |
| **Churn prediction** | F1, PR-AUC (imbalanced) | Precision@K |

### Regression

| Scenario | Primary Metric | Secondary Metrics |
|----------|---------------|-------------------|
| **House price prediction** | RMSE (dollar error) | MAE, R² |
| **Stock price forecasting** | MSLE (exponential) | MAPE |
| **Temperature forecasting** | MAE (outliers) | RMSE |
| **Sales forecasting** | MAPE (business %) | RMSE |
| **Model comparison** | R² or Adjusted R² | RMSE |

---

## Cross-Validation Considerations

**Stratified K-Fold** (Classification):
- Maintains class distribution in each fold
- Critical for imbalanced data
- Use with accuracy, F1, ROC-AUC

**Time Series Split** (Temporal data):
- Respects time order (no future data leakage)
- Train on past, test on future
- Critical for forecasting

**Metrics to report**:
- Mean and standard deviation across folds
- Shows stability of performance
- Example: "F1 = 0.85 ± 0.03"

---

## Metrics vs Business Objectives

**Align metrics with business goals**:

| Business Goal | ML Metric |
|---------------|-----------|
| Minimize customer complaints | Precision (reduce FP) |
| Maximize revenue from leads | Recall (catch all opportunities) |
| Optimize ROI | Custom: (Revenue from TP - Cost of FP) |
| Accurate forecasting | RMSE, MAE |
| Fair predictions across groups | Demographic parity, Equalized odds |

**Custom metrics**:
- Often needed for real-world problems
- Weight errors by business cost
- Example: FP costs $10, FN costs $100
  - Custom loss = $10 × FP + $100 × FN

---

## Key Takeaways

**Classification**:
- **Imbalanced data**: Use Precision, Recall, F1, MCC, PR-AUC (not accuracy)
- **Costly FP**: Optimize Precision
- **Costly FN**: Optimize Recall
- **Probabilities needed**: Use Log Loss
- **Threshold-independent**: Use ROC-AUC (balanced) or PR-AUC (imbalanced)

**Regression**:
- **Large errors bad**: Use MSE or RMSE
- **Outliers present**: Use MAE or Huber Loss
- **Interpretable error**: Use MAE or RMSE
- **Percentage error**: Use MAPE (if no zeros)
- **Variance explained**: Use R²

**General**:
- **No single best metric**: Use multiple metrics for complete picture
- **Match business objective**: Optimize what matters
- **Cross-validate**: Report mean ± std across folds
- **Imbalanced data**: Most common challenge, requires careful metric selection

**Common mistakes**:
1. Using accuracy on imbalanced data
2. Using ROC-AUC on highly imbalanced data (use PR-AUC)
3. Optimizing metric that doesn't match business goal
4. Reporting single metric without uncertainty
5. Ignoring calibration of predicted probabilities

---

## Further Reading

- **Metrics for imbalanced learning**: Davis & Goadrich (2006) - PR curves vs ROC curves
- **Probabilistic calibration**: Platt scaling, isotonic regression
- **Custom loss functions**: Domain-specific weighted errors
- **Fairness metrics**: Demographic parity, equalized odds, calibration by group
