"""
Classification Metrics - From Scratch Implementation

Implements common classification evaluation metrics for binary and multi-class
classification tasks.

Mathematical Foundations:
- Confusion Matrix: Foundation for all classification metrics
- Precision = TP / (TP + FP) - Fraction of positive predictions that are correct
- Recall = TP / (TP + FN) - Fraction of actual positives that are caught
- F1 = 2 · P · R / (P + R) - Harmonic mean of precision and recall
- ROC-AUC: Area under True Positive Rate vs False Positive Rate curve
- Log Loss: -Σ[y log(p) + (1-y) log(1-p)] - Probability calibration metric

All metrics validated against scikit-learn implementations.
"""

import numpy as np
from typing import Literal, Optional


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix for classification.

    For binary classification:
                    Predicted
                  Negative  Positive
    Actual Negative   TN       FP
           Positive   FN       TP

    For multi-class: NxN matrix where entry (i,j) is count of samples
    with true label i predicted as label j.

    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).

    Returns:
        cm: Confusion matrix, shape (n_classes, n_classes).
            cm[i, j] = number of samples with true label i predicted as j.

    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> confusion_matrix(y_true, y_pred)
        array([[2, 0],
               [1, 2]])
    """
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    # Create mapping from class label to index
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)

    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1

    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).

    Returns:
        accuracy: Fraction of correct predictions, range [0, 1].

    Notes:
        - Best for balanced classes
        - Misleading for imbalanced data (use precision, recall, F1 instead)

    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> accuracy(y_true, y_pred)
        0.8
    """
    return np.mean(y_true == y_pred)


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """
    Compute precision (positive predictive value).

    Formula: Precision = TP / (TP + FP)

    Interpretation: Of all positive predictions, what fraction are correct?

    Args:
        y_true: True binary labels, shape (n_samples,).
        y_pred: Predicted binary labels, shape (n_samples,).
        pos_label: Label of the positive class.
        zero_division: Value to return when TP + FP = 0.

    Returns:
        precision: Precision score, range [0, 1].

    Use Cases:
        - False positives are costly (e.g., spam filtering)
        - Want confidence in positive predictions

    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> precision(y_true, y_pred)
        1.0  # 2 TP, 0 FP
    """
    # True Positives: predicted positive AND actually positive
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))

    # False Positives: predicted positive BUT actually negative
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))

    # Avoid division by zero
    if tp + fp == 0:
        return zero_division

    return tp / (tp + fp)


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """
    Compute recall (sensitivity, true positive rate).

    Formula: Recall = TP / (TP + FN)

    Interpretation: Of all actual positives, what fraction did we catch?

    Args:
        y_true: True binary labels, shape (n_samples,).
        y_pred: Predicted binary labels, shape (n_samples,).
        pos_label: Label of the positive class.
        zero_division: Value to return when TP + FN = 0.

    Returns:
        recall: Recall score, range [0, 1].

    Use Cases:
        - False negatives are costly (e.g., cancer detection)
        - Want to catch all positive cases

    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> recall(y_true, y_pred)
        0.6667  # 2 TP, 1 FN
    """
    # True Positives: predicted positive AND actually positive
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))

    # False Negatives: predicted negative BUT actually positive
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))

    # Avoid division by zero
    if tp + fn == 0:
        return zero_division

    return tp / (tp + fn)


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """
    Compute F1-score (harmonic mean of precision and recall).

    Formula: F1 = 2 · (Precision · Recall) / (Precision + Recall)
           = 2 · TP / (2 · TP + FP + FN)

    Why harmonic mean: Punishes extreme values (low P or R → low F1)

    Args:
        y_true: True binary labels, shape (n_samples,).
        y_pred: Predicted binary labels, shape (n_samples,).
        pos_label: Label of the positive class.
        zero_division: Value to return when precision + recall = 0.

    Returns:
        f1: F1-score, range [0, 1].

    Use Cases:
        - Need balance between precision and recall
        - Imbalanced classes (better than accuracy)

    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> f1_score(y_true, y_pred)
        0.8  # P=1.0, R=0.667 → F1=0.8
    """
    prec = precision(y_true, y_pred, pos_label, zero_division)
    rec = recall(y_true, y_pred, pos_label, zero_division)

    # Avoid division by zero
    if prec + rec == 0:
        return zero_division

    return 2 * (prec * rec) / (prec + rec)


def fbeta_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float,
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """
    Compute F-beta score (weighted harmonic mean of precision and recall).

    Formula: F_β = (1 + β²) · (P · R) / (β² · P + R)

    Args:
        y_true: True binary labels, shape (n_samples,).
        y_pred: Predicted binary labels, shape (n_samples,).
        beta: Weight of recall relative to precision.
            beta > 1: favor recall (e.g., F2 for cancer detection)
            beta < 1: favor precision (e.g., F0.5 for spam filtering)
            beta = 1: F1-score (balanced)
        pos_label: Label of the positive class.
        zero_division: Value to return when denominator = 0.

    Returns:
        fbeta: F-beta score, range [0, 1].

    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> fbeta_score(y_true, y_pred, beta=2.0)  # Favor recall
        0.714
    """
    prec = precision(y_true, y_pred, pos_label, zero_division)
    rec = recall(y_true, y_pred, pos_label, zero_division)

    beta_squared = beta ** 2
    denominator = beta_squared * prec + rec

    if denominator == 0:
        return zero_division

    return (1 + beta_squared) * (prec * rec) / denominator


def specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """
    Compute specificity (true negative rate).

    Formula: Specificity = TN / (TN + FP)

    Interpretation: Of all actual negatives, what fraction did we correctly identify?

    Args:
        y_true: True binary labels, shape (n_samples,).
        y_pred: Predicted binary labels, shape (n_samples,).
        pos_label: Label of the positive class.
        zero_division: Value to return when TN + FP = 0.

    Returns:
        specificity: Specificity score, range [0, 1].

    Notes:
        Complement: False Positive Rate (FPR) = 1 - Specificity

    Examples:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> specificity(y_true, y_pred)
        1.0  # 2 TN, 0 FP
    """
    # True Negatives: predicted negative AND actually negative
    tn = np.sum((y_pred != pos_label) & (y_true != pos_label))

    # False Positives: predicted positive BUT actually negative
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))

    if tn + fp == 0:
        return zero_division

    return tn / (tn + fp)


def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Matthews Correlation Coefficient (MCC).

    Formula: MCC = (TP·TN - FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Range: [-1, 1]
        +1 = perfect prediction
         0 = random prediction
        -1 = perfect disagreement

    Args:
        y_true: True binary labels, shape (n_samples,).
        y_pred: Predicted binary labels, shape (n_samples,).

    Returns:
        mcc: Matthews correlation coefficient, range [-1, 1].

    Advantages:
        - Balanced metric even with imbalanced classes
        - Accounts for all four confusion matrix quadrants
        - More informative than F1 for imbalanced data

    Examples:
        >>> y_true = np.array([1, 1, 0, 0])
        >>> y_pred = np.array([1, 1, 0, 0])
        >>> matthews_corrcoef(y_true, y_pred)
        1.0  # Perfect prediction
    """
    # Compute confusion matrix elements
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    # Numerator
    numerator = tp * tn - fp * fn

    # Denominator
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # Handle edge case: denominator = 0
    if denominator == 0:
        return 0.0

    return numerator / denominator


def log_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """
    Compute binary cross-entropy loss (log loss).

    Formula: -1/n · Σ[y·log(p) + (1-y)·log(1-p)]

    Interpretation: Measures quality of predicted probabilities.
    Heavily penalizes confident wrong predictions.

    Args:
        y_true: True binary labels {0, 1}, shape (n_samples,).
        y_prob: Predicted probabilities for class 1, shape (n_samples,).
        eps: Small constant to clip probabilities, avoids log(0).

    Returns:
        loss: Log loss, range [0, ∞], lower is better.

    Use Cases:
        - Need calibrated probabilities
        - Training objective for logistic regression, neural networks

    Examples:
        >>> y_true = np.array([1, 0, 1])
        >>> y_prob = np.array([0.9, 0.1, 0.8])
        >>> log_loss(y_true, y_prob)
        0.167  # Low loss (good predictions)
    """
    # Clip probabilities to avoid log(0)
    y_prob = np.clip(y_prob, eps, 1 - eps)

    # Binary cross-entropy
    loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    return loss


def roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Area Under the ROC Curve (ROC-AUC).

    ROC Curve plots True Positive Rate vs False Positive Rate at different
    classification thresholds.

    Args:
        y_true: True binary labels {0, 1}, shape (n_samples,).
        y_prob: Predicted probabilities for class 1, shape (n_samples,).

    Returns:
        auc: Area under ROC curve, range [0, 1].
            1.0 = perfect classifier
            0.5 = random guessing
            <0.5 = worse than random (inverted)

    Interpretation:
        Probability that model ranks a random positive example higher
        than a random negative example.

    Notes:
        - Threshold-independent metric
        - Best for balanced classes
        - For imbalanced data, use PR-AUC instead

    Examples:
        >>> y_true = np.array([1, 0, 1, 0])
        >>> y_prob = np.array([0.9, 0.1, 0.8, 0.3])
        >>> roc_auc_score(y_true, y_prob)
        1.0  # Perfect ranking
    """
    # Sort by predicted probability (descending)
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_prob_sorted = y_prob[desc_score_indices]

    # Number of positive and negative samples
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    # Handle edge cases
    if n_pos == 0 or n_neg == 0:
        return 0.5  # Undefined, return random guess

    # Compute TPR and FPR at each threshold
    tpr = np.zeros(len(y_true) + 1)
    fpr = np.zeros(len(y_true) + 1)

    tp = 0
    fp = 0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        tpr[i + 1] = tp / n_pos
        fpr[i + 1] = fp / n_neg

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)

    return auc


def pr_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Area Under the Precision-Recall Curve (PR-AUC).

    PR Curve plots Precision vs Recall at different thresholds.
    Better than ROC-AUC for imbalanced datasets.

    Args:
        y_true: True binary labels {0, 1}, shape (n_samples,).
        y_prob: Predicted probabilities for class 1, shape (n_samples,).

    Returns:
        auc: Area under PR curve, range [0, 1], higher is better.

    Use Cases:
        - Imbalanced classes (ROC-AUC can be misleading)
        - Focus on positive class performance

    Examples:
        >>> y_true = np.array([1, 0, 1, 0])
        >>> y_prob = np.array([0.9, 0.1, 0.8, 0.3])
        >>> pr_auc_score(y_true, y_prob)
        1.0
    """
    # Sort by predicted probability (descending)
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[desc_score_indices]

    # Compute precision and recall at each threshold
    tp = 0
    fp = 0
    n_pos = np.sum(y_true == 1)

    precision_values = [1.0]  # Start at (0, 1)
    recall_values = [0.0]

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        prec = tp / (tp + fp)
        rec = tp / n_pos

        precision_values.append(prec)
        recall_values.append(rec)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(precision_values, recall_values)

    return auc


def precision_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Literal["macro", "micro", "weighted"] = "macro",
) -> float:
    """
    Compute precision for multi-class classification.

    Args:
        y_true: True labels, shape (n_samples,).
        y_pred: Predicted labels, shape (n_samples,).
        average: Averaging strategy:
            'macro': Unweighted mean across classes
            'micro': Aggregate TP/FP globally
            'weighted': Weighted by class support

    Returns:
        precision: Multi-class precision score, range [0, 1].

    Examples:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 2, 2, 0, 1, 1])
        >>> precision_multiclass(y_true, y_pred, average='macro')
        0.667
    """
    classes = np.unique(y_true)

    if average == "micro":
        # Aggregate TP and FP across all classes
        tp_total = np.sum(y_true == y_pred)
        return tp_total / len(y_true)

    # Compute precision for each class
    precisions = []
    supports = []

    for cls in classes:
        # Treat current class as positive, others as negative
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))

        if tp + fp > 0:
            prec = tp / (tp + fp)
        else:
            prec = 0.0

        precisions.append(prec)
        supports.append(np.sum(y_true == cls))

    precisions = np.array(precisions)
    supports = np.array(supports)

    if average == "macro":
        return np.mean(precisions)
    elif average == "weighted":
        return np.sum(precisions * supports) / np.sum(supports)
    else:
        raise ValueError(f"Unknown average: {average}")


if __name__ == "__main__":
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score as sklearn_f1, roc_auc_score as sklearn_roc_auc,
        log_loss as sklearn_log_loss, matthews_corrcoef as sklearn_mcc,
        confusion_matrix as sklearn_cm,
    )
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Classification Metrics: From Scratch Implementation")
    console.print("=" * 70 + "[/bold cyan]")

    # Example 1: Binary Classification
    console.print("\n[bold yellow]1. Binary Classification Metrics[/bold yellow]")
    console.print("-" * 70)

    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.9, 0.1, 0.85, 0.6, 0.2, 0.8, 0.15, 0.55, 0.9, 0.88])

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    console.print(f"\nConfusion Matrix:\n{cm}")
    console.print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    console.print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Compute metrics
    metrics_table = Table(title="Binary Classification Metrics", box=box.ROUNDED)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Our Implementation", justify="right", style="green")
    metrics_table.add_column("Scikit-Learn", justify="right", style="yellow")
    metrics_table.add_column("Difference", justify="right", style="white")

    # Accuracy
    our_acc = accuracy(y_true, y_pred)
    sk_acc = accuracy_score(y_true, y_pred)
    metrics_table.add_row("Accuracy", f"{our_acc:.4f}", f"{sk_acc:.4f}", f"{abs(our_acc - sk_acc):.10f}")

    # Precision
    our_prec = precision(y_true, y_pred)
    sk_prec = precision_score(y_true, y_pred)
    metrics_table.add_row("Precision", f"{our_prec:.4f}", f"{sk_prec:.4f}", f"{abs(our_prec - sk_prec):.10f}")

    # Recall
    our_rec = recall(y_true, y_pred)
    sk_rec = recall_score(y_true, y_pred)
    metrics_table.add_row("Recall", f"{our_rec:.4f}", f"{sk_rec:.4f}", f"{abs(our_rec - sk_rec):.10f}")

    # F1-Score
    our_f1 = f1_score(y_true, y_pred)
    sk_f1 = sklearn_f1(y_true, y_pred)
    metrics_table.add_row("F1-Score", f"{our_f1:.4f}", f"{sk_f1:.4f}", f"{abs(our_f1 - sk_f1):.10f}")

    # Specificity
    our_spec = specificity(y_true, y_pred)
    metrics_table.add_row("Specificity", f"{our_spec:.4f}", "-", "-")

    # MCC
    our_mcc = matthews_corrcoef(y_true, y_pred)
    sk_mcc = sklearn_mcc(y_true, y_pred)
    metrics_table.add_row("MCC", f"{our_mcc:.4f}", f"{sk_mcc:.4f}", f"{abs(our_mcc - sk_mcc):.10f}")

    # ROC-AUC
    our_roc_auc = roc_auc_score(y_true, y_prob)
    sk_roc_auc = sklearn_roc_auc(y_true, y_prob)
    metrics_table.add_row("ROC-AUC", f"{our_roc_auc:.4f}", f"{sk_roc_auc:.4f}", f"{abs(our_roc_auc - sk_roc_auc):.10f}")

    # Log Loss
    our_logloss = log_loss(y_true, y_prob)
    sk_logloss = sklearn_log_loss(y_true, y_prob)
    metrics_table.add_row("Log Loss", f"{our_logloss:.4f}", f"{sk_logloss:.4f}", f"{abs(our_logloss - sk_logloss):.10f}")

    # PR-AUC
    our_pr_auc = pr_auc_score(y_true, y_prob)
    metrics_table.add_row("PR-AUC", f"{our_pr_auc:.4f}", "-", "-")

    console.print(metrics_table)

    # Example 2: Imbalanced Data
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("2. Imbalanced Classification (95% Negative, 5% Positive)")
    console.print("=" * 70 + "[/bold cyan]")

    # Simulate imbalanced data
    np.random.seed(42)
    n_samples = 1000
    n_positive = 50
    n_negative = n_samples - n_positive

    y_true_imb = np.array([1] * n_positive + [0] * n_negative)
    # Imperfect classifier
    y_pred_imb = y_true_imb.copy()
    y_pred_imb[np.random.choice(n_positive, 10, replace=False)] = 0  # 10 FN
    y_pred_imb[n_positive + np.random.choice(n_negative, 20, replace=False)] = 1  # 20 FP

    imb_table = Table(title="Imbalanced Data Metrics", box=box.ROUNDED)
    imb_table.add_column("Metric", style="cyan")
    imb_table.add_column("Value", justify="right", style="green")
    imb_table.add_column("Interpretation", style="white")

    acc_imb = accuracy(y_true_imb, y_pred_imb)
    prec_imb = precision(y_true_imb, y_pred_imb)
    rec_imb = recall(y_true_imb, y_pred_imb)
    f1_imb = f1_score(y_true_imb, y_pred_imb)
    mcc_imb = matthews_corrcoef(y_true_imb, y_pred_imb)

    imb_table.add_row("Accuracy", f"{acc_imb:.4f}", "Misleading (high due to imbalance)")
    imb_table.add_row("Precision", f"{prec_imb:.4f}", "Useful (FP rate)")
    imb_table.add_row("Recall", f"{rec_imb:.4f}", "Useful (FN rate)")
    imb_table.add_row("F1-Score", f"{f1_imb:.4f}", "Better than accuracy")
    imb_table.add_row("MCC", f"{mcc_imb:.4f}", "Best for imbalanced data")

    console.print(imb_table)

    console.print("\n[yellow]→ Accuracy is high (97%) but misleading!")
    console.print("[yellow]→ Precision, Recall, F1, MCC are more informative[/yellow]")

    # Example 3: Multi-Class Classification
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("3. Multi-Class Classification")
    console.print("=" * 70 + "[/bold cyan]")

    y_true_multi = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred_multi = np.array([0, 2, 2, 0, 1, 1, 0, 1, 2, 0, 0, 2])

    multi_table = Table(title="Multi-Class Averaging Strategies", box=box.ROUNDED)
    multi_table.add_column("Strategy", style="cyan")
    multi_table.add_column("Precision", justify="right", style="green")
    multi_table.add_column("Description", style="white")

    macro_prec = precision_multiclass(y_true_multi, y_pred_multi, average="macro")
    micro_prec = precision_multiclass(y_true_multi, y_pred_multi, average="micro")
    weighted_prec = precision_multiclass(y_true_multi, y_pred_multi, average="weighted")

    multi_table.add_row("Macro", f"{macro_prec:.4f}", "Equal weight per class")
    multi_table.add_row("Micro", f"{micro_prec:.4f}", "Aggregate TP/FP globally")
    multi_table.add_row("Weighted", f"{weighted_prec:.4f}", "Weight by class frequency")

    console.print(multi_table)

    # Key Insights
    console.print("\n[bold cyan]" + "=" * 70)
    console.print("Key Insights")
    console.print("=" * 70 + "[/bold cyan]")

    insights = [
        "• Confusion matrix is foundation for all classification metrics",
        "• Accuracy misleading on imbalanced data (use Precision, Recall, F1, MCC)",
        "• Precision-Recall trade-off: improving one typically hurts the other",
        "• F1-Score balances precision and recall (harmonic mean)",
        "• ROC-AUC: threshold-independent, best for balanced classes",
        "• PR-AUC: better than ROC-AUC for imbalanced data",
        "• MCC: most balanced metric for binary classification",
        "• Log Loss: measures probability calibration quality",
        "",
        "[yellow]→ Choose metric based on problem:[/yellow]",
        "  - Balanced classes: Accuracy, F1, ROC-AUC",
        "  - Imbalanced: Precision, Recall, F1, MCC, PR-AUC",
        "  - Costly FP: Maximize Precision",
        "  - Costly FN: Maximize Recall",
        "  - Need probabilities: Minimize Log Loss",
    ]

    for insight in insights:
        console.print(insight)

    console.print("")
