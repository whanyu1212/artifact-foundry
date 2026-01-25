"""
Feature Importance Techniques Example

This script demonstrates various feature importance techniques on a classification problem
using popular libraries (scikit-learn, shap).

Techniques covered:
1. Impurity-based importance (Random Forest built-in)
2. Permutation Importance (Model-agnostic)
3. SHAP (Shapley Additive exPlanations)

Requirements:
    pip install shap scikit-learn pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from typing import Dict, Any


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """Trains a Random Forest classifier."""
    print("Training Random Forest model...")
    # Using a robust number of estimators and fixing random state for reproducibility
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf


def plot_impurity_importance(
    model: RandomForestClassifier, feature_names: pd.Index
) -> None:
    """
    Plots the specific impurity-based feature importance from the Random Forest.

    Note: Impurity-based feature importance can be misleading for high cardinality features
    (many unique values).
    """
    print("\n--- 1. Impurity-based Feature Importance ---")
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

    # Create a Series for easy sorting
    forest_importances = pd.Series(importances, index=feature_names)

    # Sort and plot top 10
    top_n = forest_importances.sort_values(ascending=False).head(10)

    print("Top 5 Impurity-based Importances:")
    print(top_n.head(5))

    # You would typically rely on a visualization library in a notebook,
    # but here we just print the data or could save a plot.


def calculate_permutation_importance(
    model: Any, X_val: pd.DataFrame, y_val: pd.Series
) -> None:
    """
    Calculates Permutation Importance.

    This is model-agnostic. It shuffles one feature at a time and measures the drop
    in model performance.
    """
    print("\n--- 2. Permutation Importance ---")
    # n_repeats=10 means we shuffle each feature 10 times to get stable results
    result = permutation_importance(
        model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
    )

    sorted_idx = result.importances_mean.argsort()[::-1]  # Descending

    print("Top 5 Permutation Importances (Mean Accuracy Decrease):")
    for i in sorted_idx[:5]:
        print(
            f"{X_val.columns[i]:<30}: {result.importances_mean[i]:.4f} +/- {result.importances_std[i]:.4f}"
        )


def explain_with_shap(model: Any, X_train: pd.DataFrame, X_val: pd.DataFrame) -> None:
    """
    Uses SHAP (SHapley Additive exPlanations) to explain the model.
    SHAP values represent the directional contribution of each feature to the prediction.
    """
    print("\n--- 3. SHAP Values ---")

    # TreeExplainer is optimized for tree-based models
    explainer = shap.TreeExplainer(model)

    # Calculating SHAP values for the validation set
    # check_additivity=False is sometimes needed for complex RFs or if margin of error is slight
    shap_values = explainer.shap_values(X_val)

    # For binary classification, shap_values can be a list of arrays or a 3D array
    # We want the values for the positive class (index 1)
    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    elif len(shap_values.shape) == 3:
        # Shape is (n_samples, n_features, n_classes)
        shap_values_class1 = shap_values[:, :, 1]
    else:
        # Assume it's already the values we want (e.g. regression or 1 output)
        shap_values_class1 = shap_values

    # Global Feature Importance: Mean absolute SHAP value
    # This tells us how much each feature pushes the model's output on average (magnitude)
    mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
    shap_importance = pd.Series(mean_abs_shap, index=X_val.columns).sort_values(
        ascending=False
    )

    print("Top 5 SHAP Importances (Mean |SHAP value|):")
    print(shap_importance.head(5))

    # In a notebook, we would use:
    # shap.summary_plot(shap_values_class1, X_val)


def main():
    # 1. Load Data
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # 2. Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42)

    # 3. Train Model
    rf_model = train_model(X_train, y_train)

    # 4. Impurity-based Importance (Built-in)
    plot_impurity_importance(rf_model, X.columns)

    # 5. Permutation Importance (Model Agnostic)
    calculate_permutation_importance(rf_model, X_val, y_val)

    # 6. SHAP Values (State of the art)
    explain_with_shap(rf_model, X_train, X_val)


if __name__ == "__main__":
    main()
