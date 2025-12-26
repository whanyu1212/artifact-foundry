"""
Machine Learning Code Snippets
==============================
Clean, educational implementations of ML algorithms.

Organized by algorithm/topic:
    - decision_trees: CART algorithm for classification and regression
"""

# Import decision tree components for convenience
from .decision_trees import (
    Node,
    TreeMetrics,
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

# Public API
__all__ = [
    # Decision Trees
    'Node',
    'TreeMetrics',
    'BaseDecisionTree',
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
]
