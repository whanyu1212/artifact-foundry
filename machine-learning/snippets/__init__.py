"""
Machine Learning Code Snippets
==============================
Clean, educational implementations of ML algorithms.

Organized by algorithm/topic:
    - decision_trees: CART algorithm for classification and regression
    - tree_ensembles: Bagging, Random Forest, Gradient Boosting
"""

# Import decision tree components for convenience
from .decision_trees import (
    Node,
    TreeMetrics,
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

# Import ensemble methods from tree_ensembles
from .tree_ensembles import (
    BaggingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)

# Public API
__all__ = [
    # Decision Trees
    'Node',
    'TreeMetrics',
    'BaseDecisionTree',
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    # Tree Ensembles
    'BaggingClassifier',
    'RandomForestClassifier',
    'GradientBoostingClassifier',
]
