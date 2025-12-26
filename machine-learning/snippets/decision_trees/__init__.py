"""
Decision Trees - CART Implementation
====================================
Classification and Regression Trees from scratch.

Components:
    - Node: Tree node data structure
    - TreeMetrics: Splitting criteria calculations (Gini, Entropy, MSE, MAE, etc.)
    - BaseDecisionTree: Common CART algorithm logic
    - DecisionTreeClassifier: Classification tree
    - DecisionTreeRegressor: Regression tree

See README.md for usage examples and architecture details.
See ../../notes/decision-trees.md for comprehensive notes.
"""

from .tree_node import Node
from .tree_metrics import TreeMetrics
from .base_tree import BaseDecisionTree
from .decision_tree_classifier import DecisionTreeClassifier
from .decision_tree_regressor import DecisionTreeRegressor

__all__ = [
    'Node',
    'TreeMetrics',
    'BaseDecisionTree',
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
]
