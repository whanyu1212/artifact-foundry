"""
Tree-Based Ensemble Methods
============================
Ensemble learning algorithms that combine multiple decision trees.

Modules:
    - bagging: Bootstrap Aggregating for variance reduction
    - random_forest: Bagging + feature randomness
    - gradient_boosting: Sequential boosting with gradient descent
    - ensemble_comparison: Side-by-side comparison with sklearn
"""

from .bagging import BaggingClassifier
from .random_forest import RandomForestClassifier
from .gradient_boosting import GradientBoostingClassifier

__all__ = [
    'BaggingClassifier',
    'RandomForestClassifier',
    'GradientBoostingClassifier',
]
