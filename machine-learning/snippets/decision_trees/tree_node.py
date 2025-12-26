"""
Tree Node - Data structure for decision tree nodes
==================================================
Represents both internal nodes (with splits) and leaf nodes (with predictions).
"""

from typing import Optional, Union


class Node:
    """
    Node in a decision tree (both internal and leaf nodes).
    Leaf node is also known as terminal node.


    Internal nodes have:
        - feature_idx: Which feature to split on
        - threshold: Split threshold value
        - left/right: Child nodes

    Leaf nodes have:
        - value: Prediction value (class label or continuous value)

    Metadata (all nodes):
        - impurity: Node impurity measure
        - n_samples: Number of samples at this node
        - depth: Depth in tree (root = 0)
    """

    def __init__(
        self,
        feature_idx: Optional[
            int
        ] = None,  # depending on whether it's a leaf or internal node
        threshold: Optional[
            float
        ] = None,  # depending on whether it's a leaf or internal node
        left: Optional["Node"] = None,  # Left child node or None if leaf
        right: Optional["Node"] = None,  # Right child node or None if leaf
        value: Optional[Union[int, float]] = None,  # Prediction value for leaf nodes
        impurity: Optional[float] = None,  # Node impurity measure
        n_samples: Optional[int] = None,  # Number of samples at this node
        depth: int = 0,  # Depth in tree (root = 0)
    ):
        # Internal node: split information
        self.feature_idx = feature_idx  # Index of feature to split on
        self.threshold = threshold  # Threshold: left <= threshold < right
        self.left = left  # Left child node
        self.right = right  # Right child node

        # Leaf node: prediction value
        self.value = value  # Prediction (class or continuous value)

        # Metadata
        self.impurity = impurity  # Node impurity (gini, entropy, mse, etc.)
        self.n_samples = n_samples  # Number of training samples at node
        self.depth = depth  # Depth in tree (root = 0)

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (has prediction value)."""
        return self.value is not None

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.is_leaf():
            return f"Leaf(value={self.value}, n_samples={self.n_samples})"
        else:
            return (
                f"Node(feature={self.feature_idx}, "
                f"threshold={self.threshold:.2f}, "
                f"n_samples={self.n_samples})"
            )


if __name__ == "__main__":
    # Example usage
    leaf_node = Node(value=1, n_samples=10, impurity=0.0, depth=2)
    internal_node = Node(
        feature_idx=0,  # Index of feature to split on
        threshold=1.5,  # Threshold value for splitting
        left=leaf_node,  # Left child node
        right=None,  # Right child node
        n_samples=20,  # Number of samples at this node
        impurity=0.5,  # Node impurity
        depth=1,  # Depth in tree (root = 0)
    )

    print(internal_node)
    print(leaf_node)
