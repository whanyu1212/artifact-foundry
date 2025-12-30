---
paths: ["**/*.py", "!**/test_*.py", "!**/*_test.py"]
---

# Python Snippet Standards

All Python code snippets in this repository must follow these standards. The goal is creating educational, archival-quality code that teaches concepts clearly.

## Core Principles

- **Educational First**: Code teaches concepts, not just solves problems
- **Self-Contained**: Each snippet should work standalone with minimal dependencies
- **Well-Documented**: Future you should understand the "why" immediately
- **Tested**: All code must be verified correct with tests
- **Type-Safe**: Use type hints to clarify interfaces and aid learning

---

## 1. Documentation Standards

### 1.1 Module-Level Docstrings

Every Python file must start with a module-level docstring explaining its purpose.

**Format:**
```python
"""
[Title] - [Brief description]
[Decorative separator line]
[1-3 sentences explaining what this module implements and key concepts]
"""
```

**Example:**
```python
"""
Decision Tree Classifier - Classification using CART algorithm
==============================================================
Uses information gain (Gini or Entropy) to find optimal splits.
Predicts the most common class in each leaf node.
"""
```

**Requirements:**
- Title should be descriptive and specific
- Separator line uses `=` symbols
- Explain the core algorithm/concept
- Mention key implementation choices

---

### 1.2 Class Docstrings (Google Style)

Classes must have comprehensive docstrings following Google style with these sections:

**Required Sections:**
1. **Summary**: 1-2 sentence overview
2. **Detailed explanation**: Key concepts, algorithms, formulas
3. **Args**: All constructor parameters with types and defaults
4. **Attributes**: Instance attributes set during methods (e.g., `fit()`)
5. **Example**: Concrete usage code

**Format:**
```python
class DecisionTreeClassifier(BaseDecisionTree):
    """
    [One-line summary of what this class does].

    [Detailed explanation with key concepts, algorithms, or formulas.
    Use multiple paragraphs if needed. Include mathematical notation
    when it helps understanding.]

    Impurity measures:
        - Option 1: [Brief explanation]
        - Option 2: [Brief explanation]

    Args:
        param1 (type): Description with default value (default: value)
        param2 (type): Description, None = meaning (default: None)
        param3 (type): Description (default: value)

    Attributes (set during fit):
        attr1_ (type): Description of what this stores
        attr2_ (type): Description
        attr3 (type): Description (if not set during fit)

    Example:
        >>> clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
        >>> clf.fit(X_train, y_train)
        >>> predictions = clf.predict(X_test)
        >>> print(predictions)  # [0, 1, 1, 0, ...]
    """
```

**Key Details:**
- **Args**: Include type, description, and default value explicitly
- **Attributes**: Use `_` suffix for fitted attributes (scikit-learn convention)
- **Example**: Must be runnable code showing typical usage
- **Formulas**: Use ASCII math notation when helpful (`Σ`, `²`, etc.)

---

### 1.3 Function/Method Docstrings (Google Style)

All public methods and complex private methods must be documented.

**Required Sections:**
1. **Summary**: One-line description (what it does)
2. **Detailed explanation**: Algorithm, formulas, or key concepts (if non-trivial)
3. **Args**: All parameters with types and descriptions
4. **Returns**: Type and detailed description of return value
5. **Raises**: Any exceptions raised (if applicable)
6. **Example**: Concrete example with inputs/outputs (for complex methods)

**Format:**
```python
def _calculate_impurity(self, y: np.ndarray) -> float:
    """
    Calculate impurity of a node using Gini or Entropy.

    Gini Impurity:
        Gini = 1 - Σ(p_i²)
        where p_i = proportion of class i

    Entropy:
        H = -Σ(p_i × log₂(p_i))
        where p_i = proportion of class i

    Args:
        y (np.ndarray): Class labels at this node

    Returns:
        float: Impurity value
            - Gini: [0, 1-1/n_classes], 0 = pure
            - Entropy: [0, log₂(n_classes)], 0 = pure

    Example:
        >>> y = np.array([0, 0, 1, 1, 1])  # 2 class-0, 3 class-1
        >>> # Gini = 1 - (0.4² + 0.6²) = 1 - 0.52 = 0.48
        >>> # Entropy = -(0.4×log₂(0.4) + 0.6×log₂(0.6)) ≈ 0.97
    ```

**Key Details:**
- **Summary**: Action verb (Calculate, Build, Predict, etc.)
- **Mathematical formulas**: Include when they clarify the concept
- **Returns**: Describe the value, range, and meaning
- **Example**: Show concrete inputs, calculations, and expected outputs
- Omit sections that don't apply (e.g., no Raises if nothing raised)

**Simple Methods** (getters, trivial operations):
```python
def get_depth(self) -> int:
    """Return the maximum depth of the tree."""
    return self._depth
```

Simple one-liners are acceptable for obvious methods.

---

### 1.4 Example Section Guidelines

Examples in docstrings must be:

1. **Runnable**: Code should work if copied (imports may be assumed)
2. **Concrete**: Use actual values, not abstract X/y
3. **Educational**: Show inputs, process, and outputs
4. **Commented**: Explain non-obvious parts inline

**Good Example:**
```python
Example:
    >>> y = np.array([0, 0, 1, 1, 1])  # 2 class-0, 3 class-1
    >>> # Gini = 1 - (0.4² + 0.6²) = 1 - 0.52 = 0.48
    >>> impurity = self._calculate_impurity(y)
    >>> print(f"{impurity:.2f}")  # 0.48
```

**Bad Example:**
```python
Example:
    >>> result = self.method(data)  # Too abstract, no insight
```

---

## 2. Type Annotations

### 2.1 Required Annotations

Type hints are **required** for:
- All function/method parameters (except `self`, `cls`)
- All function/method return values
- Class attributes in `__init__` (if type not obvious from assignment)

**Good:**
```python
def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
    """Build classifier from training data."""
    self.n_classes_: int = len(np.unique(y))
    return self

def _calculate_split_gain(
    self,
    y_parent: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray
) -> float:
    """Calculate information gain from a split."""
    ...
```

**When to use `Optional`:**
```python
from typing import Optional

def __init__(
    self,
    max_depth: Optional[int] = None,  # None = unlimited
    criterion: str = "gini"
):
    ...
```

### 2.2 Common Type Patterns

```python
from typing import Optional, List, Dict, Tuple, Union, Any

# NumPy arrays
import numpy as np
def process(X: np.ndarray) -> np.ndarray:
    ...

# Optional values
def __init__(self, max_depth: Optional[int] = None):
    ...

# Multiple return values
def split() -> Tuple[np.ndarray, np.ndarray]:
    ...

# Union types (use sparingly)
def predict(X: Union[np.ndarray, List[float]]) -> np.ndarray:
    ...

# Return self for method chaining
def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
    ...
```

### 2.3 When to Skip Type Hints

Skip type hints only for:
- Very obvious assignments: `count = 0`, `items = []`
- Internal loop variables: `for i in range(n):`
- Lambda functions (unless it aids clarity)

---

## 3. Code Structure and Style

### 3.1 Imports

**Order:**
1. Standard library imports
2. Third-party imports (numpy, pandas, etc.)
3. Local/relative imports

**Style:**
```python
# Standard library
import math
from collections import Counter
from typing import Optional, List

# Third-party
import numpy as np
import pandas as pd

# Local
from .base_tree import BaseDecisionTree
from .tree_metrics import TreeMetrics
```

**Rules:**
- Group by category with blank line between groups
- Sort alphabetically within each group
- Use `from X import Y` for specific items
- Use `import X` for modules used frequently
- Avoid `import *` (except in `__init__.py` if necessary)

### 3.2 Code Formatting

Follow **PEP 8** with these specifics:

- **Line length**: 88 characters (Black formatter default)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Use double quotes `"` for strings (except avoid escaping)
- **Trailing commas**: Use in multi-line structures

**Multi-line function signatures:**
```python
def __init__(
    self,
    criterion: str = "gini",
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
):
    ...
```

**Multi-line function calls:**
```python
return TreeMetrics.information_gain(
    parent=y_parent.tolist(),
    left=y_left.tolist(),
    right=y_right.tolist(),
    mode=self.criterion,
)
```

### 3.3 Naming Conventions

- **Classes**: `PascalCase` - `DecisionTreeClassifier`
- **Functions/Methods**: `snake_case` - `calculate_impurity`
- **Private methods**: `_leading_underscore` - `_calculate_leaf_value`
- **Constants**: `UPPER_SNAKE_CASE` - `MAX_ITERATIONS`
- **Variables**: `snake_case` - `n_samples`, `y_pred`

**Scikit-learn conventions** (when building ML algorithms):
- Fitted attributes: `trailing_underscore_` - `n_classes_`, `root_`
- Parameters: no underscore - `max_depth`, `criterion`

---

## 4. Comments

### 4.1 Inline Comments Philosophy

Comments should explain **WHY**, not **WHAT**.

**Good comments:**
```python
# Counter.most_common(1) returns list: [(label, count)]
# Extract just the label with [0][0]
return Counter(y.tolist()).most_common(1)[0][0]

# MAE uses median (more robust to outliers)
return float(np.median(y))

# Handle edge case: all y values are the same
if ss_tot == 0:
    return 1.0 if ss_res == 0 else 0.0
```

**Bad comments:**
```python
# Calculate the mean  ← Obvious from code
mean = np.mean(y)

# Loop through samples  ← Obvious from code
for sample in samples:
    ...
```

### 4.2 When to Add Comments

**Always comment:**
- Non-obvious algorithms or formulas
- Edge cases and special handling
- Performance optimizations that sacrifice clarity
- References to papers/resources for algorithms
- Limitations or known issues
- Why a particular approach was chosen over alternatives

**Example:**
```python
def _find_best_split(self, X: np.ndarray, y: np.ndarray):
    """Find the best feature and threshold to split on."""

    # We use a greedy approach (not globally optimal)
    # Testing all possible thresholds for each feature
    # Time complexity: O(n_features × n_samples × log(n_samples))

    for feature_idx in range(n_features):
        # Use unique values as candidate thresholds
        # Avoids testing thresholds that produce identical splits
        thresholds = np.unique(X[:, feature_idx])
        ...
```

### 4.3 Theory/Formula Comments

For educational code, include relevant theory in comments:

```python
def _calculate_leaf_value(self, y: np.ndarray) -> float:
    """Calculate prediction for a leaf node."""

    if self.criterion == 'mae':
        # Theory: The median minimizes the sum of absolute deviations
        # Proof: https://en.wikipedia.org/wiki/Median#Optimality_property
        return float(np.median(y))
    else:
        # Theory: The mean minimizes the sum of squared deviations
        # This is why MSE uses mean for leaf predictions
        return float(np.mean(y))
```

---

## 5. Testing Requirements

### 5.1 Test Coverage

Every snippet must have corresponding tests in `tests/[topic]/test_*.py`.

**Minimum test coverage:**
- Basic functionality (happy path)
- Edge cases (empty inputs, single values, etc.)
- Error conditions (invalid inputs)
- Mathematical correctness (known examples with expected outputs)

### 5.2 Test Style

```python
import pytest
import numpy as np
from machine_learning.snippets.decision_trees import DecisionTreeClassifier


def test_classifier_basic_fit():
    """Test basic fitting and prediction."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    predictions = clf.predict(X)
    assert predictions.shape == (4,)
    assert all(p in [0, 1] for p in predictions)


def test_gini_impurity():
    """Test Gini calculation with known example."""
    y = np.array([0, 0, 1, 1, 1])  # 40% class-0, 60% class-1

    clf = DecisionTreeClassifier(criterion='gini')
    impurity = clf._calculate_impurity(y)

    # Gini = 1 - (0.4² + 0.6²) = 1 - 0.52 = 0.48
    assert abs(impurity - 0.48) < 0.01


def test_invalid_criterion():
    """Test that invalid criterion raises error."""
    with pytest.raises(ValueError, match="criterion must be"):
        DecisionTreeClassifier(criterion='invalid')
```

### 5.3 Test Organization

- One test file per module: `test_decision_tree_classifier.py`
- Group related tests in classes if helpful
- Use descriptive test names: `test_[what]_[condition]_[expected]`
- Include docstrings for complex test scenarios

---

## 6. Error Handling

### 6.1 Input Validation

Validate inputs in public methods with clear error messages:

```python
def __init__(self, criterion: str = "gini", max_depth: Optional[int] = None):
    # Validate criterion
    if criterion not in ["gini", "entropy"]:
        raise ValueError(
            f"criterion must be 'gini' or 'entropy', got '{criterion}'"
        )

    # Validate max_depth if provided
    if max_depth is not None and max_depth < 1:
        raise ValueError(
            f"max_depth must be >= 1 or None, got {max_depth}"
        )

    self.criterion = criterion
    self.max_depth = max_depth
```

### 6.2 Error Messages

Good error messages include:
- What is wrong
- What was expected
- What was received

**Good:**
```python
raise ValueError(
    f"X must have shape (n_samples, {self.n_features_}), "
    f"got shape {X.shape}"
)
```

**Bad:**
```python
raise ValueError("Invalid input")  # Too vague
```

### 6.3 When to Validate

- **Public API**: Always validate inputs
- **Private methods**: Can assume inputs are valid (validated by caller)
- **Performance-critical code**: Document assumptions rather than validate

---

## 7. Code Organization Best Practices

### 7.1 Class Structure Order

Organize class members in this order:

1. Class-level constants
2. `__init__` method
3. Public methods (roughly in order of typical usage)
4. Private methods (prefixed with `_`)
5. Magic methods (`__str__`, `__repr__`, etc.) at the end

**Example:**
```python
class DecisionTree:
    # 1. Class constants
    MAX_DEPTH_DEFAULT = 10

    # 2. Constructor
    def __init__(self, ...):
        ...

    # 3. Public methods (in usage order)
    def fit(self, X, y):
        ...

    def predict(self, X):
        ...

    def score(self, X, y):
        ...

    # 4. Private methods
    def _build_tree(self, X, y):
        ...

    def _find_best_split(self, X, y):
        ...

    # 5. Magic methods
    def __str__(self):
        ...
```

### 7.2 Function Length

- **Target**: 20-30 lines max per function
- **Extract** complex logic into helper methods
- **Each function** should do one thing well

If a function is getting long, ask:
- Can this be split into smaller functions?
- Is there repeated logic to extract?
- Can a complex expression become a well-named helper function?

### 7.3 Complexity Management

Break down complex operations:

**Before:**
```python
def predict(self, X):
    results = []
    for x in X:
        node = self.root
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        results.append(node.value)
    return np.array(results)
```

**After:**
```python
def predict(self, X: np.ndarray) -> np.ndarray:
    """Predict class labels for samples."""
    return np.array([self._predict_sample(x) for x in X])

def _predict_sample(self, x: np.ndarray) -> int:
    """Traverse tree to predict single sample."""
    node = self.root
    while not node.is_leaf:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value
```

---

## 8. Educational Code Patterns

### 8.1 Show Alternative Approaches

When multiple valid approaches exist, show and explain them:

```python
def calculate_distance(self, x1, x2, method='euclidean'):
    """
    Calculate distance between two points.

    Methods:
        - euclidean: sqrt(Σ(x1 - x2)²) - Most common
        - manhattan: Σ|x1 - x2| - Good for high dimensions
        - cosine: 1 - (x1·x2)/(||x1||·||x2||) - For angle similarity
    """
    if method == 'euclidean':
        return np.sqrt(np.sum((x1 - x2) ** 2))
    elif method == 'manhattan':
        return np.sum(np.abs(x1 - x2))
    elif method == 'cosine':
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
```

### 8.2 Document Limitations

Be explicit about what the code doesn't handle:

```python
class SimpleLinearRegression:
    """
    Simple linear regression using ordinary least squares.

    Limitations:
        - Only supports single feature (univariate)
        - No regularization (may overfit)
        - Assumes linear relationship
        - Sensitive to outliers
        - No handling of missing values

    For production use, consider sklearn.linear_model.LinearRegression
    """
```

### 8.3 Link to Related Concepts

Cross-reference related topics:

```python
class RandomForest(BaseEnsemble):
    """
    Random Forest ensemble of decision trees.

    Key concepts:
        - Bootstrap aggregating (bagging) - See: bagging.py
        - Feature randomization - Reduces correlation between trees
        - Out-of-bag (OOB) estimation - See: model_evaluation.py

    Related implementations:
        - decision_trees/decision_tree_classifier.py - Base learner
        - ensemble/bagging.py - Bagging framework
        - ensemble/gradient_boosting.py - Alternative ensemble method
    """
```

### 8.4 Include References

For algorithms from papers or books:

```python
class AdaBoost:
    """
    AdaBoost (Adaptive Boosting) ensemble method.

    References:
        - Original paper: Freund & Schapire (1997)
          "A Decision-Theoretic Generalization of On-Line Learning"
        - ESLII Section 10.1: "Boosting Methods"
        - https://scikit-learn.org/stable/modules/ensemble.html#adaboost

    Algorithm:
        1. Initialize weights uniformly: w_i = 1/n
        2. For m = 1 to M:
            a. Fit classifier to weighted data
            b. Compute error: err = Σ(w_i × I(y_i ≠ ŷ_i))
            c. Compute alpha: α = log((1-err)/err)
            d. Update weights: w_i *= exp(α × I(y_i ≠ ŷ_i))
        3. Final prediction: sign(Σ(α_m × h_m(x)))
    """
```

---

## 9. Performance and Optimization

### 9.1 Optimize After Correctness

**Priority order:**
1. Correctness (algorithm works)
2. Clarity (code is understandable)
3. Performance (code is fast)

**Only optimize when:**
- Profiling shows a bottleneck
- The optimization doesn't sacrifice clarity significantly
- You document why the optimization is needed

### 9.2 Document Optimizations

```python
def _find_best_split(self, X, y):
    """Find best feature and threshold to split on."""

    # Optimization: Pre-sort feature values once
    # Avoids O(n log n) sort for each threshold test
    # Trade-off: O(n_features × n) memory for O(n_features × n²) time savings
    sorted_indices = np.argsort(X, axis=0)

    for feature_idx in range(n_features):
        # Use pre-sorted indices instead of sorting again
        sorted_y = y[sorted_indices[:, feature_idx]]
        ...
```

### 9.3 Premature Optimization

Avoid premature optimization. Prefer clarity:

**Good (clear):**
```python
def mean(values):
    """Calculate arithmetic mean."""
    return sum(values) / len(values)
```

**Bad (prematurely optimized):**
```python
def mean(values):
    """Calculate arithmetic mean."""
    # Using Kahan summation for numerical stability
    # (Probably overkill for most learning scenarios)
    s = 0.0
    c = 0.0
    for v in values:
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
    return s / len(values)
```

Only add complexity like Kahan summation if you have a specific numerical stability requirement.

---

## 10. File Organization

### 10.1 Directory Structure

```
machine-learning/
├── snippets/
│   ├── decision_trees/
│   │   ├── __init__.py
│   │   ├── base_tree.py          # Shared base class
│   │   ├── decision_tree_classifier.py
│   │   ├── decision_tree_regressor.py
│   │   └── tree_metrics.py       # Utility functions
│   └── ...
├── notes/
│   ├── decision-trees.md
│   └── ...
└── resources.md
```

### 10.2 Module Files

**When to create separate files:**
- Each major class gets its own file
- Shared utilities in separate utility modules
- Base classes in `base_*.py` files
- Keep files under 500 lines (split if larger)

**When to combine in one file:**
- Tightly coupled classes (e.g., Node and Tree)
- Very small helper classes
- Dataclasses or simple containers

### 10.3 `__init__.py` Files

Make imports easy for users:

```python
"""Decision tree implementations."""

from .decision_tree_classifier import DecisionTreeClassifier
from .decision_tree_regressor import DecisionTreeRegressor
from .tree_metrics import TreeMetrics

__all__ = [
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    'TreeMetrics',
]
```

---

## 11. Special Cases

### 11.1 Magic Methods

Document magic methods like regular methods:

```python
def __str__(self) -> str:
    """
    Return human-readable string representation.

    Returns:
        str: String showing tree structure with depth and nodes

    Example:
        >>> tree = DecisionTree(max_depth=3)
        >>> print(tree)
        DecisionTree(max_depth=3, n_nodes=15, n_leaves=8)
    """
    return f"{self.__class__.__name__}(...)"
```

### 11.2 Property Decorators

Use properties for computed attributes:

```python
@property
def n_nodes(self) -> int:
    """
    Total number of nodes in the tree.

    Returns:
        int: Count of all nodes (internal + leaves)
    """
    return self._count_nodes(self.root)

@property
def depth(self) -> int:
    """
    Maximum depth of the tree.

    Returns:
        int: Depth from root to deepest leaf (root has depth 0)
    """
    return self._compute_depth(self.root)
```

### 11.3 Dataclasses

For simple data containers, use dataclasses:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Node:
    """
    Node in a decision tree.

    Attributes:
        feature (Optional[int]): Feature index to split on (None for leaf)
        threshold (Optional[float]): Threshold value (None for leaf)
        left (Optional[Node]): Left child (feature <= threshold)
        right (Optional[Node]): Right child (feature > threshold)
        value (Optional[float]): Prediction value (only for leaf nodes)
        is_leaf (bool): Whether this is a leaf node
    """
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[float] = None
    is_leaf: bool = False
```

---

## 12. Checklist for Code Review

Before committing a snippet, verify:

- [ ] Module-level docstring present and descriptive
- [ ] All classes have comprehensive Google-style docstrings
- [ ] All public methods have docstrings with Args/Returns/Examples
- [ ] Type hints on all parameters and return values
- [ ] Inline comments explain "why" not "what"
- [ ] Examples in docstrings are concrete and runnable
- [ ] Mathematical formulas included where relevant
- [ ] Code follows PEP 8 (use `black` formatter)
- [ ] Tests exist and pass
- [ ] Input validation with clear error messages
- [ ] No obvious performance issues
- [ ] Cross-references to related concepts where applicable
- [ ] Limitations documented
- [ ] Educational value is clear (teaches a concept)

---

## 13. Tools and Automation

### 13.1 Recommended Tools

```bash
# Code formatting
pip install black isort

# Linting
pip install ruff  # Fast, combines multiple linters

# Type checking
pip install mypy

# Testing
pip install pytest pytest-cov
```

### 13.2 Pre-commit Checks

Run before committing:

```bash
# Format code
black .
isort .

# Type check
mypy snippets/

# Run tests
pytest tests/ -v --cov=snippets
```

### 13.3 Editor Configuration

**VS Code settings** (`.vscode/settings.json`):
```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "editor.formatOnSave": true,
  "editor.rulers": [88]
}
```

---

## 14. Example, Comparison, and Benchmarking Scripts

### 14.1 When This Applies

Scripts that serve these purposes must use `rich` for terminal formatting:

- **Example scripts**: Demonstrate usage of implementations (e.g., `examples_decision_tree.py`)
- **Comparison scripts**: Compare implementations (ours vs. sklearn, algorithm A vs. B)
- **Benchmarking scripts**: Show performance metrics or model comparisons
- **Demo scripts**: Interactive demonstrations of concepts

### 14.2 Required: Rich Library for Formatting

**Why use Rich:**
- Professional, readable terminal output
- Tables for structured data comparison
- Panels for visual organization
- Color coding for better comprehension
- Tree visualizations for hierarchical data
- Consistent formatting across all examples

**Installation:**
```bash
pip install rich
```

### 14.3 Core Rich Components to Use

Import the essential components:

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich import box

console = Console()
```

**When to use each component:**

| Component | Use Case | Example |
|-----------|----------|---------|
| `Console` | All output (replaces `print`) | `console.print("[bold]Training...[/bold]")` |
| `Table` | Structured data, metrics, comparisons | Results, hyperparameters, feature info |
| `Panel` | Section headers, warnings, key insights | `Panel.fit("[bold]Results[/bold]")` |
| `Tree` | Hierarchical structures | Decision trees, directory structures |
| `Text` | Styled text objects | Titles, formatted output |
| `box` | Table/panel borders | `box.ROUNDED`, `box.SIMPLE` |

### 14.4 Standard Patterns

**Pattern 1: Section Headers**
```python
# Use panels for major sections
console.print(Panel.fit(
    "[bold cyan]COMPARISON RESULTS[/bold cyan]",
    border_style="cyan"
))

# Use rules for subsections
console.print()
console.rule("[bold cyan]Detailed Analysis[/bold cyan]", style="cyan")
console.print()
```

**Pattern 2: Comparison Tables**
```python
# Compare implementations side-by-side
table = Table(title="Performance Comparison", box=box.ROUNDED)
table.add_column("Metric", style="cyan", width=25)
table.add_column("Our Impl.", justify="right", style="green")
table.add_column("Sklearn", justify="right", style="blue")
table.add_column("Difference", justify="right", style="magenta")

table.add_row(
    "Accuracy",
    f"{our_acc:.4f}",
    f"{sk_acc:.4f}",
    f"{abs(our_acc - sk_acc):.4f}"
)
console.print(table)
```

**Pattern 3: Dataset Information**
```python
# Show dataset info clearly
table = Table(title="Dataset Information", box=box.ROUNDED, show_header=False)
table.add_column("Property", style="cyan", width=20)
table.add_column("Value", style="magenta")

table.add_row("Dataset", "Iris")
table.add_row("Training samples", str(X_train.shape[0]))
table.add_row("Features", str(X_train.shape[1]))
console.print(table)
```

**Pattern 4: Progress and Status Messages**
```python
# Clear status updates
console.print("\n[bold yellow]Training Model...[/bold yellow]")
# ... training code ...
console.print("[green]✓ Training complete[/green]")

# Use dim for supplementary info
console.print("[dim]Using 100 estimators with max_depth=5[/dim]")
```

**Pattern 5: Insights and Warnings**
```python
# Highlight key insights
insight = Panel(
    "[cyan]💡 Key Insight:[/cyan]\n\n"
    "• Random Forest uses [green]feature randomness[/green]\n"
    "• This decorrelates trees and improves generalization\n"
    "• Typical setting: max_features='sqrt'",
    title="[bold]Understanding Random Forests[/bold]",
    border_style="blue",
    box=box.ROUNDED
)
console.print(insight)

# Warnings and alerts
warning = Panel(
    "[yellow]⚠️  Warning:[/yellow] Perfect training accuracy often means overfitting!",
    title="[bold red]Overfitting Alert[/bold red]",
    border_style="red"
)
console.print(warning)
```

**Pattern 6: Tree Visualization**
```python
# For decision trees or hierarchical structures
def build_rich_tree(node, label="Root"):
    """Build rich Tree visualization."""
    if node.is_leaf():
        node_label = f"[green]{label}[/green] → predict {node.value}"
    else:
        node_label = f"[blue]{label}[/blue] X[{node.feature}] ≤ {node.threshold:.2f}"

    tree = Tree(node_label)
    if not node.is_leaf():
        tree.add(build_rich_tree(node.left, "Left"))
        tree.add(build_rich_tree(node.right, "Right"))
    return tree

tree_viz = build_rich_tree(clf.root)
console.print(tree_viz)
```

### 14.5 Color Scheme Guidelines

Use consistent semantic colors:

- **Cyan/Blue**: Headers, section titles, feature names
- **Green**: Our implementations, success messages, positive values
- **Yellow**: Status messages, warnings (non-critical)
- **Magenta/Purple**: Differences, special metrics
- **Red**: Errors, critical warnings
- **Dim/Gray**: Supplementary info, less important details

```python
# Good: Semantic color usage
console.print("[cyan]Dataset:[/cyan] Breast Cancer")
console.print("[green]✓ Test passed[/green]")
console.print("[yellow]⚠️  High variance detected[/yellow]")
console.print("[red]✗ Test failed[/red]")
```

### 14.6 Complete Example Structure

Every comparison/example script should follow this structure:

```python
"""
[Script Title] - [Purpose]
[Separator]
[Description of what this script demonstrates]

Key demonstrations:
    - Point 1
    - Point 2
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# Import implementations to compare
from .our_implementation import OurModel
from sklearn.model import SklearnModel


def main():
    """Run all demonstrations."""
    # Header
    console.print()
    console.print(Panel.fit(
        "[bold white]SCRIPT TITLE[/bold white]\n"
        "[dim]Brief description[/dim]",
        border_style="bold cyan",
        padding=(1, 2)
    ))
    console.print()

    # Load and display data
    X_train, X_test, y_train, y_test = load_data()

    # Run comparisons
    compare_implementations(X_train, X_test, y_train, y_test)

    # Show insights
    show_insights()

    # Completion
    console.print(Panel.fit(
        "[bold green]✓ DEMONSTRATION COMPLETE[/bold green]",
        border_style="green"
    ))


if __name__ == "__main__":
    main()
```

### 14.7 Reference Examples

See these files for complete examples:

- **Decision trees**: `machine-learning/snippets/decision_trees/examples_decision_tree.py`
  - Shows: Tables, trees, panels, rules, color coding
  - Demonstrates: Classification and regression examples

- **Ensemble comparison**: `machine-learning/snippets/tree_ensembles/ensemble_comparison.py`
  - Shows: Side-by-side comparisons, metrics tables, usage guidelines
  - Demonstrates: Multiple model comparisons with consistent formatting

### 14.8 Checklist for Example Scripts

Before committing an example/comparison/benchmarking script:

- [ ] Uses `rich.console.Console` instead of `print()`
- [ ] Results shown in `Table` with appropriate styling
- [ ] Sections organized with `Panel` or `console.rule()`
- [ ] Color scheme is semantic and consistent
- [ ] Key insights highlighted in colored panels
- [ ] Script has clear header and completion messages
- [ ] Module docstring explains what is demonstrated
- [ ] Output is readable and professionally formatted
- [ ] No plain `print()` statements (use `console.print()`)

**Remember**: Example scripts are often the first thing people run. Make them visually impressive and easy to understand!

---

## Summary

These standards ensure your code snippets serve as an **educational archive** that:

1. **Teaches clearly** through comprehensive documentation
2. **Works correctly** via tests and validation
3. **Remains maintainable** with clean structure and comments
4. **Demonstrates understanding** of concepts, not just syntax
5. **Presents professionally** with rich terminal formatting for examples

When in doubt, prioritize **clarity and educational value** over brevity or cleverness.

**Good snippet indicators:**
- Future you can understand it in 6 months
- Someone learning the concept gains insight
- The code demonstrates understanding, not just functionality
- Examples show both how to use it and why it works
- Example scripts are visually clear and professional

**Remember**: You're building a learning archive, not a production library. The goal is understanding, not optimization.
