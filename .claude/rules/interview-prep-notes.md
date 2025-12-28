---
paths: ["interview-prep/**/*.md"]
---

# Interview Prep Notes Standards

Interview prep notes optimize for **speed and recall under pressure**. These are condensed references for pattern recognition, not deep learning.

## Core Principles

- **Concise over comprehensive**: No fluff, just what you need to recall
- **Pattern-first**: Recognize question types instantly
- **Template-driven**: Code scaffolding you can adapt quickly
- **Complexity visible**: Always show time/space upfront
- **Quick lookup**: Should find what you need in < 30 seconds

**Remember:** Deep understanding lives in topic folders. Interview prep is for fast recall and pattern matching.

---

## Organization Strategies

### Recommended: Hybrid Approach

Organize interview prep by **domain + pattern**:

```
interview-prep/
├── algorithms/              # Coding interviews
│   ├── array-patterns.md           # Two pointers, sliding window
│   ├── tree-graph-patterns.md      # DFS, BFS, traversals
│   ├── dynamic-programming.md      # DP patterns
│   └── complexity-cheatsheet.md    # Big-O quick reference
├── machine-learning/        # ML interviews
│   ├── algorithms-cheatsheet.md    # Core ML algorithms
│   ├── metrics-formulas.md         # Precision, recall, AUC
│   └── common-questions.md         # Bias-variance, overfitting
├── system-design/           # System design interviews
│   ├── components-checklist.md     # Load balancers, caches
│   └── ml-systems.md               # ML pipelines, serving
└── sql/                     # Data/analytics interviews
    ├── query-patterns.md
    └── optimization-tricks.md
```

### Alternative: Pure Pattern-Based

For focused coding interview prep:

```
interview-prep/coding/
├── two-pointers.md
├── sliding-window.md
├── binary-search.md
├── backtracking.md
├── dfs-bfs.md
└── dynamic-programming.md
```

**Choose based on:**
- **Hybrid**: Preparing for multiple interview types (SWE, MLE, data)
- **Pattern-based**: Focused on coding interviews only

---

## File Structure

### Title Format

```markdown
# [Pattern/Topic] - [One-line description]

**Use for:** [What types of problems]
**Signal keywords:** [Words in problem that indicate this pattern]
```

**Example:**
```markdown
# Two Pointers - Process array from both ends

**Use for:** Sorted arrays, palindromes, pair problems
**Signal keywords:** "sorted array", "pair sum", "palindrome", "two elements"
```

### Section Order

1. **When to use** (recognition)
2. **Template code** (scaffolding)
3. **Time/Space complexity** (always visible)
4. **Common variations** (twists)
5. **Example problems** (with companies if relevant)
6. **Gotchas** (edge cases, common mistakes)
7. **Deep dive link** (to learning notes)

---

## Template Format

### Pattern Template Structure

```markdown
## [Pattern Name]

**When to recognize:**
- [Signal 1]
- [Signal 2]
- [Signal 3]

**Template:**
```python
def pattern_name(input):
    # Setup
    initialization

    # Core logic
    while/for condition:
        # Pattern-specific logic
        pass

    # Return
    return result
```

**Complexity:** Time O(?) | Space O(?)

**Variations:**
- Variation 1: [Brief description]
- Variation 2: [Brief description]

**Example problems:**
- [LeetCode #123 - Problem Name] (Easy)
- [LeetCode #456 - Problem Name] (Medium)

**Common mistakes:**
- Mistake 1
- Mistake 2

**Deep dive:** See [Topic](../topic/notes/file.md) for theory
```

### Example: Two Pointers

```markdown
## Two Pointers

**When to recognize:**
- Sorted array + find pair/triplet
- Palindrome check
- Reverse/rearrange in-place
- Remove duplicates from sorted array

**Template:**
```python
def two_pointers(arr):
    left, right = 0, len(arr) - 1

    while left < right:
        # Process current state
        if condition_met(arr[left], arr[right]):
            return result

        # Move pointers
        if some_condition:
            left += 1
        else:
            right -= 1

    return default_result
```

**Complexity:** Time O(n) | Space O(1)

**Variations:**
- **Fast/slow pointers**: Cycle detection (Floyd's algorithm)
- **Three pointers**: 3Sum problem
- **Sliding window**: Dynamic window size

**Example problems:**
- [LC 167 - Two Sum II] (Easy) - Sorted array, target sum
- [LC 15 - 3Sum] (Medium) - Find triplets that sum to 0
- [LC 125 - Valid Palindrome] (Easy) - Compare from both ends

**Common mistakes:**
- Off-by-one: Use `left < right` not `left <= right` for most cases
- Forgetting to handle duplicates in 3Sum variants
- Not returning early when condition is met

**Deep dive:** See [Two Pointers Pattern](../../algorithms/notes/array-techniques.md#two-pointers)
```

---

## Cheat Sheet Format

For condensed reference (algorithms, formulas, complexity):

### Table-Based Format

```markdown
# ML Metrics Cheatsheet

| Metric | Formula | When to Use | Range |
|--------|---------|-------------|-------|
| Accuracy | (TP+TN) / Total | Balanced classes | [0, 1] |
| Precision | TP / (TP+FP) | Cost of FP high | [0, 1] |
| Recall | TP / (TP+FN) | Cost of FN high | [0, 1] |
| F1 | 2·P·R / (P+R) | Balance P & R | [0, 1] |

**Quick decision:**
- Spam detection → Precision (don't mark real email as spam)
- Cancer screening → Recall (don't miss cancer cases)
- Balanced problem → F1 or Accuracy
```

### List-Based Format

```markdown
# Tree Traversal Patterns

## DFS (Depth-First Search)

**Inorder (Left, Root, Right):**
```python
def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)
```
**Use:** BST → sorted order | Time: O(n) | Space: O(h)

**Preorder (Root, Left, Right):**
```python
def preorder(root):
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)
```
**Use:** Copy tree, prefix expression | Time: O(n) | Space: O(h)

**Postorder (Left, Right, Root):**
```python
def postorder(root):
    if not root: return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```
**Use:** Delete tree, postfix expression | Time: O(n) | Space: O(h)

## BFS (Breadth-First Search)

**Level-order:**
```python
from collections import deque

def levelorder(root):
    if not root: return []
    queue, result = deque([root]), []

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)

        result.append(level)

    return result
```
**Use:** Level-by-level processing | Time: O(n) | Space: O(w)
```

---

## Complexity Quick Reference

Always include complexity with templates. Use standard notation:

### Notation

```markdown
**Time:** O(n log n)
**Space:** O(1) auxiliary (O(n) if counting output)
```

### Common Complexities Table

```markdown
| Complexity | Operations | Examples |
|------------|------------|----------|
| O(1) | 1-10 | Hash lookup, array access |
| O(log n) | ~20 (n=10⁶) | Binary search, balanced tree |
| O(n) | 10⁶ | Linear scan, two pointers |
| O(n log n) | ~20M | Merge sort, heap operations |
| O(n²) | 10¹² | Nested loops, bubble sort |
| O(2ⁿ) | Huge | Backtracking, subsets |

**Interview rule of thumb:** n ≤ 10⁶, operations ≤ 10⁸
```

---

## Problem Organization

### By Pattern

```markdown
# Sliding Window Pattern

## Fixed-size Window

**[LC 643 - Maximum Average Subarray I]** (Easy)
```python
def findMaxAverage(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum

    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i-k]
        max_sum = max(max_sum, window_sum)

    return max_sum / k
```
**Pattern:** Maintain window sum, slide by +1/-1

## Variable-size Window

**[LC 3 - Longest Substring Without Repeating]** (Medium)
```python
def lengthOfLongestSubstring(s):
    seen = set()
    left = max_len = 0

    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        max_len = max(max_len, right - left + 1)

    return max_len
```
**Pattern:** Expand right, contract left when invalid
```

### By Difficulty/Company (Optional)

Only if targeting specific companies:

```markdown
# Google - Medium Difficulty Trees

- [LC 236 - Lowest Common Ancestor] - DFS with parent tracking
- [LC 297 - Serialize/Deserialize Tree] - Level-order encoding
- [LC 105 - Construct from Preorder/Inorder] - Recursive partition
```

---

## Cross-Referencing

### Link to Deep Learning

Every interview prep note should reference the learning notes:

```markdown
## For Deep Understanding

**Theory:** See [Decision Trees](../machine-learning/notes/decision-trees.md)
**Implementation:** See [DecisionTreeClassifier](../machine-learning/snippets/decision_trees/decision_tree_classifier.py)
**Math:** See [Information Theory](../foundations/notes/information-theory.md)
```

### Link from Learning to Interview Prep

In learning notes, reference interview resources:

```markdown
<!-- In machine-learning/notes/decision-trees.md -->

## Interview Preparation

For quick reference and common questions, see:
- [ML Algorithms Cheatsheet](../../interview-prep/machine-learning/algorithms-cheatsheet.md)
- [Tree-based Methods](../../interview-prep/machine-learning/tree-ensemble-questions.md)
```

---

## Common Question Types

### Coding Interviews

Organize by pattern:

```markdown
# Common Coding Patterns

1. **Two Pointers** - Sorted arrays, palindromes
2. **Sliding Window** - Subarray/substring problems
3. **Binary Search** - Sorted input, find boundary
4. **DFS/BFS** - Trees, graphs, connected components
5. **Backtracking** - Combinations, permutations, subsets
6. **Dynamic Programming** - Optimization, counting
7. **Greedy** - Intervals, scheduling
8. **Heap** - K-th element, merge K sorted
9. **Trie** - Prefix matching, autocomplete
10. **Union-Find** - Connectivity, cycles
```

### ML Interviews

Organize by topic:

```markdown
# ML Interview Topics

## Algorithms
- Decision Trees: Splitting criteria, pruning, bias-variance
- Random Forests: Bagging, OOB error, feature importance
- Gradient Boosting: Sequential learning, learning rate
- Neural Networks: Backprop, activation functions, regularization

## Evaluation
- Metrics: When to use each (precision vs recall)
- Cross-validation: K-fold, stratified, time series
- Overfitting: Detection and prevention

## System Design
- Training pipelines: Data, features, model, serving
- Scaling: Distributed training, online learning
- Monitoring: Drift detection, retraining triggers
```

### System Design Interviews

Organize by component:

```markdown
# System Design Components

## Load Balancing
**Options:** Round-robin, least connections, consistent hashing
**Trade-offs:** [Brief table]

## Caching
**Strategies:** Write-through, write-back, cache-aside
**Eviction:** LRU, LFU, TTL
```

---

## Code Snippets in Interview Prep

### Keep It Minimal

**Don't:**
```python
def binary_search(arr: List[int], target: int) -> int:
    """
    Perform binary search on a sorted array.

    Binary search works by repeatedly dividing the search interval
    in half. At each step, we compare the middle element...
    [200 lines of explanation]
    """
```

**Do:**
```python
def binary_search(arr, target):
    """Find target in sorted array. O(log n)"""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: left = mid + 1
        else: right = mid - 1

    return -1
```

### Template Comments

Use comments to mark the pattern structure:

```python
def sliding_window_template(arr, k):
    # Initialize window
    window = []
    result = 0

    # Expand window
    for right in range(len(arr)):
        window.append(arr[right])

        # Maintain window condition
        while window_invalid():
            window.pop(0)  # Contract from left

        # Update result
        result = max(result, len(window))

    return result
```

---

## Formatting Conventions

### Quick Facts

Use bullet points for rapid scanning:

```markdown
**Decision Trees:**
- **Splitting:** Greedy, max information gain
- **Impurity:** Gini (fast) or Entropy (balanced)
- **Prediction:** Mode (classification), Mean (regression)
- **Complexity:** O(n × m × log m) train, O(log d) predict
- **Pros:** Interpretable, handles non-linear, no scaling needed
- **Cons:** Overfits, high variance, unstable
- **Fix:** Ensemble (Random Forest, Gradient Boosting)
```

### Comparison Tables

Use for choosing between alternatives:

```markdown
| Algorithm | Train Time | Predict | Interpretable | Handles Non-linear |
|-----------|------------|---------|---------------|-------------------|
| Linear Reg | O(n·m²) | O(m) | ✅ | ❌ |
| Decision Tree | O(n·m·log n) | O(log d) | ✅ | ✅ |
| Random Forest | O(k·n·m·log n) | O(k·log d) | ⚠️ | ✅ |
| Neural Net | O(epochs·n·params) | O(params) | ❌ | ✅ |
```

### Formulas

Include only essential formulas with brief context. **Use LaTeX math for better readability:**

```markdown
**Information Gain:** $IG = H(parent) - \sum \frac{|child|}{|parent|} \times H(child)$

**Gini:** $1 - \sum p_i^2$ — Fast, no log

**Entropy:** $-\sum p_i \log_2(p_i)$ — Info-theoretic

**Where:** $p_i$ = class probability
```

**For complex formulas**, use display math:

```markdown
**Gradient Descent Update:**

$$
\theta_{new} = \theta_{old} - \alpha \nabla J(\theta)
$$

where $\alpha$ is learning rate
```

**Why LaTeX?** Clearer subscripts, superscripts, and symbols than Unicode

---

## Edge Cases and Gotchas

Always include common mistakes:

```markdown
## Binary Search Gotchas

**Off-by-one errors:**
- ❌ `while left < right:` → Misses single element
- ✅ `while left <= right:` → Correct for finding element
- ⚠️ Use `left < right` for finding boundary

**Integer overflow:**
- ❌ `mid = (left + right) / 2` → Overflow in some languages
- ✅ `mid = left + (right - left) // 2` → Safe

**Infinite loop:**
- ❌ `left = mid` or `right = mid` → Can loop forever
- ✅ `left = mid + 1` or `right = mid - 1` → Converges
```

---

## Quality Checklist

Interview prep notes should be:

- [ ] **Scannable**: Find key info in < 30 seconds
- [ ] **Template-driven**: Code you can adapt immediately
- [ ] **Pattern-focused**: Clear recognition signals
- [ ] **Complexity visible**: Time/space always shown
- [ ] **Concise**: No fluff, just essentials
- [ ] **Cross-referenced**: Link to deep learning notes
- [ ] **Edge cases**: Common mistakes documented

---

## What NOT to Include

**❌ Skip in interview prep:**
- Long explanations (put in learning notes)
- Mathematical derivations (link to learning notes)
- Historical context
- Multiple approaches without clear winner
- Extensive examples (one or two max)

**✅ Keep in interview prep:**
- Recognition patterns
- Template code
- Complexity
- Common variations
- Edge cases and gotchas

---

## Example: Complete Pattern Note

```markdown
# Dynamic Programming - 1D Array

**When to recognize:**
- "Maximum/minimum" + "subarray/subsequence"
- "Count ways to..."
- "Can you partition..."
- Overlapping subproblems + optimal substructure

**Template:**
```python
def dp_1d(arr):
    n = len(arr)
    dp = [base_case] * n

    # Fill dp table
    for i in range(n):
        for j in range(i):  # Check previous states
            dp[i] = max(dp[i], dp[j] + transition)

    return dp[-1]  # Or max(dp)
```

**Complexity:** Time O(n²) | Space O(n) → Can optimize to O(1)

**Variations:**
- **House Robber:** Can't take adjacent → `dp[i] = max(dp[i-1], dp[i-2] + arr[i])`
- **Longest Increasing Subsequence:** Check all previous smaller elements
- **Coin Change:** Unbounded knapsack variant

**Example:**
```python
# LC 198 - House Robber
def rob(nums):
    if not nums: return 0
    if len(nums) == 1: return nums[0]

    prev2, prev1 = 0, 0
    for num in nums:
        prev2, prev1 = prev1, max(prev1, prev2 + num)

    return prev1
```

**Common mistakes:**
- Forgetting base cases (empty array, single element)
- Not optimizing space when only need previous states
- Off-by-one in index calculations

**Deep dive:** See [Dynamic Programming](../../algorithms/notes/dynamic-programming.md)
```

---

## Summary

Interview prep notes are:
- **Concise** (templates, not essays)
- **Pattern-driven** (recognize question types)
- **Quick reference** (< 30 second lookup)
- **Linked to learning** (depth lives elsewhere)

**Golden rule:** If you're writing paragraphs of explanation, it belongs in learning notes, not interview prep.
