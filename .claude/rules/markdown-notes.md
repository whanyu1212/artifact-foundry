---
paths: ["*/notes/*.md", "!interview-prep/**/*.md"]
---

# Markdown Notes Standards

Notes in topic folders are optimized for **deep understanding** and serve as a permanent learning archive. These are living documents that capture not just what you learned, but why it matters and how concepts connect.

## Core Principles

- **Understanding over coverage**: One well-understood concept beats surface-level summaries
- **Concept-first organization**: Organize by ideas, not by learning source (book/course)
- **Archival quality**: Write for future you (6+ months later)
- **Cross-reference liberally**: Connect related concepts across topics
- **Show your work**: Include derivations, examples, and "why" not just "what"

---

## File Naming and Organization

### Naming Convention

Use descriptive, specific names in kebab-case:

**Good:**
- `decision-trees.md`
- `gradient-descent-optimization.md`
- `probability-distributions.md`
- `bagging-ensemble-methods.md`

**Avoid:**
- `ml.md` (too vague)
- `DecisionTrees.md` (use kebab-case, not PascalCase)
- `chapter-3-notes.md` (organize by concept, not source)
- `lecture_5.md` (not archival; what's the concept?)

### When to Split Files

**Keep in one file when:**
- Concepts are tightly coupled (bagging and random forests)
- Total length under ~1000 lines
- Natural progression of related ideas

**Split into separate files when:**
- Topics are conceptually distinct (decision trees vs neural networks)
- File exceeds ~1000 lines
- You find yourself scrolling too much to find content

**Split by aspect if needed:**
- `neural-networks-foundations.md`
- `neural-networks-architectures.md`
- `neural-networks-training.md`

---

## Note Structure

### Title and Overview

Start with a clear H1 title and brief overview:

```markdown
# Decision Trees

Classification and regression using recursive binary splits.

**Core idea**: Partition feature space into regions by asking yes/no questions about features. Predict using the most common value in each region.

**Key concepts**: Information gain, impurity measures, greedy splitting, tree depth
```

**Why this matters:**
- Future you can quickly recall what this is about
- Overview provides context before diving into details

### Organizing Content

Use H2 headers for major concepts, H3 for sub-concepts:

```markdown
## How Decision Trees Work

### The Splitting Algorithm

[Explanation of CART algorithm...]

### Impurity Measures

#### Gini Impurity
[Details...]

#### Entropy
[Details...]

## When to Use Decision Trees

### Strengths
- Interpretable...
- Handles non-linear...

### Limitations
- Prone to overfitting...
- High variance...
```

**Guidelines:**
- Use H2 for major sections you'd put in a table of contents
- Use H3 for supporting concepts within a section
- Use H4 sparingly (only when truly needed for hierarchy)
- Each section should be self-contained enough to understand on its own

---

## Mathematical Content

### Formulas and Equations

Include formulas when they clarify concepts. Use both symbolic and explained forms:

**Good approach:**
```markdown
## Gini Impurity

Measures how often a randomly chosen element would be incorrectly labeled:

$$
Gini = 1 - \sum_{i=1}^{n} p_i^2
$$

where:
- $p_i$ = proportion of samples belonging to class $i$
- Sum is over all classes

**Example:**
- 5 samples: [0, 0, 1, 1, 1]
- $p_0 = 2/5 = 0.4$, $p_1 = 3/5 = 0.6$
- $Gini = 1 - (0.4^2 + 0.6^2) = 1 - 0.52 = 0.48$

**Interpretation**: 0 = pure (all same class), higher = more mixed
```

### Mathematical Notation

**Preferred: Use LaTeX inline math** for better readability and rendering:

**Inline math** (within text):
```markdown
The Gini impurity is calculated as $Gini = 1 - \sum_{i=1}^{n} p_i^2$ where $p_i$ is the proportion of class $i$.
```

**Display math** (standalone equations):
```markdown
The gradient descent update rule:

$$
\theta_{new} = \theta_{old} - \alpha \nabla J(\theta)
$$

where $\theta$ represents parameters, $\alpha$ is the learning rate, and $\nabla J(\theta)$ is the gradient.
```

**Why LaTeX math?**
- Renders beautifully in GitHub, VS Code, most markdown viewers
- Standard notation (matches papers and textbooks)
- Better for complex formulas (fractions, subscripts, matrices)

**Fallback: Unicode symbols** (if LaTeX not supported in your viewer):

**Common symbols:**
- Greek: α, β, γ, θ, μ, σ, Σ, Π, Δ
- Math: ±, ×, ÷, ≤, ≥, ≈, ≠, √, ²,³
- Logic: ∀, ∃, ∈, ∉, ∅
- Arrows: →, ←, ↔, ⇒

**Last resort: Code blocks** (when neither LaTeX nor Unicode works):
```markdown
The gradient descent update rule:

```
θ_new = θ_old - α × ∇J(θ)

where:
  θ = parameters (vector)
  α = learning rate (scalar)
  ∇J(θ) = gradient of cost function
```
```

**When to show derivations:**
- When understanding the derivation helps grasp the concept
- When it reveals insights about why the formula works
- When you struggled with it and want to remember
- Skip if it's just algebra without insight

---

## Code Examples in Notes

### When to Include Code

Include code when it clarifies the concept:

```markdown
## Information Gain Calculation

```python
def information_gain(parent, left, right):
    """
    IG = Parent_Impurity - Weighted_Children_Impurity
    """
    n = len(parent)
    n_left, n_right = len(left), len(right)

    weighted_impurity = (n_left/n * gini(left) +
                        n_right/n * gini(right))

    return gini(parent) - weighted_impurity
```

**Example:**
- Parent: [0,0,1,1,1] → Gini = 0.48
- Left: [0,0] → Gini = 0 (pure!)
- Right: [1,1,1] → Gini = 0 (pure!)
- IG = 0.48 - 0 = 0.48 (perfect split!)
```

### Linking to Implementations

Reference your actual implementations:

```markdown
## Implementation

See full implementation: [decision_tree_classifier.py](../snippets/decision_trees/decision_tree_classifier.py)

Key methods:
- `_calculate_impurity()` - Computes Gini or Entropy
- `_find_best_split()` - Greedy search for optimal split
- `_build_tree()` - Recursive tree construction
```

---

## Cross-Referencing

### Linking Related Concepts

**Within same topic:**
```markdown
This builds on [Probability Distributions](./probability-distributions.md).
See also [Ensemble Methods](./ensemble-methods.md) for combining multiple trees.
```

**Across topics:**
```markdown
**Prerequisites:**
- [Probability Theory](../../foundations/notes/probability-basics.md)
- [Calculus - Optimization](../../foundations/notes/calculus-optimization.md)

**Related topics:**
- [Random Forests](./random-forests.md) - Ensemble of decision trees
- [Gradient Boosting](./gradient-boosting.md) - Sequential tree building
```

**To code snippets:**
```markdown
**Implementation:**
- [Decision Tree Classifier](../snippets/decision_trees/decision_tree_classifier.py:48-68) - `__init__` method
- [Tree Metrics](../snippets/decision_trees/tree_metrics.py) - Impurity calculations
```

### When to Cross-Reference

- **Always**: Prerequisites (foundational concepts needed)
- **Often**: Related implementations, similar algorithms, extensions
- **Sometimes**: Historical context, alternative approaches
- **Rarely**: Tangentially related topics (avoid noise)

---

## Explanatory Style

### Write for Understanding

**Good explanation:**
```markdown
## Why Decision Trees Use Greedy Splitting

Decision trees use a greedy algorithm: at each node, choose the split that gives the best immediate information gain.

**Why greedy?** Finding the globally optimal tree is NP-complete. With n features and m samples, there are O(n^m) possible trees.

**Trade-off:** Greedy doesn't guarantee global optimum, but runs in O(n × m × log m) time - practical for real datasets.

**Implication:** Trees can get stuck in local optima. Ensemble methods (Random Forests, Boosting) address this by building multiple trees differently.
```

**What makes this good:**
- Explains the "why" behind the decision
- Provides complexity context
- Shows the trade-off explicitly
- Connects to solutions (ensembles)

### Include Mental Models

Help future you remember by including intuition:

```markdown
## Random Forest Intuition

**Mental model:** "Wisdom of crowds"

Individual trees overfit and have high variance (different training data → very different trees). But their *errors are uncorrelated* because:
1. Bootstrap sampling → different training sets
2. Feature randomization → different split decisions

Averaging uncorrelated predictions reduces variance without increasing bias.

**Analogy:** Asking 100 people to estimate vs asking 1 expert - the average of 100 is often better even if individually they're worse.
```

### Document Gotchas and Pitfalls

Capture what you struggled with:

```markdown
## Common Pitfalls

**Gotcha #1: Gini vs Entropy**
- Both work well in practice
- Gini is faster (no logarithm)
- Entropy slightly more balanced trees
- **Don't overthink it** - tree depth matters more than criterion

**Gotcha #2: Max depth**
- `max_depth=None` → guaranteed overfitting on noisy data
- Rule of thumb: Start with `max_depth=log₂(n_samples)`
- Tune with cross-validation

**Gotcha #3: Feature scaling**
- Decision trees DON'T need feature scaling
- Only split points matter, not distances
- Unlike linear models, gradient descent, k-NN
```

---

## Practical Guidance

### Applications and Use Cases

Include when to use (and not use) concepts:

```markdown
## When to Use Decision Trees

**Use when:**
- Need interpretability (explain predictions to stakeholders)
- Have mixed feature types (categorical + numerical)
- Have non-linear relationships
- Want feature importance automatically
- Need fast predictions

**Don't use when:**
- Need smooth decision boundaries (use SVM, neural nets)
- Have high-dimensional data (n_features > n_samples)
- Need extrapolation beyond training data
- Linear model would suffice (simpler is better)

**Better alternatives:**
- Random Forests: More robust, same interpretability benefits
- Gradient Boosting: Higher accuracy, less interpretable
- Linear models: If relationship is actually linear
```

### Examples with Real Context

Use concrete examples from real domains:

```markdown
## Example: Email Spam Classification

**Setup:**
- Features: word frequencies, sender domain, has_attachments, etc.
- Target: spam (1) or not spam (0)

**Decision tree might learn:**
```
if "viagra" frequency > 0.01:
    return spam
else:
    if sender_domain in ["unknown", "suspicious"]:
        return spam
    else:
        return not_spam
```

**Why this works:**
- Captures non-linear rules ("viagra" is strong signal regardless of other features)
- Interpretable to users ("why was this marked spam?")
- Handles mixed features (frequencies + categories)
```

---

## Version Control and Updates

### Living Documents

Notes evolve as understanding deepens:

```markdown
## Understanding Evolution

**2024-12**: Initial notes on decision trees
**2025-01**: Added complexity analysis after implementing CART
**2025-02**: Expanded pruning section after studying Random Forests

The best approach: Update notes when you learn something new that deepens understanding, not just accumulate facts.
```

### When to Update

- **Update when:** You discover deeper insight, find better explanation, correct misunderstanding
- **Don't update:** Just to add more content without better understanding
- **Archive:** If approach is superseded, keep notes with "Historical" section

---

## Formatting Conventions

### Code Blocks

Always specify language for syntax highlighting:

```markdown
```python
def example():
    pass
```

```sql
SELECT * FROM table;
```

```bash
pip install numpy
```
```

### Lists

**Bullet points** for related items (unordered):
```markdown
Decision trees can:
- Handle non-linear relationships
- Work with mixed feature types
- Provide feature importance
```

**Numbered lists** for sequential steps or ranked items:
```markdown
CART algorithm:
1. Start with all data at root
2. Find best split (max information gain)
3. Create left and right children
4. Recurse on each child
5. Stop at max depth or pure nodes
```

### Emphasis

- **Bold** for key terms and concepts: `**information gain**`
- *Italic* for emphasis or introducing terms: `*impurity measure*`
- `Code font` for code, variable names, file names: `` `max_depth` ``
- > Blockquotes for important notes or citations

### Tables

Use for comparisons:

```markdown
| Metric | Range | Pure | Formula |
|--------|-------|------|---------|
| Gini | [0, 0.5] | 0 | 1 - Σ(p_i²) |
| Entropy | [0, 1] | 0 | -Σ(p_i × log₂(p_i)) |
| MSE | [0, ∞) | 0 | Σ(y_i - ȳ)² / n |
```

---

## Interview Prep vs Learning Notes

**These notes (topic folders)** optimize for understanding:
- Deep dives with derivations
- Multiple approaches and why
- Context and applications
- "How it works internally"

**Interview prep notes** optimize for recall:
- Quick reference format
- Common patterns and templates
- Concise formulas without derivation
- Time complexity cheat sheets

**Duplication is OK!** The same topic can exist in both with different treatments:
- `machine-learning/notes/decision-trees.md` - Deep dive with theory
- `interview-prep/ml-algorithms-cheatsheet.md` - Quick reference

---

## Quality Checklist

Before considering a note "done" (though they're always living documents):

- [ ] Clear title and overview at the top
- [ ] Core concepts explained with "why" not just "what"
- [ ] Mathematical formulas include concrete examples
- [ ] Code snippets have language tags and context
- [ ] Cross-references to related topics and implementations
- [ ] Practical guidance (when to use, pitfalls, trade-offs)
- [ ] Proper formatting (headers, lists, emphasis)
- [ ] Would be understandable to you in 6 months

---

## Anti-Patterns to Avoid

**❌ Source-based organization:**
```markdown
# Week 3 Lecture Notes
# Chapter 5 Summary
# Course XYZ - Module 4
```
→ Organize by concept, not learning source

**❌ Copy-paste without understanding:**
```markdown
The algorithm works by optimizing the objective function using stochastic gradient descent.
```
→ If you can't explain it simply, you don't understand it yet

**❌ No examples:**
```markdown
Gini impurity measures node purity using the formula: Gini = 1 - Σ(p_i²)
```
→ Always include concrete numerical examples

**❌ Missing context:**
```markdown
Use max_depth=5 for best results.
```
→ Explain why, what are the trade-offs, when does this apply?

**❌ Too much detail on wrong things:**
```markdown
[10 paragraphs on history of decision trees in 1960s-1980s]
[1 paragraph on how they actually work]
```
→ Focus on understanding concepts, not historical trivia (unless history provides insight)

---

## Summary

Great notes are:
- **Concept-first** (not source-first)
- **Understanding-focused** (why and how, not just what)
- **Cross-referenced** (show connections)
- **Practical** (when to use, pitfalls, examples)
- **Archival quality** (understandable in 6+ months)

Remember: You're building a knowledge archive for deep understanding, not a textbook summary. Capture insights, connections, and mental models - not just facts.
