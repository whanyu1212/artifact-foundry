# Organization Guide

## Folder Structure Philosophy

This repository separates **learning/skill development** from **interview preparation**. Here's how to decide where content belongs:

---

## Decision Tree

```
Is this content primarily for interview prep?
│
├─ YES → Put in interview-prep/
│   ├─ Cheatsheets and quick references
│   ├─ Common interview patterns
│   ├─ Template solutions
│   └─ Frequently asked questions
│
└─ NO → Put in the appropriate topic folder
    ├─ Deep conceptual understanding
    ├─ Comprehensive documentation
    ├─ Real-world applications
    └─ Project-related code
```

---

## Concrete Examples

### SQL Content

| Content Type | Location | Example |
|-------------|----------|---------|
| **Cheatsheet** | `interview-prep/notes/` | Quick syntax reference for all SQL commands |
| **Interview Patterns** | `interview-prep/notes/` | "Find Nth highest salary" templates |
| **Top Interview Questions** | `interview-prep/notes/` | Common SQL interview questions with solutions |
| **Deep Dive Notes** | `data-analytics/notes/` | How query optimization works internally |
| **Database Design** | `data-analytics/notes/` | Normalization, indexing strategies, when to denormalize |
| **Real-world Patterns** | `data-analytics/notes/` | How to design analytics tables, star schema, etc. |
| **Practice Queries** | `data-analytics/snippets/` | Queries you wrote for actual analysis |

### Machine Learning Content

| Content Type | Location | Example |
|-------------|----------|---------|
| **Algorithm Cheatsheet** | `interview-prep/notes/` | Quick ML algorithm comparison table |
| **Interview Questions** | `interview-prep/notes/` | "Explain bias-variance tradeoff" with concise answer |
| **Algorithm Deep Dive** | `machine-learning/notes/` | How gradient descent actually works with derivations |
| **Implementation** | `machine-learning/snippets/` | Your implementation of algorithms from scratch |
| **Case Studies** | `machine-learning/notes/` | How you applied ML to solve real problems |

### Probability/Statistics Content

| Content Type | Location | Example |
|-------------|----------|---------|
| **Probability Cheatsheet** | `interview-prep/notes/` | Quick formulas and distributions (like you already have!) |
| **Interview Patterns** | `interview-prep/notes/` | Common probability interview questions |
| **Derivations** | `foundations/notes/` | Step-by-step proofs and mathematical derivations |
| **Practice Problems** | `foundations/notes/` | Problem sets with detailed solutions |

---

## Key Principles

### interview-prep/
**Optimized for**: Speed and recall during high-pressure situations

**Characteristics**:
- ✅ Condensed, no fluff
- ✅ Pattern-focused (common question types)
- ✅ Template solutions you can adapt
- ✅ Quick lookup tables and formulas
- ✅ "What interviewers commonly ask"
- ❌ Not comprehensive
- ❌ Not for deep understanding

**Think**: "I have 15 minutes to review before my interview"

### Topic Folders (data-analytics/, machine-learning/, etc.)
**Optimized for**: Understanding and application

**Characteristics**:
- ✅ Comprehensive explanations
- ✅ "Why" and "how it works internally"
- ✅ Multiple approaches and trade-offs
- ✅ Real-world context and applications
- ✅ Your personal notes from learning
- ❌ Not optimized for quick lookup

**Think**: "I want to truly understand this topic and apply it to real problems"

---

## Duplication is OK!

**You can have the same topic in multiple places with different treatments:**

### Example: "GROUP BY in SQL"

**In data-analytics/notes/sql-advanced.md**:
```markdown
## GROUP BY and Aggregations

GROUP BY allows us to collapse rows that share common values into summary rows.

### How it works internally:
1. The database creates a hash table...
2. For each row, it computes the hash of the grouping columns...
3. ...

### When to use GROUP BY vs Window Functions:
- Use GROUP BY when you need to reduce the number of rows
- Use window functions when you need to keep all rows but add aggregate information

### Real-world example:
In our user analytics dashboard, we use GROUP BY to compute daily active users:
[detailed example with explanation]

### Performance considerations:
- GROUP BY can be expensive on large datasets...
- Consider pre-aggregating in a materialized view...
```

**In interview-prep/notes/sql-cheatsheet.md**:
```markdown
## GROUP BY

SELECT department_id, COUNT(*)
FROM employees
GROUP BY department_id;

-- With HAVING (filter after grouping)
SELECT department_id, AVG(salary)
FROM employees
GROUP BY department_id
HAVING AVG(salary) > 70000;

Common interview question: "What's the difference between WHERE and HAVING?"
Answer: WHERE filters before grouping, HAVING filters after grouping.
```

---

## Special Folder Rules

### foundations/
- Math, statistics, algorithms, data structures, systems
- Theory-heavy content
- Could be used for both learning AND interview prep depending on content

### interview-prep/
- **ONLY** interview-specific content
- Think "what would I review the night before an interview?"
- Pattern recognition, not deep understanding

### PROJECTS.md
- Links to actual implementations and projects
- Bridges theory to practice

---

## Quick Reference

**Ask yourself**:

1. **"Would I read this the night before an interview?"**
   - YES → `interview-prep/`
   - NO → Topic folder

2. **"Is this optimized for quick recall or deep understanding?"**
   - Quick recall → `interview-prep/`
   - Deep understanding → Topic folder

3. **"Is this a common interview pattern or real-world application?"**
   - Interview pattern → `interview-prep/`
   - Real-world application → Topic folder

---

## Revision History

- 2025-12-20: Initial organization guide created
