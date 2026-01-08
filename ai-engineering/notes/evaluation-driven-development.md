# Evaluation-Driven Development (EDD)

## Overview

Evaluation-Driven Development is a methodology for building LLM applications where evaluation comes first, not as an afterthought. The core principle: **define success metrics before writing application code**, then continuously measure and improve against those metrics.

## Why EDD Matters

Traditional software testing doesn't work well for LLM applications because:

1. **Non-deterministic outputs**: Same input can produce different outputs
2. **Subjective quality**: "Good" output is often context-dependent
3. **No ground truth**: Many tasks don't have single correct answers
4. **Prompt sensitivity**: Small changes can drastically affect behavior
5. **Model updates**: Provider model updates can change behavior without code changes

EDD addresses these challenges by making evaluation a first-class citizen in development.

## Core Principles

### 1. Define Evals First

Before implementing a feature:
- Define what "good" looks like
- Create test cases with expected behaviors
- Establish quantitative and qualitative metrics
- Document edge cases and failure modes

### 2. Build Eval Infrastructure Early

Components needed:
- **Test dataset**: Representative examples covering common and edge cases
- **Evaluation metrics**: Quantitative measures (accuracy, latency, cost)
- **Human evaluation**: Qualitative assessments for subjective outputs
- **Regression suite**: Prevent degradation when making changes
- **Version tracking**: Link evals to model versions, prompts, code

### 3. Iterate Based on Evals

Development loop:
1. Run evals on baseline
2. Make a change (prompt, model, architecture)
3. Run evals again
4. Compare results
5. Keep improvements, reject regressions
6. Repeat

### 4. Continuous Evaluation

Production monitoring:
- Sample real user interactions
- Run evals on production data
- Monitor for drift and degradation
- A/B test changes with real traffic

## Types of Evaluations

### Unit Evals
Test individual components in isolation.

**Examples**:
- Single prompt template effectiveness
- Tool call accuracy
- Retrieval relevance for specific queries
- Parsing correctness

**When to use**: During development of individual components

### Integration Evals
Test how components work together.

**Examples**:
- Multi-step agent workflows
- RAG pipeline end-to-end
- Tool chaining behavior
- Error handling across system

**When to use**: When combining components into workflows

### System Evals
Test entire application behavior.

**Examples**:
- User task completion rate
- Response quality for full conversations
- End-to-end latency
- Cost per interaction

**When to use**: Before deployment and in production monitoring

## Evaluation Metrics

### Quantitative Metrics

**Correctness Metrics**:
- Exact match (strict)
- Semantic similarity (embeddings)
- Contains required elements
- Follows format constraints
- Factual accuracy

**Performance Metrics**:
- Latency (p50, p95, p99)
- Token usage
- Cost per request
- Throughput

**Reliability Metrics**:
- Success rate
- Error rate by type
- Retry frequency
- Timeout rate

### Qualitative Metrics

**Human Evaluation**:
- Response quality (1-5 scale)
- Helpfulness
- Truthfulness
- Harmlessness
- Alignment with intent

**LLM-as-Judge**:
Use stronger model to evaluate weaker model outputs:
- Rubric-based scoring
- Comparison ranking (A vs B)
- Criteria checking
- COT reasoning about quality

⚠️ **Caveat**: LLM judges have biases (length, format, position). Validate against human judgments.

## Building an Eval Suite

### 1. Create Test Dataset

**Sources**:
- Real user examples (anonymized)
- Synthetic generation (use LLM to create diverse cases)
- Edge cases from bug reports
- Adversarial examples
- Golden dataset (hand-crafted high-quality examples)

**Best practices**:
- Start with 20-50 examples minimum
- Cover diverse scenarios
- Include failure modes
- Balance common and rare cases
- Version control test data

### 2. Define Evaluation Functions

```python
def evaluate_response(
    input: str,
    expected: str,
    actual: str,
    context: dict
) -> dict:
    """
    Evaluate a single response.
    
    Returns:
        {
            'passed': bool,
            'score': float,
            'metrics': dict,
            'feedback': str
        }
    """
    pass
```

### 3. Run and Track Results

**Track over time**:
- Metric trends
- Per-example performance
- Aggregate statistics
- Cost and latency
- Model/prompt versions

**Tools**:
- LangSmith (LangChain)
- Weights & Biases
- MLflow
- Custom dashboards

## Practical Patterns

### Pattern 1: Golden Dataset
Maintain curated high-quality examples with expected outputs.

**When to use**: 
- When ground truth exists
- For regression testing
- High-stakes applications

### Pattern 2: LLM-as-Judge
Use powerful model to evaluate outputs against criteria.

**When to use**:
- Subjective quality assessment
- No clear ground truth
- Scale human evaluation

**Example criteria**:
```
Score the response on:
1. Accuracy: Does it correctly answer the question?
2. Completeness: Does it cover all aspects?
3. Clarity: Is it easy to understand?
4. Conciseness: Is it appropriately brief?

Rate each 1-5, provide reasoning.
```

### Pattern 3: A/B Testing
Run two variants simultaneously in production.

**When to use**:
- Validating improvements
- Optimizing prompts
- Model selection

**Metrics to compare**:
- User satisfaction (thumbs up/down)
- Task completion rate
- Follow-up questions
- Session length

### Pattern 4: Adversarial Testing
Deliberately test edge cases and failure modes.

**Examples**:
- Jailbreak attempts
- Confusing inputs
- Ambiguous requests
- Out-of-domain queries
- Prompt injection attempts

### Pattern 5: Regression Monitoring
Continuously check that changes don't degrade existing functionality.

**Implementation**:
- Run full eval suite on every change
- Block deployment if metrics regress
- Track metrics over deployments
- Alert on degradation

## Common Pitfalls

### 1. Eval-Dataset Mismatch
Testing on data unlike real usage.

**Solution**: Regularly sample production data into eval set.

### 2. Overfitting to Evals
Optimizing for test cases at expense of general performance.

**Solution**: Hold out test set, diverse examples, blind testing.

### 3. Ignoring Latency/Cost
Focusing only on quality metrics.

**Solution**: Multi-objective optimization, set thresholds.

### 4. Not Versioning Evals
Can't compare across time.

**Solution**: Version test datasets, track all changes.

### 5. Manual-Only Evaluation
Doesn't scale, slows iteration.

**Solution**: Automate where possible, sample for human review.

## EDD Workflow Example

### Initial Setup
1. Define task clearly
2. Create 30 test examples
3. Implement baseline solution
4. Run evals, establish baseline metrics

### Iteration Loop
```
baseline_score = run_evals(baseline_system)
# baseline_score = 0.65

for variant in [prompt_v2, prompt_v3, different_model]:
    new_score = run_evals(variant)
    if new_score > baseline_score:
        baseline_system = variant
        baseline_score = new_score
        print(f"Improvement: {new_score}")
    else:
        print(f"No improvement: {new_score}")

# Final: baseline_score = 0.82
```

### Production Deployment
1. Run full eval suite (100% pass on critical cases)
2. Deploy to staging
3. A/B test with 5% traffic
4. Monitor metrics for 24h
5. Roll out to 100% if metrics hold
6. Continue monitoring

## Tools and Frameworks

### Open Source
- **OpenAI Evals**: Framework and test suite
- **LangSmith**: LangChain's evaluation platform
- **Weights & Biases**: Experiment tracking
- **PromptFoo**: Prompt testing and comparison
- **DeepEval**: LLM evaluation framework

### Commercial
- **Humanloop**: Prompt management + evals
- **Helicone**: LLM observability
- **Braintrust**: AI product evaluation
- **Context.ai**: Production monitoring

## When to Apply EDD

**High value**:
- Customer-facing applications
- High-stakes decisions
- Complex multi-step workflows
- Frequent prompt/model changes
- Production monitoring needed

**Lower value**:
- Simple prototypes
- One-off scripts
- Personal tools
- Minimal user impact

## Further Reading

- [Hamel Husain - A Guide to LLM Evals](https://hamel.dev/blog/posts/evals/)
- [OpenAI Evals Repository](https://github.com/openai/evals)
- [LangSmith Evaluation Guide](https://docs.smith.langchain.com/evaluation)
- [Anthropic's Measuring Model Persuasiveness](https://www.anthropic.com/research/measuring-model-persuasiveness)

## Key Takeaways

1. **Evals first**: Define success before implementing
2. **Automate**: Make running evals fast and easy
3. **Iterate**: Use eval results to guide improvements
4. **Monitor**: Continue evaluating in production
5. **Version everything**: Code, prompts, models, test data
6. **Balance metrics**: Quality, cost, latency together
7. **Human in loop**: Automate bulk, sample for human review
