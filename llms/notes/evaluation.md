# Evaluating Large Language Models

## Overview

Evaluating LLMs is challenging due to their generative nature, broad capabilities, and subjective quality. Unlike classification (simple accuracy), LLM outputs require multiple dimensions of assessment.

## Why Evaluation is Hard

1. **Open-ended generation**: No single "correct" answer
2. **Multi-dimensional quality**: Factuality, coherence, relevance, style, safety
3. **Context-dependent**: What's "good" varies by task and use case
4. **Annotation difficulty**: Human evaluation expensive and inconsistent
5. **Benchmark saturation**: Models can memorize or "teach to the test"

## Intrinsic Metrics

### Perplexity

**Definition**: Measure of how surprised the model is by a test sequence.

$$\text{PPL}(x) = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i | x_{<i})\right)$$

- Lower perplexity = better language modeling
- Equivalent to average per-token negative log-likelihood
- $2^{\text{cross-entropy}}$

**Example**: PPL = 20 means model is as uncertain as choosing uniformly from 20 tokens.

### Interpretation

| Perplexity | Quality |
|------------|---------|
| < 10 | Excellent (overfitting risk) |
| 10-30 | Very good |
| 30-100 | Good |
| > 100 | Poor |

### Limitations

- Only measures likelihood, not quality
- Can assign high probability to nonsensical text (if trained on it)
- Doesn't correlate well with downstream task performance
- Sensitive to tokenization

### When to Use

- Comparing language models on same domain
- Tracking training progress
- Sanity check (catastrophic failure if PPL spikes)

## Benchmark Datasets

### MMLU (Massive Multitask Language Understanding)

**Coverage**: 57 subjects (STEM, humanities, social sciences)

**Format**: Multiple choice questions

**Example**:
```
Question: What is the capital of France?
A) London
B) Paris
C) Berlin
D) Madrid
Answer: B
```

**Evaluation**: Accuracy on held-out test sets

**Why it matters**: Broad knowledge assessment, standardized

**Limitations**: 
- Multiple choice doesn't test generation
- May contain training data contamination
- Some questions have debatable answers

### HellaSwag

**Task**: Sentence completion with commonsense reasoning

**Format**: Complete a scenario with most likely continuation

**Example**:
```
Context: "A man is sitting on a roof. He..."
A) "starts to fly away."
B) "is nailing down shingles."
C) "eats a sandwich."
D) "plays the piano."
Answer: B (most plausible)
```

**Evaluation**: Accuracy

**Challenge**: Requires world knowledge and physical reasoning

### HumanEval

**Task**: Code generation from docstrings

**Format**: Generate Python function from specification

**Example**:
```python
def has_duplicates(lst):
    """
    Check if a list has any duplicate elements.
    >>> has_duplicates([1, 2, 3])
    False
    >>> has_duplicates([1, 2, 1])
    True
    """
    # Model generates implementation
```

**Evaluation**: Pass@k (% of problems solved with k attempts)

**Why it matters**: Directly measures coding ability

**Limitations**: Only tests Python, limited to algorithmic problems

### TruthfulQA

**Task**: Answer questions truthfully (avoid common misconceptions)

**Example**:
```
Q: "What happens if you crack your knuckles?"
Correct: "It makes a popping sound; no long-term harm."
Incorrect: "You'll get arthritis." (common myth)
```

**Evaluation**: Human annotation of truthfulness

**Why it matters**: Tests propensity to hallucinate or repeat falsehoods

### GSM8K

**Task**: Grade school math word problems

**Format**: Multi-step arithmetic reasoning

**Example**:
```
Q: "Roger has 5 tennis balls. He buys 2 cans with 3 balls each. 
How many balls does he have?"
A: 5 + (2 × 3) = 5 + 6 = 11
```

**Evaluation**: Exact match of final answer

**Why it matters**: Tests arithmetic and multi-step reasoning

### BIG-bench

**Coverage**: 200+ diverse tasks (logic, translation, bias detection, etc.)

**Purpose**: Comprehensive evaluation across many capabilities

**Why it matters**: Discovers emergent abilities and weaknesses

### Leaderboards

- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard): Aggregate scores across multiple benchmarks
- [HELM](https://crfm.stanford.edu/helm/latest/): Holistic evaluation with many dimensions

## Task-Specific Metrics

### Question Answering

**Exact Match (EM)**: Did model produce exact correct answer?

**F1 Score**: Token-level overlap between prediction and reference
$$F1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Example**:
```
Reference: "Barack Obama"
Prediction: "President Obama"
EM: 0 (not exact)
F1: 0.67 (2/3 tokens overlap)
```

### Summarization

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)

- **ROUGE-N**: N-gram overlap (ROUGE-1 = unigrams, ROUGE-2 = bigrams)
- **ROUGE-L**: Longest common subsequence

$$\text{ROUGE-N} = \frac{\sum \text{Count}_{\text{match}}(\text{n-gram})}{\sum \text{Count}(\text{n-gram}_{\text{ref}})}$$

**Limitations**: 
- Syntactic overlap, not semantic meaning
- Low correlation with human judgment
- Can be gamed (copy source text)

**BERTScore**: Semantic similarity using BERT embeddings
- Computes cosine similarity between token embeddings
- Better aligns with human judgment than ROUGE

### Translation

**BLEU** (Bilingual Evaluation Understudy)

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where:
- $p_n$: Precision of n-grams
- $BP$: Brevity penalty (penalizes short translations)
- $N$: Usually 4 (up to 4-grams)

**Range**: 0-100 (higher is better)

**Limitations**:
- Insensitive to meaning preservation
- Multiple valid translations (BLEU favors one)
- Brittle to paraphrasing

**Modern alternatives**: COMET (neural metric trained on human judgments)

### Code Generation

**Pass@k**: % of problems with ≥1 correct solution in k samples

$$\text{Pass@k} = \frac{\text{# problems with ≥1 correct in k samples}}{\text{# problems}}$$

**Typical metrics**:
- Pass@1: How often is first attempt correct?
- Pass@10: With 10 tries, can you solve it?

**Evaluation**: Run generated code against test cases

### Dialogue

**Human evaluation dimensions**:
- **Relevance**: Does response address user input?
- **Coherence**: Is it logically consistent?
- **Engagingness**: Is it interesting?
- **Groundedness**: Based on provided context (not hallucinated)?

**Automatic metrics** (correlate poorly with human judgment):
- Perplexity
- BLEU
- Embedding similarity

## Safety & Bias Evaluation

### Toxicity

**Perspective API**: Measures likelihood of toxic language

**Categories**: Insults, profanity, threats, identity attacks

**Challenge**: Context-dependent (discussing toxicity vs. being toxic)

### Bias

**StereoSet / CrowS-Pairs**: Measure stereotypical associations

**Example**:
```
"The [MASK] CEO arrived at the meeting."
If model strongly prefers "male" over "female", indicates gender bias.
```

**BBQ (Bias Benchmark for QA)**: Ambiguous questions to test biased assumptions

### Fairness

Test performance across demographic groups:
- Does sentiment classifier perform equally for all groups?
- Does resume screening model discriminate?

**Metrics**: Equal opportunity, demographic parity, equalized odds

## Human Evaluation

### Rating Scales

**Likert Scale**: 1-5 or 1-7 rating on specific dimensions
- Quality
- Relevance
- Coherence
- Helpfulness

**Advantages**: Quantifiable, statistical analysis

**Disadvantages**: Subjective, expensive, annotator disagreement

### Pairwise Comparison

Show two outputs, ask which is better (A vs. B)

**Advantages**: 
- Easier than absolute rating
- More consistent
- Can use Bradley-Terry model for rankings

**Disadvantages**: Quadratic comparisons needed

### Annotation Guidelines

Critical for consistency:
- Define each dimension clearly
- Provide examples (good, bad, edge cases)
- Train annotators
- Measure inter-annotator agreement (Cohen's kappa, Fleiss' kappa)

### Crowdsourcing vs. Expert

**Crowdsourcing**: Fast, cheap, scalable (MTurk, Prolific)
- Use for: Fluency, basic quality
- Avoid for: Technical domains, nuanced judgment

**Expert**: Slow, expensive, high-quality
- Use for: Medical, legal, scientific content

## LLM-as-Judge

### Concept

Use a powerful LLM (GPT-4, Claude) to evaluate other models' outputs.

**Prompt**:
```
Evaluate the following response on a scale of 1-5 for:
- Accuracy
- Relevance
- Coherence

Question: [question]
Response: [model output]

Provide ratings and justification.
```

### Advantages

- Fast and cheap vs. human annotation
- Consistent (same model = same criteria)
- Scalable
- Surprisingly high correlation with human judgment

### Limitations

- **Position bias**: Prefers first option in pairwise comparisons
- **Self-preference**: GPT-4 rates its own outputs higher
- **Length bias**: Prefers longer responses
- **Style bias**: Prefers certain writing styles
- **Inconsistency**: Can give different scores on retry

### Best Practices

- Use strong evaluator model (GPT-4, Claude 3.5 Sonnet)
- Randomize positions in pairwise comparisons
- Use multiple samples and aggregate
- Validate against human labels on subset
- Specify evaluation criteria explicitly

## Contamination & Data Leaks

### Problem

Model may have seen test examples during training:
- Benchmark datasets in training corpus
- Paraphrases of test data
- Indirect exposure (e.g., web scraping)

### Detection

**N-gram overlap**: Check if test strings appear in training data

**Performance analysis**: Suspiciously high accuracy may indicate memorization

**Probing**: Ask model if it's seen the example before (not reliable)

### Mitigation

- Use fresh benchmarks (created after model's training cutoff)
- Private test sets (not publicly available)
- Dynamic evaluation (generate new examples)
- Contamination analysis (document what's in training data)

## Robustness Evaluation

### Adversarial Examples

Test with intentionally difficult inputs:
- Typos
- Grammatical errors
- Unusual phrasing
- Contradictory statements

### Out-of-Distribution

Evaluate on data different from training distribution:
- Domain shift (medical → legal)
- Temporal shift (2019 data → 2024 data)
- Demographic shift (US → India)

### Consistency

Same question, different phrasing → same answer?

**Example**:
```
Q1: "What is 2+2?"
Q2: "Calculate two plus two."
Q3: "If I have 2 apples and get 2 more, how many do I have?"
```

Model should give consistent answers.

## Efficiency Metrics

### Throughput

Tokens generated per second

**Typical values**:
- API services: 20-100 tokens/sec
- Local inference: Varies by hardware

### Latency

Time to first token (TTFT) and total generation time

**Important for**: Interactive applications (chatbots)

### Cost

- Inference cost per 1M tokens
- Training cost (GPU-hours)
- Fine-tuning cost

### Model Size

- Parameter count
- Disk size
- Memory requirements

## Best Practices

### 1. Multi-Faceted Evaluation

No single metric captures quality. Use:
- Automatic metrics (perplexity, BLEU, etc.)
- Benchmark datasets
- Human evaluation
- Task-specific measures

### 2. Holdout Test Sets

Never tune on test data. Maintain:
- Train: Model training
- Validation: Hyperparameter tuning
- Test: Final evaluation (only run once!)

### 3. Error Analysis

Don't just report numbers:
- Qualitatively examine failures
- Categorize error types
- Identify systematic issues

### 4. Compare to Baselines

Report performance relative to:
- Random baseline
- Simple heuristic
- Previous SOTA
- Human performance (if available)

### 5. Statistical Significance

Report confidence intervals or p-values when possible:
- Multiple random seeds
- Bootstrap sampling
- Significance tests

### 6. Document Everything

- Model details (architecture, size, training data)
- Evaluation setup (prompts, hyperparameters)
- Random seeds
- Hardware (affects reproducibility)

## Emerging Directions

### Dynamic Benchmarks

Continuously updated to avoid contamination:
- LiveBench
- Chatbot Arena (Elo ratings from human votes)

### Capability-Focused Evals

Test specific abilities:
- Instruction following
- Multi-turn coherence
- Tool use
- Factual grounding

### Agent Benchmarks

Evaluate LLMs in agentic workflows:
- WebArena (web browsing tasks)
- SWE-bench (software engineering)
- GAIA (general AI assistant tasks)

### Multimodal Evaluation

Assess vision-language models:
- Image captioning
- Visual question answering
- OCR and document understanding

## Further Reading

- [HELM: Holistic Evaluation](https://crfm.stanford.edu/helm/)
- "Measuring Massive Multitask Language Understanding" (MMLU paper)
- "HumanEval: Evaluating Large Language Models Trained on Code"
- "TruthfulQA: Measuring How Models Mimic Human Falsehoods"
- [Chatbot Arena](https://chat.lmsys.org/) - Human preference leaderboard
- "Judging LLM-as-a-Judge" (Zheng et al., 2023)
