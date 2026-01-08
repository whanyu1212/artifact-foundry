# LLM Pretraining

## Overview

Pretraining is the unsupervised (or self-supervised) learning phase where a language model learns general language understanding from vast amounts of unlabeled text. The learned representations transfer to downstream tasks with little task-specific data.

## Core Paradigm

```
Large Corpus (unlabeled) → Pretraining Objective → Pretrained Model → Fine-tuning (task-specific)
```

**Key insight**: Language modeling objectives force models to learn syntax, semantics, world knowledge, and reasoning as a byproduct of predicting text.

## Causal Language Modeling (CLM)

### Objective

Predict next token given previous tokens:

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

### Architecture

- **Autoregressive**: Left-to-right generation
- **Causal masking**: Position $t$ only sees tokens $< t$
- **Use case**: Text generation, dialogue, code completion

### Examples

- **GPT family** (GPT-1/2/3, GPT-4)
- **LLaMA**, **Mistral**, **Falcon**

### Training Process

```python
for batch in data:
    # Input:  [The, cat, sat, on]
    # Target: [cat, sat, on, mat]
    logits = model(batch['input_ids'])  # Shape: (batch, seq_len, vocab_size)
    loss = cross_entropy(logits[:, :-1], targets[:, 1:])  # Shift by one
    loss.backward()
    optimizer.step()
```

### Advantages

- Natural for generation tasks
- Simple objective, scales well
- Emergent abilities with sufficient scale (in-context learning, reasoning)

### Limitations

- Can only condition on past context
- May struggle with tasks requiring bidirectional understanding

## Masked Language Modeling (MLM)

### Objective

Predict randomly masked tokens using bidirectional context:

$$\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P(x_t | x_{\backslash \mathcal{M}})$$

where $\mathcal{M}$ is the set of masked positions.

### Masking Strategy (BERT)

For each token selected for masking (15% of tokens):
- 80%: Replace with `[MASK]`
- 10%: Replace with random token
- 10%: Keep original

**Rationale**: Prevents model from only learning `[MASK]` representations; encourages robustness.

### Architecture

- **Bidirectional**: Each position sees full context
- **Encoder-only**: No causal masking
- **Use case**: Classification, sentence embeddings, NER

### Examples

- **BERT** (base, large, variants)
- **RoBERTa**: Optimized BERT training
- **ALBERT**: Parameter-efficient BERT
- **DeBERTa**: Enhanced position encoding

### Advantages

- Full context understanding
- Excellent for discriminative tasks
- More sample-efficient for understanding tasks

### Limitations

- Not designed for generation (no left-to-right ordering)
- Mismatch between pretraining (`[MASK]`) and fine-tuning (no masks)

## Prefix/Span Language Modeling

### T5 Approach

Combines aspects of CLM and MLM:

1. Corrupt spans of text with sentinel tokens
2. Predict missing spans autoregressively

**Example**:
```
Input:  "Thank you for inviting <X> to your party <Y> week."
Output: "<X> me <Y> last </s>"
```

### Advantages

- Unified framework for generation and understanding
- More flexible than pure MLM or CLM
- Handles variable-length predictions

### Use Cases

- **T5**, **BART**, **PEGASUS**: seq2seq tasks (translation, summarization)

## Permutation Language Modeling

### XLNet Approach

Train on all permutations of factorization orders:

$$\mathcal{L} = \mathbb{E}_{z \sim Z_T} \left[ \sum_{t=1}^{T} \log P(x_{z_t} | x_{z_{<t}}) \right]$$

where $Z_T$ is all permutations of $[1, \ldots, T]$.

### Key Idea

- Avoid `[MASK]` token discrepancy
- Maintain autoregressive formulation
- Leverage bidirectional context through permutation

### Implementation

- Two-stream self-attention: content and query streams
- Complex training procedure

### Trade-offs

- Theoretically appealing but complex
- Didn't significantly outperform simpler approaches in practice
- Less widely adopted than BERT or GPT

## Multi-Task Pretraining

### Approach

Pretrain on diverse objectives simultaneously or sequentially:

1. **GLaM**: Mix of languages and modalities
2. **Gopher**: Code + text
3. **PaLM**: Multitask mixture

### Mixture of Denoising Experts (MoDE)

Different corruption strategies in training batches:
- Standard MLM
- Span corruption
- Document rotation
- Sentence shuffling

## Pretraining Data

### Scale Trends

| Model | Parameters | Training Tokens |
|-------|------------|-----------------|
| BERT-large | 340M | 3.3B |
| GPT-3 | 175B | 300B |
| PaLM | 540B | 780B |
| LLaMA 2 | 70B | 2T |

**Chinchilla scaling laws**: Models should be trained on ~20 tokens per parameter for optimal compute efficiency.

### Data Quality

Key considerations:
- **Deduplication**: Remove near-duplicates (improves downstream performance)
- **Filtering**: Remove low-quality, toxic, or biased content
- **Diversity**: Multiple domains, languages, styles
- **Currency**: Recent data for temporal knowledge

### Common Sources

- **Web crawls**: Common Crawl, C4
- **Books**: BookCorpus, Pile-Books3
- **Code**: GitHub, Stack Overflow
- **Wikipedia**: High-quality encyclopedic text
- **Academic**: ArXiv, PubMed

## Training Dynamics

### Learning Rate Schedule

Typical: Warmup + cosine decay
```
Initial: 0 → Peak (1-10% of training)
Main: Peak → Minimum (cosine decay)
Final: Low learning rate, long training (improves stability)
```

### Batch Size

- **Large batches**: Better gradient estimates, scale to many devices
- **Typical**: 2M-4M tokens per batch for large models
- **Gradient accumulation**: Simulate large batches with limited memory

### Computational Cost

GPT-3 (175B): ~3.14 × 10²³ FLOPs → thousands of GPU-months

### Challenges

- **Instability**: Loss spikes, divergence
- **Checkpoint selection**: Which checkpoint generalizes best?
- **Reproducibility**: Hardware, random seeds, data order matter

## Emergent Abilities

Capabilities that appear suddenly with scale:
- **In-context learning**: Few-shot prompting
- **Chain-of-thought reasoning**: Step-by-step problem solving
- **Instruction following**: Zero-shot task completion

Not fully understood theoretically, but empirically observed at sufficient scale (~100B+ parameters).

## Further Reading

- BERT: Devlin et al. (2019)
- GPT-2: Radford et al. (2019)
- GPT-3: Brown et al. (2020)
- T5: Raffel et al. (2020)
- Scaling Laws: Kaplan et al. (2020), Hoffmann et al. (2022)
