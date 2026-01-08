# Attention Mechanisms in Transformers

## Core Intuition

Attention allows a model to focus on relevant parts of the input when processing each position, rather than treating all inputs equally. Instead of fixed-weight connections, attention computes dynamic weights based on input content.

## Scaled Dot-Product Attention

### Mathematical Definition

Given queries $Q$, keys $K$, and values $V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:
- $d_k$ is the dimension of keys (scaling factor prevents softmax saturation)
- $QK^T$ computes similarity scores between queries and keys
- Softmax normalizes scores to attention weights
- Weighted sum of values produces output

### Why Scaling by $\sqrt{d_k}$?

Without scaling, dot products grow large in magnitude as dimensionality increases, pushing softmax into regions with extremely small gradients.

**Example**: If $d_k = 512$, dot products can be $\sim 10-20 \times$ larger than with $d_k = 64$, causing softmax to saturate.

### Computational Steps

1. **Score**: Compute $QK^T$ → attention scores $(n \times n)$ matrix
2. **Scale**: Divide by $\sqrt{d_k}$
3. **Mask** (optional): Set attention to $-\infty$ for invalid positions
4. **Normalize**: Apply softmax row-wise → attention weights
5. **Aggregate**: Multiply weights by values $V$ → weighted output

## Self-Attention

Each position attends to all positions in the same sequence.

### Process

```
Input: X with shape (seq_len, d_model)

1. Project to Q, K, V:
   Q = XW_Q  # (seq_len, d_k)
   K = XW_K  # (seq_len, d_k)
   V = XW_V  # (seq_len, d_v)

2. Compute attention:
   scores = QK^T / sqrt(d_k)  # (seq_len, seq_len)
   weights = softmax(scores)   # (seq_len, seq_len)
   output = weights @ V        # (seq_len, d_v)
```

### Interpretability

The attention weight matrix $(n \times n)$ shows which positions influence each other:
- Row $i$: what position $i$ attends to
- High weights = strong relevance

## Multi-Head Attention

### Motivation

Single attention focuses on one type of relationship. Multiple heads allow learning different aspects simultaneously:
- Head 1: Syntactic dependencies
- Head 2: Semantic relationships
- Head 3: Long-range dependencies
- etc.

### Architecture

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

### Dimensions

- Input: $d_{\text{model}}$ (e.g., 512)
- Number of heads: $h$ (e.g., 8)
- Per-head dimension: $d_k = d_v = d_{\text{model}} / h$ (e.g., 64)
- Total parameters same as single large head, but more expressive

### Implementation Detail

Efficiently computed with batched matrix operations:
1. Project to $Q, K, V$ with shape $(n, d_{\text{model}})$
2. Reshape to $(h, n, d_k)$ - split heads
3. Compute attention in parallel for all heads
4. Reshape back to $(n, h \cdot d_k)$
5. Project with $W^O$ to $(n, d_{\text{model}})$

## Attention Variants

### Causal (Masked) Self-Attention

**Purpose**: Autoregressive generation (GPT family)

**Mechanism**: Position $i$ can only attend to positions $\leq i$

```python
mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))
```

**Effect**: Future tokens are invisible during training, matching inference

### Cross-Attention

**Purpose**: Attend from one sequence to another (encoder-decoder)

**Mechanism**:
- Queries from decoder
- Keys and values from encoder
- Decoder conditions on encoder's output

**Use cases**: Translation, summarization, image captioning

### Bidirectional Self-Attention

**Purpose**: Full context understanding (BERT family)

**Mechanism**: Each position sees all positions (no masking)

**Use cases**: Classification, sentence embeddings, MLM pretraining

## Positional Information

Attention has **no inherent notion of position** - it's permutation equivariant.

### Solutions

1. **Positional Encoding**: Add position signals to embeddings
   - Sinusoidal: $PE_{pos, 2i} = \sin(pos / 10000^{2i/d})$
   - Learned: Trainable position embeddings

2. **Relative Positional Encoding**: Bias attention based on distance
   - Used in Transformer-XL, T5, DeBERTa

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Attention scores | $O(n^2 \cdot d)$ | Bottleneck for long sequences |
| Softmax | $O(n^2)$ | |
| Memory | $O(n^2)$ | Stores attention matrix |

For $n = 512, d = 512$: Attention is ~260K operations per layer

### Long Sequence Challenges

- Quadratic scaling makes $n > 10K$ infeasible
- Solutions: Sparse attention, linear attention, chunked attention

## Key Properties

1. **Dynamic routing**: Weights depend on input content
2. **Parallelizable**: All positions computed simultaneously (unlike RNNs)
3. **Long-range dependencies**: Direct connections between distant positions
4. **Permutation equivariant**: Without positional encoding, order-agnostic

## Practical Tips

- **Dropout**: Apply to attention weights for regularization
- **Attention visualization**: Useful for debugging and interpretation
- **Gradient flow**: Attention provides multiple paths for gradients
- **Initialization**: Careful initialization of $W^Q, W^K, W^V$ important

## Further Reading

- "Attention Is All You Need" (Vaswani et al., 2017)
- "The Annotated Transformer" (Rush, 2018)
