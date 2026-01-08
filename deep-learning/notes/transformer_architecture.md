# Transformers

**The Architecture That Changed Everything**

Transformers introduced the **attention mechanism** as the core building block, replacing recurrence entirely. This enabled parallel processing, better long-range dependencies, and ultimately powered GPT, BERT, and the modern AI revolution.

---

## Table of Contents

1. [The Paradigm Shift](#the-paradigm-shift)
2. [Attention Mechanism](#attention-mechanism)
3. [Self-Attention](#self-attention)
4. [Multi-Head Attention](#multi-head-attention)
5. [Positional Encoding](#positional-encoding)
6. [Transformer Architecture](#transformer-architecture)
7. [Training Transformers](#training-transformers)
8. [Variants & Evolution](#variants--evolution)
9. [Applications](#applications)
10. [Implementation Details](#implementation-details)
11. [Best Practices](#best-practices)

---

## The Paradigm Shift

### Before Transformers: The Sequential Bottleneck

**RNN/LSTM approach** (2014-2017):
```
Process left-to-right:
h₁ → h₂ → h₃ → h₄ → ...
```

**Problems**:
1. **Sequential processing**: Must compute h₁ before h₂ (can't parallelize)
2. **Long sequences**: Information loss over many steps despite LSTM
3. **Attention bolted on**: Attention was added to RNNs, not core mechanism
4. **Slow**: GPUs underutilized (sequential = inefficient)

### The Transformer Breakthrough (2017)

**Paper**: "Attention Is All You Need" (Vaswani et al.)

**Core insight**: **Attention is sufficient** - no recurrence needed!

**Key innovations**:
1. **Self-attention**: Direct connections between all positions
2. **Parallel processing**: All positions computed simultaneously
3. **Scalability**: Leverages modern hardware (GPUs/TPUs)
4. **Better long-range dependencies**: Direct path between any two positions

**Impact**:
- BERT (2018): Breakthrough in NLP understanding
- GPT-2/3/4 (2019-2023): Language generation revolution
- Vision Transformers (2020): Transformers beat CNNs on vision
- Now: Dominant architecture across ML (NLP, vision, audio, multimodal)

---

## Attention Mechanism

### The Core Idea

**Question**: How do you know what to focus on?

**Example**: Translating "I ate an apple"
- When translating "ate" to French ("mangé"), you need to look at:
  - "I" (for subject agreement: "j'ai")
  - "ate" itself (for the verb)
  - "apple" (what was eaten)

**Attention**: Learn which parts of input are relevant for each output

### Attention Formula

**The fundamental equation**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Components**:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I contain?"
- **V** (Value): "What information do I have?"
- **d_k**: Dimension of keys (for scaling)

**Process**:
1. **Compute similarity**: Q × K^T (how much does query match each key?)
2. **Scale**: Divide by √d_k (prevent large values in softmax)
3. **Normalize**: softmax (convert to probability distribution)
4. **Weight values**: Multiply by V (get weighted sum of values)

### Intuition: Database Lookup

Think of attention as a **soft database lookup**:

**Traditional database**:
```sql
SELECT value FROM table WHERE key = "exact_match"
```

**Attention**:
```
For each query:
  1. Compare query to all keys (how similar?)
  2. Weight each value by similarity
  3. Return weighted sum (soft lookup, not hard)
```

**Example**:
```
Query: "capital of France"
Keys: ["Paris info", "London info", "Berlin info"]
Values: [Paris details, London details, Berlin details]

Attention weights: [0.9, 0.05, 0.05]  ← Paris info is most relevant
Output: 0.9 × Paris_details + 0.05 × London_details + 0.05 × Berlin_details
        ≈ Paris_details
```

---

## Self-Attention

### What Makes It "Self"?

**Self-attention**: Q, K, V all come from the **same sequence**

**Process**: Each position attends to all positions (including itself)

```
Sentence: "The cat sat on the mat"

When processing "sat":
- Attend to "The" (low weight)
- Attend to "cat" (high weight - subject!)
- Attend to "sat" (medium weight - self)
- Attend to "on" (medium weight)
- Attend to "the" (low weight)
- Attend to "mat" (medium weight - object)
```

### Self-Attention Step-by-Step

**Input**: Sequence of word embeddings
```
X = [x₁, x₂, x₃, ..., x_n]  ← Each x_i is a d-dimensional vector
```

**Step 1**: Create Q, K, V from input
```
Q = X × W_Q  ← Query matrix
K = X × W_K  ← Key matrix
V = X × W_V  ← Value matrix
```

Where W_Q, W_K, W_V are learned weight matrices.

**Step 2**: Compute attention scores
```
Scores = Q × K^T / √d_k
```

This gives an n×n matrix:
```
      k₁    k₂    k₃   ...  k_n
q₁  [0.9   0.1   0.0  ...  0.0]
q₂  [0.2   0.7   0.1  ...  0.0]
q₃  [0.1   0.8   0.1  ...  0.0]
...
q_n [0.0   0.0   0.1  ...  0.9]
```

Each row: How much position i attends to each position j

**Step 3**: Apply softmax (row-wise)
```
Attention_weights = softmax(Scores)  ← Each row sums to 1
```

**Step 4**: Weight the values
```
Output = Attention_weights × V
```

Each output position is a weighted combination of all value vectors.

### Why Scaling by √d_k?

**Problem**: Without scaling, dot products can be very large

**Example**: Two random d-dimensional vectors
- Dot product variance ≈ d
- For d=512, dot products can be ~20-30
- Softmax saturates (one element ≈ 1, others ≈ 0)
- Gradients vanish!

**Solution**: Divide by √d_k
- Brings variance back to ~1
- Softmax remains smooth
- Better gradient flow

### Self-Attention Properties

**1. Permutation Equivariant**
- Change input order → output order changes identically
- No inherent notion of position (need positional encoding!)

**2. Parallel Computation**
- All attention scores computed simultaneously
- Huge speedup on GPUs vs sequential RNNs

**3. Constant Path Length**
- Position 1 to position 100: one attention step
- RNN: 99 sequential steps
- Better for long-range dependencies

**4. Computational Complexity**
- Time: O(n²·d) where n = sequence length, d = dimension
- Space: O(n²) for attention matrix
- Bottleneck for very long sequences!

---

## Multi-Head Attention

### Why Multiple Heads?

**Problem**: Single attention focuses on one aspect

**Example**: In "The cat sat on the mat"
- Syntactic attention: cat → sat (subject-verb)
- Semantic attention: sat → mat (action-location)
- Need to capture **multiple relationships**!

**Solution**: Multiple attention "heads" in parallel

### Multi-Head Attention Formula

```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., head_h) × W_O

where head_i = Attention(Q×W_Q^i, K×W_K^i, V×W_V^i)
```

**Process**:
1. **Split into h heads**: Use different W_Q, W_K, W_V for each head
2. **Compute attention** for each head independently
3. **Concatenate** all head outputs
4. **Linear projection**: W_O mixes information from all heads

### Dimensionality

**Typical setup**:
- Model dimension: d_model = 512
- Number of heads: h = 8
- Each head dimension: d_k = d_v = d_model / h = 64

**Why smaller dimensions per head?**
- Total computation same as single head with d_model
- More heads = more diverse attention patterns

**Parameters**:
- Each head: 3 × (d_model × d_k) for W_Q, W_K, W_V
- Output projection: d_model × d_model for W_O
- Total: 4 × d_model²

### What Do Different Heads Learn?

**Empirical observations** (from BERT, GPT studies):

**Head 1**: Might learn syntactic relationships
- Subject → Verb
- Adjective → Noun

**Head 2**: Might learn semantic relationships
- Word → Similar words
- Pronoun → Referent

**Head 3**: Might learn positional patterns
- Current word → Next word
- Beginning → End of phrase

**Head 4-8**: Various other patterns

**Key**: Model learns diverse attention patterns automatically!

---

## Positional Encoding

### The Position Problem

**Problem**: Self-attention is permutation equivariant
- "I love cats" and "cats love I" produce same representation!
- No notion of word order

**Need**: Inject position information

### Learned vs Fixed Positional Encodings

**Option 1: Learned** (used in BERT)
```
Position_embedding[i] = lookup_table[i]  ← Learned during training
```
- Pro: Model learns best encoding
- Con: Fixed maximum sequence length

**Option 2: Fixed sinusoidal** (used in original Transformer)
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
- Pro: Can extrapolate to longer sequences
- Con: Not necessarily optimal

### Why Sinusoidal?

**Properties**:
1. **Unique encoding** for each position
2. **Relative positions**: PE(pos+k) can be expressed as linear function of PE(pos)
3. **Smooth**: Nearby positions have similar encodings
4. **Different frequencies**: Each dimension encodes position at different scale
   - Low dimensions: Vary slowly (capture long-range position)
   - High dimensions: Vary quickly (capture short-range position)

### How It's Used

```python
# Word embeddings
X = embedding_layer(input_tokens)  # Shape: (seq_len, d_model)

# Add positional encoding
X = X + positional_encoding[:seq_len]  # Element-wise addition

# Now X has both content and position information
```

**Alternative**: Concatenate instead of add (less common)

---

## Transformer Architecture

### The Full Architecture

**Two main components**:
1. **Encoder**: Processes input sequence
2. **Decoder**: Generates output sequence

```
Input Sequence → [Encoder] → Context
                             ↓
Output Sequence ← [Decoder] ← Context
```

### Encoder

**Single encoder layer**:
```
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Normalize  ← Residual connection
  ↓
Feed-Forward Network
  ↓
Add & Normalize  ← Residual connection
  ↓
Output
```

**Complete encoder**: Stack N encoder layers (typically N=6)

### Decoder

**Single decoder layer**:
```
Input (shifted right)
  ↓
Masked Multi-Head Self-Attention  ← Can't look at future tokens!
  ↓
Add & Normalize
  ↓
Multi-Head Cross-Attention  ← Attend to encoder output
  ↓
Add & Normalize
  ↓
Feed-Forward Network
  ↓
Add & Normalize
  ↓
Output
```

**Complete decoder**: Stack N decoder layers (typically N=6)

### Key Components

**1. Feed-Forward Network (FFN)**
```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
```
- Two linear layers with ReLU in between
- Applied to each position **independently** (same network for all positions)
- Typical dimension: d_model → 4×d_model → d_model (2048 for d_model=512)

**2. Layer Normalization**
```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β
```
- Normalize across features (not batch like BatchNorm)
- Stabilizes training, allows deeper networks

**3. Residual Connections**
```
Output = LayerNorm(x + Sublayer(x))
```
- Helps gradient flow
- Allows very deep networks (6-24+ layers)

**4. Masked Attention (Decoder)**
```
When generating word i, can only attend to positions 1...i-1
(Can't cheat by looking at future!)
```
Implemented by setting future positions to -∞ before softmax:
```
Mask:  [ 0   -∞  -∞  -∞]
       [ 0    0  -∞  -∞]
       [ 0    0   0  -∞]
       [ 0    0   0   0 ]
```

### Full Architecture Diagram

```
Input: "Hello world"              Output: "Bonjour monde"

[Embedding + Positional]          [Embedding + Positional]
        ↓                                   ↓
    [Encoder Layer 1]               [Decoder Layer 1]
        ↓                           ← Cross-Attention ←
    [Encoder Layer 2]               [Decoder Layer 2]
        ↓                           ← Cross-Attention ←
        ...                                 ...
        ↓                                   ↓
    [Encoder Layer 6] ────────────→ [Decoder Layer 6]
                                            ↓
                                    [Linear + Softmax]
                                            ↓
                                      "Bonjour monde"
```

---

## Training Transformers

### Loss Function

**Machine Translation** (original Transformer):
```
Loss = CrossEntropy(predicted_tokens, target_tokens)
```
Averaged over all positions in sequence.

**Language Modeling** (GPT):
```
Loss = -log P(token_i | token_1, ..., token_{i-1})
```
Predict next token given previous tokens.

**Masked Language Modeling** (BERT):
```
Loss = -log P(masked_token | context)
```
Predict masked tokens given surrounding context.

### Teacher Forcing

**During training**: Use ground-truth previous tokens (not model's predictions)

```
Target: "Bonjour monde"

Step 1: Input: <START>     → Predict: Bonjour ✓
Step 2: Input: Bonjour     → Predict: monde   ✓
Step 3: Input: monde       → Predict: <END>   ✓
```

**Why?**: Faster convergence, more stable training

**During inference**: Use model's own predictions (autoregressive generation)

### Warmup + Learning Rate Scheduling

**Original Transformer schedule**:
```
LR = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
```

**Intuition**:
1. **Warmup phase**: Gradually increase LR (first 4000 steps)
   - Prevents instability at start
2. **Decay phase**: Gradually decrease LR
   - Fine-tune parameters

**Modern approach**: Often use AdamW with cosine annealing

### Regularization

**1. Dropout**
- After attention (before Add & Norm)
- After FFN (before Add & Norm)
- On embeddings
- Typical rate: 0.1

**2. Label Smoothing**
```
Instead of [0, 0, 1, 0]:  (one-hot)
Use:        [0.025, 0.025, 0.9, 0.025]  (smoothed)
```
- Prevents overconfidence
- Better generalization

**3. Weight Decay**
- L2 regularization on parameters
- Typical: 0.01

---

## Variants & Evolution

### Encoder-Only: BERT (2018)

**Architecture**: Just the encoder stack

**Key innovations**:
1. **Bidirectional**: Can attend to both left and right context
2. **Masked Language Modeling**: Predict masked words
3. **Next Sentence Prediction**: Predict if sentences are consecutive

**Use case**: Understanding tasks (classification, QA, NER)

**Descendants**: RoBERTa, ALBERT, ELECTRA, DeBERTa

### Decoder-Only: GPT (2018-2023)

**Architecture**: Just the decoder stack (no encoder, no cross-attention)

**Key innovations**:
1. **Causal masking**: Only attend to previous tokens
2. **Autoregressive generation**: Predict next token
3. **Scaling**: GPT-2 (1.5B), GPT-3 (175B), GPT-4 (rumored ~1T)

**Use case**: Generation tasks (text completion, dialogue, code)

**Descendants**: GPT-2, GPT-3, GPT-4, LLaMA, PaLM

### Encoder-Decoder: T5, BART (2019-2020)

**Architecture**: Full Transformer (both encoder and decoder)

**Key idea**: Frame all tasks as text-to-text

**Examples**:
- Translation: "translate English to French: Hello" → "Bonjour"
- Summarization: "summarize: [long text]" → "[summary]"
- Classification: "sentiment: Great movie!" → "positive"

**Use case**: Versatile, can handle any NLP task

### Vision Transformers (ViT) (2020)

**Key insight**: Images are sequences of patches!

**Process**:
1. Split image into patches (e.g., 16×16 pixels)
2. Flatten each patch to a vector
3. Add positional embeddings
4. Feed to standard Transformer encoder

**Result**: Beat CNNs on ImageNet with sufficient data!

**Impact**: Transformers now dominant in vision too

### Efficient Transformers

**Problem**: O(n²) attention is expensive for long sequences

**Solutions**:

**1. Sparse Attention** (e.g., Longformer)
- Only attend to nearby tokens + a few global tokens
- Reduces to O(n)

**2. Linear Attention** (e.g., Performer, Linformer)
- Approximate attention with linear complexity
- O(n) instead of O(n²)

**3. Memory-based** (e.g., Transformer-XL)
- Cache previous segments
- Attend to cache + current segment

### Multimodal Transformers

**CLIP** (2021): Vision + Language
- Jointly train image and text encoders
- Match images to captions

**Flamingo, GPT-4V**: Vision-Language generation
- Generate text from images

**Whisper**: Speech Transformer
- Audio → Text transcription

---

## Applications

### Natural Language Processing

**1. Machine Translation**
- Original Transformer application
- SOTA: Google Translate, DeepL

**2. Text Generation**
- GPT models: Write essays, code, dialogue
- ChatGPT: Conversational AI

**3. Question Answering**
- BERT-based models: Extract answers from context
- T5: Generative QA

**4. Summarization**
- BART, T5, PEGASUS
- Extract key information

**5. Sentiment Analysis, NER, etc.**
- BERT fine-tuning
- Classification tasks

### Computer Vision

**1. Image Classification**
- Vision Transformers (ViT)
- Beats CNNs with enough data

**2. Object Detection**
- DETR (Detection Transformer)
- No anchor boxes, end-to-end

**3. Segmentation**
- Segmenter, SegFormer
- Transformer-based segmentation

### Speech & Audio

**1. Speech Recognition**
- Whisper: Multi-language transcription
- Wav2Vec: Self-supervised speech

**2. Text-to-Speech**
- FastSpeech 2, VITS
- Natural-sounding synthesis

### Multimodal

**1. Image Captioning**
- ViT encoder + GPT decoder
- Describe images in natural language

**2. Visual Question Answering**
- Answer questions about images
- CLIP-based models

**3. Video Understanding**
- TimeSformer, VideoMAE
- Action recognition, captioning

### Science & Beyond

**1. Protein Structure**
- AlphaFold 2: Uses attention for structure prediction
- Revolutionary impact on biology

**2. Drug Discovery**
- Molecule generation, property prediction

**3. Game Playing**
- Decision Transformers: RL with Transformers
- Gato: Generalist agent

---

## Implementation Details

### Typical Hyperparameters

**Original Transformer** (base model):
- Layers: 6 (encoder + decoder)
- d_model: 512
- d_ff: 2048
- Heads: 8
- d_k = d_v: 64
- Dropout: 0.1
- Parameters: ~65M

**BERT-base**:
- Layers: 12 (encoder only)
- d_model: 768
- d_ff: 3072
- Heads: 12
- Parameters: ~110M

**GPT-3**:
- Layers: 96 (decoder only)
- d_model: 12,288
- d_ff: 49,152
- Heads: 96
- Parameters: 175B

### Memory Complexity

**Attention matrix**: O(n² × d_model)
- For n=512, d=512: ~512MB per batch element
- For n=2048, d=512: ~8GB per batch element!

**Tricks to reduce**:
1. Gradient checkpointing: Recompute activations during backward pass
2. Mixed precision (FP16): Half memory, 2-3× speedup
3. Gradient accumulation: Simulate large batch with small batches

### Inference

**Autoregressive generation** (GPT-style):
```python
for i in range(max_length):
    output = model(input[:i])  # Process all tokens so far
    next_token = sample(output[-1])  # Sample next token
    input = append(input, next_token)
```

**Problem**: Quadratic in sequence length (reprocess all previous tokens)

**Solution**: **KV caching**
- Cache key and value matrices from previous steps
- Only compute Q, K, V for new token
- Dramatically faster inference

---

## Best Practices

### Architecture Choices

**1. Encoder-only** (BERT-style):
- For: Classification, NER, QA (understanding)
- Pro: Bidirectional context
- Con: Can't generate sequences

**2. Decoder-only** (GPT-style):
- For: Text generation, language modeling
- Pro: Simple, scales well
- Con: Only left context (causal)

**3. Encoder-decoder** (T5-style):
- For: Translation, summarization, general seq2seq
- Pro: Versatile, bidirectional encoder + flexible decoder
- Con: More parameters, more complex

### Hyperparameter Tuning

**Start with standard values**:
- Layers: 6-12 (small/medium data), 12-24+ (large data)
- d_model: 512 (base), 768-1024 (large)
- Heads: 8-16
- Dropout: 0.1

**Scale up carefully**:
- More layers > wider layers (usually)
- More heads helps (up to a point)
- Bigger models need more data

### Training Tips

**1. Warmup is critical**
- 4000-8000 steps typical
- Prevents early instability

**2. Batch size**
- Larger is better (if memory allows)
- Use gradient accumulation if needed

**3. Mixed precision**
- FP16 training: 2-3× speedup
- Watch for numerical instability

**4. Gradient clipping**
- Clip norm to 1.0
- Prevents exploding gradients

### Fine-Tuning Pre-trained Models

**Don't train from scratch!** (unless you have massive data)

**Approach**:
1. Load pre-trained model (BERT, GPT, T5)
2. Add task-specific head
3. Fine-tune on your data

**Learning rate**:
- Much smaller than pre-training (1e-5 to 5e-5)
- Warmup helps

**Freezing layers**:
- Freeze early layers (keep general features)
- Fine-tune later layers (task-specific)

---

## Key Takeaways

1. **Attention is All You Need**
   - Replaced recurrence with attention
   - Parallel processing, better long-range dependencies

2. **Self-Attention is Magic**
   - Each position attends to all positions
   - Learns what to focus on automatically

3. **Multi-Head = Multiple Perspectives**
   - Different heads learn different relationships
   - Richer representations

4. **Position Encoding is Essential**
   - Self-attention has no notion of order
   - Sinusoidal or learned embeddings inject position

5. **Scaling Works**
   - Bigger models, more data → better performance
   - GPT-3 (175B) > GPT-2 (1.5B) > GPT-1 (117M)

6. **Transformers Are Universal**
   - Started in NLP, now everywhere
   - Vision, speech, multimodal, protein folding, ...

7. **Efficiency Matters**
   - O(n²) attention is bottleneck
   - Sparse/linear attention for long sequences

---

## The Transformer Revolution

**2017**: "Attention Is All You Need" paper
- Beat RNNs on machine translation

**2018**: BERT + GPT-1
- Pre-training + fine-tuning paradigm
- Transformers dominate NLP

**2019-2020**: Scaling up
- GPT-2, GPT-3, T5, BART
- Bigger models, more data, better results

**2020-2021**: Beyond NLP
- Vision Transformers (ViT), CLIP, DALL-E
- Multimodal models

**2022-2023**: Foundation models era
- GPT-4, Gemini, LLaMA, Claude
- Transformers power ChatGPT, Copilot, etc.
- Transformers are the foundation of modern AI

**Why Transformers won**:
1. Scalability (parallel processing)
2. Expressiveness (attention captures complex relationships)
3. Transfer learning (pre-train once, fine-tune for many tasks)
4. Simplicity (one architecture for many domains)

---

## Next Steps

**To truly understand Transformers**:
1. ✅ Read these notes
2. 🔨 **Implement attention from scratch** (NumPy/PyTorch)
3. 🔨 **Build a mini-Transformer** for simple translation
4. 🔨 **Visualize attention weights** (see what model attends to)
5. 🔨 **Fine-tune BERT** on text classification
6. 🔨 **Generate text with GPT-2** (HuggingFace Transformers)
7. 📚 **Read the paper**: "Attention Is All You Need"

**Related topics**:
- Efficient Transformers (Sparse, Linear attention)
- Vision Transformers (ViT, DETR)
- Pre-training methods (MLM, CLM, denoising)
- Prompt engineering (for GPT-style models)
- RLHF (Reinforcement Learning from Human Feedback)

**The most important architecture in modern AI** - master it!
