# Recurrent Neural Networks (RNNs)

**The Architecture for Sequential Data**

RNNs introduced the concept of memory to neural networks, enabling them to process sequences of arbitrary length by maintaining a hidden state that captures information from previous time steps.

---

## Table of Contents

1. [Core Intuition](#core-intuition)
2. [The Recurrence Mechanism](#the-recurrence-mechanism)
3. [Vanilla RNN Architecture](#vanilla-rnn-architecture)
4. [The Vanishing Gradient Problem](#the-vanishing-gradient-problem)
5. [LSTM: Long Short-Term Memory](#lstm-long-short-term-memory)
6. [GRU: Gated Recurrent Unit](#gru-gated-recurrent-unit)
7. [Training RNNs](#training-rnns)
8. [Common Architectures & Patterns](#common-architectures--patterns)
9. [Applications](#applications)
10. [RNNs vs Transformers](#rnns-vs-transformers)
11. [Best Practices](#best-practices)

---

## Core Intuition

### The Sequential Data Problem

**Question**: How do you process sequences where order matters?

**Examples**:
- Text: "I love cats" ≠ "cats love I"
- Time series: Stock prices, weather data
- Audio: Speech, music
- Video: Frames in temporal order

**Challenge**: Traditional neural networks:
- Fixed input size (can't handle variable-length sequences)
- No memory (treat each input independently)
- No notion of time or order

### The RNN Solution

**Key insight**: Maintain a **hidden state** that carries information across time steps

```
Traditional NN:  x → [NN] → y

RNN:  x₁ → [RNN] → y₁
           ↓  ↑
      x₂ → [RNN] → y₂
           ↓  ↑
      x₃ → [RNN] → y₃
```

**What's special**:
- Same network applied at each time step (parameter sharing)
- Hidden state `h` passed from t to t+1 (memory)
- Can process sequences of any length

**Analogy**: Reading a book
- Each word is an input
- Your understanding (hidden state) accumulates as you read
- Previous context helps understand current word

---

## The Recurrence Mechanism

### Unfolding Through Time

A single RNN cell, unfolded across time steps:

```
       h₀ ──→ h₁ ──→ h₂ ──→ h₃
        ↑      ↑      ↑      ↑
        │      │      │      │
       [RNN]  [RNN]  [RNN]  [RNN]  ← Same weights!
        ↑      ↑      ↑      ↑
        │      │      │      │
       x₁     x₂     x₃     x₄
```

**Key components**:
- **x_t**: Input at time t
- **h_t**: Hidden state at time t (the "memory")
- **y_t**: Output at time t (optional, depending on task)
- **Same weights** used at each time step (parameter sharing)

### The Recurrence Equation

At each time step t:

```
h_t = f(h_{t-1}, x_t)
```

More explicitly:
```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

Where:
- **W_hh**: Hidden-to-hidden weights (how past affects present)
- **W_xh**: Input-to-hidden weights (how input affects hidden state)
- **W_hy**: Hidden-to-output weights (how hidden state produces output)
- **tanh**: Non-linearity (squashes values to [-1, 1])

**Interpretation**:
- `h_{t-1}`: What we remember from the past
- `x_t`: New information arriving now
- `h_t`: Updated memory (combines old memory + new input)

---

## Vanilla RNN Architecture

### Forward Propagation

For each time step t = 1 to T:

```python
h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
y_t = W_hy @ h_t + b_y
```

**Dimensions**:
- `x_t`: (input_size,)
- `h_t`: (hidden_size,)
- `y_t`: (output_size,)
- `W_xh`: (hidden_size, input_size)
- `W_hh`: (hidden_size, hidden_size)
- `W_hy`: (output_size, hidden_size)

### Backpropagation Through Time (BPTT)

**Challenge**: Network is unfolded across time, must backprop through all steps

**Process**:
1. Compute forward pass for entire sequence (t = 1 to T)
2. Compute loss at final step (or all steps, depending on task)
3. Backpropagate from T back to 1

**Gradient flow**:
```
Loss ← y_T ← h_T ← h_{T-1} ← ... ← h_1 ← x_1
```

**Key equation**: Gradient of h_t depends on gradient of h_{t+1}
```
∂L/∂h_t = ∂L/∂h_{t+1} · ∂h_{t+1}/∂h_t + ∂L/∂y_t · ∂y_t/∂h_t
```

**The problem**: Gradients must flow through many time steps
- Many matrix multiplications
- Gradients can vanish or explode (see next section)

### Types of RNN Architectures

**1. One-to-One** (Standard NN, not really RNN)
```
x → [RNN] → y
```
Example: Image classification (single image → single label)

**2. One-to-Many** (Sequence generation)
```
x → [RNN] → y₁
       ↓
     [RNN] → y₂
       ↓
     [RNN] → y₃
```
Example: Image captioning (image → sequence of words)

**3. Many-to-One** (Sequence classification)
```
x₁ → [RNN] →
       ↓
x₂ → [RNN] →
       ↓
x₃ → [RNN] → y
```
Example: Sentiment analysis (sentence → positive/negative)

**4. Many-to-Many (Same length)** (Sequence labeling)
```
x₁ → [RNN] → y₁
       ↓
x₂ → [RNN] → y₂
       ↓
x₃ → [RNN] → y₃
```
Example: POS tagging (each word → part of speech)

**5. Many-to-Many (Different length)** (Seq2Seq)
```
Encoder:           Decoder:
x₁ → [RNN] →       c → [RNN] → y₁
       ↓                 ↓
x₂ → [RNN] → c    → [RNN] → y₂
       ↓                 ↓
x₃ → [RNN] →       → [RNN] → y₃
```
Example: Machine translation (English sentence → French sentence)

---

## The Vanishing Gradient Problem

### The Core Issue

**Problem**: Gradients diminish exponentially when backpropagating through time

**Why?** Consider gradient flowing from h_T back to h_1:
```
∂h_T/∂h_1 = ∂h_T/∂h_{T-1} · ∂h_{T-1}/∂h_{T-2} · ... · ∂h_2/∂h_1
```

Each term `∂h_t/∂h_{t-1}` involves `W_hh` (the recurrent weight matrix).

**Mathematical insight**:
```
∂h_t/∂h_{t-1} = diag(tanh'(z_t)) · W_hh
```

After T time steps:
```
∂h_T/∂h_1 ≈ (W_hh)^T · product of derivatives
```

**What happens**:
- If largest eigenvalue of `W_hh` < 1: **Gradients vanish** (→ 0)
- If largest eigenvalue of `W_hh` > 1: **Gradients explode** (→ ∞)
- `tanh'(z) ≤ 1`, further contributing to vanishing

**Consequence**:
- Can't learn long-term dependencies
- Network "forgets" information from many steps ago
- Example: "The cat, which was very fluffy and enjoyed playing with yarn, ___ hungry"
  - Need to remember "cat" (singular) to predict "was" (not "were")
  - If "cat" is 20+ words back, vanilla RNN struggles

### Gradient Clipping (for exploding gradients)

**Solution**: Cap gradient magnitude
```python
if gradient_norm > threshold:
    gradient = gradient * (threshold / gradient_norm)
```

**Works for**: Gradient explosion
**Doesn't help**: Vanishing gradients (can't make 0 bigger by clipping)

---

## LSTM: Long Short-Term Memory

**The breakthrough solution to vanishing gradients**

### Core Idea: Cell State

**Key innovation**: Separate **cell state** (long-term memory) from **hidden state** (working memory)

```
Cell state: c_t ────────────────────────→ c_{t+1}
                    (highway for gradients)

Hidden state: h_t → [LSTM] → h_{t+1}
```

**Why it works**:
- Cell state flows through time with minimal transformations
- Gradients can flow backward without vanishing
- "Information highway" that preserves long-term dependencies

### The Three Gates

LSTM controls information flow with three **gates** (learned neural networks):

**1. Forget Gate** (f_t): What to forget from cell state
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```
- Output: 0 (forget completely) to 1 (remember completely)
- Decides what information to discard from c_{t-1}

**2. Input Gate** (i_t): What new information to add
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  ← Candidate values
```
- `i_t`: How much to update (0 to 1)
- `c̃_t`: What values to add (new candidate memory)

**3. Output Gate** (o_t): What to output
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
```
- Decides what part of cell state to output as hidden state

### LSTM Forward Pass

Complete LSTM equations:

```python
# Forget gate: what to keep from old cell state
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)

# Input gate: what new info to add
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
c̃_t = tanh(W_c @ [h_{t-1}, x_t] + b_c)

# Update cell state
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
     ^^^^^^^^^^^^^^^^   ^^^^^^^^^^
     forget old memory  add new memory

# Output gate: what to reveal
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(c_t)
```

**Symbols**:
- `σ`: Sigmoid (outputs 0 to 1, acts as gate)
- `⊙`: Element-wise multiplication
- `[h_{t-1}, x_t]`: Concatenation of previous hidden state and current input

### Intuition: The Gates at Work

**Example**: Processing "The cat sat on the mat"

**At "cat"**:
- Input gate: "This is important!" (opens to store "cat" info)
- Cell state: Stores subject="cat", singular

**At "sat"**:
- Forget gate: "Keep subject info" (doesn't forget "cat")
- Output gate: "Use subject for verb agreement"

**At "mat"**:
- Forget gate: "Sentence ending, can forget subject" (starts to forget)

**Why it works**:
- Gates learn WHAT to remember and for HOW LONG
- Cell state preserves info across many time steps
- Gradient flows through cell state without vanishing

### LSTM Parameters

**Weight matrices** (8 in total):
- Forget gate: W_f, b_f
- Input gate: W_i, b_i, W_c, b_c
- Output gate: W_o, b_o

**Parameter count** (hidden_size = H, input_size = I):
```
Each gate: (H + I) × H  weights + H biases
Total: 4 × [(H + I) × H + H] = 4 × (H² + I×H + H)
```

**Much more parameters** than vanilla RNN, but worth it for long sequences!

---

## GRU: Gated Recurrent Unit

**Simplified LSTM** with fewer parameters

### Key Differences from LSTM

1. **No separate cell state** (only hidden state h_t)
2. **Two gates instead of three**:
   - Update gate (z_t): Combines forget & input gates
   - Reset gate (r_t): Controls how much past info to use

### GRU Equations

```python
# Reset gate: how much past to use
r_t = sigmoid(W_r @ [h_{t-1}, x_t] + b_r)

# Update gate: how much to update
z_t = sigmoid(W_z @ [h_{t-1}, x_t] + b_z)

# Candidate hidden state
h̃_t = tanh(W_h @ [r_t ⊙ h_{t-1}, x_t] + b_h)

# Final hidden state (interpolate between old and new)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
     ^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^
     keep old hidden state    add new candidate
```

### LSTM vs GRU

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| States | 2 (cell, hidden) | 1 (hidden) |
| Parameters | ~4×(H²+I×H) | ~3×(H²+I×H) |
| Performance | Slightly better for long sequences | Similar, faster training |
| When to use | Very long dependencies | Default choice, faster |

**Rule of thumb**:
- Start with GRU (fewer parameters, faster)
- Try LSTM if GRU doesn't work well
- Performance difference usually small in practice

---

## Training RNNs

### Backpropagation Through Time (BPTT)

**Full BPTT**: Backprop through entire sequence
- Memory intensive (must store all hidden states)
- Slow for long sequences

**Truncated BPTT**: Backprop only k steps back
```python
for i in range(0, sequence_length, k):
    forward_pass(sequence[i:i+k])
    backward_pass(k_steps)  # Only k steps
    detach_hidden_state()   # Break gradient flow
```

**Tradeoff**: Speed vs learning long-term dependencies

### Loss Functions

**Sequence classification** (many-to-one):
```
Loss = CrossEntropy(y_predicted, y_true)  # Only at final step
```

**Sequence generation** (one-to-many, many-to-many):
```
Loss = (1/T) × Σ_t CrossEntropy(y_t, target_t)  # Average over all steps
```

**Sequence labeling** (many-to-many, same length):
```
Loss = (1/T) × Σ_t CrossEntropy(y_t, target_t)
```

### Optimization Challenges

**1. Gradient clipping** (for exploding gradients):
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**2. Learning rate scheduling**:
- RNNs sensitive to learning rate
- Often need to reduce LR during training

**3. Initialization**:
- Initialize recurrent weights carefully
- Orthogonal initialization for W_hh (helps stability)

**4. Regularization**:
- Dropout: Apply to non-recurrent connections (not between time steps)
- Recurrent dropout: Dropout on h_{t-1} → h_t (use same mask across time)

---

## Common Architectures & Patterns

### Bidirectional RNNs

**Idea**: Process sequence in both directions

```
Forward:  h₁→ → h₂→ → h₃→
Backward: h₁← ← h₂← ← h₃←

Output: [h_t→, h_t←] (concatenation)
```

**Use cases**:
- When entire sequence is available (not online/streaming)
- POS tagging, named entity recognition
- Machine translation (encoder)

**Parameters**: 2× regular RNN (two separate RNNs)

### Encoder-Decoder (Seq2Seq)

**Architecture**: Two RNNs, one encodes input, one decodes to output

```
Encoder:          Context    Decoder:
x₁ → [RNN]           c → [RNN] → y₁
      ↓                     ↓
x₂ → [RNN]              [RNN] → y₂
      ↓                     ↓
x₃ → [RNN] → c →        [RNN] → y₃
```

**Process**:
1. Encoder processes input sequence → final hidden state = **context vector** (c)
2. Decoder uses context vector to generate output sequence

**Applications**:
- Machine translation
- Summarization
- Question answering

**Limitation**: Context vector is a bottleneck (must compress entire input)
- Solution: **Attention mechanism** (see Transformers notes)

### Stacked/Deep RNNs

**Idea**: Stack multiple RNN layers

```
Layer 2:  h₁⁽²⁾ → h₂⁽²⁾ → h₃⁽²⁾
           ↑       ↑       ↑
Layer 1:  h₁⁽¹⁾ → h₂⁽¹⁾ → h₃⁽¹⁾
           ↑       ↑       ↑
Input:    x₁      x₂      x₃
```

**Each layer**: Learns different level of abstraction
- Layer 1: Low-level patterns (e.g., character combinations)
- Layer 2: Mid-level patterns (e.g., words, phrases)
- Layer 3: High-level patterns (e.g., sentence structure)

**Typical depth**: 2-4 layers (diminishing returns beyond that)

---

## Applications

### Natural Language Processing

**1. Language Modeling**
- Task: Predict next word in sequence
- Architecture: Many-to-many (predict at each step)
- Example: "The cat sat on the ___" → "mat"

**2. Sentiment Analysis**
- Task: Classify sentiment of text
- Architecture: Many-to-one
- Example: "This movie is great!" → Positive

**3. Named Entity Recognition**
- Task: Tag each word (person, location, organization, etc.)
- Architecture: Many-to-many (same length), bidirectional
- Example: "Barack Obama visited Paris" → [PERSON] [PERSON] [O] [LOCATION]

**4. Machine Translation**
- Task: Translate sentence to another language
- Architecture: Encoder-decoder
- Example: "Hello world" → "Bonjour monde"

### Time Series

**1. Stock Price Prediction**
- Input: Historical prices
- Output: Future price

**2. Weather Forecasting**
- Input: Past weather data
- Output: Future conditions

**3. Anomaly Detection**
- Detect unusual patterns in sequences
- Example: Fraud detection, system monitoring

### Speech & Audio

**1. Speech Recognition**
- Input: Audio waveform (spectrogram)
- Output: Text transcription

**2. Music Generation**
- Input: Seed melody
- Output: Continuation

### Video Analysis

**1. Action Recognition**
- Input: Sequence of video frames
- Output: Action class

**2. Video Captioning**
- Input: Video
- Output: Textual description

---

## RNNs vs Transformers

### Why Transformers Replaced RNNs

**RNN Limitations**:
1. **Sequential processing**: Can't parallelize (must compute h_t before h_{t+1})
2. **Long sequences**: Still struggle with very long dependencies despite LSTM/GRU
3. **Slow training**: Sequential = slow, even with GPUs
4. **Information bottleneck**: Context vector in seq2seq compresses everything

**Transformer Advantages**:
1. **Parallel processing**: All positions computed simultaneously
2. **Attention**: Direct connections between any two positions
3. **Scalability**: Much faster training on modern hardware
4. **Long-range dependencies**: Attention can look at entire sequence

### When to Still Use RNNs

**RNNs still useful for**:
1. **Online/Streaming**: Process data as it arrives (can't wait for full sequence)
2. **Small datasets**: Fewer parameters than Transformers
3. **Short sequences**: Overhead of Transformers not worth it
4. **Memory constraints**: RNNs more memory-efficient for inference
5. **Temporal dynamics**: When you explicitly want sequential processing

**Rule of thumb**:
- New projects: Start with Transformers
- Existing RNN code: If it works, no need to change
- Specific requirements (streaming, etc.): RNNs might be better

---

## Best Practices

### Architecture Choices

1. **Start with GRU**: Fewer parameters, usually works as well as LSTM
2. **Try LSTM if**: Very long sequences, GRU doesn't work
3. **Bidirectional**: Use if entire sequence is available
4. **Depth**: 2-3 layers usually sufficient, more = diminishing returns

### Hyperparameters

**Hidden size**:
- Small datasets: 128-256
- Medium datasets: 256-512
- Large datasets: 512-1024

**Learning rate**:
- Start with 0.001 (Adam)
- Reduce if training unstable
- Use learning rate scheduling

**Dropout**:
- 0.2-0.5 between layers
- Don't apply between time steps (or use recurrent dropout carefully)

### Training Tips

1. **Gradient clipping**: Always use (clip_norm=1.0 to 5.0)
2. **Batch size**: Larger is better if memory allows (16-128)
3. **Sequence length**: Truncate very long sequences (diminishing returns)
4. **Pad sequences**: Batch sequences of different lengths with padding
5. **Early stopping**: Monitor validation loss, stop if no improvement

### Debugging

**1. Vanishing gradients**:
- Symptom: Loss doesn't decrease
- Solution: Use LSTM/GRU, gradient clipping, check learning rate

**2. Exploding gradients**:
- Symptom: Loss becomes NaN
- Solution: Gradient clipping, lower learning rate

**3. Overfitting**:
- Symptom: Train loss ↓, validation loss ↑
- Solution: More dropout, regularization, more data

**4. Slow convergence**:
- Check: Is sequence too long? Is model too deep? Is learning rate too small?

---

## Key Takeaways

1. **RNNs = Sequential Processing + Memory**
   - Hidden state h_t carries information across time
   - Same weights applied at each step (parameter sharing)

2. **Vanilla RNN: Simple but Limited**
   - Vanishing gradients prevent learning long-term dependencies
   - Good for short sequences only

3. **LSTM: The Solution**
   - Gates control information flow
   - Cell state preserves long-term dependencies
   - Much better for long sequences

4. **GRU: Simplified LSTM**
   - Fewer parameters, similar performance
   - Good default choice

5. **Training is Tricky**
   - Backpropagation through time
   - Gradient clipping essential
   - Careful hyperparameter tuning needed

6. **Transformers are Better (Usually)**
   - Parallel processing, attention mechanism
   - RNNs still useful for specific use cases (streaming, small data)

---

## Next Steps

**To truly understand RNNs**:
1. ✅ Read these notes
2. 🔨 **Implement vanilla RNN from scratch** (NumPy)
3. 🔨 **Implement LSTM from scratch** (understand the gates!)
4. 🔨 **Build character-level language model** (generate text)
5. 🔨 **Implement seq2seq** for simple translation task
6. 🔨 **Visualize hidden states** (see what network learns over time)

**Related topics**:
- Attention mechanisms (bridge to Transformers)
- Sequence-to-sequence models
- Teacher forcing vs autoregressive generation
- Beam search for decoding

**Then**: Move to Transformers notes to see how attention replaced recurrence!
