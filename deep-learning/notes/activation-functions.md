# Activation Functions - Comprehensive Guide

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. This guide covers all major activation functions, their properties, use cases, and derivatives.

## Table of Contents

1. [Why Activation Functions?](#why-activation-functions)
2. [Properties of Good Activations](#properties-of-good-activations)
3. [Classical Activation Functions](#classical-activation-functions)
4. [Modern Activation Functions](#modern-activation-functions)
5. [Output Layer Activations](#output-layer-activations)
6. [Choosing Activation Functions](#choosing-activation-functions)
7. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Why Activation Functions?

### The Linearity Problem

Without activation functions, neural networks collapse to linear models:

$$
\begin{align}
h_1 &= W_1 x \\
h_2 &= W_2 h_1 = W_2 (W_1 x) \\
\hat{y} &= W_3 h_2 = W_3 (W_2 (W_1 x)) = (W_3 W_2 W_1) x = W_{\text{effective}} x
\end{align}
$$

**Result**: Multiple layers reduce to a single linear transformation, losing representational power.

### Non-Linearity Enables Complexity

Activation functions $g(z)$ break linearity:

$$
h = g(Wx + b)
$$

**Effect**: Network can learn:
- XOR and other non-linearly separable functions
- Complex decision boundaries
- Hierarchical feature representations

### Where to Apply

**Typical placement:**
```
z^[l] = W^[l] a^[l-1] + b^[l]    # Linear transformation
a^[l] = g(z^[l])                  # Activation function
```

**Per-layer choice:**
- Hidden layers: ReLU, Leaky ReLU, ELU, etc.
- Output layer: Sigmoid, softmax, or linear (task-dependent)

---

## Properties of Good Activations

### 1. Non-Linearity
- **Required** for deep networks to work
- Enables learning complex functions

### 2. Differentiability
- **Required** for gradient-based optimization
- Smooth derivatives → stable training
- Exception: ReLU (non-differentiable at 0, but sub-gradient works)

### 3. Monotonicity
- **Desirable** but not required
- Simplifies optimization landscape
- Examples: Sigmoid, tanh, ReLU are monotonic

### 4. Range
- **Bounded** (sigmoid, tanh): Output stays in fixed range
  - Pro: Stable activations
  - Con: Vanishing gradients

- **Unbounded** (ReLU): Output can be arbitrarily large
  - Pro: No vanishing gradient (for positive inputs)
  - Con: Potential for exploding activations

### 5. Zero-Centered
- **Desirable**: Outputs centered around zero
- **Why**: Reduces bias in gradients
- Examples: Tanh (✓), ReLU (✗), Sigmoid (✗)

### 6. Computational Efficiency
- **Important** for large-scale training
- ReLU: Just max(0, z) (very fast)
- Sigmoid: Requires exp() (slower)

---

## Classical Activation Functions

### 1. Sigmoid

**Formula:**
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Derivative:**
$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

**Properties:**
- **Range**: (0, 1)
- **Zero-centered**: ✗ No (outputs always positive)
- **Monotonic**: ✓ Yes (always increasing)
- **Computational**: Moderate (exp required)

**Advantages:**
- Smooth, continuous everywhere
- Output interpretable as probability
- Historically important

**Disadvantages:**
- **Vanishing gradients**: Gradient ~0 when |z| is large
- **Not zero-centered**: Slows learning
- **Expensive**: exp() computation

**When to use:**
- Binary classification (output layer only)
- Gate mechanisms (LSTM)
- Rarely used in hidden layers anymore

**Gradient saturation:**
- When $z \ll 0$: $\sigma(z) \approx 0$, $\sigma'(z) \approx 0$
- When $z \gg 0$: $\sigma(z) \approx 1$, $\sigma'(z) \approx 0$
- Only strong gradient near $z = 0$

---

### 2. Hyperbolic Tangent (Tanh)

**Formula:**
$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{2}{1 + e^{-2z}} - 1 = 2\sigma(2z) - 1
$$

**Derivative:**
$$
\tanh'(z) = 1 - \tanh^2(z)
$$

**Properties:**
- **Range**: (-1, 1)
- **Zero-centered**: ✓ Yes
- **Monotonic**: ✓ Yes
- **Computational**: Moderate (exp required)

**Advantages:**
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid
- Smooth

**Disadvantages:**
- **Vanishing gradients**: Still present for large |z|
- **Expensive**: exp() computation

**When to use:**
- Hidden layers (historically)
- RNNs (still common)
- Rarely used in modern feedforward networks (ReLU preferred)

**Comparison to Sigmoid:**
- Same shape, just shifted and scaled
- Better: Zero-centered, stronger gradients
- Still suffers from vanishing gradients

---

### 3. Rectified Linear Unit (ReLU)

**Formula:**
$$
\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}
$$

**Derivative:**
$$
\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \\ \text{undefined} & \text{if } z = 0 \end{cases}
$$

In practice, use 0 at z=0 for sub-gradient.

**Properties:**
- **Range**: [0, ∞)
- **Zero-centered**: ✗ No
- **Monotonic**: ✓ Yes
- **Computational**: Very fast (just max)

**Advantages:**
- **No vanishing gradient** for positive inputs
- **Very fast** to compute
- **Sparsity**: Many neurons output 0 (efficient representations)
- **Works well** in practice

**Disadvantages:**
- **Dying ReLU**: Neurons can get stuck at 0
  - If $z \leq 0$ always, gradient is always 0
  - Neuron stops learning
  - Caused by: large learning rates, bad initialization

- **Not zero-centered**: All positive outputs
- **Unbounded**: Can lead to large activations

**When to use:**
- **Default choice** for hidden layers in feedforward networks
- CNNs
- Most modern architectures

**Dying ReLU Problem:**
- Neuron outputs 0 for all inputs
- Gradient always 0 → weights never update
- Solutions: Leaky ReLU, proper initialization, lower learning rate

---

## Modern Activation Functions

### 4. Leaky ReLU

**Formula:**
$$
\text{LeakyReLU}(z) = \max(\alpha z, z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}
$$

Where $\alpha$ is a small constant (typically 0.01).

**Derivative:**
$$
\text{LeakyReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z < 0 \end{cases}
$$

**Properties:**
- **Range**: (-∞, ∞)
- **Zero-centered**: ✗ No
- **Monotonic**: ✓ Yes
- **Computational**: Very fast

**Advantages:**
- **Fixes dying ReLU**: Always has non-zero gradient
- Still very fast
- Usually performs as well or better than ReLU

**Disadvantages:**
- Additional hyperparameter ($\alpha$)
- Still not zero-centered

**Variants:**

**Parametric ReLU (PReLU):**
- $\alpha$ is learned during training
- Formula: Same as Leaky ReLU, but $\alpha$ is a parameter

**Randomized Leaky ReLU:**
- $\alpha$ sampled randomly during training
- Acts as regularizer

**When to use:**
- Alternative to ReLU when dying neurons are a problem
- Good default choice for hidden layers

---

### 5. Exponential Linear Unit (ELU)

**Formula:**
$$
\text{ELU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}
$$

Where $\alpha > 0$ (typically 1.0).

**Derivative:**
$$
\text{ELU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha e^z = \text{ELU}(z) + \alpha & \text{if } z < 0 \end{cases}
$$

**Properties:**
- **Range**: (-α, ∞)
- **Zero-centered**: ✓ Approximately (mean activations closer to 0)
- **Monotonic**: ✓ Yes
- **Computational**: Slower (exp required)

**Advantages:**
- **Zero-centered**: Better than ReLU/Leaky ReLU
- **No dying neurons**: Always has gradient
- **Smooth**: Differentiable everywhere
- Often **faster convergence** than ReLU

**Disadvantages:**
- **Slower** to compute (exp)
- Still has hyperparameter ($\alpha$)

**When to use:**
- Hidden layers when faster convergence is needed
- Alternative to ReLU with better properties
- Good for deeper networks

---

### 6. Scaled Exponential Linear Unit (SELU)

**Formula:**
$$
\text{SELU}(z) = \lambda \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}
$$

Where:
- $\lambda \approx 1.0507$
- $\alpha \approx 1.67326$

These constants chosen to ensure self-normalizing property.

**Derivative:**
$$
\text{SELU}'(z) = \lambda \begin{cases} 1 & \text{if } z > 0 \\ \alpha e^z & \text{if } z < 0 \end{cases}
$$

**Properties:**
- **Range**: (-λα, ∞)
- **Zero-centered**: ✓ Yes (self-normalizing)
- **Monotonic**: ✓ Yes
- **Self-normalizing**: Maintains mean ~0, variance ~1

**Advantages:**
- **Self-normalizing**: Internal normalization during forward pass
- Can replace batch normalization
- Enables very deep networks (>30 layers) without batch norm

**Disadvantages:**
- **Requires specific conditions**:
  - LeCun normal initialization
  - AlphaDropout (not standard dropout)
  - Fully connected layers (not conv layers)

- More restrictive than other activations

**When to use:**
- Deep fully connected networks
- When batch normalization is problematic
- Specific architectures designed for SELU

---

### 7. Swish (SiLU)

**Formula:**
$$
\text{Swish}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}
$$

Also called **Sigmoid-weighted Linear Unit (SiLU)**.

**Derivative:**
$$
\text{Swish}'(z) = \sigma(z) + z \cdot \sigma(z)(1 - \sigma(z)) = \sigma(z)(1 + z(1 - \sigma(z)))
$$

**Properties:**
- **Range**: (-∞, ∞)
- **Zero-centered**: ✗ No (but smooth around 0)
- **Monotonic**: ✗ No (has small bump below 0)
- **Smooth**: ✓ Yes (infinitely differentiable)

**Advantages:**
- **Smooth**: Better gradient flow than ReLU
- **Non-monotonic**: More expressive
- Often **outperforms ReLU** in deep networks
- Simple formula

**Disadvantages:**
- **Slower** than ReLU (sigmoid computation)
- Not zero-centered

**When to use:**
- Deep networks (especially very deep)
- When ReLU performance plateaus
- Modern architectures (EfficientNet, Transformer variants)

---

### 8. Gaussian Error Linear Unit (GELU)

**Formula (exact):**
$$
\text{GELU}(z) = z \cdot \Phi(z) = z \cdot P(Z \leq z), \quad Z \sim \mathcal{N}(0, 1)
$$

Where $\Phi$ is the CDF of standard normal distribution.

**Approximation** (faster):
$$
\text{GELU}(z) \approx 0.5z\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(z + 0.044715z^3)\right]\right)
$$

Or:
$$
\text{GELU}(z) \approx z \cdot \sigma(1.702z)
$$

**Derivative** (approximate):
$$
\text{GELU}'(z) \approx \sigma(1.702z)(1 + 1.702z(1 - \sigma(1.702z)))
$$

**Properties:**
- **Range**: (-∞, ∞)
- **Zero-centered**: ✗ No
- **Monotonic**: ✗ No (similar to Swish)
- **Smooth**: ✓ Yes

**Advantages:**
- **State-of-the-art** in Transformers (BERT, GPT)
- Smooth, probabilistic interpretation
- Often **best performance** on NLP tasks

**Disadvantages:**
- **Expensive**: Requires erf or approximation
- More complex than simpler activations

**When to use:**
- Transformer models
- NLP tasks
- When computational cost is acceptable

**Why popular in Transformers?**
- Smooth non-linearity helps attention mechanisms
- Probabilistic interpretation aligns with dropout (stochastic regularization)
- Empirically works very well

---

## Output Layer Activations

### 9. Sigmoid (Binary Classification)

See [Classical Activation Functions](#1-sigmoid).

**Use case:**
- Binary classification output
- Probability of positive class

**Paired with:**
- Binary cross-entropy loss

**Formula:**
$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

### 10. Softmax (Multi-Class Classification)

**Formula:**
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Properties:**
- **Range**: (0, 1) for each output
- **Sum**: $\sum_i \text{softmax}(z_i) = 1$ (probability distribution)
- **Monotonic**: Preserves relative ordering

**Derivative:**

For output $i$ with respect to input $j$:
$$
\frac{\partial \text{softmax}(z_i)}{\partial z_j} = \begin{cases}
\text{softmax}(z_i)(1 - \text{softmax}(z_i)) & \text{if } i = j \\
-\text{softmax}(z_i) \cdot \text{softmax}(z_j) & \text{if } i \neq j
\end{cases}
$$

**Paired with:**
- Categorical cross-entropy loss
- Gradient (with cross-entropy): $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$

**Numerical stability:**

Naive implementation can overflow/underflow. Use:
$$
\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}
$$

Subtracting max(z) prevents overflow without changing result.

**When to use:**
- Multi-class classification (output layer only)
- Mutually exclusive classes
- Need probability interpretation

**Variants:**

**Temperature-scaled softmax:**
$$
\text{softmax}_T(z_i) = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

- $T < 1$: Sharper distribution (more confident)
- $T > 1$: Smoother distribution (less confident)
- Used in distillation, calibration

---

### 11. Linear (Regression)

**Formula:**
$$
\text{Linear}(z) = z
$$

**Derivative:**
$$
\text{Linear}'(z) = 1
$$

**When to use:**
- Regression output layer
- Predicting continuous values
- No constraints on output range

**Paired with:**
- MSE, MAE, Huber, or other regression losses

---

## Choosing Activation Functions

### Decision Tree

```
Where is the activation?

├─ Hidden Layers
│  │
│  ├─ Default choice → ReLU
│  │
│  ├─ Dying ReLU problem? → Leaky ReLU or ELU
│  │
│  ├─ Want faster convergence? → ELU or SELU
│  │
│  ├─ Very deep network?
│  │  ├─ Feedforward → SELU (with proper init)
│  │  └─ General → Swish or GELU
│  │
│  └─ Transformer/NLP? → GELU
│
└─ Output Layer
   │
   ├─ Binary classification → Sigmoid
   │
   ├─ Multi-class classification → Softmax
   │
   └─ Regression → Linear
```

### Summary Table

| Activation | Hidden Layer | Output Layer | Pros | Cons |
|------------|--------------|--------------|------|------|
| **ReLU** | ✓✓ Default | ✗ No | Fast, no vanishing gradient | Dying ReLU, not zero-centered |
| **Leaky ReLU** | ✓ Good | ✗ No | Fixes dying ReLU, fast | Not zero-centered |
| **ELU** | ✓ Good | ✗ No | Zero-centered, smooth | Slower (exp) |
| **SELU** | ✓ Specialized | ✗ No | Self-normalizing | Requires specific setup |
| **Swish** | ✓ Advanced | ✗ No | Smooth, often beats ReLU | Slower than ReLU |
| **GELU** | ✓ Advanced | ✗ No | SOTA for Transformers | Expensive |
| **Sigmoid** | ✗ Rarely | ✓ Binary | Probabilistic output | Vanishing gradients |
| **Tanh** | △ Legacy | ✗ No | Zero-centered | Vanishing gradients |
| **Softmax** | ✗ No | ✓ Multi-class | Probability distribution | Output layer only |
| **Linear** | ✗ No | ✓ Regression | Simple | No non-linearity |

### Practical Recommendations

**For most problems:**
1. Start with **ReLU** for hidden layers
2. Use **Sigmoid** (binary) or **Softmax** (multi-class) for output
3. If ReLU doesn't work well, try **Leaky ReLU** or **ELU**

**For specific cases:**
- **Transformers/NLP**: GELU
- **Very deep networks**: Swish or SELU
- **RNNs**: Tanh (still common)
- **GANs**: Leaky ReLU or ELU

**Don't use:**
- Sigmoid/Tanh in hidden layers (unless RNN)
- ReLU in output layer

---

## Common Issues and Solutions

### 1. Vanishing Gradients

**Problem**: Gradients become very small in early layers, preventing learning.

**Causes:**
- Sigmoid/tanh activations (saturate)
- Deep networks
- Poor initialization

**Solutions:**
- ✓ Use ReLU family activations
- ✓ Batch normalization
- ✓ Skip connections (ResNet)
- ✓ Proper initialization (He for ReLU, Xavier for tanh)

---

### 2. Exploding Gradients

**Problem**: Gradients become very large, causing instability.

**Causes:**
- Deep networks
- Large weight initialization
- Unbounded activations (ReLU)

**Solutions:**
- ✓ Gradient clipping
- ✓ Batch normalization
- ✓ Lower learning rate
- ✓ Proper initialization

---

### 3. Dying ReLU

**Problem**: ReLU neurons output 0 for all inputs, stop learning.

**Causes:**
- Large learning rates
- Bad initialization
- Large negative bias

**Solutions:**
- ✓ Use Leaky ReLU or ELU
- ✓ Lower learning rate
- ✓ Better initialization (He initialization)
- ✓ Batch normalization

**Diagnosis:**
```python
# Check fraction of dead neurons
dead_neurons = (activations == 0).mean()
if dead_neurons > 0.5:
    print("Many dead ReLU neurons!")
```

---

### 4. Saturation

**Problem**: Activation function output is in saturated region (flat gradient).

**Occurs with:**
- Sigmoid (|z| > 5)
- Tanh (|z| > 2)

**Solutions:**
- ✓ Use ReLU family (don't saturate for z > 0)
- ✓ Batch normalization (keeps activations in good range)
- ✓ Proper initialization

---

### 5. Internal Covariate Shift

**Problem**: Distribution of layer inputs changes during training.

**Causes:**
- Parameter updates in previous layers
- Makes learning harder

**Solutions:**
- ✓ Batch normalization
- ✓ Layer normalization
- ✓ SELU (self-normalizing)

---

## Derivatives Reference

Quick reference for backpropagation:

| Activation | $g(z)$ | $g'(z)$ |
|------------|--------|---------|
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $g(z)(1 - g(z))$ |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - g(z)^2$ |
| **ReLU** | $\max(0, z)$ | $\begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$ |
| **Leaky ReLU** | $\max(\alpha z, z)$ | $\begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases}$ |
| **ELU** | $\begin{cases} z & z > 0 \\ \alpha(e^z-1) & z \leq 0 \end{cases}$ | $\begin{cases} 1 & z > 0 \\ g(z) + \alpha & z \leq 0 \end{cases}$ |
| **Swish** | $z \cdot \sigma(z)$ | $\sigma(z)(1 + z(1-\sigma(z)))$ |
| **Linear** | $z$ | $1$ |

---

## Summary

### Key Takeaways

1. **Always use non-linear activations** in hidden layers (except final layer for regression)

2. **ReLU is the default** for hidden layers:
   - Fast, effective, no vanishing gradient for z > 0
   - If issues arise, try Leaky ReLU or ELU

3. **Output layer depends on task**:
   - Binary classification → Sigmoid
   - Multi-class → Softmax
   - Regression → Linear

4. **Modern alternatives** (Swish, GELU) often work better but are slower

5. **Avoid** sigmoid/tanh in hidden layers (vanishing gradients)

### Quick Decision Guide

**Starting a new project?**
- Hidden: ReLU
- Output: Task-specific (sigmoid/softmax/linear)

**ReLU not working?**
- Try: Leaky ReLU → ELU → Swish

**Working on Transformers?**
- Use: GELU

**Very deep network?**
- Try: SELU (with proper setup) or Swish

**The golden rule**: Start simple (ReLU), only add complexity if needed.
