# Deep Learning Basics

A comprehensive introduction to neural networks and deep learning fundamentals.

## Table of Contents

1. [What is Deep Learning?](#what-is-deep-learning)
2. [Neural Network Architecture](#neural-network-architecture)
3. [Forward Propagation](#forward-propagation)
4. [Activation Functions](#activation-functions)
5. [Loss Functions](#loss-functions)
6. [Backward Propagation](#backward-propagation)
7. [Optimization](#optimization)
8. [Training Process](#training-process)
9. [Key Concepts](#key-concepts)
10. [Common Architectures](#common-architectures)

---

## What is Deep Learning?

**Deep Learning** is a subset of machine learning based on artificial neural networks with multiple layers (hence "deep"). These networks learn hierarchical representations of data by composing simple functions.

### Key Characteristics

1. **Hierarchical Feature Learning**: Automatically learns features at multiple levels of abstraction
2. **End-to-End Learning**: Can learn directly from raw data to final output
3. **Scalability**: Performance improves with more data and computation
4. **Flexibility**: Applicable to images, text, audio, time series, etc.

### Why "Deep"?

- **Traditional ML**: Hand-crafted features + simple classifier
- **Deep Learning**: Raw data → learned features (multiple layers) → output

**Example: Image Classification**
- **Layer 1**: Edges and simple patterns
- **Layer 2**: Textures and object parts
- **Layer 3**: Complete objects
- **Output**: Class prediction

---

## Neural Network Architecture

### The Perceptron (Single Neuron)

The fundamental building block:

```
Inputs: x₁, x₂, ..., xₙ
Weights: w₁, w₂, ..., wₙ
Bias: b

z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
a = σ(z)  (activation function)
```

**Components:**
1. **Inputs** ($x$): Features from data or previous layer
2. **Weights** ($w$): Learnable parameters (strength of connections)
3. **Bias** ($b$): Learnable offset (shifts activation)
4. **Linear combination** ($z$): Weighted sum of inputs
5. **Activation** ($a$): Non-linear transformation

### Multi-Layer Perceptron (MLP)

**Architecture:**
```
Input Layer → Hidden Layer(s) → Output Layer
```

**Layer Types:**

1. **Input Layer**
   - No computation, just holds input features
   - Size = number of features

2. **Hidden Layers**
   - Intermediate representations
   - Can have multiple layers (deep network)
   - Each layer transforms data

3. **Output Layer**
   - Final predictions
   - Size depends on task:
     - Binary classification: 1 neuron (sigmoid)
     - Multi-class: K neurons (softmax)
     - Regression: 1 or more neurons (linear)

### Notation

For a network with $L$ layers:

- $x$: Input vector
- $a^{[l]}$: Activations at layer $l$ (output of layer $l$)
- $z^{[l]}$: Pre-activation at layer $l$ (before activation function)
- $W^{[l]}$: Weight matrix for layer $l$ (shape: $n^{[l]} \times n^{[l-1]}$)
- $b^{[l]}$: Bias vector for layer $l$ (shape: $n^{[l]} \times 1$)
- $n^{[l]}$: Number of neurons in layer $l$
- $a^{[0]} = x$: Input
- $a^{[L]} = \hat{y}$: Output/prediction

---

## Forward Propagation

**Forward propagation** computes the output by passing inputs through the network layer by layer.

### Algorithm

For each layer $l = 1, 2, ..., L$:

1. **Linear step**: $z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$
2. **Activation step**: $a^{[l]} = g^{[l]}(z^{[l]})$

Where $g^{[l]}$ is the activation function for layer $l$.

### Example: 3-Layer Network

**Architecture**: Input (2) → Hidden (3) → Hidden (3) → Output (1)

**Layer 1:**
$$
z^{[1]} = W^{[1]} x + b^{[1]} \quad \text{(shape: 3×1)}
$$
$$
a^{[1]} = \text{ReLU}(z^{[1]})
$$

**Layer 2:**
$$
z^{[2]} = W^{[2]} a^{[1]} + b^{[2]} \quad \text{(shape: 3×1)}
$$
$$
a^{[2]} = \text{ReLU}(z^{[2]})
$$

**Layer 3 (Output):**
$$
z^{[3]} = W^{[3]} a^{[2]} + b^{[3]} \quad \text{(shape: 1×1)}
$$
$$
a^{[3]} = \sigma(z^{[3]}) = \hat{y}
$$

### Vectorized Implementation

For a batch of $m$ examples:

$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
$$
$$
A^{[l]} = g^{[l]}(Z^{[l]})
$$

Where:
- $A^{[l]}$ has shape $(n^{[l]}, m)$ (each column is one example)
- $Z^{[l]}$ has shape $(n^{[l]}, m)$
- $W^{[l]}$ has shape $(n^{[l]}, n^{[l-1]})$
- $b^{[l]}$ has shape $(n^{[l]}, 1)$ (broadcast across examples)

---

## Activation Functions

**Purpose**: Introduce non-linearity into the network.

**Why needed?** Without activation functions, multiple layers collapse to a single linear transformation:
$$
\text{If } a^{[l]} = z^{[l]} \text{ for all layers} \implies \text{entire network is linear}
$$

### Common Activation Functions

See [activation-functions.md](activation-functions.md) for detailed coverage. Quick overview:

1. **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
   - Output range: (0, 1)
   - Use: Binary classification output layer

2. **Tanh**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
   - Output range: (-1, 1)
   - Use: Hidden layers (zero-centered)

3. **ReLU**: $\text{ReLU}(z) = \max(0, z)$
   - Output range: [0, ∞)
   - Use: Default choice for hidden layers

4. **Leaky ReLU**: $\text{LeakyReLU}(z) = \max(0.01z, z)$
   - Fixes "dying ReLU" problem

5. **Softmax**: $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$
   - Output: Probability distribution
   - Use: Multi-class classification output

### Choosing Activation Functions

**Output Layer:**
- Binary classification → Sigmoid
- Multi-class classification → Softmax
- Regression → Linear (no activation)

**Hidden Layers:**
- Default → ReLU
- Alternative → Leaky ReLU, ELU
- Legacy → Tanh, Sigmoid (rarely used now)

---

## Loss Functions

**Purpose**: Quantify how wrong the predictions are.

For deep learning, loss functions are the same as in traditional ML (see [machine-learning/notes/loss-functions.md](../../machine-learning/notes/loss-functions.md)), but paired with appropriate output activations.

### Common Pairings

**Classification:**
- **Binary**: Sigmoid + Binary Cross-Entropy
  $$
  L(y, \hat{y}) = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
  $$

- **Multi-class**: Softmax + Categorical Cross-Entropy
  $$
  L(y, \hat{y}) = -\sum_{k} y_k \log(\hat{y}_k)
  $$

**Regression:**
- **Standard**: Linear + MSE
  $$
  L(y, \hat{y}) = (y - \hat{y})^2
  $$

### Why These Pairings?

**Sigmoid + Binary Cross-Entropy:**
- Gradient: $\frac{\partial L}{\partial z} = \hat{y} - y$ (clean!)
- No vanishing gradient at output

**Softmax + Categorical Cross-Entropy:**
- Gradient: $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$ (also clean!)
- Stable numerically when combined

---

## Backward Propagation

**Backward propagation** (backprop) computes gradients of the loss with respect to all parameters, enabling gradient descent optimization.

### The Chain Rule

Core idea: Chain rule of calculus applied recursively through layers.

For a function $f(g(x))$:
$$
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

### Backprop Algorithm

**Goal**: Compute $\frac{\partial L}{\partial W^{[l]}}$ and $\frac{\partial L}{\partial b^{[l]}}$ for all layers.

**Step 1: Output layer** ($l = L$)
$$
dZ^{[L]} = \frac{\partial L}{\partial Z^{[L]}} = A^{[L]} - Y
$$
(assuming softmax + cross-entropy or sigmoid + binary cross-entropy)

**Step 2: Gradients for layer** $l$

$$
dW^{[l]} = \frac{\partial L}{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T
$$

$$
db^{[l]} = \frac{\partial L}{\partial b^{[l]}} = \frac{1}{m} \sum_i dZ^{[l]}
$$

$$
dA^{[l-1]} = \frac{\partial L}{\partial A^{[l-1]}} = (W^{[l]})^T dZ^{[l]}
$$

**Step 3: Backward through activation**

$$
dZ^{[l-1]} = dA^{[l-1]} \odot g'^{[l-1]}(Z^{[l-1]})
$$

Where $\odot$ is element-wise multiplication and $g'$ is the derivative of the activation function.

### Example: Single Layer

**Forward:**
```
z = Wx + b
a = σ(z)
L = loss(a, y)
```

**Backward:**
```
dL/da = ∂L/∂a              # From loss function
dL/dz = dL/da * σ'(z)       # Through activation
dL/dW = dL/dz * x^T         # Weight gradient
dL/db = dL/dz               # Bias gradient
```

### Computational Graph

Backprop is essentially reverse-mode automatic differentiation on the computational graph:

```
Forward:  x → z → a → L
Backward: x ← z ← a ← L
```

Each operation stores its inputs during forward pass, then uses them to compute gradients during backward pass.

---

## Optimization

**Optimization** finds the parameter values that minimize the loss function.

### Gradient Descent

**Update rule:**
$$
W^{[l]} := W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}}
$$
$$
b^{[l]} := b^{[l]} - \alpha \frac{\partial L}{\partial b^{[l]}}
$$

Where $\alpha$ is the learning rate.

### Variants

**1. Batch Gradient Descent**
- Use entire dataset to compute gradient
- Accurate but slow for large datasets

**2. Stochastic Gradient Descent (SGD)**
- Use one example at a time
- Fast but noisy

**3. Mini-Batch Gradient Descent** (most common)
- Use small batches (32, 64, 128, 256)
- Balance between speed and stability

### Advanced Optimizers

**Momentum:**
$$
v := \beta v + (1-\beta) dW
$$
$$
W := W - \alpha v
$$

- Smooths gradient updates
- Accelerates in consistent directions

**Adam** (Adaptive Moment Estimation):
- Combines momentum and adaptive learning rates
- Most popular optimizer in practice
- Default choice for most problems

**RMSprop:**
- Adapts learning rate per parameter
- Good for RNNs

### Learning Rate

**Critical hyperparameter**: Too small → slow, too large → divergence

**Common strategies:**
- **Fixed**: 0.001, 0.01, 0.1
- **Decay**: Reduce over time
- **Cyclical**: Cycle between bounds
- **Adaptive**: Let optimizer decide (Adam)

---

## Training Process

### Full Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataset:
        # 1. Forward propagation
        predictions = model.forward(batch.x)

        # 2. Compute loss
        loss = loss_function(predictions, batch.y)

        # 3. Backward propagation
        gradients = model.backward(loss)

        # 4. Update parameters
        optimizer.step(gradients)

    # 5. Validation
    val_loss = evaluate(model, validation_data)

    # 6. Early stopping / checkpointing
    if val_loss < best_val_loss:
        save_checkpoint(model)
```

### Steps Explained

**1. Forward Propagation**
- Pass batch through network
- Compute predictions

**2. Compute Loss**
- Measure prediction quality
- Average over batch

**3. Backward Propagation**
- Compute gradients via backprop
- Chain rule through all layers

**4. Update Parameters**
- Apply optimizer (SGD, Adam, etc.)
- Move in direction of negative gradient

**5. Validation**
- Evaluate on held-out data
- Monitor generalization

**6. Checkpointing**
- Save best model
- Enable early stopping

---

## Key Concepts

### Epochs, Batches, Iterations

- **Epoch**: One complete pass through the entire training dataset
- **Batch**: Subset of training data processed together
- **Iteration**: One gradient update (one batch processed)

**Example:**
- Dataset: 10,000 samples
- Batch size: 100
- Iterations per epoch: 10,000 / 100 = 100
- Total iterations: 100 epochs × 100 iterations = 10,000

### Overfitting and Regularization

**Overfitting**: Model learns training data too well, fails on new data

**Signs:**
- Training loss ↓, validation loss ↑
- Large gap between train and val performance

**Solutions:**

**1. L2 Regularization (Weight Decay)**
$$
L_{\text{total}} = L_{\text{data}} + \frac{\lambda}{2} \sum_l \|W^{[l]}\|^2
$$

**2. Dropout**
- Randomly drop neurons during training
- Prevents co-adaptation
- Typical rate: 0.2-0.5

**3. Early Stopping**
- Stop when validation loss stops improving
- Simple and effective

**4. Data Augmentation**
- Artificially increase training data
- Images: rotation, flipping, cropping
- Text: back-translation, synonym replacement

**5. Batch Normalization**
- Normalize layer inputs
- Reduces internal covariate shift
- Acts as regularizer

### Initialization

**Why important?** Bad initialization → vanishing/exploding gradients

**Common strategies:**

**1. Xavier/Glorot Initialization** (for sigmoid/tanh)
$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
$$

**2. He Initialization** (for ReLU)
$$
W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
$$

**Rule of thumb:**
- ReLU → He initialization
- Tanh → Xavier initialization

### Vanishing and Exploding Gradients

**Vanishing Gradients:**
- Gradients become very small in early layers
- Network doesn't learn
- Caused by: sigmoid/tanh, deep networks, poor initialization

**Solutions:**
- Use ReLU activations
- Skip connections (ResNet)
- Batch normalization
- Better initialization

**Exploding Gradients:**
- Gradients become very large
- Unstable training

**Solutions:**
- Gradient clipping
- Lower learning rate
- Better initialization

### Batch Normalization

**Idea**: Normalize layer inputs to have zero mean and unit variance

**Formula** (for layer $l$):
$$
\hat{z}^{[l]} = \frac{z^{[l]} - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$
$$
\tilde{z}^{[l]} = \gamma \hat{z}^{[l]} + \beta
$$

Where $\gamma, \beta$ are learnable parameters.

**Benefits:**
- Faster training
- Higher learning rates possible
- Regularization effect
- Reduces internal covariate shift

**Placement**: After linear layer, before activation
```
z = Wx + b → BatchNorm → activation
```

---

## Common Architectures

### Feedforward Networks (MLPs)

**Structure**: Fully connected layers

**Use cases:**
- Tabular data
- Simple classification/regression
- Embedding layers

**Example:**
```
Input (784) → FC(512) → ReLU → FC(256) → ReLU → FC(10) → Softmax
```

### Convolutional Neural Networks (CNNs)

**Key operation**: Convolution (local connectivity + weight sharing)

**Components:**
- **Convolutional layers**: Feature extraction
- **Pooling layers**: Spatial downsampling
- **Fully connected layers**: Classification

**Use cases:**
- Image classification
- Object detection
- Segmentation

**Example (LeNet-like):**
```
Input → Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Softmax
```

**Famous architectures:**
- **LeNet-5** (1998): Early CNN for digit recognition
- **AlexNet** (2012): ImageNet breakthrough
- **VGGNet** (2014): Very deep (16-19 layers)
- **ResNet** (2015): Skip connections, 152 layers
- **Inception** (2014): Multi-scale features

### Recurrent Neural Networks (RNNs)

**Key idea**: Process sequences by maintaining hidden state

**Variants:**
- **Vanilla RNN**: Simple recurrence
- **LSTM**: Long Short-Term Memory (gates)
- **GRU**: Gated Recurrent Unit (simplified LSTM)

**Use cases:**
- Time series prediction
- Natural language processing
- Speech recognition

**Structure:**
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
y_t = W_hy * h_t
```

### Transformers

**Key mechanism**: Self-attention (all-to-all connections)

**Components:**
- **Multi-head attention**: Learn different relationships
- **Feedforward layers**: Per-position transformations
- **Positional encoding**: Inject sequence order

**Use cases:**
- Language models (GPT, BERT)
- Machine translation
- Vision (ViT)

**Why popular?**
- Parallelizable (unlike RNNs)
- Long-range dependencies
- Transfer learning (pre-training)

---

## Summary

### Neural Network Fundamentals

1. **Architecture**: Input → Hidden Layer(s) → Output
2. **Forward Pass**: Compute predictions layer by layer
3. **Loss**: Measure prediction quality
4. **Backward Pass**: Compute gradients via chain rule
5. **Optimization**: Update parameters to minimize loss

### Key Success Factors

**Architecture:**
- ✓ Appropriate depth and width
- ✓ Right activation functions (ReLU for hidden, task-specific for output)
- ✓ Proper initialization (He for ReLU, Xavier for tanh)

**Training:**
- ✓ Good optimizer (Adam is safe default)
- ✓ Appropriate learning rate
- ✓ Regularization (dropout, L2, early stopping)
- ✓ Batch normalization for deep networks

**Data:**
- ✓ Sufficient training data
- ✓ Proper preprocessing/normalization
- ✓ Data augmentation if needed

### Common Pitfalls

**Training issues:**
- ✗ Learning rate too high → divergence
- ✗ Learning rate too low → slow convergence
- ✗ Vanishing gradients → use ReLU, batch norm
- ✗ Exploding gradients → gradient clipping

**Overfitting:**
- ✗ Training too long → early stopping
- ✗ Too complex model → regularization
- ✗ Insufficient data → data augmentation

### Next Steps

- **Activation Functions**: See [activation-functions.md](activation-functions.md)
- **Optimization**: Dive deeper into Adam, learning rate schedules
- **Regularization**: Advanced techniques (dropout variants, label smoothing)
- **Architectures**: CNNs, RNNs, Transformers in detail

---

## Quick Reference

### Forward Propagation
```python
# Layer l
Z[l] = W[l] @ A[l-1] + b[l]
A[l] = activation(Z[l])
```

### Backward Propagation
```python
# Output layer
dZ[L] = A[L] - Y

# Hidden layer l
dW[l] = (1/m) * dZ[l] @ A[l-1].T
db[l] = (1/m) * sum(dZ[l])
dA[l-1] = W[l].T @ dZ[l]
dZ[l-1] = dA[l-1] * activation_derivative(Z[l-1])
```

### Parameter Update
```python
# Gradient descent
W[l] = W[l] - learning_rate * dW[l]
b[l] = b[l] - learning_rate * db[l]
```

### Dimensions
- **Weights**: $(n^{[l]}, n^{[l-1]})$
- **Biases**: $(n^{[l]}, 1)$
- **Activations** (batch): $(n^{[l]}, m)$
- **Gradients**: Same shape as corresponding parameter
