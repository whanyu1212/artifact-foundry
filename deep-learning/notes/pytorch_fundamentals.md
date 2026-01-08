# PyTorch Fundamentals

**Understanding PyTorch Through the Lens of From-Scratch Implementations**

Now that you've implemented neural networks, CNNs, and understand backpropagation intimately, PyTorch will make perfect sense. These notes connect PyTorch to what you've already built, showing you **why** PyTorch does things the way it does.

---

## Table of Contents

1. [What PyTorch Automates](#what-pytorch-automates)
2. [Core Concepts](#core-concepts)
3. [Tensors vs NumPy Arrays](#tensors-vs-numpy-arrays)
4. [Autograd: Automatic Differentiation](#autograd-automatic-differentiation)
5. [Building Models with nn.Module](#building-models-with-nnmodule)
6. [Your NumPy Code → PyTorch](#your-numpy-code--pytorch)
7. [Training Loop Pattern](#training-loop-pattern)
8. [Common Layers](#common-layers)
9. [Loss Functions & Optimizers](#loss-functions--optimizers)
10. [Debugging & Best Practices](#debugging--best-practices)
11. [When to Use Custom Layers](#when-to-use-custom-layers)

---

## What PyTorch Automates

### What You Implemented Manually

From your NumPy implementations:

**1. Backpropagation** (`neural_network.py`, `convolution.py`):
```python
# You wrote:
def backward(self, da, learning_rate):
    dz = da * self.activation_derivative(self.z)
    self.dW = (1/m) * np.dot(dz, self.a_prev.T)
    self.db = (1/m) * np.sum(dz, axis=1, keepdims=True)
    da_prev = np.dot(self.W.T, dz)
    # Update weights
    self.W -= learning_rate * self.dW
    self.b -= learning_rate * self.db
    return da_prev
```

**PyTorch does ALL of this automatically**:
```python
loss.backward()  # ← Computes ALL gradients through entire network!
optimizer.step()  # ← Updates ALL parameters!
```

**2. Shape Management**:
```python
# You carefully tracked shapes:
# Input: (batch, channels, height, width)
# After conv: (batch, out_channels, out_h, out_w)
# After flatten: (batch, channels * h * w)
```

**PyTorch handles this automatically** - layers figure out shapes.

**3. Gradient Accumulation**:
```python
# You manually accumulated gradients in conv backward:
dw = dout_reshaped @ x_col.T
```

**PyTorch tracks everything** in computational graph.

### What PyTorch Is

**PyTorch = NumPy + Autograd + GPU + Ecosystem**

1. **NumPy++**: Tensors work like NumPy arrays (same indexing, broadcasting)
2. **Autograd**: Automatic differentiation (tracks operations, computes gradients)
3. **GPU Support**: Run on CUDA with `.cuda()` or `.to('cuda')`
4. **Pre-built Layers**: Conv2d, Linear, LSTM - you know what they do inside!
5. **Optimizers**: Adam, SGD, RMSprop - gradient descent algorithms
6. **Utilities**: Data loading, saving, distributed training

**Key Insight**: PyTorch is a framework for what you've been doing manually.

---

## Core Concepts

### 1. Tensors: The Foundation

**Tensor** = Multi-dimensional array (like NumPy) + gradient tracking

```python
import torch

# Create tensors (similar to NumPy)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.randn(3, 4)  # Random 3x4 tensor
z = torch.zeros(2, 2)  # 2x2 zeros

# Operations (just like NumPy!)
a = x + 1
b = y @ z  # Matrix multiplication
c = torch.sin(x)
```

**Key difference from NumPy**: Can track gradients!

```python
# Enable gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Compute something
y = x ** 2
loss = y.sum()

# Get gradients automatically!
loss.backward()
print(x.grad)  # ← Gradients computed automatically!
# Output: tensor([2., 4., 6.])  because d(x²)/dx = 2x
```

### 2. Computational Graph

**What PyTorch builds behind the scenes**:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2      # y = 4
z = y + 1       # z = 5
w = z * 3       # w = 15

w.backward()  # Compute dw/dx
```

**PyTorch builds this graph**:
```
x (2.0) → [**2] → y (4.0) → [+1] → z (5.0) → [*3] → w (15.0)
  ↑                                                      ↓
  └────────────── gradient flows backward ───────────────┘
```

**Backward pass** (chain rule):
```
dw/dx = dw/dz * dz/dy * dy/dx
      = 3 * 1 * 2x
      = 3 * 1 * 2(2)
      = 12
```

PyTorch computes this automatically using the graph!

### 3. Three Magic Methods

**Every PyTorch workflow uses these three**:

```python
# 1. Forward pass - compute predictions
output = model(input)

# 2. Backward pass - compute gradients
loss.backward()

# 3. Update parameters
optimizer.step()
```

That's it! PyTorch handles the complexity.

---

## Tensors vs NumPy Arrays

### Similarities

```python
import numpy as np
import torch

# Almost identical syntax!
np_array = np.array([1, 2, 3])
tensor = torch.tensor([1, 2, 3])

# Indexing
np_array[0]  # 1
tensor[0]    # tensor(1)

# Slicing
np_array[1:]   # array([2, 3])
tensor[1:]     # tensor([2, 3])

# Broadcasting
np_array + 10   # array([11, 12, 13])
tensor + 10     # tensor([11, 12, 13])

# Reshaping
np_array.reshape(3, 1)
tensor.reshape(3, 1)  # or tensor.view(3, 1)
```

### Key Differences

| Feature | NumPy | PyTorch |
|---------|-------|---------|
| **Gradient tracking** | ❌ No | ✅ `requires_grad=True` |
| **GPU support** | ❌ CPU only | ✅ `.cuda()` or `.to('cuda')` |
| **Automatic differentiation** | ❌ Manual | ✅ `.backward()` |
| **Deep learning ops** | ❌ DIY | ✅ Built-in (conv, pool, etc.) |
| **Dynamic graphs** | N/A | ✅ Graph built on-the-fly |

### Converting Between NumPy and PyTorch

```python
# NumPy → PyTorch
np_array = np.array([1, 2, 3])
tensor = torch.from_numpy(np_array)

# PyTorch → NumPy
tensor = torch.tensor([1, 2, 3])
np_array = tensor.numpy()

# Note: They share memory (changes in one affect the other!)
# To avoid this, use .clone() or .copy()
tensor = torch.from_numpy(np_array).clone()
```

### Device Management (CPU vs GPU)

```python
# Check GPU availability
print(torch.cuda.is_available())

# Create tensor on GPU
x = torch.randn(3, 3).cuda()  # Old way
x = torch.randn(3, 3).to('cuda')  # New way (better)

# Move between devices
x_cpu = x.to('cpu')
x_gpu = x.to('cuda')

# Best practice: device-agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 3).to(device)
```

---

## Autograd: Automatic Differentiation

### How Autograd Works

**You implemented this manually**:
```python
# Your manual backprop (from neural_network.py)
dz = da * sigmoid_derivative(z)
dW = (1/m) * dz @ a_prev.T
```

**PyTorch does this automatically**:
```python
# Forward pass (PyTorch builds graph)
z = W @ x + b
a = torch.sigmoid(z)
loss = (a - y) ** 2

# Backward pass (PyTorch computes ALL derivatives)
loss.backward()

# Gradients now in W.grad, b.grad
```

### The `.backward()` Method

**What it does**:
1. Traverses computational graph in reverse
2. Applies chain rule at each node
3. Accumulates gradients in `.grad` attribute

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3  # y = 8
y.backward()

print(x.grad)  # tensor([12.])  because dy/dx = 3x² = 3(2²) = 12
```

### Gradient Accumulation

**Important**: Gradients **accumulate** by default!

```python
x = torch.tensor([2.0], requires_grad=True)

# First backward
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.])

# Second backward (without zeroing)
y = x ** 2
y.backward()
print(x.grad)  # tensor([8.])  ← Accumulated!

# Solution: Zero gradients before each backward
x.grad.zero_()
y = x ** 2
y.backward()
print(x.grad)  # tensor([4.])  ← Correct
```

**In training loops**: Use `optimizer.zero_grad()`

### Detaching from Graph

```python
# Sometimes you don't want gradient tracking
x = torch.randn(3, 3, requires_grad=True)

# Stop tracking (doesn't affect x)
y = x.detach()  # y has no gradient tracking

# Temporarily disable gradient tracking (saves memory)
with torch.no_grad():
    y = x ** 2  # No graph built, faster
```

**Use `torch.no_grad()` for**:
- Inference (no need for gradients)
- Validation loop
- Computing metrics

---

## Building Models with nn.Module

### The nn.Module Pattern

**All PyTorch models inherit from `nn.Module`**:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()  # ← Always call this!
        # Define layers here

    def forward(self, x):
        # Define forward pass here
        return output
```

**Why `nn.Module`?**
- Automatically tracks parameters
- Provides `.parameters()` method (for optimizer)
- Handles device movement (`.to(device)`)
- Enables saving/loading (`.state_dict()`)

### Simple Example: Your MLP in PyTorch

**Your NumPy version**:
```python
# From your neural_network.py
nn = NeuralNetwork()
nn.add_layer(Dense(784, 128, activation='relu'))
nn.add_layer(Dense(128, 10, activation='softmax'))
```

**PyTorch equivalent**:
```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)  # ← Your Dense layer!
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))     # ← Your relu activation!
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = MLP()
```

**Even simpler with `nn.Sequential`**:
```python
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)
```

### Accessing Parameters

```python
model = MLP()

# All parameters (for optimizer)
for param in model.parameters():
    print(param.shape)

# Named parameters (for debugging)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Example output:
# fc1.weight: torch.Size([128, 784])
# fc1.bias: torch.Size([128])
# fc2.weight: torch.Size([10, 128])
# fc2.bias: torch.Size([10])
```

### Your CNN in PyTorch

**Your NumPy version**:
```python
# From your cnn_network.py
cnn = CNN(input_shape=(28, 28, 1))
cnn.add_conv_layer(32, filter_size=3, padding=1)
cnn.add_pool_layer(pool_size=2)
cnn.add_conv_layer(64, filter_size=3, padding=1)
cnn.add_pool_layer(pool_size=2)
cnn.add_flatten()
cnn.add_dense_layer(128, activation='relu')
cnn.add_dense_layer(10, activation='softmax')
```

**PyTorch equivalent**:
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # ← After two 2x2 pools: 28→14→7
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Conv→ReLU→Pool
        x = self.pool(torch.relu(self.conv2(x)))  # Conv→ReLU→Pool
        x = x.view(x.size(0), -1)  # Flatten (your Flatten layer!)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax (use with CrossEntropyLoss)
        return x
```

---

## Your NumPy Code → PyTorch

### Dense Layer Comparison

**Your NumPy implementation**:
```python
class Dense:
    def __init__(self, input_size, output_size, activation='relu'):
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.b = np.zeros((output_size, 1))

    def forward(self, a_prev):
        self.z = np.dot(self.W, a_prev) + self.b
        self.a = self.activation(self.z)
        return self.a

    def backward(self, da, learning_rate):
        dz = da * self.activation_derivative(self.z)
        self.dW = (1/m) * np.dot(dz, self.a_prev.T)
        self.db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        return np.dot(self.W.T, dz)
```

**PyTorch equivalent** (you don't write this, it's built-in!):
```python
# Just use nn.Linear!
layer = nn.Linear(input_size, output_size)

# Forward pass
output = torch.relu(layer(input))

# Backward pass (automatic)
loss.backward()

# Parameter update (optimizer handles it)
optimizer.step()
```

**What PyTorch's `nn.Linear` does internally** (simplified):
```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight.T + self.bias

    # backward() is automatic via autograd!
```

### Conv2D Layer Comparison

**Your NumPy implementation**:
```python
class Conv2D:
    def forward(self, x):
        z, self.cache = conv2d_forward(x, self.W, self.b, self.stride, self.padding)
        return self.activation(z)

    def backward(self, da, learning_rate):
        dz = da * self.activation_derivative(self.cache_activation)
        dx, self.dW, self.db = conv2d_backward(dz, self.cache)
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        return dx
```

**PyTorch equivalent**:
```python
# Just use nn.Conv2d!
conv = nn.Conv2d(in_channels=1, out_channels=32,
                 kernel_size=3, stride=1, padding=1)

# Forward pass
output = torch.relu(conv(input))

# Backward pass (automatic!)
loss.backward()
```

**Appreciation moment**: You know what `conv2d_forward` and `conv2d_backward` do! PyTorch just automates it.

---

## Training Loop Pattern

### Standard PyTorch Training Loop

**You wrote this pattern manually** in your `fit()` methods:

```python
# Your NumPy pattern (neural_network.py)
for epoch in range(epochs):
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        # Forward
        y_pred = self.forward(X_batch)
        loss = self.loss_function(y_batch, y_pred)

        # Backward
        self.backward(y_batch, y_pred, learning_rate)
```

**PyTorch equivalent**:
```python
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        # Zero gradients (they accumulate!)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
```

### Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Prepare data
X_train = torch.randn(1000, 784)  # 1000 samples, 784 features
y_train = torch.randint(0, 10, (1000,))  # 10 classes

dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Define model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 3. Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # ← Combines softmax + cross-entropy!
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training loop
for epoch in range(10):
    for X_batch, y_batch in train_loader:
        # Zero gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward
        loss.backward()

        # Update
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()  # ← Set to evaluation mode (affects dropout, batch norm)
with torch.no_grad():  # ← Don't track gradients
    outputs = model(X_test)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()
```

---

## Common Layers

### Fully Connected Layers

```python
# Dense/Linear layer
nn.Linear(in_features, out_features, bias=True)

# Example
fc = nn.Linear(784, 128)
output = fc(input)  # input: (batch, 784) → output: (batch, 128)
```

### Convolutional Layers

```python
# 2D Convolution (for images)
nn.Conv2d(in_channels, out_channels, kernel_size,
          stride=1, padding=0)

# Example
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
# input: (batch, 3, 224, 224) → output: (batch, 64, 224, 224)
```

### Pooling Layers

```python
# Max Pooling
nn.MaxPool2d(kernel_size, stride=None)

# Example
pool = nn.MaxPool2d(2, stride=2)
# input: (batch, 64, 28, 28) → output: (batch, 64, 14, 14)

# Average Pooling
nn.AvgPool2d(kernel_size, stride=None)
```

### Activation Functions

```python
# ReLU (most common)
nn.ReLU()
# or inline: torch.relu(x)

# Leaky ReLU
nn.LeakyReLU(negative_slope=0.01)

# Sigmoid
nn.Sigmoid()
# or inline: torch.sigmoid(x)

# Tanh
nn.Tanh()

# Softmax (usually in loss function, not as layer)
nn.Softmax(dim=1)
```

### Normalization Layers

```python
# Batch Normalization (your notes mention this!)
nn.BatchNorm1d(num_features)  # For linear layers
nn.BatchNorm2d(num_features)  # For conv layers

# Layer Normalization
nn.LayerNorm(normalized_shape)

# Example
bn = nn.BatchNorm2d(64)
# input: (batch, 64, 28, 28) → output: (batch, 64, 28, 28)
# (normalized across batch dimension)
```

### Dropout

```python
# Dropout (regularization)
nn.Dropout(p=0.5)  # Drop 50% of neurons during training

# Example in model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training!
        x = self.fc2(x)
        return x
```

### Recurrent Layers

```python
# LSTM (you have notes on this!)
nn.LSTM(input_size, hidden_size, num_layers=1,
        batch_first=False)

# GRU
nn.GRU(input_size, hidden_size, num_layers=1)

# Example
lstm = nn.LSTM(input_size=100, hidden_size=256, num_layers=2)
# input: (seq_len, batch, 100) → output: (seq_len, batch, 256)
```

---

## Loss Functions & Optimizers

### Common Loss Functions

```python
# Classification
criterion = nn.CrossEntropyLoss()  # ← Multi-class (combines softmax!)
# Expects: (batch, n_classes), (batch,)

criterion = nn.BCELoss()  # Binary cross-entropy (needs sigmoid output)
criterion = nn.BCEWithLogitsLoss()  # BCE + sigmoid (more numerically stable)

# Regression
criterion = nn.MSELoss()  # Mean squared error
criterion = nn.L1Loss()   # Mean absolute error

# Usage
loss = criterion(predictions, targets)
loss.backward()
```

**Important**: `nn.CrossEntropyLoss` includes softmax!
```python
# DON'T do this:
output = torch.softmax(model(x), dim=1)
loss = nn.CrossEntropyLoss()(output, y)  # ✗ Wrong! Double softmax

# DO this:
output = model(x)  # No softmax!
loss = nn.CrossEntropyLoss()(output, y)  # ✓ Correct
```

### Common Optimizers

```python
import torch.optim as optim

# SGD (basic gradient descent)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (adaptive learning rates, most popular)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW (Adam with weight decay, modern standard)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Step decay: reduce LR every N epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Usage in training loop
for epoch in range(epochs):
    train(...)
    scheduler.step()  # Update learning rate
```

---

## Debugging & Best Practices

### Check Shapes

```python
# Print shapes at each step
x = torch.randn(32, 3, 224, 224)
print(f"Input: {x.shape}")

x = conv1(x)
print(f"After conv1: {x.shape}")

x = pool(x)
print(f"After pool: {x.shape}")
```

### Check Gradients

```python
# Are gradients flowing?
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm():.4f}")
    else:
        print(f"{name}: NO GRADIENT!")
```

### Common Errors

**1. Shape mismatch**:
```python
# Error: input (batch, 784) but linear expects (batch, 512)
# Solution: Check shapes, fix model or reshape input
```

**2. Forgot to zero gradients**:
```python
# Gradients accumulate!
for epoch in range(epochs):
    for X, y in loader:
        optimizer.zero_grad()  # ← Don't forget this!
        loss = ...
        loss.backward()
        optimizer.step()
```

**3. Wrong loss function**:
```python
# CrossEntropyLoss expects class indices, not one-hot!
# Input: (batch, n_classes) logits
# Target: (batch,) with class indices [0, 1, 2, ...]

# BCELoss expects probabilities (after sigmoid)
# BCEWithLogitsLoss expects logits (before sigmoid) ← Use this!
```

**4. Model not in training/eval mode**:
```python
# Training
model.train()  # Enables dropout, batch norm in training mode
loss.backward()

# Evaluation
model.eval()  # Disables dropout, batch norm in eval mode
with torch.no_grad():
    predictions = model(X_test)
```

### Best Practices

**1. Device-agnostic code**:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
X, y = X.to(device), y.to(device)
```

**2. Save/load models**:
```python
# Save
torch.save(model.state_dict(), 'model.pth')

# Load
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

**3. Reproducibility**:
```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
```

---

## When to Use Custom Layers

**You can implement custom operations** when built-ins aren't enough:

```python
class MyCustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(10, 10))

    def forward(self, x):
        # Custom forward pass
        return x @ self.weight

# PyTorch handles backward automatically!
```

**When to implement custom**:
- Novel architecture from papers
- Research experiments
- Specific domain requirements

**When to use built-ins**:
- Standard architectures (99% of cases)
- Production code
- When debugging/speed matters

---

## Key Takeaways

### 1. **PyTorch Automates Your Manual Work**

**You implemented**:
- ✅ Backpropagation (chain rule, gradients)
- ✅ Convolution forward/backward
- ✅ Parameter updates
- ✅ Training loops

**PyTorch does all this** with `.backward()` and `optimizer.step()`!

### 2. **Three Core Components**

```python
model = nn.Module(...)      # Architecture
criterion = nn.Loss(...)    # Loss function
optimizer = optim.SGD(...)  # Optimization algorithm
```

### 3. **Standard Training Pattern**

```python
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
```

### 4. **You Understand What's Happening**

**Because you implemented from scratch**, you know:
- What `.backward()` does (backpropagation!)
- What `nn.Linear` does (your Dense layer!)
- What `nn.Conv2d` does (your Conv2D layer!)
- Why gradients are zeroed (they accumulate!)

**This knowledge makes you better at**:
- Debugging (know what could go wrong)
- Custom layers (understand the interface)
- Optimization (know what affects gradients)
- Architecture design (understand tradeoffs)

---

## Next Steps

**Practice with PyTorch**:
1. ✅ **Reimplement your NumPy MLP** in PyTorch
2. ✅ **Reimplement your NumPy CNN** in PyTorch
3. ✅ **Train on real MNIST** (easy with torchvision)
4. ✅ **Compare performance** (PyTorch will be much faster!)

**Then**:
- Fine-tune pre-trained models (transfer learning)
- Implement papers from scratch
- Build real projects

**Resources**:
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- Your from-scratch implementations (reference for understanding!)

---

**You've earned this knowledge**. PyTorch will now make perfect sense because you understand what it automates! 🎉
