# Convolutional Neural Networks (CNNs)

**The Architecture for Computer Vision**

CNNs revolutionized computer vision by learning spatial hierarchies automatically. Instead of hand-crafting features, CNNs learn them through convolution operations that preserve spatial relationships.

---

## Table of Contents

1. [Core Intuition](#core-intuition)
2. [The Convolution Operation](#the-convolution-operation)
3. [Key Components](#key-components)
4. [CNN Architecture Patterns](#cnn-architecture-patterns)
5. [Why CNNs Work for Vision](#why-cnns-work-for-vision)
6. [Mathematical Foundations](#mathematical-foundations)
7. [Training CNNs](#training-cnns)
8. [Common Architectures](#common-architectures)
9. [Best Practices](#best-practices)
10. [When to Use CNNs](#when-to-use-cnns)

---

## Core Intuition

### The Problem with Fully Connected Networks for Images

Consider a 224×224 RGB image:
- **Pixels**: 224 × 224 × 3 = 150,528 inputs
- **First hidden layer** (1000 neurons): 150,528 × 1,000 = **150 million parameters**!

**Problems**:
1. **Too many parameters**: Computationally expensive, prone to overfitting
2. **No spatial structure**: Treats pixel at (0,0) the same as pixel at (223,223)
3. **Not translation invariant**: A cat in the top-left requires different features than a cat in the bottom-right

### The CNN Solution

**Key insights**:
1. **Local connectivity**: Each neuron only looks at a small region (receptive field)
2. **Parameter sharing**: Same filter/kernel used across entire image
3. **Spatial hierarchy**: Learn simple features (edges) → complex features (shapes) → objects

**Result**: Dramatically fewer parameters while preserving spatial relationships.

---

## The Convolution Operation

### What is Convolution?

Convolution is a **sliding window operation** that applies a small filter/kernel across an image.

```
Image (5×5):                Filter/Kernel (3×3):
┌─────────────┐             ┌─────────┐
│ 1 2 3 4 5 │             │ 1 0 -1 │
│ 6 7 8 9 0 │             │ 1 0 -1 │
│ 1 2 3 4 5 │             │ 1 0 -1 │
│ 6 7 8 9 0 │             └─────────┘
│ 1 2 3 4 5 │
└─────────────┘
```

### How It Works

**Step 1**: Place filter on top-left of image
```
┌─────────────┐
│[1 2 3]4 5 │  ← Filter covers these 9 pixels
│[6 7 8]9 0 │
│[1 2 3]4 5 │
│ 6 7 8 9 0 │
│ 1 2 3 4 5 │
└─────────────┘
```

**Step 2**: Element-wise multiplication and sum
```
Output[0,0] = (1×1) + (2×0) + (3×-1) +
               (6×1) + (7×0) + (8×-1) +
               (1×1) + (2×0) + (3×-1)
            = 1 + 0 - 3 + 6 + 0 - 8 + 1 + 0 - 3
            = -6
```

**Step 3**: Slide filter right (by stride), repeat
**Step 4**: Slide filter down when row complete

### Mathematical Definition

For input **X** and filter **W**:

```
Y[i,j] = Σₘ Σₙ X[i+m, j+n] × W[m,n] + b
```

Where:
- **Y**: Output feature map
- **X**: Input (image or previous layer's output)
- **W**: Filter/kernel weights (learned during training)
- **b**: Bias term
- **m, n**: Iterate over filter dimensions

---

## Key Components

### 1. Convolutional Layer

**Purpose**: Extract features through learned filters

**Hyperparameters**:
- **Number of filters**: How many different features to learn (e.g., 32, 64, 128)
- **Filter size**: Usually 3×3 or 5×5 (odd sizes allow centered pixel)
- **Stride**: How many pixels to skip when sliding (1 = slide by 1 pixel)
- **Padding**: Add zeros around border to control output size

**Output size formula**:
```
output_size = ⌊(input_size - filter_size + 2×padding) / stride⌋ + 1
```

**Example**:
- Input: 32×32, Filter: 5×5, Stride: 1, Padding: 2
- Output: ⌊(32 - 5 + 4) / 1⌋ + 1 = 32×32 (same size!)

**Parameter count**:
```
# filters × (filter_h × filter_w × input_channels) + # filters (bias)
```

For 64 filters of 3×3 on RGB image: 64 × (3×3×3) + 64 = 1,792 parameters

### 2. Pooling Layer

**Purpose**: Reduce spatial dimensions (downsample), provide translation invariance

**Types**:

**Max Pooling** (most common):
```
Input (4×4):              2×2 Max Pool (stride=2):
┌──────────────┐          ┌─────┐
│ 1  3  2  4 │          │ 3 4 │
│ 5  6  7  8 │    →     │ 8 9 │
│ 3  2  1  2 │          └─────┘
│ 1  3  5  9 │
└──────────────┘
```

Takes maximum value in each window. Provides:
- **Size reduction**: 4×4 → 2×2
- **Translation invariance**: Small shifts don't change max
- **Noise reduction**: Suppresses non-maximum activations

**Average Pooling**:
Takes average instead of max. Less common now, but used in some architectures.

**No learnable parameters**: Just a fixed operation

### 3. Activation Functions

**ReLU** (most common for CNNs):
- `f(x) = max(0, x)`
- Fast, no vanishing gradient for positive values
- Applied after each convolution

**Why after convolution?**
- Convolution is linear operation
- Need non-linearity to learn complex patterns
- `Conv → ReLU → Pool` is the standard pattern

### 4. Fully Connected Layers

**Purpose**: Combine features for final classification

**Placement**: After convolutional + pooling layers
- Flatten: Convert 3D feature maps to 1D vector
- Dense layers: Learn combinations of high-level features
- Output layer: Produce class predictions

---

## CNN Architecture Patterns

### Classic Pattern: Feature Extraction → Classification

```
INPUT → [CONV → RELU → POOL]×N → [FC → RELU]×M → OUTPUT
```

**Feature Extraction Stage** (`[CONV → RELU → POOL]×N`):
- Multiple conv-relu-pool blocks
- Each block: spatial size ↓, channels ↑
- Example: 224×224×3 → 112×112×64 → 56×56×128 → 28×28×256

**Classification Stage** (`[FC → RELU]×M → OUTPUT`):
- Flatten feature maps
- Fully connected layers
- Final layer: softmax for classification

### Typical Progression

```
Layer Type    | Spatial Size | Channels | Parameters
--------------|--------------|----------|------------
Input         | 224×224      | 3        | 0
Conv1 + Pool  | 112×112      | 64       | ~2K
Conv2 + Pool  | 56×56        | 128      | ~74K
Conv3 + Pool  | 28×28        | 256      | ~295K
Conv4 + Pool  | 14×14        | 512      | ~1.2M
Conv5 + Pool  | 7×7          | 512      | ~2.4M
Flatten       | 25,088       | -        | 0
FC1           | 4096         | -        | ~103M
FC2           | 4096         | -        | ~16M
Output        | 1000         | -        | ~4M
```

**Pattern**: Spatial dimensions decrease, feature depth increases

---

## Why CNNs Work for Vision

### 1. **Local Connectivity**

**Observation**: Nearby pixels are more related than distant pixels
- Edge formed by adjacent pixels
- Corner is a local pattern
- Object parts are spatially coherent

**CNN approach**: Each neuron only connects to local region
- **Receptive field**: Small window (e.g., 3×3, 5×5)
- Reduces parameters dramatically
- Focuses learning on local patterns

### 2. **Parameter Sharing**

**Observation**: Useful features can appear anywhere in image
- Edge detector useful in top-left AND bottom-right
- Cat ears look similar regardless of position

**CNN approach**: Same filter used across entire image
- One edge detector for whole image (not one per location)
- **Translation invariance**: Feature learned once, detected everywhere
- Massively reduces parameters

**Example**:
- Fully connected: 1000 different weights for 1000 positions
- CNN: 1 filter (9 weights for 3×3) shared across all positions

### 3. **Hierarchical Feature Learning**

**Observation**: Vision is hierarchical (edges → shapes → objects)

**CNN approach**: Stack layers to build complexity
```
Layer 1: Edge detectors (horizontal, vertical, diagonal)
    ↓
Layer 2: Textures, simple shapes (curves, corners)
    ↓
Layer 3: Object parts (eyes, wheels, windows)
    ↓
Layer 4: Objects (faces, cars, buildings)
```

**Why it works**: Each layer's receptive field grows
- Layer 1: Sees 3×3 pixels
- Layer 2: Sees 5×5 (through Layer 1's 3×3 windows)
- Layer 3: Sees 7×7
- Deeper layers → Larger context → More complex features

### 4. **Translation Invariance via Pooling**

**Observation**: Object identity shouldn't change with small shifts

**Pooling approach**:
- Max pooling: "Is feature present in this region?" (yes/no, not exact position)
- Slight shifts still activate same max
- Reduces sensitivity to exact positioning

---

## Mathematical Foundations

### Forward Propagation

For layer `l`:

**Convolution**:
```
Z[l][i,j,k] = Σₘ Σₙ Σ꜀ W[l][m,n,c,k] × A[l-1][i+m, j+n, c] + b[l][k]
```

Where:
- `Z[l]`: Pre-activation output
- `W[l]`: Filters (weights)
- `A[l-1]`: Previous layer's activation
- `i,j`: Spatial location in output
- `k`: Output channel (which filter)
- `m,n`: Filter dimensions
- `c`: Input channels

**Activation**:
```
A[l] = σ(Z[l])    # Usually ReLU for hidden layers
```

**Pooling** (max pooling, 2×2):
```
A[l][i,j,k] = max(A[l-1][2i:2i+2, 2j:2j+2, k])
```

### Backpropagation Through Convolution

**Challenge**: Convolution is parameter sharing, so gradients must be accumulated

**Gradient w.r.t. filter** (sum over all positions where filter was applied):
```
∂L/∂W[m,n,c,k] = Σᵢ Σⱼ (∂L/∂Z[i,j,k]) × A[l-1][i+m, j+n, c]
```

**Gradient w.r.t. previous activation** (rotate filter 180°, convolve with gradient):
```
∂L/∂A[l-1][i,j,c] = Σₖ Σₘ Σₙ W[m,n,c,k] × (∂L/∂Z[i-m, j-n, k])
```

**Key insight**: Backprop through conv is also a convolution (with flipped filter)

### Backpropagation Through Pooling

**Max pooling**: Gradient flows only to the max element
```
∂L/∂A[l-1][i,j,k] = {
    ∂L/∂A[l][i',j',k]   if (i,j) was the max
    0                    otherwise
}
```

**Average pooling**: Gradient split equally among all elements
```
∂L/∂A[l-1][i,j,k] = (∂L/∂A[l][i',j',k]) / (pool_h × pool_w)
```

---

## Training CNNs

### Data Augmentation

**Critical for CNNs**: Limited data → overfitting

**Common augmentations**:
1. **Random crops**: 224×224 crops from 256×256 image
2. **Horizontal flips**: Mirror image
3. **Color jittering**: Adjust brightness, contrast, saturation
4. **Rotation**: Small angles (±15°)
5. **Scaling**: Zoom in/out slightly

**Why it works**: Creates "new" training examples, improves generalization

### Batch Normalization

**Problem**: Internal covariate shift (layer inputs' distribution changes during training)

**Solution**: Normalize activations within each mini-batch
```
BN(x) = γ × (x - μ_batch) / √(σ²_batch + ε) + β
```

**Benefits**:
- Faster training (higher learning rates possible)
- Less sensitive to initialization
- Regularization effect (slight)

**Placement**: Usually after Conv, before activation
```
Conv → BatchNorm → ReLU → Pool
```

### Transfer Learning

**Observation**: Low-level features (edges, textures) are universal

**Approach**:
1. **Pre-train** on large dataset (ImageNet: 1.2M images, 1000 classes)
2. **Fine-tune** on target task with smaller dataset

**Two strategies**:
1. **Feature extractor**: Freeze conv layers, train only FC layers
2. **Fine-tuning**: Train entire network with small learning rate

**Why it works**:
- Early layers: general features (useful for all vision tasks)
- Later layers: task-specific features (can be adapted)

---

## Common Architectures

### LeNet-5 (1998) - The Pioneer

**Architecture**: `Conv-Pool-Conv-Pool-FC-FC`
- Designed for handwritten digit recognition (MNIST)
- **Significance**: Proved CNNs work for real-world tasks

### AlexNet (2012) - ImageNet Breakthrough

**Architecture**: 8 layers (5 conv, 3 FC)
- **Innovation**: ReLU (instead of tanh), Dropout, GPU training
- **Impact**: Won ImageNet 2012 by huge margin, started deep learning revolution

### VGG (2014) - Simplicity & Depth

**Key idea**: Stack many small (3×3) filters instead of large filters
- VGG-16: 16 layers
- VGG-19: 19 layers

**Why 3×3?**:
- Two 3×3 convs = same receptive field as one 5×5
- Fewer parameters: 2×(3²) = 18 vs 5² = 25
- More non-linearity (ReLU after each layer)

**Pattern**: `[CONV-CONV-POOL]` repeated, channels double each pool

### ResNet (2015) - Very Deep Networks

**Problem**: Very deep networks (>20 layers) degrade performance

**Solution**: **Skip connections** (residual connections)
```
      x ────────────→ (+) → ReLU → output
       ↓              ↑
       Conv-ReLU-Conv
```

Instead of learning `H(x)`, learn `F(x) = H(x) - x` (the residual)
- Easier to learn identity mapping if needed
- Gradients flow directly through skip connections

**Impact**: Enabled 152-layer networks (ResNet-152)

### Inception (GoogLeNet) (2014) - Multi-Scale Features

**Key idea**: Apply multiple filter sizes in parallel
```
        Input
    ┌────┼────┬────┐
   1×1  3×3  5×5  MaxPool
    └────┼────┴────┘
      Concatenate
```

**Why?**: Different filter sizes capture different scale features
- 1×1: Point-wise features
- 3×3: Small local patterns
- 5×5: Larger patterns

### EfficientNet (2019) - Scaling CNNs

**Key idea**: Balanced scaling of depth, width, and resolution
- Not just deeper or wider, but all dimensions together
- Achieves SOTA with fewer parameters

---

## Best Practices

### Architecture Design

1. **Start simple**: Don't immediately use ResNet-152
   - Begin with 3-5 conv layers
   - Add complexity if needed

2. **Filter progression**:
   - Start with fewer channels (32, 64)
   - Double channels after each pooling (64 → 128 → 256)

3. **Filter sizes**:
   - First layer: Can use 5×5 or 7×7 (larger receptive field)
   - Deeper layers: Stick to 3×3 (proven to work)

4. **Pooling**:
   - Max pooling with 2×2 windows, stride 2
   - Don't pool too aggressively (preserve spatial info early on)

### Regularization

1. **Dropout**: 0.5 in FC layers (not in conv layers usually)
2. **Weight decay (L2)**: Small coefficient (1e-4 to 1e-5)
3. **Data augmentation**: Essential, use heavily
4. **Batch normalization**: Nearly always beneficial

### Training

1. **Learning rate**:
   - Start with 0.001 (Adam) or 0.01 (SGD)
   - Reduce when validation loss plateaus

2. **Batch size**:
   - Larger is better (32, 64, 128) if GPU memory allows
   - Batch normalization works better with larger batches

3. **Optimizer**:
   - Adam: Good default (adaptive learning rates)
   - SGD + Momentum: Often better final performance but needs tuning

---

## When to Use CNNs

### ✅ **Perfect for:**

1. **Image Classification**
   - Object recognition
   - Scene classification
   - Medical image diagnosis

2. **Object Detection**
   - Locating objects in images (R-CNN, YOLO, etc.)
   - Face detection

3. **Image Segmentation**
   - Semantic segmentation (label each pixel)
   - Instance segmentation (separate objects)

4. **Image Generation**
   - GANs use CNNs for generator/discriminator
   - Style transfer

5. **Video Analysis**
   - 3D convolutions (add time dimension)
   - Action recognition

6. **Any Spatial Data**
   - Audio spectrograms (time-frequency images)
   - Time series with spatial structure

### ❌ **Not ideal for:**

1. **Sequential data** without spatial structure (use RNNs/Transformers)
2. **Tabular data** (regular neural networks or tree-based models)
3. **Variable-length sequences** (unless you want to pad/resize)
4. **Graph data** (use Graph Neural Networks)

### 🤔 **Considerations:**

1. **Data requirements**: CNNs need lots of data (use transfer learning if limited)
2. **Computation**: Training is GPU-intensive
3. **Interpretability**: "Black box" - hard to explain decisions

---

## Key Takeaways

1. **Convolution = local, shared, hierarchical**
   - Local connectivity reduces parameters
   - Parameter sharing enables translation invariance
   - Hierarchical layers build complexity

2. **The magic three**: Conv, ReLU, Pool
   - Conv: Extract features
   - ReLU: Non-linearity
   - Pool: Reduce dimensions, invariance

3. **Bigger isn't always better**
   - Smart architectures (ResNet, Inception) > just deeper
   - 3×3 filters surprisingly effective

4. **Transfer learning is powerful**
   - Pre-trained models save time and data
   - Fine-tuning beats training from scratch on small datasets

5. **CNNs revolutionized vision**
   - From hand-crafted features to learned features
   - Enabled modern computer vision applications

---

## Next Steps

**To truly understand CNNs**:
1. ✅ Read these notes
2. 🔨 **Implement convolution from scratch** (NumPy)
3. 🔨 **Build a simple CNN** for MNIST
4. 🔨 **Visualize learned filters** (see what network learned)
5. 🔨 **Try transfer learning** on a small dataset

**Related topics to explore**:
- Advanced architectures (ResNet, Inception, EfficientNet)
- Object detection (R-CNN, YOLO, SSD)
- Semantic segmentation (U-Net, FCN)
- Visualization techniques (Grad-CAM, activation maximization)
