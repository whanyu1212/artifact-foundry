# Deep Learning Notes

**A Journey Through Modern Neural Network Architectures**

This directory contains comprehensive notes on the fundamental architectures that define deep learning. Each note is designed for deep understanding - the "why" and "how it works internally," not just the "what."

---

## 📚 Architecture Notes

### 1. [CNN Architecture](cnn_architecture.md) - Computer Vision Foundation

**What**: Convolutional Neural Networks
**Key Innovation**: Local connectivity + parameter sharing + spatial hierarchy
**Best For**: Images, spatial data

**Core Concepts**:
- Convolution operation (sliding window with learned filters)
- Pooling (downsampling and translation invariance)
- Hierarchical feature learning (edges → shapes → objects)

**When to Read**: After understanding basic neural networks (MLPs)
**Complexity**: ⭐⭐ Moderate - convolution is non-intuitive at first
**Implementation Difficulty**: ⭐⭐⭐ Medium - convolution and backprop need careful indexing

**Key Takeaway**: CNNs work because they preserve spatial structure while dramatically reducing parameters through local connectivity and weight sharing.

---

### 2. [RNN Architecture](rnn_architecture.md) - Sequential Data & Memory

**What**: Recurrent Neural Networks, LSTM, GRU
**Key Innovation**: Hidden state maintains memory across time steps
**Best For**: Sequential data (text, time series, audio)

**Core Concepts**:
- Recurrence (h_t depends on h_{t-1})
- Vanishing gradients (why vanilla RNN fails)
- Gates (LSTM/GRU solution for long-term dependencies)

**When to Read**: After CNNs, before Transformers
**Complexity**: ⭐⭐⭐ Hard - gates and BPTT are challenging
**Implementation Difficulty**: ⭐⭐⭐⭐ Very Hard - backpropagation through time is complex

**Key Takeaway**: RNNs process sequences by maintaining memory, but vanishing gradients limit them. LSTM gates enable long-term dependencies. Transformers have largely replaced RNNs, but understanding RNNs helps appreciate why attention is powerful.

---

### 3. [Transformer Architecture](transformer_architecture.md) - The Modern Foundation

**What**: Attention-based architecture (no recurrence!)
**Key Innovation**: Self-attention allows all positions to interact directly
**Best For**: Everything (NLP, vision, multimodal) - the dominant architecture

**Core Concepts**:
- Attention mechanism (Query, Key, Value)
- Self-attention (attend to all positions in parallel)
- Multi-head attention (multiple attention patterns)
- Positional encoding (inject position information)

**When to Read**: After RNNs (to appreciate the paradigm shift)
**Complexity**: ⭐⭐⭐⭐ Very Hard - attention mechanism requires careful study
**Implementation Difficulty**: ⭐⭐⭐⭐ Very Hard - many moving parts (attention, positional encoding, masking)

**Key Takeaway**: Transformers replaced recurrence with attention, enabling parallel processing and better long-range dependencies. Powers GPT, BERT, and modern AI.

---

### 4. [PyTorch Fundamentals](pytorch_fundamentals.md) - From NumPy to Production

**What**: Modern deep learning framework
**Key Innovation**: Automatic differentiation + GPU acceleration + ecosystem
**Best For**: Everything (research and production)

**Core Concepts**:
- Tensors (NumPy++ with gradient tracking)
- Autograd (automatic backpropagation)
- nn.Module (building blocks for models)
- Training patterns and best practices

**When to Read**: After implementing at least one architecture from scratch
**Complexity**: ⭐⭐ Moderate - concepts are familiar from NumPy implementations
**Implementation Difficulty**: ⭐ Easy - PyTorch automates what you've manually done

**Key Takeaway**: PyTorch automates backpropagation, parameter updates, and GPU acceleration. Understanding your from-scratch implementations makes PyTorch completely transparent.

---

## 🗺️ Learning Path

### Recommended Order

```
1. Basic Neural Networks (MLPs) ✅
   ↓
2. CNNs
   - Start here for computer vision
   - Relatively straightforward after MLPs
   ↓
3. RNNs/LSTMs
   - Essential for understanding sequential processing
   - Helps appreciate Transformers
   ↓
4. Transformers
   - The most important architecture today
   - Build on attention from RNNs
```

### Alternative Path (Jump to What You Need)

**For Computer Vision Projects**:
- MLPs → CNNs → Vision Transformers (skip RNNs initially)

**For NLP Projects**:
- MLPs → Transformers directly (skip CNNs and RNNs)
- Go back to RNNs later for historical context

**For Time Series**:
- MLPs → RNNs/LSTMs → Transformers for time series

---

## 📊 Quick Comparison

| Architecture | Input Type | Key Strength | Main Limitation | Status |
|--------------|------------|--------------|-----------------|--------|
| **MLP** | Fixed vectors | Simple, fast | No structure (spatial/temporal) | Still used for tabular data |
| **CNN** | Images, grids | Spatial hierarchy | Fixed receptive field | Still dominant for vision (with ViT) |
| **RNN/LSTM** | Sequences | Sequential memory | Slow (sequential), vanishing gradients | Mostly replaced by Transformers |
| **Transformer** | Any (tokenized) | Parallel, long-range | O(n²) attention cost | Dominant across all domains |
| **PyTorch** | Framework | Autograd, GPU, ecosystem | Learning curve (but easier after NumPy!) | Industry standard for research/production |

---

## 🎯 Implementation Roadmap

### Phase 1: CNNs (2-3 weeks)
**Goal**: Build a CNN from scratch for MNIST

**Steps**:
1. Implement 2D convolution (NumPy)
2. Implement max pooling
3. Implement backpropagation for conv layer
4. Build simple CNN: Conv-ReLU-Pool-FC
5. Train on MNIST, achieve >98% accuracy

**Why**: Convolution is fundamental to understanding spatial processing. Once you implement it, you'll truly get it.

**Deliverables**:
- `snippets/cnn/convolution.py` - Convolution from scratch
- `snippets/cnn/cnn_network.py` - Full CNN implementation
- `examples/mnist_cnn.py` - Training on MNIST

---

### Phase 2: RNNs/LSTMs (2-3 weeks)
**Goal**: Build character-level language model

**Steps**:
1. Implement vanilla RNN (NumPy)
2. Implement LSTM with all three gates
3. Implement backpropagation through time
4. Build character-level model (e.g., Shakespeare text)
5. Generate text autoregressively

**Why**: LSTM gates are mind-bending until you implement them. Generating text shows sequential modeling in action.

**Deliverables**:
- `snippets/rnn/vanilla_rnn.py` - Vanilla RNN
- `snippets/rnn/lstm.py` - LSTM from scratch
- `examples/char_language_model.py` - Text generation

---

### Phase 3: Transformers (3-4 weeks)
**Goal**: Build mini-Transformer for simple translation

**Steps**:
1. Implement scaled dot-product attention
2. Implement multi-head attention
3. Implement positional encoding
4. Build encoder and decoder
5. Train on simple dataset (e.g., number to words, sorting)

**Why**: Transformers are the most important architecture in modern AI. Implementing attention from scratch is enlightening.

**Deliverables**:
- `snippets/transformer/attention.py` - Attention mechanisms
- `snippets/transformer/transformer.py` - Full Transformer
- `examples/simple_translation.py` - Toy translation task

---

## 🔍 Deep Dive Topics

After mastering the three pillars, explore these advanced topics:

### Advanced Architectures
- **ResNet**: Skip connections for very deep networks
- **U-Net**: Encoder-decoder for segmentation
- **Vision Transformers (ViT)**: Transformers for images
- **DETR**: Transformers for object detection

### Training Techniques
- **Batch Normalization**: Stabilize training
- **Data Augmentation**: Improve generalization
- **Transfer Learning**: Leverage pre-trained models
- **Mixed Precision Training**: Faster training with FP16

### Optimization
- **Adam, AdamW**: Adaptive learning rates
- **Learning Rate Scheduling**: Warmup, decay, cyclical
- **Gradient Accumulation**: Simulate large batches
- **Distributed Training**: Multi-GPU, multi-node

### Advanced Transformers
- **BERT**: Bidirectional encoder (understanding)
- **GPT**: Decoder-only (generation)
- **T5**: Encoder-decoder (versatile)
- **Efficient Transformers**: Sparse, linear attention

---

## 💡 Study Tips

### For Maximum Understanding

**1. Read Actively**
- Don't just read - work through examples
- Derive equations yourself
- Sketch architectures on paper

**2. Implement from Scratch**
- Don't use frameworks until you've implemented manually
- NumPy forces you to understand every operation
- Compare your implementation to PyTorch (should match!)

**3. Visualize**
- Plot attention weights (see what model learns)
- Visualize filters (CNNs)
- Watch hidden states evolve (RNNs)

**4. Experiment**
- Change hyperparameters, observe effects
- Break things intentionally, understand failure modes
- Try simple datasets first (MNIST, tiny Shakespeare)

**5. Teach Others**
- Explain concepts to someone else
- Write your own notes/blog posts
- Answer questions on forums

### Common Pitfalls

**❌ Jumping to frameworks too quickly**
- "I'll just use PyTorch" → Miss deep understanding
- ✅ Implement from scratch first, THEN use frameworks

**❌ Skipping math**
- "I'll skip the equations" → Can't debug or innovate
- ✅ Understand the math, even if you don't memorize it

**❌ Tutorial hell**
- Watching endless videos without building
- ✅ Build projects, get your hands dirty

**❌ Trying to learn everything**
- Superficial knowledge of 10 architectures
- ✅ Deep understanding of 3 architectures > Shallow knowledge of 10

---

## 📖 Essential Papers

### Must-Read Papers

**CNNs**:
- ImageNet Classification (AlexNet) - Krizhevsky et al., 2012
- Very Deep Networks (VGG) - Simonyan & Zisserman, 2014
- Deep Residual Learning (ResNet) - He et al., 2015

**RNNs**:
- LSTM - Hochreiter & Schmidhuber, 1997
- GRU - Cho et al., 2014
- Sequence to Sequence Learning - Sutskever et al., 2014

**Transformers**:
- Attention Is All You Need - Vaswani et al., 2017 ⭐ THE PAPER
- BERT - Devlin et al., 2018
- GPT-2/3 - Radford et al., 2019; Brown et al., 2020

### How to Read Papers

1. **First pass** (10 min): Abstract, intro, figures - get the gist
2. **Second pass** (1 hour): Read carefully, skip proofs
3. **Third pass** (3+ hours): Understand every detail, reproduce results

---

## 🎓 Learning Resources

### Books
- **Deep Learning** by Goodfellow, Bengio, Courville (comprehensive)
- **Dive into Deep Learning** (interactive, code-focused)

### Courses
- **CS231n** (Stanford): CNNs for Visual Recognition
- **CS224n** (Stanford): NLP with Deep Learning
- **Fast.ai**: Practical Deep Learning

### Blogs
- **The Illustrated Transformer** - Jay Alammar
- **Distill.pub** - Visual explanations
- **Sebastian Ruder** - NLP research

### Code
- **Andrej Karpathy**: micrograd, minGPT (minimal implementations)
- **HuggingFace Transformers**: Production-ready implementations

---

## 🏗️ Project Ideas

### Beginner
- MNIST digit classification (CNN)
- IMDB sentiment analysis (RNN)
- Name generation (character RNN)

### Intermediate
- Object detection (YOLO-style CNN)
- Machine translation (Transformer)
- Image captioning (CNN encoder + RNN/Transformer decoder)

### Advanced
- Build your own Vision Transformer
- Implement efficient attention (Linformer, Performer)
- Multi-modal model (CLIP-style)

---

## ✅ Next Steps

**You are here**: ✅ Comprehensive notes for CNN, RNN, Transformer

**Recommended next action**:
1. **Review the notes** - Read through at your own pace
2. **Pick one architecture** to implement (suggest: CNN first)
3. **Build from scratch** using NumPy
4. **Create project** in your PROJECTS.md when complete

**Questions to consider**:
- Which architecture interests you most?
- What's your target application (vision, NLP, time series)?
- How much time can you dedicate to implementation?

---

## 🤝 The Foundry Philosophy Applied

**Understanding over completion**:
- Don't rush through all three architectures
- Pick one, understand it deeply, implement it thoroughly

**Implementation over theory**:
- These notes give you the theory
- Now BUILD to truly internalize

**Fundamentals over frameworks**:
- NumPy implementations before PyTorch
- Understand the mechanics before the abstraction

**Quality over quantity**:
- One well-implemented CNN > Ten copy-pasted PyTorch tutorials

---

**Ready to build?** Start with the architecture that excites you most! 🚀
