# Fine-Tuning LLMs

## Overview

Fine-tuning adapts a pretrained language model to a specific task or domain using task-specific data. The pretrained model provides a strong initialization, requiring far less data and compute than training from scratch.

## Full Fine-Tuning

### Process

1. Load pretrained model weights
2. (Optional) Add task-specific head
3. Train all parameters on task data
4. Save fine-tuned checkpoint

### Training Objective

```python
for batch in task_data:
    logits = model(batch['input_ids'])
    loss = task_loss_function(logits, batch['labels'])
    loss.backward()
    optimizer.step()
```

### Learning Rate

Much smaller than pretraining (typically 1e-5 to 5e-5):
- Pretrained weights are good; avoid catastrophic forgetting
- Fine-tuning is fine-grained adjustment, not wholesale learning

### Advantages

- Maximum performance on target task
- Simple to implement
- Full model capacity available

### Disadvantages

- Requires storing full model copy per task
- High memory requirements (backprop through all layers)
- Risk of catastrophic forgetting (overwrites pretrained knowledge)
- Expensive for billion-parameter models

## Parameter-Efficient Fine-Tuning (PEFT)

Goal: Adapt models with minimal trainable parameters while maintaining performance.

### Why PEFT?

- **Storage**: One base model + small adapters per task
- **Memory**: Fewer gradients to compute and store
- **Speed**: Faster training with fewer parameters
- **Stability**: Pretrained weights frozen, reducing overfitting

## Low-Rank Adaptation (LoRA)

### Core Idea

Model weight updates are low-rank:

$$W' = W + \Delta W = W + BA$$

where:
- $W \in \mathbb{R}^{d \times k}$: Original frozen weight matrix
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$: Trainable low-rank matrices
- $r \ll \min(d, k)$: Rank (typically 8-64)

### Why Low-Rank?

Weight updates during fine-tuning have low "intrinsic dimensionality" - they lie in a low-dimensional subspace. LoRA exploits this.

### Implementation

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)  # Frozen
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank  # alpha is hyperparameter
        
    def forward(self, x):
        frozen_output = self.W(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return frozen_output + lora_output
```

### Parameter Reduction

For a $d \times k$ weight matrix:
- Full fine-tuning: $d \times k$ parameters
- LoRA: $r \times (d + k)$ parameters
- Reduction factor: $\frac{dk}{r(d+k)}$

**Example**: $d=k=4096, r=8$
- Full: 16.7M parameters
- LoRA: 65K parameters (256× reduction)

### Which Layers to Apply LoRA?

Most common: Query and Value projection matrices in attention
- Empirically most impactful
- Can also apply to all linear layers (more parameters, better performance)

### Advantages

- 10-10,000× fewer trainable parameters
- No inference latency (merge $BA$ into $W$ after training)
- Multiple LoRA adapters can share one base model
- Works remarkably well in practice

### Disadvantages

- Slight performance drop vs. full fine-tuning (usually negligible)
- Adds hyperparameters ($r$, $\alpha$, which layers)

## Adapter Layers

### Architecture

Insert small bottleneck layers between transformer blocks:

```
Input (d_model)
    ↓
Down-project (d_model → d_adapter)  # e.g., 1024 → 64
    ↓
Non-linearity (ReLU/GELU)
    ↓
Up-project (d_adapter → d_model)    # 64 → 1024
    ↓
Residual connection (add input)
```

### Trainable Parameters

Only adapter weights are updated; rest of model frozen.

### Parameter Count

Per adapter layer: $2 \times d_{\text{model}} \times d_{\text{adapter}}$

**Example**: $d_{\text{model}}=1024, d_{\text{adapter}}=64$
- Per layer: 131K parameters
- 12 layers: 1.6M parameters

### Advantages

- Simple and interpretable
- Easy to swap adapters for different tasks
- Proven effectiveness (comparable to full fine-tuning)

### Disadvantages

- Adds inference latency (extra forward passes)
- Not as parameter-efficient as LoRA

## Prompt Tuning / Prefix Tuning

### Idea

Prepend learnable "soft prompts" (continuous embeddings) to inputs:

```
Input: [v1, v2, ..., vP, x1, x2, ..., xn]
       └─ learnable ─┘ └── input tokens ──┘
```

### Prefix Tuning

Similar but adds learnable keys and values at each layer, not just input embeddings.

### Parameter Count

Extremely small: $P \times d_{\text{model}}$ (e.g., 100 prompts × 1024 dim = 102K parameters)

### Advantages

- Minimal parameters
- No architecture changes
- One model serves all tasks simultaneously

### Disadvantages

- Reduces effective sequence length
- Performance often below LoRA or adapters
- Sensitive to initialization

## QLoRA (Quantized LoRA)

### Innovation

Combine LoRA with quantization:
1. Load base model in 4-bit precision
2. Add LoRA adapters in full precision (16-bit or 32-bit)
3. Train adapters only

### Memory Savings

- Base model: 4× smaller (4-bit vs 16-bit)
- Adapters: Small anyway (LoRA)
- Enables fine-tuning 65B+ models on single GPU

### Trade-offs

- Slightly slower training (quantization overhead)
- Negligible performance drop from quantization
- Requires careful implementation (bitsandbytes library)

## Instruction Fine-Tuning

### Objective

Teach model to follow instructions through supervised examples:

```
Input:  "Summarize this paragraph: [text]"
Output: [summary]

Input:  "Translate to French: Hello"
Output: "Bonjour"
```

### Dataset Format

Typically: Instruction + (optional) input + output

### Models

- **FLAN-T5**: T5 instruction-tuned on diverse tasks
- **InstructGPT**: GPT-3 + instruction tuning + RLHF
- **Alpaca**: LLaMA instruction-tuned with GPT-3.5-generated data

### Why It Works

Pretrained models have knowledge but don't naturally follow explicit instructions. Fine-tuning aligns model behavior with user intent.

## Reinforcement Learning from Human Feedback (RLHF)

### Motivation

Optimize for human preferences, not just likelihood of training data.

### Three-Stage Process

1. **Supervised Fine-Tuning (SFT)**: Train on high-quality human demonstrations
2. **Reward Model Training**: Train classifier to predict human preference between outputs
3. **RL Fine-Tuning**: Use PPO to optimize policy (LLM) against reward model

### Reward Model

Given two model outputs $y_1, y_2$ for prompt $x$, predict which humans prefer:

$$P(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$$

Train $r(\cdot)$ on human preference data.

### PPO Objective

$$\max_\pi \mathbb{E}_{x,y} [r(x,y) - \beta \cdot \text{KL}(\pi || \pi_{\text{SFT}})]$$

- Maximize reward
- KL penalty prevents model from deviating too far from SFT initialization

### Challenges

- Expensive: Requires human annotation and multiple training stages
- Instability: RL training can be tricky
- Reward hacking: Model exploits reward model weaknesses

### Examples

- ChatGPT (InstructGPT)
- Claude (Constitutional AI variant)

## Best Practices

### Data Quality

- Clean, task-relevant data > large, noisy data
- Balance dataset to avoid bias
- Include challenging examples

### Hyperparameters

- **Learning rate**: 1e-5 to 5e-5 for full fine-tuning; can be higher for PEFT
- **Batch size**: As large as memory allows
- **Epochs**: 2-5 typically sufficient (avoid overfitting)
- **Warmup**: 5-10% of training steps

### Evaluation

- Hold out validation set
- Monitor for overfitting (train/val gap)
- Test on diverse examples
- Measure catastrophic forgetting on general tasks

### Catastrophic Forgetting

**Problem**: Fine-tuning can degrade general capabilities

**Solutions**:
- Lower learning rate
- Mix task data with general pretraining data
- Regularization (weight decay, dropout)
- PEFT methods (freeze most parameters)

## When to Use Each Method

| Method | Use When |
|--------|----------|
| Full fine-tuning | Maximum performance, have compute/storage |
| LoRA | Best balance of efficiency and performance |
| Adapters | Need modular task-specific components |
| Prompt tuning | Extremely limited resources, many tasks |
| QLoRA | Large models, single GPU |
| Instruction tuning | Improve instruction-following ability |
| RLHF | Align with human preferences and values |

## Further Reading

- LoRA: Hu et al. (2021)
- Adapters: Houlsby et al. (2019)
- Prefix Tuning: Li & Liang (2021)
- InstructGPT: Ouyang et al. (2022)
- QLoRA: Dettmers et al. (2023)
