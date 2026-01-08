# Large Language Models (LLMs)

This folder contains notes, implementations, and resources for understanding Large Language Models from the ground up.

## Scope

Focus on **fundamental LLM concepts and architectures**:
- Tokenization strategies
- Transformer architecture and attention mechanisms
- Pretraining objectives and approaches
- Fine-tuning methods
- Model architectures (GPT, BERT, T5 families)
- Scaling laws and emergent abilities
- Evaluation metrics for language models

**Not covered here** (see other folders):
- RAG systems → `ai-engineering/`
- Agent frameworks → `ai-engineering/`
- Production deployment → `productionization/`
- General deep learning → `deep-learning/`

## Notes Organization

- `tokenization.md` - BPE, WordPiece, SentencePiece, vocabulary management
- `attention-mechanisms.md` - Self-attention, multi-head attention, variants
- `pretraining.md` - Language modeling objectives (CLM, MLM, etc.)
- `fine-tuning.md` - Full fine-tuning, LoRA, adapters, PEFT methods
- `model-architectures.md` - GPT, BERT, T5, and their variants
- `scaling-laws.md` - How model performance scales with compute, data, parameters
- `evaluation.md` - Perplexity, benchmark datasets, human evaluation

## Learning Approach

As with all content in this repository:
- Build from-scratch implementations where practical
- Understand the math and theory behind techniques
- Compare research approaches and their trade-offs
- Archive hands-on learning experiences
