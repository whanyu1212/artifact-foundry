# Tokenization in LLMs

## Overview

Tokenization is the process of converting raw text into discrete units (tokens) that language models can process. The choice of tokenization strategy significantly impacts model vocabulary size, training efficiency, and ability to handle rare words and multiple languages.

## Why Tokenization Matters

- **Vocabulary size trade-off**: Character-level (small vocab, long sequences) vs word-level (huge vocab, short sequences)
- **Subword units**: Balance between vocabulary size and sequence length
- **Handling rare words**: Break unknown words into known subword pieces
- **Multilingual capability**: Share subword pieces across languages

## Byte Pair Encoding (BPE)

### Algorithm

1. Start with character-level vocabulary
2. Find most frequent pair of consecutive tokens
3. Merge this pair into a new token
4. Repeat for desired vocabulary size

### Key Properties

- **Data-driven**: Vocabulary learned from training corpus
- **Deterministic**: Given merge operations, tokenization is reproducible
- **Greedy**: Always merges most frequent pair (not globally optimal)

### Example

```
Initial: "l o w e r"
Merge 1: "low er" (if 'lo' is most frequent pair)
Merge 2: "low er" → "lower" (if 'lower' is formed)
```

### Use Cases

- GPT-2, GPT-3, RoBERTa
- Good for: English and similar languages
- Challenges: Can struggle with morphologically rich languages

## WordPiece

### Differences from BPE

- **Selection criterion**: Choose merge that maximizes likelihood of training data (not just frequency)
- **Prefix markers**: Uses `##` to denote word-internal pieces
- **Vocabulary**: Tends to create more linguistically meaningful subwords

### Example

```
"unwanted" → ["un", "##want", "##ed"]
"playing" → ["play", "##ing"]
```

### Use Cases

- BERT, DistilBERT
- Better handling of morphology than BPE
- Prefix markers help model understand word boundaries

## SentencePiece

### Key Innovation

- **Language-agnostic**: Treats text as raw byte stream, no pre-tokenization
- **Reversible**: Can perfectly reconstruct original text including spaces
- **No language assumptions**: Works for languages without explicit word boundaries (Chinese, Japanese)

### Approaches

1. **BPE mode**: Similar to standard BPE but on raw text
2. **Unigram mode**: Train language model to select optimal subwords

### Use Cases

- T5, ALBERT, XLNet
- Essential for multilingual models
- Best for: Asian languages, code, noisy text

## Practical Considerations

### Vocabulary Size

| Size | Trade-offs |
|------|------------|
| Small (10K) | Longer sequences, more computation, better generalization |
| Medium (32K) | Balanced, common choice |
| Large (100K+) | Shorter sequences, more rare word coverage, larger embedding matrix |

### Special Tokens

- `[PAD]`: Padding for batching
- `[UNK]`: Unknown/out-of-vocabulary tokens
- `[CLS]`: Classification token (BERT)
- `[SEP]`: Separator between sequences
- `[MASK]`: Masked token for pretraining

### Encoding/Decoding

```python
# Encoding
text = "Hello world"
token_ids = tokenizer.encode(text)  # [15496, 995]

# Decoding
reconstructed = tokenizer.decode(token_ids)  # "Hello world"
```

## Common Pitfalls

1. **Tokenizer mismatch**: Must use same tokenizer for training and inference
2. **Vocabulary drift**: Retraining tokenizer changes token IDs
3. **Subword artifacts**: Model may struggle with tokens split differently than training
4. **OOV handling**: New domains may have many unknown tokens

## Implementation Notes

- **Preprocessing**: Lowercase, accent removal, punctuation handling
- **Byte-level fallback**: Handle any Unicode character
- **Caching**: Token vocabularies and merge operations are precomputed
- **Efficiency**: Fast tokenization crucial for large-scale training

## Further Reading

- Original BPE paper: Sennrich et al. (2016)
- WordPiece: Schuster & Nakajima (2012)
- SentencePiece: Kudo & Richardson (2018)
