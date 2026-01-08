# Context Engineering

## Overview

**Context engineering** is the practice of designing, optimizing, and managing the information provided to language models to maximize response quality, accuracy, and efficiency. While prompt engineering focuses on instruction design, context engineering focuses on the *information* the model uses to generate responses.

**Why It Matters**:
- Context directly influences model behavior and output quality
- Context window limits require strategic information selection
- Poorly organized context leads to "lost in the middle" phenomenon
- Context costs money (input tokens) and time (latency)

## Core Concepts

### Context Window

The maximum number of tokens a model can process in a single request (input + output).

**Model Capacities** (as of 2025-2026):
- GPT-4: 8K, 32K, 128K variants
- GPT-4 Turbo: 128K
- Claude 3 (Opus/Sonnet): 200K
- Claude 3.5 Sonnet: 200K
- Gemini 1.5 Pro: 1M (1 million tokens)
- Command R+: 128K

**Considerations**:
- Larger windows ≠ better performance (attention dilution)
- Cost scales with context size
- Latency increases with longer context
- Not all models utilize full window effectively

### Effective Context

Not all context is used equally:

```
[System Message] ← High attention
[Recent Messages] ← High attention
[Middle Context] ← Lower attention ("lost in the middle")
[Latest Message] ← Highest attention
```

**Lost in the Middle Problem**: Models pay less attention to information in the middle of long contexts. Place critical information at the beginning or end.

## Context Optimization Strategies

### 1. Context Ordering

#### Recency Bias
Place most relevant information near the end:
```
[Background info]
[Older messages]
[Recent messages]
[Most recent/relevant] ← Model focuses here
[User query]
```

#### Inverted Pyramid
Most important first, details later:
```
[Key facts and decisions]
[Supporting details]
[Historical context]
[User query]
```

#### Sandwich Strategy
Critical info at both ends:
```
[Key context]
[Supporting information]
[Reiterate key context]
[User query]
```

### 2. Context Compression

#### Summarization
Compress older context to preserve space:

```python
def compress_old_messages(messages, max_tokens=4000):
    """Summarize old messages to stay within token budget"""
    recent = messages[-5:]  # Keep last 5 messages
    old = messages[:-5]     # Summarize older ones
    
    if count_tokens(old) > max_tokens:
        summary = llm.summarize(old, max_length=500)
        return [summary] + recent
    
    return messages
```

#### Selective Pruning
Remove less relevant information:

```python
def prune_context(context_items, relevance_threshold=0.7):
    """Keep only highly relevant context"""
    scored_items = [
        (item, calculate_relevance(item, current_query))
        for item in context_items
    ]
    
    return [
        item for item, score in scored_items
        if score >= relevance_threshold
    ]
```

#### Hierarchical Context
Provide summaries with details on-demand:

```
Level 1: [High-level summary]
Level 2: [Key points]
Level 3: [Detailed information] ← Only if needed
```

### 3. Context Structuring

#### Use Delimiters
Clear boundaries help models parse context:

```xml
<document>
  <title>Product Documentation</title>
  <content>...</content>
</document>

<user_query>
How do I reset my password?
</user_query>
```

```markdown
### Document 1: User Manual
[content]

### Document 2: FAQ
[content]

---
User Question: [query]
```

#### Structured Formats
JSON, YAML, or tables for clarity:

```json
{
  "user_profile": {
    "name": "Alice",
    "preferences": ["technical", "concise"],
    "history": ["topic1", "topic2"]
  },
  "current_query": "Explain RAG systems"
}
```

#### Metadata and Labels
Add context about the context:

```markdown
[Source: Official Documentation, Last Updated: 2024-12-15]
Content: ...

[Source: User Manual, Reliability: High]
Content: ...

[Source: Forum Discussion, Reliability: Medium]
Content: ...
```

### 4. Context Filtering

#### Relevance Scoring
Only include relevant information:

```python
def filter_by_relevance(query, documents, top_k=5):
    """Retrieve and rank by relevance"""
    embeddings = embed(documents)
    query_embedding = embed(query)
    
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities.argsort()[-top_k:]
    
    return [documents[i] for i in top_indices]
```

#### Recency Filtering
Prioritize recent information:

```python
def filter_by_recency(items, days=30):
    """Keep only recent items"""
    cutoff = datetime.now() - timedelta(days=days)
    return [item for item in items if item.timestamp > cutoff]
```

#### Access Control
Only include information user has access to:

```python
def filter_by_access(documents, user):
    """Security-aware context filtering"""
    return [
        doc for doc in documents
        if user.can_access(doc)
    ]
```

## Context Window Management

### Sliding Window

Keep recent messages, drop oldest:

```python
class SlidingWindowMemory:
    def __init__(self, max_messages=10):
        self.messages = []
        self.max_messages = max_messages
    
    def add_message(self, message):
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)  # Remove oldest
    
    def get_context(self):
        return self.messages
```

### Token-Based Limiting

Track actual token count:

```python
def build_context(messages, max_tokens=4000):
    """Build context staying within token limit"""
    context = []
    total_tokens = 0
    
    # Start from most recent
    for message in reversed(messages):
        msg_tokens = count_tokens(message)
        if total_tokens + msg_tokens <= max_tokens:
            context.insert(0, message)
            total_tokens += msg_tokens
        else:
            break
    
    return context
```

### Hybrid Approach

Combine multiple strategies:

```python
def build_hybrid_context(messages, max_tokens=4000):
    """
    1. Keep system message always
    2. Keep last 5 messages always
    3. Summarize middle messages
    4. Fill remaining space with relevant history
    """
    system = messages[0]
    recent = messages[-5:]
    middle = messages[1:-5]
    
    # Essential parts
    context = [system] + recent
    used_tokens = count_tokens(context)
    
    # Summarize middle if needed
    if middle:
        summary = llm.summarize(middle, 
                                max_tokens=min(500, max_tokens - used_tokens))
        context.insert(1, summary)
        used_tokens += count_tokens(summary)
    
    return context
```

## Advanced Techniques

### Context Caching

Some providers (Anthropic Claude) support caching frequently-used context:

```python
# First request: pay for full context
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": large_system_context,  # e.g., 50K tokens
            "cache_control": {"type": "ephemeral"}  # Cache this
        }
    ],
    messages=[{"role": "user", "content": query}]
)

# Subsequent requests: cached context is free
# Saves 90% on input token costs for repeated context
```

**Benefits**:
- Significant cost savings (cache hits are ~90% cheaper)
- Lower latency (cached tokens skip encoding)

**Use Cases**:
- Large system prompts
- Document Q&A with same documents
- Chatbots with stable background info

### Incremental Context Loading

Load context progressively as needed:

```python
async def answer_with_incremental_context(query):
    """Load more context only if needed"""
    
    # Start with minimal context
    context = get_minimal_context(query)
    response = await llm.generate(query, context)
    
    # Check if model needs more info
    if response.confidence < 0.7 or "I need more information" in response:
        # Load additional context
        additional_context = get_additional_context(query, response)
        context.extend(additional_context)
        response = await llm.generate(query, context)
    
    return response
```

### Context Segmentation

Break context into logical segments:

```python
context_template = """
# Task Context
{task_description}

# User Information
{user_profile}

# Historical Context
{conversation_history}

# Reference Materials
{retrieved_documents}

# Current Query
{query}

# Instructions
{instructions}
"""
```

Each segment can be independently managed, pruned, or updated.

### Query-Specific Context

Tailor context to query type:

```python
def get_context_for_query(query):
    query_type = classify_query(query)
    
    if query_type == "factual":
        return retrieve_factual_documents(query)
    elif query_type == "procedural":
        return retrieve_how_to_guides(query)
    elif query_type == "comparative":
        return retrieve_comparison_data(query)
    else:
        return retrieve_general_context(query)
```

## Context Engineering for RAG

### Chunk Context

Include metadata with each chunk:

```python
chunk_template = """
Document: {document_title}
Section: {section_name}
Page: {page_number}
Last Updated: {timestamp}

{chunk_content}
"""
```

### Cross-References

Add links between related chunks:

```markdown
[Current Chunk Content]

Related Information:
- See also: [Chunk ID 123] for prerequisites
- For advanced usage: [Chunk ID 456]
- Contrast with: [Chunk ID 789]
```

### Reranking Context

Order retrieved chunks strategically:

```python
def optimize_chunk_order(chunks, query):
    """
    Place most relevant chunk last (recency bias)
    Place second-most relevant first
    Put rest in middle
    """
    scored = [(chunk, score_relevance(chunk, query)) for chunk in chunks]
    sorted_chunks = sorted(scored, key=lambda x: x[1], reverse=True)
    
    if len(sorted_chunks) > 2:
        # Rearrange for better attention
        optimized = [
            sorted_chunks[1],  # Second best first
            *sorted_chunks[2:-1],  # Rest in middle
            sorted_chunks[0]  # Best last
        ]
        return [chunk for chunk, score in optimized]
    
    return chunks
```

## Context for Multi-Turn Conversations

### Conversation Summarization

Compress old turns while preserving key information:

```python
def manage_conversation_context(messages):
    """
    Keep: System message, last 5 turns
    Summarize: Everything else
    """
    if len(messages) <= 6:  # System + 5 turns
        return messages
    
    system = messages[0]
    recent = messages[-5:]
    to_summarize = messages[1:-5]
    
    summary = {
        "role": "system",
        "content": f"Previous conversation summary: {llm.summarize(to_summarize)}"
    }
    
    return [system, summary] + recent
```

### Key Information Extraction

Extract and persist important facts:

```python
def extract_key_facts(conversation):
    """Extract facts to preserve across turns"""
    prompt = f"""
    From this conversation, extract key facts that should be remembered:
    - User preferences
    - Stated goals
    - Important decisions
    - Agreed-upon constraints
    
    Conversation: {conversation}
    """
    
    facts = llm.extract(prompt)
    return facts  # Store these separately, include in future context
```

### Context Reset

Sometimes full reset is best:

```python
def should_reset_context(conversation):
    """Detect when to start fresh"""
    signals = {
        "topic_change": detect_topic_shift(conversation),
        "too_long": len(conversation) > 50,
        "confusion": detect_confusion_markers(conversation),
        "user_request": "start over" in conversation[-1]["content"].lower()
    }
    
    return any(signals.values())
```

## Evaluation Metrics

### Context Utilization

Are models actually using the provided context?

```python
def measure_context_utilization(context, response):
    """Check if response uses context"""
    context_terms = extract_key_terms(context)
    response_terms = extract_key_terms(response)
    
    overlap = len(context_terms & response_terms)
    utilization = overlap / len(context_terms)
    
    return utilization
```

### Context Efficiency

How much context is needed for good results?

```
efficiency = quality_score / context_tokens_used
```

Lower token count with same quality = better efficiency.

### Context Relevance

Is the provided context actually relevant?

```python
def measure_context_relevance(query, context, response):
    """
    Use LLM-as-judge to evaluate relevance
    """
    prompt = f"""
    Query: {query}
    Context provided: {context}
    Response generated: {response}
    
    Rate context relevance (0-1):
    - Did the context help answer the query?
    - Was any context unused/irrelevant?
    - Was important information missing?
    """
    
    score = judge_llm.score(prompt)
    return score
```

## Common Pitfalls

### 1. Information Overload

**Problem**: Providing too much context dilutes attention

**Solution**: Be selective, prioritize quality over quantity

### 2. Outdated Context

**Problem**: Including stale information

**Solution**: Filter by recency, include timestamps, refresh regularly

### 3. Conflicting Context

**Problem**: Contradictory information confuses model

**Solution**: 
- Deduplicate context
- Mark source reliability
- Let model know how to handle conflicts

```
If sources conflict, prefer:
1. Official documentation
2. Recent sources over old
3. Primary sources over secondary
```

### 4. Poor Organization

**Problem**: Unstructured context dump

**Solution**: Use clear structure, delimiters, hierarchy

### 5. Security Leaks

**Problem**: Including sensitive info user shouldn't see

**Solution**: 
- Access control filtering
- PII detection and removal
- Audit context before sending

## Production Best Practices

### Monitoring

Track context-related metrics:

```python
metrics = {
    "avg_context_length": [],
    "context_utilization": [],
    "cache_hit_rate": [],
    "cost_per_query": [],
    "latency": []
}

def log_context_metrics(context, response):
    metrics["avg_context_length"].append(len(context))
    metrics["context_utilization"].append(
        calculate_utilization(context, response)
    )
    # ... log other metrics
```

### A/B Testing

Test different context strategies:

```python
strategies = [
    "recency_bias",
    "inverted_pyramid", 
    "sandwich",
    "summarization"
]

for user_query in test_queries:
    strategy = random.choice(strategies)
    context = build_context(user_query, strategy=strategy)
    response = llm.generate(user_query, context)
    
    log_experiment(strategy, user_query, response, user_feedback)
```

### Dynamic Context Sizing

Adjust context based on query complexity:

```python
def determine_context_size(query):
    """Allocate context budget based on query needs"""
    if is_simple_query(query):
        return 1000  # tokens
    elif is_complex_query(query):
        return 8000  # tokens
    else:
        return 4000  # default
```

### Cost Optimization

```python
# Use caching for repeated context
if context_hash in cache:
    context = cache[context_hash]  # Cached, cheaper
else:
    context = build_context(query)
    cache[context_hash] = context

# Choose model based on context needs
if context_length < 4000:
    model = "gpt-3.5-turbo"  # Cheaper
else:
    model = "gpt-4-turbo"  # Necessary for long context
```

## Tools and Libraries

### Token Counting

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
```

### Context Management

**LangChain**:
```python
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000
)
```

**LlamaIndex**:
```python
from llama_index import ServiceContext

service_context = ServiceContext.from_defaults(
    chunk_size=512,
    chunk_overlap=50
)
```

### Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def compress_context(text, max_length=150):
    summary = summarizer(text, max_length=max_length, min_length=50)
    return summary[0]['summary_text']
```

## Recent Developments (2024-2026)

- **Million-token contexts**: Gemini 1.5 Pro supports 1M+ tokens
- **Context caching**: Anthropic Claude offers 90% cost reduction on cached tokens
- **Attention improvements**: Models better at using middle context
- **Structured outputs**: Better support for JSON, XML in context
- **Context-aware fine-tuning**: Training models to use context more effectively

## Key Takeaways

1. Context engineering is distinct from prompt engineering—focus on information, not instructions
2. Not all context receives equal attention ("lost in the middle")
3. Strategic ordering: critical info at start or end
4. Compression and filtering are essential for long conversations
5. Structure and delimiters improve context parsing
6. Context caching can dramatically reduce costs
7. Monitor context utilization and efficiency
8. Balance context completeness with token costs and latency

## Related Topics

- [RAG](rag.md) - Retrieval-augmented context for QA
- [Agents](agents.md) - Context management for autonomous agents
- [Memory Systems](memory-systems.md) - Long-term context persistence
- [Prompting Techniques](../llms/notes/prompting-techniques.md) - Instruction design
