# Retrieval-Augmented Generation (RAG)

## Overview

RAG combines large language models with external knowledge retrieval to generate more accurate, up-to-date, and factually grounded responses. Instead of relying solely on the model's parametric knowledge (what it learned during training), RAG retrieves relevant information from external sources at inference time.

**Key Motivation**: LLMs have limitations including:
- Knowledge cutoff dates
- Hallucination (generating plausible but incorrect information)
- Inability to cite sources
- No access to private/domain-specific data

## Core Architecture

```
User Query → Retrieval → Context Assembly → LLM Generation → Response
```

### Components

1. **Document Processing Pipeline**
   - Ingestion: Load documents from various sources
   - Chunking: Split documents into manageable pieces
   - Embedding: Convert chunks to vector representations
   - Storage: Store embeddings in vector database

2. **Retrieval Pipeline**
   - Query embedding: Convert user query to vector
   - Search: Find most similar document chunks
   - Reranking (optional): Reorder results for better relevance
   - Context assembly: Format retrieved chunks for LLM

3. **Generation Pipeline**
   - Prompt construction: Combine query + retrieved context
   - LLM inference: Generate response using augmented context
   - Post-processing: Format, cite sources, filter

## Document Chunking Strategies

### Fixed-Size Chunking
```python
# Simple approach: split by token/character count
chunk_size = 512  # tokens
overlap = 50      # tokens for continuity
```

**Pros**: Simple, predictable chunk sizes  
**Cons**: May split semantic units awkwardly

### Semantic Chunking
- Split by sentences/paragraphs
- Use NLP to identify topic boundaries
- Preserve semantic coherence

**Pros**: Better semantic integrity  
**Cons**: Variable chunk sizes, more complex

### Hierarchical Chunking
```
Document → Sections → Paragraphs → Sentences
```

Store multiple granularities:
- Coarse chunks for overview
- Fine chunks for specific details

### Considerations
- **Chunk size vs. retrieval**: Larger chunks provide more context but may dilute relevance
- **Overlap**: Prevents information loss at boundaries (typically 10-20% of chunk size)
- **Metadata**: Store source, page numbers, section headers for better filtering and citation

## Embedding Models

### Dense Embeddings
Convert text to fixed-size dense vectors (e.g., 768 or 1536 dimensions).

**Popular Models**:
- `text-embedding-ada-002` (OpenAI) - 1536 dim
- `all-MiniLM-L6-v2` (Sentence Transformers) - 384 dim
- `e5-large-v2` (Microsoft) - 1024 dim
- `bge-large-en-v1.5` (BAAI) - 1024 dim

**Key Considerations**:
- **Domain adaptation**: General vs. domain-specific embeddings
- **Dimensionality**: Higher dimensions capture more nuance but increase storage/compute
- **Multi-lingual**: Some models support multiple languages

### Sparse Embeddings (BM25)
Traditional keyword-based retrieval using term frequency-inverse document frequency.

**Characteristics**:
- Exact keyword matching
- No semantic understanding
- Computationally efficient
- Works well for technical terms, IDs, codes

### Hybrid Retrieval
Combine dense and sparse approaches:
```python
# Weighted combination
score = α * dense_score + (1-α) * sparse_score
```

**Benefits**: 
- Dense captures semantics
- Sparse captures exact matches
- Better overall recall

## Vector Databases

Store and efficiently search high-dimensional embeddings.

### Popular Options

| Database | Type | Best For |
|----------|------|----------|
| Pinecone | Managed | Production, scale, low latency |
| Weaviate | Self-hosted/Cloud | Advanced filtering, GraphQL |
| Qdrant | Self-hosted/Cloud | High performance, Rust-based |
| Chroma | Embedded | Development, simple use cases |
| FAISS | Library | Research, prototyping |
| pgvector | PostgreSQL ext | Existing PostgreSQL deployments |

### Key Features to Consider
- **Index types**: HNSW, IVF, Product Quantization
- **Filtering**: Metadata filtering (date ranges, tags, access control)
- **Scalability**: Horizontal scaling, sharding
- **Hybrid search**: Support for both dense and sparse vectors
- **CRUD operations**: Update/delete support for dynamic data

## Retrieval Strategies

### Similarity Search
Basic approach: retrieve top-k most similar chunks.

```python
# Cosine similarity
query_embedding = embed(query)
results = vector_db.search(query_embedding, top_k=5)
```

### MMR (Maximal Marginal Relevance)
Balance relevance and diversity to avoid redundant results.

```
MMR = λ * Similarity(query, doc) - (1-λ) * max(Similarity(doc, selected_docs))
```

**Use case**: When you want diverse perspectives, not just similar chunks

### Parent-Child Retrieval
- Retrieve small, specific chunks
- Return larger parent chunks for context

**Example**: Retrieve sentence, return full paragraph

### Multi-Query Retrieval
Generate multiple query variations to improve recall.

```python
queries = [
    original_query,
    "What is X?",
    "How does X work?",
    "Examples of X"
]
all_results = [retrieve(q) for q in queries]
merged_results = deduplicate_and_rank(all_results)
```

### Hypothetical Document Embeddings (HyDE)
1. Generate a hypothetical answer to the query
2. Embed the hypothetical answer
3. Search using that embedding

**Intuition**: What the answer looks like may be more similar to actual documents than the question

## Reranking

After initial retrieval, reorder results for better relevance.

### Cross-Encoder Reranking
Use a more powerful but slower model to score query-document pairs.

```python
# Initial retrieval: fast but less accurate
candidates = vector_db.search(query_embedding, top_k=100)

# Reranking: slower but more accurate
scores = cross_encoder.rank(query, candidates)
final_results = candidates[scores.argsort()[:10]]
```

**Popular Models**:
- `ms-marco-MiniLM-L-12-v2` (Sentence Transformers)
- Cohere Rerank API
- `bge-reranker-large` (BAAI)

### Benefits
- Bi-encoders (embeddings): Fast but less accurate
- Cross-encoders (reranking): Slow but more accurate
- Two-stage approach balances speed and quality

## Advanced RAG Techniques

### Query Transformation

**Query Expansion**: Add related terms or context
```python
expanded_query = f"{query} including {related_terms}"
```

**Query Decomposition**: Break complex queries into sub-queries
```python
# "Compare X and Y"
sub_queries = ["What is X?", "What is Y?", "Differences between X and Y"]
```

**Step-back Prompting**: Ask more general questions first
```python
# Original: "What's the boiling point of water at 5000m elevation?"
# Step-back: "How does elevation affect boiling point?"
```

### Retrieval-Interleaved Generation
Instead of retrieve-once-then-generate:
```
1. Initial retrieval
2. Generate partial response
3. If needed, retrieve more context
4. Continue generation
5. Repeat until complete
```

**Use case**: Long-form answers, multi-hop reasoning

### Agentic RAG
Give the LLM tools to control the retrieval process:
- Decide when to retrieve
- Formulate retrieval queries
- Determine if more information is needed

### Self-RAG (Self-Reflective RAG)
Model generates reflection tokens to decide:
- Does the response need retrieval?
- Is the retrieved content relevant?
- Is the generated response supported by evidence?

## Prompt Engineering for RAG

### Basic Template
```
Context:
{retrieved_chunk_1}

{retrieved_chunk_2}

{retrieved_chunk_3}

Question: {user_query}

Based on the context above, provide a detailed answer.
```

### Best Practices

1. **Context Ordering**: 
   - Most relevant first (or last for recency bias)
   - Consider "lost in the middle" phenomenon

2. **Explicit Instructions**:
   ```
   Answer the question using ONLY the provided context.
   If the context doesn't contain the answer, say "I don't have enough information."
   Cite specific passages when making claims.
   ```

3. **Structured Output**:
   ```
   Provide your answer in this format:
   Answer: [Your response]
   Sources: [List of source documents used]
   Confidence: [High/Medium/Low]
   ```

4. **Citation Prompting**:
   ```
   When using information from the context, cite the source using [1], [2], etc.
   ```

## Evaluation Metrics

### Retrieval Quality

**Recall@k**: Fraction of relevant documents in top-k results
```python
recall_at_k = (relevant_docs_in_top_k) / (total_relevant_docs)
```

**Precision@k**: Fraction of retrieved documents that are relevant
```python
precision_at_k = (relevant_docs_in_top_k) / k
```

**MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant document
```python
MRR = mean([1/rank_of_first_relevant for query in queries])
```

**NDCG (Normalized Discounted Cumulative Gain)**: Considers ranking quality with graded relevance

### Generation Quality

**Faithfulness**: Is the response supported by retrieved context?
```python
# Use LLM-as-judge
faithfulness_score = judge_llm.evaluate(
    "Does the answer contradict or go beyond the provided context?"
)
```

**Answer Relevance**: Does it actually answer the question?
**Context Relevance**: Is the retrieved context relevant to the query?

### End-to-End Metrics

**RAGAS Framework**:
- Context Precision: Relevant chunks in top positions
- Context Recall: All relevant info was retrieved
- Faithfulness: Answer is grounded in context
- Answer Relevance: Answer addresses the question

**Human Evaluation**: Still the gold standard
- Correctness
- Completeness
- Clarity
- Citation quality

## Common Challenges and Solutions

### Challenge: Retrieval Failures

**Problem**: Relevant documents exist but aren't retrieved

**Solutions**:
- Improve embedding model (fine-tune on domain data)
- Use hybrid search (dense + sparse)
- Query expansion/reformulation
- Increase top-k
- Check chunking strategy

### Challenge: Context Window Limitations

**Problem**: Too many relevant chunks to fit in context

**Solutions**:
- Better reranking to get most relevant chunks
- Summarize retrieved content before feeding to LLM
- Use hierarchical retrieval (overview chunks)
- Retrieval-interleaved generation

### Challenge: Hallucination Despite RAG

**Problem**: Model still generates unsupported claims

**Solutions**:
- Stronger system prompts ("ONLY use provided context")
- Post-generation verification
- Structured output formats
- Fine-tune model for faithfulness
- Attribution/citation enforcement

### Challenge: Cold Start (Empty Vector DB)

**Problem**: No documents to retrieve from initially

**Solutions**:
- Hybrid approach: RAG + pre-trained knowledge
- Graceful degradation
- Synthetic data generation to bootstrap

### Challenge: Outdated Information

**Problem**: Vector DB contains stale data

**Solutions**:
- Incremental updates (add/remove/update chunks)
- Timestamp metadata for filtering
- Periodic full re-indexing
- Cache invalidation strategies

## RAG vs. Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Knowledge Update** | Easy (update DB) | Hard (retrain model) |
| **Cost** | Lower (inference cost) | Higher (training cost) |
| **Factuality** | Higher (grounded in sources) | Lower (may hallucinate) |
| **Latency** | Higher (retrieval overhead) | Lower (direct inference) |
| **Domain Adaptation** | Limited | Strong |
| **Citations** | Natural (source docs) | Not possible |

**When to use RAG**: Dynamic knowledge, factual QA, need for citations  
**When to use Fine-Tuning**: Style/tone adaptation, specific task formats  
**Best of Both**: Fine-tune model for RAG tasks specifically

## Implementation Frameworks

### LangChain
```python
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

vectorstore = Pinecone.from_documents(docs, embeddings)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)
```

### LlamaIndex
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
```

### Haystack
```python
from haystack.document_stores import PineconeDocumentStore
from haystack.nodes import EmbeddingRetriever

document_store = PineconeDocumentStore(...)
retriever = EmbeddingRetriever(document_store=document_store)
```

## Production Considerations

### Monitoring
- Retrieval latency
- Generation latency
- Retrieval precision/recall
- User feedback signals
- Cost per query

### Optimization
- **Caching**: Cache frequent queries
- **Batching**: Batch embed operations
- **Indexing**: Optimize vector index (HNSW, IVF)
- **Compression**: Quantization for smaller embeddings

### Security
- Access control on document level
- PII filtering
- Input sanitization
- Output filtering

### Scalability
- Horizontal scaling of vector DB
- Load balancing
- Asynchronous processing
- Rate limiting

## Recent Developments (2024-2026)

- **Long-context models**: 100K+ token windows reduce RAG necessity for some use cases
- **Multimodal RAG**: Retrieve images, tables, code alongside text
- **Graph RAG**: Use knowledge graphs for structured retrieval
- **Corrective RAG**: Models that self-correct using web search
- **Adaptive RAG**: Dynamically choose retrieval strategy based on query type

## Key Takeaways

1. RAG is essential for factual accuracy and up-to-date information
2. Chunking strategy significantly impacts retrieval quality
3. Hybrid search (dense + sparse) often outperforms either alone
4. Reranking improves precision at the cost of latency
5. Prompt engineering is crucial for getting models to respect context
6. Evaluation should cover both retrieval and generation quality
7. Production RAG requires monitoring, caching, and security controls

## Related Topics

- [Agent-to-Agent Communication](agent-to-agent.md) - Multi-agent RAG systems
- [Tool Use](tool-use.md) - RAG as a tool for agents
- [Context Engineering](context-engineering.md) - Optimizing RAG prompts
- [Memory Systems](memory-systems.md) - RAG vs. episodic memory
