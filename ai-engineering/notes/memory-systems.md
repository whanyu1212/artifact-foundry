# Memory Systems for AI Agents

## Overview

**Memory systems** enable AI agents to retain and recall information across interactions, making them more coherent, personalized, and effective over time. Unlike stateless models that forget everything between requests, agents with memory can learn from experience and maintain context across sessions.

**Key Benefits**:
- **Continuity**: Remember previous conversations and decisions
- **Personalization**: Adapt to user preferences and patterns
- **Learning**: Improve from past experiences
- **Efficiency**: Avoid asking for the same information repeatedly
- **Context**: Maintain long-term understanding of goals and state

## Memory Types

### Short-Term (Working) Memory

**Definition**: Temporary storage for current conversation and active tasks

**Characteristics**:
- Limited capacity (context window constraints)
- High recency—recent information most accessible
- Volatile—cleared between sessions or when context is full

**Implementation**:
```python
class ShortTermMemory:
    """In-memory buffer for current conversation"""
    
    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages
    
    def add(self, role: str, content: str):
        """Add message to short-term memory"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Trim if exceeds limit
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
    
    def get_context(self):
        """Retrieve all recent messages"""
        return self.messages
    
    def clear(self):
        """Clear short-term memory"""
        self.messages = []
```

**Use Cases**:
- Current conversation history
- Active task state
- Temporary calculations or intermediate results

### Long-Term (Episodic) Memory

**Definition**: Persistent storage of past interactions and experiences

**Characteristics**:
- Unlimited capacity (in practice, limited by storage)
- Requires retrieval mechanism (can't fit all in context)
- Persistent across sessions
- Indexed for efficient search

**Implementation**:
```python
from datetime import datetime
import sqlite3

class LongTermMemory:
    """Persistent memory storage"""
    
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """Create memory tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                content TEXT,
                embedding BLOB,
                timestamp DATETIME,
                session_id TEXT,
                importance FLOAT,
                tags TEXT
            )
        """)
    
    def store(self, content: str, session_id: str, 
              importance: float = 0.5, tags: list = None):
        """Store memory with metadata"""
        embedding = self.embed(content)
        
        self.conn.execute("""
            INSERT INTO memories 
            (content, embedding, timestamp, session_id, importance, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            content,
            embedding,
            datetime.now(),
            session_id,
            importance,
            ",".join(tags or [])
        ))
        self.conn.commit()
    
    def retrieve(self, query: str, limit: int = 5):
        """Retrieve relevant memories"""
        query_embedding = self.embed(query)
        
        # Find similar memories using vector similarity
        memories = self.conn.execute("""
            SELECT content, timestamp, importance
            FROM memories
            ORDER BY vector_similarity(embedding, ?) DESC
            LIMIT ?
        """, (query_embedding, limit)).fetchall()
        
        return memories
```

**Use Cases**:
- Past conversation summaries
- User preferences and history
- Learned facts and relationships
- Previous task outcomes

### Semantic Memory

**Definition**: Structured knowledge and facts independent of specific episodes

**Characteristics**:
- Organized by concepts and relationships
- Decontextualized facts
- Often represented as knowledge graphs or vector embeddings
- Updated through learning

**Implementation**:
```python
class SemanticMemory:
    """Structured knowledge storage"""
    
    def __init__(self):
        self.facts = {}  # key-value facts
        self.relationships = []  # (subject, relation, object) triples
    
    def add_fact(self, key: str, value: Any):
        """Store a fact"""
        self.facts[key] = {
            "value": value,
            "last_updated": datetime.now(),
            "confidence": 1.0
        }
    
    def add_relationship(self, subject: str, relation: str, obj: str):
        """Store a relationship"""
        self.relationships.append({
            "subject": subject,
            "relation": relation,
            "object": obj,
            "timestamp": datetime.now()
        })
    
    def get_fact(self, key: str):
        """Retrieve a fact"""
        return self.facts.get(key, {}).get("value")
    
    def query_relationships(self, subject: str = None, 
                           relation: str = None, 
                           obj: str = None):
        """Query knowledge graph"""
        results = []
        for rel in self.relationships:
            if (subject is None or rel["subject"] == subject) and \
               (relation is None or rel["relation"] == relation) and \
               (obj is None or rel["object"] == obj):
                results.append(rel)
        return results

# Example usage
memory = SemanticMemory()
memory.add_fact("user_name", "Alice")
memory.add_fact("user_role", "engineer")
memory.add_relationship("Alice", "works_on", "RAG_project")
memory.add_relationship("RAG_project", "uses", "LangChain")
```

**Use Cases**:
- User profile information
- Domain knowledge
- Entity relationships
- Learned rules and patterns

### Procedural Memory

**Definition**: Knowledge of how to perform tasks and skills

**Characteristics**:
- Often implicit/automatic
- Improved through practice
- Difficult to verbalize
- Represents learned procedures

**Implementation**:
```python
class ProceduralMemory:
    """Store learned procedures and skills"""
    
    def __init__(self):
        self.procedures = {}
        self.success_counts = defaultdict(int)
        self.failure_counts = defaultdict(int)
    
    def store_procedure(self, task: str, steps: list):
        """Store a procedure"""
        self.procedures[task] = {
            "steps": steps,
            "created": datetime.now(),
            "last_used": None,
            "success_rate": 0.0
        }
    
    def get_procedure(self, task: str):
        """Retrieve procedure for task"""
        if task in self.procedures:
            self.procedures[task]["last_used"] = datetime.now()
            return self.procedures[task]["steps"]
        return None
    
    def record_outcome(self, task: str, success: bool):
        """Update success rate based on outcomes"""
        if success:
            self.success_counts[task] += 1
        else:
            self.failure_counts[task] += 1
        
        total = self.success_counts[task] + self.failure_counts[task]
        success_rate = self.success_counts[task] / total
        
        if task in self.procedures:
            self.procedures[task]["success_rate"] = success_rate
    
    def get_best_procedure(self, task_type: str):
        """Get highest-success procedure for task type"""
        relevant = [
            (task, proc) for task, proc in self.procedures.items()
            if task_type in task
        ]
        
        if not relevant:
            return None
        
        best = max(relevant, key=lambda x: x[1]["success_rate"])
        return best[1]["steps"]
```

**Use Cases**:
- Common task workflows
- Error recovery strategies
- Optimization techniques
- Learned shortcuts

## Memory Architectures

### Unified Memory Pool

Single memory store with different access patterns:

```python
class UnifiedMemory:
    """Single memory system for all types"""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.metadata_db = MetadataDatabase()
    
    def store(self, content: str, memory_type: str, 
              importance: float = 0.5, **metadata):
        """Store any type of memory"""
        embedding = embed(content)
        
        memory_id = self.vector_db.add(
            embedding=embedding,
            metadata={
                "content": content,
                "type": memory_type,
                "importance": importance,
                "timestamp": time.time(),
                **metadata
            }
        )
        
        return memory_id
    
    def retrieve(self, query: str, memory_types: list = None,
                 limit: int = 5):
        """Retrieve memories with optional type filtering"""
        query_embedding = embed(query)
        
        filters = {}
        if memory_types:
            filters["type"] = {"$in": memory_types}
        
        results = self.vector_db.search(
            query_embedding,
            filter=filters,
            limit=limit
        )
        
        return results
```

### Hierarchical Memory

Multi-level memory with different granularities:

```python
class HierarchicalMemory:
    """Memory organized in hierarchical levels"""
    
    def __init__(self):
        self.immediate = []  # Last few messages
        self.recent = []     # Last few hours/sessions
        self.summary = []    # Compressed older memories
        self.important = []  # High-importance memories
    
    def add(self, content: str, importance: float = 0.5):
        """Add to appropriate level(s)"""
        memory = {
            "content": content,
            "timestamp": time.time(),
            "importance": importance
        }
        
        # Always add to immediate
        self.immediate.append(memory)
        
        # Trim immediate if too large
        if len(self.immediate) > 10:
            old = self.immediate.pop(0)
            self.recent.append(old)
        
        # Add high-importance to permanent storage
        if importance > 0.8:
            self.important.append(memory)
        
        # Periodically compress recent to summary
        if len(self.recent) > 100:
            self._compress_recent()
    
    def _compress_recent(self):
        """Summarize recent memories"""
        batch = self.recent[:50]
        summary = llm.summarize(batch)
        
        self.summary.append({
            "content": summary,
            "timestamp": time.time(),
            "original_count": len(batch)
        })
        
        self.recent = self.recent[50:]
    
    def get_context(self, query: str):
        """Build context from all levels"""
        context = []
        
        # Always include immediate
        context.extend(self.immediate)
        
        # Add relevant important memories
        context.extend([
            m for m in self.important
            if is_relevant(m["content"], query)
        ])
        
        # Add recent if space allows
        if len(context) < 20:
            context.extend(self.recent[-10:])
        
        return context
```

### Memory Stream (Generative Agents)

Continuous stream of observations with reflection:

```python
class MemoryStream:
    """Memory stream from 'Generative Agents' paper"""
    
    def __init__(self):
        self.stream = []
        self.reflections = []
    
    def observe(self, observation: str, importance: float):
        """Add observation to stream"""
        memory = {
            "content": observation,
            "timestamp": datetime.now(),
            "importance": importance,
            "type": "observation",
            "access_count": 0,
            "last_accessed": datetime.now()
        }
        self.stream.append(memory)
        
        # Trigger reflection if importance threshold reached
        total_importance = sum(m["importance"] for m in self.stream[-100:])
        if total_importance > 50:
            self._reflect()
    
    def _reflect(self):
        """Generate high-level insights from recent memories"""
        recent = self.stream[-100:]
        
        # Ask LLM to reflect on recent experiences
        prompt = f"""
        Based on these recent observations, what are the key insights?
        
        Observations:
        {format_memories(recent)}
        
        Provide 3-5 high-level insights or patterns.
        """
        
        insights = llm.generate(prompt)
        
        for insight in insights:
            self.reflections.append({
                "content": insight,
                "timestamp": datetime.now(),
                "importance": 0.8,  # Reflections are important
                "type": "reflection",
                "based_on": [m["content"] for m in recent]
            })
    
    def retrieve(self, query: str, k: int = 5):
        """Retrieve memories using recency, importance, and relevance"""
        scores = []
        current_time = datetime.now()
        
        for memory in self.stream + self.reflections:
            # Recency score (exponential decay)
            hours_ago = (current_time - memory["timestamp"]).total_seconds() / 3600
            recency = 0.99 ** hours_ago
            
            # Importance score
            importance = memory["importance"]
            
            # Relevance score (semantic similarity)
            relevance = cosine_similarity(
                embed(query),
                embed(memory["content"])
            )
            
            # Combined score
            score = recency + importance + relevance
            scores.append((memory, score))
        
        # Return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, score in scores[:k]]
```

## Retrieval Strategies

### Recency-Based

Retrieve most recent memories:

```python
def retrieve_recent(memories, limit=5):
    """Get most recent memories"""
    sorted_memories = sorted(
        memories,
        key=lambda m: m["timestamp"],
        reverse=True
    )
    return sorted_memories[:limit]
```

### Importance-Based

Retrieve most important memories:

```python
def retrieve_important(memories, threshold=0.7):
    """Get important memories above threshold"""
    return [
        m for m in memories
        if m.get("importance", 0) >= threshold
    ]
```

### Similarity-Based

Retrieve semantically similar memories:

```python
def retrieve_similar(query: str, memories, limit=5):
    """Get semantically similar memories"""
    query_embedding = embed(query)
    
    scores = []
    for memory in memories:
        memory_embedding = embed(memory["content"])
        similarity = cosine_similarity(query_embedding, memory_embedding)
        scores.append((memory, similarity))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [memory for memory, score in scores[:limit]]
```

### Hybrid Retrieval

Combine multiple signals:

```python
def retrieve_hybrid(query: str, memories, limit=5):
    """Combine recency, importance, and relevance"""
    query_embedding = embed(query)
    current_time = time.time()
    
    scores = []
    for memory in memories:
        # Recency (0-1, exponential decay)
        age = current_time - memory["timestamp"]
        recency = math.exp(-age / (24 * 3600))  # Decay over 1 day
        
        # Importance (0-1)
        importance = memory.get("importance", 0.5)
        
        # Relevance (0-1, cosine similarity)
        memory_embedding = embed(memory["content"])
        relevance = cosine_similarity(query_embedding, memory_embedding)
        
        # Weighted combination
        score = 0.3 * recency + 0.3 * importance + 0.4 * relevance
        scores.append((memory, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [memory for memory, score in scores[:limit]]
```

## Memory Management

### Forgetting Mechanisms

**Time-Based Decay**:
```python
def apply_decay(memories, decay_rate=0.01):
    """Reduce importance over time"""
    current_time = time.time()
    
    for memory in memories:
        age_days = (current_time - memory["timestamp"]) / (24 * 3600)
        decay = math.exp(-decay_rate * age_days)
        memory["importance"] *= decay
    
    # Remove memories below threshold
    return [m for m in memories if m["importance"] > 0.1]
```

**Access-Based Retention**:
```python
def update_access(memory):
    """Strengthen memory on access"""
    memory["access_count"] += 1
    memory["last_accessed"] = time.time()
    
    # Boost importance based on access frequency
    memory["importance"] = min(1.0, 
        memory["importance"] * (1 + 0.1 * memory["access_count"])
    )
```

**Capacity-Based Pruning**:
```python
def prune_memories(memories, max_size=10000):
    """Keep only most important memories when at capacity"""
    if len(memories) <= max_size:
        return memories
    
    # Sort by importance
    sorted_memories = sorted(
        memories,
        key=lambda m: m["importance"],
        reverse=True
    )
    
    return sorted_memories[:max_size]
```

### Memory Consolidation

Merge similar memories:

```python
def consolidate_memories(memories, similarity_threshold=0.9):
    """Merge highly similar memories"""
    clusters = []
    used = set()
    
    for i, mem1 in enumerate(memories):
        if i in used:
            continue
        
        cluster = [mem1]
        emb1 = embed(mem1["content"])
        
        for j, mem2 in enumerate(memories[i+1:], start=i+1):
            if j in used:
                continue
            
            emb2 = embed(mem2["content"])
            similarity = cosine_similarity(emb1, emb2)
            
            if similarity >= similarity_threshold:
                cluster.append(mem2)
                used.add(j)
        
        if len(cluster) > 1:
            # Merge cluster into single memory
            merged = {
                "content": llm.summarize([m["content"] for m in cluster]),
                "timestamp": max(m["timestamp"] for m in cluster),
                "importance": max(m["importance"] for m in cluster),
                "merged_from": len(cluster)
            }
            clusters.append(merged)
        else:
            clusters.append(mem1)
    
    return clusters
```

### Memory Updating

Update existing memories:

```python
class UpdatableMemory:
    """Memory that can be updated with new information"""
    
    def __init__(self):
        self.memories = {}  # key -> memory
    
    def store_or_update(self, key: str, content: str, 
                        importance: float = 0.5):
        """Store new or update existing memory"""
        if key in self.memories:
            # Update existing
            existing = self.memories[key]
            
            # Combine old and new information
            combined = llm.generate(f"""
            Old information: {existing['content']}
            New information: {content}
            
            Combine these into a single, coherent memory.
            Resolve any conflicts, keeping the most recent information.
            """)
            
            self.memories[key] = {
                "content": combined,
                "timestamp": time.time(),
                "importance": max(existing["importance"], importance),
                "version": existing.get("version", 1) + 1
            }
        else:
            # Store new
            self.memories[key] = {
                "content": content,
                "timestamp": time.time(),
                "importance": importance,
                "version": 1
            }
```

## Integration with Agents

### Memory-Augmented Agent

```python
class MemoryAgent:
    """Agent with comprehensive memory system"""
    
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
    
    async def process(self, user_input: str):
        """Process input with memory"""
        
        # Add to short-term memory
        self.short_term.add("user", user_input)
        
        # Retrieve relevant long-term memories
        relevant_memories = self.long_term.retrieve(user_input, limit=5)
        
        # Build context
        context = {
            "current_conversation": self.short_term.get_context(),
            "relevant_history": relevant_memories,
            "user_facts": self.semantic.query_facts(user_input),
            "known_procedures": self.procedural.get_procedure(user_input)
        }
        
        # Generate response
        response = await self.generate_response(user_input, context)
        
        # Update memories
        self.short_term.add("assistant", response)
        
        # Store important information to long-term
        if is_important(user_input, response):
            self.long_term.store(
                f"User: {user_input}\nAssistant: {response}",
                importance=calculate_importance(user_input, response)
            )
        
        # Extract and store facts
        facts = extract_facts(user_input, response)
        for fact in facts:
            self.semantic.add_fact(fact["key"], fact["value"])
        
        return response
```

### Memory Injection Points

```python
def build_prompt_with_memory(query: str, memories):
    """Inject memories into prompt"""
    
    prompt = f"""
# Relevant Context from Memory

{format_memories(memories["recent"])}

# Important Facts

{format_facts(memories["facts"])}

# User Query

{query}

# Instructions

Answer the query using the context and facts provided above.
Reference specific memories when relevant.
"""
    
    return prompt
```

## Evaluation Metrics

### Memory Utilization

```python
def calculate_utilization(response: str, provided_memories):
    """Check if memories were actually used"""
    used_count = sum(
        1 for mem in provided_memories
        if any(term in response.lower() 
               for term in extract_key_terms(mem["content"]))
    )
    
    return used_count / len(provided_memories) if provided_memories else 0
```

### Memory Accuracy

```python
def evaluate_memory_accuracy(memories, ground_truth):
    """Check if stored memories match reality"""
    correct = 0
    total = len(memories)
    
    for memory in memories:
        if matches_ground_truth(memory, ground_truth):
            correct += 1
    
    return correct / total if total > 0 else 0
```

### Retrieval Relevance

```python
def evaluate_retrieval(query: str, retrieved: list, relevant: list):
    """Precision and recall of memory retrieval"""
    retrieved_ids = {m["id"] for m in retrieved}
    relevant_ids = {m["id"] for m in relevant}
    
    true_positives = len(retrieved_ids & relevant_ids)
    
    precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
    recall = true_positives / len(relevant_ids) if relevant_ids else 0
    
    return {"precision": precision, "recall": recall}
```

## Production Considerations

### Scalability

```python
# Use efficient vector databases
from qdrant_client import QdrantClient

class ScalableMemory:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection = "agent_memories"
    
    async def store_batch(self, memories: list):
        """Batch insert for efficiency"""
        points = [
            {
                "id": mem["id"],
                "vector": embed(mem["content"]),
                "payload": mem
            }
            for mem in memories
        ]
        
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )
```

### Privacy and Security

```python
class SecureMemory:
    """Memory with encryption and access control"""
    
    def __init__(self, encryption_key):
        self.encryption_key = encryption_key
        self.access_log = []
    
    def store(self, content: str, user_id: str, classification: str):
        """Store with encryption"""
        encrypted = encrypt(content, self.encryption_key)
        
        memory = {
            "content": encrypted,
            "user_id": user_id,
            "classification": classification,  # "public", "private", "sensitive"
            "timestamp": time.time()
        }
        
        self.db.insert(memory)
    
    def retrieve(self, query: str, requesting_user: str):
        """Retrieve with access control"""
        memories = self.db.search(query)
        
        # Filter by access permissions
        accessible = []
        for mem in memories:
            if self.can_access(requesting_user, mem):
                decrypted = decrypt(mem["content"], self.encryption_key)
                accessible.append(decrypted)
                
                # Log access
                self.access_log.append({
                    "user": requesting_user,
                    "memory_id": mem["id"],
                    "timestamp": time.time()
                })
        
        return accessible
```

### Monitoring

```python
class MemoryMonitor:
    """Monitor memory system health"""
    
    def __init__(self):
        self.metrics = {
            "total_memories": 0,
            "storage_size_mb": 0,
            "avg_retrieval_time_ms": 0,
            "cache_hit_rate": 0
        }
    
    def log_retrieval(self, query: str, num_results: int, latency_ms: float):
        """Log retrieval metrics"""
        self.metrics["avg_retrieval_time_ms"] = (
            0.9 * self.metrics["avg_retrieval_time_ms"] + 
            0.1 * latency_ms
        )
        
        # Alert if slow
        if latency_ms > 500:
            self.alert(f"Slow retrieval: {latency_ms}ms for query: {query}")
    
    def check_health(self):
        """Check overall system health"""
        issues = []
        
        if self.metrics["storage_size_mb"] > 10000:
            issues.append("Storage size exceeds 10GB")
        
        if self.metrics["avg_retrieval_time_ms"] > 200:
            issues.append("Average retrieval time too high")
        
        return {"healthy": len(issues) == 0, "issues": issues}
```

## Best Practices

1. **Separate Concerns**: Different memory types for different purposes
2. **Efficient Retrieval**: Use vector databases, not linear search
3. **Smart Forgetting**: Implement decay and pruning mechanisms
4. **Context Window Awareness**: Don't overload with irrelevant memories
5. **User Control**: Allow users to view/edit/delete their memories
6. **Privacy**: Encrypt sensitive information, implement access controls
7. **Monitoring**: Track storage, retrieval latency, utilization
8. **Versioning**: Track changes to memories over time

## Key Takeaways

1. Memory enables agents to learn and maintain context across interactions
2. Four types: short-term (current), long-term (episodic), semantic (facts), procedural (skills)
3. Retrieval combines recency, importance, and relevance
4. Management: forgetting (decay), consolidation (merging), updating
5. Integration: inject memories into prompts, extract info to store
6. Production: scalability (vector DBs), privacy (encryption), monitoring
7. Memory quality directly impacts agent effectiveness

## Related Topics

- [Agents](agents.md) - Agents that use memory systems
- [RAG](rag.md) - Similar retrieval patterns for documents
- [Context Engineering](context-engineering.md) - Managing what fits in context
- [Agent-to-Agent Communication](agent-to-agent.md) - Shared memory in multi-agent systems
