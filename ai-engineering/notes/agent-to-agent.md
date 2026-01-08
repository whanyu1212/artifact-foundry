# Agent-to-Agent Communication (Multi-Agent Systems)

## Overview

**Multi-agent systems** involve multiple AI agents working together to solve complex problems. Instead of a single agent handling everything, specialized agents collaborate, each focusing on specific domains or tasks.

**Key Advantages**:
- **Specialization**: Each agent excels in a specific domain
- **Parallelism**: Agents can work concurrently
- **Modularity**: Easier to develop, test, and maintain
- **Robustness**: Failure of one agent doesn't crash the system
- **Scalability**: Add more agents for more capabilities

**Challenges**:
- **Coordination**: Ensuring agents work together effectively
- **Conflict Resolution**: Handling disagreements between agents
- **Communication Overhead**: Message passing costs
- **Consistency**: Maintaining shared state across agents
- **Debugging**: Harder to trace multi-agent behavior

## Multi-Agent Architectures

### Hierarchical (Supervisor)

One agent coordinates others:

```
       ┌─────────────┐
       │  Supervisor │ (Orchestrates work)
       │    Agent    │
       └──────┬──────┘
              │
      ┌───────┼───────┐
      │       │       │
  ┌───▼──┐ ┌──▼──┐ ┌──▼───┐
  │Agent │ │Agent│ │Agent │
  │  1   │ │  2  │ │  3   │
  └──────┘ └─────┘ └──────┘
```

**Supervisor** decides:
- Which agent to invoke
- What information to share
- When task is complete

**Worker agents** focus on:
- Specialized tasks
- Executing instructions
- Reporting results

**Example**:
```python
async def supervisor_agent(task):
    """Orchestrate specialist agents"""
    
    # Analyze task
    plan = analyze_task(task)
    
    # Delegate to specialists
    if plan.needs_research:
        research = await research_agent.execute(task)
    
    if plan.needs_code:
        code = await coding_agent.execute(task, research)
    
    if plan.needs_testing:
        tests = await testing_agent.execute(code)
    
    # Aggregate results
    return combine_results(research, code, tests)
```

### Peer-to-Peer (Collaborative)

Agents communicate directly without a central coordinator:

```
   ┌──────┐      ┌──────┐
   │Agent │◄────►│Agent │
   │  1   │      │  2   │
   └───┬──┘      └──┬───┘
       │            │
       │    ┌──────┐│
       └───►│Agent ││
            │  3   ││
            └──────┘│
                    │
```

**Characteristics**:
- Agents negotiate and coordinate directly
- No single point of failure
- More flexible but harder to control

**Example**:
```python
class PeerAgent:
    def __init__(self, name, peers):
        self.name = name
        self.peers = peers
    
    async def handle_request(self, request):
        # Can I handle this?
        if self.can_handle(request):
            return await self.execute(request)
        
        # Ask peers for help
        for peer in self.peers:
            if await peer.can_handle(request):
                return await peer.execute(request)
        
        raise ValueError("No agent can handle this request")
```

### Sequential Pipeline

Agents process work in stages:

```
Input → Agent 1 → Agent 2 → Agent 3 → Output
```

**Example**: Document processing
```
Text → Extraction Agent → Classification Agent → Summarization Agent → Summary
```

**Implementation**:
```python
async def pipeline(input_data):
    """Process through sequential agents"""
    
    # Stage 1: Extract
    extracted = await extractor_agent.process(input_data)
    
    # Stage 2: Classify
    classified = await classifier_agent.process(extracted)
    
    # Stage 3: Summarize
    summary = await summarizer_agent.process(classified)
    
    return summary
```

### Debate/Consensus

Multiple agents propose solutions, debate, and reach consensus:

```
        ┌──────────┐
        │ Problem  │
        └────┬─────┘
             │
    ┌────────┼────────┐
    │        │        │
┌───▼──┐ ┌───▼──┐ ┌───▼──┐
│Agent │ │Agent │ │Agent │
│  1   │ │  2   │ │  3   │
└───┬──┘ └───┬──┘ └───┬──┘
    │        │        │
    └────────┼────────┘
             │
        ┌────▼─────┐
        │  Debate  │
        └────┬─────┘
             │
        ┌────▼─────┐
        │Consensus │
        └──────────┘
```

**Example**:
```python
async def debate_and_decide(problem, agents, rounds=3):
    """Agents debate to reach better solution"""
    
    solutions = []
    
    # Each agent proposes solution
    for agent in agents:
        solution = await agent.propose(problem)
        solutions.append(solution)
    
    # Debate rounds
    for round in range(rounds):
        critiques = []
        for i, agent in enumerate(agents):
            # Critique others' solutions
            other_solutions = solutions[:i] + solutions[i+1:]
            critique = await agent.critique(other_solutions)
            critiques.append(critique)
        
        # Revise solutions based on critiques
        solutions = [
            await agent.revise(solutions[i], critiques)
            for i, agent in enumerate(agents)
        ]
    
    # Vote or aggregate to final decision
    final_solution = await vote(solutions)
    return final_solution
```

### Graph-Based (LangGraph)

Agents as nodes in a state graph with conditional transitions:

```python
from langgraph.graph import StateGraph

class AgentState:
    messages: List[Message]
    next_agent: str
    task_complete: bool

graph = StateGraph(AgentState)

# Add agents as nodes
graph.add_node("researcher", research_agent)
graph.add_node("coder", coding_agent)
graph.add_node("reviewer", review_agent)

# Define transitions
graph.add_edge("researcher", "coder")
graph.add_conditional_edges(
    "coder",
    lambda state: "reviewer" if state.has_code else "researcher"
)
graph.add_conditional_edges(
    "reviewer",
    lambda state: "end" if state.approved else "coder"
)

# Compile and run
app = graph.compile()
result = app.invoke(initial_state)
```

## Communication Patterns

### Message Passing

Agents send messages to each other:

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class Message:
    sender: str
    recipient: str
    content: Any
    message_type: str  # "request", "response", "notification"
    timestamp: float

class MessageBus:
    """Central message broker"""
    
    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.Queue()
    
    def register(self, agent_id: str, agent):
        self.agents[agent_id] = agent
    
    async def send(self, message: Message):
        await self.message_queue.put(message)
    
    async def dispatch(self):
        """Deliver messages to recipients"""
        while True:
            message = await self.message_queue.get()
            recipient = self.agents.get(message.recipient)
            if recipient:
                await recipient.receive(message)
```

### Shared Memory

Agents read/write to common state:

```python
class SharedMemory:
    """Shared state accessible to all agents"""
    
    def __init__(self):
        self.data = {}
        self.lock = asyncio.Lock()
    
    async def read(self, key: str):
        async with self.lock:
            return self.data.get(key)
    
    async def write(self, key: str, value):
        async with self.lock:
            self.data[key] = value
    
    async def update(self, key: str, updater):
        async with self.lock:
            current = self.data.get(key)
            self.data[key] = updater(current)

# Usage
memory = SharedMemory()

async def agent_1():
    await memory.write("user_intent", "book_flight")
    
async def agent_2():
    intent = await memory.read("user_intent")
    # Use intent for decision making
```

### Blackboard System

Shared workspace where agents post and read information:

```python
class Blackboard:
    """Shared knowledge space"""
    
    def __init__(self):
        self.knowledge = {}
        self.subscribers = defaultdict(list)
    
    def post(self, key: str, value: Any, author: str):
        """Post information to blackboard"""
        self.knowledge[key] = {
            "value": value,
            "author": author,
            "timestamp": time.time()
        }
        
        # Notify subscribers
        for subscriber in self.subscribers[key]:
            subscriber.notify(key, value)
    
    def read(self, key: str):
        """Read from blackboard"""
        return self.knowledge.get(key, {}).get("value")
    
    def subscribe(self, key: str, agent):
        """Subscribe to updates"""
        self.subscribers[key].append(agent)

# Example usage
blackboard = Blackboard()

class ResearchAgent:
    async def execute(self, query):
        results = await self.research(query)
        blackboard.post("research_results", results, author="researcher")

class SummaryAgent:
    def __init__(self):
        blackboard.subscribe("research_results", self)
    
    def notify(self, key, value):
        # Automatically triggered when research is posted
        summary = self.summarize(value)
        blackboard.post("summary", summary, author="summarizer")
```

### RPC (Remote Procedure Call)

Agents call methods on each other directly:

```python
class Agent:
    def __init__(self, name):
        self.name = name
    
    async def call(self, agent, method: str, *args, **kwargs):
        """Call method on another agent"""
        return await getattr(agent, method)(*args, **kwargs)

# Usage
researcher = Agent("researcher")
coder = Agent("coder")

# Researcher calls coder's implement method
code = await researcher.call(
    coder,
    "implement",
    specification="Create a sorting function"
)
```

## Coordination Strategies

### Task Allocation

**Round-Robin**: Distribute tasks evenly
```python
def allocate_round_robin(tasks, agents):
    """Distribute tasks equally"""
    allocation = {agent: [] for agent in agents}
    
    for i, task in enumerate(tasks):
        agent = agents[i % len(agents)]
        allocation[agent].append(task)
    
    return allocation
```

**Load Balancing**: Assign to least busy agent
```python
def allocate_by_load(task, agents):
    """Assign to agent with lightest load"""
    loads = {agent: agent.get_queue_size() for agent in agents}
    return min(loads, key=loads.get)
```

**Auction-Based**: Agents bid for tasks
```python
async def allocate_by_auction(task, agents):
    """Agents bid based on capability and availability"""
    bids = {}
    
    for agent in agents:
        # Agent evaluates how well it can handle task
        bid = await agent.evaluate_task(task)
        bids[agent] = bid
    
    # Assign to highest bidder
    winner = max(bids, key=bids.get)
    return winner
```

### Conflict Resolution

**Voting**: Majority decides
```python
def resolve_by_voting(proposals, agents):
    """Each agent votes, majority wins"""
    votes = defaultdict(int)
    
    for agent in agents:
        choice = agent.vote(proposals)
        votes[choice] += 1
    
    return max(votes, key=votes.get)
```

**Weighted Voting**: Expert opinions matter more
```python
def resolve_by_weighted_voting(proposals, agents):
    """Weight votes by agent expertise"""
    scores = defaultdict(float)
    
    for agent in agents:
        choice = agent.vote(proposals)
        scores[choice] += agent.expertise_weight
    
    return max(scores, key=scores.get)
```

**Mediator**: Third-party decides
```python
async def resolve_by_mediator(conflict, agents, mediator):
    """Mediator agent makes final decision"""
    
    # Collect perspectives
    perspectives = [
        await agent.explain_position(conflict)
        for agent in agents
    ]
    
    # Mediator decides
    decision = await mediator.mediate(perspectives)
    return decision
```

### Synchronization

**Barrier Synchronization**: Wait for all agents
```python
class Barrier:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.count = 0
        self.event = asyncio.Event()
    
    async def wait(self):
        """Block until all agents reach barrier"""
        self.count += 1
        
        if self.count >= self.num_agents:
            self.event.set()
        
        await self.event.wait()

# Usage
barrier = Barrier(num_agents=3)

async def agent_task(agent_id):
    # Do work
    result = await agent_work(agent_id)
    
    # Wait for all agents to finish
    await barrier.wait()
    
    # Continue with synchronized work
    return result
```

**Coordinator Pattern**: Central agent synchronizes
```python
class Coordinator:
    def __init__(self, agents):
        self.agents = agents
        self.results = {}
    
    async def coordinate(self, task):
        """Coordinate parallel work"""
        
        # Distribute subtasks
        subtasks = self.split_task(task)
        
        # Execute in parallel
        tasks = [
            agent.execute(subtask)
            for agent, subtask in zip(self.agents, subtasks)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine results
        return self.merge_results(results)
```

## Multi-Agent Frameworks

### CrewAI

Role-based agents with tasks:

```python
from crewai import Agent, Task, Crew

# Define agents
researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert at finding and verifying information",
    tools=[search_tool, scrape_tool],
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Write engaging content",
    backstory="Skilled at creating clear, compelling narratives",
    tools=[],
    verbose=True
)

# Define tasks
research_task = Task(
    description="Research the latest trends in AI",
    agent=researcher,
    expected_output="Detailed report on AI trends"
)

writing_task = Task(
    description="Write an article based on the research",
    agent=writer,
    context=[research_task],  # Depends on research
    expected_output="Published article"
)

# Create crew and run
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

result = crew.kickoff()
```

### AutoGen (Microsoft)

Conversational multi-agent framework:

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Create agents
assistant = AssistantAgent(
    name="Assistant",
    llm_config={"model": "gpt-4"},
    system_message="You are a helpful AI assistant"
)

user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="TERMINATE",
    code_execution_config={"work_dir": "coding"}
)

# Create group chat
groupchat = GroupChat(
    agents=[user_proxy, assistant],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat, llm_config={"model": "gpt-4"})

# Start conversation
user_proxy.initiate_chat(
    manager,
    message="Create a Python script to analyze CSV data"
)
```

### LangGraph

State machine for agent orchestration:

```python
from langgraph.graph import StateGraph, END

class ResearchState:
    query: str
    research: str
    code: str
    review: str

workflow = StateGraph(ResearchState)

workflow.add_node("research", research_node)
workflow.add_node("code", code_node)
workflow.add_node("review", review_node)

workflow.add_edge("research", "code")
workflow.add_conditional_edges(
    "code",
    lambda state: "review" if state.code else "research"
)
workflow.add_conditional_edges(
    "review",
    lambda state: END if state.approved else "code"
)

workflow.set_entry_point("research")
app = workflow.compile()
```

## Real-World Use Cases

### Software Development Team

```
Product Manager Agent → Requirements
Architect Agent → Design
Coding Agent → Implementation
Testing Agent → Tests
Review Agent → Code review
DevOps Agent → Deployment
```

### Customer Support System

```
Triage Agent → Classify issue
FAQ Agent → Check knowledge base
Technical Agent → Debug technical issues
Escalation Agent → Route to human if needed
```

### Content Creation Pipeline

```
Research Agent → Gather information
Outline Agent → Create structure
Writing Agent → Draft content
Editing Agent → Polish and refine
SEO Agent → Optimize for search
Publishing Agent → Format and publish
```

### Trading System

```
Data Agent → Collect market data
Analysis Agent → Technical analysis
Sentiment Agent → News sentiment
Strategy Agent → Generate signals
Risk Agent → Evaluate risk
Execution Agent → Place trades
```

## Evaluation Metrics

### Task Success Rate
```python
success_rate = completed_tasks / total_tasks
```

### Collaboration Efficiency
```python
# Did agents work well together?
collaboration_score = (
    successful_handoffs / total_handoffs
)
```

### Communication Overhead
```python
# How much time spent communicating vs. working?
overhead = communication_time / (communication_time + work_time)
```

### Specialization Effectiveness
```python
# Are agents handling tasks in their domain?
specialization_score = (
    tasks_handled_by_expert / total_tasks
)
```

## Common Challenges and Solutions

### Challenge: Infinite Loops

**Problem**: Agents keep passing tasks back and forth

**Solutions**:
```python
class LoopDetector:
    def __init__(self, max_iterations=10):
        self.max_iterations = max_iterations
        self.iteration_count = 0
        self.state_history = []
    
    def check(self, current_state):
        self.iteration_count += 1
        
        # Hit iteration limit?
        if self.iteration_count >= self.max_iterations:
            raise Exception("Max iterations reached")
        
        # Repeating state?
        if current_state in self.state_history[-3:]:
            raise Exception("Loop detected")
        
        self.state_history.append(current_state)
```

### Challenge: Deadlocks

**Problem**: Agents waiting for each other indefinitely

**Solutions**:
```python
# Timeouts on communication
try:
    response = await asyncio.wait_for(
        agent.request(other_agent),
        timeout=30.0
    )
except asyncio.TimeoutError:
    # Handle timeout, break deadlock
    response = None

# Deadlock detection
dependency_graph = build_dependency_graph(agents)
if has_cycle(dependency_graph):
    raise Exception("Deadlock detected")
```

### Challenge: Inconsistent State

**Problem**: Agents have different views of shared state

**Solutions**:
```python
# Locking for critical sections
async with shared_memory.lock:
    value = await shared_memory.read("counter")
    await shared_memory.write("counter", value + 1)

# Versioning
class VersionedState:
    def __init__(self):
        self.data = {}
        self.version = 0
    
    def read(self):
        return self.data, self.version
    
    def write(self, new_data, expected_version):
        if self.version != expected_version:
            raise ConflictError("State was modified")
        self.data = new_data
        self.version += 1
```

### Challenge: High Coordination Cost

**Problem**: Too much time spent on coordination

**Solutions**:
- Reduce communication frequency
- Batch messages
- Use asynchronous communication
- Give agents more autonomy
- Fewer, more capable agents

## Best Practices

### 1. Clear Agent Responsibilities

```python
class Agent:
    """
    Clearly define what each agent does
    """
    role: str  # "Researcher", "Writer", etc.
    capabilities: List[str]  # ["web_search", "data_analysis"]
    constraints: List[str]  # ["read_only", "max_cost_$1"]
```

### 2. Explicit Communication Protocols

```python
# Use structured message formats
@dataclass
class AgentMessage:
    type: Literal["request", "response", "notification"]
    sender: str
    recipient: str
    content: Dict[str, Any]
    metadata: Dict[str, Any]
```

### 3. Graceful Degradation

```python
async def robust_delegation(task, agents):
    """Fallback if preferred agent fails"""
    for agent in agents:  # Ordered by preference
        try:
            return await agent.execute(task)
        except Exception as e:
            logger.warning(f"Agent {agent} failed: {e}")
            continue
    
    # All agents failed
    return await fallback_handler(task)
```

### 4. Monitoring and Observability

```python
class AgentMonitor:
    def log_interaction(self, sender, recipient, message):
        """Track all agent interactions"""
        self.interactions.append({
            "timestamp": time.time(),
            "sender": sender,
            "recipient": recipient,
            "message": message
        })
    
    def get_interaction_graph(self):
        """Visualize agent communications"""
        # Build graph showing who talks to whom
        pass
```

## Key Takeaways

1. Multi-agent systems enable specialization and parallel work
2. Architectures: hierarchical (supervisor), peer-to-peer, pipeline, debate
3. Communication: message passing, shared memory, blackboard, RPC
4. Coordination challenges: task allocation, conflict resolution, synchronization
5. Frameworks: CrewAI (role-based), AutoGen (conversational), LangGraph (state machines)
6. Key challenges: loops, deadlocks, inconsistency, coordination overhead
7. Best practices: clear roles, explicit protocols, graceful degradation, monitoring

## Related Topics

- [Agents](agents.md) - Single agent architectures and patterns
- [Tool Use](tool-use.md) - Tools for agent capabilities
- [Memory Systems](memory-systems.md) - Shared memory for multi-agent systems
- [MCP](mcp.md) - Protocol for agent-tool communication
