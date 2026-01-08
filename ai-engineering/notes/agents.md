# AI Agents

## Overview

An **AI agent** is an autonomous system that uses a language model as its reasoning engine to perceive its environment, make decisions, and take actions to achieve goals. Unlike simple chatbots that respond once per turn, agents can plan multi-step solutions, use tools, and operate iteratively until a task is complete.

**Key Characteristics**:
- **Autonomy**: Can operate without human intervention for extended periods
- **Goal-directed**: Works toward specified objectives
- **Adaptive**: Adjusts approach based on feedback and results
- **Tool use**: Can interact with external systems and APIs
- **Multi-step reasoning**: Breaks down complex tasks into subtasks

## Core Architecture

### Basic Agent Loop

```
while not task_complete:
    1. Observe: Get current state and context
    2. Reason: Plan next action using LLM
    3. Act: Execute chosen action (tool call, API request, etc.)
    4. Reflect: Evaluate result and update memory
```

### Components

1. **Planning Module**: Break tasks into subtasks, create execution plans
2. **Memory**: Store conversation history, task progress, learned information
3. **Tool Interface**: Execute functions and interact with external systems
4. **Reflection/Evaluation**: Assess own performance and adjust strategy
5. **Safety Guardrails**: Ensure actions are safe and aligned with constraints

## Agent Architectures

### ReAct (Reasoning + Acting)

Interleave reasoning traces with action execution.

```
Thought: I need to find the current temperature in Tokyo
Action: search("current temperature Tokyo")
Observation: 18°C, cloudy
Thought: Now I need to convert to Fahrenheit
Action: calculate(18 * 9/5 + 32)
Observation: 64.4°F
Thought: I have the answer
Final Answer: The current temperature in Tokyo is 18°C (64.4°F), cloudy conditions.
```

**Benefits**:
- Interpretable reasoning chain
- Natural error recovery
- Human-like problem solving

**Implementation**:
```python
prompt = f"""
Answer the question using this format:
Thought: [your reasoning]
Action: [tool_name(arguments)]
Observation: [tool result]
... (repeat as needed)
Final Answer: [your answer]

Available tools: {tools}
Question: {question}
"""
```

### Plan-and-Execute

Two-stage approach: plan first, then execute.

```
1. Planning Phase:
   - Analyze task
   - Create step-by-step plan
   - Identify required tools/resources

2. Execution Phase:
   - Execute each step sequentially
   - Handle errors and replan if needed
   - Aggregate results
```

**Benefits**:
- More efficient (fewer LLM calls)
- Better for complex, multi-step tasks
- Easier to estimate costs upfront

**Limitations**:
- Less adaptive to unexpected results
- Plan may become outdated

### Tool-Using Agent

Focus on selecting and executing the right tools.

**Tool Format** (OpenAI function calling):
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]
```

**Execution Loop**:
```python
messages = [{"role": "user", "content": query}]

while True:
    response = llm.chat(messages, tools=tools)
    
    if response.finish_reason == "tool_calls":
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    else:
        return response.content
```

### LangGraph / State Machines

Model agent as a graph of states and transitions.

```python
from langgraph.graph import StateGraph

# Define states
class AgentState:
    messages: List[Message]
    next_step: str
    memory: Dict

# Define transitions
graph = StateGraph(AgentState)
graph.add_node("plan", planning_function)
graph.add_node("execute", execution_function)
graph.add_node("reflect", reflection_function)

graph.add_edge("plan", "execute")
graph.add_edge("execute", "reflect")
graph.add_conditional_edges("reflect", 
    lambda state: "plan" if not state.done else "end"
)
```

**Benefits**:
- Explicit control flow
- Easy to debug and visualize
- Composable and modular

## Planning Strategies

### Chain-of-Thought (CoT)

Encourage step-by-step reasoning.

```
Let's solve this step by step:
1. First, I need to...
2. Then, I should...
3. Finally, I can...
```

### Tree of Thoughts (ToT)

Explore multiple reasoning paths, backtrack if needed.

```
           Root Problem
          /      |      \
      Plan A  Plan B  Plan C
       / \      |      / | \
     A1 A2     B1    C1 C2 C3
```

Evaluate each branch, prune unsuccessful paths.

### Hierarchical Planning

Break tasks into subtasks recursively.

```
Task: "Plan a trip to Japan"
├── Subtask 1: Research destinations
│   ├── Find top cities
│   └── Read travel guides
├── Subtask 2: Book flights
│   ├── Compare prices
│   └── Select dates
└── Subtask 3: Arrange accommodation
    ├── Search hotels
    └── Make reservations
```

### Least-to-Most Prompting

Solve simpler subproblems first, build up to complex solution.

```
Question: What is the result of 15 * 23 + 47?

Simpler: What is 15 * 23?
Answer: 345

Simpler: What is 345 + 47?
Answer: 392

Final: 392
```

## Memory Systems

### Short-Term (Working) Memory

Maintains current conversation context and task state.

**Implementation**: 
- Conversation history buffer
- Recent tool results
- Current plan/goals

**Challenges**:
- Context window limits
- Managing relevance over long conversations

### Long-Term Memory

Persistent storage of learned information across sessions.

**Types**:
- **Semantic Memory**: Facts and knowledge
- **Episodic Memory**: Past experiences and interactions
- **Procedural Memory**: Learned skills and strategies

**Implementation Options**:
- Vector database (RAG-style retrieval)
- Key-value store (structured data)
- Graph database (relationships and entities)

### Memory Retrieval

**Recency**: Most recent information
**Relevance**: Semantically similar to current context
**Importance**: User-defined or learned significance

```python
def retrieve_memories(query, top_k=5):
    # Combine multiple signals
    recent = get_recent_memories(limit=20)
    relevant = vector_db.search(query, top_k=top_k*2)
    important = filter_by_importance(relevant, threshold=0.7)
    
    # Merge and deduplicate
    memories = merge_and_rank(recent, relevant, important)
    return memories[:top_k]
```

### Memory Management

**Summarization**: Compress old memories to save space
```python
if len(conversation_history) > max_length:
    old_messages = conversation_history[:100]
    summary = llm.summarize(old_messages)
    conversation_history = [summary] + conversation_history[100:]
```

**Forgetting**: Remove irrelevant or outdated memories
**Consolidation**: Merge related memories for efficiency

## Tool Design and Integration

### Good Tool Characteristics

1. **Clear Purpose**: Single, well-defined responsibility
2. **Robust Error Handling**: Graceful failures with helpful messages
3. **Descriptive Metadata**: Detailed descriptions and parameter specs
4. **Idempotent When Possible**: Safe to retry
5. **Fast Execution**: Minimize latency in agent loop

### Tool Description Best Practices

```python
{
    "name": "search_database",
    "description": """
    Search the customer database for records matching the query.
    
    Use this when you need to:
    - Find customer information by name, email, or ID
    - Check order history
    - Verify account details
    
    Do NOT use this for:
    - General web search (use web_search instead)
    - Product inventory (use check_inventory instead)
    """,
    "parameters": {...}
}
```

**Key elements**:
- What the tool does
- When to use it
- When NOT to use it (common mistakes)
- Examples (optional but helpful)

### Error Handling

```python
def execute_tool(tool_name, args):
    try:
        result = tools[tool_name](**args)
        return {"success": True, "result": result}
    except ValueError as e:
        return {
            "success": False, 
            "error": "Invalid parameters",
            "message": str(e),
            "suggestion": "Please check the parameter types and try again"
        }
    except TimeoutError:
        return {
            "success": False,
            "error": "Timeout",
            "suggestion": "The operation took too long. Try a more specific query."
        }
```

Return structured errors that help the agent recover.

### Tool Composition

Allow agents to chain tools:

```python
# Example: Search → Filter → Aggregate
result = aggregate(
    filter(
        search("machine learning papers"),
        year=2024
    ),
    by="citation_count"
)
```

## Reflection and Self-Evaluation

### Self-Critique

Agent evaluates its own responses before finalizing.

```
Draft Answer: [agent's initial response]

Critique: [evaluate for accuracy, completeness, relevance]
- Is this factually correct?
- Does it fully address the question?
- Are there any logical errors?
- Did I use the right tools?

Revised Answer: [improved response based on critique]
```

### Reward Modeling

Agent learns what good responses look like.

```python
def evaluate_action(action, result, goal):
    """Score how well the action helped achieve the goal"""
    relevance = calculate_relevance(result, goal)
    efficiency = 1.0 / number_of_steps
    success = check_goal_achievement(goal)
    
    reward = 0.5*relevance + 0.3*efficiency + 0.2*success
    return reward
```

Use rewards to guide future decisions (reinforcement learning).

### Error Analysis

Learn from mistakes:

```python
if task_failed:
    reflection = llm.generate(f"""
    Task: {task}
    Actions taken: {action_history}
    Result: {result}
    
    Why did this fail? What should I do differently next time?
    """)
    
    store_memory(reflection, importance="high")
```

## Multi-Agent Systems

See [Agent-to-Agent Communication](agent-to-agent.md) for detailed coverage.

**Quick Overview**:
- Multiple specialized agents collaborate
- Coordination mechanisms: shared memory, message passing, supervisor
- Benefits: specialization, parallelism, robustness
- Challenges: coordination overhead, conflict resolution

## Agent Frameworks

### LangChain

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

tools = [
    Tool(name="Search", func=search_function, description="..."),
    Tool(name="Calculator", func=calc_function, description="...")
]

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What's 15% of the GDP of Japan?"})
```

### LlamaIndex

Focus on data integration and querying.

```python
from llama_index.agent import ReActAgent
from llama_index.tools import QueryEngineTool

query_engine = index.as_query_engine()
tools = [
    QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="knowledge_base",
        description="Search the company knowledge base"
    )
]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
response = agent.chat("What's our return policy?")
```

### Semantic Kernel (Microsoft)

```csharp
var kernel = Kernel.Builder.Build();
kernel.ImportFunctions(new MyTools(), "Tools");

var planner = new SequentialPlanner(kernel);
var plan = await planner.CreatePlanAsync("Book a meeting with John next Tuesday");

var result = await plan.InvokeAsync();
```

### AutoGPT / BabyAGI

Highly autonomous agents with continuous loops.

```python
while True:
    # Get next task from task list
    task = task_manager.get_next_task()
    
    # Execute task using agent
    result = agent.execute(task)
    
    # Create new tasks based on result
    new_tasks = task_manager.create_new_tasks(result, objective)
    
    # Reprioritize task list
    task_manager.prioritize(objective)
```

**Warning**: Unbounded execution loops can be expensive and unpredictable.

## Prompt Engineering for Agents

### System Prompt Structure

```
You are an AI assistant with access to tools.

# Your Capabilities
- You can search the web
- You can perform calculations
- You can access the company database

# Your Process
1. Analyze the user's request
2. Break it down into steps
3. Use tools as needed
4. Provide a clear, concise answer

# Important Guidelines
- Always explain your reasoning
- If you're unsure, say so - don't guess
- Use tools rather than relying on memory
- Cite sources when providing facts
- If a task is impossible, explain why

# Response Format
Use this format:
Thought: [your reasoning]
Action: [tool name and arguments]
Observation: [tool result]
... (repeat as needed)
Final Answer: [your response to the user]
```

### Few-Shot Examples

```
Example 1:
User: What's the weather in Tokyo?
Thought: I need to get current weather data
Action: get_weather(location="Tokyo")
Observation: 18°C, cloudy
Final Answer: It's currently 18°C and cloudy in Tokyo.

Example 2:
User: Calculate 15% tip on $83.50
Thought: I need to calculate 83.50 * 0.15
Action: calculator(83.50 * 0.15)
Observation: 12.525
Thought: I should round to 2 decimal places
Final Answer: A 15% tip on $83.50 would be $12.53.
```

### Constraints and Guardrails

```
# Constraints
- You MUST get user confirmation before:
  - Making purchases
  - Sending emails
  - Deleting data
  
- You MUST NOT:
  - Share user personal information
  - Execute code that could harm the system
  - Bypass security controls

# Budget Limits
- Maximum 10 tool calls per request
- Maximum $0.50 in API costs per request
- Timeout after 60 seconds
```

## Evaluation and Testing

### Success Metrics

**Task Completion Rate**: Percentage of tasks successfully completed
```python
success_rate = successful_tasks / total_tasks
```

**Efficiency**: Steps/tokens used per task
```python
efficiency = total_steps / tasks_completed
```

**Tool Usage Accuracy**: Using the right tools
```python
tool_accuracy = correct_tool_calls / total_tool_calls
```

**Error Recovery**: Ability to recover from failures
```python
recovery_rate = recovered_failures / total_failures
```

### Testing Approaches

**Unit Tests**: Test individual components
```python
def test_planning():
    plan = agent.create_plan("Book a flight to Paris")
    assert "search flights" in plan.steps
    assert "compare prices" in plan.steps
```

**Integration Tests**: Test full agent loop
```python
def test_end_to_end():
    result = agent.execute("What's the capital of France?")
    assert result.success
    assert "Paris" in result.answer
```

**Benchmark Suites**:
- **HotPotQA**: Multi-hop reasoning
- **WebArena**: Web interaction tasks
- **ToolBench**: Tool use capabilities
- **GAIA**: General AI assistant tasks

### Debugging Strategies

1. **Verbose Logging**: Log every thought, action, observation
2. **Intermediate Checkpoints**: Save state at each step
3. **Replay**: Re-run failed attempts with modifications
4. **Tracing**: Track execution flow through the agent loop
5. **Human-in-the-Loop**: Manual review of agent decisions

## Common Challenges

### Challenge: Tool Selection Errors

**Problem**: Agent chooses wrong tool or misuses tools

**Solutions**:
- Better tool descriptions with examples
- Few-shot examples showing correct usage
- Reduce number of available tools (cognitive load)
- Tool usage feedback loop

### Challenge: Infinite Loops

**Problem**: Agent gets stuck repeating same actions

**Solutions**:
- Maximum iteration limits
- Track action history, detect cycles
- Reflection: "Am I making progress?"
- Circuit breakers

```python
if action in recent_actions[-3:]:
    return "You seem to be repeating the same action. Try a different approach."
```

### Challenge: High Costs

**Problem**: Many LLM calls add up quickly

**Solutions**:
- Caching: Store results of repeated queries
- Smaller models for simple decisions
- Batch operations when possible
- Budget limits and monitoring

### Challenge: Unreliable Tool Execution

**Problem**: External APIs fail or return unexpected results

**Solutions**:
- Retry logic with exponential backoff
- Fallback tools/methods
- Graceful degradation
- Clear error messages to agent

### Challenge: Context Window Overflow

**Problem**: Agent history exceeds model's context limit

**Solutions**:
- Summarization of old interactions
- Selective memory retrieval (most relevant)
- Sliding window: keep recent + important
- External memory systems

## Safety and Alignment

### Action Approval

Require confirmation for high-risk actions:

```python
if action.risk_level == "high":
    approval = request_user_approval(action)
    if not approval:
        return "Action cancelled by user"
```

### Sandboxing

Execute code/commands in isolated environments:
- Docker containers
- VMs
- Restricted file system access
- Network limitations

### Input/Output Filtering

```python
# Input sanitization
def sanitize_input(user_input):
    # Remove potential injection attacks
    # Check for malicious patterns
    # Validate against schema
    return clean_input

# Output filtering
def filter_output(agent_response):
    # Remove PII
    # Check for harmful content
    # Verify alignment with guidelines
    return safe_response
```

### Monitoring and Alerts

```python
monitor = AgentMonitor()

# Track unusual behavior
if agent.tool_calls > threshold:
    monitor.alert("Unusual number of tool calls")

if agent.api_cost > budget:
    monitor.alert("Budget exceeded")
    agent.stop()

# Log all actions for audit
monitor.log(action, user, timestamp, result)
```

## Production Considerations

### Observability

- **Tracing**: Track execution through the agent loop (OpenTelemetry, LangSmith)
- **Metrics**: Latency, cost, success rate, tool usage
- **Logging**: Structured logs for debugging
- **User Feedback**: Collect thumbs up/down, bug reports

### Scalability

- **Async Execution**: Handle multiple requests concurrently
- **Queuing**: Buffer requests during high load
- **Caching**: Store repeated queries/results
- **Load Balancing**: Distribute across instances

### Reliability

- **Graceful Degradation**: Fallback to simpler responses if agent fails
- **Timeouts**: Don't wait indefinitely
- **Idempotency**: Safe to retry requests
- **Circuit Breakers**: Stop calling failing services

## Recent Developments (2024-2026)

- **Function calling improvements**: Native support in GPT-4, Claude, Gemini
- **Multi-modal agents**: Process images, audio, video alongside text
- **Code execution**: Agents can write and run code safely (sandboxed environments)
- **Agent frameworks maturity**: LangGraph, CrewAI, AutoGPT improvements
- **Evaluation benchmarks**: GAIA, WebArena, AgentBench for standardized testing
- **Agentic patterns in foundation models**: Models trained specifically for tool use

## Key Takeaways

1. Agents are autonomous systems that reason, plan, and use tools to achieve goals
2. ReAct (Reason + Act) pattern is widely effective for interpretability
3. Memory systems (short-term + long-term) are crucial for complex tasks
4. Good tool design makes agent behavior more reliable
5. Reflection and self-evaluation improve agent performance
6. Safety guardrails are essential for production deployment
7. Testing and evaluation should cover both task success and efficiency
8. Cost and latency management are critical for practical applications

## Related Topics

- [Tool Use](tool-use.md) - Deep dive into function calling and tool design
- [Agent-to-Agent Communication](agent-to-agent.md) - Multi-agent systems
- [Memory Systems](memory-systems.md) - Persistent memory for agents
- [RAG](rag.md) - Knowledge retrieval for agents
- [MCP](mcp.md) - Protocol for agent-tool integration
