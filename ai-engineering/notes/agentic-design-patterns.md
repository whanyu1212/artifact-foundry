# Agentic Design Patterns

## Overview

Agentic design patterns are architectural approaches for building autonomous AI agents that can reason, plan, use tools, and iteratively improve their outputs. These patterns enable agents to move beyond simple prompt-response interactions toward more capable, goal-driven behavior.

## Why Design Patterns Matter

**Without patterns**: Ad-hoc agent implementations are brittle, hard to debug, and difficult to improve.

**With patterns**: Structured approaches provide:
- Reusable architectures that work across domains
- Clear separation of concerns (reasoning, acting, reflection)
- Predictable behavior and failure modes
- Easier testing and debugging
- Proven reliability from real-world use

## Fundamental Agent Components

Before diving into patterns, understand the building blocks:

### 1. Reasoning Engine
The LLM that powers decision-making, planning, and response generation.

### 2. Tool Interface
Mechanisms for the agent to interact with external systems (APIs, databases, search engines).

### 3. Memory System
Storage for conversation history, facts learned, and task context.

### 4. Observation Handler
Processing and interpreting results from tool calls and environment feedback.

### 5. Control Flow
Logic determining what action to take next (loop, branch, terminate).

## Core Agentic Patterns

### Pattern 1: ReAct (Reasoning + Acting)

**Concept**: Interleave reasoning (thinking) with acting (tool use) in explicit steps.

**Flow**:
```
1. Thought: Reason about current state and what to do next
2. Action: Execute tool/operation
3. Observation: Process result
4. Repeat: Continue until task complete
```

**Example**:
```
Question: What's the weather in the capital of France?

Thought: I need to first identify the capital of France.
Action: search("capital of France")
Observation: Paris is the capital of France.

Thought: Now I need to get the weather for Paris.
Action: get_weather("Paris, France")
Observation: Current weather in Paris: 15°C, partly cloudy.

Thought: I have all the information needed.
Answer: The weather in Paris (capital of France) is 15°C and partly cloudy.
```

**Advantages**:
- Transparent reasoning process
- Self-documenting behavior
- Easy to debug (see each step)
- Naturally handles multi-step tasks

**Disadvantages**:
- More tokens (explicit thoughts)
- Can get stuck in reasoning loops
- Requires careful prompt engineering

**When to use**:
- Complex tasks requiring multiple steps
- Need interpretability/debugging
- Combining multiple tools
- User wants to see reasoning

**Implementation tips**:
- Use structured output format for thoughts/actions
- Implement max iteration limit (prevent infinite loops)
- Log all steps for debugging
- Validate action syntax before execution

### Pattern 2: Plan-and-Execute

**Concept**: First create a complete plan, then execute steps sequentially.

**Flow**:
```
1. Planning: Generate full task breakdown
2. Execution: Execute each step in sequence
3. Replanning: Adjust plan if step fails or new info emerges
```

**Example**:
```
Task: Research and write a blog post about quantum computing

Plan:
1. Research quantum computing fundamentals
2. Research recent developments
3. Research applications
4. Create outline
5. Write introduction
6. Write main sections
7. Write conclusion
8. Review and edit

Execute:
[Step 1] Researching fundamentals... ✓
[Step 2] Researching developments... ✓
[Step 3] Researching applications... ✓
...
```

**Advantages**:
- Clear structure upfront
- Parallel execution possible (independent steps)
- Progress tracking
- Easier to estimate cost/time

**Disadvantages**:
- Less adaptive (plan may be wrong)
- Overhead in planning phase
- Hard to handle unexpected situations
- May over/under plan

**When to use**:
- Well-defined tasks with clear steps
- Need progress visibility
- Cost/time constraints
- Tasks benefit from upfront planning

**Implementation tips**:
- Allow replanning when steps fail
- Break plans into milestones
- Validate plan before execution
- Support partial execution

### Pattern 3: Reflexion (Self-Reflection)

**Concept**: Agent critiques and improves its own output through iterative reflection.

**Flow**:
```
1. Generate initial output
2. Self-critique: Identify weaknesses
3. Revise: Improve based on critique
4. Repeat: Until quality threshold met or max iterations
```

**Example**:
```
Task: Write a persuasive email

Iteration 1:
Output: [Generic email]
Critique: Too formal, no personalization, weak call-to-action
Score: 6/10

Iteration 2:
Output: [Improved email with personal touch]
Critique: Better, but could be more concise, stronger hook needed
Score: 8/10

Iteration 3:
Output: [Polished email]
Critique: Concise, personal, strong hook and CTA
Score: 9/10
✓ Accepted
```

**Advantages**:
- Higher quality outputs
- Self-improving
- No external feedback needed
- Works for subjective tasks

**Disadvantages**:
- Multiple LLM calls (expensive)
- May not converge
- Can hallucinate improvements
- Needs good self-evaluation

**When to use**:
- Quality matters more than speed
- Creative tasks (writing, code)
- No clear right answer
- Agent can recognize quality

**Implementation tips**:
- Set max iterations (3-5 typically)
- Define clear quality criteria
- Track scores over iterations
- Consider external validation

### Pattern 4: Tool-Augmented Generation

**Concept**: Agent uses external tools to enhance responses with real-time data and capabilities.

**Flow**:
```
1. Receive query
2. Determine which tools needed
3. Call tools with appropriate parameters
4. Integrate results into response
5. Generate final output
```

**Example**:
```
Query: Compare our Q3 sales to last year's Q3

Tools available: database_query, calculator, chart_generator

Agent:
1. Tool: database_query("SELECT sales FROM sales_data WHERE quarter='Q3' AND year=2025")
   Result: $1.2M

2. Tool: database_query("SELECT sales FROM sales_data WHERE quarter='Q3' AND year=2024")
   Result: $1.0M

3. Tool: calculator("(1.2 - 1.0) / 1.0 * 100")
   Result: 20%

4. Tool: chart_generator(data=[1.0, 1.2], labels=['2024 Q3', '2025 Q3'])
   Result: [chart_url]

Response: Q3 2025 sales were $1.2M, a 20% increase over Q3 2024 ($1.0M). [Chart]
```

**Advantages**:
- Grounds responses in facts
- Extends capabilities beyond LLM
- Reduces hallucination
- Enables complex operations

**Disadvantages**:
- Requires tool integration
- Tool selection can be tricky
- Error handling complexity
- Latency from external calls

**When to use**:
- Need factual, up-to-date information
- Tasks require specialized capabilities
- Calculations or data processing needed
- Integration with existing systems

**Implementation tips**:
- Provide clear tool descriptions
- Validate tool parameters
- Handle tool failures gracefully
- Cache tool results when possible
- Use structured output for tool calls

### Pattern 5: Multi-Step Reasoning (Chain of Thought)

**Concept**: Break complex reasoning into explicit intermediate steps.

**Flow**:
```
1. Decompose problem into sub-problems
2. Solve each sub-problem explicitly
3. Combine results for final answer
```

**Example**:
```
Question: If a store sells 15% more items in Q2 than Q1, and sold 800 items in Q1, how many items sold in both quarters combined?

Step 1: Calculate Q2 sales
- Q1 sales: 800 items
- Increase: 15% of 800 = 0.15 × 800 = 120 items
- Q2 sales: 800 + 120 = 920 items

Step 2: Calculate total
- Q1: 800 items
- Q2: 920 items  
- Total: 800 + 920 = 1,720 items

Answer: 1,720 items
```

**Advantages**:
- Better accuracy on complex reasoning
- Transparent logic
- Easier error identification
- Works with weaker models

**Disadvantages**:
- More verbose (higher cost)
- Slower response
- Can include unnecessary steps
- Reasoning quality varies

**When to use**:
- Math/logic problems
- Multi-hop reasoning
- Need to show work
- Accuracy critical

**Implementation tips**:
- Prompt for explicit steps
- Validate intermediate results
- Allow backtracking if step fails
- Consider Tree of Thoughts for exploration

### Pattern 6: Structured Output Agent

**Concept**: Agent generates responses in predefined structured format (JSON, form fields, database schema).

**Flow**:
```
1. Define output schema
2. Agent generates content
3. Validate against schema
4. Retry if invalid, return if valid
```

**Example**:
```
Task: Extract contact information from email

Schema:
{
  "name": str,
  "email": str,
  "phone": str,
  "company": str
}

Input: "Hi, I'm John Doe from Acme Corp. Reach me at john@acme.com or 555-0100."

Output:
{
  "name": "John Doe",
  "email": "john@acme.com",
  "phone": "555-0100",
  "company": "Acme Corp"
}
```

**Advantages**:
- Predictable output format
- Easy integration with systems
- Validation possible
- Reduces parsing errors

**Disadvantages**:
- Less flexibility
- Schema design upfront
- May force unnatural outputs
- Retry overhead if invalid

**When to use**:
- Feeding downstream systems
- Data extraction tasks
- API responses
- Form filling

**Implementation tips**:
- Use JSON schema validation
- Provide examples in prompt
- Set max retries (don't loop forever)
- Handle partial extractions
- Consider function calling APIs

### Pattern 7: Router Agent

**Concept**: Agent classifies request and routes to specialized handlers.

**Flow**:
```
1. Analyze incoming request
2. Classify intent/category
3. Route to appropriate specialized agent or workflow
4. Return result
```

**Example**:
```
Input: "I need to cancel my subscription"

Router:
- Category: account_management
- Subcategory: cancellation
- Route to: cancellation_specialist_agent

Cancellation Agent:
[Handles cancellation flow]
```

**Advantages**:
- Specialization improves quality
- Scalable (add new routes)
- Clear separation of concerns
- Optimize per-category

**Disadvantages**:
- Classification errors
- Extra routing step
- Need multiple specialized agents
- Complex orchestration

**When to use**:
- Diverse request types
- Domain-specific expertise needed
- High volume (specialize per category)
- Clear categorization possible

**Implementation tips**:
- Use lightweight router (small model)
- Provide confidence scores
- Fallback for ambiguous requests
- Monitor misclassification rate
- Support multi-category routing

### Pattern 8: Human-in-the-Loop

**Concept**: Agent requests human feedback at critical decision points.

**Flow**:
```
1. Agent performs task
2. At decision point, pause for human input
3. Human approves, rejects, or modifies
4. Agent continues with feedback
```

**Example**:
```
Task: Send marketing email to 10,000 customers

Agent: Generated email draft
[Shows preview]
Action: Pause for approval
Human: [Approved / "Make it more professional"]
Agent: [Executes send / Revises draft]
```

**Advantages**:
- Safety and oversight
- Increases trust
- Leverages human judgment
- Catch errors before they propagate

**Disadvantages**:
- Breaks autonomy (blocks execution)
- Slower latency
- Requires UI for human interaction
- Not fully scalable

**When to use**:
- High-stakes actions (payments, massive emails)
- Ambiguous tasks requiring judgment
- Generating training data (human teaching)
- Early stages of deployment

**Implementation tips**:
- Design clear UI for approval
- Allow modifying specific parts
- Timeout handling (if human doesn't respond)
- Log all decisions for auditing

## Advanced Patterns (from Antonio Gulli)

Based on *Agentic Design Patterns* by Antonio Gulli, these additional patterns address complex system requirements.

### Pattern 9: Prompt Chaining

**Concept**: Decompose a task into a fixed sequence of LLM calls, where the output of one step becomes the input of the next. Simpler and more deterministic than ReAct.

**Flow**:
```
Input → [Prompt 1: Extract] → [Prompt 2: Classification] → [Prompt 3: Summarize] → Output
```

**When to use**: 
- The workflow is linear and predictable
- Reliability is more important than flexibility
- Latency needs to be predictable

### Pattern 10: Parallelization

**Concept**: Execute multiple independent agent tasks concurrently to speed up processing or gather diverse perspectives.

**Flow**:
```
          ┌→ Agent A (Research) ─┐
Input ────┼→ Agent B (Code) ─────┼→ Aggregator → Output
          └→ Agent C (Review) ───┘
```

**Types**:
- **Sectioning**: Splitting a large task (e.g., writing different chapters)
- **Voting**: Asking multiple agents the same question for consensus

### Pattern 11: Goal Setting & Monitoring

**Concept**: Agents that don't just execute, but actively maintain a dynamic list of goals and track progress against them.

**Components**:
- **Goal Creator**: Decomposes high-level instructions
- **Monitor**: Checks if actions are moving towards the goal
- **Replanner**: Updates goals if blocked

### Pattern 12: Exception Handling & Recovery

**Concept**: Dedicated architectural patterns for recovering from failures without crashing.

**Strategies**:
- **Fallback Agents**: Switch to a simpler/different model on failure
- **Retry with Feedback**: Feed the error message back to the agent to fix its own mistake
- **Human Escalation**: Route to human only on repeated errors

### Pattern 13: Adaptation (Self-Evolving)

**Concept**: Agents that improve their own performance over time by updating their own system prompts or tool definitions based on success/failure history.

**Flow**:
1. Execute task
2. Evaluate performance
3. If successful, save successful trajectory
4. If failed, update "Knowledge/Strategy" memory

### Pattern 14: Prioritization

**Concept**: Agent manages a backlog of tasks and dynamically decides what to work on next based on urgency and importance, acting like a Project Manager.

**When to use**:
- Autonomous agents running over long periods
- Handling asynchronous events/interruptions

## Choosing the Right Pattern

| Pattern | Best For | Complexity | Cost |
|---------|----------|------------|------|
| **Prompt Chaining** | Predictable, linear workflows | Low | Low |
| **ReAct** | General purpose, transparent reasoning | Medium | Medium |
| **Plan-and-Execute** | Defined multi-step tasks | Medium | Low |
| **Reflexion** | High-quality generation, coding | High | High |
| **Tool Use** | Factual queries, enhancing capabilities | Medium | Low |
| **Parallelization** | Speed, diverse perspectives (voting) | Medium | High |
| **Goal Setting** | Long-running, ambiguous tasks | High | Medium |
| **Structured Output** | System integration, data extraction | Low | Low |
| **Exception Handling** | Robust production systems | Medium | Low |
| **Human-in-the-Loop** | High-stakes, safety-critical tasks | Low (Architecture) | High (Time) |

## Future Trends

1. **Agent Swarms**: Massive numbers of simple agents collaborating
2. **Long-term Memory**: Agents that learn and remember user preferences over months
3. **Multimodal Agents**: Natively processing reason/act across text, image, audio
4. **Self-Evolving Agents**: Agents that write their own tools and prompt updates

## Key Takeaways

1. **Start Simple**: Don't use a complex agent pattern if a simple prompt or chain works.
2. **Visibility**: Ensure you can see the agent's "thought process" for debugging.
3. **Failure Handling**: Agents will fail. Design for graceful degradation and retries.
4. **Reliability > Autonomy**: Constrain the agent's action space to ensure reliability.
5. **Pattern Mixing**: Real-world systems often combine patterns (e.g., Router -> ReAct -> Structured Output).

## Further Reading

- [Agentic Design Patterns](https://github.com/sarwarbeing-ai/Agentic_Design_Patterns/blob/main/Agentic_Design_Patterns.pdf) - Antonio Gulli
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Reflexion Paper](https://arxiv.org/abs/2303.11366)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [LangChain Agent Concepts](https://python.langchain.com/docs/modules/agents/concepts)