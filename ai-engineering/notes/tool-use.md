# Tool Use and Function Calling

## Overview

**Tool use** (also called function calling) enables language models to interact with external systems, APIs, and computational resources. Instead of relying solely on text generation, models can invoke predefined functions to perform actions, retrieve information, or process data.

**Key Capabilities**:
- Execute code and calculations
- Query databases and APIs
- Access real-time information
- Interact with external services
- Perform structured data transformations

**Why It Matters**:
- Extends model capabilities beyond text generation
- Provides accurate, up-to-date information
- Enables action-taking (not just conversation)
- Makes AI assistants practical and useful

## Core Concepts

### Function Definitions

Tools are defined with schemas describing their purpose and parameters:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or coordinates"
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "description": "Temperature units"
      }
    },
    "required": ["location"]
  }
}
```

### Function Calling Flow

```
1. User: "What's the weather in Tokyo?"
2. Model decides to call: get_weather(location="Tokyo", units="celsius")
3. System executes function: returns {"temp": 18, "condition": "cloudy"}
4. Model receives result
5. Model generates response: "It's 18°C and cloudy in Tokyo"
```

### Provider Implementations

Different providers have varying approaches:

**OpenAI**:
```python
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's 15% of $83.50?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"}
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
)
```

**Anthropic Claude**:
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=[
        {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    ],
    messages=[{"role": "user", "content": "What's 15% of $83.50?"}]
)
```

## Tool Design Principles

### 1. Single Responsibility

Each tool should do one thing well:

```python
# Good: Focused tools
def get_user(user_id: str) -> dict:
    """Get user information by ID"""
    pass

def update_user(user_id: str, updates: dict) -> dict:
    """Update user information"""
    pass

# Bad: Does too much
def manage_user(action: str, user_id: str, **kwargs):
    """Do various user operations"""
    if action == "get":
        # ...
    elif action == "update":
        # ...
    elif action == "delete":
        # ...
```

### 2. Clear Descriptions

Be explicit about what the tool does and when to use it:

```python
{
    "name": "search_database",
    "description": """
    Search the customer database for records matching the query.
    
    Use this when you need to:
    - Find customer information by name, email, or ID
    - Check order history for a customer
    - Verify account details
    
    Do NOT use this for:
    - General web search (use web_search instead)
    - Product inventory (use check_inventory instead)
    - Internal documentation (use search_docs instead)
    
    Examples:
    - "Find customer with email john@example.com"
    - "Get order history for customer ID 12345"
    """,
    "parameters": {...}
}
```

### 3. Type Safety

Use strict type definitions:

```python
{
    "name": "create_event",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Event title"
            },
            "date": {
                "type": "string",
                "format": "date",  # ISO 8601 date
                "description": "Event date (YYYY-MM-DD)"
            },
            "time": {
                "type": "string",
                "pattern": "^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$",
                "description": "Event time (HH:MM)"
            },
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of attendee email addresses"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Event priority level"
            }
        },
        "required": ["title", "date", "time"]
    }
}
```

### 4. Graceful Error Handling

Return informative errors that help the model recover:

```python
def search_database(query: str, limit: int = 10):
    """Search database with error handling"""
    try:
        # Validate inputs
        if not query or len(query) < 3:
            return {
                "success": False,
                "error": "Query too short",
                "suggestion": "Provide at least 3 characters to search"
            }
        
        if limit > 100:
            return {
                "success": False,
                "error": "Limit too high",
                "suggestion": "Maximum limit is 100. Use pagination for more results."
            }
        
        # Execute search
        results = db.search(query, limit=limit)
        
        if not results:
            return {
                "success": True,
                "results": [],
                "message": "No results found. Try broader search terms."
            }
        
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
        
    except DatabaseConnectionError as e:
        return {
            "success": False,
            "error": "Database unavailable",
            "suggestion": "Please try again in a moment"
        }
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return {
            "success": False,
            "error": "Search failed",
            "suggestion": "Please rephrase your query or contact support"
        }
```

### 5. Idempotency

Make tools safe to retry:

```python
def create_user(email: str, name: str):
    """Idempotent user creation"""
    
    # Check if user already exists
    existing = db.get_user_by_email(email)
    if existing:
        return {
            "success": True,
            "user": existing,
            "message": "User already exists"
        }
    
    # Create new user
    user = db.create_user(email=email, name=name)
    return {
        "success": True,
        "user": user,
        "message": "User created"
    }
```

## Implementation Patterns

### Basic Tool Execution Loop

```python
from openai import OpenAI
import json

client = OpenAI()

def execute_tool(tool_name: str, arguments: dict):
    """Execute the requested tool"""
    tools = {
        "get_weather": get_weather,
        "calculator": calculator,
        "search": search
    }
    
    if tool_name not in tools:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    return tools[tool_name](**arguments)

def chat_with_tools(messages, tools, max_iterations=10):
    """Handle conversation with tool use"""
    
    for iteration in range(max_iterations):
        # Get model response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # Check if model wants to call tools
        if message.tool_calls:
            # Execute each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute tool
                result = execute_tool(tool_name, arguments)
                
                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        else:
            # No more tool calls, return final response
            return message.content
    
    raise Exception("Max iterations reached")

# Usage
messages = [{"role": "user", "content": "What's the weather in Tokyo?"}]
response = chat_with_tools(messages, weather_tools)
```

### Async Tool Execution

Execute multiple tools in parallel:

```python
import asyncio

async def execute_tool_async(tool_name: str, arguments: dict):
    """Async tool execution"""
    # Simulate async operation
    await asyncio.sleep(0.1)
    return tools[tool_name](**arguments)

async def chat_with_tools_async(messages, tools):
    """Handle parallel tool execution"""
    
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools
    )
    
    message = response.choices[0].message
    
    if message.tool_calls:
        # Execute all tool calls in parallel
        tasks = [
            execute_tool_async(
                tc.function.name,
                json.loads(tc.function.arguments)
            )
            for tc in message.tool_calls
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Add all results to messages
        for tool_call, result in zip(message.tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
    
    return messages
```

### Tool Selection Guidance

Help the model choose the right tool:

```python
def build_tool_catalog():
    """Organize tools by category"""
    return {
        "data_retrieval": [
            "search_database",
            "get_user",
            "list_products"
        ],
        "data_modification": [
            "create_user",
            "update_order",
            "delete_item"
        ],
        "external_apis": [
            "get_weather",
            "search_web",
            "translate_text"
        ],
        "computation": [
            "calculator",
            "analyze_data",
            "generate_report"
        ]
    }

system_prompt = """
You have access to these tool categories:

DATA RETRIEVAL: Search and fetch information
- search_database: Find customer/order records
- get_user: Get specific user details
- list_products: Browse product catalog

DATA MODIFICATION: Create, update, delete records
- create_user: Add new user (requires confirmation)
- update_order: Modify order details
- delete_item: Remove items (requires confirmation)

EXTERNAL APIS: Access external services
- get_weather: Current weather data
- search_web: Web search
- translate_text: Language translation

COMPUTATION: Perform calculations and analysis
- calculator: Mathematical operations
- analyze_data: Statistical analysis
- generate_report: Create formatted reports

Choose the appropriate tool based on the user's request.
"""
```

### Tool Chaining

One tool's output feeds into another:

```python
async def execute_tool_chain(steps):
    """Execute sequence of dependent tools"""
    results = {}
    
    for step in steps:
        tool_name = step["tool"]
        
        # Resolve argument references
        arguments = {}
        for arg_name, arg_value in step["arguments"].items():
            if isinstance(arg_value, str) and arg_value.startswith("$"):
                # Reference to previous result
                ref = arg_value[1:]  # Remove $
                arguments[arg_name] = results[ref]
            else:
                arguments[arg_name] = arg_value
        
        # Execute tool
        result = await execute_tool_async(tool_name, arguments)
        results[step["name"]] = result
    
    return results

# Example chain
chain = [
    {
        "name": "user",
        "tool": "get_user",
        "arguments": {"user_id": "123"}
    },
    {
        "name": "orders",
        "tool": "get_user_orders",
        "arguments": {"user_id": "$user.id"}  # Reference previous result
    },
    {
        "name": "summary",
        "tool": "summarize_orders",
        "arguments": {"orders": "$orders"}
    }
]

results = await execute_tool_chain(chain)
```

## Advanced Techniques

### Tool Approval/Confirmation

Require human approval for sensitive operations:

```python
class ToolExecutor:
    def __init__(self, auto_approve_safe=True):
        self.auto_approve_safe = auto_approve_safe
        self.pending_approvals = []
    
    def is_safe_tool(self, tool_name: str) -> bool:
        """Check if tool is safe to auto-execute"""
        safe_tools = ["get_weather", "calculator", "search"]
        return tool_name in safe_tools
    
    async def execute_with_approval(self, tool_name: str, arguments: dict):
        """Execute tool with approval workflow"""
        
        if self.auto_approve_safe and self.is_safe_tool(tool_name):
            # Safe tool, execute immediately
            return await execute_tool(tool_name, arguments)
        
        # Requires approval
        approval_id = str(uuid.uuid4())
        self.pending_approvals.append({
            "id": approval_id,
            "tool": tool_name,
            "arguments": arguments,
            "status": "pending"
        })
        
        return {
            "requires_approval": True,
            "approval_id": approval_id,
            "message": f"Tool '{tool_name}' requires approval. Review and approve?"
        }
    
    async def approve(self, approval_id: str):
        """Approve and execute pending tool"""
        for approval in self.pending_approvals:
            if approval["id"] == approval_id:
                result = await execute_tool(
                    approval["tool"],
                    approval["arguments"]
                )
                approval["status"] = "approved"
                return result
        
        raise ValueError("Approval not found")
```

### Tool Result Caching

Cache expensive or redundant tool calls:

```python
from functools import lru_cache
import hashlib
import json

class CachedToolExecutor:
    def __init__(self):
        self.cache = {}
    
    def cache_key(self, tool_name: str, arguments: dict) -> str:
        """Generate cache key from tool and arguments"""
        key_data = json.dumps({
            "tool": tool_name,
            "args": arguments
        }, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def execute(self, tool_name: str, arguments: dict, ttl: int = 300):
        """Execute with caching"""
        
        key = self.cache_key(tool_name, arguments)
        
        # Check cache
        if key in self.cache:
            cached = self.cache[key]
            if time.time() - cached["timestamp"] < ttl:
                return cached["result"]
        
        # Execute tool
        result = await execute_tool(tool_name, arguments)
        
        # Cache result
        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }
        
        return result
```

### Dynamic Tool Generation

Create tools on-the-fly based on context:

```python
def generate_database_tools(schema):
    """Generate tools for database tables dynamically"""
    tools = []
    
    for table in schema.tables:
        # Generate query tool
        tools.append({
            "name": f"query_{table.name}",
            "description": f"Query the {table.name} table",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "description": f"Filter conditions for {table.name}"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results"
                    }
                }
            }
        })
        
        # Generate create tool
        properties = {}
        required = []
        for column in table.columns:
            properties[column.name] = {
                "type": map_sql_to_json_type(column.type),
                "description": column.description or f"{column.name} field"
            }
            if not column.nullable:
                required.append(column.name)
        
        tools.append({
            "name": f"create_{table.name}",
            "description": f"Create new {table.name} record",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })
    
    return tools
```

### Tool Retry Logic

Handle transient failures:

```python
async def execute_with_retry(
    tool_name: str,
    arguments: dict,
    max_retries: int = 3,
    backoff: float = 1.0
):
    """Execute tool with exponential backoff retry"""
    
    for attempt in range(max_retries):
        try:
            return await execute_tool(tool_name, arguments)
        except TransientError as e:
            if attempt == max_retries - 1:
                raise  # Last attempt, give up
            
            wait_time = backoff * (2 ** attempt)
            logging.warning(
                f"Tool {tool_name} failed (attempt {attempt + 1}), "
                f"retrying in {wait_time}s: {e}"
            )
            await asyncio.sleep(wait_time)
        except PermanentError as e:
            # Don't retry permanent errors
            raise
```

## Common Tool Categories

### Data Retrieval

```python
tools = [
    {
        "name": "search",
        "description": "Search knowledge base",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object"}
            }
        }
    },
    {
        "name": "get_by_id",
        "description": "Get specific record by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"}
            },
            "required": ["id"]
        }
    }
]
```

### Data Modification

```python
tools = [
    {
        "name": "create",
        "description": "Create new record",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "object"}
            },
            "required": ["data"]
        }
    },
    {
        "name": "update",
        "description": "Update existing record",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "updates": {"type": "object"}
            },
            "required": ["id", "updates"]
        }
    },
    {
        "name": "delete",
        "description": "Delete record",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"}
            },
            "required": ["id"]
        }
    }
]
```

### Computation

```python
tools = [
    {
        "name": "calculate",
        "description": "Perform mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression (e.g., '2 + 2 * 3')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "analyze_data",
        "description": "Statistical analysis of data",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["mean", "median", "std", "min", "max"]
                    }
                }
            },
            "required": ["data", "operations"]
        }
    }
]
```

### External APIs

```python
tools = [
    {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["location"]
        }
    }
]
```

## Testing Tools

### Unit Tests

```python
import pytest

def test_calculator_tool():
    """Test calculator tool"""
    result = calculator(expression="2 + 2")
    assert result == 4

def test_calculator_error_handling():
    """Test error handling"""
    result = calculator(expression="invalid")
    assert result["success"] == False
    assert "error" in result

@pytest.mark.asyncio
async def test_search_tool():
    """Test async search tool"""
    result = await search(query="Python", limit=5)
    assert isinstance(result, list)
    assert len(result) <= 5
```

### Integration Tests

```python
def test_tool_calling_flow():
    """Test end-to-end tool calling"""
    messages = [
        {"role": "user", "content": "What's 15% of $100?"}
    ]
    
    response = chat_with_tools(messages, calculator_tools)
    
    assert "$15" in response or "15" in response

def test_multi_tool_scenario():
    """Test using multiple tools"""
    messages = [
        {"role": "user", "content": "Search for user@example.com and get their orders"}
    ]
    
    response = chat_with_tools(messages, [user_tools, order_tools])
    
    # Verify both tools were used
    assert "user" in response.lower()
    assert "order" in response.lower()
```

### Mock Tools for Development

```python
def create_mock_tool(name: str, return_value: Any):
    """Create mock tool for testing"""
    def mock_function(**kwargs):
        return return_value
    
    return {
        "name": name,
        "function": mock_function,
        "schema": {
            "name": name,
            "description": f"Mock {name} tool",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }

# Usage
mock_weather = create_mock_tool(
    "get_weather",
    {"temp": 20, "condition": "sunny"}
)
```

## Best Practices

### 1. Comprehensive Descriptions

```python
{
    "name": "send_email",
    "description": """
    Send an email to one or more recipients.
    
    Use when the user explicitly asks to send an email or message.
    
    Important:
    - Verify recipient email addresses are valid
    - Keep subject lines under 100 characters
    - For sensitive content, ask for confirmation first
    
    Examples:
    - "Send an email to john@example.com about the meeting"
    - "Email the team about tomorrow's deadline"
    """,
    "parameters": {...}
}
```

### 2. Validate Inputs

```python
def send_email(to: list[str], subject: str, body: str):
    """Send email with validation"""
    
    # Validate recipients
    if not to or len(to) == 0:
        return {"error": "At least one recipient required"}
    
    for email in to:
        if not is_valid_email(email):
            return {"error": f"Invalid email: {email}"}
    
    # Validate subject
    if not subject:
        return {"error": "Subject is required"}
    
    if len(subject) > 200:
        return {"error": "Subject too long (max 200 characters)"}
    
    # Send email
    # ...
```

### 3. Return Structured Data

```python
def search_products(query: str):
    """Return well-structured results"""
    results = db.search(query)
    
    return {
        "success": True,
        "query": query,
        "results": [
            {
                "id": product.id,
                "name": product.name,
                "price": product.price,
                "in_stock": product.in_stock,
                "url": product.url
            }
            for product in results
        ],
        "total": len(results),
        "timestamp": datetime.now().isoformat()
    }
```

### 4. Log Tool Usage

```python
def execute_tool(tool_name: str, arguments: dict):
    """Execute with logging"""
    
    logging.info(f"Executing tool: {tool_name}")
    logging.debug(f"Arguments: {arguments}")
    
    start_time = time.time()
    
    try:
        result = tools[tool_name](**arguments)
        
        execution_time = time.time() - start_time
        logging.info(
            f"Tool {tool_name} completed in {execution_time:.2f}s"
        )
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logging.error(
            f"Tool {tool_name} failed after {execution_time:.2f}s: {e}"
        )
        raise
```

### 5. Rate Limiting

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self):
        self.calls = defaultdict(list)
    
    def check_limit(self, tool_name: str, limit: int = 10, window: int = 60):
        """Allow 'limit' calls per 'window' seconds"""
        now = time.time()
        calls = self.calls[tool_name]
        
        # Remove old calls
        calls[:] = [t for t in calls if now - t < window]
        
        if len(calls) >= limit:
            return False, f"Rate limit exceeded for {tool_name}"
        
        calls.append(now)
        return True, None

rate_limiter = RateLimiter()

def execute_with_rate_limit(tool_name: str, arguments: dict):
    """Execute tool with rate limiting"""
    allowed, error = rate_limiter.check_limit(tool_name)
    
    if not allowed:
        return {"success": False, "error": error}
    
    return execute_tool(tool_name, arguments)
```

## Common Challenges

### Challenge: Tool Selection Errors

**Problem**: Model calls wrong tool or no tool when it should

**Solutions**:
- Clearer tool descriptions with examples
- Explicit "when to use" / "when NOT to use" guidance
- Reduce number of similar tools
- Few-shot examples showing correct usage

### Challenge: Invalid Arguments

**Problem**: Model provides wrong types or missing required parameters

**Solutions**:
- Strict schema validation
- Good parameter descriptions with examples
- Enum constraints where applicable
- Return helpful error messages

### Challenge: Expensive Tool Calls

**Problem**: Too many unnecessary calls or repeated calls

**Solutions**:
- Caching (same inputs → same outputs)
- Encourage model to batch operations
- Rate limiting
- Cost monitoring and alerts

### Challenge: Security Risks

**Problem**: Tools could be abused or access sensitive data

**Solutions**:
- Input validation and sanitization
- Access control checks
- Approval workflows for sensitive operations
- Audit logging
- Sandbox execution

## Key Takeaways

1. Tool use extends LLM capabilities beyond text generation
2. Good tool design: single responsibility, clear descriptions, type safety
3. Error handling should help models recover gracefully
4. Tool execution patterns: basic loop, async, chaining, approval
5. Common categories: data retrieval, modification, computation, external APIs
6. Best practices: validate inputs, structure outputs, log usage, rate limit
7. Testing is essential: unit tests for tools, integration tests for flows

## Related Topics

- [Agents](agents.md) - Agents rely heavily on tool use
- [MCP](mcp.md) - Protocol for standardized tool integration
- [Agent-to-Agent Communication](agent-to-agent.md) - Tools in multi-agent systems
- [RAG](rag.md) - Tools can provide retrieval capabilities
