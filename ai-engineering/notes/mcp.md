# Model Context Protocol (MCP)

## Overview

The **Model Context Protocol (MCP)** is an open standard developed by Anthropic for connecting AI assistants to external data sources and tools. It provides a universal protocol for communication between AI applications (clients) and context providers (servers).

**Core Purpose**: Enable AI assistants to securely access context from various sources—databases, APIs, file systems, SaaS platforms—without requiring custom integrations for each source.

**Key Benefits**:
- **Standardization**: One protocol for all context sources
- **Security**: Controlled access with user consent
- **Composability**: Mix and match multiple context servers
- **Simplicity**: Easier than building custom integrations

## Architecture

### High-Level Design

```
┌─────────────────┐
│   AI Client     │ (Claude Desktop, IDEs, custom apps)
│  (MCP Client)   │
└────────┬────────┘
         │ MCP Protocol
         ├─────────────┬─────────────┬──────────────
         │             │             │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │  MCP    │   │  MCP    │   │  MCP    │
    │ Server  │   │ Server  │   │ Server  │
    │(GitHub) │   │(Slack)  │   │(Postgres)│
    └─────────┘   └─────────┘   └──────────┘
```

### Components

1. **MCP Client** (Host)
   - AI application that needs context/tools
   - Manages connections to multiple servers
   - Examples: Claude Desktop, IDEs, custom apps

2. **MCP Server** (Provider)
   - Exposes resources, tools, or prompts
   - Implements MCP protocol
   - Examples: GitHub integration, database connector, file system access

3. **Transport Layer**
   - Handles communication between client and server
   - Supports: stdio, HTTP/SSE (Server-Sent Events), WebSocket

## Core Concepts

### Resources

**Resources** are data that servers expose to clients (read-only by default).

**Examples**:
- File contents
- Database rows
- API responses
- Documentation
- Logs

**Resource Schema**:
```typescript
{
  uri: "file:///path/to/document.txt",  // Unique identifier
  name: "document.txt",                  // Human-readable name
  mimeType: "text/plain",                // Content type
  description: "Project documentation"   // Optional description
}
```

**Resource Access**:
```typescript
// Client requests resource
const response = await client.readResource({
  uri: "file:///path/to/document.txt"
});

// Server returns content
{
  contents: [
    {
      uri: "file:///path/to/document.txt",
      mimeType: "text/plain",
      text: "File contents here..."
    }
  ]
}
```

### Prompts

**Prompts** are reusable prompt templates that servers can provide.

**Examples**:
- "Analyze this code for bugs"
- "Summarize the recent commits"
- "Generate test cases for this function"

**Prompt Schema**:
```typescript
{
  name: "analyze-code",
  description: "Analyze code for potential bugs and improvements",
  arguments: [
    {
      name: "language",
      description: "Programming language",
      required: true
    },
    {
      name: "code",
      description: "Code to analyze",
      required: true
    }
  ]
}
```

**Prompt Usage**:
```typescript
// Get prompt with arguments filled
const prompt = await client.getPrompt({
  name: "analyze-code",
  arguments: {
    language: "python",
    code: "def add(a, b): return a + b"
  }
});

// Returns filled prompt ready for LLM
{
  messages: [
    {
      role: "user",
      content: {
        type: "text",
        text: "Analyze this Python code for bugs:\n\ndef add(a, b): return a + b"
      }
    }
  ]
}
```

### Tools

**Tools** are functions that servers expose for clients to execute.

**Examples**:
- Query database
- Send email
- Create GitHub issue
- Run code
- Search web

**Tool Schema**:
```typescript
{
  name: "query_database",
  description: "Execute SQL query on the database",
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "SQL query to execute"
      },
      params: {
        type: "array",
        description: "Query parameters",
        items: { type: "string" }
      }
    },
    required: ["query"]
  }
}
```

**Tool Execution**:
```typescript
// Client calls tool
const result = await client.callTool({
  name: "query_database",
  arguments: {
    query: "SELECT * FROM users WHERE id = ?",
    params: ["123"]
  }
});

// Server returns result
{
  content: [
    {
      type: "text",
      text: "Found 1 user: {id: 123, name: 'Alice', email: 'alice@example.com'}"
    }
  ]
}
```

## Transport Mechanisms

### stdio (Standard Input/Output)

Process-based communication using stdin/stdout.

**Best for**: Local integrations, desktop applications

**Client Code**:
```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const transport = new StdioClientTransport({
  command: "python",
  args: ["-m", "my_mcp_server"]
});

const client = new Client({
  name: "my-client",
  version: "1.0.0"
}, {
  capabilities: {}
});

await client.connect(transport);
```

**Server Code** (Python):
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("my-server")

@server.list_tools()
async def list_tools():
    return [...]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### HTTP with Server-Sent Events (SSE)

HTTP-based communication with streaming responses.

**Best for**: Web applications, remote servers, cloud deployments

**Server Setup**:
```python
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route

server = Server("my-server")

async def handle_sse(request):
    async with SseServerTransport("/messages") as transport:
        await server.run(
            transport.read_stream,
            transport.write_stream
        )

app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse)
])
```

## Building an MCP Server

### Basic Server Structure (Python)

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
import asyncio

# Initialize server
server = Server("example-server")

# Define resources
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="file:///example.txt",
            name="Example File",
            mimeType="text/plain",
            description="An example file"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "file:///example.txt":
        return [
            TextContent(
                type="text",
                text="This is the file content"
            )
        ]
    raise ValueError(f"Unknown resource: {uri}")

# Define tools
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="echo",
            description="Echo back the input",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo"
                    }
                },
                "required": ["message"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "echo":
        message = arguments["message"]
        return [
            TextContent(
                type="text",
                text=f"Echo: {message}"
            )
        ]
    raise ValueError(f"Unknown tool: {name}")

# Run server
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

### Basic Server Structure (TypeScript)

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { ListResourcesRequestSchema, ReadResourceRequestSchema, 
         ListToolsRequestSchema, CallToolRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  {
    name: "example-server",
    version: "1.0.0"
  },
  {
    capabilities: {
      resources: {},
      tools: {}
    }
  }
);

// List available resources
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "file:///example.txt",
        name: "Example File",
        mimeType: "text/plain",
        description: "An example file"
      }
    ]
  };
});

// Read resource content
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  if (request.params.uri === "file:///example.txt") {
    return {
      contents: [
        {
          uri: request.params.uri,
          mimeType: "text/plain",
          text: "This is the file content"
        }
      ]
    };
  }
  throw new Error(`Unknown resource: ${request.params.uri}`);
});

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "echo",
        description: "Echo back the input",
        inputSchema: {
          type: "object",
          properties: {
            message: {
              type: "string",
              description: "Message to echo"
            }
          },
          required: ["message"]
        }
      }
    ]
  };
});

// Execute tool
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "echo") {
    const message = request.params.arguments?.message;
    return {
      content: [
        {
          type: "text",
          text: `Echo: ${message}`
        }
      ]
    };
  }
  throw new Error(`Unknown tool: ${request.params.name}`);
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
```

## Client Integration

### Claude Desktop Integration

Add to Claude Desktop config (`claude_desktop_config.json`):

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "<your_token>"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/username/projects"]
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    }
  }
}
```

### Custom Client Implementation

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

async function createClient() {
  // Create transport
  const transport = new StdioClientTransport({
    command: "python",
    args: ["-m", "my_mcp_server"]
  });

  // Initialize client
  const client = new Client(
    {
      name: "my-ai-app",
      version: "1.0.0"
    },
    {
      capabilities: {
        roots: {
          listChanged: true
        }
      }
    }
  );

  // Connect
  await client.connect(transport);
  
  return client;
}

async function main() {
  const client = await createClient();
  
  // List available tools
  const tools = await client.listTools();
  console.log("Available tools:", tools);
  
  // Call a tool
  const result = await client.callTool({
    name: "query_database",
    arguments: {
      query: "SELECT * FROM users LIMIT 5"
    }
  });
  console.log("Query result:", result);
  
  // List resources
  const resources = await client.listResources();
  console.log("Available resources:", resources);
  
  // Read a resource
  const content = await client.readResource({
    uri: "file:///data/document.txt"
  });
  console.log("Resource content:", content);
}

main().catch(console.error);
```

## Common Use Cases

### File System Access

Expose local files to AI assistant:

```python
import os
from pathlib import Path

@server.list_resources()
async def list_resources():
    resources = []
    for file in Path("/project").rglob("*.py"):
        resources.append(Resource(
            uri=f"file://{file}",
            name=file.name,
            mimeType="text/x-python"
        ))
    return resources

@server.read_resource()
async def read_resource(uri: str):
    path = uri.replace("file://", "")
    with open(path) as f:
        return [TextContent(type="text", text=f.read())]
```

### Database Access

Query databases via MCP:

```python
import asyncpg

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query":
        conn = await asyncpg.connect(DATABASE_URL)
        try:
            results = await conn.fetch(arguments["query"])
            return [TextContent(
                type="text",
                text=str(results)
            )]
        finally:
            await conn.close()
```

### API Integration

Wrap external APIs:

```python
import httpx

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_github":
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/search/repositories",
                params={"q": arguments["query"]},
                headers={"Authorization": f"token {GITHUB_TOKEN}"}
            )
            return [TextContent(
                type="text",
                text=response.json()
            )]
```

### Git Integration

Expose Git operations:

```python
import subprocess

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "git_log":
        result = subprocess.run(
            ["git", "log", f"-n{arguments.get('limit', 10)}", "--oneline"],
            capture_output=True,
            text=True
        )
        return [TextContent(type="text", text=result.stdout)]
    
    if name == "git_diff":
        result = subprocess.run(
            ["git", "diff", arguments.get("file", "")],
            capture_output=True,
            text=True
        )
        return [TextContent(type="text", text=result.stdout)]
```

## Official MCP Servers

Anthropic provides reference implementations:

### @modelcontextprotocol/server-filesystem
Access local file system with read/write permissions

### @modelcontextprotocol/server-github
- Search repositories
- Read file contents
- Create issues and PRs
- Manage branches

### @modelcontextprotocol/server-postgres
Query PostgreSQL databases with read-only access

### @modelcontextprotocol/server-sqlite
Query SQLite databases

### @modelcontextprotocol/server-google-drive
Access Google Drive files and folders

### @modelcontextprotocol/server-slack
- Read channels and messages
- Send messages
- Search content

### @modelcontextprotocol/server-memory
Simple key-value memory storage for agents

## Security Considerations

### Authentication and Authorization

```python
from functools import wraps

def require_auth(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Check authentication token
        if not is_authorized():
            raise PermissionError("Unauthorized")
        return await func(*args, **kwargs)
    return wrapper

@server.call_tool()
@require_auth
async def call_tool(name: str, arguments: dict):
    # Tool implementation
    pass
```

### Input Validation

```python
def validate_query(query: str) -> bool:
    """Validate SQL query for safety"""
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER"]
    query_upper = query.upper()
    
    for keyword in dangerous_keywords:
        if keyword in query_upper:
            return False
    return True

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "query":
        query = arguments["query"]
        if not validate_query(query):
            raise ValueError("Query contains forbidden operations")
        # Execute query...
```

### Resource Access Control

```python
ALLOWED_PATHS = ["/home/user/documents", "/var/log/app"]

@server.read_resource()
async def read_resource(uri: str):
    path = uri.replace("file://", "")
    
    # Check if path is allowed
    if not any(path.startswith(allowed) for allowed in ALLOWED_PATHS):
        raise PermissionError(f"Access denied: {path}")
    
    # Read file...
```

### Rate Limiting

```python
from collections import defaultdict
from time import time

rate_limits = defaultdict(list)

def check_rate_limit(client_id: str, limit: int = 10, window: int = 60):
    """Allow 'limit' requests per 'window' seconds"""
    now = time()
    requests = rate_limits[client_id]
    
    # Remove old requests
    requests[:] = [req for req in requests if now - req < window]
    
    if len(requests) >= limit:
        raise Exception("Rate limit exceeded")
    
    requests.append(now)
```

## Testing MCP Servers

### Unit Tests

```python
import pytest
from mcp.types import CallToolRequest

@pytest.mark.asyncio
async def test_echo_tool():
    result = await server.call_tool(
        name="echo",
        arguments={"message": "Hello, MCP!"}
    )
    assert len(result) == 1
    assert result[0].text == "Echo: Hello, MCP!"

@pytest.mark.asyncio
async def test_invalid_tool():
    with pytest.raises(ValueError, match="Unknown tool"):
        await server.call_tool(name="nonexistent", arguments={})
```

### Integration Tests

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";

describe("MCP Server Integration", () => {
  let client: Client;
  
  beforeAll(async () => {
    client = await createTestClient();
  });
  
  afterAll(async () => {
    await client.close();
  });
  
  test("should list tools", async () => {
    const tools = await client.listTools();
    expect(tools.tools).toHaveLength(2);
    expect(tools.tools[0].name).toBe("echo");
  });
  
  test("should call tool successfully", async () => {
    const result = await client.callTool({
      name: "echo",
      arguments: { message: "test" }
    });
    expect(result.content[0].text).toContain("Echo: test");
  });
});
```

### Manual Testing with MCP Inspector

```bash
# Install inspector
npx @modelcontextprotocol/inspector

# Test your server
npx @modelcontextprotocol/inspector python -m my_mcp_server
```

Provides web UI to:
- View available resources/tools/prompts
- Test tool calls interactively
- Inspect request/response messages

## Best Practices

### 1. Clear Descriptions

```python
Tool(
    name="search_code",
    description="""
    Search codebase for patterns or text.
    
    Use this when you need to:
    - Find function definitions
    - Locate where variables are used
    - Search for code patterns
    
    Returns: List of file paths and line numbers
    """,
    inputSchema={...}
)
```

### 2. Structured Error Handling

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        # Tool implementation
        pass
    except FileNotFoundError as e:
        return [TextContent(
            type="text",
            text=f"Error: File not found - {e.filename}"
        )]
    except PermissionError as e:
        return [TextContent(
            type="text",
            text=f"Error: Permission denied - {e}"
        )]
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return [TextContent(
            type="text",
            text="Error: An unexpected error occurred"
        )]
```

### 3. Resource Pagination

```python
@server.list_resources()
async def list_resources(cursor: Optional[str] = None):
    """Support pagination for large resource lists"""
    page_size = 100
    offset = int(cursor) if cursor else 0
    
    resources = get_resources(limit=page_size, offset=offset)
    
    return {
        "resources": resources,
        "nextCursor": str(offset + page_size) if len(resources) == page_size else None
    }
```

### 4. Logging and Monitoring

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    logger.info(f"Tool called: {name} with args: {arguments}")
    
    try:
        result = execute_tool(name, arguments)
        logger.info(f"Tool {name} completed successfully")
        return result
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        raise
```

## Comparison to Other Protocols

| Feature | MCP | OpenAI Functions | LangChain Tools |
|---------|-----|------------------|-----------------|
| **Standardization** | Open protocol | Vendor-specific | Framework-specific |
| **Resources** | Native support | Via embeddings | Via document loaders |
| **Prompts** | Native support | Manual | Via prompt templates |
| **Transport** | stdio, HTTP/SSE | HTTP API calls | Python functions |
| **Multi-server** | Easy | Requires custom code | Via chains |
| **Streaming** | Supported | Supported | Supported |

## Future Directions

- **MCP 2.0**: Enhanced streaming, better error handling
- **Marketplace**: Shared server registry
- **Security**: OAuth integration, fine-grained permissions
- **Performance**: Connection pooling, caching
- **Observability**: Built-in tracing and metrics

## Key Takeaways

1. MCP standardizes AI-to-external-system communication
2. Three core primitives: Resources (read), Tools (execute), Prompts (templates)
3. Multiple transport options: stdio (local), HTTP/SSE (remote)
4. Easy to build custom servers for specific use cases
5. Security via validation, access control, and rate limiting
6. Well-suited for Claude Desktop and custom AI applications
7. Growing ecosystem of official and community servers

## Related Topics

- [Agents](agents.md) - MCP enables tool use for agents
- [Tool Use](tool-use.md) - Deep dive into tool design patterns
- [RAG](rag.md) - MCP resources can feed RAG systems
- [Context Engineering](context-engineering.md) - MCP resources as context sources

## Resources

- [Official MCP Documentation](https://modelcontextprotocol.io/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [MCP SDKs](https://github.com/modelcontextprotocol/sdk)
- [Claude Desktop Config Guide](https://docs.anthropic.com/en/docs/agents-and-tools)
