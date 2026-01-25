# FastAPI API Design Patterns

## Overview

FastAPI is a modern Python web framework built on Starlette (async) and Pydantic (validation). It provides automatic API documentation, type safety, and excellent performance.

## Core Concepts

### 1. Type Hints and Validation

FastAPI uses Python type hints for:
- Automatic request validation
- Response serialization
- API documentation generation
- Editor autocomplete

```python
@app.post("/items")
def create_item(item: Item) -> ItemResponse:
    # FastAPI validates input, serializes output automatically
    return ItemResponse(...)
```

### 2. Dependency Injection

FastAPI's dependency injection system enables:
- Shared logic (authentication, database connections)
- Testing (mock dependencies easily)
- Code reuse without global state

**Common use cases:**
- Database sessions
- Authentication/authorization
- Rate limiting
- Shared configurations

### 3. Path Operations

**HTTP Methods:**
- `@app.get()` - Read data
- `@app.post()` - Create data
- `@app.put()` - Update/replace data
- `@app.patch()` - Partial update
- `@app.delete()` - Delete data

**Path Parameters:**
```python
@app.get("/items/{item_id}")
def get_item(item_id: int):  # Automatic validation and conversion
    return {"item_id": item_id}
```

**Query Parameters:**
```python
@app.get("/items")
def list_items(skip: int = 0, limit: int = 10):
    # /items?skip=0&limit=10
    return items[skip:skip+limit]
```

**Request Body:**
```python
@app.post("/items")
def create_item(item: Item):  # Pydantic model
    return item
```

### 4. Pydantic Models

Define request/response schemas with validation:

```python
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    description: Optional[str] = None
```

**Benefits:**
- Automatic validation
- Clear API contracts
- JSON schema generation
- Type safety

### 5. Response Models

Control what gets returned to clients:

```python
@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate):
    # Internal User model may have password, etc.
    # response_model ensures only UserResponse fields are returned
    return user_in_db
```

**Patterns:**
- `UserCreate` - For creating (includes password)
- `UserResponse` - For reading (excludes password)
- `UserUpdate` - For updating (all optional fields)

## Common Patterns

### RESTful API Design

**Resource-based URLs:**
```
GET    /users          - List users
POST   /users          - Create user
GET    /users/{id}     - Get user
PUT    /users/{id}     - Update user
DELETE /users/{id}     - Delete user
```

**Nested Resources:**
```
GET    /users/{id}/posts       - User's posts
POST   /users/{id}/posts       - Create post for user
```

### Error Handling

**HTTPException for client errors:**
```python
from fastapi import HTTPException

@app.get("/items/{item_id}")
def get_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return items[item_id]
```

**Custom Exception Handlers:**
```python
@app.exception_handler(CustomError)
async def custom_handler(request: Request, exc: CustomError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )
```

### Middleware

Add cross-cutting concerns:
- Logging
- CORS
- Authentication
- Request timing
- Rate limiting

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url} - {duration:.2f}s")
    return response
```

### Authentication Patterns

**API Key:**
```python
async def verify_api_key(api_key: str = Header(...)):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401)
    return api_key
```

**Bearer Token:**
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify JWT or similar
    return decode_token(credentials.credentials)
```

**OAuth2:**
```python
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Verify token and return user
    return user
```

### Background Tasks

For operations that don't need to block response:

```python
from fastapi import BackgroundTasks

@app.post("/send-email")
async def send_email(
    email: EmailSchema,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_email_task, email)
    return {"message": "Email will be sent"}
```

**Use cases:**
- Sending emails
- Logging analytics
- Cleanup tasks
- Notifications

### Request/Response Lifecycle

1. **Request received**
2. **Middleware (request processing)**
3. **Dependency resolution** (e.g., auth, DB session)
4. **Path operation function** (your handler)
5. **Response model validation**
6. **Middleware (response processing)**
7. **Response sent**

## Advanced Patterns

### Sub-Applications and Routers

**APIRouter for modularity:**
```python
# users_router.py
from fastapi import APIRouter

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
def list_users():
    ...

# main.py
app.include_router(users_router)
```

### Versioning Strategies

**URL Path Versioning:**
```python
app = FastAPI()
v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")

app.include_router(v1_router)
app.include_router(v2_router)
```

**Header Versioning:**
```python
@app.get("/items")
def get_items(version: str = Header(default="v1", alias="API-Version")):
    if version == "v1":
        return v1_items()
    elif version == "v2":
        return v2_items()
```

### WebSockets

For real-time bidirectional communication:

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Echo: {data}")
```

### File Uploads

```python
from fastapi import File, UploadFile

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {"filename": file.filename, "size": len(contents)}
```

## Best Practices

### 1. Use Dependency Injection
- Avoid global state
- Makes testing easier
- Better code organization

### 2. Define Clear Schemas
- Separate create/update/response models
- Use Field() for validation and documentation
- Include examples for better docs

### 3. Proper Error Handling
- Use appropriate HTTP status codes
- Provide clear error messages
- Don't expose internal errors

### 4. Async When Beneficial
- Use `async def` for I/O-bound operations (DB, external APIs)
- Use regular `def` for CPU-bound operations
- Don't mix blocking code in async functions

### 5. Document Your API
- Use docstrings (appear in /docs)
- Add descriptions to Pydantic fields
- Include examples
- Tag endpoints for organization

### 6. Security
- Validate all inputs
- Use HTTPS in production
- Implement rate limiting
- Sanitize error messages
- Use security headers (middleware)

### 7. Testing
- Use TestClient for integration tests
- Mock dependencies for unit tests
- Test error cases
- Test edge cases

## Common Pitfalls

### 1. Mutable Default Arguments
```python
# BAD
@app.get("/items")
def get_items(ids: List[int] = []):  # Mutable default!
    ...

# GOOD
@app.get("/items")
def get_items(ids: List[int] = None):
    ids = ids or []
```

### 2. Blocking Async Functions
```python
# BAD
@app.get("/data")
async def get_data():
    data = requests.get(url)  # Blocking call in async function!
    return data

# GOOD
@app.get("/data")
async def get_data():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### 3. Not Using Response Models
```python
# BAD - Might return sensitive data
@app.get("/users/{id}")
def get_user(id: int):
    return db.get_user(id)  # Returns User with password!

# GOOD - Only returns allowed fields
@app.get("/users/{id}", response_model=UserResponse)
def get_user(id: int):
    return db.get_user(id)  # Password excluded
```

## Performance Considerations

1. **Use async for I/O-bound operations** - Better concurrency
2. **Enable compression** - Reduce response sizes
3. **Implement caching** - Redis, in-memory caches
4. **Database connection pooling** - Reuse connections
5. **Pagination** - Don't return all data at once
6. **Background tasks** - Offload non-critical work

## Tools and Extensions

- **SQLAlchemy** - ORM for databases
- **Alembic** - Database migrations
- **pytest** - Testing
- **httpx** - Async HTTP client
- **python-jose** - JWT handling
- **passlib** - Password hashing
- **python-multipart** - File upload support

## Related Topics

- See [ml-system-design/notes/fastapi-model-serving.md](../../ml-system-design/notes/fastapi-model-serving.md) for ML-specific patterns
- See [productionization/] for deployment strategies (when created)

## Resources

- Official Docs: https://fastapi.tiangolo.com/
- Pydantic Docs: https://docs.pydantic.dev/
- Starlette Docs: https://www.starlette.io/
