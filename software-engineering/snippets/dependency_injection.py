"""
Dependency Injection in FastAPI

Demonstrates FastAPI's powerful dependency injection (DI) system for:
1. Shared resources (database connections, caches)
2. Authentication and authorization
3. Common logic (pagination, filtering)
4. Testing (easy mocking)

Key Concepts:
- Dependencies are functions that can take other dependencies
- Use Depends() to inject dependencies into path operations
- Dependencies can be reused across multiple endpoints
- Makes code more modular and testable

Run:
    uvicorn dependency_injection:app --reload
"""

from typing import Generator, Optional, Annotated
from contextlib import contextmanager

from fastapi import FastAPI, Depends, HTTPException, Header, Query
from pydantic import BaseModel


# =============================================================================
# Example 1: Database Session Management
# =============================================================================

class Database:
    """Mock database for demonstration."""

    def __init__(self):
        self.connected = False
        self.data = {"users": {1: "Alice", 2: "Bob"}}

    def connect(self):
        """Establish connection."""
        print("  [DB] Connecting to database...")
        self.connected = True

    def disconnect(self):
        """Close connection."""
        print("  [DB] Disconnecting from database...")
        self.connected = False

    def get_user(self, user_id: int) -> Optional[str]:
        """Query user."""
        return self.data["users"].get(user_id)


# Global database instance (in production, use connection pool)
db = Database()


def get_db() -> Generator[Database, None, None]:
    """
    Database session dependency.

    This is a generator dependency that:
    1. Runs code before the path operation (setup)
    2. Yields the resource to the path operation
    3. Runs code after the path operation (cleanup)

    Pattern:
        setup
        yield resource
        cleanup

    FastAPI automatically handles the lifecycle.
    """
    print("  [Dependency] Setting up database session")

    # Setup: Connect to database
    db.connect()

    try:
        # Yield database to the path operation
        yield db
    finally:
        # Cleanup: Always runs, even if path operation raises exception
        db.disconnect()
        print("  [Dependency] Cleaned up database session")


# =============================================================================
# Example 2: Authentication Dependencies
# =============================================================================

class User(BaseModel):
    """User model."""
    user_id: int
    username: str
    role: str


def verify_api_key(api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Verify API key from header.

    This dependency:
    - Extracts X-API-Key header
    - Validates the key
    - Raises HTTPException if invalid

    Dependencies can access:
    - Path parameters
    - Query parameters
    - Headers
    - Cookies
    - Request body
    """
    valid_key = "secret-api-key-123"

    if api_key != valid_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    return api_key


def get_current_user(api_key: str = Depends(verify_api_key)) -> User:
    """
    Get current authenticated user.

    This dependency depends on verify_api_key (nested dependencies).
    FastAPI resolves dependencies in order:
    1. verify_api_key runs first
    2. If successful, get_current_user receives the api_key
    3. get_current_user can use api_key to look up user

    Dependency chain:
        verify_api_key → get_current_user → path operation
    """
    print("  [Dependency] Getting current user from API key")

    # In production, look up user by API key in database
    return User(user_id=1, username="alice", role="user")


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Require admin role.

    Another level of nested dependency.
    Dependency chain:
        verify_api_key → get_current_user → require_admin → path operation

    This shows how you can build complex authorization logic
    from simple, reusable dependencies.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )

    return current_user


# =============================================================================
# Example 3: Common Query Parameters (Pagination, Filtering)
# =============================================================================

class PaginationParams:
    """
    Reusable pagination parameters.

    Can be a class! FastAPI will call it and use the result.
    This is cleaner than repeating skip/limit in every endpoint.
    """

    def __init__(
        self,
        skip: int = Query(0, ge=0, description="Number of items to skip"),
        limit: int = Query(10, ge=1, le=100, description="Number of items to return")
    ):
        self.skip = skip
        self.limit = limit

    def __repr__(self):
        return f"PaginationParams(skip={self.skip}, limit={self.limit})"


class FilterParams:
    """Reusable filtering parameters."""

    def __init__(
        self,
        search: Optional[str] = Query(None, description="Search query"),
        sort_by: str = Query("created_at", description="Field to sort by"),
        order: str = Query("desc", regex="^(asc|desc)$", description="Sort order")
    ):
        self.search = search
        self.sort_by = sort_by
        self.order = order


# =============================================================================
# Example 4: Dependency with Configuration
# =============================================================================

class Settings:
    """Application settings (in production, use Pydantic Settings)."""
    APP_NAME: str = "FastAPI App"
    DEBUG: bool = True
    MAX_ITEMS_PER_REQUEST: int = 100


settings = Settings()


def get_settings() -> Settings:
    """
    Settings dependency.

    Makes settings easily mockable for testing.
    Can be replaced with different settings in test environment.
    """
    return settings


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title="Dependency Injection Patterns")


@app.get("/")
def root():
    """API information."""
    return {
        "message": "Dependency Injection Patterns",
        "endpoints": {
            "with_db": "/users/{user_id} - Database session dependency",
            "authenticated": "/me - Authentication dependency",
            "admin_only": "/admin - Authorization dependency",
            "paginated": "/items - Pagination dependencies",
            "with_settings": "/info - Settings dependency"
        }
    }


@app.get("/users/{user_id}")
def get_user(
    user_id: int,
    db: Database = Depends(get_db)  # Inject database session
):
    """
    Endpoint using database dependency.

    The database session is:
    1. Connected before this function runs
    2. Passed to this function
    3. Disconnected after this function returns

    Test:
        curl http://localhost:8000/users/1

    Watch the logs to see dependency lifecycle.
    """
    print(f"[Path Operation] Getting user {user_id}")

    user = db.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return {"user_id": user_id, "username": user}


@app.get("/me")
def get_me(
    current_user: User = Depends(get_current_user)  # Inject authenticated user
):
    """
    Endpoint requiring authentication.

    Dependency chain automatically:
    1. Verifies API key (verify_api_key)
    2. Gets user (get_current_user)
    3. Passes user to path operation

    Test:
        # Without API key (fails)
        curl http://localhost:8000/me

        # With valid API key (succeeds)
        curl -H "X-API-Key: secret-api-key-123" http://localhost:8000/me
    """
    return {
        "message": f"Hello, {current_user.username}!",
        "user": current_user
    }


@app.get("/admin")
def admin_only(
    admin: User = Depends(require_admin)  # Inject and verify admin user
):
    """
    Endpoint requiring admin role.

    Dependency chain:
    1. Verifies API key
    2. Gets user
    3. Checks admin role
    4. Passes user to path operation

    Test:
        curl -H "X-API-Key: secret-api-key-123" http://localhost:8000/admin

    Note: In this example, the user has role="user", so this will fail.
    To test success, modify User creation in get_current_user to have role="admin".
    """
    return {
        "message": "Admin access granted",
        "admin": admin
    }


@app.get("/items")
def list_items(
    pagination: PaginationParams = Depends(),  # Class dependency
    filters: FilterParams = Depends()  # Another class dependency
):
    """
    Endpoint with pagination and filtering dependencies.

    Both pagination and filters are injected as dependencies.
    This keeps the function signature clean and reuses common parameters.

    Test:
        # Default pagination
        curl http://localhost:8000/items

        # Custom pagination
        curl "http://localhost:8000/items?skip=10&limit=20"

        # With filtering
        curl "http://localhost:8000/items?search=foo&sort_by=name&order=asc"
    """
    # Mock items
    all_items = [{"id": i, "name": f"Item {i}"} for i in range(100)]

    # Apply filtering (simplified)
    filtered = all_items
    if filters.search:
        filtered = [
            item for item in filtered
            if filters.search.lower() in item["name"].lower()
        ]

    # Apply pagination
    paginated = filtered[pagination.skip:pagination.skip + pagination.limit]

    return {
        "items": paginated,
        "total": len(filtered),
        "pagination": str(pagination),
        "filters": {
            "search": filters.search,
            "sort_by": filters.sort_by,
            "order": filters.order
        }
    }


@app.get("/info")
def get_info(
    settings: Settings = Depends(get_settings)  # Inject settings
):
    """
    Endpoint using settings dependency.

    Settings can be easily mocked for testing by overriding the dependency.

    Test:
        curl http://localhost:8000/info
    """
    return {
        "app_name": settings.APP_NAME,
        "debug": settings.DEBUG,
        "max_items": settings.MAX_ITEMS_PER_REQUEST
    }


# =============================================================================
# Example 5: Type Annotations for Cleaner Code (Python 3.9+)
# =============================================================================

# You can use Annotated to make dependency injection cleaner
# This is especially useful when the same dependency is used many times

# Define type aliases for common dependencies
DbSession = Annotated[Database, Depends(get_db)]
CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_admin)]
Pagination = Annotated[PaginationParams, Depends()]


@app.get("/modern/users/{user_id}")
def get_user_modern(
    user_id: int,
    db: DbSession,  # Much cleaner than Depends(get_db)
    current_user: CurrentUser  # Cleaner than Depends(get_current_user)
):
    """
    Modern syntax using Annotated type aliases.

    Same functionality as /users/{user_id}, but cleaner code.
    The dependency injection works exactly the same way.
    """
    print(f"[Path Operation] User {current_user.username} getting user {user_id}")

    user = db.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return {"user_id": user_id, "username": user}


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*70)
    print("Dependency Injection Patterns")
    print("="*70)
    print("\nKey Concepts Demonstrated:")
    print("  1. Database session management (setup/cleanup)")
    print("  2. Nested dependencies (auth chain)")
    print("  3. Class-based dependencies (pagination)")
    print("  4. Configuration injection (settings)")
    print("  5. Modern Annotated syntax")
    print("\nWatch the console logs to see dependency lifecycle!")
    print("="*70 + "\n")

    uvicorn.run(
        "dependency_injection:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
