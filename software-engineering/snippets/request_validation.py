"""
Request Validation Patterns with Pydantic

Demonstrates comprehensive input validation using Pydantic models in FastAPI:
1. Basic field validation (types, constraints)
2. Custom validators
3. Field-level vs model-level validation
4. Advanced validation patterns
5. Nested models

Key Concepts:
- FastAPI uses Pydantic for automatic request validation
- Validation happens before path operation executes (fail fast)
- Invalid requests return 422 with detailed error messages
- Pydantic models provide type safety and documentation

Run:
    uvicorn request_validation:app --reload
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, date
from enum import Enum

from fastapi import FastAPI, HTTPException, Body
from pydantic import (
    BaseModel,
    Field,
    EmailStr,
    HttpUrl,
    validator,
    root_validator,
    constr,
    conint,
    confloat
)


# =============================================================================
# Example 1: Basic Field Validation
# =============================================================================

class Priority(str, Enum):
    """Enum for valid priority values."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskCreate(BaseModel):
    """
    Task creation with basic field validation.

    Pydantic validators:
    - Field(...): Mark field as required
    - min_length, max_length: String length constraints
    - ge, gt, le, lt: Numeric comparisons (greater/less than)
    - regex: Pattern matching
    - example: Documentation example
    """

    title: str = Field(
        ...,  # ... means required (equivalent to Field(required=True))
        min_length=1,
        max_length=200,
        description="Task title",
        example="Deploy new feature"
    )

    description: Optional[str] = Field(
        None,  # Optional field with default None
        max_length=1000,
        description="Detailed task description"
    )

    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Task priority level"
    )

    due_date: Optional[date] = Field(
        None,
        description="Task due date"
    )

    estimated_hours: Optional[float] = Field(
        None,
        ge=0,  # Greater than or equal to 0
        le=1000,  # Less than or equal to 1000
        description="Estimated hours to complete"
    )

    tags: List[str] = Field(
        default=[],
        max_length=10,  # Max 10 tags
        description="Task tags"
    )


# =============================================================================
# Example 2: Custom Field Types (Constrained Types)
# =============================================================================

class UserCreate(BaseModel):
    """
    User creation with constrained types.

    Pydantic provides constrained types for common patterns:
    - constr: Constrained string (length, regex)
    - conint: Constrained integer (bounds)
    - confloat: Constrained float (bounds)
    - EmailStr: Valid email address
    - HttpUrl: Valid URL
    """

    # Username: 3-20 alphanumeric characters
    username: constr(min_length=3, max_length=20, regex=r'^[a-zA-Z0-9_]+$') = Field(
        ...,
        description="Username (alphanumeric and underscore only)",
        example="john_doe"
    )

    # Email validation
    email: EmailStr = Field(
        ...,
        description="Valid email address",
        example="john@example.com"
    )

    # Password: min 8 characters
    password: constr(min_length=8) = Field(
        ...,
        description="Password (min 8 characters)",
        example="SecurePass123!"
    )

    # Age: 13-120
    age: conint(ge=13, le=120) = Field(
        ...,
        description="User age (must be 13 or older)",
        example=25
    )

    # Website URL (optional)
    website: Optional[HttpUrl] = Field(
        None,
        description="Personal website URL",
        example="https://example.com"
    )


# =============================================================================
# Example 3: Custom Validators
# =============================================================================

class Event(BaseModel):
    """
    Event with custom validation logic.

    Custom validators allow complex validation rules beyond basic constraints.
    """

    name: str = Field(..., min_length=1, max_length=100)

    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")

    attendees: List[EmailStr] = Field(
        default=[],
        description="List of attendee emails"
    )

    max_attendees: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum number of attendees"
    )

    @validator('end_time')
    def validate_end_time(cls, v, values):
        """
        Field-level validator: Ensure end_time is after start_time.

        Args:
            cls: The model class
            v: The field value being validated (end_time)
            values: Dict of previously validated fields

        Returns:
            The validated value (or raises ValueError)

        Note: Validators run AFTER basic type validation passes.
        """
        if 'start_time' in values and v <= values['start_time']:
            raise ValueError('end_time must be after start_time')
        return v

    @validator('attendees')
    def validate_attendees(cls, v, values):
        """
        Validate attendee list against max_attendees.

        Demonstrates accessing other fields in validator.
        """
        max_attendees = values.get('max_attendees')
        if max_attendees and len(v) > max_attendees:
            raise ValueError(
                f'Cannot have more than {max_attendees} attendees'
            )
        return v

    @validator('name')
    def validate_name(cls, v):
        """
        Validate name doesn't contain forbidden words.

        Simple field validation example.
        """
        forbidden_words = ['spam', 'test123', 'delete']
        if any(word in v.lower() for word in forbidden_words):
            raise ValueError(f'Name contains forbidden words')
        return v.strip()  # Can also transform the value


# =============================================================================
# Example 4: Root Validators (Model-Level Validation)
# =============================================================================

class Discount(BaseModel):
    """
    Discount with model-level validation.

    Root validators check relationships between multiple fields.
    """

    code: str = Field(..., min_length=1, max_length=20)

    # Either percentage OR fixed amount, not both
    percentage: Optional[confloat(ge=0, le=100)] = None
    fixed_amount: Optional[confloat(ge=0)] = None

    # Minimum purchase amount to qualify
    min_purchase: Optional[confloat(ge=0)] = None
    max_discount: Optional[confloat(ge=0)] = None

    @root_validator
    def validate_discount_type(cls, values):
        """
        Model-level validation: Ensure exactly one discount type is set.

        Root validators receive ALL field values and can validate
        relationships between fields.

        Args:
            cls: The model class
            values: Dict of all field values

        Returns:
            The values dict (or raises ValueError)
        """
        percentage = values.get('percentage')
        fixed_amount = values.get('fixed_amount')

        # Must have exactly one discount type
        if percentage is None and fixed_amount is None:
            raise ValueError('Must specify either percentage or fixed_amount')

        if percentage is not None and fixed_amount is not None:
            raise ValueError('Cannot specify both percentage and fixed_amount')

        # If percentage discount, max_discount makes sense
        # If fixed discount, max_discount is redundant
        if fixed_amount is not None and values.get('max_discount'):
            raise ValueError('max_discount not applicable to fixed amount discounts')

        return values


# =============================================================================
# Example 5: Nested Models
# =============================================================================

class Address(BaseModel):
    """Address component."""

    street: str = Field(..., min_length=1, max_length=200)
    city: str = Field(..., min_length=1, max_length=100)
    state: str = Field(..., min_length=2, max_length=2)  # US state code
    zip_code: constr(regex=r'^\d{5}(-\d{4})?$') = Field(
        ...,
        description="US ZIP code (12345 or 12345-6789)"
    )

    @validator('state')
    def validate_state(cls, v):
        """Ensure state code is uppercase."""
        return v.upper()


class ContactInfo(BaseModel):
    """Contact information."""

    email: EmailStr
    phone: constr(regex=r'^\+?1?\d{10,15}$') = Field(
        ...,
        description="Phone number (10-15 digits)",
        example="+11234567890"
    )


class OrderItem(BaseModel):
    """Individual order item."""

    product_id: int = Field(..., gt=0)
    quantity: int = Field(..., ge=1, le=100)
    unit_price: float = Field(..., gt=0)


class Order(BaseModel):
    """
    Order with nested models.

    Demonstrates:
    - Nested single model (shipping_address, contact)
    - Nested list of models (items)
    - Computed fields (@property or validators)
    """

    order_id: str = Field(..., min_length=1)

    items: List[OrderItem] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Order items"
    )

    shipping_address: Address = Field(..., description="Shipping address")

    contact: ContactInfo = Field(..., description="Contact information")

    discount_code: Optional[str] = None

    @property
    def total_items(self) -> int:
        """Computed property: total number of items."""
        return sum(item.quantity for item in self.items)

    @property
    def subtotal(self) -> float:
        """Computed property: order subtotal."""
        return sum(item.quantity * item.unit_price for item in self.items)


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Request Validation Patterns",
    description="Comprehensive examples of Pydantic validation in FastAPI"
)


@app.post("/tasks", status_code=201)
def create_task(task: TaskCreate):
    """
    Create task with basic validation.

    Test valid request:
        curl -X POST http://localhost:8000/tasks \
             -H "Content-Type: application/json" \
             -d '{
               "title": "Deploy feature",
               "priority": "high",
               "estimated_hours": 5.5
             }'

    Test invalid requests:
        # Empty title (min_length violation)
        -d '{"title": ""}'

        # Invalid priority (not in enum)
        -d '{"title": "Task", "priority": "super-high"}'

        # Negative hours (ge=0 violation)
        -d '{"title": "Task", "estimated_hours": -1}'
    """
    return {
        "message": "Task created",
        "task": task.dict()
    }


@app.post("/users", status_code=201)
def create_user(user: UserCreate):
    """
    Create user with constrained types.

    Test valid:
        curl -X POST http://localhost:8000/users \
             -H "Content-Type: application/json" \
             -d '{
               "username": "john_doe",
               "email": "john@example.com",
               "password": "SecurePass123!",
               "age": 25
             }'

    Test invalid:
        # Invalid email
        -d '{"username": "john", "email": "not-an-email", ...}'

        # Username with invalid characters
        -d '{"username": "john-doe!", ...}'

        # Age too young
        -d '{"username": "john_doe", "email": "john@example.com", "password": "pass1234", "age": 10}'
    """
    return {
        "message": "User created",
        "username": user.username,
        "email": user.email
    }


@app.post("/events", status_code=201)
def create_event(event: Event):
    """
    Create event with custom validators.

    Test valid:
        curl -X POST http://localhost:8000/events \
             -H "Content-Type: application/json" \
             -d '{
               "name": "Team Meeting",
               "start_time": "2026-01-10T10:00:00",
               "end_time": "2026-01-10T11:00:00",
               "attendees": ["alice@example.com", "bob@example.com"],
               "max_attendees": 10
             }'

    Test invalid:
        # end_time before start_time
        -d '{"name": "Event", "start_time": "2026-01-10T10:00:00", "end_time": "2026-01-10T09:00:00"}'

        # Too many attendees
        -d '{"name": "Event", ..., "attendees": [...], "max_attendees": 2}'

        # Forbidden word in name
        -d '{"name": "spam event", ...}'
    """
    return {
        "message": "Event created",
        "event": event.dict()
    }


@app.post("/discounts", status_code=201)
def create_discount(discount: Discount):
    """
    Create discount with model-level validation.

    Test valid:
        # Percentage discount
        curl -X POST http://localhost:8000/discounts \
             -H "Content-Type: application/json" \
             -d '{
               "code": "SAVE20",
               "percentage": 20,
               "min_purchase": 50,
               "max_discount": 100
             }'

        # Fixed amount discount
        -d '{"code": "SAVE10", "fixed_amount": 10, "min_purchase": 30}'

    Test invalid:
        # Missing both discount types
        -d '{"code": "INVALID"}'

        # Both discount types specified
        -d '{"code": "INVALID", "percentage": 20, "fixed_amount": 10}'

        # max_discount with fixed_amount
        -d '{"code": "INVALID", "fixed_amount": 10, "max_discount": 50}'
    """
    return {
        "message": "Discount created",
        "discount": discount.dict()
    }


@app.post("/orders", status_code=201)
def create_order(order: Order):
    """
    Create order with nested models.

    Test valid:
        curl -X POST http://localhost:8000/orders \
             -H "Content-Type: application/json" \
             -d '{
               "order_id": "ORD-001",
               "items": [
                 {"product_id": 1, "quantity": 2, "unit_price": 19.99},
                 {"product_id": 2, "quantity": 1, "unit_price": 49.99}
               ],
               "shipping_address": {
                 "street": "123 Main St",
                 "city": "San Francisco",
                 "state": "CA",
                 "zip_code": "94102"
               },
               "contact": {
                 "email": "customer@example.com",
                 "phone": "+11234567890"
               }
             }'

    Note: Nested validation works recursively through all models.
    """
    return {
        "message": "Order created",
        "order_id": order.order_id,
        "total_items": order.total_items,
        "subtotal": order.subtotal
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*70)
    print("Request Validation Patterns")
    print("="*70)
    print("\nValidation Features Demonstrated:")
    print("  1. Basic field constraints (length, range)")
    print("  2. Constrained types (constr, conint, EmailStr, HttpUrl)")
    print("  3. Custom field validators (@validator)")
    print("  4. Model-level validators (@root_validator)")
    print("  5. Nested model validation")
    print("\nTry invalid requests to see detailed error messages!")
    print("="*70 + "\n")

    uvicorn.run(
        "request_validation:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
