# Python Class Anatomy

A comprehensive guide to understanding Python's object-oriented features, from decorators and dunder methods to modern data containers and class architectures.

## Table of Contents

1. [Class Basics](#class-basics)
2. [Method Types & Decorators](#method-types--decorators)
3. [Dunder Methods (Magic Methods)](#dunder-methods-magic-methods)
4. [Properties & Descriptors](#properties--descriptors)
5. [Data Containers](#data-containers)
6. [Class Architectures](#class-architectures)
7. [Advanced Patterns](#advanced-patterns)

---

## Class Basics

### Minimal Class

```python
class Point:
    """A point in 2D space."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
```

**Anatomy**:
- `class` keyword declares the class
- `Point` is the class name (PascalCase convention)
- `__init__` is the constructor (initializer)
- `self` is the instance reference (like `this` in other languages)

### Class vs Instance Attributes

```python
class Counter:
    # Class attribute (shared by all instances)
    total_count = 0

    def __init__(self, name):
        # Instance attribute (unique to each instance)
        self.name = name
        self.count = 0
        Counter.total_count += 1

    def increment(self):
        self.count += 1

# Usage
c1 = Counter("first")
c2 = Counter("second")

c1.increment()
print(c1.count)           # 1
print(c2.count)           # 0
print(Counter.total_count)  # 2 (class attribute)
```

**Key Difference**:
- **Class attributes**: Defined at class level, shared across all instances
- **Instance attributes**: Defined in `__init__`, unique to each instance

---

## Method Types & Decorators

### Instance Methods

The default method type. Receives `self` as the first parameter.

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):  # Instance method
        return 3.14159 * self.radius ** 2

circle = Circle(5)
print(circle.area())  # 78.53975
```

### @staticmethod

A method that doesn't access instance or class state. Just a function scoped to the class.

```python
class MathUtils:
    @staticmethod
    def add(x, y):
        """Doesn't need self or cls - just a utility function."""
        return x + y

# Can call on class or instance
print(MathUtils.add(3, 4))        # 7
m = MathUtils()
print(m.add(3, 4))                # 7 (works but not idiomatic)
```

**When to use**:
- Utility functions related to the class but don't need instance/class data
- Grouping related functions under a class namespace
- Replacing module-level functions when you want OOP organization

### @classmethod

A method that receives the class itself as the first parameter (`cls`).

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def from_string(cls, date_string):
        """Alternative constructor (factory method)."""
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)

    @classmethod
    def today(cls):
        """Another factory method."""
        import datetime
        now = datetime.date.today()
        return cls(now.year, now.month, now.day)

# Usage
d1 = Date(2024, 12, 25)
d2 = Date.from_string("2024-12-25")
d3 = Date.today()
```

**When to use**:
- Alternative constructors (factory methods)
- Methods that need to modify class state
- Methods that work with subclasses polymorphically

**classmethod vs staticmethod**:
```python
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients

    @classmethod
    def margherita(cls):
        return cls(['mozzarella', 'tomatoes'])

    @staticmethod
    def circle_area(radius):
        return 3.14 * radius ** 2

# classmethod gets the class, can instantiate
pizza = Pizza.margherita()  # Returns Pizza instance

# staticmethod doesn't get class, just utility
area = Pizza.circle_area(12)  # Returns float
```

### @property

Turns a method into a computed attribute (getter).

```python
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def celsius(self):
        """Getter for celsius."""
        return self._celsius

    @celsius.setter
    def celsius(self, value):
        """Setter for celsius."""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value

    @property
    def fahrenheit(self):
        """Computed property."""
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value):
        self._celsius = (value - 32) * 5/9

# Usage
temp = Temperature(25)
print(temp.celsius)      # 25 (looks like attribute access)
print(temp.fahrenheit)   # 77.0

temp.celsius = 30        # Uses setter
temp.fahrenheit = 100    # Sets celsius via fahrenheit setter
```

**Benefits**:
- Clean API (no getter/setter method names)
- Validation logic
- Computed attributes
- Can add logic later without changing client code

### Custom Decorators

```python
import functools
import time

def timer(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper

def validate_positive(func):
    """Decorator to validate arguments are positive."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args[1:]:  # Skip self
            if isinstance(arg, (int, float)) and arg < 0:
                raise ValueError(f"Expected positive number, got {arg}")
        return func(*args, **kwargs)
    return wrapper

class Calculator:
    @timer
    @validate_positive
    def slow_multiply(self, x, y):
        time.sleep(0.1)
        return x * y

calc = Calculator()
result = calc.slow_multiply(5, 10)  # slow_multiply took 0.1001s
```

**Common Decorators**:
- `@functools.wraps` - Preserves function metadata
- `@functools.lru_cache` - Memoization
- `@functools.cached_property` - Cached property
- `@abc.abstractmethod` - Abstract method marker

---

## Dunder Methods (Magic Methods)

Methods with double underscores (dunders) that Python calls implicitly.

### Object Lifecycle

```python
class Resource:
    def __new__(cls, *args, **kwargs):
        """Controls object creation (rarely overridden)."""
        print(f"Creating instance of {cls.__name__}")
        instance = super().__new__(cls)
        return instance

    def __init__(self, name):
        """Initialize the instance."""
        print(f"Initializing {name}")
        self.name = name

    def __del__(self):
        """Called when object is garbage collected (unreliable)."""
        print(f"Deleting {self.name}")

r = Resource("data")
# Creating instance of Resource
# Initializing data
del r
# Deleting data
```

**Key difference**:
- `__new__`: Creates the object (returns instance)
- `__init__`: Initializes the object (returns None)

### String Representation

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        """Unambiguous representation for developers."""
        return f"Point({self.x}, {self.y})"

    def __str__(self):
        """Readable representation for users."""
        return f"({self.x}, {self.y})"

p = Point(3, 4)
print(repr(p))  # Point(3, 4) - should be eval()-able
print(str(p))   # (3, 4) - human readable
print(p)        # (3, 4) - uses __str__
```

**Best Practice**:
- `__repr__`: Debugging, should be unambiguous
- `__str__`: Display to users, can be pretty
- If only one, implement `__repr__`

### Comparison Operators

```python
from functools import total_ordering

@total_ordering
class Version:
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __eq__(self, other):
        """Equality: =="""
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) == \
               (other.major, other.minor, other.patch)

    def __lt__(self, other):
        """Less than: <"""
        if not isinstance(other, Version):
            return NotImplemented
        return (self.major, self.minor, self.patch) < \
               (other.major, other.minor, other.patch)

    # @total_ordering provides: <=, >, >=, !=

v1 = Version(1, 2, 3)
v2 = Version(1, 3, 0)
print(v1 < v2)   # True
print(v1 == v2)  # False
print(v1 >= v2)  # False (provided by @total_ordering)
```

**Comparison Methods**:
- `__eq__`: `==`
- `__ne__`: `!=` (defaults to `not __eq__`)
- `__lt__`: `<`
- `__le__`: `<=`
- `__gt__`: `>`
- `__ge__`: `>=`

**Tip**: Use `@functools.total_ordering` to generate missing comparisons from `__eq__` and one ordering method.

### Arithmetic Operators

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        """Addition: +"""
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        """Subtraction: -"""
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        """Multiplication: * (scalar)"""
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        """Right multiplication: scalar * vector"""
        return self.__mul__(scalar)

    def __abs__(self):
        """Absolute value: abs()"""
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1 + v2)      # Vector(4, 6)
print(v1 * 3)       # Vector(3, 6)
print(3 * v1)       # Vector(3, 6) - uses __rmul__
print(abs(v1))      # 2.23606797749979
```

**Arithmetic Methods**:
- `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__floordiv__`, `__mod__`, `__pow__`
- `__r*__` versions for right-hand operations
- `__i*__` versions for in-place operations (`+=`, etc.)

### Container Methods

```python
class Playlist:
    def __init__(self, songs):
        self._songs = songs

    def __len__(self):
        """len(playlist)"""
        return len(self._songs)

    def __getitem__(self, index):
        """playlist[index] or playlist[start:stop]"""
        return self._songs[index]

    def __setitem__(self, index, value):
        """playlist[index] = value"""
        self._songs[index] = value

    def __delitem__(self, index):
        """del playlist[index]"""
        del self._songs[index]

    def __contains__(self, item):
        """item in playlist"""
        return item in self._songs

    def __iter__(self):
        """for song in playlist"""
        return iter(self._songs)

    def __reversed__(self):
        """reversed(playlist)"""
        return reversed(self._songs)

playlist = Playlist(["Song A", "Song B", "Song C"])

print(len(playlist))           # 3
print(playlist[1])             # Song B
print("Song A" in playlist)    # True

for song in playlist:
    print(song)

for song in reversed(playlist):
    print(song)
```

### Context Managers

```python
class DatabaseConnection:
    def __init__(self, db_name):
        self.db_name = db_name
        self.connection = None

    def __enter__(self):
        """Called when entering 'with' block."""
        print(f"Opening connection to {self.db_name}")
        self.connection = f"Connection to {self.db_name}"
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Called when exiting 'with' block."""
        print(f"Closing connection to {self.db_name}")
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        return False  # Don't suppress exceptions

# Usage
with DatabaseConnection("mydb") as conn:
    print(f"Using {conn}")
    # raise ValueError("Oops!")  # __exit__ will still be called

# Opening connection to mydb
# Using Connection to mydb
# Closing connection to mydb
```

### Callable Objects

```python
class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        """Makes instance callable like a function."""
        return x * self.factor

double = Multiplier(2)
triple = Multiplier(3)

print(double(5))    # 10
print(triple(5))    # 15
print(callable(double))  # True
```

**Use Cases**:
- Stateful functions
- Function factories
- Decorators implemented as classes

### Attribute Access

```python
class DynamicAttributes:
    def __init__(self):
        self._data = {}

    def __getattr__(self, name):
        """Called when attribute doesn't exist."""
        print(f"Getting {name}")
        return self._data.get(name, f"No attribute {name}")

    def __setattr__(self, name, value):
        """Called on all attribute assignments."""
        print(f"Setting {name} = {value}")
        if name.startswith('_'):
            # Allow private attributes normally
            super().__setattr__(name, value)
        else:
            # Store public attributes in _data
            self._data[name] = value

    def __delattr__(self, name):
        """Called when deleting attribute."""
        print(f"Deleting {name}")
        if name in self._data:
            del self._data[name]

obj = DynamicAttributes()
obj.x = 10          # Setting x = 10
print(obj.x)        # Getting x -> 10
del obj.x           # Deleting x
print(obj.y)        # Getting y -> No attribute y
```

**Attribute Access Methods**:
- `__getattr__`: Called when attribute not found
- `__getattribute__`: Called on **every** attribute access (be careful!)
- `__setattr__`: Called on attribute assignment
- `__delattr__`: Called on attribute deletion

---

## Properties & Descriptors

### Property Decorator (Revisited)

```python
class Temperature:
    def __init__(self, kelvin):
        self.kelvin = kelvin

    @property
    def celsius(self):
        return self.kelvin - 273.15

    @celsius.setter
    def celsius(self, value):
        self.kelvin = value + 273.15

    @celsius.deleter
    def celsius(self):
        del self.kelvin

temp = Temperature(300)
print(temp.celsius)  # 26.85
temp.celsius = 25    # Sets kelvin to 298.15
del temp.celsius     # Deletes kelvin
```

### Descriptors (Advanced)

Descriptors control attribute access at the class level.

```python
class Validator:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value

    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to class attribute."""
        self.name = f"_{name}"

    def __get__(self, instance, owner):
        """Called when attribute is accessed."""
        if instance is None:
            return self
        return getattr(instance, self.name, None)

    def __set__(self, instance, value):
        """Called when attribute is set."""
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{value} is below minimum {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{value} exceeds maximum {self.max_value}")
        setattr(instance, self.name, value)

class Person:
    age = Validator(min_value=0, max_value=150)
    height = Validator(min_value=0)

    def __init__(self, age, height):
        self.age = age
        self.height = height

p = Person(25, 175)
print(p.age)      # 25
p.age = 30        # OK
# p.age = -5      # ValueError: -5 is below minimum 0
# p.age = 200     # ValueError: 200 exceeds maximum 150
```

**Descriptor Protocol**:
- `__get__(self, instance, owner)`: Attribute access
- `__set__(self, instance, value)`: Attribute assignment
- `__delete__(self, instance)`: Attribute deletion
- `__set_name__(self, owner, name)`: Called when descriptor assigned

**Built on descriptors**: `@property`, `@staticmethod`, `@classmethod`

---

## Data Containers

### dataclass (Python 3.7+)

Modern way to create data-holding classes.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Point:
    """Automatically generates __init__, __repr__, __eq__."""
    x: float
    y: float

    def distance_from_origin(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

p1 = Point(3, 4)
p2 = Point(3, 4)
print(p1)           # Point(x=3, y=4)
print(p1 == p2)     # True (auto-generated __eq__)

@dataclass(frozen=True)
class ImmutablePoint:
    """Immutable dataclass."""
    x: float
    y: float

ip = ImmutablePoint(1, 2)
# ip.x = 5  # FrozenInstanceError

@dataclass
class Student:
    name: str
    age: int
    grades: List[float] = field(default_factory=list)
    # NEVER: grades: List[float] = []  # Mutable default!

    def average_grade(self):
        return sum(self.grades) / len(self.grades) if self.grades else 0.0

s1 = Student("Alice", 20)
s2 = Student("Bob", 21, [85, 90, 95])
```

**dataclass Parameters**:
- `frozen=True`: Make immutable
- `order=True`: Generate comparison methods
- `kw_only=True`: Make all params keyword-only
- `slots=True`: Use `__slots__` (memory efficient)

### NamedTuple

Immutable, lightweight alternative to dataclass.

```python
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float

    def distance_from_origin(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Point(3, 4)
print(p.x, p.y)     # 3 4
print(p[0], p[1])   # 3 4 (also a tuple!)
# p.x = 5           # AttributeError (immutable)

# Unpacking works
x, y = p
```

**NamedTuple vs dataclass**:
- NamedTuple: Immutable, tuple-like, less memory
- dataclass: Mutable by default, more features, can add validation

### attrs (Third-party)

More powerful than dataclass, but requires installation.

```python
import attr

@attr.s(auto_attribs=True)
class Point:
    x: float = attr.ib(validator=attr.validators.instance_of(float))
    y: float = attr.ib(validator=attr.validators.instance_of(float))

    @y.validator
    def check_y(self, attribute, value):
        if value < 0:
            raise ValueError("y must be non-negative")
```

### Pydantic (Third-party)

Data validation using Python type hints.

```python
from pydantic import BaseModel, validator

class User(BaseModel):
    name: str
    age: int
    email: str

    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('age must be positive')
        return v

# Validates on instantiation
user = User(name="Alice", age=25, email="alice@example.com")
# user = User(name="Bob", age=-5, email="bob@example.com")  # ValidationError
```

**Comparison**:

| Feature | dataclass | NamedTuple | attrs | Pydantic |
|---------|-----------|------------|-------|----------|
| **Mutability** | Mutable | Immutable | Configurable | Mutable |
| **Validation** | Manual | Manual | Built-in | Built-in |
| **Performance** | Fast | Fastest | Fast | Slower |
| **JSON Support** | Manual | Manual | Yes | Excellent |
| **Type Checking** | Yes | Yes | Yes | Runtime |
| **External Dep** | No | No | Yes | Yes |

---

## Class Architectures

### Inheritance

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

dog = Dog("Buddy")
cat = Cat("Whiskers")
print(dog.speak())  # Buddy says Woof!
print(cat.speak())  # Whiskers says Meow!

# Check inheritance
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True
print(issubclass(Dog, Animal))  # True
```

**Method Resolution Order (MRO)**:
```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.method())    # "B" (B comes before C in MRO)
print(D.__mro__)     # (D, B, C, A, object)
```

### Composition (Preferred over Inheritance)

```python
class Engine:
    def start(self):
        return "Engine started"

    def stop(self):
        return "Engine stopped"

class Wheels:
    def rotate(self):
        return "Wheels rotating"

class Car:
    """Composition: Car HAS-A engine and wheels."""
    def __init__(self):
        self.engine = Engine()
        self.wheels = Wheels()

    def drive(self):
        return f"{self.engine.start()}, {self.wheels.rotate()}"

car = Car()
print(car.drive())  # Engine started, Wheels rotating
```

**Composition vs Inheritance**:
- **Inheritance**: "is-a" relationship (Dog is an Animal)
- **Composition**: "has-a" relationship (Car has an Engine)
- **Prefer composition**: More flexible, easier to test, looser coupling

### Mixins

Small classes that add specific functionality.

```python
class JSONMixin:
    """Adds JSON serialization."""
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class TimestampMixin:
    """Adds timestamp tracking."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from datetime import datetime
        self.created_at = datetime.now()

class User(JSONMixin, TimestampMixin):
    def __init__(self, name, email):
        super().__init__()
        self.name = name
        self.email = email

user = User("Alice", "alice@example.com")
print(user.to_json())  # {"name": "Alice", "email": "alice@example.com", ...}
print(user.created_at)
```

**Mixin Guidelines**:
- Small, focused functionality
- No state (or minimal state)
- Multiple inheritance friendly
- Name with "Mixin" suffix

### Abstract Base Classes (ABC)

Define interfaces that subclasses must implement.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        """Subclasses must implement."""
        pass

    @abstractmethod
    def perimeter(self):
        """Subclasses must implement."""
        pass

    def describe(self):
        """Concrete method."""
        return f"Area: {self.area()}, Perimeter: {self.perimeter()}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# shape = Shape()  # TypeError: Can't instantiate abstract class

rect = Rectangle(5, 3)
print(rect.describe())  # Area: 15, Perimeter: 16
```

### Protocols (Python 3.8+)

Structural subtyping (duck typing with type checking).

```python
from typing import Protocol

class Drawable(Protocol):
    """Any class with a draw() method."""
    def draw(self) -> str:
        ...

class Circle:
    def draw(self) -> str:
        return "Drawing circle"

class Square:
    def draw(self) -> str:
        return "Drawing square"

def render(shape: Drawable) -> None:
    """Accepts anything with draw() method."""
    print(shape.draw())

# Works without explicit inheritance
render(Circle())  # Drawing circle
render(Square())  # Drawing square
```

**Protocol vs ABC**:
- **ABC**: Nominal typing (explicit inheritance)
- **Protocol**: Structural typing (duck typing)
- **Use Protocol when**: You don't control the classes, want duck typing with type hints

---

## Advanced Patterns

### Singleton

```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True
```

### Factory Pattern

```python
class ShapeFactory:
    @staticmethod
    def create(shape_type, *args):
        shapes = {
            'circle': Circle,
            'rectangle': Rectangle,
            'triangle': Triangle
        }
        return shapes[shape_type](*args)

shape = ShapeFactory.create('circle', radius=5)
```

### Builder Pattern

```python
class QueryBuilder:
    def __init__(self):
        self._select = []
        self._from = None
        self._where = []

    def select(self, *fields):
        self._select.extend(fields)
        return self

    def from_table(self, table):
        self._from = table
        return self

    def where(self, condition):
        self._where.append(condition)
        return self

    def build(self):
        query = f"SELECT {', '.join(self._select)}"
        query += f" FROM {self._from}"
        if self._where:
            query += f" WHERE {' AND '.join(self._where)}"
        return query

# Fluent interface
query = (QueryBuilder()
    .select('name', 'age')
    .from_table('users')
    .where('age > 18')
    .where('active = true')
    .build())

print(query)
# SELECT name, age FROM users WHERE age > 18 AND active = true
```

### Slots (Memory Optimization)

```python
class Point:
    __slots__ = ('x', 'y')

    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(3, 4)
# p.z = 5  # AttributeError: 'Point' object has no attribute 'z'

# Memory comparison
import sys
class NormalPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

normal = NormalPoint(1, 2)
slotted = Point(1, 2)

print(sys.getsizeof(normal.__dict__))  # ~280 bytes
# Slotted classes don't have __dict__
```

**Benefits of __slots__**:
- ~50% memory reduction
- Faster attribute access
- Prevents typos (no dynamic attributes)

**Drawbacks**:
- No `__dict__` (can't add attributes dynamically)
- Inheritance complications
- No weak references (unless `__weakref__` in slots)

---

## Best Practices

### 1. Prefer Composition Over Inheritance

```python
# ❌ Bad: Deep inheritance hierarchy
class Vehicle:
    pass

class LandVehicle(Vehicle):
    pass

class Car(LandVehicle):
    pass

class SportsCar(Car):
    pass

# ✅ Good: Composition
class Engine:
    pass

class Wheels:
    pass

class Car:
    def __init__(self):
        self.engine = Engine()
        self.wheels = Wheels()
```

### 2. Use dataclasses for Data Containers

```python
# ❌ Verbose
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

# ✅ Concise
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
```

### 3. Implement `__repr__` First

```python
class Node:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Node({self.value!r})"

    # __str__ is optional

node = Node("hello")
print(node)          # Node('hello')
print([node])        # [Node('hello')] - works in containers
```

### 4. Use Properties for Computed Attributes

```python
# ❌ Methods for simple access
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def get_area(self):
        return 3.14159 * self.radius ** 2

# ✅ Property for computed attribute
class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return 3.14159 * self.radius ** 2

circle = Circle(5)
print(circle.area)  # Cleaner API
```

### 5. Avoid Mutable Default Arguments

```python
# ❌ Dangerous!
class Team:
    def __init__(self, members=[]):
        self.members = members

t1 = Team()
t2 = Team()
t1.members.append("Alice")
print(t2.members)  # ['Alice'] - Oops!

# ✅ Correct
class Team:
    def __init__(self, members=None):
        self.members = members if members is not None else []
```

---

## References

- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [Python Descriptor HowTo](https://docs.python.org/3/howto/descriptor.html)
- [PEP 557 – Data Classes](https://peps.python.org/pep-0557/)
- [PEP 544 – Protocols](https://peps.python.org/pep-0544/)
- [Real Python: Python's __init__() Method](https://realpython.com/python-class-constructor/)
- [Real Python: Inheritance and Composition](https://realpython.com/inheritance-composition-python/)
