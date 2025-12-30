# Python Dependency Management in Docker

Comparison of different approaches for managing Python dependencies in Docker.

## Quick Comparison

| Tool | Speed | Reproducibility | Complexity | Image Size | Use Case |
|------|-------|----------------|------------|------------|----------|
| **pip + requirements.txt** | Slow | ❌ Poor | ✅ Simple | Small | Quick prototypes, legacy |
| **pip + requirements.txt (pinned)** | Slow | ⚠️ Medium | ✅ Simple | Small | Simple projects |
| **uv** | ✅ Very Fast | ✅ Excellent | Medium | Small | **Recommended for new projects** |
| **Poetry** | Medium | ✅ Excellent | Medium | Small | Teams already using Poetry |
| **pip-tools** | Slow | ✅ Excellent | Medium | Small | Alternative to Poetry |

## The Traditional Approach: requirements.txt

### Basic Pattern

```dockerfile
FROM python:3.11-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### Why It's Common

✅ **Simplicity** - One file, everyone understands it
✅ **No extra tools** - pip is built-in
✅ **Layer caching** - Easy to optimize
✅ **Universal** - Works everywhere

### Problems

❌ **Non-reproducible** - Different installs can get different versions
❌ **Slow** - pip is slow at resolving dependencies
❌ **Dependency conflicts** - pip's resolver isn't great
❌ **No lock file** - Can't guarantee exact environment

### Example Issue

```txt
# requirements.txt
flask==3.0.0
```

**Problem**: Flask depends on werkzeug. Today `pip install` might get werkzeug 3.0.1,
tomorrow it might get 3.0.2 (if released). Your builds are non-reproducible!

### Workaround: Pin Everything

```bash
# Generate fully pinned requirements
pip freeze > requirements.txt
```

```txt
# requirements.txt (pinned)
flask==3.0.0
werkzeug==3.0.1
jinja2==3.1.2
click==8.1.7
...
```

**Better**, but pip is still slow and you lose the distinction between direct and transitive dependencies.

---

## Modern Approach 1: uv (Recommended)

### Why uv?

- 🚀 **10-100x faster** than pip (written in Rust)
- 🔒 **Lock file** (`uv.lock`) for reproducibility
- 📦 **Drop-in replacement** for pip
- 🎯 **Simple** - Less complex than Poetry
- 💾 **Small** - Doesn't add much to image size

### Setup

```bash
# On your machine (not in Docker)
uv init  # Creates pyproject.toml
uv add flask fastapi  # Add dependencies
uv lock  # Creates uv.lock (commit this!)
```

### Dockerfile Pattern

See [Dockerfile.python-uv](./Dockerfile.python-uv) for complete example.

```dockerfile
FROM python:3.11-slim as builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-dev --locked

FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["python", "app.py"]
```

### Advantages

✅ **Fast builds** - Installs are 10-100x faster
✅ **Reproducible** - uv.lock guarantees exact versions
✅ **Simple** - Similar to pip workflow
✅ **Layer caching** - Works perfectly
✅ **No bloat** - Final image doesn't include uv

### When to Use

- ✅ New projects
- ✅ When build speed matters
- ✅ When you want reproducibility without complexity
- ✅ CI/CD pipelines (faster builds = cheaper)

---

## Modern Approach 2: Poetry

### Why Poetry?

- 🔒 **Lock file** (`poetry.lock`) for reproducibility
- 📚 **Mature** - Been around since 2018, stable
- 🎯 **Dev dependencies** - Separate dev and production deps
- 🛠️ **Full tool** - Handles packaging, publishing, versioning

### Setup

```bash
# On your machine
poetry init
poetry add flask fastapi
poetry add --group dev pytest black
poetry lock  # Creates poetry.lock (commit this!)
```

### Dockerfile Pattern

See [Dockerfile.python-poetry](./Dockerfile.python-poetry) for complete example.

```dockerfile
FROM python:3.11-slim as builder
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_CREATE=false

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --no-dev

FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["python", "app.py"]
```

### Advantages

✅ **Reproducible** - poetry.lock guarantees versions
✅ **Mature ecosystem** - Lots of tooling support
✅ **Dev/prod separation** - `poetry add --group dev`
✅ **Good dependency resolution** - Better than pip

### Disadvantages

⚠️ **Slower than uv** - Dependency resolution can be slow
⚠️ **More complex** - Learning curve, more configuration
⚠️ **Heavier** - Need to install Poetry in builder

### When to Use

- ✅ Team already uses Poetry
- ✅ Need advanced features (publishing packages, etc.)
- ✅ Build speed not critical

---

## Modern Approach 3: pip-tools

### Why pip-tools?

- 🔒 **Lock file** (`requirements.lock`)
- 📦 **Uses pip** - Familiar tool
- 🎯 **Simple** - Just adds locking to pip

### Setup

```bash
# requirements.in (your actual dependencies)
flask==3.0.0
fastapi==0.109.0

# Generate lock file
pip-compile requirements.in
# Creates requirements.txt with all pinned dependencies
```

### Dockerfile

```dockerfile
FROM python:3.11-slim as builder
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["python", "app.py"]
```

### Advantages

✅ **Simple** - Just pip with locking
✅ **Reproducible** - Pinned versions
✅ **Familiar** - Uses requirements.txt format

### Disadvantages

⚠️ **Still slow** - Uses pip under the hood
⚠️ **Manual workflow** - Need to run pip-compile

### When to Use

- ✅ Want reproducibility without new tools
- ✅ Transitioning from plain requirements.txt

---

## Layer Caching Comparison

### requirements.txt

```dockerfile
COPY requirements.txt .     # Changes rarely → good caching
RUN pip install -r requirements.txt
COPY . .                    # Changes often → doesn't invalidate install layer ✅
```

### uv

```dockerfile
COPY pyproject.toml uv.lock .  # Changes rarely → good caching
RUN uv pip install --system --locked
COPY . .                       # Changes often → doesn't invalidate install layer ✅
```

### Poetry (Proper Way)

```dockerfile
COPY pyproject.toml poetry.lock .  # Changes rarely → good caching
RUN poetry install --no-root --no-dev
COPY . .                           # Changes often → doesn't invalidate install layer ✅
RUN poetry install --only-root     # Install project itself
```

### Poetry (Wrong Way - Don't Do This)

```dockerfile
COPY . .                           # ❌ Copies everything first
RUN poetry install --no-dev        # ❌ Cache invalidated on ANY file change
```

This breaks caching! Any code change invalidates the entire dependency install.

---

## Build Speed Comparison

Real-world example: Installing FastAPI + dependencies

| Tool | First Build | Cached Build | Lock Generation |
|------|-------------|--------------|-----------------|
| pip + requirements.txt | 45s | 2s | - |
| pip + requirements.txt (pinned) | 45s | 2s | Manual |
| uv | 5s | 0.5s | 1s |
| Poetry | 60s | 2s | 30s |
| pip-tools | 45s | 2s | 45s |

**Winner: uv** (10x faster first build, 4x faster cached build)

---

## Recommendations

### For New Projects

**Use uv** - It's the best of all worlds:
- Fast (10-100x faster than pip)
- Simple (similar workflow to pip)
- Reproducible (lock file)
- Modern (actively developed)

```bash
# Get started
uv init
uv add your-dependencies
uv lock
# Use Dockerfile.python-uv
```

### For Existing Poetry Projects

**Keep Poetry** - No need to migrate if it's working:
- You already have poetry.lock
- Team knows the tool
- Just optimize your Dockerfile (see Dockerfile.python-poetry)

### For Simple Scripts

**Plain requirements.txt is fine**:
- Quick prototypes
- Single-file scripts
- Learning projects
- When reproducibility doesn't matter

### For Production ML Systems

**Use uv or Poetry**:
- ML dependencies are complex (tensorflow, pytorch, etc.)
- Reproducibility is critical
- Build speed matters in CI/CD
- Lock files prevent "works on my machine" issues

---

## Migration Paths

### From requirements.txt to uv

```bash
# Convert existing requirements.txt
uv pip compile requirements.txt -o requirements.lock

# Or start fresh with pyproject.toml
uv init
uv add $(cat requirements.txt | grep -v "^#" | grep -v "^$")
uv lock
```

### From Poetry to uv

```bash
# Export from poetry
poetry export -f requirements.txt --output requirements.txt

# Import to uv
uv init
uv add $(cat requirements.txt | grep -v "^#" | grep -v "^$")
uv lock
```

### From requirements.txt to Poetry

```bash
poetry init
poetry add $(cat requirements.txt | grep -v "^#" | grep -v "^$")
poetry lock
```

---

## Best Practices (Regardless of Tool)

### 1. Always Use Lock Files in Production

```dockerfile
# ❌ Bad - non-reproducible
RUN pip install flask

# ✅ Good - locked versions
COPY requirements.txt .
RUN pip install -r requirements.txt
```

### 2. Leverage Layer Caching

```dockerfile
# ✅ Good order
COPY pyproject.toml uv.lock .
RUN uv pip install --system --locked
COPY . .

# ❌ Bad order
COPY . .
RUN uv pip install --system --locked
```

### 3. Use Multi-Stage Builds

```dockerfile
# Builder stage: includes build tools
FROM python:3.11 as builder
RUN install dependencies...

# Runtime stage: minimal, no build tools
FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages ...
```

### 4. Pin Base Image Versions

```dockerfile
# ❌ Bad - python:3.11 tag can change
FROM python:3.11

# ✅ Better - specific version
FROM python:3.11.7-slim

# ✅ Best - digest for absolute reproducibility
FROM python:3.11.7-slim@sha256:abc123...
```

---

## Summary

| Scenario | Recommendation | Reason |
|----------|---------------|---------|
| New project | **uv** | Fast, simple, reproducible |
| Existing Poetry project | **Poetry** | Don't fix what's not broken |
| Quick prototype | **requirements.txt** | Simple, fast to set up |
| Team project | **uv or Poetry** | Reproducibility matters |
| CI/CD pipeline | **uv** | Fast builds = lower costs |
| ML production | **uv or Poetry** | Complex deps need locking |

**The future is probably uv** - It's the newest, fastest, and strikes the best balance between simplicity and features.

---

## Examples in This Directory

- [Dockerfile.python](./Dockerfile.python) - Traditional with requirements.txt
- [Dockerfile.python-uv](./Dockerfile.python-uv) - Modern with uv (recommended)
- [Dockerfile.python-poetry](./Dockerfile.python-poetry) - Modern with Poetry
- [Dockerfile.ml](./Dockerfile.ml) - ML-specific (uses requirements.txt for simplicity)

Pick the one that matches your project's needs!
