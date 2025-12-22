# Python Build Systems & Backends

## Overview

Python build systems handle converting your project source code into distributable packages (wheels, sdists). The build backend is specified in `pyproject.toml` under the `[build-system]` section and determines how your package is built and what features are available.

```toml
[build-system]
requires = ["<backend>"]
build-backend = "<backend.module>"
```

## Why Build Backends Matter

1. **Package Discovery**: How the backend finds and includes your Python files
2. **Metadata Generation**: How project information is processed and included
3. **Build Features**: What build-time operations are supported (C extensions, data files, etc.)
4. **Developer Experience**: Error messages, configuration verbosity, build speed
5. **Compatibility**: Support for editable installs, different Python versions, platforms

## Common Build Backends

### 1. Setuptools

**Status**: Traditional, most widely used, highest compatibility

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["my_package"]  # Explicit package list
# OR
py-modules = []  # For dependency-only projects (no packages)

[tool.setuptools.package-data]
my_package = ["data/*.json"]
```

**Strengths**:
- Widest ecosystem compatibility and support
- Most forgiving about project structure
- Excellent documentation and community knowledge
- Handles complex build scenarios (C extensions, data files, etc.)
- Best for projects with non-standard layouts
- Can be configured to work as dependency manager only

**Weaknesses**:
- Verbose configuration for simple projects
- Slower build times compared to modern alternatives
- Legacy baggage from setup.py era
- More configuration options can be overwhelming

**Best For**:
- Projects with C extensions or complex builds
- Learning repositories that aren't traditional packages
- Maximum compatibility with existing tools
- Projects that need fine-grained control over packaging

**Example Use Cases**:
- Scientific computing packages with native extensions
- Legacy projects migrating from setup.py
- Non-package repos using pyproject.toml for dependency management

---

### 2. Hatchling

**Status**: Modern, fast, default for `uv`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

# Automatic package discovery by default
# Looks for: src/<package>/ or <package>/ directories

[tool.hatch.build.targets.wheel]
packages = ["src/my_package"]  # Override if needed
```

**Strengths**:
- Fast build times
- Modern, clean design (no legacy baggage)
- Smart automatic package discovery
- Excellent default configurations
- Good error messages
- Growing ecosystem momentum

**Weaknesses**:
- Stricter about project structure (expects standard layout)
- Less forgiving for non-package projects
- Smaller ecosystem than setuptools (though growing rapidly)
- Fewer complex build scenario examples

**Best For**:
- New pure-Python projects
- Projects following standard layout (src/ or flat)
- When build speed matters
- Modern greenfield projects

**Package Discovery Rules**:
1. Looks for `src/<name>/` where `<name>` matches project name
2. Looks for `<name>/` at root
3. Errors if ambiguous or not found

---

### 3. Flit / Flit-Core

**Status**: Minimalist, simple, opinionated

```toml
[build-system]
requires = ["flit_core>=3.2"]
build-backend = "flit_core.buildapi"

# Minimal configuration needed
# Uses module docstring for description
# Uses __version__ variable for version
```

**Strengths**:
- Extremely simple for pure Python projects
- Minimal configuration required
- Uses Python conventions (docstrings, __version__)
- Fast and lightweight
- Opinionated defaults reduce decision fatigue

**Weaknesses**:
- Only for pure Python (no C extensions)
- Limited customization options
- Less flexibility for complex projects
- Opinionated structure requirements

**Best For**:
- Simple pure-Python libraries
- Projects that fit the opinionated structure
- Developers who want minimal configuration
- Single-module packages

**Special Features**:
- Reads metadata from module docstrings
- Version from `__version__` attribute
- No need for version in pyproject.toml

---

### 4. Poetry / Poetry-Core

**Status**: Feature-rich, integrated tool ecosystem

```toml
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "my-package"
version = "0.1.0"
description = "My awesome package"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = "^3.10"
requests = "^2.28.0"

[tool.poetry.dev-dependencies]
pytest = "^7.0.0"
```

**Strengths**:
- Rich dependency management features (lock files, constraints)
- Integrated virtual environment management
- Excellent dependency resolution
- Nice CLI for package management
- Good for projects using Poetry ecosystem

**Weaknesses**:
- Heavier than alternatives (more dependencies)
- Uses custom dependency syntax (^, ~) vs standard
- Tightly coupled to Poetry CLI (though poetry-core is standalone)
- Can be overkill for simple projects

**Best For**:
- Projects using Poetry for dependency management
- Teams wanting opinionated, all-in-one tooling
- Projects with complex dependency constraints
- When lock file reproducibility is critical

**Note**: poetry-core can be used as standalone build backend without full Poetry

---

### 5. PDM-Backend

**Status**: Modern, PEP-compliant, growing

```toml
[build-system]
requires = ["pdm-backend"]
build-backend = "pdm.backend"

# Follows PEP 621 standards closely
# Similar philosophy to hatchling but independent ecosystem
```

**Strengths**:
- Standards-compliant (PEP 621, PEP 517, etc.)
- Modern design
- Part of PDM ecosystem (alternative to Poetry)
- Good performance

**Weaknesses**:
- Smaller community than setuptools/poetry
- Less documentation and examples
- Ecosystem still maturing

**Best For**:
- Projects using PDM for dependency management
- Teams wanting modern PEP-compliant tooling
- Alternative to Poetry ecosystem

---

## Comparison Table

| Feature | Setuptools | Hatchling | Flit-Core | Poetry-Core | PDM-Backend |
|---------|-----------|-----------|-----------|-------------|-------------|
| **Maturity** | Very High | Medium | High | High | Medium |
| **Ecosystem** | Largest | Growing | Stable | Large | Growing |
| **Build Speed** | Slower | Fast | Fast | Medium | Fast |
| **C Extensions** | Yes | Yes | No | Yes | Yes |
| **Configuration** | Verbose | Moderate | Minimal | Verbose | Moderate |
| **Auto-discovery** | Basic | Smart | Smart | Smart | Smart |
| **Default for** | pip | uv | — | Poetry | PDM |
| **Learning Curve** | Medium | Low | Low | Medium | Low |
| **Non-package Projects** | Excellent | Poor | Poor | Medium | Medium |

## Special Case: Dependency-Only Projects

For learning repositories or projects that don't distribute packages but want to use `pyproject.toml` for dependency management:

### Setuptools Approach (Recommended)

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = []  # Explicitly specify no modules to package
```

**Why this works**: Setuptools is forgiving and allows editable installs without actual packages.

### Alternative: No Build System

```toml
# Remove [build-system] section entirely
[project]
name = "my-learning-repo"
dependencies = [...]

[project.optional-dependencies]
ml = [...]
```

**Limitation**: Can't use `pip install -e .`, but can use dependency groups via other means.

## Decision Tree

```
Do you need to distribute a package?
│
├─ No (learning repo, dependency management only)
│  └─ Use: setuptools with py-modules = []
│
└─ Yes (will publish to PyPI or internal registry)
   │
   ├─ Do you have C extensions or complex builds?
   │  └─ Use: setuptools
   │
   ├─ Are you already using Poetry/PDM CLI?
   │  ├─ Poetry → Use: poetry-core
   │  └─ PDM → Use: pdm-backend
   │
   ├─ Is it a simple single-module library?
   │  └─ Use: flit-core
   │
   └─ Modern pure-Python project?
      └─ Use: hatchling (uv default)
```

## Migration Examples

### From Hatchling to Setuptools (Non-Package Project)

**Before**:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**After**:
```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
py-modules = []
```

### From setup.py to Modern Backend

**Before** (setup.py):
```python
from setuptools import setup, find_packages

setup(
    name="myproject",
    version="0.1.0",
    packages=find_packages(),
)
```

**After** (pyproject.toml):
```toml
[project]
name = "myproject"
version = "0.1.0"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = {find = {}}  # Equivalent to find_packages()
```

## Common Issues & Solutions

### Issue: "Unable to determine which files to ship"

**Backend**: Hatchling
**Cause**: No matching package directory found
**Solutions**:
1. Create directory matching project name: `myproject/`
2. Use src layout: `src/myproject/`
3. Switch to setuptools with `py-modules = []`

### Issue: "No module named 'myproject' after install"

**Cause**: Package not discovered by build backend
**Solutions**:
1. Check package directory naming matches project name
2. Use explicit `packages = ["myproject"]` configuration
3. Verify __init__.py exists in package directory

### Issue: "Could not build wheels"

**Possible Causes**:
1. C extensions without proper compiler
2. Missing build dependencies in `requires`
3. Incompatible build backend version

**Debug Steps**:
```bash
# Verbose build to see errors
pip install -e . -v

# Check build dependencies
pip install build
python -m build --wheel
```

## Best Practices

1. **Choose based on your needs**, not trends:
   - Simple package → flit-core
   - Complex package → setuptools
   - Modern pure-Python → hatchling
   - Dependency-only → setuptools

2. **Specify minimum versions** in build-system.requires:
   ```toml
   requires = ["setuptools>=61.0"]  # PEP 621 support
   ```

3. **Test editable installs** work as expected:
   ```bash
   pip install -e ".[dev]"
   python -c "import mypackage"
   ```

4. **Document your choice** for future maintainers

5. **Stay standards-compliant** (PEP 517, PEP 621) for maximum tool compatibility

## References

- [PEP 517](https://peps.python.org/pep-0517/) - Build system interface
- [PEP 518](https://peps.python.org/pep-0518/) - pyproject.toml specification
- [PEP 621](https://peps.python.org/pep-0621/) - Project metadata in pyproject.toml
- [Setuptools Documentation](https://setuptools.pypa.io/)
- [Hatchling Documentation](https://hatch.pypa.io/latest/config/build/)
- [Flit Documentation](https://flit.pypa.io/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [PDM Documentation](https://pdm.fming.dev/)
