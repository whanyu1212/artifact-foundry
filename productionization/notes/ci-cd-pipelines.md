# CI/CD Pipelines

**Last Updated**: 2025-12-30

A comprehensive guide to Continuous Integration and Continuous Deployment from fundamentals to production implementation.

---

## Table of Contents

1. [What is CI/CD?](#what-is-cicd)
2. [Why CI/CD Matters](#why-cicd-matters)
3. [Core Concepts](#core-concepts)
4. [The CI/CD Pipeline Stages](#the-cicd-pipeline-stages)
5. [Version Control and Branching Strategies](#version-control-and-branching-strategies)
6. [Continuous Integration (CI)](#continuous-integration-ci)
7. [Continuous Deployment vs Continuous Delivery](#continuous-deployment-vs-continuous-delivery)
8. [Testing in CI/CD](#testing-in-cicd)
9. [Build Automation](#build-automation)
10. [Artifact Management](#artifact-management)
11. [Deployment Strategies](#deployment-strategies)
12. [Infrastructure as Code (IaC)](#infrastructure-as-code-iac)
13. [Configuration Management](#configuration-management)
14. [Monitoring and Observability](#monitoring-and-observability)
15. [Security in CI/CD (DevSecOps)](#security-in-cicd-devsecops)
16. [CI/CD Tools and Platforms](#cicd-tools-and-platforms)
17. [GitHub Actions Deep Dive](#github-actions-deep-dive)
18. [GitLab CI/CD](#gitlab-cicd)
19. [Jenkins](#jenkins)
20. [Best Practices](#best-practices)
21. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
22. [ML-Specific CI/CD Considerations](#ml-specific-cicd-considerations)

---

## What is CI/CD?

**CI/CD** stands for Continuous Integration and Continuous Deployment/Delivery. It's a methodology that automates the software delivery process from code commit to production deployment.

### The Core Idea

**Traditional Development** (Manual):
```
Developer writes code
→ Manual build
→ Manual testing
→ Manual deployment
→ Hope nothing breaks
```

**CI/CD** (Automated):
```
Developer commits code
→ Automatic build
→ Automatic testing
→ Automatic deployment
→ Immediate feedback
```

### Three Components

1. **Continuous Integration (CI)**: Automatically build and test code changes
2. **Continuous Delivery (CD)**: Automatically prepare releases for production
3. **Continuous Deployment (CD)**: Automatically deploy to production

---

## Why CI/CD Matters

### The Problems CI/CD Solves

#### 1. Integration Hell

**Without CI**:
```
Multiple developers work for weeks on separate branches
→ Try to merge everything before release
→ Conflicts everywhere
→ Days of debugging integration issues
```

**With CI**:
```
Developers integrate code multiple times per day
→ Small, manageable changes
→ Conflicts caught immediately
→ Always in a deployable state
```

#### 2. Manual Testing Bottleneck

**Without CI**:
```
Developer: "I'm done!"
→ Wait for QA team
→ Manual testing takes days
→ Bug found
→ Fix and repeat
```

**With CI**:
```
Developer commits code
→ Automated tests run in minutes
→ Immediate feedback
→ Fix issues before they accumulate
```

#### 3. Deployment Anxiety

**Without CD**:
```
Deployments are rare, risky events
→ "Deployment Day" = stress
→ Manual steps, high error rate
→ Difficult to rollback
```

**With CD**:
```
Deployments are routine, low-risk
→ Deploy multiple times per day
→ Automated, repeatable process
→ Easy rollback if needed
```

### Benefits

1. **Faster Time to Market**: Ship features in hours/days, not weeks/months
2. **Higher Quality**: Catch bugs early when they're cheap to fix
3. **Reduced Risk**: Small, frequent changes are less risky than big releases
4. **Better Collaboration**: Clear process, immediate feedback
5. **Developer Productivity**: Automation frees developers to focus on features
6. **Business Agility**: Respond quickly to market changes

---

## Core Concepts

### The Feedback Loop

```
Code Commit → Build → Test → Deploy → Monitor
     ↑                                    ↓
     └────────── Feedback ────────────────┘
```

**Fast feedback is critical**: Developers should know within minutes if their change broke something.

### Pipeline as Code

CI/CD pipelines should be:
- **Versioned**: In source control alongside application code
- **Reviewable**: Changes go through code review
- **Testable**: Can test pipeline changes safely
- **Reproducible**: Same results every time

### Immutability

**Key principle**: Build once, deploy everywhere.

```
Build Image (v1.2.3)
    ↓
Deploy to Dev (same image)
    ↓
Deploy to Staging (same image)
    ↓
Deploy to Production (same image)
```

**Never rebuild** between environments - eliminates "works on staging but not prod" issues.

### Shift Left

**Shift Left**: Move testing, security, and quality checks earlier in the development process.

```
Traditional (Shift Right):
Code → Build → Deploy → Test → Security Scan → Production

Modern (Shift Left):
Code → Test → Security Scan → Build → Deploy → Production
```

**Why?** Finding bugs in development costs $1, in production costs $1000.

---

## The CI/CD Pipeline Stages

A typical pipeline has several stages that code passes through:

```
┌─────────────┐
│   Commit    │  Developer pushes code to version control
└──────┬──────┘
       ↓
┌─────────────┐
│   Build     │  Compile code, install dependencies
└──────┬──────┘
       ↓
┌─────────────┐
│    Test     │  Unit tests, integration tests, linters
└──────┬──────┘
       ↓
┌─────────────┐
│  Security   │  Vulnerability scans, SAST, dependency checks
└──────┬──────┘
       ↓
┌─────────────┐
│   Package   │  Build Docker image, create artifacts
└──────┬──────┘
       ↓
┌─────────────┐
│  Deploy Dev │  Deploy to development environment
└──────┬──────┘
       ↓
┌─────────────┐
│ Test Staging│  Integration tests, E2E tests
└──────┬──────┘
       ↓
┌─────────────┐
│Deploy Prod  │  Deploy to production (manual approval or automatic)
└──────┬──────┘
       ↓
┌─────────────┐
│  Monitor    │  Track metrics, logs, errors
└─────────────┘
```

### Stage Characteristics

**Each stage should**:
- **Fail fast**: If tests fail, stop immediately
- **Be idempotent**: Running twice produces same result
- **Provide clear feedback**: What failed and why
- **Be parallelizable**: Run tests concurrently when possible

---

## Version Control and Branching Strategies

CI/CD is built on version control. Your branching strategy determines how code flows to production.

### Git Flow

**Branches**:
- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: Feature branches
- `release/*`: Release preparation
- `hotfix/*`: Emergency production fixes

```
main ────●────────●─────────●───────→ (production)
          ↖      ↗         ↗
develop ───●●●●●●───●●●●●●●──────────→
            ↖  ↗     ↖  ↗
feature/x    ●●●       (merged)
feature/y          ●●●●●
```

**When to use**: Large teams, scheduled releases, need release branches.

### GitHub Flow (Simpler)

**Branches**:
- `main`: Always deployable
- `feature/*`: Short-lived feature branches

```
main ──●─────●────●────●────●────●───→ (always deployable)
        ↖   ↗      ↖  ↗
feature/x ●●●
feature/y      ●●●●●
```

**Process**:
1. Create branch from `main`
2. Add commits
3. Open Pull Request
4. Review and discuss
5. Merge to `main`
6. Deploy `main` to production

**When to use**: Small teams, continuous deployment, web applications.

### Trunk-Based Development

**Main idea**: Everyone commits to `main` (trunk) frequently.

```
main ──●●●●●●●●●●●●●●●●●●●●●●───→ (everyone commits here)
```

**Feature flags** control what's visible in production.

**When to use**: High-performing teams, continuous deployment, need high velocity.

### Choosing a Strategy

| Strategy | Team Size | Release Frequency | Complexity |
|----------|-----------|-------------------|------------|
| Git Flow | Large | Scheduled releases | High |
| GitHub Flow | Small-Medium | Continuous | Low |
| Trunk-Based | Any (with discipline) | Continuous | Medium |

---

## Continuous Integration (CI)

### What is Continuous Integration?

**CI** is the practice of merging code changes into the main branch frequently (multiple times per day) and automatically verifying each change.

### Core Practices

#### 1. Maintain a Single Source Repository

All code, configuration, build scripts, tests in version control.

```
repo/
├── src/           # Application code
├── tests/         # Test code
├── .github/       # CI/CD workflows
├── Dockerfile     # Container definition
└── requirements.txt
```

#### 2. Automate the Build

**Build should be one command**:

```bash
# Bad: Complex manual steps
export PYTHON_PATH=/usr/local/...
pip install -r requirements.txt
python setup.py build
cp config.yaml.template config.yaml
# ... 15 more steps

# Good: Single command
make build
```

#### 3. Make Builds Self-Testing

Every build should include automated tests:

```yaml
# .github/workflows/ci.yml
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build
        run: make build
      - name: Test
        run: make test  # ← Tests are part of build process
```

#### 4. Everyone Commits to Main Daily

**Avoid long-lived branches**. They lead to integration hell.

```
# Anti-pattern: Long-lived feature branch
feature-branch: 2 weeks old, 500 commits behind main

# Best practice: Short-lived branches
feature-branch: 1 day old, merge to main daily
```

#### 5. Every Commit Triggers a Build

**CI server watches repository**:

```
Developer commits code
    ↓
CI server detects change
    ↓
Checks out code
    ↓
Runs build + tests
    ↓
Notifies developer of result (✓ or ✗)
```

#### 6. Keep Build Fast

**Target**: Build + tests in < 10 minutes.

**Strategies**:
- Parallel testing
- Incremental builds
- Caching dependencies
- Split long tests into separate pipeline

```yaml
# Parallel test execution
strategy:
  matrix:
    test-suite: [unit, integration, e2e]
  parallel: 3
```

#### 7. Test in a Clone of Production

Development environment should match production:

```yaml
services:
  postgres:
    image: postgres:15  # Same version as production
  redis:
    image: redis:7      # Same version as production
```

#### 8. Make it Easy to Get Latest Deliverables

Everyone should be able to access latest build artifacts:

```
CI builds Docker image → Pushes to registry
Developers: docker pull myapp:latest
```

#### 9. Everyone Can See Build Results

**Transparency is key**:
- Build status visible to whole team
- Notifications on failures
- Dashboards showing trends

#### 10. Automate Deployment

CI should produce deployable artifacts automatically.

---

## Continuous Deployment vs Continuous Delivery

### Continuous Delivery

**Continuous Delivery**: Automatically prepare code for release, but **require manual approval** for production deployment.

```
Commit → Build → Test → Package → Deploy to Staging → [Manual Approval] → Deploy to Production
```

**When to use**:
- Regulated industries (finance, healthcare)
- Business wants control over release timing
- Need coordination with marketing/support

**Benefits**:
- Always ready to deploy
- Business chooses when
- Reduced deployment risk
- Still maintains automation benefits

### Continuous Deployment

**Continuous Deployment**: Automatically deploy to production **without manual approval**.

```
Commit → Build → Test → Package → Deploy to Production (automatic)
```

**When to use**:
- SaaS applications
- High-performing teams
- Fast feedback loops needed

**Requirements**:
- Excellent test coverage
- Strong monitoring
- Feature flags for risk management
- Automated rollback capability

### Comparison

| Aspect | Continuous Delivery | Continuous Deployment |
|--------|-------------------|---------------------|
| Production Deploy | Manual | Automatic |
| Release Frequency | On-demand | Every commit |
| Risk | Lower (human gate) | Higher (but mitigated) |
| Speed | Fast | Fastest |
| Best For | Enterprise, regulated | SaaS, web apps |

---

## Testing in CI/CD

Testing is the foundation of CI/CD. Without reliable tests, automation is impossible.

### The Testing Pyramid

```
          ╱╲
         ╱  ╲        E2E Tests (Few, slow, expensive)
        ╱────╲
       ╱      ╲      Integration Tests (Some, medium speed)
      ╱────────╲
     ╱          ╲    Unit Tests (Many, fast, cheap)
    ╱────────────╲
```

**Ideal distribution**:
- 70% Unit tests
- 20% Integration tests
- 10% E2E tests

### Test Types

#### 1. Unit Tests

**What**: Test individual functions/classes in isolation.

**Characteristics**:
- Very fast (milliseconds)
- No external dependencies
- High coverage possible

```python
def test_calculate_discount():
    assert calculate_discount(100, 0.1) == 90
    assert calculate_discount(100, 0) == 100
```

**In CI**:
```yaml
- name: Run unit tests
  run: pytest tests/unit -v --cov=src --cov-report=xml
```

#### 2. Integration Tests

**What**: Test interaction between components.

**Characteristics**:
- Medium speed (seconds)
- May need database, external services
- Test realistic scenarios

```python
def test_user_registration_flow():
    # Tests database, email service, validation
    response = client.post('/register', data={'email': 'test@example.com'})
    assert response.status_code == 201
    assert User.query.filter_by(email='test@example.com').first() is not None
```

**In CI**:
```yaml
services:
  postgres:
    image: postgres:15

steps:
  - name: Run integration tests
    run: pytest tests/integration -v
    env:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/test
```

#### 3. End-to-End (E2E) Tests

**What**: Test complete user workflows through UI.

**Characteristics**:
- Slow (minutes)
- Brittle (UI changes break tests)
- Test critical user journeys

```python
def test_purchase_flow(browser):
    browser.visit('http://localhost:8000')
    browser.find_by_text('Add to Cart').click()
    browser.find_by_text('Checkout').click()
    browser.fill('card_number', '4242424242424242')
    browser.find_by_text('Purchase').click()
    assert browser.is_text_present('Order Confirmed')
```

**In CI**:
```yaml
- name: Run E2E tests
  run: |
    docker-compose up -d
    pytest tests/e2e -v
    docker-compose down
```

#### 4. Static Analysis / Linting

**What**: Analyze code without running it.

**Tools**:
- **Python**: `pylint`, `flake8`, `mypy` (type checking), `black` (formatting)
- **JavaScript**: `eslint`, `prettier`
- **Go**: `golint`, `go vet`

```yaml
- name: Lint code
  run: |
    flake8 src/ --max-line-length=100
    mypy src/
    black --check src/
```

#### 5. Security Scanning

**What**: Detect vulnerabilities in code and dependencies.

**Tools**:
- **SAST** (Static): `bandit`, `semgrep`, `SonarQube`
- **Dependency**: `safety`, `snyk`, `dependabot`
- **Container**: `trivy`, `grype`, `clair`

```yaml
- name: Security scan
  run: |
    # Scan dependencies
    safety check

    # Scan code for vulnerabilities
    bandit -r src/

    # Scan Docker image
    docker build -t myapp:${{ github.sha }} .
    trivy image myapp:${{ github.sha }}
```

#### 6. Performance Tests

**What**: Ensure application meets performance requirements.

```python
def test_api_response_time():
    start = time.time()
    response = requests.get('http://localhost:8000/api/users')
    duration = time.time() - start
    assert duration < 0.5  # Must respond in < 500ms
```

### Test Coverage

**Code coverage** measures what percentage of code is executed by tests.

```yaml
- name: Run tests with coverage
  run: |
    pytest --cov=src --cov-report=html --cov-report=term

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

**Good targets**:
- New code: 80%+ coverage
- Critical paths: 90%+ coverage
- Overall: 70%+ coverage

**Remember**: 100% coverage doesn't mean bug-free! Quality > quantity.

---

## Build Automation

### What is a Build?

**Build** transforms source code into executable artifacts.

**Steps typically include**:
1. Fetch dependencies
2. Compile code (if needed)
3. Run tests
4. Create distributable package

### Build Tools by Language

| Language | Build Tools |
|----------|-------------|
| Python | `pip`, `poetry`, `setuptools`, `hatch` |
| JavaScript | `npm`, `yarn`, `pnpm`, `webpack`, `vite` |
| Java | `Maven`, `Gradle` |
| Go | Built-in `go build` |
| C/C++ | `make`, `cmake`, `bazel` |
| .NET | `dotnet build`, `MSBuild` |

### Build Script Example (Python)

```makefile
# Makefile
.PHONY: install test build clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/
	mypy src/
	black --check src/

build: clean lint test
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +

docker-build:
	docker build -t myapp:latest .

docker-test:
	docker run --rm myapp:latest pytest
```

**Usage**:
```bash
make install  # Install dependencies
make test     # Run tests
make build    # Full build process
```

### Caching in Builds

**Problem**: Installing dependencies on every build is slow.

**Solution**: Cache dependencies.

**GitHub Actions example**:
```yaml
- name: Cache Python dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Install dependencies
  run: pip install -r requirements.txt
```

**Before caching**: 2 minutes to install dependencies
**After caching**: 10 seconds

### Docker Build Caching

```dockerfile
# Efficient layer ordering
FROM python:3.11-slim

# Layer 1: System dependencies (rarely change)
RUN apt-get update && apt-get install -y git

# Layer 2: Python dependencies (change occasionally)
COPY requirements.txt .
RUN pip install -r requirements.txt  # ← Cached if requirements.txt unchanged

# Layer 3: Application code (changes frequently)
COPY . .

CMD ["python", "app.py"]
```

**BuildKit cache**:
```bash
# Build with cache export
docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t myapp:latest .

# Build using remote cache
docker build --cache-from myapp:latest -t myapp:latest .
```

---

## Artifact Management

**Artifacts** are the outputs of your build process (Docker images, compiled binaries, packages).

### Types of Artifacts

1. **Container Images**: Docker images
2. **Language Packages**: Python wheels, npm packages, JAR files
3. **Compiled Binaries**: Go binaries, C++ executables
4. **Static Assets**: JavaScript bundles, CSS files

### Artifact Repositories

**Purpose**: Store and version build artifacts.

**Options**:

| Type | Tool | Use Case |
|------|------|----------|
| Container Registry | Docker Hub, ECR, GCR, ACR | Docker images |
| Package Registry | PyPI, npm, Maven Central | Language packages |
| Generic Artifact Store | Artifactory, Nexus | Any artifact type |
| Cloud Storage | S3, GCS, Azure Blob | Static files, backups |

### Artifact Versioning

**Semantic Versioning** (SemVer): `MAJOR.MINOR.PATCH`

```
1.0.0 → Initial release
1.0.1 → Bug fix (backwards compatible)
1.1.0 → New feature (backwards compatible)
2.0.0 → Breaking change
```

**In CI/CD**:
```yaml
- name: Extract version from tag
  run: echo "VERSION=${GITHUB_REF#refs/tags/v}" >> $GITHUB_ENV

- name: Build and tag Docker image
  run: |
    docker build -t myapp:${{ env.VERSION }} .
    docker tag myapp:${{ env.VERSION }} myapp:latest

- name: Push to registry
  run: |
    docker push myapp:${{ env.VERSION }}
    docker push myapp:latest
```

### Artifact Promotion

**Build once, promote through environments**:

```
Build (CI)
  ↓
Push to registry with SHA tag: myapp:sha-abc123
  ↓
Test in Dev
  ↓
Tag as: myapp:dev
  ↓
Test in Staging
  ↓
Tag as: myapp:staging
  ↓
Deploy to Production
  ↓
Tag as: myapp:v1.2.3, myapp:latest
```

**Same artifact** moves through all environments - eliminates "works in staging, not in prod" issues.

---

## Deployment Strategies

How you deploy determines your risk, downtime, and rollback capability.

### 1. Recreate (Simple, Downtime)

**Process**:
1. Stop old version
2. Deploy new version
3. Start new version

```
Before:  [v1] [v1] [v1]
During:  [  ] [  ] [  ]  ← Downtime!
After:   [v2] [v2] [v2]
```

**Pros**: Simple, complete environment refresh
**Cons**: Downtime during deployment

**When to use**: Development environments, non-critical applications

### 2. Rolling Update (No Downtime)

**Process**: Gradually replace instances one at a time.

```
Start:   [v1] [v1] [v1] [v1]
Step 1:  [v2] [v1] [v1] [v1]
Step 2:  [v2] [v2] [v1] [v1]
Step 3:  [v2] [v2] [v2] [v1]
End:     [v2] [v2] [v2] [v2]
```

**Pros**: No downtime, gradual rollout
**Cons**: Both versions running simultaneously (need backwards compatibility)

**Kubernetes example**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 4
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1  # At most 1 pod down
      maxSurge: 1        # At most 1 extra pod
```

### 3. Blue-Green Deployment (Zero Downtime)

**Process**: Run two identical environments, switch traffic instantly.

```
Blue (v1):   [v1] [v1] [v1]  ← 100% traffic
Green (v2):  [v2] [v2] [v2]  ← 0% traffic (ready)

[Switch traffic]

Blue (v1):   [v1] [v1] [v1]  ← 0% traffic (keep for rollback)
Green (v2):  [v2] [v2] [v2]  ← 100% traffic
```

**Pros**:
- Instant rollback (switch back to blue)
- Test green environment before switching
- Zero downtime

**Cons**:
- Requires 2x resources
- Database migrations can be tricky

**Implementation**:
```yaml
# Load balancer switches between blue and green
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
    version: green  # ← Change to 'blue' to rollback
```

### 4. Canary Deployment (Risk Mitigation)

**Process**: Route small percentage of traffic to new version, gradually increase.

```
Stage 1:  90% traffic → [v1] [v1] [v1]
          10% traffic → [v2]

Stage 2:  50% traffic → [v1] [v1]
          50% traffic → [v2] [v2]

Stage 3:   0% traffic → [v1]
         100% traffic → [v2] [v2] [v2]
```

**Pros**:
- Limit blast radius (only affects small % of users)
- Real production testing with minimal risk
- Can automate rollback based on metrics

**Cons**:
- Complex traffic routing needed
- Monitoring must be excellent

**Example (using feature flags)**:
```python
# Feature flag controls who sees new version
def get_recommendation_algorithm(user_id):
    if canary_enabled(user_id, percentage=10):
        return new_algorithm()  # 10% of users
    else:
        return old_algorithm()  # 90% of users
```

**Automated canary with metrics**:
```yaml
# If error rate < 1% for 5 minutes, increase traffic
# If error rate > 1%, rollback automatically
```

### 5. A/B Testing Deployment

**Process**: Similar to canary, but for testing different versions against each other.

```
50% traffic → [v1 Algorithm A]
50% traffic → [v2 Algorithm B]

Measure: Which version has better metrics?
```

**Use case**: Testing product changes, ML models, UX variations.

### Comparison

| Strategy | Downtime | Complexity | Rollback Speed | Resource Cost |
|----------|----------|------------|----------------|---------------|
| Recreate | Yes | Low | Medium | Low |
| Rolling Update | No | Medium | Slow | Low |
| Blue-Green | No | Medium | Instant | High (2x) |
| Canary | No | High | Fast | Medium |
| A/B Testing | No | High | Fast | Medium |

---

## Infrastructure as Code (IaC)

**Infrastructure as Code**: Manage infrastructure through code files instead of manual configuration.

### Why IaC?

**Without IaC** (Manual):
```
1. Login to cloud console
2. Click "Create Server"
3. Choose size, region, settings
4. Install software manually
5. Hope you remember all steps next time
```

**With IaC** (Automated):
```terraform
# infrastructure.tf
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.medium"

  tags = {
    Name = "web-server"
  }
}
```

```bash
terraform apply  # Creates infrastructure
```

### Benefits

1. **Version Control**: Infrastructure changes tracked in Git
2. **Reproducible**: Create identical environments easily
3. **Documentation**: Code documents current state
4. **Testable**: Can test infrastructure changes
5. **Disaster Recovery**: Rebuild infrastructure from code

### IaC Tools

#### 1. Terraform (Cloud-Agnostic)

**Pros**: Works with all cloud providers, mature ecosystem
**Cons**: State management can be complex

```hcl
# main.tf
provider "aws" {
  region = "us-west-2"
}

resource "aws_instance" "app_server" {
  ami           = var.ami_id
  instance_type = "t3.medium"

  tags = {
    Name = "AppServer"
    Environment = var.environment
  }
}

resource "aws_security_group" "app_sg" {
  name = "app-security-group"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

#### 2. CloudFormation (AWS)

**Pros**: Native AWS, no state management needed
**Cons**: AWS only, YAML can get verbose

```yaml
# template.yml
Resources:
  AppServer:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: t3.medium
      ImageId: !Ref AmiId
      Tags:
        - Key: Name
          Value: AppServer
```

#### 3. Pulumi (Use Programming Languages)

**Pros**: Use real programming languages (Python, TypeScript, Go)
**Cons**: Newer, smaller ecosystem

```python
# __main__.py
import pulumi
from pulumi_aws import ec2

# Create EC2 instance
instance = ec2.Instance('app-server',
    instance_type='t3.medium',
    ami='ami-0c55b159cbfafe1f0',
    tags={'Name': 'AppServer'}
)

pulumi.export('instance_id', instance.id)
```

#### 4. Ansible (Configuration Management)

**Pros**: Agentless, simple YAML syntax
**Cons**: Not purely declarative

```yaml
# playbook.yml
- hosts: webservers
  tasks:
    - name: Install nginx
      apt:
        name: nginx
        state: present

    - name: Start nginx
      service:
        name: nginx
        state: started
        enabled: yes
```

### IaC in CI/CD

```yaml
# .github/workflows/infrastructure.yml
name: Deploy Infrastructure

on:
  push:
    branches: [main]
    paths:
      - 'infrastructure/**'

jobs:
  terraform:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v2

      - name: Terraform Init
        run: terraform init
        working-directory: ./infrastructure

      - name: Terraform Plan
        run: terraform plan -out=tfplan
        working-directory: ./infrastructure

      - name: Terraform Apply
        if: github.ref == 'refs/heads/main'
        run: terraform apply -auto-approve tfplan
        working-directory: ./infrastructure
```

---

## Configuration Management

**Configuration**: Settings that vary between environments (API keys, database URLs, feature flags).

### The Twelve-Factor App: Config

**Rule**: Store config in environment variables, never in code.

**Bad** (hardcoded):
```python
# DON'T DO THIS
DATABASE_URL = "postgresql://user:password@prod-db:5432/myapp"
API_KEY = "sk-1234567890abcdef"
```

**Good** (environment variables):
```python
import os

DATABASE_URL = os.environ['DATABASE_URL']
API_KEY = os.environ['API_KEY']
```

### Configuration Approaches

#### 1. Environment Variables

```yaml
# docker-compose.yml
services:
  app:
    environment:
      DATABASE_URL: postgresql://user:pass@db:5432/myapp
      REDIS_URL: redis://redis:6379
      LOG_LEVEL: info
```

**Pros**: Simple, universally supported
**Cons**: Not great for complex nested config, secrets visible in process list

#### 2. Configuration Files

```yaml
# config.yaml
database:
  url: postgresql://user:pass@db:5432/myapp
  pool_size: 10

redis:
  url: redis://redis:6379
  max_connections: 20

logging:
  level: info
  format: json
```

**Load in app**:
```python
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

DATABASE_URL = config['database']['url']
```

**Pros**: Supports complex structures, readable
**Cons**: Harder to inject per environment

#### 3. Secret Management Services

**Tools**:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault
- Google Secret Manager

**Example (AWS Secrets Manager)**:
```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

db_creds = get_secret('prod/database/credentials')
DATABASE_URL = f"postgresql://{db_creds['username']}:{db_creds['password']}@..."
```

**Pros**: Centralized, audited, encrypted, rotation support
**Cons**: Added complexity, dependency on external service

### Environment-Specific Config

**Directory structure**:
```
config/
├── base.yaml        # Shared config
├── development.yaml # Dev overrides
├── staging.yaml     # Staging overrides
└── production.yaml  # Prod overrides
```

**Load with precedence**:
```python
def load_config(env='development'):
    # Load base config
    with open('config/base.yaml') as f:
        config = yaml.safe_load(f)

    # Overlay environment-specific config
    with open(f'config/{env}.yaml') as f:
        env_config = yaml.safe_load(f)
        config.update(env_config)

    return config
```

### Best Practices

1. **Never commit secrets** to version control
2. **Use `.env` files** for local development (add to `.gitignore`)
3. **Inject secrets** at deployment time
4. **Rotate secrets** regularly
5. **Principle of least privilege**: Services only get secrets they need

---

## Monitoring and Observability

**You can't improve what you don't measure.** CI/CD enables frequent deployments, but you need to know if they're working.

### The Three Pillars of Observability

#### 1. Metrics (What)

**Quantitative measurements** over time.

**Examples**:
- Request rate (requests/second)
- Error rate (errors/second)
- Latency (p50, p95, p99)
- CPU/Memory usage
- Database connections

**Tools**: Prometheus, Datadog, New Relic, CloudWatch

**Example (Prometheus)**:
```python
from prometheus_client import Counter, Histogram
import time

# Define metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.route('/api/users')
def get_users():
    request_count.labels(method='GET', endpoint='/api/users').inc()

    with request_duration.time():
        # Handle request
        users = fetch_users()
        return jsonify(users)
```

#### 2. Logs (Why)

**Event records** with context.

**Structured logging**:
```python
import logging
import json

logger = logging.getLogger(__name__)

# Structured log
logger.info(json.dumps({
    'event': 'user_login',
    'user_id': user.id,
    'ip_address': request.remote_addr,
    'timestamp': datetime.utcnow().isoformat()
}))
```

**Tools**: ELK Stack (Elasticsearch, Logstash, Kibana), Splunk, Loki

#### 3. Traces (Where)

**Request flow** through distributed system.

**Example**:
```
User Request → API Gateway (10ms)
              ↓
            Web Server (50ms)
              ↓
            Database (200ms)
              ↓
            Cache (5ms)

Total: 265ms (where did time go? → Database!)
```

**Tools**: Jaeger, Zipkin, AWS X-Ray, Datadog APM

### Monitoring in CI/CD

#### Health Checks

**Readiness probe**: Is service ready to handle traffic?
**Liveness probe**: Is service still alive?

```yaml
# Kubernetes health checks
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 3
```

```python
# Health check endpoint
@app.route('/health')
def health():
    # Check critical dependencies
    if not database.is_connected():
        return jsonify({'status': 'unhealthy'}), 503

    return jsonify({'status': 'healthy'}), 200
```

#### Deployment Validation

**Automated checks** after deployment:

```yaml
- name: Deploy to production
  run: kubectl apply -f k8s/

- name: Wait for rollout
  run: kubectl rollout status deployment/myapp

- name: Smoke test
  run: |
    sleep 10  # Wait for service to stabilize
    curl -f https://api.example.com/health || exit 1

- name: Check error rate
  run: |
    # Query Prometheus for error rate
    ERROR_RATE=$(curl -s 'http://prometheus:9090/api/v1/query?query=rate(http_errors[5m])')
    # Rollback if error rate > threshold
    if [ "$ERROR_RATE" -gt 0.01 ]; then
      kubectl rollout undo deployment/myapp
      exit 1
    fi
```

### Alerting

**Alert on symptoms, not causes**:

```yaml
# Good: Alert on user impact
alert: HighErrorRate
expr: rate(http_errors[5m]) > 0.05
message: "Error rate above 5% for 5 minutes"

# Bad: Alert on low-level metrics
alert: HighCPU
expr: cpu_usage > 80%
message: "CPU usage high"  # So what? Does this affect users?
```

---

## Security in CI/CD (DevSecOps)

**DevSecOps**: Integrate security into the entire development lifecycle.

### Shift Left Security

**Traditional**: Security testing at the end (after production deploy)
**DevSecOps**: Security testing at every stage

```
Code → SAST → Dependency Scan → Build → Container Scan → Deploy → Runtime Security
```

### Security Checks in CI/CD

#### 1. Secret Scanning

**Prevent secrets from being committed**:

```yaml
- name: Scan for secrets
  uses: trufflesecurity/trufflehog@main
  with:
    path: ./

# Will fail if API keys, passwords, tokens found in commits
```

**Pre-commit hook**:
```bash
# .git/hooks/pre-commit
#!/bin/bash
gitleaks protect --staged --redact --verbose
```

#### 2. Static Application Security Testing (SAST)

**Scan code for vulnerabilities** without running it:

```yaml
- name: Run SAST scan
  run: |
    # Python
    bandit -r src/ -f json -o bandit-report.json

    # JavaScript
    npm audit

    # Multi-language
    semgrep --config auto src/
```

#### 3. Dependency Scanning

**Check for known vulnerabilities** in dependencies:

```yaml
- name: Check Python dependencies
  run: |
    pip install safety
    safety check --json

- name: Check npm dependencies
  run: npm audit --audit-level=moderate
```

**Automated dependency updates**:
```yaml
# Dependabot config (.github/dependabot.yml)
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
```

#### 4. Container Image Scanning

**Scan Docker images** for vulnerabilities:

```yaml
- name: Build Docker image
  run: docker build -t myapp:${{ github.sha }} .

- name: Scan image with Trivy
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: myapp:${{ github.sha }}
    format: 'sarif'
    output: 'trivy-results.sarif'
    severity: 'CRITICAL,HIGH'

- name: Fail if critical vulnerabilities found
  run: |
    trivy image --severity CRITICAL --exit-code 1 myapp:${{ github.sha }}
```

#### 5. Infrastructure Security Scanning

**Scan IaC files** for misconfigurations:

```yaml
- name: Scan Terraform with Checkov
  uses: bridgecrewio/checkov-action@master
  with:
    directory: infrastructure/
    framework: terraform

- name: Scan Kubernetes manifests with Kubesec
  run: |
    docker run -v $(pwd):/work kubesec/kubesec:512c5e0 \
      scan /work/k8s/*.yaml
```

#### 6. Dynamic Application Security Testing (DAST)

**Test running application** for vulnerabilities:

```yaml
- name: Deploy to test environment
  run: kubectl apply -f k8s/staging/

- name: Run OWASP ZAP scan
  uses: zaproxy/action-baseline@v0.7.0
  with:
    target: 'https://staging.example.com'
```

### Security Best Practices

1. **Least Privilege**: CI/CD should have minimal permissions needed
2. **Audit Logging**: Log all CI/CD activities
3. **Separate Environments**: Dev/staging/prod isolation
4. **Immutable Artifacts**: Don't modify artifacts after build
5. **Sign Artifacts**: Cryptographically sign builds
6. **Secure Secrets**: Use secret management tools, not environment variables in CI logs

---

## CI/CD Tools and Platforms

### Comparison Matrix

| Tool | Type | Pricing | Best For | Cloud/Self-Hosted |
|------|------|---------|----------|-------------------|
| GitHub Actions | Hosted | Free tier, usage-based | GitHub repos, simplicity | Cloud |
| GitLab CI/CD | Both | Free tier, usage-based | All-in-one platform | Both |
| Jenkins | Self-hosted | Free (OSS) | Customization, on-prem | Self-hosted |
| CircleCI | Hosted | Free tier, usage-based | Docker-first workflows | Cloud |
| Travis CI | Hosted | Free for OSS, paid | Open source projects | Cloud |
| Azure DevOps | Both | Free tier, usage-based | Microsoft ecosystem | Both |
| AWS CodePipeline | Hosted | Usage-based | AWS-native apps | Cloud (AWS) |
| ArgoCD | Self-hosted | Free (OSS) | GitOps, Kubernetes | Self-hosted |
| Buildkite | Hybrid | Usage-based | Security, bring your own compute | Hybrid |

---

## GitHub Actions Deep Dive

**GitHub Actions**: Cloud-based CI/CD integrated into GitHub.

### Core Concepts

**Workflow**: Automated process defined in YAML
**Job**: Set of steps that execute on same runner
**Step**: Individual task (run command, use action)
**Action**: Reusable unit of code
**Runner**: Server that runs workflows

### Workflow Structure

```yaml
# .github/workflows/ci.yml
name: CI Pipeline                    # Workflow name

on:                                  # Triggers
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'             # Daily at midnight

env:                                 # Global environment variables
  PYTHON_VERSION: '3.11'

jobs:                                # Jobs
  test:
    runs-on: ubuntu-latest          # Runner

    services:                        # Service containers
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s

    steps:                           # Steps
      - name: Checkout code
        uses: actions/checkout@v3    # Reusable action

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Install dependencies
        run: |                       # Run shell commands
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest tests/ -v --cov=src
        env:
          DATABASE_URL: postgresql://postgres:postgres@postgres:5432/test

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  build:
    needs: test                      # Wait for test job
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'  # Only on main branch

    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Login to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Push image
        run: |
          docker tag myapp:${{ github.sha }} myapp:latest
          docker push myapp:${{ github.sha }}
          docker push myapp:latest
```

### Advanced Patterns

#### Matrix Strategy (Parallel Testing)

Test across multiple versions:

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.9', '3.10', '3.11']

    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - run: pytest

# Runs 9 jobs in parallel (3 OS × 3 Python versions)
```

#### Reusable Workflows

```yaml
# .github/workflows/reusable-deploy.yml
on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    steps:
      - name: Deploy to ${{ inputs.environment }}
        run: ./deploy.sh ${{ inputs.environment }}
```

```yaml
# .github/workflows/main.yml
jobs:
  deploy-staging:
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: staging

  deploy-production:
    needs: deploy-staging
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: production
```

#### Conditional Execution

```yaml
steps:
  - name: Run only on PR
    if: github.event_name == 'pull_request'
    run: echo "This is a PR"

  - name: Run only on main branch
    if: github.ref == 'refs/heads/main'
    run: echo "This is main"

  - name: Run on version tags
    if: startsWith(github.ref, 'refs/tags/v')
    run: echo "This is a version tag"
```

---

## GitLab CI/CD

**GitLab CI/CD**: Integrated CI/CD in GitLab.

### Pipeline Structure

```yaml
# .gitlab-ci.yml
stages:                    # Define stages
  - build
  - test
  - deploy

variables:                 # Global variables
  DOCKER_IMAGE: myapp
  PYTHON_VERSION: "3.11"

before_script:            # Run before every job
  - echo "Starting job..."

build:
  stage: build
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - python -m build
  artifacts:              # Save build outputs
    paths:
      - dist/
    expire_in: 1 week
  cache:                  # Cache dependencies
    key: ${CI_COMMIT_REF_SLUG}
    paths:
      - .cache/pip

test:unit:
  stage: test
  image: python:3.11
  services:
    - postgres:15
  variables:
    POSTGRES_PASSWORD: postgres
  script:
    - pip install -r requirements.txt
    - pytest tests/unit -v
  coverage: '/TOTAL.*\s+(\d+%)$/'  # Extract coverage percentage

test:integration:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest tests/integration -v
  only:
    - merge_requests
    - main

deploy:staging:
  stage: deploy
  image: alpine
  script:
    - apk add --no-cache curl
    - curl -X POST $STAGING_WEBHOOK
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop

deploy:production:
  stage: deploy
  image: alpine
  script:
    - curl -X POST $PRODUCTION_WEBHOOK
  environment:
    name: production
    url: https://example.com
  only:
    - main
  when: manual           # Require manual trigger
```

### Features

**Child Pipelines**:
```yaml
trigger-child:
  trigger:
    include: .gitlab-ci-child.yml
    strategy: depend
```

**Multi-Project Pipelines**:
```yaml
trigger-downstream:
  trigger:
    project: group/downstream-project
    branch: main
```

**Dynamic Pipelines**:
```yaml
generate-pipeline:
  script:
    - python generate_pipeline.py > pipeline.yml
  artifacts:
    paths:
      - pipeline.yml

trigger-generated:
  trigger:
    include:
      - artifact: pipeline.yml
        job: generate-pipeline
```

---

## Jenkins

**Jenkins**: Most popular self-hosted CI/CD tool.

### Pipeline as Code (Jenkinsfile)

**Declarative Pipeline**:
```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'myapp'
        PYTHON_VERSION = '3.11'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build') {
            agent {
                docker {
                    image 'python:3.11'
                }
            }
            steps {
                sh 'pip install -r requirements.txt'
                sh 'python -m build'
            }
        }

        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh 'pytest tests/unit -v'
                    }
                }
                stage('Lint') {
                    steps {
                        sh 'flake8 src/'
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${BUILD_NUMBER}")
                }
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'develop'
            }
            steps {
                sh './deploy.sh staging'
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            input {
                message "Deploy to production?"
                ok "Deploy"
            }
            steps {
                sh './deploy.sh production'
            }
        }
    }

    post {
        always {
            junit 'test-reports/*.xml'
            archiveArtifacts artifacts: 'dist/*', fingerprint: true
        }
        failure {
            emailext (
                subject: "Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Check console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
        success {
            echo 'Pipeline succeeded!'
        }
    }
}
```

**Scripted Pipeline** (more flexible):
```groovy
node {
    stage('Checkout') {
        checkout scm
    }

    stage('Build') {
        docker.image('python:3.11').inside {
            sh 'pip install -r requirements.txt'
            sh 'python -m build'
        }
    }

    stage('Test') {
        try {
            sh 'pytest tests/ -v'
        } catch (Exception e) {
            currentBuild.result = 'FAILURE'
            throw e
        }
    }
}
```

---

## Best Practices

### 1. Keep Pipelines Fast

**Target**: < 10 minutes from commit to deployment

**Strategies**:
- Parallelize independent steps
- Cache dependencies
- Use incremental builds
- Split slow tests into separate pipeline

### 2. Fail Fast

**Stop immediately** when a critical step fails:

```yaml
# Don't do this (waste time running all tests when build fails)
jobs:
  build: ...
  test-unit: ...
  test-integration: ...
  test-e2e: ...

# Do this (stop if build fails)
jobs:
  build: ...
  test-unit:
    needs: build
  test-integration:
    needs: build
  test-e2e:
    needs: [test-unit, test-integration]
```

### 3. Make Pipelines Deterministic

**Same input → Same output**

**Bad** (non-deterministic):
```dockerfile
FROM python:latest  # ← Version changes over time
RUN pip install flask  # ← Gets latest version
```

**Good** (deterministic):
```dockerfile
FROM python:3.11.5
COPY requirements.txt .
RUN pip install -r requirements.txt
```

```
# requirements.txt
flask==2.3.3
requests==2.31.0
```

### 4. Use Pipeline as Code

**Don't configure CI in UI**. Put pipeline config in version control:

```
repo/
├── .github/workflows/ci.yml    # GitHub Actions
├── .gitlab-ci.yml              # GitLab
├── Jenkinsfile                 # Jenkins
└── azure-pipelines.yml         # Azure DevOps
```

**Benefits**:
- Version controlled
- Code reviewed
- Documented
- Portable

### 5. Secure Secrets

**Never hardcode secrets** in pipeline files:

```yaml
# Bad
env:
  API_KEY: sk-1234567890abcdef

# Good
env:
  API_KEY: ${{ secrets.API_KEY }}  # GitHub secrets
```

**Use secret management**:
- GitHub Secrets
- GitLab CI/CD Variables (masked)
- Jenkins Credentials
- HashiCorp Vault

### 6. Monitor Pipeline Performance

**Track metrics**:
- Build duration (trend over time)
- Success rate
- Flaky test rate
- Queue time

**Set up alerts** when pipelines degrade.

### 7. Make Pipelines Idempotent

**Running pipeline twice** should produce same result:

```bash
# Bad
echo "build-$(date)" > version.txt  # ← Different every time

# Good
echo "build-$GIT_COMMIT_SHA" > version.txt  # ← Same for same commit
```

### 8. Provide Clear Feedback

**Good error messages**:

```yaml
- name: Check code quality
  run: |
    if ! flake8 src/; then
      echo "❌ Code quality check failed"
      echo "Run 'flake8 src/' locally and fix issues"
      exit 1
    fi
```

**Status badges**:
```markdown
# README.md
![Build Status](https://github.com/user/repo/actions/workflows/ci.yml/badge.svg)
```

### 9. Clean Up Resources

**Don't leak resources**:

```yaml
- name: Start test environment
  run: docker-compose up -d

- name: Run tests
  run: pytest

- name: Cleanup (always runs)
  if: always()
  run: docker-compose down
```

### 10. Test the Pipeline

**Pipeline changes should be tested**:

```yaml
# Test on feature branch before merging
on:
  pull_request:
    paths:
      - '.github/workflows/**'
```

---

## Common Pitfalls and Solutions

### Pitfall 1: Slow Pipelines

**Problem**: Pipeline takes 30+ minutes
**Impact**: Developers don't get quick feedback, avoid running tests

**Solutions**:
- Parallelize steps
- Cache dependencies
- Use faster runners (more CPU/memory)
- Split into fast/slow pipelines

```yaml
# Fast pipeline (runs on every commit)
- Unit tests
- Linting
- Basic security scans

# Slow pipeline (runs nightly or before deploy)
- E2E tests
- Performance tests
- Extensive security scans
```

### Pitfall 2: Flaky Tests

**Problem**: Tests pass/fail randomly
**Impact**: Developers lose trust, ignore failures

**Solutions**:
- Fix or delete flaky tests
- Retry flaky tests automatically (last resort)
- Isolate test environments
- Use test containers

```yaml
# Retry flaky tests (not ideal, but pragmatic)
- name: Run E2E tests
  uses: nick-invision/retry@v2
  with:
    timeout_minutes: 10
    max_attempts: 3
    command: pytest tests/e2e
```

### Pitfall 3: Tight Coupling to CI Platform

**Problem**: Pipeline only works on specific CI platform
**Impact**: Vendor lock-in, hard to migrate

**Solution**: Use standard tools (Makefile, scripts)

```makefile
# Makefile works everywhere
test:
	pytest tests/ -v

build:
	docker build -t myapp:latest .

deploy:
	./deploy.sh
```

```yaml
# CI just calls Makefile
steps:
  - run: make test
  - run: make build
  - run: make deploy
```

**Now easy to switch** CI platforms.

### Pitfall 4: No Rollback Strategy

**Problem**: Deployment breaks production, no way to rollback
**Impact**: Extended outages

**Solution**: Always have rollback plan

```yaml
- name: Deploy
  run: ./deploy.sh

- name: Health check
  run: |
    sleep 30
    curl -f https://api.example.com/health || {
      echo "Health check failed, rolling back"
      ./rollback.sh
      exit 1
    }
```

### Pitfall 5: Ignoring Security Scans

**Problem**: Security scans fail, but deploy anyway
**Impact**: Vulnerabilities in production

**Solution**: Fail pipeline on critical vulnerabilities

```yaml
- name: Security scan
  run: |
    trivy image --severity CRITICAL,HIGH --exit-code 1 myapp:latest
```

---

## ML-Specific CI/CD Considerations

Machine learning adds unique challenges to CI/CD.

### Challenges

1. **Data is code**: Model depends on data, not just code
2. **Non-deterministic**: Training can produce different models
3. **Long-running**: Training takes hours/days
4. **Large artifacts**: Models can be GBs
5. **Model validation**: Need to validate model quality, not just code

### ML Pipeline Stages

```
Data Validation → Data Processing → Training → Model Validation → Model Deployment → Monitoring
```

### Data Validation

**Check data quality** before training:

```python
# data_validation.py
def validate_data(df):
    assert len(df) > 10000, "Not enough data"
    assert df['target'].isna().sum() == 0, "Missing target values"
    assert df['feature'].between(0, 100).all(), "Feature out of range"
```

```yaml
- name: Validate training data
  run: python data_validation.py --data data/train.csv
```

### Model Versioning

**Track model versions** with code versions:

```
Model Registry:
- model-v1.2.3 (trained from git commit abc123, data version v2)
- model-v1.2.4 (trained from git commit def456, data version v2)
```

**Tools**: MLflow, Weights & Biases, Neptune

### Model Validation

**Don't just test code, test model quality**:

```python
# model_validation.py
def validate_model(model, test_data):
    predictions = model.predict(test_data)
    accuracy = accuracy_score(test_data['target'], predictions)

    # Fail if model quality degrades
    assert accuracy > 0.85, f"Model accuracy {accuracy} below threshold"

    # Check for bias
    check_fairness(predictions, test_data)
```

```yaml
- name: Train model
  run: python train.py

- name: Validate model
  run: python model_validation.py --model models/latest.pkl
```

### Continuous Training

**Retrain models** automatically when data updates:

```yaml
# .github/workflows/retrain.yml
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2am

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - name: Fetch latest data
        run: python fetch_data.py

      - name: Validate data
        run: python validate_data.py

      - name: Train model
        run: python train.py

      - name: Validate model
        run: python validate_model.py

      - name: Deploy if better
        run: |
          if python compare_models.py; then
            python deploy_model.py
          fi
```

### A/B Testing Models

**Deploy multiple models**, route traffic for comparison:

```python
# model_server.py
def get_model(user_id):
    if hash(user_id) % 100 < 10:
        return load_model('model-v2')  # 10% traffic
    else:
        return load_model('model-v1')  # 90% traffic
```

**Track metrics** for each model version, promote winner.

---

## Conclusion

**CI/CD is not a tool, it's a practice**. The goal is to make software delivery:
- **Fast**: Deploy in minutes, not weeks
- **Reliable**: Automated testing catches issues early
- **Repeatable**: Same process every time
- **Safe**: Easy to rollback, low-risk deployments

**Start simple**:
1. Automate your build
2. Add automated tests
3. Deploy to staging automatically
4. Gradually add more sophistication

**Remember**: Perfect is the enemy of good. A simple CI/CD pipeline you actually use beats a complex one you don't.

---

## Next Steps

1. **Implement basic CI**: Start with automated testing
2. **Add deployment automation**: Start with dev/staging
3. **Improve monitoring**: You can't improve what you don't measure
4. **Iterate**: Continuously improve your pipeline
5. **Learn from failures**: Post-mortems on pipeline failures

---

**Key Takeaway**: CI/CD enables fast, reliable, low-risk software delivery by automating the path from code to production. Invest in your pipeline—it's the foundation of modern software engineering.
