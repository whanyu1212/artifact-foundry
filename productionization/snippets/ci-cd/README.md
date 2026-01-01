# CI/CD Pipeline Snippets

This directory contains comprehensive, commented examples of CI/CD pipelines for different platforms and use cases.

## Contents

### 1. `github-actions-python.yml`
**Complete GitHub Actions pipeline for Python applications**

Demonstrates:
- Multi-stage pipeline (build, test, scan, deploy)
- Code quality checks (linting, formatting, type checking)
- Testing with service containers (PostgreSQL, Redis)
- Security scanning (dependency vulnerabilities, SAST, container scanning)
- Docker image building and pushing
- Deployment to staging and production with approval gates
- Extensive inline comments explaining each section

**Key Features:**
- Matrix strategy for testing multiple Python versions
- Caching to speed up builds
- Artifact management
- Environment protection rules
- Notifications and reporting

---

### 2. `gitlab-ci-python.yml`
**Complete GitLab CI/CD pipeline for Python applications**

Demonstrates:
- GitLab-specific syntax (stages, services, artifacts)
- Built-in security scanning (SAST, secret detection, container scanning)
- Pipeline templates and job dependencies
- Coverage reporting integrated into GitLab UI
- Manual approval for production deployments
- Extensive inline comments explaining each section

**Key Features:**
- GitLab Container Registry integration
- Service containers for integration tests
- Artifacts and reports for visualization
- Environment tracking
- Scheduled pipelines

---

### 3. `jenkins-pipeline.groovy`
**Complete Jenkins declarative pipeline**

Demonstrates:
- Declarative pipeline syntax
- Docker agents for build isolation
- Parameters for manual triggering
- Post-build actions and notifications
- Multi-environment deployment (staging, production)
- Approval gates for production
- Extensive inline comments explaining each section

**Key Features:**
- Docker integration
- Kubernetes deployment
- Credential management
- HTML report publishing
- Email and Slack notifications

---

### 4. `ml-pipeline-example.yml`
**Machine Learning-specific CI/CD pipeline**

Demonstrates:
- Data validation and versioning (DVC)
- Model training and evaluation
- Model quality gates (accuracy, F1, inference time)
- Model registry integration (MLflow)
- Fairness/bias checks
- Canary deployment for models
- Extensive inline comments explaining ML-specific concerns

**Key Features:**
- DVC for data version control
- MLflow for experiment tracking
- Model versioning and registry
- A/B testing support
- Performance monitoring

---

## How to Use These Snippets

### 1. As Learning Material
Read through the files to understand:
- Pipeline structure and stages
- Best practices for each platform
- Security considerations
- Deployment strategies

### 2. As Templates
Copy and adapt for your projects:
1. Copy the relevant file to your repository
2. Customize environment variables
3. Adjust stages for your needs
4. Configure secrets in your CI platform

### 3. As Reference
Use when implementing specific features:
- "How do I set up service containers?"
- "What's the syntax for manual approval?"
- "How do I cache dependencies?"

---

## Platform Comparison

| Feature | GitHub Actions | GitLab CI/CD | Jenkins |
|---------|---------------|--------------|---------|
| **Syntax** | YAML (declarative) | YAML (declarative) | Groovy (declarative or scripted) |
| **Hosting** | Cloud (free tier) | Cloud or self-hosted | Self-hosted |
| **Docker Support** | ✅ Native | ✅ Native | ✅ Plugin |
| **Secret Management** | GitHub Secrets | CI/CD Variables | Credentials |
| **Caching** | Built-in | Built-in | Plugin |
| **Matrix Builds** | ✅ Strategy | ✅ Matrix | ✅ Manual |
| **Environment Protection** | ✅ Native | ✅ Native | Manual |
| **Security Scanning** | Actions marketplace | Built-in (Ultimate) | Plugins |
| **Learning Curve** | Low | Low | Medium-High |

---

## Best Practices Demonstrated

### Security
- ✅ No hardcoded secrets
- ✅ Dependency vulnerability scanning
- ✅ Container image scanning
- ✅ SAST (Static Application Security Testing)
- ✅ Least privilege principles

### Testing
- ✅ Fast feedback (fail fast)
- ✅ Test isolation (service containers)
- ✅ Coverage reporting
- ✅ Multiple test types (unit, integration, E2E)

### Deployment
- ✅ Build once, deploy everywhere (immutable artifacts)
- ✅ Environment progression (dev → staging → production)
- ✅ Approval gates for production
- ✅ Automated rollback on health check failure
- ✅ Zero-downtime deployments

### Performance
- ✅ Dependency caching
- ✅ Parallel execution
- ✅ Incremental builds
- ✅ Minimal Docker image layers

---

## Common Customizations

### Adjust Python Version
```yaml
env:
  PYTHON_VERSION: '3.11'  # Change to your version
```

### Add More Test Types
```yaml
- name: Run performance tests
  run: pytest tests/performance -v

- name: Run security tests
  run: pytest tests/security -v
```

### Change Docker Registry
```yaml
env:
  DOCKER_REGISTRY: 'gcr.io'  # Google Container Registry
  # or
  DOCKER_REGISTRY: 'public.ecr.aws'  # AWS ECR Public
```

### Add Notification Channels
```yaml
- name: Notify Slack
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Prerequisites

### For GitHub Actions
- GitHub repository
- GitHub Actions enabled
- Secrets configured in repository settings

### For GitLab CI/CD
- GitLab repository
- GitLab Runner configured (or use shared runners)
- CI/CD variables configured in project settings

### For Jenkins
- Jenkins server installed
- Docker plugin installed
- Credentials configured in Jenkins
- Jenkinsfile in repository root

### For ML Pipeline
- MLflow server for experiment tracking
- DVC for data versioning
- Cloud storage (S3, GCS) for data/models
- GPU resources for training (optional but recommended)

---

## Testing the Pipelines

### Local Testing

**GitHub Actions:**
```bash
# Use act to run GitHub Actions locally
brew install act
act -j test  # Run 'test' job
```

**GitLab CI/CD:**
```bash
# Use gitlab-runner locally
gitlab-runner exec docker test:unit
```

**Jenkins:**
- Use Jenkins Pipeline Linter in Jenkins UI
- Or test with local Jenkins in Docker

---

## Further Reading

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitLab CI/CD Documentation](https://docs.gitlab.com/ee/ci/)
- [Jenkins Pipeline Documentation](https://www.jenkins.io/doc/book/pipeline/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [DVC Documentation](https://dvc.org/doc)

---

## Notes

- All examples use Python, but concepts apply to any language
- Adjust resource limits (memory, CPU) based on your needs
- Security scanning tools may require licenses (check vendor documentation)
- Production deployments should have stricter controls than shown (customize to your org's requirements)
- ML pipelines may need significant compute resources (consider cloud GPU instances)

---

**Remember:** These are educational examples with extensive comments. In production, you might:
- Remove some comments for brevity
- Add organization-specific steps
- Integrate with your monitoring/alerting systems
- Implement more sophisticated deployment strategies
- Add compliance/audit steps
