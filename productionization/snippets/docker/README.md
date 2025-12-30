# Docker and Docker Compose Examples

This directory contains production-ready Docker configurations demonstrating best practices for containerizing applications.

## Files Overview

### Dockerfiles

- **`Dockerfile.python`** - Production-ready Python application with multi-stage build
  - Demonstrates: Layer caching, non-root user, health checks, minimal base image
  - Use case: Flask, FastAPI, Django applications

- **`Dockerfile.nodejs`** - Production-ready Node.js application with multi-stage build
  - Demonstrates: Dependency optimization, TypeScript compilation, signal handling with dumb-init
  - Use case: Express, Next.js, React applications

### Docker Compose Files

- **`docker-compose.yml`** - Complete production stack
  - Services: PostgreSQL, Redis, Web App, Nginx, Prometheus
  - Demonstrates: Health checks, networks, volumes, dependencies, resource limits

- **`docker-compose.dev.yml`** - Development environment overrides
  - Features: Live code reloading, debug ports, development tools (pgAdmin, Redis Commander, Mailhog)
  - Use as override file or reference for development configurations

### Other Files

- **`.dockerignore`** - Excludes unnecessary files from build context
  - Speeds up builds, reduces image size, protects secrets

## Usage Examples

### Building Images

```bash
# Python application
docker build -f Dockerfile.python -t myapp:latest .

# Node.js application
docker build -f Dockerfile.nodejs -t myapp:latest .

# With build arguments
docker build -f Dockerfile.python --build-arg PYTHON_VERSION=3.11 -t myapp:latest .
```

### Running with Docker Compose

```bash
# Production
docker compose up -d

# Development (with override)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# View logs
docker compose logs -f web

# Stop and remove
docker compose down

# Stop and remove including volumes (⚠️ deletes data!)
docker compose down -v
```

### Development Workflow

```bash
# Start development environment
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Access services:
# - Application: http://localhost:8080
# - pgAdmin: http://localhost:5050
# - Redis Commander: http://localhost:8081
# - Mailhog: http://localhost:8025
# - Prometheus: http://localhost:9090

# Rebuild after dependency changes
docker compose up --build

# Execute commands in running container
docker compose exec web bash
docker compose exec web pytest
```

## Key Concepts Demonstrated

### Multi-Stage Builds

Both Dockerfiles use multi-stage builds to:
1. **Builder stage**: Install build dependencies, compile code
2. **Runtime stage**: Copy only compiled artifacts, exclude build tools

**Result**: Final images are 50-70% smaller

### Layer Caching

Dockerfiles order instructions from least to most frequently changed:
1. Base image
2. System dependencies
3. Application dependencies (requirements.txt, package.json)
4. Application code

**Result**: Faster rebuilds when only code changes

### Security Best Practices

- ✅ Non-root users
- ✅ Minimal base images (slim, alpine)
- ✅ No secrets in images
- ✅ Health checks for production readiness
- ✅ Resource limits to prevent DoS

### Network Isolation

Docker Compose uses two networks:
- **frontend**: nginx ↔ web
- **backend**: web ↔ database/redis

**Result**: Database not directly accessible from nginx (defense in depth)

### Data Persistence

Named volumes persist data across container restarts:
- `postgres_data`: Database files
- `redis_data`: Redis persistence
- `uploads`: User-uploaded files

## Customization Guide

### Adapting for Your Application

1. **Update base images** in Dockerfiles to match your language/framework
2. **Modify dependencies** installation commands (pip, npm, etc.)
3. **Adjust ports** in docker-compose.yml to match your app's configuration
4. **Add/remove services** as needed (remove nginx if not needed, add Elasticsearch, etc.)
5. **Update health check endpoints** to match your application's health check route
6. **Configure environment variables** in .env file (see .env.example)

### Environment Variables

Create a `.env` file in the same directory as docker-compose.yml:

```bash
# .env
DB_PASSWORD=secure_password_here
SECRET_KEY=your_secret_key
API_KEY=your_api_key
```

**⚠️ Never commit .env to version control!**

## Common Issues and Solutions

### Issue: Container exits immediately

```bash
# Check logs
docker compose logs web

# Run interactively to debug
docker compose run web bash
```

### Issue: Cannot connect to database

1. Check service is healthy: `docker compose ps`
2. Verify environment variables: `docker compose config`
3. Check network: `docker network inspect [network_name]`

### Issue: Port already in use

```bash
# Change port mapping in docker-compose.yml
ports:
  - "8081:8080"  # Use different host port
```

### Issue: Changes not reflected

```bash
# Rebuild image
docker compose up --build

# For dev environment, check volume mounts are correct
```

## Production Deployment Checklist

Before deploying to production:

- [ ] Use specific image tags (not `:latest`)
- [ ] Set `restart: unless-stopped` for all services
- [ ] Configure resource limits (CPU, memory)
- [ ] Enable health checks for all services
- [ ] Use secrets management (AWS Secrets Manager, Vault)
- [ ] Set up log aggregation (ELK, CloudWatch)
- [ ] Configure backups for volumes
- [ ] Use HTTPS with valid SSL certificates
- [ ] Review security (non-root users, minimal images, no secrets in images)
- [ ] Set up monitoring (Prometheus, Grafana)

## Learning Resources

For comprehensive explanations, see:
- **Deep dive**: `/productionization/notes/docker-containerization.md`
- **Quick reference**: `/interview-prep/notes/docker-cheatsheet.md`
- **Additional resources**: `/productionization/resources.md`

## Notes

- These examples assume the application code exists (app.py, package.json, etc.)
- Adapt paths and configurations to match your project structure
- Comments explain the "why" behind each configuration choice
- Use these as templates and modify based on your specific needs
