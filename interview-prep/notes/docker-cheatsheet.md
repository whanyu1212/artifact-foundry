# Docker & Docker Compose Cheatsheet - Interview Prep

**Quick reference for Docker interviews and deployment scenarios**

---

## Table of Contents

1. [Key Concepts](#key-concepts)
2. [Docker Images](#docker-images)
3. [Docker Containers](#docker-containers)
4. [Dockerfile Commands](#dockerfile-commands)
5. [Docker Networks](#docker-networks)
6. [Docker Volumes](#docker-volumes)
7. [Docker Compose](#docker-compose)
8. [Common Commands](#common-commands)
9. [Best Practices](#best-practices)
10. [Interview Questions](#interview-questions)

---

## Key Concepts

### Containers vs VMs

| Feature | Containers | Virtual Machines |
|---------|-----------|------------------|
| **Isolation** | Process-level (namespaces) | Full OS isolation |
| **Size** | MBs | GBs |
| **Startup** | Seconds | Minutes |
| **Resource** | Share host kernel | Each has full OS |
| **Performance** | Near-native | Some overhead |

### Core Components

- **Image**: Read-only template to create containers
- **Container**: Running instance of an image
- **Dockerfile**: Instructions to build an image
- **Registry**: Storage for images (Docker Hub, ECR, GCR)
- **Volume**: Persistent data storage
- **Network**: Container communication

---

## Docker Images

### Image Naming
```
[registry/][username/]repository[:tag]

Examples:
ubuntu:22.04                    # Official from Docker Hub
myusername/myapp:v1.0          # User image
gcr.io/project/myapp:latest    # Google Container Registry
```

### Image Commands

```bash
# Pull image
docker pull ubuntu:22.04

# List images
docker images
docker images -a                # Include intermediate layers

# Build image
docker build -t myapp:latest .
docker build -t myapp:v1.0 --no-cache .  # Fresh build

# Tag image
docker tag myapp:latest myapp:v1.0

# Push to registry
docker push username/myapp:v1.0

# Remove image
docker rmi myapp:latest
docker rmi $(docker images -q)  # Remove all

# Inspect image
docker inspect myapp:latest
docker history myapp:latest     # View layers
```

---

## Docker Containers

### Container Lifecycle

```
Created → Running → Paused → Stopped → Removed
```

### Container Commands

```bash
# Run container
docker run ubuntu:22.04                    # Run and exit
docker run -it ubuntu:22.04 /bin/bash     # Interactive
docker run -d nginx                        # Detached (background)
docker run --rm alpine echo "hello"        # Auto-remove after exit
docker run --name mycontainer nginx        # Named container

# With options
docker run -d \
  --name web \
  -p 8080:80 \
  -e ENV=production \
  --env-file .env \
  -v /host/path:/container/path \
  --memory="512m" \
  --cpus="1.5" \
  nginx

# List containers
docker ps                      # Running containers
docker ps -a                   # All containers

# Start/Stop
docker start mycontainer
docker stop mycontainer
docker restart mycontainer
docker pause mycontainer       # Pause processes
docker unpause mycontainer

# Execute command in running container
docker exec -it mycontainer /bin/bash
docker exec mycontainer ls /app

# View logs
docker logs mycontainer
docker logs -f mycontainer     # Follow logs
docker logs --tail 100 mycontainer

# Inspect container
docker inspect mycontainer
docker stats mycontainer       # Resource usage
docker top mycontainer         # Processes

# Copy files
docker cp mycontainer:/app/file.txt ./
docker cp ./file.txt mycontainer:/app/

# Remove container
docker rm mycontainer
docker rm -f mycontainer       # Force remove running
docker rm $(docker ps -aq)     # Remove all stopped
```

---

## Dockerfile Commands

### Basic Structure

```dockerfile
# Base image
FROM python:3.11-slim

# Metadata
LABEL maintainer="email@example.com"
LABEL version="1.0"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    APP_HOME=/app

# Working directory
WORKDIR $APP_HOME

# Copy files
COPY requirements.txt .
COPY . .

# Run commands
RUN pip install --no-cache-dir -r requirements.txt

# Create user
RUN useradd -m appuser
USER appuser

# Expose port (documentation only)
EXPOSE 8000

# Default command
CMD ["python", "app.py"]
```

### All Dockerfile Instructions

| Instruction | Purpose | Example |
|------------|---------|---------|
| `FROM` | Base image | `FROM python:3.11-slim` |
| `RUN` | Execute command during build | `RUN apt-get update` |
| `COPY` | Copy files from host | `COPY app.py /app/` |
| `ADD` | Like COPY + tar extraction | `ADD archive.tar.gz /app/` |
| `WORKDIR` | Set working directory | `WORKDIR /app` |
| `ENV` | Set environment variable | `ENV DEBUG=false` |
| `EXPOSE` | Document port | `EXPOSE 8000` |
| `CMD` | Default command (overridable) | `CMD ["python", "app.py"]` |
| `ENTRYPOINT` | Command prefix | `ENTRYPOINT ["python"]` |
| `USER` | Switch user | `USER appuser` |
| `ARG` | Build-time variable | `ARG VERSION=1.0` |
| `VOLUME` | Mount point | `VOLUME /data` |
| `HEALTHCHECK` | Health check | `HEALTHCHECK CMD curl localhost/` |

### CMD vs ENTRYPOINT

```dockerfile
# CMD - can be overridden
CMD ["python", "app.py"]
# docker run myapp other.py  → runs other.py

# ENTRYPOINT - always runs
ENTRYPOINT ["python"]
CMD ["app.py"]
# docker run myapp other.py  → runs: python other.py
```

### Multi-Stage Build (Optimize Size)

```dockerfile
# Stage 1: Build
FROM python:3.11 as builder
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-deps --wheel-dir /wheels -r requirements.txt

# Stage 2: Runtime (smaller)
FROM python:3.11-slim
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
COPY . .
CMD ["python", "app.py"]
```

### .dockerignore

```
.git
.env
__pycache__
*.pyc
node_modules/
.DS_Store
*.log
```

---

## Docker Networks

### Network Types

| Type | Description | Use Case |
|------|-------------|----------|
| `bridge` | Default, private network | Single host, container isolation |
| `host` | Use host's network | No isolation, better performance |
| `none` | No networking | Isolated container |
| `overlay` | Multi-host networking | Swarm, Kubernetes |

### Network Commands

```bash
# Create network
docker network create mynetwork
docker network create --driver bridge mynetwork

# List networks
docker network ls

# Connect container to network
docker run -d --name web --network mynetwork nginx

# Connect existing container
docker network connect mynetwork mycontainer

# Disconnect
docker network disconnect mynetwork mycontainer

# Inspect network
docker network inspect mynetwork

# Remove network
docker network rm mynetwork
```

### Inter-Container Communication

```bash
# Create network
docker network create app-net

# Run containers on same network
docker run -d --name db --network app-net postgres
docker run -d --name app --network app-net myapp

# app can reach db by name: postgresql://db:5432/mydb
```

---

## Docker Volumes

### Volume Types

1. **Named Volumes** (managed by Docker) - Recommended
2. **Bind Mounts** (host directory) - Development
3. **tmpfs** (memory only) - Temporary data

### Volume Commands

```bash
# Create volume
docker volume create mydata

# List volumes
docker volume ls

# Inspect volume
docker volume inspect mydata

# Use volume
docker run -d -v mydata:/var/lib/postgresql/data postgres

# Bind mount (development)
docker run -d -v $(pwd):/app myapp
docker run -d -v /host/path:/container/path myapp

# Read-only mount
docker run -d -v mydata:/data:ro nginx

# Remove volume
docker volume rm mydata
docker volume prune  # Remove unused volumes
```

### Backup and Restore

```bash
# Backup volume
docker run --rm \
  -v mydata:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/backup.tar.gz -C /data .

# Restore volume
docker run --rm \
  -v mydata:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/backup.tar.gz -C /data
```

---

## Docker Compose

### Basic docker-compose.yml

```yaml
version: '3.8'

services:
  # Database
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: ${DB_PASSWORD}  # From .env
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - backend
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    networks:
      - backend

  # Web Application
  web:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        BUILD_ENV: production
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://user:${DB_PASSWORD}@db:5432/myapp
      REDIS_URL: redis://redis:6379
    volumes:
      - ./uploads:/app/uploads
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - backend
      - frontend
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '2'

volumes:
  postgres_data:
  redis_data:

networks:
  backend:
  frontend:
```

### Docker Compose Commands

```bash
# Start services
docker compose up                    # Foreground
docker compose up -d                 # Background (detached)
docker compose up --build            # Rebuild before starting
docker compose up -d --scale web=3   # Scale service

# Stop services
docker compose stop                  # Stop (don't remove)
docker compose down                  # Stop and remove
docker compose down -v               # Also remove volumes

# View status
docker compose ps                    # Running services
docker compose logs                  # All logs
docker compose logs -f web           # Follow specific service

# Execute commands
docker compose exec web bash         # Shell in running service
docker compose run web pytest        # One-off command

# Other commands
docker compose restart
docker compose pause
docker compose unpause
docker compose config                # Validate and view config
```

### Environment Variables

```yaml
# Method 1: .env file (auto-loaded)
services:
  web:
    environment:
      DB_PASSWORD: ${DB_PASSWORD}

# Method 2: Separate env file
services:
  web:
    env_file:
      - .env.production

# Method 3: Inline
services:
  web:
    environment:
      - DEBUG=true
      - LOG_LEVEL=info
```

### Override Files

```bash
# Base: docker-compose.yml
# Dev: docker-compose.override.yml (auto-loaded)
# Prod: docker-compose.prod.yml

# Use specific file
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Common Commands

### Cleanup

```bash
# Remove all stopped containers
docker container prune

# Remove unused images
docker image prune
docker image prune -a  # Remove all unused, not just dangling

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Remove everything
docker system prune
docker system prune -a --volumes  # Nuclear option

# Check disk usage
docker system df
```

### Debugging

```bash
# View container logs
docker logs -f mycontainer

# Enter running container
docker exec -it mycontainer /bin/bash

# Inspect container details
docker inspect mycontainer

# View resource usage
docker stats mycontainer

# View processes
docker top mycontainer

# View port mappings
docker port mycontainer

# View filesystem changes
docker diff mycontainer

# Export container
docker export mycontainer > container.tar
```

---

## Best Practices

### Image Optimization

```dockerfile
# ✅ Use slim/alpine base images
FROM python:3.11-slim

# ✅ Combine RUN commands
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# ✅ Order layers: least → most frequently changed
COPY requirements.txt .       # Changes rarely
RUN pip install -r requirements.txt
COPY . .                      # Changes often

# ✅ Use .dockerignore
# Exclude .git, node_modules, __pycache__, etc.

# ✅ Multi-stage builds
FROM builder as build
# ... build steps
FROM runtime
COPY --from=build /app/dist ./
```

### Security

```dockerfile
# ✅ Don't run as root
RUN useradd -m appuser
USER appuser

# ✅ Use specific image tags
FROM python:3.11-slim  # Not :latest

# ✅ Scan images
docker scan myapp:latest

# ✅ No secrets in images
# Use environment variables at runtime
```

### Production

```yaml
# ✅ Health checks
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost/health"]
  interval: 30s
  timeout: 3s
  retries: 3

# ✅ Resource limits
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '2'

# ✅ Restart policy
restart: unless-stopped

# ✅ Logging
# Log to stdout/stderr, configure log driver

# ✅ Use specific versions, not :latest
image: myapp:1.2.3  # Not myapp:latest
```

---

## Interview Questions

### Q1: What's the difference between containers and VMs?

**Answer**: Containers share the host OS kernel and provide process-level isolation using namespaces and cgroups. VMs include a full OS and use a hypervisor for isolation. Containers are lighter (MBs vs GBs) and start faster (seconds vs minutes).

---

### Q2: Explain Docker layers and image caching

**Answer**: Each Dockerfile instruction creates a read-only layer. Layers are cached and reused if unchanged, speeding up builds. Order instructions from least to most frequently changed. Containers add a writable layer on top.

```dockerfile
FROM python:3.11        # Layer 1 (cached)
COPY requirements.txt   # Layer 2 (cached if file unchanged)
RUN pip install ...     # Layer 3 (cached if layer 2 unchanged)
COPY . .               # Layer 4 (changes often, rebuilt)
```

---

### Q3: What's the difference between CMD and ENTRYPOINT?

**Answer**:
- **CMD**: Default command, can be overridden by user
- **ENTRYPOINT**: Always runs, CMD becomes arguments

```dockerfile
# Just CMD
CMD ["python", "app.py"]
# docker run myapp → python app.py
# docker run myapp other.py → other.py

# ENTRYPOINT + CMD
ENTRYPOINT ["python"]
CMD ["app.py"]
# docker run myapp → python app.py
# docker run myapp other.py → python other.py
```

---

### Q4: How do containers communicate with each other?

**Answer**: Use user-defined networks. Containers on the same network can communicate using service names as hostnames via built-in DNS.

```bash
docker network create mynet
docker run --name db --network mynet postgres
docker run --name app --network mynet myapp
# app connects to: postgresql://db:5432/mydb
```

---

### Q5: How do you persist data in containers?

**Answer**: Use volumes (not container filesystem, which is ephemeral).

- **Named volumes**: Managed by Docker, recommended for production
- **Bind mounts**: Mount host directory, good for development
- **tmpfs**: In-memory, temporary data

```bash
docker volume create mydata
docker run -v mydata:/var/lib/postgresql/data postgres
```

---

### Q6: What's a multi-stage build and why use it?

**Answer**: Separate build dependencies from runtime. Final image only contains what's needed to run, reducing size and attack surface.

```dockerfile
FROM node:18 as builder
RUN npm install && npm run build

FROM node:18-alpine
COPY --from=builder /app/dist ./dist
# Final image doesn't include dev dependencies
```

---

### Q7: How do you debug a container that exits immediately?

**Answer**:
1. Check logs: `docker logs <container>`
2. Check exit code: `docker ps -a`
3. Run interactively: `docker run -it myapp /bin/bash`
4. Override CMD: `docker run -it myapp sh`

Common causes: missing dependencies, wrong CMD, application crash

---

### Q8: Explain depends_on in Docker Compose

**Answer**: Controls startup order, but doesn't wait for service to be ready.

```yaml
# Simple dependency (starts after, doesn't wait)
depends_on:
  - db

# Wait for healthy (requires healthcheck)
depends_on:
  db:
    condition: service_healthy
```

---

### Q9: How do you handle secrets in Docker?

**Answer**:
- ❌ Never hardcode in Dockerfile/image
- ✅ Pass at runtime via environment variables
- ✅ Use secret management tools (AWS Secrets Manager, Vault)
- ✅ Use Docker secrets (Swarm mode)

```bash
docker run -e DB_PASSWORD=$(cat secret.txt) myapp
```

---

### Q10: What are namespaces and cgroups?

**Answer**:
- **Namespaces**: Provide isolation (PID, network, mount, user, IPC, UTS)
  - Each container sees its own process tree, network interfaces, filesystem
- **cgroups**: Limit and monitor resources (CPU, memory, disk I/O)
  - Prevent containers from using excessive resources

These are Linux kernel features that enable containerization.

---

### Q11: How do you optimize Docker image size?

**Answer**:
1. Use slim/alpine base images
2. Multi-stage builds
3. Combine RUN commands (fewer layers)
4. Clean up in same layer: `apt-get update && apt-get install && rm -rf /var/lib/apt/lists/*`
5. Use `.dockerignore`
6. Remove build dependencies in final stage

---

### Q12: What's the difference between COPY and ADD?

**Answer**:
- **COPY**: Simple copy from host to image
- **ADD**: COPY + tar extraction + URL download

**Best practice**: Use COPY unless you need ADD's extra features.

```dockerfile
COPY app.py /app/           # Preferred
ADD archive.tar.gz /app/    # Auto-extracts tar
```

---

## Common Patterns

### Pattern: Development with Live Reload

```bash
# Mount code directory for live changes
docker run -d \
  -v $(pwd):/app \
  -p 8080:8080 \
  myapp:dev
```

### Pattern: Database with Init Scripts

```yaml
services:
  db:
    image: postgres
    volumes:
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
      - postgres_data:/var/lib/postgresql/data
```

### Pattern: Multi-Service Stack

```yaml
services:
  nginx:      # Reverse proxy
    depends_on: [web]
  web:        # Application
    depends_on: [db, redis]
  db:         # Database
  redis:      # Cache
```

### Pattern: Environment-Specific Config

```bash
# Development
docker compose up

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Quick Reference Card

```bash
# Images
docker build -t name .
docker pull image
docker push image
docker images
docker rmi image

# Containers
docker run -d -p 8080:80 --name web nginx
docker ps
docker stop/start/restart name
docker exec -it name bash
docker logs -f name
docker rm name

# Networks
docker network create net
docker run --network net name

# Volumes
docker volume create vol
docker run -v vol:/data name

# Compose
docker compose up -d
docker compose down
docker compose logs -f
docker compose exec service bash

# Cleanup
docker system prune -a
docker volume prune
```

---

**Last Updated**: 2025-12-30
