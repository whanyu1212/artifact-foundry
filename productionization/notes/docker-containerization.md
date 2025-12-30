# Docker and Containerization

**Last Updated**: 2025-12-30

A comprehensive guide to understanding Docker and containerization from fundamentals to production deployment.

---

## Table of Contents

1. [What Are Containers?](#what-are-containers)
2. [Why Containers Matter](#why-containers-matter)
3. [How Containers Work Internally](#how-containers-work-internally)
4. [Docker Architecture](#docker-architecture)
5. [Core Docker Concepts](#core-docker-concepts)
6. [Docker Images](#docker-images)
7. [Docker Containers](#docker-containers)
8. [Dockerfile Best Practices](#dockerfile-best-practices)
9. [Docker Networking](#docker-networking)
10. [Docker Volumes and Data Persistence](#docker-volumes-and-data-persistence)
11. [Docker Compose](#docker-compose)
12. [Multi-Stage Builds](#multi-stage-builds)
13. [Container Orchestration](#container-orchestration)
14. [Security Considerations](#security-considerations)
15. [Production Best Practices](#production-best-practices)
16. [Common Patterns and Anti-Patterns](#common-patterns-and-anti-patterns)
17. [Debugging and Troubleshooting](#debugging-and-troubleshooting)

---

## What Are Containers?

**Containers** are lightweight, standalone, executable packages that include everything needed to run an application: code, runtime, system tools, libraries, and settings.

### Key Characteristics

- **Isolated**: Each container runs in its own isolated environment
- **Portable**: Runs consistently across different environments (dev, staging, prod)
- **Lightweight**: Shares the host OS kernel, unlike VMs which need full OS
- **Ephemeral**: Designed to be disposable and replaceable

### Containers vs Virtual Machines

```
Virtual Machine Architecture:
┌─────────────────────────────────────┐
│         Application Layer           │
├─────────────────────────────────────┤
│    Guest OS (Full Operating System) │
├─────────────────────────────────────┤
│         Hypervisor (VMware, etc)    │
├─────────────────────────────────────┤
│         Host Operating System       │
├─────────────────────────────────────┤
│         Physical Hardware           │
└─────────────────────────────────────┘

Container Architecture:
┌─────────────────────────────────────┐
│         Application Layer           │
├─────────────────────────────────────┤
│    Container Runtime (Docker)       │
├─────────────────────────────────────┤
│         Host Operating System       │
├─────────────────────────────────────┤
│         Physical Hardware           │
└─────────────────────────────────────┘
```

**Key Differences**:
- **VMs**: Each has full OS (GBs), slow startup (minutes), complete isolation
- **Containers**: Share host kernel (MBs), fast startup (seconds), process-level isolation

---

## Why Containers Matter

### The "It Works on My Machine" Problem

Before containers:
```
Developer's machine: Python 3.8, specific libraries, macOS
Staging server: Python 3.7, different library versions, Ubuntu
Production server: Python 3.9, yet different versions, RHEL

Result: Inconsistencies, bugs that only appear in certain environments
```

With containers:
```
Same container image runs everywhere
All dependencies bundled inside
Guaranteed consistency from dev to prod
```

### Benefits

1. **Consistency**: Same environment everywhere
2. **Isolation**: Dependencies don't conflict
3. **Portability**: Run anywhere Docker is installed
4. **Resource Efficiency**: Much lighter than VMs
5. **Scalability**: Spin up/down instances quickly
6. **DevOps Integration**: CI/CD pipelines, microservices
7. **Version Control**: Infrastructure as code

---

## How Containers Work Internally

Containers leverage Linux kernel features to create isolated environments. Understanding these internals helps debug issues and optimize deployments.

### Linux Namespaces

**Namespaces** provide isolation by giving processes their own view of system resources.

**Types of Namespaces**:

1. **PID Namespace**: Process isolation
   - Container sees its main process as PID 1
   - Can't see host processes

2. **Network Namespace**: Network isolation
   - Own network interfaces, IP addresses, routing tables
   - Containers can have same port bindings without conflict

3. **Mount Namespace**: Filesystem isolation
   - Own filesystem hierarchy
   - Can't access host files unless explicitly mounted

4. **UTS Namespace**: Hostname/domain isolation
   - Each container can have its own hostname

5. **IPC Namespace**: Inter-process communication isolation
   - Shared memory, semaphores, message queues isolated

6. **User Namespace**: User ID isolation
   - Root in container != root on host (when configured properly)

### Control Groups (cgroups)

**cgroups** limit and monitor resource usage.

**What they control**:
- **CPU**: Limit CPU time, set priorities
- **Memory**: Set memory limits, prevent OOM on host
- **Disk I/O**: Limit read/write speeds
- **Network**: Bandwidth control

**Example**: Limit container to 512MB RAM and 50% of one CPU core
```bash
docker run --memory="512m" --cpus="0.5" myapp
```

### Union Filesystems (UnionFS)

Docker uses layered filesystems to make images efficient.

**How it works**:
```
Container Writable Layer  ← Changes made during runtime
    ↓
Image Layer 3: App code
    ↓
Image Layer 2: Dependencies
    ↓
Image Layer 1: Base OS
    ↓
Base Image
```

**Benefits**:
- **Deduplication**: Multiple containers share base layers
- **Fast builds**: Only rebuild changed layers
- **Efficient storage**: Layers are reused

**Copy-on-Write (CoW)**:
- Read from lower layers
- Write creates a copy in the top writable layer
- Original layers remain unchanged

---

## Docker Architecture

### Components

```
┌──────────────────────────────────────────────────┐
│              Docker Client (CLI)                 │
│              docker build, run, push             │
└────────────────┬─────────────────────────────────┘
                 │ REST API
┌────────────────▼─────────────────────────────────┐
│              Docker Daemon (dockerd)             │
│  • Manages containers, images, networks, volumes │
│  • Builds images from Dockerfiles                │
│  • Communicates with containerd                  │
└────────────────┬─────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────┐
│              containerd                          │
│  • Container lifecycle management                │
│  • Image distribution                            │
│  • Lower-level than Docker daemon                │
└────────────────┬─────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────┐
│              runc                                │
│  • Low-level container runtime                   │
│  • Creates and runs containers                   │
│  • Implements OCI specification                  │
└──────────────────────────────────────────────────┘
```

### Workflow

1. **Client**: User runs `docker run myapp`
2. **Daemon**: Receives request, checks for image locally
3. **Registry**: If not found, pulls from Docker Hub
4. **containerd**: Manages container lifecycle
5. **runc**: Creates container using namespaces and cgroups
6. **Container**: Application runs in isolated environment

---

## Core Docker Concepts

### Images

**Immutable templates** used to create containers. Think of them as "classes" in OOP.

```bash
# Pull an image from Docker Hub
docker pull ubuntu:22.04

# List local images
docker images

# Remove an image
docker rmi ubuntu:22.04
```

### Containers

**Running instances** of images. Think of them as "objects" in OOP.

```bash
# Create and start a container
docker run -d --name mycontainer ubuntu:22.04

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop mycontainer

# Remove a container
docker rm mycontainer
```

### Registry

**Storage and distribution system** for images (e.g., Docker Hub, AWS ECR, Google GCR).

```bash
# Login to Docker Hub
docker login

# Tag an image for registry
docker tag myapp:latest username/myapp:v1.0

# Push to registry
docker push username/myapp:v1.0

# Pull from registry
docker pull username/myapp:v1.0
```

---

## Docker Images

### Image Naming Convention

```
[registry/][username/]repository[:tag]

Examples:
ubuntu:22.04                           # Official image from Docker Hub
docker.io/library/python:3.11-slim    # Full name of official Python image
myusername/myapp:v1.0                 # User image on Docker Hub
gcr.io/myproject/myapp:latest         # Google Container Registry
localhost:5000/myapp:dev              # Private local registry
```

### Image Layers

Each Dockerfile instruction creates a layer:

```dockerfile
FROM python:3.11-slim              # Layer 1: Base image
RUN apt-get update && apt-get install -y git  # Layer 2: Add git
COPY requirements.txt .            # Layer 3: Add requirements file
RUN pip install -r requirements.txt  # Layer 4: Install dependencies
COPY . .                          # Layer 5: Add application code
```

**Inspect layers**:
```bash
docker history myapp:latest
```

### Image Optimization

**Before** (Bad - Large image):
```dockerfile
FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y pip
RUN pip install flask
COPY . .
CMD ["python3", "app.py"]
# Result: ~500MB
```

**After** (Good - Optimized):
```dockerfile
FROM python:3.11-slim              # Use slim base
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*    # Clean up in same layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python3", "app.py"]
# Result: ~150MB
```

**Key optimizations**:
- Use slim/alpine base images
- Combine RUN commands to reduce layers
- Clean up package manager caches in same layer
- Use `.dockerignore` to exclude unnecessary files
- Order layers from least to most frequently changed

---

## Docker Containers

### Container Lifecycle

```
Created → Running → Paused → Stopped → Removed
   ↓         ↓         ↓         ↓
  start    pause    unpause   start
```

### Running Containers

**Interactive mode** (for development):
```bash
# Run with interactive terminal
docker run -it ubuntu:22.04 /bin/bash

# Run in background (detached)
docker run -d nginx

# Run with auto-removal after exit
docker run --rm ubuntu:22.04 echo "Hello"
```

**With resource limits**:
```bash
docker run \
  --memory="512m" \
  --cpus="1.5" \
  --name myapp \
  -d myapp:latest
```

**With environment variables**:
```bash
docker run \
  -e DATABASE_URL="postgres://..." \
  -e DEBUG=true \
  --env-file .env \
  myapp:latest
```

**With port mapping**:
```bash
# Map container port 80 to host port 8080
docker run -p 8080:80 nginx

# Map to random host port
docker run -P nginx

# Bind to specific interface
docker run -p 127.0.0.1:8080:80 nginx
```

### Executing Commands in Running Containers

```bash
# Open a shell in running container
docker exec -it mycontainer /bin/bash

# Run a single command
docker exec mycontainer ls /app

# Run as different user
docker exec -u root mycontainer apt-get update
```

### Viewing Logs

```bash
# View all logs
docker logs mycontainer

# Follow logs (like tail -f)
docker logs -f mycontainer

# Show last 100 lines
docker logs --tail 100 mycontainer

# Show logs with timestamps
docker logs -t mycontainer
```

### Inspecting Containers

```bash
# Get all container details (JSON)
docker inspect mycontainer

# Get specific field
docker inspect -f '{{.NetworkSettings.IPAddress}}' mycontainer

# View resource usage
docker stats mycontainer

# View processes inside container
docker top mycontainer
```

---

## Dockerfile Best Practices

### Basic Dockerfile Structure

```dockerfile
# 1. Base image
FROM python:3.11-slim

# 2. Metadata
LABEL maintainer="your-email@example.com"
LABEL version="1.0"
LABEL description="My Python application"

# 3. Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOME=/app

# 4. Working directory
WORKDIR $APP_HOME

# 5. Install system dependencies (if needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# 6. Copy dependency files first (for caching)
COPY requirements.txt .

# 7. Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 8. Copy application code
COPY . .

# 9. Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser $APP_HOME
USER appuser

# 10. Expose ports
EXPOSE 8000

# 11. Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 12. Default command
CMD ["python", "app.py"]
```

### Layer Caching Strategy

**Order matters!** Place instructions in order from least to most frequently changed:

```dockerfile
FROM python:3.11-slim

# 1. System packages (rarely change)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# 2. Application dependencies (change occasionally)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Application code (changes frequently)
COPY . .

CMD ["python", "app.py"]
```

**Why?** If you change app code, Docker only rebuilds from that layer onward, reusing cached layers above.

### .dockerignore

Exclude unnecessary files to speed up builds and reduce image size:

```
# .dockerignore
.git
.gitignore
README.md
.env
.env.*
__pycache__
*.pyc
*.pyo
*.pyd
.pytest_cache
.coverage
htmlcov/
dist/
build/
*.egg-info/
venv/
.venv/
node_modules/
.DS_Store
*.log
```

### Multi-Stage Builds

Separate build dependencies from runtime dependencies:

```dockerfile
# Stage 1: Build
FROM python:3.11 as builder

WORKDIR /build

# Install build dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY . .

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/app.py .

# Update PATH
ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

**Result**: Final image only contains runtime dependencies, much smaller.

### Common Dockerfile Instructions

| Instruction | Purpose | Example |
|------------|---------|---------|
| `FROM` | Base image | `FROM python:3.11-slim` |
| `RUN` | Execute command during build | `RUN pip install flask` |
| `COPY` | Copy files from host to image | `COPY app.py /app/` |
| `ADD` | Like COPY but with tar extraction and URL support | `ADD archive.tar.gz /app/` |
| `WORKDIR` | Set working directory | `WORKDIR /app` |
| `ENV` | Set environment variable | `ENV DEBUG=false` |
| `EXPOSE` | Document port (doesn't publish) | `EXPOSE 8000` |
| `CMD` | Default command (can be overridden) | `CMD ["python", "app.py"]` |
| `ENTRYPOINT` | Command always runs | `ENTRYPOINT ["python"]` |
| `USER` | Switch user | `USER appuser` |
| `ARG` | Build-time variables | `ARG VERSION=1.0` |
| `VOLUME` | Mount point for external storage | `VOLUME /data` |
| `HEALTHCHECK` | Container health check | `HEALTHCHECK CMD curl -f http://localhost/` |

### ENTRYPOINT vs CMD

**CMD**: Can be overridden
```dockerfile
CMD ["python", "app.py"]
```
```bash
docker run myapp python other.py  # Overrides CMD
```

**ENTRYPOINT**: Always runs, CMD becomes arguments
```dockerfile
ENTRYPOINT ["python"]
CMD ["app.py"]
```
```bash
docker run myapp other.py  # Runs: python other.py
```

**Combined pattern** (recommended for flexibility):
```dockerfile
ENTRYPOINT ["python"]
CMD ["app.py"]
```

---

## Docker Networking

Docker provides several network drivers for different use cases.

### Network Drivers

1. **bridge** (default): Private network on host, NAT to external
2. **host**: Container uses host's network directly (no isolation)
3. **none**: No networking
4. **overlay**: Multi-host networking (Swarm/Kubernetes)
5. **macvlan**: Assign MAC address to container

### Default Bridge Network

When you run a container without specifying network:

```bash
docker run -d --name web nginx
```

**Limitations**:
- Containers communicate via IP only, not names
- Must manually link containers
- All on same network (less isolation)

### User-Defined Bridge Networks (Recommended)

**Create network**:
```bash
docker network create my-network
```

**Run containers on network**:
```bash
docker run -d --name db --network my-network postgres
docker run -d --name web --network my-network nginx
```

**Benefits**:
- **DNS resolution**: `web` can reach `db` by name
- **Better isolation**: Only containers on same network can communicate
- **On-the-fly attachment**: Connect/disconnect without restart

### Network Commands

```bash
# List networks
docker network ls

# Inspect network
docker network inspect my-network

# Connect container to network
docker network connect my-network mycontainer

# Disconnect container from network
docker network disconnect my-network mycontainer

# Remove network
docker network rm my-network
```

### Inter-Container Communication Example

```bash
# Create network
docker network create app-network

# Run PostgreSQL
docker run -d \
  --name postgres \
  --network app-network \
  -e POSTGRES_PASSWORD=secret \
  postgres:15

# Run application (connects to postgres by name)
docker run -d \
  --name webapp \
  --network app-network \
  -p 8080:8080 \
  -e DATABASE_URL=postgresql://postgres:secret@postgres:5432/mydb \
  myapp:latest
```

The webapp can connect to `postgres:5432` directly!

### Port Publishing

**Publish specific port**:
```bash
docker run -p 8080:80 nginx  # host:8080 → container:80
```

**Publish all exposed ports to random host ports**:
```bash
docker run -P nginx
```

**Bind to specific interface**:
```bash
docker run -p 127.0.0.1:8080:80 nginx  # Only accessible locally
```

**View port mappings**:
```bash
docker port mycontainer
```

---

## Docker Volumes and Data Persistence

Containers are ephemeral - data inside is lost when container is removed. **Volumes** persist data.

### Three Types of Mounts

1. **Volumes** (recommended): Managed by Docker
2. **Bind Mounts**: Mount host directory/file
3. **tmpfs Mounts**: In-memory only (not persisted)

### Volumes (Recommended for Production)

**Create volume**:
```bash
docker volume create mydata
```

**Use volume**:
```bash
docker run -d \
  --name postgres \
  -v mydata:/var/lib/postgresql/data \
  postgres:15
```

**Benefits**:
- Docker manages storage location
- Easy to backup and migrate
- Work on Windows/Mac/Linux
- Can use volume drivers (cloud storage, etc.)
- Better performance than bind mounts on Mac/Windows

**Volume commands**:
```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect mydata

# Remove volume
docker volume rm mydata

# Remove all unused volumes
docker volume prune
```

### Bind Mounts (Good for Development)

Mount host directory into container:

```bash
docker run -d \
  --name dev-app \
  -v /host/path:/container/path \
  -v $(pwd):/app \
  myapp:latest
```

**Use cases**:
- Development: Live code updates without rebuild
- Configuration files
- Logs you want on host filesystem

**Example: Development workflow**:
```bash
docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  python:3.11 \
  python app.py
# Changes to app.py on host immediately reflected in container
```

### Volume Sharing Between Containers

```bash
# Create and use volume
docker run -d --name writer -v shared-data:/data alpine sh -c "echo hello > /data/file.txt"

# Another container uses same volume
docker run --rm -v shared-data:/data alpine cat /data/file.txt
# Output: hello
```

### Backup and Restore

**Backup volume**:
```bash
docker run --rm \
  -v mydata:/data \
  -v $(pwd):/backup \
  alpine \
  tar czf /backup/backup.tar.gz -C /data .
```

**Restore volume**:
```bash
docker run --rm \
  -v mydata:/data \
  -v $(pwd):/backup \
  alpine \
  tar xzf /backup/backup.tar.gz -C /data
```

### Read-Only Mounts

Prevent container from modifying data:

```bash
docker run -v mydata:/data:ro nginx
```

---

## Docker Compose

**Docker Compose** is a tool for defining and running multi-container applications using a YAML file.

### Why Docker Compose?

**Without Compose** (manual):
```bash
docker network create myapp-network
docker run -d --name db --network myapp-network -e POSTGRES_PASSWORD=secret postgres
docker run -d --name redis --network myapp-network redis
docker run -d --name web --network myapp-network -p 8080:8080 myapp
```

**With Compose** (declarative):
```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: secret

  redis:
    image: redis:7

  web:
    image: myapp:latest
    ports:
      - "8080:8080"
    depends_on:
      - db
      - redis
```

```bash
docker compose up -d  # Start all services
docker compose down   # Stop and remove all
```

### Complete docker-compose.yml Example

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: myapp_db
    restart: unless-stopped
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: ${DB_PASSWORD}  # From .env file
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U myuser"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: myapp_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    networks:
      - backend

  # Web Application
  web:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        BUILD_ENV: production
    container_name: myapp_web
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://myuser:${DB_PASSWORD}@postgres:5432/myapp
      REDIS_URL: redis://redis:6379
      DEBUG: "false"
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - backend
      - frontend
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: myapp_nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - static_files:/usr/share/nginx/html:ro
    depends_on:
      - web
    networks:
      - frontend

# Named volumes (managed by Docker)
volumes:
  postgres_data:
  redis_data:
  static_files:

# Networks
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
```

### Docker Compose Commands

```bash
# Start services
docker compose up              # Foreground
docker compose up -d           # Background (detached)
docker compose up --build      # Rebuild images before starting

# Stop services
docker compose stop            # Stop containers (don't remove)
docker compose down            # Stop and remove containers, networks
docker compose down -v         # Also remove volumes

# View status
docker compose ps              # Running services
docker compose logs            # View logs
docker compose logs -f web     # Follow logs for specific service

# Execute commands
docker compose exec web bash   # Open shell in running service
docker compose run web pytest  # Run one-off command

# Scaling
docker compose up -d --scale web=3  # Run 3 instances of web

# Restart services
docker compose restart
docker compose restart web     # Restart specific service
```

### Environment Variables

**Method 1: .env file** (recommended)
```bash
# .env
DB_PASSWORD=secret123
API_KEY=abc123
```

**Method 2: Environment file per service**
```yaml
services:
  web:
    env_file:
      - ./config/.env.web
```

**Method 3: Inline**
```yaml
services:
  web:
    environment:
      - DEBUG=true
      - LOG_LEVEL=info
```

### Depends On and Service Health

**Simple dependency** (starts in order, doesn't wait for ready):
```yaml
services:
  web:
    depends_on:
      - db
```

**Wait for healthy** (requires healthcheck):
```yaml
services:
  db:
    healthcheck:
      test: ["CMD", "pg_isready"]
      interval: 10s

  web:
    depends_on:
      db:
        condition: service_healthy
```

### Override Files

**Base config**: `docker-compose.yml`
**Development**: `docker-compose.override.yml` (auto-loaded)
**Production**: `docker-compose.prod.yml`

```bash
# Use specific override
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

---

## Multi-Stage Builds

Reduce final image size by separating build and runtime environments.

### Python Example

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11 as builder

WORKDIR /build

# Install build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy only the built wheels from builder stage
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application
COPY . .

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "app.py"]
```

**Result**: Final image doesn't include gcc, python-dev, or build tools.

### Node.js Example

```dockerfile
# Stage 1: Dependencies
FROM node:18 as dependencies
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Stage 2: Build
FROM node:18 as builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 3: Runtime
FROM node:18-alpine
WORKDIR /app

# Copy only production dependencies
COPY --from=dependencies /app/node_modules ./node_modules

# Copy built application
COPY --from=builder /app/dist ./dist
COPY package*.json ./

USER node
CMD ["node", "dist/index.js"]
```

### Go Example (Extreme size reduction)

```dockerfile
# Stage 1: Build
FROM golang:1.21 as builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# Stage 2: Runtime (scratch = empty image!)
FROM scratch
COPY --from=builder /app/main /main
EXPOSE 8080
CMD ["/main"]
```

**Result**: Final image is ~10MB (just the compiled binary)!

---

## Container Orchestration

For production, you typically need more than Docker Compose.

### Beyond Docker Compose

**Docker Compose is great for**:
- Development environments
- Single-host deployments
- Simple production setups

**But it can't**:
- Distribute containers across multiple hosts
- Auto-scale based on load
- Self-heal when containers crash
- Rolling updates without downtime
- Advanced service discovery

### Orchestration Options

#### 1. Docker Swarm

**Pros**:
- Built into Docker
- Easy to learn
- Good for simple multi-host setups

**Cons**:
- Less feature-rich than Kubernetes
- Smaller ecosystem

**Basic example**:
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml myapp

# Scale service
docker service scale myapp_web=5
```

#### 2. Kubernetes (k8s)

**Pros**:
- Industry standard
- Massive ecosystem
- Advanced features (auto-scaling, self-healing, rolling updates)
- Cloud-agnostic

**Cons**:
- Steep learning curve
- Complex for simple use cases
- Operational overhead

**When to use**: Production workloads at scale, multiple teams, complex microservices

#### 3. Managed Services

- **AWS ECS/Fargate**: AWS-native, serverless option
- **Google Cloud Run**: Serverless containers
- **Azure Container Instances**: Simple container deployments

---

## Security Considerations

### 1. Use Official and Verified Images

```dockerfile
# Good: Official image
FROM python:3.11-slim

# Bad: Random user's image
FROM randomuser/python-custom
```

**Check**:
- Official images (Docker Official Images badge)
- Verified publishers
- Scan images: `docker scan myimage:latest`

### 2. Don't Run as Root

```dockerfile
# Create and use non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
```

**Why?** If attacker escapes container, they have limited privileges.

### 3. Scan for Vulnerabilities

```bash
# Using Docker Scout (built-in)
docker scout cves myimage:latest

# Using Trivy
docker run aquasec/trivy image myimage:latest

# Using Snyk
snyk container test myimage:latest
```

### 4. Minimize Attack Surface

```dockerfile
# Use minimal base images
FROM python:3.11-slim  # Not python:3.11 (full)
FROM alpine:3.18       # Minimal (~5MB)
FROM scratch           # Empty (Go binaries)
```

**Remove unnecessary tools**:
```dockerfile
# Don't include shells, curl, wget in production if not needed
RUN apt-get remove --purge -y curl wget && \
    rm -rf /var/lib/apt/lists/*
```

### 5. Secret Management

**Bad** (secrets in image):
```dockerfile
ENV DATABASE_PASSWORD=secret123  # DON'T DO THIS
```

**Good** (secrets at runtime):
```bash
docker run -e DATABASE_PASSWORD=$(cat /path/to/secret) myapp

# Or use Docker secrets (Swarm)
docker secret create db_password /path/to/secret
```

**Use secret management tools**:
- AWS Secrets Manager
- HashiCorp Vault
- Kubernetes Secrets

### 6. Read-Only Filesystem

```bash
docker run --read-only \
  --tmpfs /tmp \
  --tmpfs /run \
  myapp:latest
```

Prevents attacker from writing malicious files.

### 7. Limit Resources

```bash
docker run \
  --memory="512m" \
  --memory-swap="512m" \
  --cpus="1.0" \
  --pids-limit=100 \
  myapp:latest
```

Prevents DoS attacks and resource exhaustion.

### 8. Drop Capabilities

```bash
docker run \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  myapp:latest
```

Minimizes container capabilities.

### 9. Use Multi-Stage Builds

Don't ship build tools and source code in production images.

### 10. Keep Images Updated

```bash
# Regularly rebuild with updated base images
docker pull python:3.11-slim
docker build --no-cache -t myapp:latest .
```

---

## Production Best Practices

### 1. Health Checks

**In Dockerfile**:
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

**In Docker Compose**:
```yaml
services:
  web:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s
```

### 2. Logging

**Best practice**: Log to stdout/stderr, let Docker handle collection.

```python
import logging
import sys

# Configure to log to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
```

**View logs**:
```bash
docker logs -f mycontainer
```

**Configure log driver**:
```bash
docker run \
  --log-driver=json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  myapp:latest
```

### 3. Resource Limits

Always set limits in production:

```yaml
services:
  web:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### 4. Restart Policies

```yaml
services:
  web:
    restart: unless-stopped  # Restart unless manually stopped
```

Options:
- `no`: Never restart
- `always`: Always restart
- `on-failure`: Restart on error
- `unless-stopped`: Restart unless explicitly stopped

### 5. Use Labels for Metadata

```dockerfile
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="team@example.com"
LABEL org.opencontainers.image.source="https://github.com/user/repo"
```

### 6. Graceful Shutdown

Handle SIGTERM properly:

```python
import signal
import sys

def signal_handler(sig, frame):
    print('Shutting down gracefully...')
    # Close connections, save state, etc.
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

### 7. Container Registry Strategy

```bash
# Use semantic versioning
docker tag myapp:latest myapp:1.2.3
docker tag myapp:latest myapp:1.2
docker tag myapp:latest myapp:1

# Never deploy :latest in production
# Always use specific version tags
```

### 8. CI/CD Pipeline

```yaml
# Example: GitHub Actions
name: Build and Push Docker Image

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build image
        run: docker build -t myapp:${{ github.ref_name }} .

      - name: Run tests in container
        run: docker run --rm myapp:${{ github.ref_name }} pytest

      - name: Scan image
        run: docker scan myapp:${{ github.ref_name }}

      - name: Push to registry
        run: |
          docker login -u ${{ secrets.DOCKER_USERNAME }} -p ${{ secrets.DOCKER_PASSWORD }}
          docker push myapp:${{ github.ref_name }}
```

---

## Common Patterns and Anti-Patterns

### ✅ Good Patterns

#### Pattern: One Process Per Container

```dockerfile
# Good: One service per container
# Container 1: Web server
FROM nginx
COPY nginx.conf /etc/nginx/nginx.conf

# Container 2: Application (separate container)
FROM python:3.11
COPY app.py .
CMD ["python", "app.py"]
```

**Why?** Easier to scale, update, and debug individual services.

#### Pattern: Immutable Infrastructure

```bash
# Don't modify running containers
# Instead, build new image and redeploy

# Bad
docker exec mycontainer apt-get install vim

# Good
# Add to Dockerfile, rebuild, redeploy
```

#### Pattern: 12-Factor App

- Store config in environment variables
- Treat logs as streams
- Disposable processes
- Dev/prod parity

### ❌ Anti-Patterns

#### Anti-Pattern: Installing SSH in Containers

```dockerfile
# Bad: Don't do this
RUN apt-get install -y openssh-server
```

**Why bad?** Containers should be ephemeral and accessed via `docker exec`, not SSH.

#### Anti-Pattern: Storing Data in Containers

```bash
# Bad: Data saved in container, lost when container removed
docker run -d postgres  # Data in container

# Good: Use volumes
docker run -d -v pgdata:/var/lib/postgresql/data postgres
```

#### Anti-Pattern: Using :latest in Production

```yaml
# Bad
services:
  web:
    image: myapp:latest  # Which version is this?

# Good
services:
  web:
    image: myapp:1.2.3  # Explicit version
```

#### Anti-Pattern: Building in Production

```bash
# Bad: Build on production server
docker build -t myapp:latest .

# Good: Build in CI/CD, push to registry, pull in production
docker pull myregistry/myapp:1.2.3
```

#### Anti-Pattern: Hardcoding Configuration

```dockerfile
# Bad
ENV DATABASE_URL=postgresql://user:pass@prod-db:5432/db

# Good: Inject at runtime
# docker run -e DATABASE_URL=$DATABASE_URL myapp
```

---

## Debugging and Troubleshooting

### Common Issues and Solutions

#### Issue: Container Exits Immediately

```bash
# Check exit code and logs
docker ps -a  # Find container ID
docker logs <container-id>

# Run interactively to debug
docker run -it myapp /bin/bash
```

**Common causes**:
- Application crashes on startup
- Wrong CMD/ENTRYPOINT
- Missing dependencies

#### Issue: Can't Connect to Container

```bash
# Check container is running
docker ps

# Check port mappings
docker port mycontainer

# Check network
docker network inspect bridge

# Test from inside container
docker exec mycontainer curl localhost:8080
```

#### Issue: Image Build Fails

```bash
# Build with verbose output
docker build --progress=plain --no-cache -t myapp .

# Check specific layer
docker build --target builder -t debug-layer .
docker run -it debug-layer /bin/bash
```

#### Issue: Out of Disk Space

```bash
# Check disk usage
docker system df

# Clean up
docker system prune -a  # Remove all unused images, containers, networks
docker volume prune     # Remove unused volumes

# Remove specific items
docker rm $(docker ps -a -q)  # Remove all stopped containers
docker rmi $(docker images -q)  # Remove all images
```

#### Issue: Performance Problems

```bash
# Check resource usage
docker stats

# Inspect container config
docker inspect mycontainer | grep -i memory
docker inspect mycontainer | grep -i cpu

# Increase limits
docker update --memory="1g" --cpus="2" mycontainer
```

### Debugging Tools

```bash
# Enter running container
docker exec -it mycontainer /bin/bash

# View processes
docker top mycontainer

# View events
docker events

# View filesystem changes
docker diff mycontainer

# Export container filesystem
docker export mycontainer > container.tar

# Copy files from container
docker cp mycontainer:/app/logs/error.log ./
```

### Debugging Networks

```bash
# List networks
docker network ls

# Inspect network (see connected containers)
docker network inspect my-network

# Test connectivity
docker run --rm --network my-network alpine ping other-container

# Inspect container's network settings
docker inspect -f '{{.NetworkSettings.Networks}}' mycontainer
```

### Debugging Volumes

```bash
# List volumes
docker volume ls

# Inspect volume (find mount point)
docker volume inspect myvolume

# Check volume contents
docker run --rm -v myvolume:/data alpine ls -la /data

# Backup volume for inspection
docker run --rm -v myvolume:/data -v $(pwd):/backup alpine tar czf /backup/volume.tar.gz /data
```

---

## Real-World Example: Complete ML Application Stack

Putting it all together: A production-ready ML application.

### Directory Structure

```
ml-app/
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── model/
├── nginx/
│   ├── Dockerfile
│   └── nginx.conf
└── monitoring/
    └── prometheus.yml
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mlapp
      POSTGRES_USER: mluser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mluser"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    networks:
      - backend

  ml-api:
    build:
      context: ./app
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://mluser:${DB_PASSWORD}@postgres:5432/mlapp
      REDIS_URL: redis://redis:6379
      MODEL_PATH: /models
    volumes:
      - ./models:/models:ro
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - backend
      - frontend
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  nginx:
    build:
      context: ./nginx
    ports:
      - "80:80"
    depends_on:
      - ml-api
    networks:
      - frontend

  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - backend

volumes:
  postgres_data:
  redis_data:
  prometheus_data:

networks:
  frontend:
  backend:
```

### app/Dockerfile

```dockerfile
# Multi-stage build for ML app
FROM python:3.11 as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app
USER mluser

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

This comprehensive guide covers the fundamentals through production deployment. Practice these concepts hands-on to truly understand Docker!

---

## Next Steps

1. **Hands-on Practice**: Build and deploy a simple application
2. **Read Official Docs**: [docs.docker.com](https://docs.docker.com)
3. **Explore Kubernetes**: When you outgrow single-host deployments
4. **Security Hardening**: Dive deeper into container security
5. **Monitoring**: Add logging and metrics (Prometheus, Grafana)

---

**Key Takeaway**: Containers solve the "works on my machine" problem by packaging applications with all dependencies. Docker makes this practical and production-ready. Start simple, but understand the internals—this knowledge will help you debug issues and optimize deployments.
