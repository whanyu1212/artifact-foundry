# Jenkins - Automation Server Deep Dive

**Last Updated**: 2025-12-30

A comprehensive guide to understanding Jenkins from fundamentals to advanced pipeline development, with special focus on ML/data science workflows.

---

## Table of Contents

1. [What is Jenkins?](#what-is-jenkins)
2. [Why Jenkins Still Matters](#why-jenkins-still-matters)
3. [Jenkins Architecture](#jenkins-architecture)
4. [Core Concepts](#core-concepts)
5. [Jenkins vs Other CI/CD Tools](#jenkins-vs-other-cicd-tools)
6. [Installation and Setup](#installation-and-setup)
7. [Pipeline as Code](#pipeline-as-code)
8. [Declarative Pipeline Syntax](#declarative-pipeline-syntax)
9. [Scripted Pipeline Syntax](#scripted-pipeline-syntax)
10. [Agents and Distributed Builds](#agents-and-distributed-builds)
11. [Plugin Ecosystem](#plugin-ecosystem)
12. [Credentials Management](#credentials-management)
13. [Docker Integration](#docker-integration)
14. [Kubernetes Integration](#kubernetes-integration)
15. [Jenkins for Machine Learning](#jenkins-for-machine-learning)
16. [Best Practices](#best-practices)
17. [Troubleshooting Common Issues](#troubleshooting-common-issues)
18. [Security Considerations](#security-considerations)
19. [Performance Optimization](#performance-optimization)
20. [Migration Strategies](#migration-strategies)

---

## What is Jenkins?

**Jenkins** is an open-source automation server used to automate software development tasks including building, testing, and deploying applications.

### Historical Context

```
2004: Kohsuke Kawaguchi starts Hudson at Sun Microsystems
2005: Hudson released as open source
2011: Oracle acquires Sun → Community forks to Jenkins
2011-Present: Jenkins becomes dominant CI/CD tool
2020s: Still widely used, especially in enterprises
```

**Why it succeeded:**
- Free and open source
- Extensible (plugin architecture)
- Self-hosted (full control)
- Language agnostic
- Active community

### What Problems Does Jenkins Solve?

**Before Jenkins** (Manual process):
```
Developer: "I finished feature X"
→ Manually build application
→ Manually run tests
→ Email team if tests fail
→ Manually deploy to server
→ Hope nothing breaks
→ Repeat for every change
```

**With Jenkins** (Automated):
```
Developer: Pushes code to Git
→ Jenkins detects change automatically
→ Jenkins builds application
→ Jenkins runs all tests
→ Jenkins notifies team of results
→ Jenkins deploys if tests pass
→ All automatically, every time
```

### Core Capabilities

1. **Continuous Integration**: Automatically build and test code changes
2. **Continuous Deployment**: Automatically deploy to environments
3. **Job Orchestration**: Coordinate complex multi-step workflows
4. **Distributed Builds**: Scale across multiple machines
5. **Extensive Integration**: Connect to anything via plugins

---

## Why Jenkins Still Matters

Despite newer alternatives (GitHub Actions, GitLab CI, CircleCI), Jenkins remains relevant:

### Strengths

**1. Maximum Flexibility**
- Not opinionated - can do anything
- Full control over execution environment
- Unlimited customization via plugins

**2. Self-Hosted Benefits**
- Complete data control (critical for sensitive data/models)
- No usage limits or costs
- Can use own hardware (including GPUs for ML)
- Air-gapped deployments possible

**3. Enterprise Features**
- Role-based access control (RBAC)
- Audit logging
- Integration with enterprise systems (LDAP, SSO)
- Mature, battle-tested in production

**4. Ecosystem**
- 1800+ plugins
- Integrates with virtually everything
- Large community and knowledge base

**5. ML/Data Science Friendly**
- Can route jobs to GPU nodes
- Handles long-running processes (training)
- Job queuing and resource management
- Integration with ML tools (MLflow, Kubeflow, etc.)

### When to Choose Jenkins

✅ **Good for:**
- Existing Jenkins infrastructure
- Complex, custom workflows
- Self-hosted requirements (compliance, privacy)
- GPU-intensive ML workloads
- Hybrid/multi-cloud environments
- Need to integrate with many systems

❌ **Consider alternatives when:**
- Starting from scratch (GitHub Actions simpler)
- Small team without DevOps expertise
- Prefer managed service
- Simple workflows only

---

## Jenkins Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Jenkins Controller                     │
│  (Master Node)                                          │
│                                                         │
│  • Manages configuration                                │
│  • Schedules builds                                     │
│  • Monitors agents                                      │
│  • Stores build results                                 │
│  • Serves web UI                                        │
└────────┬────────────────┬────────────────┬──────────────┘
         │                │                │
         ↓                ↓                ↓
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │ Agent 1 │     │ Agent 2 │     │ Agent 3 │
    │ Linux   │     │ Windows │     │ GPU     │
    │ x86_64  │     │ Server  │     │ Server  │
    └─────────┘     └─────────┘     └─────────┘
         ↓                ↓                ↓
    Execute Jobs    Execute Jobs    Execute Jobs
```

### Components Explained

#### Jenkins Controller (Master)

**Responsibilities:**
- **Job Scheduling**: Decides what runs when and where
- **Configuration Management**: Stores all Jenkins configuration
- **Build Monitoring**: Tracks running and completed builds
- **Plugin Management**: Manages installed plugins
- **Web Interface**: Provides UI for users
- **API**: REST API for programmatic access

**Important**: Controller should NOT execute builds directly (security and performance).

#### Jenkins Agents (formerly Slaves)

**Responsibilities:**
- Execute build jobs
- Run tests
- Deploy applications
- Report results back to controller

**Types of Agents:**

1. **Permanent Agents**: Always available, SSH connection
2. **Cloud Agents**: Dynamically provisioned (Docker, Kubernetes, AWS)
3. **Static Agents**: Manually configured machines

**Agent Selection:**
```groovy
// Run on any available agent
agent any

// Run on specific agent by label
agent { label 'linux && docker' }

// Run on specific node by name
agent { node { label 'gpu-server-1' } }

// Run in Docker container
agent { docker 'python:3.11' }
```

### Communication Flow

```
1. Developer pushes code to Git
         ↓
2. Webhook notifies Jenkins Controller
         ↓
3. Controller finds available agent
         ↓
4. Agent checks out code
         ↓
5. Agent executes pipeline steps
         ↓
6. Agent reports results to Controller
         ↓
7. Controller stores results and notifies user
```

---

## Core Concepts

### 1. Jobs (Items)

**Job** = Unit of work in Jenkins.

**Types:**
- **Freestyle Project**: Simple, UI-configured job
- **Pipeline**: Code-based job (Jenkinsfile)
- **Multi-branch Pipeline**: Automatically creates pipelines for branches
- **Folder**: Organize jobs hierarchically
- **Organization Folder**: Automatically discovers repositories

### 2. Builds

**Build** = Single execution of a job.

```
Job: "Build MyApp"
  ├─ Build #1 (Success) - 2025-01-01 10:00
  ├─ Build #2 (Failed) - 2025-01-01 11:30
  ├─ Build #3 (Success) - 2025-01-01 14:15
  └─ Build #4 (Running) - 2025-01-01 16:00
```

**Build Information:**
- Build number
- Git commit SHA
- Start time, duration
- Console output
- Test results
- Artifacts

### 3. Workspace

**Workspace** = Working directory for a build on an agent.

```
/var/jenkins/workspace/
├── my-pipeline/           # Workspace for 'my-pipeline' job
│   ├── src/
│   ├── tests/
│   └── Jenkinsfile
└── another-job/
    └── ...
```

**Important:**
- Each job has its own workspace
- Cleaned between builds (usually)
- Contains checked-out code

### 4. Executors

**Executor** = Slot for running a build on an agent.

```
Agent: build-server-1
  Executors: 4
    ├─ Executor #1: Running Build #45 of job-a
    ├─ Executor #2: Running Build #12 of job-b
    ├─ Executor #3: Idle
    └─ Executor #4: Idle
```

**Configuration:**
- Controller: 0 executors (don't run builds on controller)
- Agents: 1-8 executors typically (depends on resources)

### 5. Queue

**Queue** = Waiting list for builds when no executors available.

```
Build Queue:
  1. job-c Build #5 (waiting 2 min)
  2. job-d Build #18 (waiting 30 sec)
  3. job-e Build #1 (waiting 10 sec)
```

### 6. Artifacts

**Artifacts** = Files preserved after build completes.

**Examples:**
- JAR/WAR files
- Docker images
- Test reports
- Log files
- Built binaries

```groovy
// Archive artifacts
archiveArtifacts artifacts: 'dist/*.jar', fingerprint: true
```

### 7. Triggers

**Triggers** = Events that start builds.

**Types:**

1. **SCM Polling**: Check repository for changes
```groovy
triggers {
    pollSCM('H/5 * * * *')  // Check every 5 minutes
}
```

2. **Webhooks**: Repository notifies Jenkins directly (preferred)

3. **Scheduled**: Run at specific times
```groovy
triggers {
    cron('H 2 * * *')  // Nightly at ~2 AM
}
```

4. **Upstream**: Triggered by another job
```groovy
triggers {
    upstream(upstreamProjects: 'job-a', threshold: hudson.model.Result.SUCCESS)
}
```

5. **Manual**: User clicks "Build Now"

---

## Jenkins vs Other CI/CD Tools

### Comparison Matrix

| Feature | Jenkins | GitHub Actions | GitLab CI | CircleCI | Azure DevOps |
|---------|---------|---------------|-----------|----------|--------------|
| **Hosting** | Self-hosted | Cloud | Both | Cloud | Both |
| **Cost** | Free (infra cost) | Free tier | Free tier | Free tier | Free tier |
| **Setup Complexity** | High | Low | Low | Low | Medium |
| **Flexibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Plugins** | 1800+ | Marketplace | Built-in | Orbs | Extensions |
| **Pipeline as Code** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Multi-branch** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Docker Support** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **K8s Integration** | ✅ | Limited | ✅ | ✅ | ✅ |
| **GPU Support** | ✅✅ | Limited | ✅ | ✅ | ✅ |
| **Learning Curve** | High | Low | Low | Medium | Medium |
| **Community** | Huge | Growing | Large | Medium | Medium |

### When Each Tool Shines

**Jenkins:**
- Complex enterprise workflows
- Need self-hosted
- GPU-intensive ML workloads
- Maximum customization

**GitHub Actions:**
- GitHub-hosted projects
- Simple to medium complexity
- Quick to get started
- Don't want to manage infrastructure

**GitLab CI:**
- GitLab-hosted projects
- All-in-one platform (Git + CI/CD + Registry)
- Built-in security scanning

**CircleCI:**
- Fast builds with excellent caching
- Docker-first workflows
- Good documentation

**Azure DevOps:**
- Microsoft ecosystem
- Enterprise features out of the box
- Good Windows support

---

## Installation and Setup

### Installation Methods

#### 1. Docker (Recommended for Testing)

```bash
# Run Jenkins in Docker
docker run -d \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  --name jenkins \
  jenkins/jenkins:lts

# Get initial admin password
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

**Pros**: Quick setup, isolated
**Cons**: Not persistent by default, needs volumes

#### 2. Package Manager (Linux)

**Ubuntu/Debian:**
```bash
# Add Jenkins repository
wget -q -O - https://pkg.jenkins.io/debian/jenkins.io.key | sudo apt-key add -
sudo sh -c 'echo deb http://pkg.jenkins.io/debian-stable binary/ > /etc/apt/sources.list.d/jenkins.list'

# Install
sudo apt update
sudo apt install jenkins

# Start
sudo systemctl start jenkins
sudo systemctl enable jenkins
```

**RHEL/CentOS:**
```bash
sudo wget -O /etc/yum.repos.d/jenkins.repo https://pkg.jenkins.io/redhat-stable/jenkins.repo
sudo rpm --import https://pkg.jenkins.io/redhat-stable/jenkins.io.key
sudo yum install jenkins
sudo systemctl start jenkins
```

#### 3. Kubernetes (Production)

```yaml
# jenkins-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jenkins
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jenkins
  template:
    metadata:
      labels:
        app: jenkins
    spec:
      containers:
      - name: jenkins
        image: jenkins/jenkins:lts
        ports:
        - containerPort: 8080
        - containerPort: 50000
        volumeMounts:
        - name: jenkins-home
          mountPath: /var/jenkins_home
      volumes:
      - name: jenkins-home
        persistentVolumeClaim:
          claimName: jenkins-pvc
```

### Initial Setup

**1. Access Jenkins**: `http://localhost:8080`

**2. Unlock Jenkins**: Enter initial admin password

**3. Install Plugins**:
- Select "Install suggested plugins" (good starting point)
- Or select plugins individually

**Essential Plugins:**
- Git Plugin
- Pipeline Plugin
- Docker Plugin
- Kubernetes Plugin
- Credentials Plugin
- Blue Ocean (modern UI)

**4. Create Admin User**: Set username and password

**5. Configure Jenkins URL**: Set proper URL for webhooks

---

## Pipeline as Code

The modern way to use Jenkins: define pipelines in code (Jenkinsfile).

### Why Pipeline as Code?

**Benefits:**
1. **Version Control**: Pipeline versioned with application code
2. **Code Review**: Changes go through pull request process
3. **Reproducibility**: Same pipeline definition every time
4. **Collaboration**: Team can see and modify pipeline
5. **Testing**: Can test pipeline changes in branches

### Two Syntaxes

```
Pipeline as Code
    ├─ Declarative (recommended for most)
    │    • Structured, opinionated
    │    • Easier to learn
    │    • Built-in validation
    │
    └─ Scripted (for advanced needs)
         • Full Groovy programming
         • Maximum flexibility
         • More complex
```

### Jenkinsfile Location

**Option 1: In repository** (recommended)
```
myproject/
├── src/
├── tests/
├── Jenkinsfile         ← Pipeline definition
└── README.md
```

**Option 2: In Jenkins**
- Define pipeline in Jenkins UI
- Less preferred (not version controlled)

---

## Declarative Pipeline Syntax

**Declarative Pipeline** is the recommended approach for most use cases.

### Basic Structure

```groovy
pipeline {
    agent any                    // Where to run

    environment {                // Variables
        VAR = 'value'
    }

    stages {                     // Stages
        stage('Build') {
            steps {              // Steps to execute
                sh 'make build'
            }
        }
    }

    post {                       // Post-build actions
        always {
            echo 'Done'
        }
    }
}
```

### Complete Example

```groovy
pipeline {
    // =========================================================================
    // AGENT: Where to execute
    // =========================================================================
    agent {
        docker {
            image 'python:3.11-slim'
            args '-v $HOME/.cache:/cache'
        }
    }

    // =========================================================================
    // PARAMETERS: Input from user when triggering build
    // =========================================================================
    parameters {
        string(name: 'BRANCH', defaultValue: 'main', description: 'Branch to build')
        choice(name: 'ENVIRONMENT', choices: ['dev', 'staging', 'prod'], description: 'Deploy environment')
        booleanParam(name: 'RUN_TESTS', defaultValue: true, description: 'Run tests?')
    }

    // =========================================================================
    // ENVIRONMENT: Variables available to all stages
    // =========================================================================
    environment {
        APP_NAME = 'myapp'
        BUILD_VERSION = "${env.GIT_BRANCH}-${env.BUILD_NUMBER}"
        // Credentials binding
        DOCKER_CREDS = credentials('docker-hub-credentials')
    }

    // =========================================================================
    // OPTIONS: Pipeline behavior settings
    // =========================================================================
    options {
        timeout(time: 1, unit: 'HOURS')          // Max pipeline duration
        buildDiscarder(logRotator(numToKeepStr: '10'))  // Keep last 10 builds
        timestamps()                              // Add timestamps to output
        disableConcurrentBuilds()                // Don't run concurrent builds
    }

    // =========================================================================
    // TRIGGERS: When to run
    // =========================================================================
    triggers {
        pollSCM('H/5 * * * *')   // Poll every 5 minutes
        cron('H 2 * * *')        // Nightly build at ~2 AM
    }

    // =========================================================================
    // STAGES: Main pipeline stages
    // =========================================================================
    stages {

        stage('Checkout') {
            steps {
                echo "Checking out ${params.BRANCH}"
                checkout scm
                sh 'git log -1'
            }
        }

        stage('Build') {
            steps {
                echo 'Building application...'
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Test') {
            when {
                expression { params.RUN_TESTS == true }
            }

            parallel {
                stage('Unit Tests') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            pytest tests/unit --junitxml=results.xml
                        '''
                    }
                }

                stage('Lint') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            flake8 src/
                        '''
                    }
                }
            }
        }

        stage('Docker Build') {
            steps {
                script {
                    dockerImage = docker.build("${APP_NAME}:${BUILD_VERSION}")
                }
            }
        }

        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                echo "Deploying to ${params.ENVIRONMENT}"
                sh "./deploy.sh ${params.ENVIRONMENT}"
            }
        }
    }

    // =========================================================================
    // POST: Actions after stages complete
    // =========================================================================
    post {
        always {
            echo 'Cleaning up...'
            cleanWs()
        }

        success {
            echo 'Pipeline succeeded!'
            junit 'results.xml'
        }

        failure {
            echo 'Pipeline failed!'
            emailext(
                subject: "Build Failed: ${env.JOB_NAME}",
                body: "Build ${env.BUILD_NUMBER} failed",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

### Key Sections Explained

#### Agent

**Where to run the pipeline:**

```groovy
// Any available agent
agent any

// Specific label
agent { label 'linux' }

// Docker container
agent {
    docker {
        image 'python:3.11'
        args '-u root'  // Run as root
    }
}

// Kubernetes pod
agent {
    kubernetes {
        yaml '''
            spec:
              containers:
              - name: python
                image: python:3.11
        '''
    }
}

// None (define per stage)
agent none
```

#### Stages and Steps

**Stages** = Major phases
**Steps** = Individual actions

```groovy
stages {
    stage('Example') {
        steps {
            // Shell command
            sh 'echo "Hello"'

            // Groovy script block
            script {
                def x = 5 + 5
                echo "Result: ${x}"
            }

            // Built-in step
            archiveArtifacts 'dist/*.jar'
        }
    }
}
```

#### When Directive

**Conditional execution:**

```groovy
stage('Deploy to Prod') {
    when {
        // Only on main branch
        branch 'main'

        // And only if environment variable set
        environment name: 'DEPLOY', value: 'true'

        // And custom condition
        expression {
            return currentBuild.result == 'SUCCESS'
        }
    }
    steps {
        sh './deploy.sh production'
    }
}
```

#### Parallel Stages

**Run stages concurrently:**

```groovy
stage('Test') {
    parallel {
        stage('Unit Tests') {
            steps { sh 'pytest tests/unit' }
        }
        stage('Integration Tests') {
            steps { sh 'pytest tests/integration' }
        }
        stage('Lint') {
            steps { sh 'flake8 src/' }
        }
    }
}
```

---

## Scripted Pipeline Syntax

**Scripted Pipeline** = Full Groovy programming language.

### Basic Structure

```groovy
node {  // Allocate agent
    stage('Build') {
        // Groovy code here
        sh 'make build'
    }
}
```

### Complete Example

```groovy
// Variables
def appName = 'myapp'
def dockerImage

// Allocate node
node('docker') {
    try {
        // Checkout stage
        stage('Checkout') {
            checkout scm
            env.GIT_COMMIT_SHORT = sh(
                script: 'git rev-parse --short HEAD',
                returnStdout: true
            ).trim()
        }

        // Build stage
        stage('Build') {
            sh '''
                python -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
            '''
        }

        // Test stage with conditional
        stage('Test') {
            def runTests = params.RUN_TESTS ?: true
            if (runTests) {
                parallel(
                    'Unit Tests': {
                        sh 'pytest tests/unit'
                    },
                    'Lint': {
                        sh 'flake8 src/'
                    }
                )
            } else {
                echo 'Skipping tests'
            }
        }

        // Docker build
        stage('Docker Build') {
            dockerImage = docker.build("${appName}:${env.GIT_COMMIT_SHORT}")
        }

        // Deploy
        stage('Deploy') {
            if (env.BRANCH_NAME == 'main') {
                docker.withRegistry('https://registry.example.com', 'docker-credentials') {
                    dockerImage.push()
                }

                sh './deploy.sh production'
            } else {
                echo 'Not main branch, skipping deployment'
            }
        }

        // Mark as success
        currentBuild.result = 'SUCCESS'

    } catch (Exception e) {
        // Mark as failure
        currentBuild.result = 'FAILURE'
        throw e

    } finally {
        // Always cleanup
        stage('Cleanup') {
            cleanWs()
        }

        // Send notifications
        if (currentBuild.result == 'FAILURE') {
            mail(
                to: 'team@example.com',
                subject: "Build Failed: ${env.JOB_NAME}",
                body: "Build ${env.BUILD_NUMBER} failed"
            )
        }
    }
}
```

### Declarative vs Scripted

| Aspect | Declarative | Scripted |
|--------|------------|----------|
| **Syntax** | Structured, opinionated | Free-form Groovy |
| **Learning Curve** | Easier | Harder |
| **Validation** | Built-in | Manual |
| **Flexibility** | Limited | Unlimited |
| **Best For** | Most use cases | Complex logic |
| **Recommended** | ✅ Yes | Only when needed |

**Rule of thumb**: Start with declarative, use scripted only when you hit limitations.

---

## Agents and Distributed Builds

### Why Distributed Builds?

**Problems with single server:**
- Limited resources (CPU, memory, disk)
- No platform diversity (can't build Linux and Windows)
- Single point of failure
- Can't scale horizontally

**Solution**: Multiple agents

```
           Jenkins Controller
                  │
    ┌─────────────┼─────────────┐
    │             │             │
 Agent 1       Agent 2       Agent 3
 Linux x64     Windows       Linux ARM
 8 CPUs        16 CPUs       GPU
```

### Agent Types

#### 1. Permanent Agent

**Always available, connected via SSH/JNLP**

**Setup:**
1. Install Java on agent machine
2. Add agent in Jenkins UI (Manage Jenkins → Nodes)
3. Configure connection method (SSH recommended)
4. Set labels and executors

**Configuration:**
```
Name: build-server-1
Labels: linux docker gpu
Executors: 4
Remote root directory: /var/jenkins
Launch method: SSH
```

#### 2. Dynamic Cloud Agents

**Provisioned on-demand, destroyed after use**

**Docker Agents:**
```groovy
agent {
    docker {
        image 'python:3.11'
        // Container exists only during build
    }
}
```

**Kubernetes Agents:**
```groovy
agent {
    kubernetes {
        yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: python
    image: python:3.11
    command: ['cat']
    tty: true
  - name: docker
    image: docker:dind
    securityContext:
      privileged: true
        '''
    }
}
```

**AWS EC2 Agents:**
- Jenkins EC2 Plugin
- Spin up instances on demand
- Terminate when idle

### Labels

**Labels** = Tags for selecting agents

**Example:**
```
Agent 1: labels = [linux, docker, x86_64]
Agent 2: labels = [linux, docker, gpu]
Agent 3: labels = [windows, visual-studio]
```

**Usage in pipeline:**
```groovy
stage('Train Model') {
    agent { label 'gpu' }  // Only runs on Agent 2
    steps {
        sh 'python train.py --gpu'
    }
}
```

### Multi-Agent Pipeline

```groovy
pipeline {
    agent none  // Don't allocate agent at pipeline level

    stages {
        stage('Build Linux') {
            agent { label 'linux' }
            steps {
                sh 'make build-linux'
            }
        }

        stage('Build Windows') {
            agent { label 'windows' }
            steps {
                bat 'build.bat'
            }
        }

        stage('Train Model') {
            agent { label 'gpu' }
            steps {
                sh 'python train.py --gpu'
            }
        }
    }
}
```

---

## Plugin Ecosystem

Jenkins' power comes from its **1800+ plugins**.

### Essential Plugins

#### Core Functionality

1. **Pipeline Plugin** (pre-installed)
   - Pipeline as code support
   - Jenkinsfile support

2. **Git Plugin** (pre-installed)
   - Git repository integration
   - Checkout code from Git

3. **Credentials Plugin** (pre-installed)
   - Secure credential storage
   - Use secrets in pipelines

#### Docker & Kubernetes

4. **Docker Pipeline Plugin**
   ```groovy
   docker.build('myimage:latest')
   docker.image('python:3.11').inside {
       sh 'python script.py'
   }
   ```

5. **Kubernetes Plugin**
   - Dynamic Kubernetes pod agents
   - Run builds in K8s cluster

#### Build Tools

6. **Blue Ocean**
   - Modern, visual pipeline editor
   - Better UI than classic Jenkins

7. **Warnings Next Generation**
   - Parse compiler warnings
   - Visualize code quality trends

#### Testing & Quality

8. **JUnit Plugin**
   ```groovy
   junit 'test-results.xml'
   ```

9. **Cobertura/JaCoCo**
   - Code coverage reporting

10. **SonarQube Scanner**
    - Code quality analysis
    - Integration with SonarQube

#### Notifications

11. **Email Extension**
    ```groovy
    emailext(
        subject: "Build ${currentBuild.result}",
        body: "Check ${env.BUILD_URL}",
        to: "${env.CHANGE_AUTHOR_EMAIL}"
    )
    ```

12. **Slack Notification**
    ```groovy
    slackSend(
        color: 'good',
        message: "Build succeeded: ${env.JOB_NAME}"
    )
    ```

#### Deployment

13. **SSH Plugin**
    - Deploy via SSH
    - Copy files to remote servers

14. **Kubernetes CLI**
    - kubectl commands in pipeline

15. **AWS Steps**
    - Interact with AWS services

### Plugin Management

**Install plugins:**
1. Manage Jenkins → Plugins → Available
2. Search for plugin
3. Install without restart (if possible)

**Update plugins:**
1. Manage Jenkins → Plugins → Updates
2. Select plugins
3. Update

**Plugin dependencies:**
- Plugins can depend on other plugins
- Jenkins automatically installs dependencies

---

## Credentials Management

**Never hardcode secrets!** Use Jenkins credentials.

### Credential Types

1. **Username with password**
2. **Secret text**: Single secret value (API key)
3. **Secret file**: File containing secret (kubeconfig)
4. **SSH Username with private key**
5. **Certificate**

### Adding Credentials

**Via UI:**
1. Manage Jenkins → Credentials
2. Select domain (usually "Global")
3. Add Credentials
4. Enter details and give it an ID

**Via pipeline:**
```groovy
// Not recommended, but possible
// Better to add via UI
```

### Using Credentials in Pipeline

#### Method 1: Environment Variable

```groovy
environment {
    DOCKER_CREDS = credentials('docker-hub-credentials')
    // Creates:
    // DOCKER_CREDS_USR = username
    // DOCKER_CREDS_PSW = password
}

steps {
    sh 'docker login -u $DOCKER_CREDS_USR -p $DOCKER_CREDS_PSW'
}
```

#### Method 2: withCredentials Block

```groovy
steps {
    withCredentials([
        usernamePassword(
            credentialsId: 'docker-hub',
            usernameVariable: 'USER',
            passwordVariable: 'PASS'
        )
    ]) {
        sh 'docker login -u $USER -p $PASS'
    }
}
```

#### Method 3: String Credentials

```groovy
withCredentials([string(credentialsId: 'api-key', variable: 'API_KEY')]) {
    sh 'curl -H "Authorization: Bearer $API_KEY" https://api.example.com'
}
```

#### Method 4: File Credentials

```groovy
withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
    sh 'kubectl get pods'
}
```

### Security Best Practices

1. **Use Credentials Plugin** - Don't hardcode
2. **Limit Credential Scope** - Folder-level credentials when possible
3. **Use Different Credentials** - Don't reuse prod creds in dev
4. **Rotate Regularly** - Update credentials periodically
5. **Audit Access** - Monitor who uses which credentials

---

## Docker Integration

Jenkins + Docker = Powerful combination

### Use Cases

1. **Build in Docker**: Consistent build environment
2. **Run Tests in Docker**: Isolated test execution
3. **Build Docker Images**: Create deployable artifacts
4. **Dynamic Agents**: Disposable build environments

### Docker Pipeline Plugin

#### Build and Run Inside Container

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
            args '-v $HOME/.cache:/cache'
        }
    }

    stages {
        stage('Test') {
            steps {
                sh '''
                    python --version
                    pip install pytest
                    pytest
                '''
            }
        }
    }
}
```

**What happens:**
1. Jenkins pulls `python:3.11-slim` image
2. Starts container with workspace mounted
3. Executes steps inside container
4. Stops and removes container

#### Build Docker Image

```groovy
pipeline {
    agent any

    stages {
        stage('Build Image') {
            steps {
                script {
                    // Build image
                    dockerImage = docker.build("myapp:${env.BUILD_NUMBER}")

                    // Tag image
                    dockerImage.tag('latest')

                    // Push to registry
                    docker.withRegistry('https://registry.example.com', 'docker-creds') {
                        dockerImage.push("${env.BUILD_NUMBER}")
                        dockerImage.push('latest')
                    }
                }
            }
        }
    }
}
```

#### Run Commands in Container

```groovy
stage('Test') {
    steps {
        script {
            docker.image('python:3.11').inside {
                sh 'python --version'
                sh 'pytest'
            }
        }
    }
}
```

#### Multi-Container Setup

```groovy
stage('Integration Test') {
    steps {
        script {
            // Start database container
            docker.image('postgres:15').withRun('-e POSTGRES_PASSWORD=secret') { db ->
                // Get database IP
                def dbUrl = "postgresql://postgres:secret@${db.id}:5432/test"

                // Run tests against database
                docker.image('python:3.11').inside("--link ${db.id}:postgres") {
                    sh "DATABASE_URL=${dbUrl} pytest tests/integration"
                }
            }
        }
    }
}
```

### Docker-in-Docker (DinD)

**For building Docker images inside Docker agent:**

```groovy
agent {
    docker {
        image 'docker:dind'
        args '--privileged -v /var/run/docker.sock:/var/run/docker.sock'
    }
}
```

**Security warning**: Mounting Docker socket gives full access to host!

---

## Kubernetes Integration

Run Jenkins agents as Kubernetes pods.

### Kubernetes Plugin

**Benefits:**
- Dynamic agent provisioning
- Automatic scaling
- Resource isolation
- Cost-effective (pods created on-demand)

### Configuration

**1. Install Kubernetes Plugin**

**2. Configure Kubernetes Cloud:**
- Manage Jenkins → Clouds → Add Kubernetes
- Kubernetes URL: Your cluster URL
- Credentials: Kubeconfig or token
- Jenkins URL: How pods reach Jenkins

**3. Define Pod Template:**

```groovy
agent {
    kubernetes {
        // YAML pod definition
        yaml '''
apiVersion: v1
kind: Pod
metadata:
  labels:
    jenkins: agent
spec:
  containers:
  - name: python
    image: python:3.11
    command: ['cat']
    tty: true
  - name: docker
    image: docker:dind
    securityContext:
      privileged: true
  - name: kubectl
    image: bitnami/kubectl:latest
    command: ['cat']
    tty: true
        '''
    }
}
```

### Using Kubernetes Agents

```groovy
pipeline {
    agent {
        kubernetes {
            yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: python
    image: python:3.11
    command: ['sleep']
    args: ['infinity']
  - name: docker
    image: docker:dind
    securityContext:
      privileged: true
            '''
        }
    }

    stages {
        stage('Test') {
            steps {
                container('python') {
                    sh 'python --version'
                    sh 'pytest'
                }
            }
        }

        stage('Build Image') {
            steps {
                container('docker') {
                    sh 'docker build -t myapp:latest .'
                }
            }
        }
    }
}
```

### ML Workload Example

```groovy
// GPU-enabled pod for model training
agent {
    kubernetes {
        yaml '''
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: pytorch
    image: pytorch/pytorch:latest
    command: ['sleep']
    args: ['infinity']
    resources:
      limits:
        nvidia.com/gpu: 1  # Request GPU
      requests:
        memory: "16Gi"
        cpu: "4"
        '''
    }
}

stages {
    stage('Train Model') {
        steps {
            container('pytorch') {
                sh '''
                    python train.py \
                        --data /data/train.csv \
                        --epochs 100 \
                        --gpu
                '''
            }
        }
    }
}
```

---

## Jenkins for Machine Learning

Jenkins is particularly useful for ML workflows due to its flexibility and ability to handle long-running jobs.

### ML Pipeline Architecture

```
Data Updates → Jenkins Pipeline
                    ↓
         ┌──────────┴──────────┐
         │                     │
    Data Validation      Feature Engineering
         │                     │
         └──────────┬──────────┘
                    ↓
              Model Training (GPU)
                    ↓
         ┌──────────┴──────────┐
         │                     │
    Model Evaluation    Fairness Check
         │                     │
         └──────────┬──────────┘
                    ↓
         Model Registry (MLflow)
                    ↓
         ┌──────────┴──────────┐
         │                     │
    Deploy Staging      A/B Test Setup
         │                     │
         └──────────┬──────────┘
                    ↓
            Deploy Production
```

### Complete ML Pipeline Example

```groovy
pipeline {
    agent none

    parameters {
        string(name: 'DATA_VERSION', defaultValue: 'latest', description: 'Data version to use')
        choice(name: 'MODEL_TYPE', choices: ['xgboost', 'random_forest', 'neural_net'], description: 'Model algorithm')
        booleanParam(name: 'FORCE_RETRAIN', defaultValue: false, description: 'Force retraining even if data unchanged')
    }

    environment {
        MLFLOW_TRACKING_URI = 'https://mlflow.example.com'
        MODEL_REGISTRY = 's3://models/'
        DVC_REMOTE = 's3://dvc-storage/'
    }

    stages {
        // Stage 1: Data Validation
        stage('Data Validation') {
            agent { label 'cpu' }
            steps {
                script {
                    // Pull data with DVC
                    sh "dvc pull data/train.csv.dvc"

                    // Validate data quality
                    sh '''
                        python scripts/validate_data.py \
                            --data data/train.csv \
                            --output reports/data_validation.json
                    '''

                    // Check if retraining needed
                    def dataChanged = sh(
                        script: 'python scripts/check_data_drift.py',
                        returnStdout: true
                    ).trim()

                    if (dataChanged == 'false' && !params.FORCE_RETRAIN) {
                        echo "Data unchanged, skipping retraining"
                        currentBuild.result = 'SUCCESS'
                        return
                    }
                }
            }
        }

        // Stage 2: Feature Engineering
        stage('Feature Engineering') {
            agent {
                label 'high-memory'  // Node with lots of RAM
            }
            steps {
                sh '''
                    python scripts/feature_engineering.py \
                        --input data/train.csv \
                        --output data/features.parquet
                '''

                // Save features with DVC
                sh 'dvc add data/features.parquet'
                sh 'dvc push'
            }
        }

        // Stage 3: Model Training (GPU)
        stage('Train Model') {
            agent {
                label 'gpu'  // GPU-enabled node
            }
            steps {
                script {
                    // Train with MLflow logging
                    sh """
                        python scripts/train.py \
                            --data data/features.parquet \
                            --model-type ${params.MODEL_TYPE} \
                            --mlflow-run-name jenkins-${env.BUILD_NUMBER} \
                            --mlflow-experiment production \
                            --gpu \
                            --output models/model.pkl
                    """
                }

                // Archive model
                archiveArtifacts artifacts: 'models/model.pkl', fingerprint: true
            }
        }

        // Stage 4: Model Evaluation
        stage('Evaluate Model') {
            agent { label 'cpu' }
            steps {
                script {
                    // Pull test data
                    sh 'dvc pull data/test.csv.dvc'

                    // Evaluate model
                    sh '''
                        python scripts/evaluate.py \
                            --model models/model.pkl \
                            --test-data data/test.csv \
                            --output reports/evaluation.json
                    '''

                    // Extract metrics
                    def metrics = readJSON file: 'reports/evaluation.json'
                    echo "Accuracy: ${metrics.accuracy}"
                    echo "F1 Score: ${metrics.f1_score}"

                    // Quality gate
                    if (metrics.accuracy < 0.85) {
                        error("Model accuracy ${metrics.accuracy} below threshold 0.85")
                    }

                    // Check inference speed
                    if (metrics.avg_inference_time_ms > 100) {
                        error("Inference time ${metrics.avg_inference_time_ms}ms exceeds 100ms")
                    }
                }
            }
        }

        // Stage 5: Fairness Check
        stage('Fairness Analysis') {
            agent { label 'cpu' }
            steps {
                sh '''
                    python scripts/fairness_check.py \
                        --model models/model.pkl \
                        --test-data data/test.csv \
                        --protected-attributes age,gender \
                        --output reports/fairness.json
                '''

                script {
                    def fairness = readJSON file: 'reports/fairness.json'
                    if (fairness.bias_detected) {
                        error("Bias detected in model predictions")
                    }
                }
            }
        }

        // Stage 6: Register Model
        stage('Register Model') {
            agent { label 'cpu' }
            steps {
                withCredentials([
                    usernamePassword(
                        credentialsId: 'mlflow-credentials',
                        usernameVariable: 'MLFLOW_USER',
                        passwordVariable: 'MLFLOW_PASS'
                    )
                ]) {
                    sh '''
                        python scripts/register_model.py \
                            --model-path models/model.pkl \
                            --model-name my-ml-model \
                            --stage Staging \
                            --git-commit ${GIT_COMMIT} \
                            --jenkins-build ${BUILD_NUMBER}
                    '''
                }

                // Upload to S3
                sh '''
                    aws s3 cp models/model.pkl \
                        s3://models/production/${GIT_COMMIT}/model.pkl
                '''
            }
        }

        // Stage 7: Deploy to Staging
        stage('Deploy Staging') {
            agent { label 'kubernetes' }
            steps {
                withCredentials([file(credentialsId: 'kubeconfig-staging', variable: 'KUBECONFIG')]) {
                    sh """
                        kubectl set image deployment/ml-api \
                            ml-api=ml-api:${GIT_COMMIT} \
                            -n staging

                        kubectl rollout status deployment/ml-api -n staging
                    """
                }

                // Smoke tests
                sh '''
                    sleep 30
                    curl -f https://ml-api-staging.example.com/health
                    python scripts/smoke_test.py --url https://ml-api-staging.example.com
                '''
            }
        }

        // Stage 8: Deploy Production (Manual Approval)
        stage('Deploy Production') {
            when {
                branch 'main'
            }

            agent { label 'kubernetes' }

            steps {
                // Manual approval required
                timeout(time: 24, unit: 'HOURS') {
                    input(
                        message: 'Deploy to production?',
                        ok: 'Deploy',
                        submitter: 'admin,ml-team'
                    )
                }

                withCredentials([file(credentialsId: 'kubeconfig-production', variable: 'KUBECONFIG')]) {
                    // Canary deployment: 10% traffic first
                    sh """
                        kubectl set image deployment/ml-api-canary \
                            ml-api=ml-api:${GIT_COMMIT} \
                            -n production

                        kubectl rollout status deployment/ml-api-canary -n production
                    """

                    // Monitor canary
                    echo "Monitoring canary for 10 minutes..."
                    sleep time: 10, unit: 'MINUTES'

                    // Check metrics
                    def canaryOk = sh(
                        script: 'python scripts/check_canary_metrics.py',
                        returnStatus: true
                    )

                    if (canaryOk != 0) {
                        error("Canary metrics failed, rolling back")
                    }

                    // Full rollout
                    sh """
                        kubectl set image deployment/ml-api \
                            ml-api=ml-api:${GIT_COMMIT} \
                            -n production

                        kubectl rollout status deployment/ml-api -n production
                    """
                }

                // Update MLflow model stage
                sh '''
                    python scripts/promote_model.py \
                        --model-name my-ml-model \
                        --stage Production
                '''
            }
        }
    }

    post {
        always {
            // Archive all reports
            archiveArtifacts artifacts: 'reports/**/*', allowEmptyArchive: true
        }

        success {
            slackSend(
                color: 'good',
                message: "ML Pipeline Succeeded: ${env.JOB_NAME} #${env.BUILD_NUMBER}\nModel: ${params.MODEL_TYPE}\nAccuracy: See reports"
            )
        }

        failure {
            slackSend(
                color: 'danger',
                message: "ML Pipeline Failed: ${env.JOB_NAME} #${env.BUILD_NUMBER}\nCheck: ${env.BUILD_URL}"
            )
        }
    }
}
```

### Scheduling Retraining

```groovy
// Retrain model weekly
triggers {
    cron('H 2 * * 0')  // Every Sunday at ~2 AM
}

pipeline {
    agent none

    stages {
        stage('Check Data Drift') {
            steps {
                script {
                    def drift = sh(
                        script: 'python check_drift.py',
                        returnStdout: true
                    ).trim().toFloat()

                    echo "Data drift score: ${drift}"

                    // Only retrain if significant drift
                    if (drift > 0.3) {
                        echo "Significant drift detected, retraining..."
                        build job: 'ml-training-pipeline', wait: true
                    } else {
                        echo "No significant drift, skipping retrain"
                    }
                }
            }
        }
    }
}
```

---

## Best Practices

### 1. Pipeline Design

✅ **Keep Pipelines Fast**
- Target: < 10 minutes
- Use parallel stages
- Cache dependencies
- Split slow tests to separate pipeline

```groovy
// Fast pipeline (on every commit)
stage('Quick Checks') {
    parallel {
        stage('Unit Tests') { ... }
        stage('Lint') { ... }
        stage('Security Scan') { ... }
    }
}

// Slow pipeline (nightly)
stage('Comprehensive Tests') {
    triggers { cron('H 2 * * *') }
    steps {
        sh 'pytest tests/e2e'  // Slow tests
    }
}
```

✅ **Fail Fast**
- Check syntax before running tests
- Run quick checks before slow ones

```groovy
stage('Validate') {
    steps {
        sh 'python -m py_compile src/*.py'  // Syntax check (fast)
    }
}

stage('Test') {  // Only runs if validate passes
    steps {
        sh 'pytest'  // Tests (slow)
    }
}
```

✅ **Make Pipelines Idempotent**
- Running twice should produce same result
- Clean workspace before starting

```groovy
options {
    skipDefaultCheckout()  // Don't auto-checkout
}

stages {
    stage('Setup') {
        steps {
            cleanWs()  // Clean workspace
            checkout scm  // Then checkout
        }
    }
}
```

### 2. Resource Management

✅ **Don't Run Builds on Controller**
```groovy
// Controller configuration
Number of executors: 0  // Never run builds on controller
```

✅ **Use Appropriate Agents**
```groovy
// Use specific agents for specific tasks
stage('Build Frontend') {
    agent { label 'node && docker' }
}

stage('Train Model') {
    agent { label 'gpu && high-memory' }
}
```

✅ **Clean Up**
```groovy
post {
    always {
        cleanWs()  // Clean workspace after build
    }
}
```

### 3. Security

✅ **Use Credentials Plugin**
```groovy
// Never hardcode secrets
environment {
    API_KEY = credentials('api-key-id')
}
```

✅ **Limit Agent Permissions**
- Agents shouldn't have admin access
- Use separate credentials per environment

✅ **Enable CSRF Protection**
- Enabled by default in modern Jenkins

✅ **Use Role-Based Access Control**
- Different permissions for different teams
- Use folders to organize jobs

### 4. Notifications

✅ **Notify on State Changes**
```groovy
post {
    regression {  // First failure after success
        mail to: 'team@example.com',
             subject: "Build Broke: ${env.JOB_NAME}"
    }

    fixed {  // First success after failure
        mail to: 'team@example.com',
             subject: "Build Fixed: ${env.JOB_NAME}"
    }
}
```

✅ **Don't Spam**
- Don't notify on every build
- Notify on failures and state changes only

### 5. Maintainability

✅ **Use Shared Libraries**

Create reusable pipeline code:

```groovy
// vars/buildDockerImage.groovy (in shared library)
def call(String imageName) {
    script {
        docker.build("${imageName}:${env.BUILD_NUMBER}")
    }
}
```

```groovy
// Jenkinsfile (using shared library)
@Library('my-shared-library') _

pipeline {
    stages {
        stage('Build') {
            steps {
                buildDockerImage('myapp')  // Reusable function
            }
        }
    }
}
```

✅ **Document Your Pipeline**
```groovy
/*
 * MyApp Build Pipeline
 *
 * This pipeline builds, tests, and deploys MyApp.
 *
 * Agents required:
 * - 'docker': For building images
 * - 'gpu': For model training
 *
 * Credentials required:
 * - 'docker-hub': Docker Hub credentials
 * - 'kubeconfig-prod': Production Kubernetes config
 */

pipeline {
    // ...
}
```

---

## Troubleshooting Common Issues

### Issue 1: Agent Offline

**Symptom**: Builds stuck in queue, agent shows offline

**Causes:**
- Network connectivity
- Agent machine down
- Incorrect credentials
- Port blocked (50000 for JNLP)

**Solutions:**
```bash
# Check agent logs on Jenkins
# Manage Jenkins → Nodes → [agent] → Log

# Test connectivity from controller
ping agent-hostname
telnet agent-hostname 22  # SSH
telnet agent-hostname 50000  # JNLP

# Restart Jenkins agent
sudo systemctl restart jenkins-agent
```

### Issue 2: Workspace Issues

**Symptom**: "Cannot find file X" despite checkout

**Cause**: Workspace not cleaned or wrong directory

**Solution:**
```groovy
stage('Setup') {
    steps {
        cleanWs()  // Clean workspace
        checkout scm
        sh 'ls -la'  // Verify files present
    }
}
```

### Issue 3: Out of Disk Space

**Symptom**: Builds fail with disk full errors

**Cause**: Old workspaces, build artifacts accumulate

**Solution:**
```groovy
// Auto-discard old builds
options {
    buildDiscarder(logRotator(
        numToKeepStr: '10',  // Keep last 10 builds
        artifactNumToKeepStr: '5'  // Keep artifacts for last 5
    ))
}
```

```bash
# Manual cleanup on Jenkins server
# Clean workspaces
find /var/jenkins/workspace -type d -mtime +30 -exec rm -rf {} +

# Clean old builds
# Manage Jenkins → Manage Old Data
```

### Issue 4: Pipeline Syntax Errors

**Symptom**: Pipeline fails immediately with Groovy errors

**Solution:**
- Use Pipeline Syntax snippet generator
- Validate Jenkinsfile before committing

```bash
# Validate Jenkinsfile locally (requires Jenkins CLI)
java -jar jenkins-cli.jar -s http://jenkins.example.com \
    declarative-linter < Jenkinsfile
```

### Issue 5: Docker Permission Denied

**Symptom**: "permission denied while trying to connect to Docker daemon"

**Cause**: Jenkins user not in docker group

**Solution:**
```bash
# On agent machine
sudo usermod -aG docker jenkins
sudo systemctl restart jenkins

# Verify
su - jenkins
docker ps  # Should work
```

### Issue 6: Credential Not Found

**Symptom**: "could not find credentials with ID 'xyz'"

**Cause:**
- Credential doesn't exist
- Wrong scope (job vs folder vs global)
- Typo in credential ID

**Solution:**
- Check Manage Jenkins → Credentials
- Verify credential ID exactly matches
- Ensure credential in correct scope

---

## Security Considerations

### 1. Authentication & Authorization

✅ **Enable Security**
- Manage Jenkins → Security
- Enable "Jenkins' own user database" or LDAP/AD

✅ **Use Authorization Strategy**
- Matrix-based security: Fine-grained permissions
- Role-Based Strategy Plugin: Manage roles

✅ **Require Authentication**
- Never run Jenkins without authentication

### 2. Agent Security

✅ **Don't Trust Agents**
- Agents are less trusted than controller
- Don't give agents admin access
- Use separate credentials for agents

✅ **Agent-to-Master Security**
- Enable by default in modern Jenkins
- Prevents agents from modifying controller

### 3. Secret Management

✅ **Use Credentials Plugin**
- Never hardcode secrets in Jenkinsfile
- Use `credentials()` function

✅ **Mask Secrets in Logs**
```groovy
withCredentials([string(credentialsId: 'api-key', variable: 'KEY')]) {
    sh '''
        set +x  # Don't print commands (hides KEY)
        curl -H "Authorization: Bearer $KEY" ...
    '''
}
```

✅ **Rotate Credentials Regularly**

### 4. Plugin Security

✅ **Keep Plugins Updated**
- Vulnerabilities discovered regularly
- Update plugins monthly

✅ **Only Install Necessary Plugins**
- More plugins = larger attack surface

✅ **Check Plugin Security Warnings**
- Jenkins displays warnings for plugins with known issues

### 5. Network Security

✅ **Use HTTPS**
- Never run Jenkins over plain HTTP in production
- Configure reverse proxy (nginx) with SSL

✅ **Restrict Access**
- Firewall rules to limit access
- VPN for external access

✅ **Enable CSRF Protection**
- Enabled by default

---

## Performance Optimization

### 1. Pipeline Performance

**Parallelize Stages:**
```groovy
stage('Test') {
    parallel {
        stage('Unit') { ... }
        stage('Integration') { ... }
        stage('E2E') { ... }
    }
}
```

**Cache Dependencies:**
```groovy
// Use Docker volume for pip cache
agent {
    docker {
        image 'python:3.11'
        args '-v pip-cache:/root/.cache/pip'
    }
}
```

**Incremental Builds:**
```groovy
// Only build changed modules
sh 'make build-changed-only'
```

### 2. Jenkins Server Performance

**Increase Heap Size:**
```bash
# In /etc/default/jenkins or systemd unit file
JAVA_OPTS="-Xmx4g -Xms2g"
```

**Use SSD for Jenkins Home:**
- Significant performance improvement
- Jenkins does lots of disk I/O

**Separate Build Data:**
- Store workspaces on separate disk
- Keep build artifacts on different storage

### 3. Agent Optimization

**Right-Size Agents:**
- Sufficient CPU/RAM for workloads
- Don't oversubscribe executors

**Use Local Docker Registry:**
- Cache images locally
- Avoid pulling from Docker Hub repeatedly

---

## Migration Strategies

### Migrating to Jenkins

**From GitLab CI:**
- Convert `.gitlab-ci.yml` to Jenkinsfile
- Stages map directly
- Services → Docker in Docker

**From GitHub Actions:**
- Convert workflows to Jenkinsfile
- Jobs → Stages
- Actions → Plugins or custom scripts

### Migrating from Jenkins

**To GitHub Actions:**
- Simpler for GitHub-hosted projects
- Less maintenance overhead

**To GitLab CI:**
- If using GitLab for source control
- Better integrated experience

**To Kubernetes-native (Tekton, Argo):**
- Cloud-native approach
- Better if already on Kubernetes

---

## Conclusion

**Jenkins is a powerful, flexible automation server** that remains relevant despite newer alternatives.

### When to Use Jenkins:
- ✅ Need self-hosted solution
- ✅ Complex, custom workflows
- ✅ GPU-intensive ML workloads
- ✅ Integration with many systems
- ✅ Existing Jenkins infrastructure

### When to Consider Alternatives:
- ❌ Starting fresh with simple needs
- ❌ Small team without DevOps expertise
- ❌ Prefer managed service
- ❌ GitHub/GitLab hosted projects

**Key Takeaway**: Jenkins provides maximum flexibility and control at the cost of complexity. Master the basics (agents, pipelines, plugins), and you have an automation platform that can handle virtually any workflow.

---

## Further Reading

- [Jenkins Documentation](https://www.jenkins.io/doc/)
- [Pipeline Syntax Reference](https://www.jenkins.io/doc/book/pipeline/syntax/)
- [Jenkins Plugin Index](https://plugins.jenkins.io/)
- [CloudBees Jenkins Best Practices](https://www.cloudbees.com/jenkins/best-practices)
- [Jenkins for ML (Medium)](https://medium.com/search?q=jenkins+machine+learning)

---

**Remember**: Jenkins is a tool, not a solution. Success comes from good practices (pipeline as code, clean separation of concerns, proper testing) more than the tool itself.
