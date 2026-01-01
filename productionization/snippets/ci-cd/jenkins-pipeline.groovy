// Jenkins Pipeline for Python Application
//
// This Jenkinsfile defines a declarative pipeline for building, testing, and deploying
// a Python application. Jenkins uses Groovy syntax.
//
// Key Concepts:
// - Declarative Pipeline: Structured, opinionated syntax (easier for most use cases)
// - Scripted Pipeline: Full Groovy programming (more flexible, shown in some sections)
// - Stages: Major phases of the pipeline
// - Steps: Individual actions within a stage
// - Agent: Where to run (Docker, Kubernetes, specific nodes)
// - Post: Actions to run after stages complete
//
// Prerequisites:
// - Jenkins with Docker support
// - Required plugins: Docker Pipeline, Pipeline, Git, Credentials Binding
// - Credentials configured in Jenkins (Docker registry, Kubernetes config, etc.)

// =============================================================================
// PIPELINE DEFINITION - Declarative Pipeline Syntax
// =============================================================================

pipeline {
    // Agent defines where the pipeline will execute
    // 'any' means use any available Jenkins agent
    agent any

    // =============================================================================
    // ENVIRONMENT VARIABLES - Available to all stages
    // =============================================================================
    environment {
        // Application configuration
        APP_NAME = 'python-app'
        PYTHON_VERSION = '3.11'

        // Docker configuration
        // Credentials are managed in Jenkins and accessed via credentials() function
        DOCKER_REGISTRY = 'docker.io'
        DOCKER_IMAGE = "mycompany/${APP_NAME}"
        // Docker credentials ID (configured in Jenkins Credentials)
        DOCKER_CREDENTIALS_ID = 'docker-hub-credentials'

        // Kubernetes configuration
        KUBE_CREDENTIALS_ID = 'kubernetes-credentials'
        STAGING_NAMESPACE = 'staging'
        PRODUCTION_NAMESPACE = 'production'

        // Test configuration
        MIN_COVERAGE = '80'

        // Build metadata
        // Jenkins built-in variables: BUILD_NUMBER, BUILD_ID, GIT_COMMIT, etc.
        BUILD_VERSION = "${env.GIT_BRANCH}-${env.BUILD_NUMBER}"
        BUILD_TIMESTAMP = sh(script: 'date -u +%Y-%m-%dT%H:%M:%SZ', returnStdout: true).trim()
    }

    // =============================================================================
    // PARAMETERS - Allow manual input when triggering build
    // =============================================================================
    parameters {
        // Choice parameter for deployment environment
        choice(
            name: 'DEPLOY_ENVIRONMENT',
            choices: ['none', 'staging', 'production'],
            description: 'Environment to deploy to (none = build only)'
        )

        // Boolean parameter for skipping tests (use cautiously!)
        booleanParam(
            name: 'SKIP_TESTS',
            defaultValue: false,
            description: 'Skip test execution (NOT recommended for production)'
        )

        // String parameter for custom Docker tag
        string(
            name: 'DOCKER_TAG',
            defaultValue: 'latest',
            description: 'Additional Docker tag to apply'
        )
    }

    // =============================================================================
    // TRIGGERS - Automated pipeline execution
    // =============================================================================
    triggers {
        // Poll SCM every 5 minutes for changes
        // Syntax: minute hour day month weekday
        // H = hash, distributes load across Jenkins
        pollSCM('H/5 * * * *')

        // CRON trigger for nightly builds
        cron('H 2 * * *')  // Every night at 2 AM (approximately)
    }

    // =============================================================================
    // OPTIONS - Pipeline-level settings
    // =============================================================================
    options {
        // Keep only last 10 builds
        buildDiscarder(logRotator(numToKeepStr: '10'))

        // Timeout entire pipeline after 1 hour
        timeout(time: 1, unit: 'HOURS')

        // Don't allow concurrent builds of same branch
        disableConcurrentBuilds()

        // Add timestamps to console output
        timestamps()

        // Use ANSI colors in console
        ansiColor('xterm')
    }

    // =============================================================================
    // STAGES - Main pipeline stages
    // =============================================================================
    stages {

        // -------------------------------------------------------------------------
        // STAGE: Checkout
        // Get source code from version control
        // -------------------------------------------------------------------------
        stage('Checkout') {
            steps {
                // Echo for visibility in logs
                echo "Checking out code from ${env.GIT_BRANCH}"

                // Checkout code (Jenkins automatically configures based on job settings)
                checkout scm

                // Display git information
                script {
                    // Script block allows arbitrary Groovy code
                    GIT_COMMIT_SHORT = sh(
                        script: 'git rev-parse --short HEAD',
                        returnStdout: true
                    ).trim()

                    GIT_AUTHOR = sh(
                        script: 'git log -1 --pretty=format:%an',
                        returnStdout: true
                    ).trim()

                    echo "Commit: ${GIT_COMMIT_SHORT} by ${GIT_AUTHOR}"
                }
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Build
        // Install dependencies and prepare application
        // -------------------------------------------------------------------------
        stage('Build') {
            // Use Docker agent for this stage
            // This creates a container with Python 3.11
            agent {
                docker {
                    image "python:${PYTHON_VERSION}-slim"
                    // Reuse container across steps for efficiency
                    reuseNode true
                }
            }

            steps {
                echo 'Installing dependencies...'

                // Create virtual environment and install dependencies
                sh '''
                    python -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install -r requirements-dev.txt
                '''

                // Verify installation
                sh '''
                    . .venv/bin/activate
                    pip list
                    python -c "import src; print('Application imports successful')"
                '''

                // Create build metadata file
                script {
                    writeFile file: 'build-info.txt', text: """
Build Number: ${env.BUILD_NUMBER}
Build ID: ${env.BUILD_ID}
Git Commit: ${env.GIT_COMMIT}
Git Branch: ${env.GIT_BRANCH}
Build Timestamp: ${BUILD_TIMESTAMP}
Jenkins URL: ${env.BUILD_URL}
"""
                }
            }

            // Post actions for this stage only
            post {
                success {
                    echo 'Build successful!'
                    // Archive build info
                    archiveArtifacts artifacts: 'build-info.txt', fingerprint: true
                }
                failure {
                    echo 'Build failed!'
                }
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Code Quality
        // Run linting, formatting checks, static analysis
        // -------------------------------------------------------------------------
        stage('Code Quality') {
            agent {
                docker {
                    image "python:${PYTHON_VERSION}-slim"
                    reuseNode true
                }
            }

            steps {
                echo 'Running code quality checks...'

                sh '''
                    . .venv/bin/activate

                    # Install linting tools
                    pip install flake8 black mypy isort bandit pylint

                    # Create reports directory
                    mkdir -p reports

                    # Check code formatting with Black
                    echo "Checking code formatting..."
                    black --check --diff src/ tests/ || true

                    # Check import sorting
                    echo "Checking import sorting..."
                    isort --check-only --diff src/ tests/ || true

                    # Lint with Flake8 (generate report)
                    echo "Running Flake8..."
                    flake8 src/ tests/ --format=html --htmldir=reports/flake8 --max-line-length=88 --statistics
                    flake8 src/ tests/ --max-line-length=88 --count

                    # Type checking with mypy
                    echo "Running type checker..."
                    mypy src/ --html-report reports/mypy || true

                    # Security check with Bandit
                    echo "Running security checks..."
                    bandit -r src/ -f html -o reports/bandit.html
                    bandit -r src/ -ll

                    # Code quality with Pylint
                    echo "Running Pylint..."
                    pylint src/ --output-format=html > reports/pylint.html || true
                    pylint src/ --output-format=text
                '''
            }

            post {
                always {
                    // Publish HTML reports
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'reports',
                        reportFiles: 'flake8/index.html,bandit.html,pylint.html',
                        reportName: 'Code Quality Reports'
                    ])
                }
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Unit Tests
        // Run unit tests with coverage
        // -------------------------------------------------------------------------
        stage('Unit Tests') {
            // Skip tests if parameter set (not recommended!)
            when {
                expression { params.SKIP_TESTS == false }
            }

            agent {
                docker {
                    image "python:${PYTHON_VERSION}-slim"
                    reuseNode true
                }
            }

            steps {
                echo 'Running unit tests...'

                sh '''
                    . .venv/bin/activate

                    # Run pytest with coverage
                    pytest tests/ \
                        --verbose \
                        --cov=src \
                        --cov-report=html:reports/coverage-html \
                        --cov-report=xml:reports/coverage.xml \
                        --cov-report=term \
                        --junitxml=reports/junit.xml \
                        --cov-fail-under=${MIN_COVERAGE}
                '''
            }

            post {
                always {
                    // Publish JUnit test results
                    junit 'reports/junit.xml'

                    // Publish coverage report
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'reports/coverage-html',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])

                    // Cobertura plugin for coverage visualization
                    // Requires Cobertura plugin installed
                    // cobertura coberturaReportFile: 'reports/coverage.xml'
                }
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Integration Tests
        // Run integration tests with actual services (database, cache, etc.)
        // -------------------------------------------------------------------------
        stage('Integration Tests') {
            when {
                expression { params.SKIP_TESTS == false }
                // Only run on main and develop branches
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }

            steps {
                echo 'Running integration tests...'

                script {
                    // Use docker-compose to start services
                    sh '''
                        # Start services in background
                        docker-compose -f docker-compose.test.yml up -d

                        # Wait for services to be ready
                        sleep 10

                        # Run integration tests
                        docker-compose -f docker-compose.test.yml exec -T app \
                            pytest tests/integration/ --verbose
                    '''
                }
            }

            post {
                always {
                    // Always clean up services
                    sh 'docker-compose -f docker-compose.test.yml down -v || true'
                }
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Security Scan
        // Scan dependencies for known vulnerabilities
        // -------------------------------------------------------------------------
        stage('Security Scan') {
            agent {
                docker {
                    image "python:${PYTHON_VERSION}-slim"
                    reuseNode true
                }
            }

            steps {
                echo 'Scanning for security vulnerabilities...'

                sh '''
                    . .venv/bin/activate

                    # Install security scanning tools
                    pip install safety pip-audit

                    # Scan with Safety
                    echo "Running Safety scan..."
                    safety check --json --output reports/safety.json || true
                    safety check || true

                    # Scan with pip-audit
                    echo "Running pip-audit..."
                    pip-audit --requirement requirements.txt --format json --output reports/pip-audit.json || true
                    pip-audit --requirement requirements.txt || true
                '''
            }

            post {
                always {
                    // Archive security reports
                    archiveArtifacts artifacts: 'reports/safety.json,reports/pip-audit.json', allowEmptyArchive: true
                }
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Build Docker Image
        // Create Docker image for deployment
        // -------------------------------------------------------------------------
        stage('Build Docker Image') {
            // Only build on main and develop branches
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }

            steps {
                echo "Building Docker image: ${DOCKER_IMAGE}:${BUILD_VERSION}"

                script {
                    // Build Docker image using Docker Pipeline plugin
                    dockerImage = docker.build(
                        "${DOCKER_IMAGE}:${BUILD_VERSION}",
                        "--build-arg BUILD_DATE=${BUILD_TIMESTAMP} " +
                        "--build-arg VCS_REF=${env.GIT_COMMIT} " +
                        "--build-arg VERSION=${BUILD_VERSION} " +
                        "."
                    )

                    // Tag with additional tags
                    dockerImage.tag(env.GIT_BRANCH)
                    dockerImage.tag(params.DOCKER_TAG)
                }
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Scan Docker Image
        // Scan built image for vulnerabilities using Trivy
        // -------------------------------------------------------------------------
        stage('Scan Docker Image') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }

            steps {
                echo 'Scanning Docker image for vulnerabilities...'

                script {
                    // Run Trivy in a container to scan our image
                    sh """
                        docker run --rm \
                            -v /var/run/docker.sock:/var/run/docker.sock \
                            -v \$(pwd)/reports:/reports \
                            aquasec/trivy:latest image \
                            --format json \
                            --output /reports/trivy.json \
                            ${DOCKER_IMAGE}:${BUILD_VERSION}

                        docker run --rm \
                            -v /var/run/docker.sock:/var/run/docker.sock \
                            aquasec/trivy:latest image \
                            --severity CRITICAL,HIGH \
                            --exit-code 0 \
                            ${DOCKER_IMAGE}:${BUILD_VERSION}
                    """
                }
            }

            post {
                always {
                    archiveArtifacts artifacts: 'reports/trivy.json', allowEmptyArchive: true
                }
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Push Docker Image
        // Push image to Docker registry
        // -------------------------------------------------------------------------
        stage('Push Docker Image') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }

            steps {
                echo "Pushing Docker image to ${DOCKER_REGISTRY}"

                script {
                    // Use Jenkins credentials to login and push
                    docker.withRegistry("https://${DOCKER_REGISTRY}", DOCKER_CREDENTIALS_ID) {
                        dockerImage.push(BUILD_VERSION)
                        dockerImage.push(env.GIT_BRANCH)

                        // Push 'latest' tag only from main branch
                        if (env.GIT_BRANCH == 'main') {
                            dockerImage.push('latest')
                        }
                    }
                }
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Deploy to Staging
        // Deploy to staging environment
        // -------------------------------------------------------------------------
        stage('Deploy to Staging') {
            when {
                anyOf {
                    // Deploy to staging on develop branch
                    branch 'develop'
                    // Or if manually selected
                    expression { params.DEPLOY_ENVIRONMENT == 'staging' }
                }
            }

            steps {
                echo 'Deploying to staging environment...'

                script {
                    // Use Kubernetes credentials
                    withKubeConfig([credentialsId: KUBE_CREDENTIALS_ID]) {
                        sh """
                            # Update deployment with new image
                            kubectl set image deployment/${APP_NAME} \
                                ${APP_NAME}=${DOCKER_IMAGE}:${BUILD_VERSION} \
                                -n ${STAGING_NAMESPACE}

                            # Wait for rollout to complete
                            kubectl rollout status deployment/${APP_NAME} \
                                -n ${STAGING_NAMESPACE} \
                                --timeout=5m

                            # Get deployment info
                            kubectl get deployment ${APP_NAME} -n ${STAGING_NAMESPACE}
                        """
                    }
                }

                // Run smoke tests
                echo 'Running smoke tests...'
                sh '''
                    sleep 30
                    curl -f https://staging.example.com/health || exit 1
                    curl -f https://staging.example.com/api/status || exit 1
                '''
            }
        }

        // -------------------------------------------------------------------------
        // STAGE: Deploy to Production
        // Deploy to production with approval gate
        // -------------------------------------------------------------------------
        stage('Deploy to Production') {
            when {
                anyOf {
                    branch 'main'
                    expression { params.DEPLOY_ENVIRONMENT == 'production' }
                }
            }

            steps {
                // Manual approval required for production
                script {
                    timeout(time: 24, unit: 'HOURS') {
                        input(
                            message: 'Deploy to production?',
                            ok: 'Deploy',
                            submitter: 'admin,release-managers',  // Only these users can approve
                            parameters: [
                                booleanParam(
                                    name: 'CONFIRMED',
                                    defaultValue: false,
                                    description: 'Confirm production deployment'
                                )
                            ]
                        )
                    }
                }

                echo 'Deploying to production environment...'

                script {
                    withKubeConfig([credentialsId: KUBE_CREDENTIALS_ID]) {
                        sh """
                            # Production deployment
                            kubectl set image deployment/${APP_NAME} \
                                ${APP_NAME}=${DOCKER_IMAGE}:${BUILD_VERSION} \
                                -n ${PRODUCTION_NAMESPACE}

                            # Wait and monitor
                            kubectl rollout status deployment/${APP_NAME} \
                                -n ${PRODUCTION_NAMESPACE} \
                                --timeout=10m
                        """
                    }
                }

                // Comprehensive health checks
                echo 'Verifying production deployment...'
                sh '''
                    sleep 60

                    # Run multiple health checks
                    for i in {1..5}; do
                        if curl -f https://example.com/health; then
                            echo "Production health check passed"
                            exit 0
                        fi
                        echo "Health check $i/5 failed, retrying..."
                        sleep 10
                    done

                    echo "Production health checks failed!"
                    exit 1
                '''
            }

            post {
                failure {
                    echo 'Production deployment failed, consider rollback!'
                    // Could trigger automatic rollback here
                }
            }
        }
    }

    // =============================================================================
    // POST ACTIONS - Run after all stages complete
    // =============================================================================
    post {
        // Always run, regardless of pipeline result
        always {
            echo 'Pipeline completed'

            // Clean workspace
            cleanWs(
                deleteDirs: true,
                disableDeferredWipeout: true,
                notFailBuild: true
            )
        }

        // Only on success
        success {
            echo 'Pipeline succeeded!'

            // Send success notification
            // emailext(
            //     subject: "SUCCESS: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
            //     body: "Build succeeded: ${env.BUILD_URL}",
            //     to: "${env.CHANGE_AUTHOR_EMAIL}"
            // )
        }

        // Only on failure
        failure {
            echo 'Pipeline failed!'

            // Send failure notification
            // emailext(
            //     subject: "FAILURE: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
            //     body: "Build failed: ${env.BUILD_URL}\nConsole: ${env.BUILD_URL}console",
            //     to: "${env.CHANGE_AUTHOR_EMAIL},dev-team@example.com"
            // )

            // Could also notify Slack, Teams, etc.
            // slackSend(
            //     color: 'danger',
            //     message: "Pipeline failed: ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
            // )
        }

        // Only on first failure after success
        regression {
            echo 'Pipeline regressed!'
        }

        // Only on first success after failure
        fixed {
            echo 'Pipeline fixed!'
        }

        // Only when build status changed
        changed {
            echo 'Pipeline status changed!'
        }
    }
}
