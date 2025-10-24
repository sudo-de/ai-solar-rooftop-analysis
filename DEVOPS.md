# DevOps Guide - Solar Rooftop Analysis

This document provides comprehensive DevOps instructions for deploying and managing the Solar Rooftop Analysis application.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Monitoring & Observability](#monitoring--observability)
7. [Security](#security)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Docker**: 20.10+ with Docker Compose 2.0+
- **Node.js**: 18+ (for local development)
- **Python**: 3.11+ (for local development)
- **Kubernetes**: 1.24+ (for K8s deployment)
- **kubectl**: Latest version
- **Helm**: 3.8+ (optional, for advanced deployments)

### Environment Variables

Create environment files for different environments:

```bash
# .env.development
ENVIRONMENT=development
LOG_LEVEL=DEBUG
OPENWEATHER_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
SENTINEL_API_KEY=your_key_here
LANDSAT_API_KEY=your_key_here

# .env.staging
ENVIRONMENT=staging
LOG_LEVEL=INFO
# ... same API keys

# .env.production
ENVIRONMENT=production
LOG_LEVEL=WARNING
# ... same API keys
```

## Local Development

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ai-solar-rooftop-analysis

# Start development environment
./deploy.sh --environment development

# Or manually with Docker Compose
docker-compose -f docker-compose.dev.yml up --build
```

### Development Commands

```bash
# Backend development
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend development
cd frontend
npm install
npm run dev

# Run tests
cd backend && python -m pytest test_main.py -v
cd frontend && npm test
```

## Docker Deployment

### Production Deployment

```bash
# Deploy to production
./deploy.sh --environment production --clean

# Or manually
docker-compose up --build -d
```

### Development Deployment

```bash
# Deploy to development
./deploy.sh --environment development

# Or manually
docker-compose -f docker-compose.dev.yml up --build -d
```

### Docker Commands

```bash
# View logs
docker-compose logs -f [service]

# Scale services
docker-compose up --scale backend=3

# Stop services
docker-compose down

# Clean up
docker-compose down -v --remove-orphans
docker system prune -f
```

### Docker Images

The application uses multi-stage builds for optimized production images:

- **Backend**: `ghcr.io/your-org/ai-solar-rooftop-analysis-backend:latest`
- **Frontend**: `ghcr.io/your-org/ai-solar-rooftop-analysis-frontend:latest`

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Persistent volume support

### Staging Deployment

```bash
# Create namespace
kubectl apply -f k8s/staging/namespace.yaml

# Deploy persistent volumes
kubectl apply -f k8s/staging/persistent-volumes.yaml

# Deploy services
kubectl apply -f k8s/staging/redis-deployment.yaml
kubectl apply -f k8s/staging/backend-deployment.yaml
kubectl apply -f k8s/staging/frontend-deployment.yaml

# Deploy ingress
kubectl apply -f k8s/staging/ingress.yaml
```

### Production Deployment

```bash
# Similar to staging but with production configurations
kubectl apply -f k8s/production/
```

### Kubernetes Commands

```bash
# Check pod status
kubectl get pods -n solar-analysis-staging

# View logs
kubectl logs -f deployment/backend -n solar-analysis-staging

# Scale deployment
kubectl scale deployment backend --replicas=3 -n solar-analysis-staging

# Port forward for testing
kubectl port-forward service/backend-service 8000:8000 -n solar-analysis-staging
kubectl port-forward service/frontend-service 3000:3000 -n solar-analysis-staging
```

## CI/CD Pipeline

### GitHub Actions

The project includes a comprehensive CI/CD pipeline:

1. **Test Stage**: Runs linting, unit tests, and coverage reports
2. **Build Stage**: Builds and pushes Docker images
3. **Security Stage**: Runs vulnerability scanning
4. **Deploy Stage**: Deploys to staging/production

### Pipeline Features

- **Multi-environment support**: Development, staging, production
- **Security scanning**: Trivy vulnerability scanner
- **Code quality**: Linting with flake8, black, isort
- **Testing**: pytest for Python, Jest for TypeScript
- **Coverage**: Code coverage reporting with Codecov
- **Caching**: Docker layer caching for faster builds

### Manual Deployment

```bash
# Deploy specific environment
./deploy.sh --environment production --clean

# Deploy with monitoring
./deploy.sh --environment production --monitoring

# Skip tests (not recommended for production)
./deploy.sh --environment staging --skip-tests
```

## Monitoring & Observability

### Prometheus & Grafana

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access monitoring
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3001 (admin/admin)
```

### Health Checks

All services include health check endpoints:

- **Backend**: `http://localhost:8000/api/health`
- **Frontend**: `http://localhost:3000`
- **Redis**: `redis-cli ping`

### Logging

Logs are structured and include:

- **Application logs**: Application-specific events
- **Access logs**: HTTP request/response logs
- **Error logs**: Error tracking and debugging
- **Performance logs**: Performance metrics and timing

### Metrics

Key metrics monitored:

- **Response times**: API endpoint performance
- **Error rates**: 4xx/5xx HTTP status codes
- **Resource usage**: CPU, memory, disk
- **Business metrics**: Analysis requests, success rates

## Security

### Container Security

- **Non-root users**: All containers run as non-root
- **Minimal base images**: Alpine Linux for smaller attack surface
- **Security scanning**: Trivy vulnerability scanning
- **Secrets management**: Environment variables for sensitive data

### Network Security

- **Internal networking**: Services communicate via internal networks
- **Rate limiting**: Nginx rate limiting for API protection
- **SSL/TLS**: HTTPS support with proper certificates
- **Firewall rules**: Restricted port access

### Best Practices

1. **Regular updates**: Keep base images and dependencies updated
2. **Secret rotation**: Rotate API keys and secrets regularly
3. **Access control**: Implement proper RBAC for Kubernetes
4. **Audit logging**: Enable audit logs for security events

## Troubleshooting

### Common Issues

#### Backend Won't Start

```bash
# Check logs
docker-compose logs backend

# Common fixes
# 1. Port already in use
sudo lsof -i :8000
sudo kill -9 <PID>

# 2. Missing dependencies
docker-compose down
docker-compose up --build

# 3. Permission issues
sudo chown -R $USER:$USER outputs uploads logs
```

#### Frontend Build Fails

```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

#### Database Connection Issues

```bash
# Check Redis status
docker-compose exec redis redis-cli ping

# Restart Redis
docker-compose restart redis
```

### Performance Issues

#### High Memory Usage

```bash
# Check resource usage
docker stats

# Scale services
docker-compose up --scale backend=2
```

#### Slow Response Times

```bash
# Check logs for errors
docker-compose logs -f

# Monitor with Prometheus
# Access http://localhost:9090
```

### Debugging Commands

```bash
# Enter container
docker-compose exec backend bash
docker-compose exec frontend sh

# Check service health
curl http://localhost:8000/api/health
curl http://localhost:3000

# View detailed logs
docker-compose logs --tail=100 -f backend
```

### Recovery Procedures

#### Complete Reset

```bash
# Stop all services
docker-compose down -v --remove-orphans

# Clean up everything
docker system prune -a -f
docker volume prune -f

# Rebuild from scratch
./deploy.sh --environment production --clean
```

#### Data Recovery

```bash
# Backup data
docker run --rm -v solar-analysis_outputs:/data -v $(pwd):/backup alpine tar czf /backup/outputs-backup.tar.gz -C /data .

# Restore data
docker run --rm -v solar-analysis_outputs:/data -v $(pwd):/backup alpine tar xzf /backup/outputs-backup.tar.gz -C /data
```

## Support

For DevOps-related issues:

1. Check the troubleshooting section above
2. Review application logs
3. Check system resources
4. Verify network connectivity
5. Contact the development team

## Contributing

When contributing to DevOps configurations:

1. Test changes in development environment first
2. Update documentation for any new procedures
3. Ensure security best practices are followed
4. Test rollback procedures
5. Update CI/CD pipeline if needed
