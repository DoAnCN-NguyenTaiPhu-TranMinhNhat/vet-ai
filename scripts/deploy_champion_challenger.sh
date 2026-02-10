#!/bin/bash

# Champion-Challenger Deployment Script
# Implements zero-downtime deployment with model registry

set -e

echo "🚀 Deploying Vet-AI Champion-Challenger Strategy..."

# Configuration
NAMESPACE="vet-ai"
DOCKER_REGISTRY="your-registry.com"
IMAGE_TAG="v2.0.0"

# Build and push Docker image
echo "🔨 Building Docker image..."
docker build -t ${DOCKER_REGISTRY}/vet-ai:${IMAGE_TAG} .
docker push ${DOCKER_REGISTRY}/vet-ai:${IMAGE_TAG}

# Create namespace if not exists
echo "📁 Creating namespace..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Create secrets
echo "🔐 Creating secrets..."
kubectl create secret generic vet-ai-secrets \
  --from-literal=admin-token=$(openssl rand -hex 16) \
  --namespace=${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy infrastructure
echo "🏗️ Deploying infrastructure..."
kubectl apply -f k8s/champion-challenger-deployment.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/vet-ai-champion-challenger -n ${NAMESPACE}

# Verify deployment
echo "🔍 Verifying deployment..."
kubectl get pods -n ${NAMESPACE}
kubectl get services -n ${NAMESPACE}
kubectl get ingress -n ${NAMESPACE}

# Test API endpoints
echo "🧪 Testing API endpoints..."
API_URL=$(kubectl get ingress vet-ai-champion-challenger-ingress -n ${NAMESPACE} -o jsonpath='{.spec.rules[0].host}')

# Health check
curl -f https://${API_URL}/health || {
  echo "❌ Health check failed"
  exit 1
}

# MLOps health check
curl -f https://${API_URL}/mlops/v2/health || {
  echo "❌ MLOps health check failed"
  exit 1
}

echo "✅ Champion-Challenger deployment completed!"
echo ""
echo "🌐 API available at: https://${API_URL}"
echo "📊 MLOps endpoints: https://${API_URL}/mlops/v2"
echo ""
echo "🔑 Admin token: $(kubectl get secret vet-ai-secrets -n ${NAMESPACE} -o jsonpath='{.data.admin-token}' | base64 -d)"
echo ""
echo "📋 Next steps:"
echo "1. Train a new model: curl -X POST https://${API_URL}/mlops/v2/continuous-training"
echo "2. Evaluate staging: curl -X POST https://${API_URL}/mlops/v2/staging/{version}/evaluate"
echo "3. Request approval: curl -X POST https://${API_URL}/mlops/v2/staging/{version}/request-approval"
echo "4. Approve promotion: curl -X POST https://${API_URL}/mlops/v2/staging/{version}/approve"
echo ""
echo "📊 Monitor deployment:"
echo "kubectl logs -f deployment/vet-ai-champion-challenger -n ${NAMESPACE}"
echo "kubectl get events -n ${NAMESPACE} --sort-by=.metadata.creationTimestamp"
