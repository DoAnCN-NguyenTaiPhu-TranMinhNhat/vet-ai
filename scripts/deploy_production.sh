#!/bin/bash

# Production Deployment Script
# This script handles the deployment of Vet-AI with persistent model storage

set -e

echo "🚀 Starting Vet-AI Production Deployment..."

# Check if .env.production exists
if [ ! -f ".env.production" ]; then
    echo "❌ .env.production file not found. Please create it first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ./ai_service/models
mkdir -p ./logs
mkdir -p ./data

# Build and start services
echo "🔨 Building Docker images..."
docker-compose -f docker-compose.prod.yml build --no-cache

echo "🔄 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."
docker-compose -f docker-compose.prod.yml ps

# Verify model storage
echo "🔍 Verifying model storage..."
docker-compose -f docker-compose.prod.yml exec vet-ai python scripts/verify_model_storage.py

echo "✅ Production deployment completed!"
echo ""
echo "🌐 Services available at:"
echo "  - API: http://localhost:8000"
echo "  - MLflow: http://localhost:5000"
echo "  - PostgreSQL: localhost:5432"
echo ""
echo "📊 To check logs:"
echo "  docker-compose -f docker-compose.prod.yml logs -f vet-ai"
echo ""
echo "🛑 To stop services:"
echo "  docker-compose -f docker-compose.prod.yml down"
