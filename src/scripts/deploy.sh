### 15. **scripts/deploy.sh** - CREATE NEW FILE
```bash
#!/bin/bash

# YouTube MLOps Deployment Script

echo "🚀 Deploying YouTube MLOps Pipeline..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if models exist
if [ ! -f "models/xgboost_view_count.joblib" ]; then
    echo "⚠️  Model files not found. Training models first..."
    source venv/bin/activate
    python src/models/train.py
fi

# Build and start services
echo "🏗️  Building and starting services..."
cd deployment/docker

# Pull latest images
echo "📥 Pulling latest images..."
docker-compose pull

# Build custom images
echo "🔨 Building custom images..."
docker-compose build

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    echo "Checking $service_name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -f $url &> /dev/null; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 5
        ((attempt++))
    done
    
    echo "❌ $service_name failed to become healthy"
    return 1
}

# Test service endpoints
echo "🧪 Testing service endpoints..."
check_service "API" "http://localhost:8000/health"
check_service "MLflow" "http://localhost:5000"
check_service "Grafana" "http://localhost:3000"
check_service "Prometheus" "http://localhost:9090"

# Test API prediction endpoint
echo "🎯 Testing API prediction endpoint..."
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Video 🎥",
    "description": "Test description",
    "channel_id": "UC123456",
    "duration_seconds": 300,
    "publish_hour": 14,
    "publish_day_of_week": 1,
    "tags": ["test", "video"]
  }' | jq . || echo "⚠️  API prediction test failed"

echo "✅ Deployment completed!"
echo ""
echo "Services available at:"
echo "📊 API: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "📈 MLflow: http://localhost:5000"
echo "📋 Grafana: http://localhost:3000 (admin/admin)"
echo "🎯 Prometheus: http://localhost:9090"
echo "🌊 Airflow: http://localhost:8080 (admin/admin)"
echo "📱 Dashboard: http://localhost:8501"