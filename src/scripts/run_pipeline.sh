#!/bin/bash

# YouTube MLOps Pipeline Runner
echo "🚀 Running Complete YouTube MLOps Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}🔄 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found. Please run setup.sh first."
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Step 1: Data Collection
print_step "Step 1: Collecting YouTube data..."
python src/data/data_collector.py --output data/raw/youtube_trending.csv --max-results 50

if [ $? -eq 0 ]; then
    print_success "Data collection completed"
else
    print_error "Data collection failed"
fi

# Step 2: Feature Engineering
print_step "Step 2: Engineering features..."
python src/data/feature_engineering.py --input data/raw/youtube_trending.csv --output data/processed/features.csv

if [ $? -eq 0 ]; then
    print_success "Feature engineering completed"
else
    print_error "Feature engineering failed"
fi

# Step 3: Model Training
print_step "Step 3: Training models..."
python src/models/train.py --input data/processed/features.csv --output models/

if [ $? -eq 0 ]; then
    print_success "Model training completed"
else
    print_error "Model training failed"
fi

# Step 4: Model Evaluation (Optional)
if [ -f "src/models/evaluate.py" ]; then
    print_step "Step 4: Evaluating models..."
    python src/models/evaluate.py --model-path models/ --data-path data/processed/features.csv
    
    if [ $? -eq 0 ]; then
        print_success "Model evaluation completed"
    else
        print_warning "Model evaluation failed, but continuing..."
    fi
fi

# Step 5: Start API Server
print_step "Step 5: Starting API server..."
echo "Starting FastAPI server in background..."

# Kill any existing server on port 8000
pkill -f "uvicorn.*main:app" 2>/dev/null || true
sleep 2

# Start the API server in background
cd src/api && python main.py &
API_PID=$!
cd ../..

# Wait for API to start
echo "Waiting for API to start..."
sleep 5

# Test if API is running
for i in {1..10}; do
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_success "API server is running"
        break
    fi
    if [ $i -eq 10 ]; then
        print_error "API server failed to start"
    fi
    sleep 2
done

# Step 6: Test the API
print_step "Step 6: Testing API endpoints..."

echo "Testing health endpoint..."
health_response=$(curl -s http://localhost:8000/health)
if echo "$health_response" | grep -q "healthy\|degraded"; then
    print_success "Health check passed"
    echo "Response: $health_response"
else
    print_warning "Health check unexpected response: $health_response"
fi

echo ""
echo "Testing prediction endpoint..."
prediction_response=$(curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Amazing AI Tutorial 🤖",
    "description": "Learn machine learning in this comprehensive tutorial",
    "channel_id": "UC123456789",
    "duration_seconds": 600,
    "publish_hour": 14,
    "publish_day_of_week": 1,
    "tags": ["ai", "tutorial", "machine learning"]
  }')

if echo "$prediction_response" | grep -q "predicted_views"; then
    print_success "Prediction endpoint test passed"
    echo "Sample prediction:"
    echo "$prediction_response" | python -m json.tool 2>/dev/null || echo "$prediction_response"
else
    print_warning "Prediction endpoint test failed"
    echo "Response: $prediction_response"
fi

echo ""
echo "Testing model info endpoint..."
model_info=$(curl -s http://localhost:8000/model-info)
if echo "$model_info" | grep -q "models"; then
    print_success "Model info endpoint test passed"
    echo "Model info: $model_info"
else
    print_warning "Model info unexpected response: $model_info"
fi

# Display final results
echo ""
echo "🎉 Pipeline execution completed!"
echo ""
echo "📊 Results Summary:"
echo "=================="

if [ -f "data/raw/youtube_trending.csv" ]; then
    row_count=$(tail -n +2 data/raw/youtube_trending.csv | wc -l)
    echo "📥 Raw data: $row_count videos collected"
fi

if [ -f "data/processed/features.csv" ]; then
    feature_count=$(head -1 data/processed/features.csv | tr ',' '\n' | wc -l)
    echo "🔧 Features: $feature_count columns engineered"
fi

if [ -f "models/model_metadata.json" ]; then
    echo "🤖 Models: Trained and saved"
    if command -v python &> /dev/null; then
        python -c "
import json
try:
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    performance = metadata.get('model_performance', {})
    for target, metrics in performance.items():
        r2 = metrics.get('test_r2', 0)
        rmse = metrics.get('test_rmse', 0)
        print(f'  {target}: R² = {r2:.3f}, RMSE = {rmse:.3f}')
except:
    print('  Model metadata not readable')
"
    fi
fi

echo "🌐 API: Running on http://localhost:8000"
echo ""
echo "📋 Available Endpoints:"
echo "  GET  /health      - Health check"
echo "  GET  /docs        - API documentation"
echo "  GET  /model-info  - Model information"
echo "  POST /predict     - Make predictions"
echo ""
echo "🧪 Test the API:"
echo "curl -X POST \"http://localhost:8000/predict\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"title\": \"Your Video Title\", \"channel_id\": \"UC123\", \"duration_seconds\": 300, \"publish_hour\": 14, \"publish_day_of_week\": 1}'"
echo ""
echo "🎯 Next Steps:"
echo "1. Visit http://localhost:8000/docs for interactive API documentation"
echo "2. Customize the model training parameters in src/models/train.py"
echo "3. Add your YouTube API key to .env for real data collection"
echo "4. Explore the data in data/processed/features.csv"
echo ""

# Keep API running
if [ "$1" != "--no-serve" ]; then
    echo "🔄 API server is running in background (PID: $API_PID)"
    echo "   To stop: kill $API_PID"
    echo "   To run in foreground: python src/api/main.py"
    echo ""
    echo "Press Ctrl+C to stop the pipeline and API server..."
    
    # Wait for interrupt
    trap 'print_step "Stopping API server..."; kill $API_PID 2>/dev/null; print_success "Pipeline stopped"; exit 0' INT
    
    # Keep script running
    wait $API_PID
else
    # Stop the API server
    kill $API_PID 2>/dev/null
    print_success "Pipeline completed (API server stopped)"
fi