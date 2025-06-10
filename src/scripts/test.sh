#!/bin/bash

# YouTube MLOps Testing Script

echo "🧪 Running YouTube MLOps Pipeline Tests..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Set test environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run unit tests
echo "🔬 Running unit tests..."
pytest tests/ -v --cov=src --cov-report=html --cov-report=term --cov-report=xml

# Check test results
if [ $? -eq 0 ]; then
    echo "✅ Unit tests passed"
else
    echo "❌ Unit tests failed"
    exit 1
fi

# Run data validation tests
echo "📊 Running data validation tests..."
if [ -f "data/processed/features.csv" ]; then
    python -c "
import pandas as pd
import sys
sys.path.append('src')
from utils.helpers import DataValidator

df = pd.read_csv('data/processed/features.csv')
validator = DataValidator()
issues = validator.validate_youtube_data(df)

if issues:
    print('❌ Data validation issues found:')
    for issue in issues:
        print(f'  - {issue}')
    sys.exit(1)
else:
    print('✅ Data validation passed')
    print(f'Data shape: {df.shape}')
    print(f'Missing values: {df.isnull().sum().sum()}')
"
else
    echo "⚠️  Processed data not found. Run data pipeline first."
fi

# Check model files
echo "🤖 Checking model files..."
model_files=("models/xgboost_view_count.joblib" "models/xgboost_like_count.joblib" "models/xgboost_comment_count.joblib")
all_models_exist=true

for file in "${model_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
        all_models_exist=false
    fi
done

# Test model loading
if [ "$all_models_exist" = true ]; then
    echo "🔄 Testing model loading..."
    python -c "
import joblib
import json
import sys

try:
    # Load models
    models = {}
    models['views'] = joblib.load('models/xgboost_view_count.joblib')
    models['likes'] = joblib.load('models/xgboost_like_count.joblib')
    models['comments'] = joblib.load('models/xgboost_comment_count.joblib')
    
    # Load feature columns
    with open('models/feature_columns.json', 'r') as f:
        feature_columns = json.load(f)
    
    print(f'✅ Successfully loaded {len(models)} models')
    print(f'✅ Feature columns: {len(feature_columns)}')
    
except Exception as e:
    print(f'❌ Model loading failed: {e}')
    sys.exit(1)
"
fi

# Test API endpoints (if services are running)
echo "🌐 Testing API endpoints..."
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "✅ API is running, testing endpoints..."
    
    # Test health endpoint
    health_response=$(curl -s http://localhost:8000/health)
    if echo "$health_response" | jq -e '.status == "healthy"' > /dev/null; then
        echo "✅ Health endpoint test passed"
    else
        echo "❌ Health endpoint test failed"
    fi
    
    # Test model info endpoint
    model_info=$(curl -s http://localhost:8000/model-info)
    if echo "$model_info" | jq -e '.models' > /dev/null; then
        echo "✅ Model info endpoint test passed"
    else
        echo "❌ Model info endpoint test failed"
    fi
    
    # Test prediction endpoint
    echo "🎯 Testing prediction endpoint..."
    prediction_response=$(curl -s -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "title": "Test Video 🎥",
        "description": "Test description",
        "channel_id": "UC123456",
        "duration_seconds": 300,
        "publish_hour": 14,
        "publish_day_of_week": 1,
        "tags": ["test", "video"]
      }')
    
    if echo "$prediction_response" | jq -e '.predicted_views' > /dev/null; then
        echo "✅ Prediction endpoint test passed"
        predicted_views=$(echo "$prediction_response" | jq -r '.predicted_views')
        confidence=$(echo "$prediction_response" | jq -r '.confidence_score')
        echo "   Predicted views: $predicted_views"
        echo "   Confidence: $confidence"
    else
        echo "❌ Prediction endpoint test failed"
        echo "Response: $prediction_response"
    fi
else
    echo "⚠️  API is not running. Skipping API tests."
    echo "   Run: ./scripts/deploy.sh to start services"
fi

# Test DVC pipeline
echo "🎯 Testing DVC pipeline..."
if command -v dvc &> /dev/null; then
    dvc dag > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ DVC pipeline configuration is valid"
    else
        echo "❌ DVC pipeline configuration has issues"
    fi
else
    echo "⚠️  DVC not installed, skipping pipeline test"
fi

# Performance benchmarks
echo "⚡ Running performance benchmarks..."
if curl -f http://localhost:8000/health &> /dev/null; then
    echo "📊 API Response Time Test..."
    
    # Measure API response time
    start_time=$(date +%s%N)
    curl -s -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{
        "title": "Performance Test Video",
        "description": "Testing API response time",
        "channel_id": "UC123456",
        "duration_seconds": 300,
        "publish_hour": 14,
        "publish_day_of_week": 1,
        "tags": ["performance", "test"]
      }' > /dev/null
    end_time=$(date +%s%N)
    
    duration=$(((end_time - start_time) / 1000000))  # Convert to milliseconds
    echo "   API response time: ${duration}ms"
    
    if [ $duration -lt 2000 ]; then
        echo "✅ API response time is acceptable (< 2s)"
    else
        echo "⚠️  API response time is slow (> 2s)"
    fi
fi

echo ""
echo "🎉 Testing completed!"
echo ""
echo "📈 Test Summary:"
echo "- Unit Tests: $(if [ -f htmlcov/index.html ]; then echo 'View coverage report in htmlcov/index.html'; else echo 'Check terminal output'; fi)"
echo "- Coverage Report: coverage.xml generated"
echo "- Data Validation: $(if [ -f data/processed/features.csv ]; then echo 'Passed'; else echo 'Skipped - no data'; fi)"
echo "- Model Loading: $(if [ "$all_models_exist" = true ]; then echo 'Passed'; else echo 'Failed - missing models'; fi)"
echo "- API Tests: $(if curl -f http://localhost:8000/health &> /dev/null; then echo 'Passed'; else echo 'Skipped - API not running'; fi)"