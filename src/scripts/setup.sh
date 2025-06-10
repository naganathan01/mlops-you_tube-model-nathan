#!/bin/bash

# YouTube MLOps Pipeline Setup Script
echo "🚀 Setting up YouTube MLOps Pipeline..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
fi

print_status "Python 3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt
print_status "Dependencies installed"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,external}
mkdir -p models
mkdir -p logs
mkdir -p metrics
mkdir -p src/{data,models,api,monitoring,utils}
mkdir -p tests/{test_data,test_models,test_api}
mkdir -p deployment/docker
mkdir -p airflow/dags
mkdir -p monitoring/{prometheus,grafana}

print_status "Directory structure created"

# Create __init__.py files
echo "🐍 Creating Python package structure..."
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/api/__init__.py
touch src/monitoring/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
touch tests/test_data/__init__.py
touch tests/test_models/__init__.py
touch tests/test_api/__init__.py

print_status "Python packages initialized"

# Copy environment template
if [ ! -f .env ]; then
    echo "🔧 Creating environment file..."
    cp .env.template .env
    print_status "Environment file created"
    print_warning "Please edit .env file with your API keys and configuration"
else
    print_warning ".env file already exists"
fi

# Initialize git repository if not exists
if [ ! -d ".git" ]; then
    echo "📚 Initializing git repository..."
    git init
    print_status "Git repository initialized"
else
    print_warning "Git repository already exists"
fi

# Create .gitignore if not exists
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Environment variables
.env

# Models and data
models/*.joblib
models/*.pkl
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# MLflow
mlruns/
mlartifacts/

# Jupyter
.ipynb_checkpoints/

# Testing
.coverage
htmlcov/
.pytest_cache/
EOF
    print_status ".gitignore created"
fi

# Create .gitkeep files for empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch logs/.gitkeep
touch models/.gitkeep

# Test data collection with sample data
echo "🧪 Testing data collection..."
python3 -c "
import sys
sys.path.append('src')
from data.data_collector import YouTubeDataCollector
import os

# Test with dummy key (will generate sample data)
collector = YouTubeDataCollector('dummy_key')
df = collector._generate_sample_data()
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/youtube_trending.csv', index=False)
print(f'✅ Sample data created: {len(df)} videos')
"

if [ $? -eq 0 ]; then
    print_status "Data collection test passed"
else
    print_error "Data collection test failed"
fi

# Test feature engineering
echo "🔧 Testing feature engineering..."
python3 -c "
import sys
sys.path.append('src')
from data.feature_engineering import YouTubeFeatureEngineer
import pandas as pd
import os

# Load sample data
df = pd.read_csv('data/raw/youtube_trending.csv')
engineer = YouTubeFeatureEngineer()
df_features = engineer.engineer_features(df)

os.makedirs('data/processed', exist_ok=True)
df_features.to_csv('data/processed/features.csv', index=False)

# Save feature columns
import json
feature_cols = [col for col in df_features.columns 
               if col not in ['video_id', 'title', 'description', 'published_at', 
                            'collected_at', 'tags', 'channel_id', 'channel_title', 
                            'duration', 'view_count', 'like_count', 'comment_count']]

with open('data/processed/feature_columns.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

print(f'✅ Features engineered: {df_features.shape}')
"

if [ $? -eq 0 ]; then
    print_status "Feature engineering test passed"
else
    print_error "Feature engineering test failed"
fi

# Test model training
echo "🤖 Testing model training..."
python3 -c "
import sys
sys.path.append('src')
from models.train import YouTubePerformancePredictor
import pandas as pd

# Load processed data
df = pd.read_csv('data/processed/features.csv')
predictor = YouTubePerformancePredictor()

# Train models
X, y = predictor.prepare_data(df)
results = predictor.train_models(X, y)
predictor.save_models('models/')

print(f'✅ Models trained and saved')
for target, info in results.items():
    print(f'  {target}: R² = {info[\"metrics\"][\"test_r2\"]:.3f}')
"

if [ $? -eq 0 ]; then
    print_status "Model training test passed"
else
    print_error "Model training test failed"
fi

# Make scripts executable
chmod +x src/scripts/*.sh 2>/dev/null || true

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file with your YouTube API key (optional)"
echo "2. Test the API: python src/api/main.py"
echo "3. Open another terminal and test prediction:"
echo "   curl -X POST \"http://localhost:8000/predict\" \\"
echo "   -H \"Content-Type: application/json\" \\"
echo "   -d '{\"title\": \"Test Video\", \"channel_id\": \"UC123\", \"duration_seconds\": 300, \"publish_hour\": 14, \"publish_day_of_week\": 1}'"
echo ""
echo "📊 Available commands:"
echo "  - python src/data/data_collector.py (collect YouTube data)"
echo "  - python src/data/feature_engineering.py (engineer features)"  
echo "  - python src/models/train.py (train models)"
echo "  - python src/api/main.py (start API server)"
echo ""
echo "🎯 Project structure:"
echo "  - data/: Raw and processed data"
echo "  - models/: Trained models"
echo "  - src/: Source code"
echo "  - tests/: Test files"
echo ""
print_status "MLOps pipeline is ready to use!"