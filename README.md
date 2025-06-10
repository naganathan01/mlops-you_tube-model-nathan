# 🎥 YouTube Performance Predictor - MLOps Pipeline

A complete end-to-end MLOps pipeline for predicting YouTube video performance using machine learning. This beginner-friendly project demonstrates data collection, feature engineering, model training, API deployment, and monitoring.

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Clone and setup
git clone <your-repo>
cd youtube-mlops-project

# 2. Run automated setup
chmod +x setup.sh
./setup.sh

# 3. Run complete pipeline
chmod +x run_pipeline.sh
./run_pipeline.sh
```

That's it! Your MLOps pipeline is running at http://localhost:8000

## 📁 Project Structure

```
youtube-mlops-project/
├── data/
│   ├── raw/                    # Raw YouTube data
│   └── processed/              # Processed features
├── models/                     # Trained ML models
├── src/
│   ├── data/
│   │   ├── data_collector.py   # YouTube API data collection
│   │   └── feature_engineering.py # Feature creation
│   ├── models/
│   │   ├── train.py           # Model training
│   │   └── evaluate.py        # Model evaluation
│   ├── api/
│   │   └── main.py            # FastAPI server
│   └── utils/                 # Helper functions
├── tests/                     # Test files
├── requirements.txt           # Python dependencies
├── setup.sh                  # Setup script
├── run_pipeline.sh           # Pipeline runner
└── README.md                 # This file
```

## 🎯 Features

### ✅ Complete MLOps Pipeline
- **Data Collection**: YouTube API integration with fallback sample data
- **Feature Engineering**: 20+ engineered features (text, time, engagement)
- **Model Training**: XGBoost models for views, likes, and comments
- **API Deployment**: FastAPI with automatic documentation
- **Monitoring**: Health checks and performance tracking
- **Testing**: Comprehensive test suite

### 🤖 Machine Learning
- **Multi-target prediction**: Views, likes, comments
- **Feature engineering**: Title analysis, timing, content features
- **Model evaluation**: R², RMSE, cross-validation
- **Model persistence**: Joblib serialization

### 🌐 API Features
- **RESTful API**: GET/POST endpoints
- **Input validation**: Pydantic schemas
- **Error handling**: Graceful failures
- **Documentation**: Auto-generated OpenAPI docs
- **Health monitoring**: System status endpoints

## 📋 Prerequisites

- Python 3.8+
- Git
- curl (for testing)
- (Optional) YouTube API key for real data

## 🛠️ Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd youtube-mlops-project

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/{raw,processed} models logs metrics

# Set up environment
cp .env.template .env
```

## 🚀 Usage

### Run Complete Pipeline

```bash
# Automated pipeline (recommended)
./run_pipeline.sh

# Or step by step:
python src/data/data_collector.py
python src/data/feature_engineering.py
python src/models/train.py
python src/api/main.py
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Amazing AI Tutorial 🤖",
    "description": "Learn machine learning basics",
    "channel_id": "UC123456789",
    "duration_seconds": 600,
    "publish_hour": 14,
    "publish_day_of_week": 1,
    "tags": ["ai", "tutorial", "ml"]
  }'
```

### Use the Dashboard

```bash
# Install Streamlit
pip install streamlit

# Run dashboard
streamlit run dashboard.py

# Open http://localhost:8501
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and system status |
| `/predict` | POST | Predict video performance |
| `/model-info` | GET | Model metadata and info |
| `/docs` | GET | Interactive API documentation |

### Example API Response

```json
{
  "predicted_views": 15432.7,
  "predicted_likes": 1205.3,
  "predicted_comments": 89.2,
  "confidence_score": 0.85,
  "recommendations": [
    "Consider adding emojis to increase engagement",
    "Optimal posting time detected"
  ],
  "model_version": "1.0.0",
  "prediction_id": "pred_20240101_120000_abc123"
}
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# YouTube API (optional - uses sample data if not provided)
YOUTUBE_API_KEY=your_youtube_api_key_here

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Model Configuration
MODEL_REGISTRY_PATH=./models
```

### Model Parameters

Edit `src/models/train.py` to customize:

```python
# XGBoost parameters
model = xgb.XGBRegressor(
    n_estimators=100,     # Number of trees
    max_depth=6,          # Tree depth
    learning_rate=0.1,    # Learning rate
    random_state=42
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test specific component
pytest tests/test_api/ -v
```

## 📈 Monitoring

### Health Monitoring

```bash
# Check system health
curl http://localhost:8000/health

# Response:
{
  "status": "healthy",
  "models_loaded": 3,
  "features_available": 18,
  "version": "1.0.0"
}
```

### Model Performance

Check `metrics/model_performance.json` for:
- R² scores
- RMSE values
- Training metrics

## 🐳 Docker Deployment

```bash
# Build image
docker build -t youtube-mlops-api .

# Run container
docker run -p 8000:8000 youtube-mlops-api

# Or use Docker Compose
docker-compose up -d
```

## 🎓 Learning Objectives

This project teaches:

1. **Data Engineering**: API integration, data cleaning, feature engineering
2. **Machine Learning**: Multi-target regression, model evaluation, serialization
3. **MLOps**: Model versioning, API deployment, monitoring
4. **Software Engineering**: Testing, documentation, containerization
5. **API Development**: FastAPI, validation, error handling

## 🔍 Troubleshooting

### Common Issues

**API won't start:**
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
pkill -f "uvicorn.*main:app"
```

**Models not loading:**
```bash
# Retrain models
python src/models/train.py

# Check model files
ls -la models/
```

**Data collection fails:**
```bash
# Check if using sample data
grep "Generating sample data" logs/*

# Verify API key in .env
cat .env | grep YOUTUBE_API_KEY
```

### Getting Help

1. Check the logs: `tail -f logs/mlops_pipeline.log`
2. Test individual components: `python src/data/data_collector.py`
3. Validate data: `python -c "import pandas as pd; print(pd.read_csv('data/processed/features.csv').info())"`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Run tests: `pytest tests/`
4. Submit pull request

## 📚 Next Steps

### Beginner Enhancements
- [ ] Add more video features (thumbnail analysis)
- [ ] Implement data validation checks
- [ ] Add model retraining scheduler
- [ ] Create performance dashboards

### Advanced Features
- [ ] A/B testing framework
- [ ] Real-time data streaming
- [ ] Model drift detection
- [ ] Kubernetes deployment
- [ ] CI/CD pipelines

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- YouTube Data API v3
- FastAPI and Uvicorn
- XGBoost team
- Streamlit community

---

## 🆘 Quick Commands Reference

```bash
# Setup and run everything
./setup.sh && ./run_pipeline.sh

# Individual components
python src/data/data_collector.py          # Collect data
python src/data/feature_engineering.py     # Engineer features  
python src/models/train.py                 # Train models
python src/api/main.py                     # Start API

# Testing
pytest tests/ -v                           # Run tests
curl http://localhost:8000/health          # Test API

# Cleanup
pkill -f "uvicorn.*main:app"              # Stop API
deactivate                                # Exit venv
```

**Happy Learning! 🚀**