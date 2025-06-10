import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Mock the model loading to avoid file dependencies in tests
@patch('src.api.main.joblib.load')
@patch('builtins.open')
@patch('src.api.main.redis.Redis.from_url')
def test_api_startup(mock_redis, mock_open, mock_joblib):
    """Test API startup and model loading"""
    # Mock model loading
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([10.0])
    mock_joblib.return_value = mock_model
    
    # Mock feature columns file
    mock_file = MagicMock()
    mock_file.read.return_value = '["feature1", "feature2", "feature3"]'
    mock_open.return_value.__enter__.return_value = mock_file
    
    # Mock Redis
    mock_redis_instance = MagicMock()
    mock_redis_instance.ping.return_value = True
    mock_redis.return_value = mock_redis_instance
    
    from src.api.main import app
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data

class TestAPIEndpoints:
    
    def setup_method(self):
        """Setup test client with mocked dependencies"""
        with patch('src.api.main.joblib.load'), \
             patch('builtins.open'), \
             patch('src.api.main.redis.Redis.from_url'):
            
            from src.api.main import app
            self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        
        required_fields = ["status", "timestamp", "models_loaded", "features_available", "redis_connected", "version"]
        for field in required_fields:
            assert field in data
    
    @patch('src.api.main.models')
    @patch('src.api.main.feature_columns')
    def test_predict_endpoint_valid_input(self, mock_feature_columns, mock_models):
        """Test prediction with valid input"""
        # Setup mocks
        mock_feature_columns.__len__ = MagicMock(return_value=10)
        mock_feature_columns.__iter__ = MagicMock(return_value=iter(['feature1', 'feature2']))
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([10.0])
        mock_models.__getitem__ = MagicMock(return_value=mock_model)
        mock_models.__len__ = MagicMock(return_value=3)
        
        test_data = {
            "title": "Test Video Title",
            "description": "Test description",
            "channel_id": "UC123456",
            "duration_seconds": 300,
            "publish_hour": 14,
            "publish_day_of_week": 1,
            "tags": ["test", "video"]
        }
        
        with patch('src.api.main.create_features') as mock_create_features:
            mock_create_features.return_value = pd.DataFrame({'feature1': [1], 'feature2': [2]})
            
            response = self.client.post("/predict", json=test_data)
            assert response.status_code == 200
            data = response.json()
            
            required_fields = [
                "predicted_views", "predicted_likes", "predicted_comments",
                "confidence_score", "recommendations", "model_version", "prediction_id"
            ]
            
            for field in required_fields:
                assert field in data
            
            assert 0 <= data["confidence_score"] <= 1
            assert data["predicted_views"] >= 0
            assert data["predicted_likes"] >= 0
            assert data["predicted_comments"] >= 0
    
    def test_predict_endpoint_invalid_input(self):
        """Test prediction with invalid input"""
        invalid_data = {
            "title": "",  # Empty title
            "duration_seconds": -1,  # Negative duration
            "publish_hour": 25  # Invalid hour
        }
        
        response = self.client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = self.client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    @patch('src.api.main.models')
    @patch('src.api.main.feature_columns')
    def test_model_info_endpoint(self, mock_feature_columns, mock_models):
        """Test model info endpoint"""
        mock_models.__len__ = MagicMock(return_value=3)
        mock_models.items = MagicMock(return_value=[
            ('views', MagicMock(__class__=MagicMock(__name__='XGBRegressor'))),
            ('likes', MagicMock(__class__=MagicMock(__name__='XGBRegressor'))),
            ('comments', MagicMock(__class__=MagicMock(__name__='XGBRegressor')))
        ])
        mock_feature_columns.__len__ = MagicMock(return_value=42)
        
        response = self.client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        
        assert "model_version" in data
        assert "models" in data
        assert "feature_count" in data
        assert "last_updated" in data

class TestInputValidation:
    
    def setup_method(self):
        """Setup test client"""
        with patch('src.api.main.joblib.load'), \
             patch('builtins.open'), \
             patch('src.api.main.redis.Redis.from_url'):
            
            from src.api.main import app
            self.client = TestClient(app)
    
    def test_title_length_validation(self):
        """Test title length validation"""
        # Title too long
        long_title = "x" * 201
        data = {
            "title": long_title,
            "channel_id": "UC123456",
            "duration_seconds": 300,
            "publish_hour": 14,
            "publish_day_of_week": 1
        }
        
        response = self.client.post("/predict", json=data)
        assert response.status_code == 422
    
    def test_duration_validation(self):
        """Test duration validation"""
        # Duration too long (> 12 hours)
        data = {
            "title": "Test Video",
            "channel_id": "UC123456",
            "duration_seconds": 50000,
            "publish_hour": 14,
            "publish_day_of_week": 1
        }
        
        response = self.client.post("/predict", json=data)
        assert response.status_code == 422
    
    def test_hour_validation(self):
        """Test hour validation"""
        data = {
            "title": "Test Video",
            "channel_id": "UC123456",
            "duration_seconds": 300,
            "publish_hour": 25,  # Invalid hour
            "publish_day_of_week": 1
        }
        
        response = self.client.post("/predict", json=data)
        assert response.status_code == 422

@pytest.fixture
def sample_video_data():
    """Sample video data for testing"""
    return {
        "title": "Amazing AI Tutorial 🤖",
        "description": "Learn machine learning basics",
        "channel_id": "UC123456789",
        "duration_seconds": 600,
        "publish_hour": 14,
        "publish_day_of_week": 1,
        "tags": ["ai", "tutorial", "ml"],
        "category_id": "22"
    }

def test_feature_creation(sample_video_data):
    """Test feature creation function"""
    from src.api.main import VideoInput
    
    video_input = VideoInput(**sample_video_data)
    
    # Test basic attributes
    assert video_input.title == "Amazing AI Tutorial 🤖"
    assert video_input.duration_seconds == 600
    assert video_input.publish_hour == 14
    assert len(video_input.tags) == 3

def test_recommendation_generation():
    """Test recommendation generation"""
    from src.api.main import generate_recommendations, VideoInput
    
    # Test short title recommendation
    video = VideoInput(
        title="Short",
        channel_id="UC123",
        duration_seconds=300,
        publish_hour=14,
        publish_day_of_week=1
    )
    
    recommendations = generate_recommendations(video, 10000, 1000)
    assert any("longer" in rec for rec in recommendations)
    
    # Test late posting recommendation
    video = VideoInput(
        title="Test Video Title That Is Long Enough",
        channel_id="UC123",
        duration_seconds=300,
        publish_hour=2,  # Very early hour
        publish_day_of_week=1
    )
    
    recommendations = generate_recommendations(video, 10000, 1000)
    assert any("6 AM - 10 PM" in rec for rec in recommendations)