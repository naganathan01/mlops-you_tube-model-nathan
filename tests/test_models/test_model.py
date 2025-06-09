import pytest
import pandas as pd
import numpy as np
from src.models.train import YouTubePerformancePredictor
from src.data.feature_engineering import YouTubeFeatureEngineer

class TestYouTubePredictor:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        data = {
            'video_id': ['test1', 'test2', 'test3'],
            'title': ['Test Video 1', 'Test Video 2 🔥', 'Test Video 3 #trending'],
            'channel_id': ['UC123', 'UC456', 'UC789'],
            'channel_title': ['Channel 1', 'Channel 2', 'Channel 3'],
            'published_at': ['2024-01-01T10:00:00Z', '2024-01-02T15:00:00Z', '2024-01-03T20:00:00Z'],
            'description': ['Description 1', 'Description 2', 'Description 3'],
            'view_count': [1000, 5000, 2000],
            'like_count': [100, 500, 200],
            'comment_count': [10, 50, 20],
            'duration': ['PT5M30S', 'PT10M15S', 'PT45S'],
            'category_id': ['22', '22', '22'],
            'tags': [['tag1'], ['tag2', 'tag3'], ['tag4']],
            'default_language': ['en', 'en', 'en']
        }
        return pd.DataFrame(data)
    
    def test_feature_engineering(self, sample_data):
        """Test feature engineering"""
        engineer = YouTubeFeatureEngineer()
        df_features = engineer.engineer_features(sample_data)
        
        # Check that new features are created
        assert 'title_length' in df_features.columns
        assert 'has_emoji' in df_features.columns
        assert 'duration_seconds' in df_features.columns
        assert 'engagement_rate' in df_features.columns
        
        # Check data types
        assert df_features['title_length'].dtype in [np.int64, np.int32]
        assert df_features['has_emoji'].dtype == bool
        assert df_features['duration_seconds'].dtype in [np.int64, np.int32]
    
    def test_model_training(self, sample_data):
        """Test model training"""
        engineer = YouTubeFeatureEngineer()
        df_features = engineer.engineer_features(sample_data)
        
        predictor = YouTubePerformancePredictor(experiment_name="test_experiment")
        X, y = predictor.prepare_data(df_features)
        
        # Check data preparation
        assert not X.empty
        assert not y.empty
        assert len(X) == len(y)
        
        # Check that targets are log-transformed
        assert all(y['view_count'] >= 0)
    
    def test_model_prediction_format(self):
        """Test prediction output format"""
        # Create mock model predictions
        predictions = {
            'predicted_views': 1000.0,
            'predicted_likes': 100.0,
            'predicted_comments': 10.0,
            'confidence_score': 0.85
        }
        
        # Validate prediction format
        assert isinstance(predictions['predicted_views'], float)
        assert predictions['predicted_views'] >= 0
        assert 0 <= predictions['confidence_score'] <= 1

class TestDataValidation:
    
    def test_data_quality_checks(self, sample_data):
        """Test data quality validation"""
        # Check for required columns
        required_columns = ['video_id', 'title', 'view_count', 'like_count', 'comment_count']
        for col in required_columns:
            assert col in sample_data.columns
        
        # Check data types
        assert sample_data['view_count'].dtype in [np.int64, np.int32]
        assert sample_data['like_count'].dtype in [np.int64, np.int32]
        
        # Check for negative values
        assert all(sample_data['view_count'] >= 0)
        assert all(sample_data['like_count'] >= 0)
        assert all(sample_data['comment_count'] >= 0)
    
    def test_feature_consistency(self, sample_data):
        """Test feature consistency across data processing"""
        engineer = YouTubeFeatureEngineer()
        df1 = engineer.engineer_features(sample_data.copy())
        df2 = engineer.engineer_features(sample_data.copy())
        
        # Features should be consistent across runs
        pd.testing.assert_frame_equal(df1, df2)