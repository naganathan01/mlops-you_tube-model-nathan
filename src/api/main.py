from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime
import json
import hashlib
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="YouTube Performance Predictor API",
    version="1.0.0",
    description="MLOps API for predicting YouTube video performance"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Video title")
    description: str = Field(default="", max_length=5000, description="Video description")
    channel_id: str = Field(..., min_length=1, description="YouTube channel ID")
    duration_seconds: int = Field(..., gt=0, le=43200, description="Video duration in seconds")
    publish_hour: int = Field(..., ge=0, le=23, description="Hour of publication (0-23)")
    publish_day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    tags: List[str] = Field(default=[], description="Video tags")
    category_id: str = Field(default="22", description="YouTube category ID")

    class Config:
        schema_extra = {
            "example": {
                "title": "Amazing AI Tutorial 🤖",
                "description": "Learn machine learning in this tutorial",
                "channel_id": "UC123456789",
                "duration_seconds": 600,
                "publish_hour": 14,
                "publish_day_of_week": 1,
                "tags": ["ai", "tutorial", "machine learning"],
                "category_id": "22"
            }
        }

class PredictionOutput(BaseModel):
    predicted_views: float = Field(..., description="Predicted view count")
    predicted_likes: float = Field(..., description="Predicted like count")
    predicted_comments: float = Field(..., description="Predicted comment count")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    model_version: str = Field(..., description="Model version used")
    prediction_id: str = Field(..., description="Unique prediction identifier")

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: int
    features_available: int
    version: str

# Global variables
models = {}
feature_columns = None
model_version = "1.0.0"

@app.on_event("startup")
async def load_models():
    """Load models and feature metadata at startup"""
    global models, feature_columns
    try:
        model_path = os.getenv("MODEL_REGISTRY_PATH", "./models")
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            logger.warning(f"Model directory {model_path} not found. Creating dummy models for testing.")
            _create_dummy_models()
            return
        
        # Load models
        model_files = {
            'views': 'xgboost_view_count.joblib',
            'likes': 'xgboost_like_count.joblib', 
            'comments': 'xgboost_comment_count.joblib'
        }
        
        models_loaded = 0
        for key, filename in model_files.items():
            file_path = os.path.join(model_path, filename)
            if os.path.exists(file_path):
                models[key] = joblib.load(file_path)
                models_loaded += 1
                logger.info(f"Loaded model: {filename}")
            else:
                logger.warning(f"Model file not found: {file_path}")
        
        # Load feature columns
        feature_file = os.path.join(model_path, "feature_columns.json")
        if os.path.exists(feature_file):
            with open(feature_file, "r") as f:
                feature_columns = json.load(f)
            logger.info(f"Loaded {len(feature_columns)} feature columns")
        else:
            logger.warning("Feature columns file not found, using default features")
            feature_columns = _get_default_features()
        
        if models_loaded == 0:
            logger.warning("No models loaded, creating dummy models for testing")
            _create_dummy_models()
        else:
            logger.info(f"Successfully loaded {models_loaded} models")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        _create_dummy_models()

def _create_dummy_models():
    """Create dummy models for testing when real models aren't available"""
    global models, feature_columns
    
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Create dummy feature columns
    feature_columns = _get_default_features()
    
    # Create dummy models
    for target in ['views', 'likes', 'comments']:
        dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
        # Fit with dummy data
        X_dummy = np.random.rand(100, len(feature_columns))
        y_dummy = np.random.rand(100) * 10  # Log scale
        dummy_model.fit(X_dummy, y_dummy)
        models[target] = dummy_model
    
    logger.info("Created dummy models for testing")

def _get_default_features():
    """Get default feature column names"""
    return [
        'title_length', 'title_word_count', 'has_emoji', 'has_hashtag',
        'has_question', 'has_exclamation', 'description_length', 
        'duration_seconds', 'is_shorts', 'publish_hour', 'publish_day_of_week',
        'tag_count', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'title_uppercase_ratio', 'is_weekend', 'is_prime_time'
    ]

@app.post("/predict", response_model=PredictionOutput)
async def predict_performance(video: VideoInput, background_tasks: BackgroundTasks):
    """Predict video performance"""
    try:
        # Create features
        features_df = create_features(video)
        
        # Make predictions
        pred_views = np.expm1(models['views'].predict(features_df)[0])
        pred_likes = np.expm1(models['likes'].predict(features_df)[0]) 
        pred_comments = np.expm1(models['comments'].predict(features_df)[0])
        
        # Calculate confidence (simplified)
        confidence = calculate_confidence(features_df, pred_views)
        
        # Generate recommendations
        recommendations = generate_recommendations(video, pred_views, pred_likes)
        
        # Create prediction ID
        prediction_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_generate_hash(video.dict())[:8]}"
        
        result = PredictionOutput(
            predicted_views=float(max(0, pred_views)),
            predicted_likes=float(max(0, pred_likes)),
            predicted_comments=float(max(0, pred_comments)),
            confidence_score=float(confidence),
            recommendations=recommendations,
            model_version=model_version,
            prediction_id=prediction_id
        )
        
        # Log prediction asynchronously
        background_tasks.add_task(log_prediction, video.dict(), result.dict())
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthResponse(
            status="healthy" if len(models) > 0 else "degraded",
            timestamp=datetime.now(),
            models_loaded=len(models),
            features_available=len(feature_columns) if feature_columns else 0,
            version=model_version
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    try:
        model_details = {}
        for name, model in models.items():
            model_details[name] = {
                "type": type(model).__name__,
                "features": len(feature_columns) if feature_columns else 0
            }
        
        return {
            "model_version": model_version,
            "models": model_details,
            "feature_count": len(feature_columns) if feature_columns else 0,
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def create_features(video: VideoInput) -> pd.DataFrame:
    """Create feature vector from input"""
    # Create base features
    features = {
        'title_length': len(video.title),
        'title_word_count': len(video.title.split()),
        'has_emoji': 1 if any(ord(char) > 127 for char in video.title) else 0,
        'has_hashtag': 1 if '#' in video.title else 0,
        'has_question': 1 if '?' in video.title else 0,
        'has_exclamation': 1 if '!' in video.title else 0,
        'description_length': len(video.description),
        'duration_seconds': video.duration_seconds,
        'is_shorts': 1 if video.duration_seconds <= 60 else 0,
        'publish_hour': video.publish_hour,
        'publish_day_of_week': video.publish_day_of_week,
        'tag_count': len(video.tags),
        'hour_sin': np.sin(2 * np.pi * video.publish_hour / 24),
        'hour_cos': np.cos(2 * np.pi * video.publish_hour / 24),
        'day_sin': np.sin(2 * np.pi * video.publish_day_of_week / 7),
        'day_cos': np.cos(2 * np.pi * video.publish_day_of_week / 7),
        'title_uppercase_ratio': sum(1 for c in video.title if c.isupper()) / len(video.title) if video.title else 0,
        'is_weekend': 1 if video.publish_day_of_week >= 5 else 0,
        'is_prime_time': 1 if 18 <= video.publish_hour <= 22 else 0
    }
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Ensure all required columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the columns used during training
    df = df[feature_columns]
    
    return df

def calculate_confidence(features_df: pd.DataFrame, prediction: float) -> float:
    """Calculate prediction confidence"""
    base_confidence = 0.7
    
    # Adjust based on prediction magnitude
    if 1000 <= prediction <= 100000:
        confidence_adj = 0.2
    elif 100 <= prediction <= 1000000:
        confidence_adj = 0.1
    else:
        confidence_adj = -0.1
    
    return min(0.95, max(0.5, base_confidence + confidence_adj))

def generate_recommendations(video: VideoInput, pred_views: float, pred_likes: float) -> List[str]:
    """Generate optimization recommendations"""
    recommendations = []
    
    # Title recommendations
    if len(video.title) < 30:
        recommendations.append("Consider making your title longer (30-60 characters) for better SEO")
    elif len(video.title) > 100:
        recommendations.append("Consider shortening your title for better readability")
    
    if not any(ord(char) > 127 for char in video.title):
        recommendations.append("Adding emojis to your title can increase click-through rates")
    
    if '#' not in video.title and len(video.tags) < 3:
        recommendations.append("Add relevant hashtags to improve discoverability")
    
    # Duration recommendations
    if video.duration_seconds > 600 and pred_views < 10000:
        recommendations.append("Consider creating shorter content (under 10 minutes) for better retention")
    elif video.duration_seconds < 60:
        recommendations.append("Shorts format detected - great for viral potential!")
    
    # Timing recommendations
    if video.publish_hour < 6 or video.publish_hour > 22:
        recommendations.append("Consider posting between 6 AM - 10 PM for better initial engagement")
    
    # Description recommendations
    if len(video.description) < 100:
        recommendations.append("Add a more detailed description to improve SEO and provide context")
    
    if not recommendations:
        recommendations.append("Your video settings look optimized for good performance!")
    
    return recommendations

def _generate_hash(data: dict) -> str:
    """Generate hash for caching"""
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

async def log_prediction(input_data: dict, prediction_data: dict):
    """Log prediction for monitoring"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "prediction": prediction_data,
        "model_version": model_version
    }
    logger.info(f"Prediction logged: {prediction_data['prediction_id']}")

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)