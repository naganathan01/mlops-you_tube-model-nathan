from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import os
from datetime import datetime
import redis
import json
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import asyncio
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

app = FastAPI(
    title="YouTube Performance Predictor API",
    version="2.0.0",
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

# Redis cache
try:
    redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Running without cache.")
    redis_client = None

class VideoInput(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="Video title")
    description: str = Field(default="", max_length=5000, description="Video description")
    channel_id: str = Field(..., min_length=1, description="YouTube channel ID")
    duration_seconds: int = Field(..., gt=0, le=43200, description="Video duration in seconds")
    publish_hour: int = Field(..., ge=0, le=23, description="Hour of publication (0-23)")
    publish_day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    tags: List[str] = Field(default=[], description="Video tags")
    category_id: str = Field(default="22", description="YouTube category ID")

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
    redis_connected: bool
    version: str

# Model storage
models = {}
feature_columns = None
model_version = "2.0.0"

@app.on_event("startup")
async def load_models():
    """Load models and feature metadata at startup"""
    global models, feature_columns
    try:
        model_path = os.getenv("MODEL_REGISTRY_PATH", "./models")
        
        # Load models
        models['views'] = joblib.load(f"{model_path}/xgboost_view_count.joblib")
        models['likes'] = joblib.load(f"{model_path}/xgboost_like_count.joblib")
        models['comments'] = joblib.load(f"{model_path}/xgboost_comment_count.joblib")
        
        # Load feature columns
        with open(f"{model_path}/feature_columns.json", "r") as f:
            feature_columns = json.load(f)
            
        logger.info(f"Successfully loaded {len(models)} models with {len(feature_columns)} features")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {e}")

@app.post("/predict", response_model=PredictionOutput)
async def predict_performance(
    video: VideoInput,
    background_tasks: BackgroundTasks
):
    """Predict video performance with caching and monitoring"""
    REQUEST_COUNT.inc()
    
    with PREDICTION_LATENCY.time():
        try:
            # Generate cache key
            cache_key = f"prediction:{_generate_hash(video.dict())}"
            
            # Check cache
            if redis_client:
                cached_result = redis_client.get(cache_key)
                if cached_result:
                    logger.info("Returning cached prediction")
                    return PredictionOutput.parse_raw(cached_result)
            
            # Create features
            features_df = create_features(video)
            
            # Make predictions
            pred_views = np.expm1(models['views'].predict(features_df)[0])
            pred_likes = np.expm1(models['likes'].predict(features_df)[0])
            pred_comments = np.expm1(models['comments'].predict(features_df)[0])
            
            # Calculate confidence
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
            
            # Cache result (expire in 1 hour)
            if redis_client:
                redis_client.setex(cache_key, 3600, result.json())
            
            # Log prediction asynchronously
            background_tasks.add_task(log_prediction, video.dict(), result.dict())
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check"""
    try:
        # Check model availability
        model_status = len(models) > 0
        
        # Check Redis connection
        redis_status = False
        if redis_client:
            try:
                redis_status = redis_client.ping()
            except:
                redis_status = False
        
        # Check feature columns
        feature_status = feature_columns is not None
        
        status = "healthy" if all([model_status, feature_status]) else "unhealthy"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            models_loaded=len(models),
            features_available=len(feature_columns) if feature_columns else 0,
            redis_connected=redis_status,
            version=model_version
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.get("/model-info")
async def model_info():
    """Get detailed model information"""
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
        'has_emoji': any(ord(char) > 127 for char in video.title),
        'has_hashtag': '#' in video.title,
        'has_question': '?' in video.title,
        'has_exclamation': '!' in video.title,
        'description_length': len(video.description),
        'description_word_count': len(video.description.split()),
        'duration_seconds': video.duration_seconds,
        'is_shorts': video.duration_seconds <= 60,
        'publish_hour': video.publish_hour,
        'publish_day_of_week': video.publish_day_of_week,
        'tag_count': len(video.tags),
        'hour_sin': np.sin(2 * np.pi * video.publish_hour / 24),
        'hour_cos': np.cos(2 * np.pi * video.publish_hour / 24),
        'day_sin': np.sin(2 * np.pi * video.publish_day_of_week / 7),
        'day_cos': np.cos(2 * np.pi * video.publish_day_of_week / 7),
        'title_uppercase_ratio': sum(1 for c in video.title if c.isupper()) / len(video.title) if video.title else 0
    }
    
    # Duration category features
    if video.duration_seconds <= 60:
        duration_cat = 'Short'
    elif video.duration_seconds <= 300:
        duration_cat = 'Medium'
    elif video.duration_seconds <= 600:
        duration_cat = 'Long'
    else:
        duration_cat = 'Very_Long'
    
    # One-hot encode duration categories
    for cat in ['Short', 'Medium', 'Long', 'Very_Long']:
        features[f'duration_category_{cat}'] = 1 if duration_cat == cat else 0
    
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
    """Log prediction for monitoring and retraining"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "prediction": prediction_data,
        "model_version": model_version
    }
    
    # In production, send this to a proper logging system
    logger.info(f"Prediction logged: {prediction_data['prediction_id']}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)