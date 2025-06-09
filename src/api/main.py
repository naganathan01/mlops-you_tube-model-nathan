from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict
import logging
from datetime import datetime

app = FastAPI(title="YouTube Performance Predictor API", version="1.0.0")

class VideoInput(BaseModel):
    title: str
    description: str
    channel_id: str
    duration_seconds: int
    publish_hour: int
    publish_day_of_week: int
    tags: List[str] = []
    category_id: str = "22"

class PredictionOutput(BaseModel):
    predicted_views: float
    predicted_likes: float
    predicted_comments: float
    confidence_score: float
    recommendations: List[str]

# Load models at startup
models = {}
feature_columns = None

@app.on_event("startup")
async def load_models():
    global models, feature_columns
    try:
        # Load trained models
        models['views'] = joblib.load("models/xgboost_view_count.joblib")
        models['likes'] = joblib.load("models/xgboost_like_count.joblib")
        models['comments'] = joblib.load("models/xgboost_comment_count.joblib")
        
        # Load feature columns
        with open("models/feature_columns.txt", "r") as f:
            feature_columns = [line.strip() for line in f.readlines()]
            
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Error loading models: {e}")

@app.post("/predict", response_model=PredictionOutput)
async def predict_performance(video: VideoInput):
    """Predict video performance"""
    try:
        # Create feature dataframe
        features_df = create_features(video)
        
        # Make predictions
        pred_views = np.expm1(models['views'].predict(features_df)[0])
        pred_likes = np.expm1(models['likes'].predict(features_df)[0])
        pred_comments = np.expm1(models['comments'].predict(features_df)[0])
        
        # Calculate confidence (simplified)
        confidence = min(0.95, max(0.5, 1.0 - abs(pred_views - 10000) / 100000))
        
        # Generate recommendations
        recommendations = generate_recommendations(video, pred_views, pred_likes)
        
        return PredictionOutput(
            predicted_views=float(pred_views),
            predicted_likes=float(pred_likes),
            predicted_comments=float(pred_comments),
            confidence_score=float(confidence),
            recommendations=recommendations
        )
        
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
    
    # Duration category
    if video.duration_seconds <= 60:
        features['duration_category'] = 'Short'
    elif video.duration_seconds <= 300:
        features['duration_category'] = 'Medium'
    elif video.duration_seconds <= 600:
        features['duration_category'] = 'Long'
    else:
        features['duration_category'] = 'Very_Long'
    
    # One-hot encode categorical features
    duration_categories = ['Short', 'Medium', 'Long', 'Very_Long']
    for cat in duration_categories:
        features[f'duration_category_{cat}'] = 1 if features['duration_category'] == cat else 0
    
    del features['duration_category']
    
    # Create DataFrame and ensure all required columns are present
    df = pd.DataFrame([features])
    
    # Add missing columns with default values
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the columns used during training
    df = df[feature_columns]
    
    return df

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "models_loaded": len(models),
        "feature_count": len(feature_columns) if feature_columns else 0,
        "model_types": list(models.keys()) if models else []
    }