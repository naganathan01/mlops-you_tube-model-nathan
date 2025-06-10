from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class VideoInputSchema(BaseModel):
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
                "description": "Learn the basics of machine learning in this comprehensive tutorial",
                "channel_id": "UC123456789",
                "duration_seconds": 600,
                "publish_hour": 14,
                "publish_day_of_week": 1,
                "tags": ["ai", "tutorial", "machine learning"],
                "category_id": "22"
            }
        }

class PredictionResponseSchema(BaseModel):
    predicted_views: float = Field(..., description="Predicted view count")
    predicted_likes: float = Field(..., description="Predicted like count")
    predicted_comments: float = Field(..., description="Predicted comment count")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    model_version: str = Field(..., description="Model version used")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    timestamp: datetime = Field(default_factory=datetime.now)

class HealthResponseSchema(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: int
    features_available: int
    redis_connected: bool
    version: str

class BatchPredictionSchema(BaseModel):
    videos: List[VideoInputSchema]
    
class BatchPredictionResponseSchema(BaseModel):
    predictions: List[PredictionResponseSchema]
    batch_id: str
    processed_count: int

class ErrorResponseSchema(BaseModel):
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)
    path: Optional[str] = None