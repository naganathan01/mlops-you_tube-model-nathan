import os
from typing import Optional
from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    # YouTube API
    youtube_api_key: Optional[str] = None
    
    # Database
    database_url: str = "postgresql://postgres:password@localhost:5432/youtube_mlops"
    redis_url: str = "redis://localhost:6379/0"
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "youtube_performance_prediction"
    
    # Model Configuration
    model_registry_path: str = "./models"
    feature_store_path: str = "./features"
    
    # Monitoring
    prometheus_port: int = 9090
    grafana_port: int = 3000
    
    # Production
    environment: str = "development"
    debug: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings