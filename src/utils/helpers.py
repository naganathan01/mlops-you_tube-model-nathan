import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import json
import yaml

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mlops_pipeline.log')
        ]
    )
    return logging.getLogger(__name__)

def create_directories(dirs: List[str]) -> None:
    """Create necessary directories"""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"Created/verified directory: {directory}")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate dataframe has required columns"""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    return True

def generate_hash(data: Dict[str, Any]) -> str:
    """Generate hash for data dictionary"""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(data_str.encode()).hexdigest()

def parse_duration_to_seconds(duration_str: str) -> int:
    """Parse ISO 8601 duration to seconds"""
    if not duration_str:
        return 0
    
    import re
    duration_str = duration_str.replace('PT', '')
    
    hours = 0
    minutes = 0
    seconds = 0
    
    if 'H' in duration_str:
        hours = int(re.search(r'(\d+)H', duration_str).group(1))
    if 'M' in duration_str:
        minutes = int(re.search(r'(\d+)M', duration_str).group(1))
    if 'S' in duration_str:
        seconds = int(re.search(r'(\d+)S', duration_str).group(1))
    
    return hours * 3600 + minutes * 60 + seconds

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load data from YAML file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], filepath: str) -> None:
    """Save data to YAML file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, indent=2)

class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_youtube_data(df: pd.DataFrame) -> List[str]:
        """Validate YouTube data quality"""
        issues = []
        
        # Check required columns
        required_cols = ['video_id', 'title', 'view_count', 'like_count', 'comment_count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for negative values
        numeric_cols = ['view_count', 'like_count', 'comment_count']
        for col in numeric_cols:
            if col in df.columns and (df[col] < 0).any():
                issues.append(f"Negative values found in {col}")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        high_missing = missing_data[missing_data > len(df) * 0.5]
        if not high_missing.empty:
            issues.append(f"High missing data in columns: {high_missing.index.tolist()}")
        
        return issues
    
    @staticmethod
    def validate_features(df: pd.DataFrame, feature_columns: List[str]) -> List[str]:
        """Validate feature data"""
        issues = []
        
        # Check if all feature columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            issues.append(f"Missing feature columns: {missing_features}")
        
        # Check for infinite values
        for col in feature_columns:
            if col in df.columns and np.isinf(df[col]).any():
                issues.append(f"Infinite values found in {col}")
        
        return issues

def format_number(num: float, precision: int = 2) -> str:
    """Format large numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def calculate_engagement_rate(likes: int, comments: int, views: int) -> float:
    """Calculate engagement rate"""
    if views == 0:
        return 0.0
    return (likes + comments) / views * 100

def is_viral_potential(views: int, duration: int, engagement_rate: float) -> bool:
    """Determine if a video has viral potential"""
    if duration <= 60:  # Shorts
        return views > 100000 and engagement_rate > 5
    else:  # Regular videos
        return views > 1000000 and engagement_rate > 3