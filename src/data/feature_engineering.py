import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder
import logging
import argparse
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeFeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline with robust error handling"""
        try:
            df = df.copy()
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Starting feature engineering for {len(df)} videos")
            
            # Data cleaning first
            df = self._clean_data(df)
            
            # Time-based features
            df = self._create_time_features(df)
            
            # Content features
            df = self._create_content_features(df)
            
            # Text features
            df = self._create_text_features(df)
            
            # Engagement features (if target columns exist)
            if all(col in df.columns for col in ['view_count', 'like_count', 'comment_count']):
                df = self._create_engagement_features(df)
            
            # Channel features
            df = self._create_channel_features(df)
            
            # Fill any remaining NaN values
            df = self._handle_missing_values(df)
            
            self.logger.info(f"Feature engineering completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate input data"""
        # Ensure required columns exist
        required_cols = ['title', 'published_at', 'duration']
        for col in required_cols:
            if col not in df.columns:
                df[col] = '' if col == 'title' else '2024-01-01T00:00:00Z' if col == 'published_at' else 'PT0S'
        
        # Clean numeric columns
        numeric_cols = ['view_count', 'like_count', 'comment_count']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                df[col] = df[col].clip(lower=0)  # Ensure non-negative
        
        # Clean text columns
        df['title'] = df['title'].fillna('').astype(str)
        df['description'] = df.get('description', '').fillna('').astype(str)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features with error handling"""
        try:
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            
            # If parsing fails, use current time
            df['published_at'] = df['published_at'].fillna(pd.Timestamp.now())
            
            # Extract time components
            df['publish_hour'] = df['published_at'].dt.hour
            df['publish_day_of_week'] = df['published_at'].dt.dayofweek
            df['publish_month'] = df['published_at'].dt.month
            df['publish_year'] = df['published_at'].dt.year
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['publish_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['publish_hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['publish_day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['publish_day_of_week'] / 7)
            
            # Weekend indicator
            df['is_weekend'] = (df['publish_day_of_week'] >= 5).astype(int)
            
            # Prime time indicator (6 PM - 10 PM)
            df['is_prime_time'] = ((df['publish_hour'] >= 18) & (df['publish_hour'] <= 22)).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"Error in time features: {e}")
            # Add default time features
            df['publish_hour'] = 12
            df['publish_day_of_week'] = 1
            df['hour_sin'] = df['hour_cos'] = df['day_sin'] = df['day_cos'] = 0
            df['is_weekend'] = df['is_prime_time'] = 0
            return df
    
    def _create_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create content-related features"""
        # Duration features
        df['duration_seconds'] = df['duration'].apply(self._parse_duration)
        
        # Duration categories
        df['is_shorts'] = (df['duration_seconds'] <= 60).astype(int)
        df['is_medium'] = ((df['duration_seconds'] > 60) & (df['duration_seconds'] <= 600)).astype(int)
        df['is_long'] = (df['duration_seconds'] > 600).astype(int)
        
        # Content type detection
        df['is_live'] = df['title'].str.contains('live|LIVE|streaming', na=False, case=False).astype(int)
        df['is_tutorial'] = df['title'].str.contains('tutorial|how to|guide', na=False, case=False).astype(int)
        df['is_review'] = df['title'].str.contains('review|unboxing', na=False, case=False).astype(int)
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features"""
        # Title features
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        
        # Emoji detection
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
        df['has_emoji'] = df['title'].str.contains(emoji_pattern, na=False).astype(int)
        df['emoji_count'] = df['title'].str.count(emoji_pattern)
        
        # Special characters
        df['has_hashtag'] = df['title'].str.contains('#', na=False).astype(int)
        df['has_question'] = df['title'].str.contains(r'\?', na=False).astype(int)
        df['has_exclamation'] = df['title'].str.contains('!', na=False).astype(int)
        df['has_numbers'] = df['title'].str.contains(r'\d', na=False).astype(int)
        
        # Title case analysis
        df['title_uppercase_ratio'] = df['title'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if str(x) else 0
        )
        
        # Description features
        df['description_length'] = df['description'].str.len()
        df['description_word_count'] = df['description'].str.split().str.len()
        
        # Tags features
        df['tag_count'] = df.get('tags', []).apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Clickbait indicators
        clickbait_words = ['amazing', 'shocking', 'unbelievable', 'secret', 'hack', 'trick']
        df['clickbait_score'] = df['title'].apply(
            lambda x: sum(1 for word in clickbait_words if word.lower() in str(x).lower())
        )
        
        return df
    
    def _create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engagement-based features"""
        # Basic engagement metrics
        df['engagement_rate'] = (df['like_count'] + df['comment_count']) / (df['view_count'] + 1) * 100
        df['like_rate'] = df['like_count'] / (df['view_count'] + 1) * 100
        df['comment_rate'] = df['comment_count'] / (df['view_count'] + 1) * 100
        
        # Engagement ratios
        df['likes_per_comment'] = df['like_count'] / (df['comment_count'] + 1)
        
        # Performance categories
        df['is_viral'] = (df['view_count'] > df['view_count'].quantile(0.9)).astype(int)
        df['is_high_engagement'] = (df['engagement_rate'] > df['engagement_rate'].quantile(0.8)).astype(int)
        
        return df
    
    def _create_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create channel-based features"""
        if 'channel_id' not in df.columns:
            df['channel_id'] = 'unknown'
        
        # Channel-level aggregations
        channel_stats = df.groupby('channel_id').agg({
            'view_count': ['mean', 'count'] if 'view_count' in df.columns else [lambda x: 1000, lambda x: 1],
            'like_count': 'mean' if 'like_count' in df.columns else lambda x: 100,
            'comment_count': 'mean' if 'comment_count' in df.columns else lambda x: 10
        }).reset_index()
        
        # Flatten column names
        if 'view_count' in df.columns:
            channel_stats.columns = ['channel_id', 'channel_avg_views', 'channel_video_count', 
                                   'channel_avg_likes', 'channel_avg_comments']
        else:
            channel_stats.columns = ['channel_id', 'channel_avg_views', 'channel_video_count', 
                                   'channel_avg_likes', 'channel_avg_comments']
            channel_stats['channel_avg_views'] = 1000
            channel_stats['channel_video_count'] = 1
            channel_stats['channel_avg_likes'] = 100
            channel_stats['channel_avg_comments'] = 10
        
        # Merge back
        df = df.merge(channel_stats, on='channel_id', how='left')
        
        # Fill missing values for new channels
        df['channel_avg_views'] = df['channel_avg_views'].fillna(1000)
        df['channel_video_count'] = df['channel_video_count'].fillna(1)
        df['channel_avg_likes'] = df['channel_avg_likes'].fillna(100)
        df['channel_avg_comments'] = df['channel_avg_comments'].fillna(10)
        
        # Channel performance indicators
        if 'view_count' in df.columns:
            df['views_vs_channel_avg'] = df['view_count'] / (df['channel_avg_views'] + 1)
        else:
            df['views_vs_channel_avg'] = 1.0
        
        df['is_channel_popular'] = (df['channel_video_count'] > 10).astype(int)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle any remaining missing values"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds with error handling"""
        try:
            if not duration_str or duration_str == 'PT0S':
                return 0
            
            duration_str = str(duration_str).replace('PT', '')
            
            hours = 0
            minutes = 0
            seconds = 0
            
            if 'H' in duration_str:
                match = re.search(r'(\d+)H', duration_str)
                hours = int(match.group(1)) if match else 0
            if 'M' in duration_str:
                match = re.search(r'(\d+)M', duration_str)
                minutes = int(match.group(1)) if match else 0
            if 'S' in duration_str:
                match = re.search(r'(\d+)S', duration_str)
                seconds = int(match.group(1)) if match else 0
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return max(0, min(total_seconds, 43200))  # Cap at 12 hours
            
        except Exception as e:
            logger.warning(f"Error parsing duration {duration_str}: {e}")
            return 300  # Default to 5 minutes

def main():
    """Main function for feature engineering"""
    parser = argparse.ArgumentParser(description='Engineer features for YouTube data')
    parser.add_argument('--input', default='data/raw/youtube_trending.csv', help='Input CSV file')
    parser.add_argument('--output', default='data/processed/features.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    try:
        # Load data
        print(f"📥 Loading data from {args.input}")
        df = pd.read_csv(args.input)
        print(f"📊 Input data shape: {df.shape}")
        
        # Engineer features
        engineer = YouTubeFeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save processed data
        df_features.to_csv(args.output, index=False)
        print(f"✅ Saved processed features to {args.output}")
        print(f"📊 Output data shape: {df_features.shape}")
        
        # Save feature column names for model training
        feature_cols = [col for col in df_features.columns 
                       if col not in ['video_id', 'title', 'description', 'published_at', 
                                    'collected_at', 'tags', 'channel_id', 'channel_title', 
                                    'duration', 'view_count', 'like_count', 'comment_count']]
        
        feature_cols_path = 'data/processed/feature_columns.json'
        os.makedirs(os.path.dirname(feature_cols_path), exist_ok=True)
        with open(feature_cols_path, 'w') as f:
            json.dump(feature_cols, f, indent=2)
        
        print(f"📋 Saved {len(feature_cols)} feature columns to {feature_cols_path}")
        print("✨ Feature engineering completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        raise

if __name__ == "__main__":
    main()