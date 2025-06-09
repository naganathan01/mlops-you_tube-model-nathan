import pandas as pd
import numpy as np
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from textblob import TextBlob

class YouTubeFeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.scalers = {}
        self.tfidf_vectorizers = {}
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        df = df.copy()
        
        # Time-based features
        df = self._create_time_features(df)
        
        # Content features
        df = self._create_content_features(df)
        
        # Engagement features
        df = self._create_engagement_features(df)
        
        # Text features
        df = self._create_text_features(df)
        
        # Channel features
        df = self._create_channel_features(df)
        
        return df
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df['published_at'] = pd.to_datetime(df['published_at'])
        
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
        
        # Time since upload (if collected_at is available)
        if 'collected_at' in df.columns:
            df['collected_at'] = pd.to_datetime(df['collected_at'])
            df['hours_since_upload'] = (df['collected_at'] - df['published_at']).dt.total_seconds() / 3600
        
        return df
    
    def _create_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create content-related features"""
        # Duration features
        df['duration_seconds'] = df['duration'].apply(self._parse_duration)
        df['duration_category'] = pd.cut(df['duration_seconds'], 
                                       bins=[0, 60, 300, 600, 1800, float('inf')],
                                       labels=['Short', 'Medium', 'Long', 'Very_Long', 'Extra_Long'])
        
        # Content type detection
        df['is_shorts'] = (df['duration_seconds'] <= 60) & (df['duration_seconds'] > 0)
        df['is_live'] = df['title'].str.contains('live|LIVE', na=False)
        df['is_premiere'] = df['title'].str.contains('premiere|PREMIERE', na=False)
        
        return df
    
    def _create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engagement-based features"""
        # Basic engagement metrics
        df['engagement_rate'] = (df['like_count'] + df['comment_count']) / (df['view_count'] + 1)
        df['like_rate'] = df['like_count'] / (df['view_count'] + 1)
        df['comment_rate'] = df['comment_count'] / (df['view_count'] + 1)
        
        # Engagement ratios
        df['likes_per_comment'] = df['like_count'] / (df['comment_count'] + 1)
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create text-based features"""
        # Title features
        df['title_length'] = df['title'].str.len()
        df['title_word_count'] = df['title'].str.split().str.len()
        df['has_emoji'] = df['title'].str.contains(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]', na=False)
        df['has_hashtag'] = df['title'].str.contains('#', na=False)
        df['has_question'] = df['title'].str.contains(r'\?', na=False)
        df['has_exclamation'] = df['title'].str.contains('!', na=False)
        df['title_uppercase_ratio'] = df['title'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if x else 0)
        
        # Description features
        df['description_length'] = df['description'].str.len()
        df['description_word_count'] = df['description'].str.split().str.len()
        
        # Tags features
        df['tag_count'] = df['tags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        # Sentiment analysis
        df['title_sentiment'] = df['title'].apply(self._get_sentiment)
        
        return df
    
    def _create_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create channel-based features"""
        # Channel-level aggregations
        channel_stats = df.groupby('channel_id').agg({
            'view_count': ['mean', 'std', 'count'],
            'like_count': 'mean',
            'comment_count': 'mean',
            'engagement_rate': 'mean'
        }).reset_index()
        
        channel_stats.columns = ['channel_id', 'channel_avg_views', 'channel_std_views', 
                               'channel_video_count', 'channel_avg_likes', 'channel_avg_comments',
                               'channel_avg_engagement']
        
        df = df.merge(channel_stats, on='channel_id', how='left')
        
        # Channel performance relative to channel average
        df['views_vs_channel_avg'] = df['view_count'] / (df['channel_avg_views'] + 1)
        
        return df
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        if not duration_str:
            return 0
        
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
    
    def _get_sentiment(self, text: str) -> float:
        """Get sentiment polarity of text"""
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0