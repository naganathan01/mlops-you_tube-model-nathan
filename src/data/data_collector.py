import pandas as pd
import requests
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

class YouTubeDataCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.logger = logging.getLogger(__name__)
    
    def collect_trending_videos(self, region_code: str = "IN", max_results: int = 200) -> pd.DataFrame:
        """Collect trending videos data"""
        videos_data = []
        
        try:
            # Get trending videos
            request = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                chart="mostPopular",
                regionCode=region_code,
                maxResults=min(50, max_results)
            )
            
            next_page_token = None
            collected = 0
            
            while collected < max_results:
                if next_page_token:
                    request = self.youtube.videos().list(
                        part="snippet,statistics,contentDetails",
                        chart="mostPopular",
                        regionCode=region_code,
                        maxResults=min(50, max_results - collected),
                        pageToken=next_page_token
                    )
                
                response = request.execute()
                
                for item in response.get('items', []):
                    video_data = self._extract_video_features(item)
                    videos_data.append(video_data)
                    collected += 1
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            return pd.DataFrame(videos_data)
            
        except Exception as e:
            self.logger.error(f"Error collecting trending videos: {e}")
            return pd.DataFrame()
    
    def collect_channel_videos(self, channel_id: str, max_results: int = 100) -> pd.DataFrame:
        """Collect videos from specific channel"""
        # Implementation similar to previous channel analyzer
        pass
    
    def _extract_video_features(self, video_item: Dict) -> Dict:
        """Extract features from video API response"""
        snippet = video_item.get('snippet', {})
        statistics = video_item.get('statistics', {})
        content_details = video_item.get('contentDetails', {})
        
        return {
            'video_id': video_item.get('id'),
            'title': snippet.get('title'),
            'channel_id': snippet.get('channelId'),
            'channel_title': snippet.get('channelTitle'),
            'published_at': snippet.get('publishedAt'),
            'description': snippet.get('description', ''),
            'view_count': int(statistics.get('viewCount', 0)),
            'like_count': int(statistics.get('likeCount', 0)),
            'comment_count': int(statistics.get('commentCount', 0)),
            'duration': content_details.get('duration'),
            'category_id': snippet.get('categoryId'),
            'tags': snippet.get('tags', []),
            'default_language': snippet.get('defaultLanguage'),
            'collected_at': datetime.now().isoformat()
        }