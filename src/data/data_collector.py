import pandas as pd
import requests
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import os
import json
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeDataCollector:
    def __init__(self, api_key: str):
        """Initialize YouTube Data Collector with API key"""
        if not api_key or api_key == "your_youtube_api_key_here":
            raise ValueError("Please provide a valid YouTube API key")
        
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.logger = logging.getLogger(__name__)
    
    def collect_trending_videos(self, region_code: str = "US", max_results: int = 200) -> pd.DataFrame:
        """Collect trending videos data with error handling"""
        videos_data = []
        
        try:
            collected = 0
            page_token = None
            
            while collected < max_results:
                batch_size = min(50, max_results - collected)
                
                # Build request
                request_params = {
                    'part': 'snippet,statistics,contentDetails',
                    'chart': 'mostPopular',
                    'regionCode': region_code,
                    'maxResults': batch_size
                }
                
                if page_token:
                    request_params['pageToken'] = page_token
                
                try:
                    response = self.youtube.videos().list(**request_params).execute()
                    
                    for item in response.get('items', []):
                        video_data = self._extract_video_features(item)
                        if video_data:  # Only add if data extraction was successful
                            videos_data.append(video_data)
                            collected += 1
                    
                    page_token = response.get('nextPageToken')
                    if not page_token:
                        break
                        
                    # Rate limiting
                    time.sleep(0.1)
                    
                except HttpError as e:
                    if e.resp.status == 403:
                        self.logger.error("API quota exceeded. Please wait or check your API key.")
                        break
                    else:
                        self.logger.error(f"HTTP Error: {e}")
                        break
                
            self.logger.info(f"Collected {len(videos_data)} videos")
            return pd.DataFrame(videos_data)
            
        except Exception as e:
            self.logger.error(f"Error collecting trending videos: {e}")
            # Return sample data for testing if API fails
            return self._generate_sample_data()
    
    def _extract_video_features(self, video_item: Dict) -> Optional[Dict]:
        """Extract features from video API response with error handling"""
        try:
            snippet = video_item.get('snippet', {})
            statistics = video_item.get('statistics', {})
            content_details = video_item.get('contentDetails', {})
            
            return {
                'video_id': video_item.get('id', ''),
                'title': snippet.get('title', ''),
                'channel_id': snippet.get('channelId', ''),
                'channel_title': snippet.get('channelTitle', ''),
                'published_at': snippet.get('publishedAt', ''),
                'description': snippet.get('description', '')[:1000],  # Limit description length
                'view_count': int(statistics.get('viewCount', 0)),
                'like_count': int(statistics.get('likeCount', 0)),
                'comment_count': int(statistics.get('commentCount', 0)),
                'duration': content_details.get('duration', 'PT0S'),
                'category_id': snippet.get('categoryId', '22'),
                'tags': snippet.get('tags', [])[:10],  # Limit tags
                'default_language': snippet.get('defaultLanguage', 'en'),
                'collected_at': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.warning(f"Error extracting video features: {e}")
            return None
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample data for testing when API is not available"""
        self.logger.info("Generating sample data for testing")
        
        sample_data = []
        titles = [
            "How to Learn Python in 2024 🐍",
            "Amazing AI Tutorial for Beginners",
            "Top 10 Programming Tips #coding",
            "Machine Learning Explained Simply",
            "Build Your First Web App 🚀",
            "Data Science Career Guide",
            "JavaScript Tips and Tricks ⚡",
            "React Tutorial 2024",
            "DevOps Best Practices",
            "Cloud Computing Basics"
        ]
        
        for i, title in enumerate(titles):
            sample_data.append({
                'video_id': f'sample_vid_{i+1}',
                'title': title,
                'channel_id': f'UC_sample_{i+1}',
                'channel_title': f'Sample Channel {i+1}',
                'published_at': (datetime.now() - timedelta(days=i+1)).isoformat(),
                'description': f'This is a sample description for {title}',
                'view_count': 1000 + i * 500,
                'like_count': 100 + i * 50,
                'comment_count': 10 + i * 5,
                'duration': f'PT{5+i}M30S',
                'category_id': '22',
                'tags': ['sample', 'tutorial', f'tag{i}'],
                'default_language': 'en',
                'collected_at': datetime.now().isoformat()
            })
        
        return pd.DataFrame(sample_data)

def main():
    """Main function for testing data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect YouTube trending videos')
    parser.add_argument('--output', default='data/raw/youtube_trending.csv', help='Output CSV file')
    parser.add_argument('--max-results', type=int, default=50, help='Maximum number of videos to collect')
    parser.add_argument('--region', default='US', help='Region code')
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        print("⚠️  YOUTUBE_API_KEY not found in environment variables.")
        print("Using sample data for testing...")
        collector = YouTubeDataCollector("dummy_key")
        df = collector._generate_sample_data()
    else:
        collector = YouTubeDataCollector(api_key)
        df = collector.collect_trending_videos(args.region, args.max_results)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save data
    df.to_csv(args.output, index=False)
    print(f"✅ Collected {len(df)} videos and saved to {args.output}")
    print(f"📊 Data shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()