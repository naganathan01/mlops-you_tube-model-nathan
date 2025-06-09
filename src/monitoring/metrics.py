import logging
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3

# Prometheus metrics
prediction_counter = Counter('model_predictions_total', 'Total number of predictions made')
prediction_latency = Histogram('model_prediction_duration_seconds', 'Time spent on predictions')
model_accuracy = Gauge('model_accuracy_score', 'Current model accuracy score')
data_drift_score = Gauge('data_drift_score', 'Data drift detection score')

class ModelMonitor:
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.setup_database()
        self.logger = logging.getLogger(__name__)
    
    def setup_database(self):
        """Setup monitoring database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_features TEXT,
                predicted_views FLOAT,
                predicted_likes FLOAT,
                predicted_comments FLOAT,
                actual_views FLOAT DEFAULT NULL,
                actual_likes FLOAT DEFAULT NULL,
                actual_comments FLOAT DEFAULT NULL,
                model_version TEXT,
                latency_ms FLOAT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_name TEXT,
                metric_name TEXT,
                metric_value FLOAT,
                data_period TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, input_data: dict, predictions: dict, latency: float, model_version: str = "v1.0"):
        """Log prediction for monitoring"""
        prediction_counter.inc()
        prediction_latency.observe(latency)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (input_features, predicted_views, predicted_likes, 
                                   predicted_comments, model_version, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(input_data),
            predictions.get('views', 0),
            predictions.get('likes', 0),
            predictions.get('comments', 0),
            model_version,
            latency * 1000
        ))
        
        conn.commit()
        conn.close()
    
    def update_actual_performance(self, prediction_id: int, actual_views: float, 
                                actual_likes: float, actual_comments: float):
        """Update with actual performance data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE predictions 
            SET actual_views = ?, actual_likes = ?, actual_comments = ?
            WHERE id = ?
        ''', (actual_views, actual_likes, actual_comments, prediction_id))
        
        conn.commit()
        conn.close()
    
    def calculate_model_accuracy(self, days_back: int = 7):
        """Calculate model accuracy for recent predictions"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT predicted_views, actual_views, predicted_likes, actual_likes,
                   predicted_comments, actual_comments
            FROM predictions 
            WHERE actual_views IS NOT NULL 
            AND timestamp >= datetime('now', '-{} days')
        '''.format(days_back)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return None
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        def mape(actual, predicted):
            return np.mean(np.abs((actual - predicted) / actual)) * 100
        
        views_mape = mape(df['actual_views'], df['predicted_views'])
        likes_mape = mape(df['actual_likes'], df['predicted_likes'])
        comments_mape = mape(df['actual_comments'], df['predicted_comments'])
        
        overall_accuracy = 100 - np.mean([views_mape, likes_mape, comments_mape])
        model_accuracy.set(overall_accuracy)
        
        return {
            'overall_accuracy': overall_accuracy,
            'views_mape': views_mape,
            'likes_mape': likes_mape,
            'comments_mape': comments_mape
        }
    
    def detect_data_drift(self, new_data: pd.DataFrame, reference_data: pd.DataFrame):
        """Detect data drift using statistical tests"""
        from scipy import stats
        
        drift_scores = []
        
        for column in new_data.select_dtypes(include=[np.number]).columns:
            if column in reference_data.columns:
                # Kolmogorov-Smirnov test
                ks_stat, p_value = stats.ks_2samp(reference_data[column].dropna(), 
                                                 new_data[column].dropna())
                drift_scores.append(ks_stat)
        
        overall_drift = np.mean(drift_scores) if drift_scores else 0
        data_drift_score.set(overall_drift)
        
        return overall_drift

def start_monitoring_server(port: int = 8000):
    """Start Prometheus metrics server"""
    start_http_server(port)
    logging.info(f"Monitoring server started on port {port}")