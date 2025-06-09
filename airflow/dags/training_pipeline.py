from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
import sys
import os

# Add project path
sys.path.append('/opt/airflow/dags/src')

from data.data_collector import YouTubeDataCollector
from data.feature_engineering import YouTubeFeatureEngineer
from models.train import YouTubePerformancePredictor

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'youtube_ml_training_pipeline',
    default_args=default_args,
    description='YouTube ML Model Training Pipeline',
    schedule_interval='@weekly',
    catchup=False
)

def collect_data(**context):
    """Collect fresh data"""
    api_key = os.getenv('YOUTUBE_API_KEY')
    collector = YouTubeDataCollector(api_key)
    
    # Collect trending videos
    df = collector.collect_trending_videos(max_results=1000)
    
    # Save raw data
    data_path = f"/opt/airflow/data/raw/trending_videos_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(data_path, index=False)
    
    return data_path

def engineer_features(**context):
    """Engineer features"""
    data_path = context['task_instance'].xcom_pull(task_ids='collect_data')
    
    import pandas as pd
    df = pd.read_csv(data_path)
    
    engineer = YouTubeFeatureEngineer()
    df_engineered = engineer.engineer_features(df)
    
    # Save processed data
    processed_path = f"/opt/airflow/data/processed/features_{datetime.now().strftime('%Y%m%d')}.csv"
    df_engineered.to_csv(processed_path, index=False)
    
    return processed_path

def train_models(**context):
    """Train ML models"""
    data_path = context['task_instance'].xcom_pull(task_ids='engineer_features')
    
    import pandas as pd
    df = pd.read_csv(data_path)
    
    predictor = YouTubePerformancePredictor()
    X, y = predictor.prepare_data(df)
    results = predictor.train_models(X, y)
    
    # Save models
    predictor.save_models("/opt/airflow/models/")
    
    return "Training completed"

def validate_models(**context):
    """Validate trained models"""
    # Load test data and validate models
    # Compare with previous model performance
    # Decide whether to promote to production
    pass

# Define tasks
collect_data_task = PythonOperator(
    task_id='collect_data',
    python_callable=collect_data,
    dag=dag
)

engineer_features_task = PythonOperator(
    task_id='engineer_features',
    python_callable=engineer_features,
    dag=dag
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag
)

validate_models_task = PythonOperator(
    task_id='validate_models',
    python_callable=validate_models,
    dag=dag
)

# Set dependencies
collect_data_task >> engineer_features_task >> train_models_task >> validate_models_task