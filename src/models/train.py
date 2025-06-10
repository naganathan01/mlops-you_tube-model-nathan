import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import json
import os
import yaml
import argparse
from typing import Dict, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    experiment_name: str = "youtube_performance_prediction"
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    target_columns: list = None
    
    def __post_init__(self):
        if self.target_columns is None:
            self.target_columns = ['view_count', 'like_count', 'comment_count']

class EnhancedYouTubePredictor:
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.feature_columns = None
        self.scalers = {}
        self.logger = logging.getLogger(__name__)
        
        # Set MLflow experiment
        mlflow.set_experiment(self.config.experiment_name)
    
    def load_params(self, params_file: str = "params.yaml") -> Dict:
        """Load parameters from yaml file"""
        try:
            with open(params_file, 'r') as f:
                params = yaml.safe_load(f)
            return params
        except FileNotFoundError:
            self.logger.warning(f"Params file {params_file} not found, using defaults")
            return {}
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Enhanced data preparation with validation"""
        # Data validation
        self._validate_data(df)
        
        # Feature selection with better logic
        exclude_cols = [
            'video_id', 'title', 'description', 'published_at', 'collected_at',
            'tags', 'channel_id', 'channel_title', 'duration'
        ] + self.config.target_columns
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        feature_cols = [col for col in feature_cols if col not in categorical_cols]
        
        self.feature_columns = feature_cols
        
        X = df[feature_cols].copy()
        y = df[self.config.target_columns].copy()
        
        # Enhanced missing value handling
        X = self._handle_missing_values(X)
        y = y.fillna(y.median())
        
        # Log transform targets with handling for zeros
        y_log = np.log1p(y.clip(lower=0))
        
        self.logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y_log
    
    def _validate_data(self, df: pd.DataFrame):
        """Validate input data quality"""
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        required_cols = ['view_count', 'like_count', 'comment_count']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for negative values in target columns
        for col in required_cols:
            if (df[col] < 0).any():
                self.logger.warning(f"Negative values found in {col}, will be clipped to 0")
                df[col] = df[col].clip(lower=0)
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enhanced missing value handling"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col] = X[col].fillna(median_val)
                self.logger.info(f"Filled {X[col].isnull().sum()} missing values in {col} with median: {median_val}")
        
        return X
    
    def train_models_with_hyperparameter_tuning(self, X: pd.DataFrame, y: pd.DataFrame, params: Dict = None) -> Dict[str, Any]:
        """Train models with hyperparameter tuning"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        results = {}
        
        # Get hyperparameters from config
        if params and 'hyperparameters' in params:
            xgb_params = params['hyperparameters'].get('xgboost', {
                'n_estimators': [100, 200, 300],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            })
        else:
            xgb_params = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        
        for target_col in self.config.target_columns:
            self.logger.info(f"Training models for target: {target_col}")
            
            with mlflow.start_run(run_name=f"xgboost_tuned_{target_col}"):
                # Hyperparameter tuning
                xgb_model = xgb.XGBRegressor(random_state=self.config.random_state)
                grid_search = GridSearchCV(
                    xgb_model, xgb_params, cv=3, scoring='neg_root_mean_squared_error',
                    n_jobs=-1, verbose=1
                )
                
                self.logger.info(f"Starting hyperparameter tuning for {target_col}")
                grid_search.fit(X_train, y_train[target_col])
                best_model = grid_search.best_estimator_
                
                # Evaluation
                y_pred_train = best_model.predict(X_train)
                y_pred_test = best_model.predict(X_test)
                
                metrics = self._calculate_metrics(
                    y_train[target_col], y_pred_train,
                    y_test[target_col], y_pred_test
                )
                
                # Log everything to MLflow
                mlflow.log_params(grid_search.best_params_)
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.xgboost.log_model(best_model, f"xgboost_{target_col}")
                
                results[target_col] = {
                    'model': best_model,
                    'metrics': metrics,
                    'best_params': grid_search.best_params_
                }
                
                self.logger.info(f"Best params for {target_col}: {grid_search.best_params_}")
                self.logger.info(f"Test R² for {target_col}: {metrics['test_r2']:.3f}")
        
        self.models = results
        return results
    
    def _calculate_metrics(self, y_train_true, y_train_pred, y_test_true, y_test_pred) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        return {
            'train_rmse': np.sqrt(mean_squared_error(y_train_true, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test_true, y_test_pred)),
            'train_r2': r2_score(y_train_true, y_train_pred),
            'test_r2': r2_score(y_test_true, y_test_pred),
            'train_mae': mean_absolute_error(y_train_true, y_train_pred),
            'test_mae': mean_absolute_error(y_test_true, y_test_pred)
        }
    
    def save_models_and_metadata(self, output_dir: str = "models/"):
        """Save models with comprehensive metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for target_col, model_info in self.models.items():
            model_path = f"{output_dir}/xgboost_{target_col}.joblib"
            joblib.dump(model_info['model'], model_path)
            self.logger.info(f"Saved model: {model_path}")
        
        # Save feature columns
        with open(f"{output_dir}/feature_columns.json", "w") as f:
            json.dump(self.feature_columns, f)
        
        # Save model metadata
        metadata = {
            'model_version': '2.0.0',
            'feature_count': len(self.feature_columns),
            'target_columns': self.config.target_columns,
            'training_date': datetime.now().isoformat(),
            'model_performance': {
                target: info['metrics'] for target, info in self.models.items()
            },
            'best_parameters': {
                target: info['best_params'] for target, info in self.models.items()
            }
        }
        
        with open(f"{output_dir}/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save performance metrics for DVC
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/model_performance.json", "w") as f:
            json.dump({
                target: {
                    'test_r2': info['metrics']['test_r2'],
                    'test_rmse': info['metrics']['test_rmse']
                } for target, info in self.models.items()
            }, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Train YouTube performance prediction models')
    parser.add_argument('--input', default='data/processed/features.csv', help='Input features file')
    parser.add_argument('--output', default='models/', help='Output directory for models')
    parser.add_argument('--params', default='params.yaml', help='Parameters file')
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        config = ModelConfig()
        predictor = EnhancedYouTubePredictor(config)
        
        # Load parameters
        params = predictor.load_params(args.params)
        
        # Load and prepare data
        logger.info(f"Loading data from {args.input}")
        df = pd.read_csv(args.input)
        X, y = predictor.prepare_data(df)
        
        # Train models
        logger.info("Starting model training with hyperparameter tuning")
        results = predictor.train_models_with_hyperparameter_tuning(X, y, params)
        
        # Save everything
        predictor.save_models_and_metadata(args.output)
        
        logger.info("Training completed successfully!")
        
        # Print summary
        print("\n🎯 Training Summary:")
        for target, info in results.items():
            print(f"{target}:")
            print(f"  - Test R²: {info['metrics']['test_r2']:.3f}")
            print(f"  - Test RMSE: {info['metrics']['test_rmse']:.3f}")
            print(f"  - Best params: {info['best_params']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()