import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import joblib
from typing import Dict, Tuple, Any
import logging

class YouTubePerformancePredictor:
    def __init__(self, experiment_name: str = "youtube_performance_prediction"):
        self.models = {}
        self.feature_columns = None
        self.target_columns = ['view_count', 'like_count', 'comment_count']
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        
        mlflow.set_experiment(experiment_name)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare features and targets for training"""
        # Select feature columns (exclude target and identifier columns)
        exclude_cols = ['video_id', 'title', 'description', 'published_at', 'collected_at', 
                       'tags', 'channel_id', 'channel_title'] + self.target_columns
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df[feature_cols].copy()
        y = df[self.target_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # Log transform targets to handle skewed distributions
        y_log = np.log1p(y)
        
        return X, y_log
    
    def train_models(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple models and track with MLflow"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define models
        models_config = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for target_col in self.target_columns:
            self.logger.info(f"Training models for target: {target_col}")
            target_results = {}
            
            for model_name, model in models_config.items():
                with mlflow.start_run(run_name=f"{model_name}_{target_col}"):
                    # Train model
                    model.fit(X_train, y_train[target_col])
                    
                    # Predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)
                    
                    # Metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train[target_col], y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test[target_col], y_pred_test))
                    train_r2 = r2_score(y_train[target_col], y_pred_train)
                    test_r2 = r2_score(y_test[target_col], y_pred_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train[target_col], 
                                              cv=5, scoring='neg_root_mean_squared_error')
                    cv_rmse = -cv_scores.mean()
                    
                    # Log metrics
                    mlflow.log_metric("train_rmse", train_rmse)
                    mlflow.log_metric("test_rmse", test_rmse)
                    mlflow.log_metric("train_r2", train_r2)
                    mlflow.log_metric("test_r2", test_r2)
                    mlflow.log_metric("cv_rmse", cv_rmse)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, f"model_{model_name}_{target_col}")
                    
                    # Store results
                    target_results[model_name] = {
                        'model': model,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'cv_rmse': cv_rmse
                    }
            
            results[target_col] = target_results
        
        self.models = results
        return results
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Perform hyperparameter tuning for best models"""
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # XGBoost hyperparameter tuning
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        with mlflow.start_run(run_name=f"xgb_tuning_{target_col}"):
            xgb_model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                xgb_model, xgb_params, cv=3, scoring='neg_root_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train[target_col])
            
            # Log best parameters
            mlflow.log_params(grid_search.best_params_)
            
            # Evaluate best model
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test[target_col], y_pred))
            test_r2 = r2_score(y_test[target_col], y_pred)
            
            mlflow.log_metric("best_test_rmse", test_rmse)
            mlflow.log_metric("best_test_r2", test_r2)
            mlflow.sklearn.log_model(best_model, f"best_xgb_{target_col}")
        
        return grid_search.best_estimator_
    
    def save_models(self, output_dir: str = "models/"):
        """Save trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for target_col, target_models in self.models.items():
            for model_name, model_info in target_models.items():
                model_path = f"{output_dir}/{model_name}_{target_col}.joblib"
                joblib.dump(model_info['model'], model_path)
                self.logger.info(f"Saved model: {model_path}")