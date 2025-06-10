import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import json
import os
import argparse
from typing import Dict, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubePerformancePredictor:
    def __init__(self):
        self.models = {}
        self.feature_columns = None
        self.target_columns = ['view_count', 'like_count', 'comment_count']
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for model training with robust handling"""
        try:
            self.logger.info(f"Preparing data from {df.shape[0]} samples")
            
            # Identify feature columns (exclude metadata and target columns)
            exclude_cols = [
                'video_id', 'title', 'description', 'published_at', 'collected_at',
                'tags', 'channel_id', 'channel_title', 'duration', 'default_language'
            ] + self.target_columns
            
            # Get feature columns
            available_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Only keep numeric columns for features
            numeric_cols = df[available_cols].select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = numeric_cols
            
            if not self.feature_columns:
                raise ValueError("No numeric feature columns found!")
            
            # Prepare feature matrix
            X = df[self.feature_columns].copy()
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Check for infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Prepare targets
            y_data = {}
            for target in self.target_columns:
                if target in df.columns:
                    target_values = pd.to_numeric(df[target], errors='coerce').fillna(0)
                    target_values = target_values.clip(lower=0)  # Ensure non-negative
                    # Log transform for better model performance
                    y_data[target] = np.log1p(target_values)
                else:
                    # Create dummy target for testing
                    y_data[target] = np.log1p(pd.Series(np.random.randint(100, 10000, len(X))))
            
            y = pd.DataFrame(y_data)
            
            self.logger.info(f"Data prepared: {X.shape[1]} features, {len(self.target_columns)} targets")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {e}")
            raise
    
    def train_models(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """Train models for each target variable"""
        try:
            results = {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.logger.info(f"Training set: {X_train.shape[0]} samples")
            self.logger.info(f"Test set: {X_test.shape[0]} samples")
            
            for target in self.target_columns:
                self.logger.info(f"Training model for {target}...")
                
                # Train XGBoost model
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    objective='reg:squarederror'
                )
                
                model.fit(X_train, y_train[target])
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train[target], y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test[target], y_pred_test))
                train_r2 = r2_score(y_train[target], y_pred_train)
                test_r2 = r2_score(y_test[target], y_pred_test)
                
                metrics = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
                
                results[target] = {
                    'model': model,
                    'metrics': metrics
                }
                
                self.logger.info(f"{target} - Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f}")
            
            self.models = results
            return results
            
        except Exception as e:
            self.logger.error(f"Error in model training: {e}")
            raise
    
    def save_models(self, output_dir: str = "models/"):
        """Save trained models and metadata"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save individual models
            for target, model_info in self.models.items():
                model_path = os.path.join(output_dir, f"xgboost_{target}.joblib")
                joblib.dump(model_info['model'], model_path)
                self.logger.info(f"Saved model: {model_path}")
            
            # Save feature columns
            feature_columns_path = os.path.join(output_dir, "feature_columns.json")
            with open(feature_columns_path, "w") as f:
                json.dump(self.feature_columns, f, indent=2)
            
            # Save model metadata
            metadata = {
                'model_version': '1.0.0',
                'feature_count': len(self.feature_columns),
                'target_columns': self.target_columns,
                'training_date': datetime.now().isoformat(),
                'model_performance': {
                    target: info['metrics'] for target, info in self.models.items()
                }
            }
            
            metadata_path = os.path.join(output_dir, "model_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Save performance metrics for tracking
            os.makedirs("metrics", exist_ok=True)
            performance_path = "metrics/model_performance.json"
            with open(performance_path, "w") as f:
                json.dump({
                    target: {
                        'test_r2': info['metrics']['test_r2'],
                        'test_rmse': info['metrics']['test_rmse']
                    } for target, info in self.models.items()
                }, f, indent=2)
            
            self.logger.info(f"All models and metadata saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            raise

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train YouTube performance prediction models')
    parser.add_argument('--input', default='data/processed/features.csv', help='Input features file')
    parser.add_argument('--output', default='models/', help='Output directory for models')
    
    args = parser.parse_args()
    
    try:
        # Check if input file exists
        if not os.path.exists(args.input):
            print(f"❌ Input file {args.input} not found!")
            print("Please run data collection and feature engineering first:")
            print("  python src/data/data_collector.py")
            print("  python src/data/feature_engineering.py")
            return
        
        # Load data
        print(f"📥 Loading data from {args.input}")
        df = pd.read_csv(args.input)
        print(f"📊 Data shape: {df.shape}")
        
        # Initialize predictor
        predictor = YouTubePerformancePredictor()
        
        # Prepare data
        X, y = predictor.prepare_data(df)
        print(f"✨ Features: {X.shape[1]}, Samples: {X.shape[0]}")
        
        # Train models
        print("🚀 Starting model training...")
        results = predictor.train_models(X, y)
        
        # Save models
        predictor.save_models(args.output)
        
        # Print results summary
        print("\n🎯 Training Results:")
        print("-" * 50)
        for target, info in results.items():
            metrics = info['metrics']
            print(f"{target:15} | R²: {metrics['test_r2']:.3f} | RMSE: {metrics['test_rmse']:.3f}")
        
        print(f"\n✅ Training completed! Models saved to {args.output}")
        print(f"📁 Model files:")
        for target in predictor.target_columns:
            print(f"  - {args.output}/xgboost_{target}.joblib")
        print(f"  - {args.output}/feature_columns.json")
        print(f"  - {args.output}/model_metadata.json")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        logger.exception("Training error details:")
        raise

if __name__ == "__main__":
    main()