import pandas as pd
import numpy as np
import joblib
import json
import os
import argparse
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str = "models/", data_path: str = "data/processed/features.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.models = {}
        self.test_data = None
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """Load all trained models"""
        model_files = {
            'views': 'xgboost_view_count.joblib',
            'likes': 'xgboost_like_count.joblib',
            'comments': 'xgboost_comment_count.joblib'
        }
        
        for target, filename in model_files.items():
            model_file = os.path.join(self.model_path, filename)
            if os.path.exists(model_file):
                self.models[target] = joblib.load(model_file)
                self.logger.info(f"Loaded model for {target}")
            else:
                self.logger.error(f"Model file not found: {model_file}")
                raise FileNotFoundError(f"Model file not found: {model_file}")
    
    def load_test_data(self):
        """Load and prepare test data"""
        df = pd.read_csv(self.data_path)
        
        # Same feature preparation as in training
        exclude_cols = [
            'video_id', 'title', 'description', 'published_at', 'collected_at',
            'tags', 'channel_id', 'channel_title', 'duration',
            'view_count', 'like_count', 'comment_count'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        feature_cols = [col for col in feature_cols if col not in categorical_cols]
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[['view_count', 'like_count', 'comment_count']].fillna(0)
        y_log = np.log1p(y.clip(lower=0))
        
        self.test_data = {'X': X, 'y': y, 'y_log': y_log}
        self.logger.info(f"Loaded test data: {X.shape[0]} samples, {X.shape[1]} features")
        
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """Evaluate all models and return metrics"""
        if not self.models or self.test_data is None:
            raise ValueError("Models and test data must be loaded first")
        
        results = {}
        
        for target in ['view_count', 'like_count', 'comment_count']:
            target_key = target.replace('_count', 's')
            if target_key in self.models:
                model = self.models[target_key]
                
                # Make predictions
                y_pred_log = model.predict(self.test_data['X'])
                y_pred = np.expm1(y_pred_log)
                y_true = self.test_data['y'][target]
                
                # Calculate metrics
                metrics = {
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred),
                    'mape': self._calculate_mape(y_true, y_pred)
                }
                
                results[target] = metrics
                self.logger.info(f"{target} - RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        return results
    
    def _calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report"""
        self.load_models()
        self.load_test_data()
        results = self.evaluate_models()
        
        # Save results
        os.makedirs("metrics", exist_ok=True)
        with open('metrics/evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self._create_evaluation_plots(results)
        
        # Generate text report
        report = self._generate_text_report(results)
        
        with open('metrics/evaluation_report.txt', 'w') as f:
            f.write(report)
        
        return "Evaluation completed. Results saved to metrics/"
    
    def _create_evaluation_plots(self, results: Dict[str, Dict[str, float]]):
        """Create evaluation plots"""
        # Model performance comparison
        metrics_df = pd.DataFrame(results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, metric in enumerate(['rmse', 'mae', 'r2', 'mape']):
            ax = axes[i//2, i%2]
            bars = metrics_df[metric].plot(kind='bar', ax=ax, color=['skyblue', 'lightgreen', 'lightcoral'])
            ax.set_title(f'{metric.upper()} by Target Variable', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('metrics/evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance plot (for the first model)
        if self.models:
            first_model = list(self.models.values())[0]
            if hasattr(first_model, 'feature_importances_'):
                self._plot_feature_importance(first_model)
    
    def _plot_feature_importance(self, model):
        """Plot feature importance"""
        feature_names = self.test_data['X'].columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Top 20 Feature Importances", fontsize=16, fontweight='bold')
        top_20_indices = indices[:20]
        plt.bar(range(20), importances[top_20_indices])
        plt.xticks(range(20), [feature_names[i] for i in top_20_indices], rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('metrics/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, results: Dict[str, Dict[str, float]]) -> str:
        """Generate text evaluation report"""
        report = []
        report.append("="*60)
        report.append("MODEL EVALUATION REPORT")
        report.append("="*60)
        report.append(f"Generated: {pd.Timestamp.now()}")
        report.append("")
        
        for target, metrics in results.items():
            report.append(f"{target.upper()}:")
            report.append("-" * 40)
            for metric, value in metrics.items():
                if metric == 'r2':
                    interpretation = "Excellent" if value > 0.9 else "Good" if value > 0.7 else "Fair" if value > 0.5 else "Poor"
                    report.append(f"  {metric.upper():>8}: {value:>8.3f} ({interpretation})")
                else:
                    report.append(f"  {metric.upper():>8}: {value:>8.3f}")
            report.append("")
        
        # Summary
        avg_r2 = np.mean([metrics['r2'] for metrics in results.values()])
        report.append("SUMMARY:")
        report.append("-" * 40)
        report.append(f"Average R² Score: {avg_r2:.3f}")
        
        if avg_r2 > 0.8:
            report.append("✅ Models show excellent performance")
        elif avg_r2 > 0.6:
            report.append("⚠️  Models show good performance with room for improvement")
        else:
            report.append("❌ Models need significant improvement")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Evaluate YouTube performance prediction models')
    parser.add_argument('--model-path', default='models/', help='Path to model directory')
    parser.add_argument('--data-path', default='data/processed/features.csv', help='Path to test data')
    
    args = parser.parse_args()
    
    try:
        evaluator = ModelEvaluator(args.model_path, args.data_path)
        report = evaluator.generate_evaluation_report()
        print(report)
        
        # Print results summary
        results = evaluator.evaluate_models()
        print("\n🎯 Evaluation Summary:")
        for target, metrics in results.items():
            print(f"{target}: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.0f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()