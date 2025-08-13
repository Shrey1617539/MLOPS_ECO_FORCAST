# src/models/evaluate.py
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logging import get_logger, log_function_call
from src.config import get_config
import os

logger = get_logger(__name__)

class ModelEvaluator:
    """Evaluates model performance using various metrics."""
    
    def __init__(self):
        self.DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
        self.MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "models"))
        self.predictions_dir = self.DATA_ROOT / "predictions"
        self.raw_data_dir = self.DATA_ROOT / "raw"
        self.evaluation_dir = self.MODEL_ROOT / "evaluation"
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
    
    @log_function_call()
    def evaluate_model(self, city_name, period_days=30):
        """
        Evaluate model performance for a specific city.
        
        Args:
            city_name: Name of the city
            period_days: Number of days to look back for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Load predictions and actual data
            pred_file = self.predictions_dir / f"{city_name}.csv"
            actual_file = self.raw_data_dir / f"{city_name}.csv"
            
            if not pred_file.exists() or not actual_file.exists():
                logger.error(f"Data files missing for {city_name}")
                return {"status": "error", "error": "Data files missing"}
            
            # Load data
            pred_df = pd.read_csv(pred_file)
            actual_df = pd.read_csv(actual_file)
            
            # Convert dates
            pred_df['date'] = pd.to_datetime(pred_df['date'])
            actual_df['date'] = pd.to_datetime(actual_df['date'])
            
            # Get target variable from config
            model_config = get_config('model')
            target = model_config.get('target', 'temperature')
            
            # Merge data
            merged_df = pd.merge(
                pred_df, 
                actual_df, 
                on='date',
                suffixes=('_pred', '')
            )
            
            # Filter for recent data
            from datetime import datetime, timedelta
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=period_days)
            
            mask = (merged_df['date'].dt.date >= start_date) & (merged_df['date'].dt.date < end_date)
            recent_df = merged_df[mask].copy()
            
            if len(recent_df) == 0:
                logger.warning(f"No recent data available for {city_name}")
                return {"status": "no_data"}
            
            # Calculate metrics
            y_true = recent_df[target]
            y_pred = recent_df[f"predicted_{target}"]
            
            metrics = {}
            
            # Mean Absolute Error
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            
            # Root Mean Squared Error
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            
            # Mean Absolute Percentage Error
            metrics["mape"] = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            
            # R-squared
            metrics["r2"] = float(r2_score(y_true, y_pred))
            
            # Relative Absolute Error
            metrics["rae"] = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true))))
            
            # Save evaluation results
            result = {
                "city": city_name,
                "timestamp": datetime.now().isoformat(),
                "period_days": period_days,
                "sample_size": len(recent_df),
                "metrics": metrics,
                "status": "success"
            }
            
            # Save to file
            self._save_evaluation_results(city_name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating model for {city_name}: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _save_evaluation_results(self, city_name, results):
        """Save evaluation results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d")
        eval_file = self.evaluation_dir / f"{city_name}_evaluation_{timestamp}.json"
        
        with open(eval_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_file}")
