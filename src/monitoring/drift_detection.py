# src/monitoring/drift_detection.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import scipy.stats as stats
from src.logging import get_logger, log_function_call
from src.config import get_config
import os

logger = get_logger(__name__)

class DriftDetector:
    """Detects and analyzes drift between predictions and actual values."""
    
    def __init__(self):
        self.DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
        self.MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "models"))
        self.predictions_dir = self.DATA_ROOT / "predictions"
        
        self.raw_data_dir = self.DATA_ROOT / "raw"
        self.drift_metrics_dir = self.MODEL_ROOT / "drift_metrics"
        self.drift_metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure drift thresholds from config
        drift_config = get_config('pipeline', 'drift_detection')
        self.lookback_period = drift_config.get('lookback_period', 30)  # Default 30 days
        self.drift_threshold = drift_config.get('threshold', 0.1)  # Default threshold
    
    @log_function_call()
    def run_weekly_drift_detection(self):
        """Run drift detection for all cities over the past month."""
        # Get city data files
        cities = []
        for pred_file in self.predictions_dir.glob("*.csv"):
            city_name = pred_file.stem
            raw_file = self.raw_data_dir / f"{city_name}.csv"
            
            if raw_file.exists():
                cities.append(city_name)
            else:
                logger.warning(f"Raw data file not found for {city_name}")
        
        results = {}
        for city in cities:
            try:
                drift_result = self.detect_drift_for_city(city)
                results[city] = drift_result
            except Exception as e:
                logger.error(f"Error detecting drift for {city}: {str(e)}")
                results[city] = {"status": "error", "error": str(e)}
        
        # Save overall results
        self.save_drift_results(results)
        
        # Determine if retraining is needed
        needs_retraining = any(
            result.get("status") == "drift_detected" for result in results.values()
        )
        
        return {"results": results, "needs_retraining": needs_retraining}
    
    @log_function_call()
    def detect_drift_for_city(self, city_name):
        """Detect drift for a specific city by comparing predictions and actuals."""
        # Load predictions data
        pred_file = self.predictions_dir / f"{city_name}.csv"
        pred_df = pd.read_csv(pred_file)
        
        # Load actual data
        raw_file = self.raw_data_dir / f"{city_name}.csv"
        actual_df = pd.read_csv(raw_file)
        
        # Convert dates
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        actual_df['date'] = pd.to_datetime(actual_df['date'])
        
        # Filter for the lookback period
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.lookback_period)
        
        mask_pred = (pred_df['date'].dt.date >= start_date) & (pred_df['date'].dt.date < end_date)
        pred_recent = pred_df[mask_pred].copy()
        
        mask_actual = (actual_df['date'].dt.date >= start_date) & (actual_df['date'].dt.date < end_date)
        actual_recent = actual_df[mask_actual].copy()
        
        if len(pred_recent) == 0 or len(actual_recent) == 0:
            logger.warning(f"Insufficient data for {city_name} in the lookback period")
            return {"status": "insufficient_data"}
        
        # Merge predictions and actuals on date
        merged_df = pd.merge(
            pred_recent, 
            actual_recent, 
            on='date', 
            suffixes=('_pred', '_actual')
        )
        
        if len(merged_df) == 0:
            logger.warning(f"No matching dates between predictions and actuals for {city_name}")
            return {"status": "no_matching_dates"}
        
        # Perform statistical tests for drift detection
        drift_metrics = {}
        
        # 1. Get model target variable from config
        model_config = get_config('model')
        target = model_config.get('target', 'temperature')
        
        # 2. Calculate drift metrics
        pred_col = f"predicted_{target}"
        actual_col = target
        
        # KS test for distribution comparison
        ks_statistic, ks_pvalue = stats.ks_2samp(
            merged_df[pred_col], 
            merged_df[actual_col]
        )
        drift_metrics["ks_statistic"] = float(ks_statistic)
        drift_metrics["ks_pvalue"] = float(ks_pvalue)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(merged_df[pred_col] - merged_df[actual_col]))
        drift_metrics["mae"] = float(mae)
        
        # Mean Percentage Error
        mape = np.mean(np.abs((merged_df[actual_col] - merged_df[pred_col]) / merged_df[actual_col])) * 100
        drift_metrics["mape"] = float(mape)
        
        # Calculate Population Stability Index (PSI)
        # For PSI, we need to bin the data first
        bins = 10
        pred_hist, bin_edges = np.histogram(merged_df[pred_col], bins=bins)
        actual_hist, _ = np.histogram(merged_df[actual_col], bins=bin_edges)
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        pred_pct = pred_hist / (np.sum(pred_hist) + epsilon)
        actual_pct = actual_hist / (np.sum(actual_hist) + epsilon)
        
        # Calculate PSI
        psi = np.sum((pred_pct - actual_pct) * np.log((pred_pct + epsilon) / (actual_pct + epsilon)))
        drift_metrics["psi"] = float(psi)
        
        # Determine if drift is detected based on metrics
        # PSI > 0.25 indicates significant drift
        is_drift_detected = psi > 0.25 or ks_pvalue < 0.05
        
        result = {
            "status": "drift_detected" if is_drift_detected else "no_drift",
            "metrics": drift_metrics,
            "analyzed_dates": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "sample_size": len(merged_df)
        }
        
        return result
    
    def save_drift_results(self, results):
        """Save drift detection results to JSON file."""
        import json
        
        # Create timestamp for the report
        timestamp = datetime.now().strftime("%Y%m%d")
        result_file = self.drift_metrics_dir / f"drift_report_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "lookback_period_days": self.lookback_period,
                "results": results
            }, f, indent=2)
        
        logger.info(f"Drift detection results saved to {result_file}")
