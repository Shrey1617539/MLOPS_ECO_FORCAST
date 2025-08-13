import sys
import os
import glob
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.monitoring.drift_detection import DriftDetector
from src.models.retraining import ModelRetrainer
from src.models.evaluate import ModelEvaluator
from src.config import get_cities
from src.logging import get_logger

logger = get_logger(__name__)

def has_sufficient_prediction_data(min_days=30):
    """
    Checks if there is at least min_days worth of prediction data.
    """
    DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
    predictions_base_dir = DATA_ROOT / "predictions"
    # Find all files in the DATA_ROOT directory
    DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
    all_files_in_dataroot = list(DATA_ROOT.rglob('*'))  # Recursively find all files
    if not predictions_base_dir.exists():
        logger.warning(f"Predictions base directory not found: {predictions_base_dir}")
        return False

    all_prediction_dates = set()
    
    # Look directly for all CSV files in the predictions directory
    pattern = str(predictions_base_dir / "*.csv")
    
    for f_path in glob.glob(pattern):
        try:
            # Read the CSV file to extract dates from the 'date' column
            import pandas as pd
            df = pd.read_csv(f_path)
            
            if 'date' in df.columns:
                for date_str in df['date'].unique():
                    try:
                        # Parse date in YYYY-MM-DD format
                        pred_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        all_prediction_dates.add(pred_date)
                        logger.info(f"Found prediction date: {pred_date} in file {f_path}")
                    except ValueError:
                        logger.warning(f"Could not parse date {date_str} from file {f_path}")
            else:
                logger.warning(f"No 'date' column found in {f_path}")
        except Exception as e:
            logger.warning(f"Error processing file {f_path}: {str(e)}")
            continue
    
    if not all_prediction_dates:
        logger.warning("No valid prediction data files found.")
        return False
    
    date_span = max(all_prediction_dates) - min(all_prediction_dates)
    return date_span.days >= min_days

def run_drift_monitoring_pipeline():
    """Run the complete drift monitoring pipeline."""
    MODEL_ROOT = Path(os.getenv("MODEL_ROOT", "models"))
    
    try:
        logger.info("Starting drift monitoring pipeline")
        
        # 1. Check for sufficient prediction data
        logger.info("Checking for sufficient prediction data")
        has_data = has_sufficient_prediction_data(min_days=30)
        
        if not has_data:
            logger.warning("Insufficient prediction data (less than 1 month). Skipping drift detection.")
            pipeline_results = {
                "timestamp": datetime.now().isoformat(),
                "status": "skipped",
                "message": "Not enough prediction data (less than 1 month) to run drift detection."
            }
        else:
            # 2. Detect drift
            logger.info("Running drift detection")
            drift_detector = DriftDetector()
            drift_results = drift_detector.run_weekly_drift_detection()
            
            # 3. Retrain models if drift detected
            if drift_results.get("needs_retraining", False):
                logger.info("Data drift detected, proceeding with model retraining")


                retrainer = ModelRetrainer()
                retraining_results = retrainer.retrain_models_if_needed(drift_results)
                
                # 4. Evaluate retrained models
                logger.info("Evaluating retrained models")
                evaluator = ModelEvaluator()
                
                evaluation_results = {}
                for city, result in retraining_results.get("results", {}).items():
                    if result.get("status") == "retrained":
                        eval_result = evaluator.evaluate_model(city)
                        evaluation_results[city] = eval_result
                        logger.info(f"Evaluated retrained model for {city}")
                
                pipeline_results = {
                    "timestamp": datetime.now().isoformat(),
                    "drift_detected": True,
                    "models_retrained": list(retraining_results.get("results", {}).keys()),
                    "evaluation_results": evaluation_results
                }
            else:
                logger.info("No data drift detected, skipping retraining")
                pipeline_results = {
                    "timestamp": datetime.now().isoformat(),
                    "drift_detected": False,
                    "message": "No drift detected, models are up-to-date"
                }
        
        # 5. Save pipeline results
        logger.info("Saving pipeline results")
        reports_dir = MODEL_ROOT / "evaluation" / "pipeline_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"drift_pipeline_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        logger.info(f"Pipeline report saved to {report_file}")
        logger.info("Drift monitoring pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in drift monitoring pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_drift_monitoring_pipeline()
