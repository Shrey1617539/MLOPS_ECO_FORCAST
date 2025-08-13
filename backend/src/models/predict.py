import pandas as pd
from pathlib import Path
from datetime import datetime
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_latest_predictions(city_name):
    """
    Load the latest prediction for a city
    
    Args:
        city_name: Normalized city name (lowercase, underscores)
        
    Returns:
        Dictionary with prediction data or None if not found
    """
    try:
        predictions_path = Path(f"/app/data/predictions/{city_name}.csv")
        
        if not predictions_path.exists():
            logger.warning(f"No predictions file found for {city_name}")
            return None
        
        # Load predictions
        pred_df = pd.read_csv(predictions_path)
        
        # Convert date to datetime for sorting
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        
        # Sort by date and get the latest
        pred_df = pred_df.sort_values('date', ascending=False)
        
        if pred_df.empty:
            logger.warning(f"Predictions file for {city_name} is empty")
            return None
        
        # Get the latest prediction
        latest = pred_df.iloc[0].to_dict()
        
        # Convert date back to string for JSON serialization
        latest['date'] = latest['date'].strftime('%Y-%m-%d')
        
        return latest
    except Exception as e:
        logger.error(f"Error loading predictions for {city_name}: {str(e)}")
        return None
