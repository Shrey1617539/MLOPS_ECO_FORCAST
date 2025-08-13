import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
import sys

from src.logging import get_logger, log_function_call
from src.config import get_cities

logger = get_logger(__name__)

class WeatherDataLoader:
    """Loads and prepares weather data for prediction."""
    
    def __init__(self):
        self.DATA_ROOT = os.getenv("DATA_ROOT", "data")
        self.processed_data_dir = Path(self.DATA_ROOT) / "processed"
    
    @log_function_call()
    def load_previous_days_data(self, city_name, lookback_days=7):
        """
        Load the previous N days of data for a city for model input.
        
        Args:
            city_name: Name of the city
            lookback_days: Number of previous days to load
        
        Returns:
            DataFrame containing the previous days' data
        """
        normalized_city_name = city_name.lower().replace(' ', '_')
        
        # Path to the city's processed data file
        data_file = self.processed_data_dir / f"{normalized_city_name}.csv"
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Read the full dataset
        df = pd.read_csv(data_file)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date to ensure correct order
        df = df.sort_values('date')
        
        # Get today's date and calculate the date range we need
        today = datetime.now().date()
        start_date = today - timedelta(days=lookback_days)
        
        # Filter for the date range
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date < today)
        recent_data = df[mask].copy()
        
        if len(recent_data) < lookback_days:
            logger.warning(f"Only found {len(recent_data)} days of data, expected {lookback_days}")
        
        logger.info(f"Loaded {len(recent_data)} days of historical data for {city_name}")
        
        return recent_data
