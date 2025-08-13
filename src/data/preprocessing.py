# src/data/preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.logging import get_logger, log_function_call
from src.config import get_cities
import os

logger = get_logger(__name__)

class WeatherDataPreprocessor:
    """Preprocesses and validates raw weather data."""
    
    def __init__(self):
        self.cities = get_cities()
        self.DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
        self.raw_data_dir = self.DATA_ROOT / "raw"
        self.processed_data_dir = self.DATA_ROOT / "processed"
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    @log_function_call()
    def preprocess_all_cities(self, date=None):
        """
        Preprocess weather data for all cities for a specific date.
        If date is None, process previous day's data.
        """
        if date is None:
            date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        
        results = []
        
        for city in self.cities:
            try:
                city_name = city['name'].lower().replace(' ', '_')
                result = self.preprocess_city_data(city_name, date)
                results.append(result)
            except Exception as e:
                logger.error(f"Error preprocessing data for {city['name']}: {str(e)}")
                results.append({
                    'city': city['name'],
                    'date': date,
                    'status': 'error',
                    'error': str(e)
                })
        
        return results

    
    @log_function_call()
    def preprocess_city_data(self, city_name, date):
        """Preprocess weather data for a specific city and date."""
        input_path = self.raw_data_dir / f"{city_name}.csv"
        
        if not input_path.exists():
            logger.error(f"Data file not found: {input_path}")
            raise FileNotFoundError(f"Data file not found: {input_path}")
        
        # Read data
        df = pd.read_csv(input_path)
        
        # Filter only the requested date
        date_df = df[df['date'] == date]
        
        if date_df.empty:
            logger.error(f"No data found for {city_name} on {date}")
            raise ValueError(f"No data found for {city_name} on {date}")
        
        # Validate data
        self._validate_data(date_df)
        
        # Preprocess data
        clean_df = self._clean_data(date_df)
        
        # Read existing processed data file if it exists
        output_path = self.processed_data_dir / f"{city_name}.csv"
        
        if output_path.exists():
            processed_df = pd.read_csv(output_path)
            # Remove this date if it already exists
            processed_df = processed_df[processed_df['date'] != date]
            # Append new data
            processed_df = pd.concat([processed_df, clean_df], ignore_index=True)
            # Sort by date
            processed_df = processed_df.sort_values('date')
            processed_df.to_csv(output_path, index=False)
        else:
            # Create new file
            clean_df.to_csv(output_path, index=False)
        
        logger.info(f"Preprocessed data for {date} saved to {output_path}")
        
        return {
            'city': city_name,
            'date': date,
            'status': 'success',
            'input_path': str(input_path),
            'output_path': str(output_path),
            'row_count': len(clean_df)
        }
    
    def _validate_data(self, df):
        """Validate the data for completeness and correctness."""
        # Check for required columns
        required_columns = ['city', 'date', 'temperature', 'humidity', 'pressure', 'wind_speed']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing columns in data: {missing_columns}")
            raise ValueError(f"Missing columns in data: {missing_columns}")
        
        # Check for null values in critical columns
        null_counts = df[required_columns].isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0].index.tolist()
        
        if columns_with_nulls:
            logger.warning(f"Null values found in columns: {columns_with_nulls}")
    
    def _clean_data(self, df):
        """Clean and transform the data."""
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values for numeric columns
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for col in numeric_columns:
            if col in df_clean.columns:
                # Replace extreme outliers with NaN
                q1 = df_clean[col].quantile(0.25)
                q3 = df_clean[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                
                df_clean.loc[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound), col] = np.nan
                
                # Fill remaining NaN with median
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Ensure date is in correct format
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date']).dt.strftime('%Y-%m-%d')
        
        return df_clean
