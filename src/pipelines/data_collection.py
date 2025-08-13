# src/pipelines/data_collection.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.api_client import OpenWeatherMapClient
from src.data.preprocessing import WeatherDataPreprocessor
from src.data.storage_setup import setup_storage_structure
from src.data.version_control import track_new_data
from src.config import get_cities
from src.logging import get_logger
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import os

logger = get_logger(__name__)

def run_data_collection_pipeline():
    DATA_ROOT = Path(os.getenv("DATA_ROOT", "data"))
    """Run the complete data collection pipeline."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    
    try:
        # 1. Set up storage structure
        logger.info("Setting up storage structure")
        setup_storage_structure()
        
        # 2. Fetch weather data for all cities
        logger.info(f"Fetching weather data for {yesterday}")
        client = OpenWeatherMapClient()
        cities = get_cities()
        
        for city in cities:
            try:
                city_name = city['name'].lower().replace(' ', '_')
                logger.info(f"Processing {city['name']}")
                
                # Fetch data
                weather_data = client.get_previous_day_weather(city)
                
                # Save to CSV - append to existing file

                output_path =  DATA_ROOT / "raw" / f"{city_name}.csv"
                df = pd.DataFrame([weather_data])
                logger.info(f"path: {output_path}")
                
                if output_path.exists():
                    # Check if this date already exists
                    existing_df = pd.read_csv(output_path)
                    if (existing_df['date'] == yesterday).any():
                        logger.warning(f"Data for {yesterday} already exists for {city_name}. Skipping.")
                        continue
                    
                    # Append without headers
                    df.to_csv(output_path, mode='a', header=False, index=False)
                    logger.info(f"Appended raw data to {output_path}")
                else:
                    # Create new file with headers
                    df.to_csv(output_path, index=False)
                    logger.info(f"Created new file with raw data at {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {city['name']}: {str(e)}")
        
        # 3. Preprocess the data
        logger.info("Preprocessing weather data")
        preprocessor = WeatherDataPreprocessor()
        results = preprocessor.preprocess_all_cities(date=yesterday)
        
        # Log preprocessing results
        for result in results:
            if result.get('status') == 'success':
                logger.info(f"Preprocessed {result['city']} data: {result['row_count']} rows")
            else:
                logger.error(f"Preprocessing failed for {result['city']}: {result.get('error')}")
        
        # 4. Update DVC tracking
        logger.info("Updating DVC tracking")
        track_new_data()
        
        logger.info("Data collection pipeline completed successfully")
    
    except Exception as e:
        logger.error(f"Error in data collection pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    run_data_collection_pipeline()
