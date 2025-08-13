# src/data/api_client.py
import requests
import os
import time
from datetime import datetime, timedelta, timezone
from src.logging import get_logger, log_function_call
from src.config import get_config, get_cities
from dotenv import load_dotenv

logger = get_logger(__name__)

class OpenWeatherMapClient:
    """Client for fetching weather data from OpenWeatherMap API."""
    
    def __init__(self):
        api_config = get_config('pipeline', 'data_collection').get('api', {})
        self.base_url = api_config.get('base_url')
        self.endpoint = api_config.get('endpoint')
        self.units = api_config.get('units', 'metric')
        self.timeout = api_config.get('timeout', 30)
        self.retry_attempts = api_config.get('retry_attempts', 3)
        self.retry_delay = api_config.get('retry_delay', 5)
        
        # Get API key from environment variable
        load_dotenv()
        api_key_env = api_config.get('api_key', '').strip('${}')
        self.api_key = os.environ.get(api_key_env)
        
        if not self.api_key:
            logger.error(f"API key not found in environment variable {api_key_env}")
            raise ValueError(f"API key not found in environment variable {api_key_env}")
    
    @log_function_call()
    def get_previous_day_weather(self, city):
        """
        Fetch weather data for a specific city for the previous day.
        
        Args:
            city: City information dictionary with name, id, etc.
            
        Returns:
            Weather data as a dictionary
        """
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        
        params = {
            'q': f"{city['name']},{city['country']}",
            'appid': self.api_key,
            'units': self.units,
            'dt': int(yesterday.timestamp())
        }
        
        url = f"{self.base_url}/{self.endpoint}"
        
        for attempt in range(self.retry_attempts):
            try:
                logger.info(f"Fetching weather data for {city['name']} on {yesterday_str}")
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                weather_data = response.json()
                processed_data = self._process_weather_data(weather_data, city, yesterday_str)
                
                logger.info(f"Successfully fetched weather data for {city['name']}")
                return processed_data
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt+1}/{self.retry_attempts} failed for {city['name']}: {str(e)}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to fetch weather data for {city['name']} after {self.retry_attempts} attempts")
                    raise
    
    def _process_weather_data(self, data, city, date_str):
        """Extract relevant fields from the API response."""
        try:
            return {
                'city': city['name'],
                'country': city['country'],
                'date': date_str,
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'timestamp': datetime.now().isoformat()
            }
        except KeyError as e:
            logger.error(f"Error processing weather data: {e}")
            raise ValueError(f"Missing expected field in weather data: {e}")
