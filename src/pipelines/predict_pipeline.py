import os
import sys
import h5py

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import tensorflow as tf
from src.logging import get_logger, log_function_call
from src.config import get_cities, get_config
from src.data.load_data import WeatherDataLoader
from src.data.preprocessing import WeatherDataPreprocessor

logger = get_logger(__name__)

class WeatherPredictionPipeline:
    """Pipeline to generate weather predictions for the current day."""
    
    def __init__(self):
        self.DATA_ROOT = os.getenv("DATA_ROOT", "data")
        self.MODEL_ROOT = os.getenv("MODEL_ROOT", "models")
        self.data_loader = WeatherDataLoader()
        self.preprocessor = WeatherDataPreprocessor()
        self.predictions_dir = Path(self.DATA_ROOT) / "predictions"
        self.model_dir = Path(self.MODEL_ROOT) / "models"
        
        # Ensure predictions directory exists
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
    
    @log_function_call()
    def predict_for_city(self, city):
        """Generate predictions for a specific city."""
        city_name = city['name'].lower().replace(' ', '_')
        
        # try:
            # 1. Load previous days' data
        model_config = get_config('model')
        lookback_days = model_config.get('parameters', {}).get('sequence_length', 7)
        recent_data = self.data_loader.load_previous_days_data(city_name, lookback_days)
        
        # 2. Preprocess data for model input
        features = model_config.get('features', [])
        X = self._prepare_sequence(recent_data, features)
        
        # 3. Load the trained model
        
        model_path = self.model_dir / f"{city_name}_model.h5"
        logger.info(
            f"model_path.exists={model_path.exists()}, "
            f"is_file={model_path.is_file()}, is_dir={model_path.is_dir()}, "
            f"is_hdf5={h5py.is_hdf5(model_path)}, size={model_path.stat().st_size}"
        )
        if not model_path.exists():
            raise FileNotFoundError(f"No model found at {model_path}")

        # if it's genuine HDF5, load it; otherwise assume it's weights-only
        if h5py.is_hdf5(model_path):
            model = tf.keras.models.load_model(str(model_path))
        else:
            # you must have a function that rebuilds the exact architecture
            from src.models import build_model  
            model = build_model(get_config('model'))
            model.load_weights(str(model_path))

        # 4. Generate prediction for current day
        prediction = model.predict(X)

        # 5. Save prediction to CSV
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Create prediction data
        target_feature = model_config.get('target', 'temperature')
        pred_value = float(prediction[0][0])
        
        # Save to CSV
        pred_df = pd.DataFrame({
            'city': city['name'],
            'date': today,
            'predicted_' + target_feature: pred_value,
            'timestamp': datetime.now().isoformat()
        }, index=[0])
        
        # Path to save prediction
        pred_file = self.predictions_dir / f"{city_name}.csv"
        
        # If file exists, append; otherwise create new
        if pred_file.exists():
            existing_df = pd.read_csv(pred_file)
            
            # Check if prediction for today already exists
            if (existing_df['date'] == today).any():
                # Update existing prediction
                existing_df.loc[existing_df['date'] == today, 'predicted_' + target_feature] = pred_value
                existing_df.loc[existing_df['date'] == today, 'timestamp'] = datetime.now().isoformat()
            else:
                # Append new prediction
                existing_df = pd.concat([existing_df, pred_df], ignore_index=True)
            
            existing_df.to_csv(pred_file, index=False)
        else:
            # Create new file
            pred_df.to_csv(pred_file, index=False)
        
        logger.info(f"Saved prediction for {city['name']} on {today} to {pred_file}")
        
        return {
            'city': city['name'],
            'date': today,
            'predicted_' + target_feature: pred_value,
            'status': 'success'
        }
            
        # except Exception as e:
        #     logger.error(f"Error generating prediction for {city['name']}: {str(e)}")
        #     return {
        #         'city': city['name'],
        #         'date': datetime.now().strftime("%Y-%m-%d"),
        #         'status': 'error',
        #         'error': str(e)
        #     }
    
    def _prepare_sequence(self, df, features):
        """Prepare data sequence for LSTM input."""
        # Extract features and normalize
        X = df[features].values
        
        # Normalize data (simple min-max scaling)
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
        
        # Reshape to [1, sequence_length, n_features] for LSTM input
        X_reshaped = np.expand_dims(X_norm, axis=0)
        
        return X_reshaped
    
    @log_function_call()
    def run_predictions_for_all_cities(self):
        """Generate predictions for all configured cities."""
        cities = get_cities()
        results = []
        
        for city in cities:
            result = self.predict_for_city(city)
            results.append(result)
        
        return results

def prediction_pipeline():
    """Main function to run the weather prediction pipeline."""
    pipeline = WeatherPredictionPipeline()
    results = pipeline.run_predictions_for_all_cities()
    
    # Log results
    for result in results:
        if result.get('status') == 'success':
            logger.info(f"Prediction successful for {result['city']} on {result['date']}: {result['predicted_temperature']}")
        else:
            logger.error(f"Prediction failed for {result['city']}: {result.get('error')}")

if __name__ == "__main__":
    # Run the pipeline for testing purposes
    prediction_pipeline()

