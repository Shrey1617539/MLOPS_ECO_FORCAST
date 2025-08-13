import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from src.config import get_config
from src.logging import get_logger

logger = get_logger(__name__)

def create_lstm_model(config):
    """Create an LSTM model for weather prediction."""
    model_config = get_config('model')
    sequence_length = model_config.get('parameters', {}).get('sequence_length', 7)
    
    # Get number of features from config
    features = model_config.get('features', [])
    n_features = len(features)
    
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(sequence_length, n_features), 
                   return_sequences=True))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
    
    logger.info(f"Created LSTM model with {sequence_length} sequence length and {n_features} features")
    return model
