import os
import pickle
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# LSTM dependencies
import tensorflow as tf
# Import Keras directly from tensorflow package
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Prophet
# from prophet import Prophet

# # ARIMA
# from statsmodels.tsa.arima.model import ARIMA

def positional_encoding(max_len, d_model):
    """Create positional encodings for the transformer model."""
    # Create position values as a range from 0 to max_len
    position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
    
    angle_rads = get_angles(
        position,
        tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model)
    
    # Apply sin to even indices
    sines = tf.math.sin(angle_rads[:, 0::2])
    # Apply cos to odd indices
    cosines = tf.math.cos(angle_rads[:, 1::2])
    
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    
    return tf.cast(pos_encoding, tf.float32)

def get_angles(positions, i, d_model):
    """Calculate angles for positional encoding."""
    angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return positions * angle_rates

def train_model_for_city(
    city_info: dict,
    batch_size: int = None,
    learning_rate: float = None,
    epochs: int = None
) -> dict:
    """
    Train a model for one city according to config/model.yaml.
    Returns {"metrics": {...}, "model_path": str}
    """
    # --- load config ---
    root = Path(__file__).resolve().parent.parent.parent
    cfg = yaml.safe_load((root / "config" / "model.yaml").read_text())
    mdl_cfg = cfg["model"]
    model_type = mdl_cfg["type"]
    params = mdl_cfg["parameters"][model_type].copy()
    # apply overrides
    params["batch_size"] = batch_size
    # if "learning_rate" in params and learning_rate is not None:
    params["learning_rate"] = learning_rate
    params["epochs"] = epochs
    test_size = cfg["model"]["evaluation"]["test_size"]
    target_col = cfg["model"]["target"]
    feature_cols = cfg["model"]["features"]
    
    city_name = city_info["name"].lower().replace(" ", "_")
    # --- load processed data ---
    data_root = Path(os.getenv("DATA_ROOT", "data"))
    df = pd.read_csv(data_root / "processed" / f"{city_name}.csv", parse_dates=["date"])
    
    # prepare X,y
    if model_type == "lstm":
        # create sequences
        seq_len = params.get("sequence_length", 1)
        values = df[feature_cols + [target_col]].values
        X, y = [], []
        
        for i in range(len(values) - seq_len):
            X.append(values[i : i + seq_len, :-1])
            y.append(values[i + seq_len, -1])
        X = np.array(X)
        y = np.array(y)
    else:
        X = df[feature_cols].values
        y = df[target_col].values

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=(model_type!="arima"), random_state=42
    )

    # --- training & prediction ---
    if model_type == "lstm":
        tf.keras.backend.clear_session()
        
        # Replace simple LSTM with a Transformer-based model
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Input, Dense, Dropout, GlobalAveragePooling1D
        from tensorflow.keras.models import Model
        
        # Input shape from data preparation remains the same
        input_shape = X_train.shape[1:]
        inputs = Input(shape=input_shape)
        
        # Extract scalar value for sequence length to ensure it's not a tensor
        seq_length = input_shape[0]
        if isinstance(seq_length, tf.Tensor):
            seq_length = tf.cast(seq_length, tf.int32).numpy()
        elif not isinstance(seq_length, (int, float)):
            seq_length = int(seq_length)  # Convert to scalar if it's numpy array
            
        # Add positional encoding with scalar sequence length
        pos_encoding = positional_encoding(seq_length, input_shape[-1])
        x = inputs + pos_encoding[:, :seq_length, :]
        
        # Increased capacity - more layers and attention heads
        for _ in range(params.get("layers", 3)):  # Increased from 2 to 3 layers
            # Self-attention layer with more heads and larger key dimension
            attention_output = MultiHeadAttention(
                num_heads=params.get("num_heads", 8),  # Increased from 4 to 8 heads
                key_dim=params.get("key_dim", 96)      # Increased from 64 to 96
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(attention_output + x)
            
            # Feed-forward network with more units and L2 regularization
            ffn = Dense(params.get("units", 256),      # Increased from 128 to 256
                       activation="relu",
                       kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
            ffn = Dropout(params.get("dropout", 0.15))(ffn)  # Slightly increased dropout
            ffn = Dense(input_shape[-1], 
                       kernel_regularizer=tf.keras.regularizers.l2(1e-5))(ffn)
            x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global pooling with additional dense layer
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation="relu")(x)  # Additional layer for better feature extraction
        
        # Final prediction layer
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model with learning rate scheduler
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        model.compile(optimizer=opt, loss="mse")
        
        # Train model - same interface as before
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        preds = model.predict(X_test).flatten()
        model_obj = model

    # elif model_type == "prophet":
    #     prophet_df = df.rename(columns={"date": "ds", target_col: "y"})
    #     m = Prophet()
    #     m.fit(prophet_df)
    #     future = pd.DataFrame({"ds": df["date"].iloc[-int(len(df)*test_size):]})
    #     fcst = m.predict(future)
    #     preds = fcst["yhat"].values
    #     model_obj = m

    # elif model_type == "arima":
    #     # fit on entire series, forecast test_size steps
    #     series = pd.Series(y, index=df["date"])
    #     order = (params.get("p",1), params.get("d",1), params.get("q",1))
    #     m = ARIMA(series.iloc[:-int(len(series)*test_size)], order=order).fit()
    #     preds = m.forecast(steps=int(len(series)*test_size))
    #     preds = np.array(preds)
    #     model_obj = m

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # --- metrics ---
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1e-8))) * 100
    metrics = {"mae": mae, "rmse": rmse, "mape": mape}

    # --- persist model ---
    model_root = Path(os.getenv("MODEL_ROOT", "models")) / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if model_type == "lstm":
        # save full model (architecture + weights) in HDF5
        model_path = model_root / f"{city_name}_model_{timestamp}.h5"
        model_obj.save(str(model_path), save_format="h5")
    else:
        # fallback to pickle for non‐Keras models
        model_path = model_root / f"{city_name}_model_{timestamp}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)

    return {
        "metrics": metrics,
        "model_path": str(model_path)
    }