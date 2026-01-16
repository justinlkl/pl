"""LSTM model architecture for FPL point prediction.

Builds a sequence-to-sequence LSTM model that predicts player points
for multiple future gameweeks based on historical performance.
"""

from __future__ import annotations

import tensorflow as tf


def build_lstm_model(*, seq_length: int, num_features: int, horizon: int) -> tf.keras.Model:
    """Build an LSTM model for multi-step FPL point prediction.
    
    Architecture:
    - LSTM layer (128 units) to capture temporal patterns
    - Dropout for regularization
    - Dense layers for feature processing
    - Output layer for multi-step prediction (6 gameweeks)
    
    Args:
        seq_length: Length of input sequences (gameweeks to look back)
        num_features: Number of input features per timestep
        horizon: Number of gameweeks to predict ahead
        
    Returns:
        Compiled Keras model ready for training
    """
    inputs = tf.keras.Input(shape=(seq_length, num_features))
    
    # First LSTM layer: capture temporal dependencies
    x = tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Second LSTM layer: deeper temporal modeling
    x = tf.keras.layers.LSTM(64, activation="tanh")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Dense layers for final predictions
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    
    # Output layer: predict points for next 'horizon' gameweeks
    outputs = tf.keras.layers.Dense(horizon, activation="linear")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    
    return model
