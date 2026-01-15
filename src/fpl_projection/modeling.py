from __future__ import annotations

import tensorflow as tf


def build_lstm_model(*, seq_length: int, num_features: int, horizon: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(seq_length, num_features))
    x = tf.keras.layers.LSTM(64, activation="tanh")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(horizon, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
