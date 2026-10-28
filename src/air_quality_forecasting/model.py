"""Keras model construction and chronological train/test model fitting."""
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import MinMaxScaler

@dataclass
class TrainingResult:
    model: object
    scaler: MinMaxScaler
    test_index: object
    actual: np.ndarray
    predicted: np.ndarray
    history: object
    split_index: int
    lookback: int

def make_sequences(values, lookback: int):
    x, y = [], []
    for i in range(lookback, len(values)):
        x.append(values[i-lookback:i])
        y.append(values[i])
    return np.asarray(x, dtype=np.float32).reshape(-1, lookback, 1), np.asarray(y, dtype=np.float32)

def chronological_split_index(n_observations: int, train_fraction: float = 0.67) -> int:
    """Return the first held-out position for a chronological split."""
    if n_observations < 0 or not 0 < train_fraction < 1:
        raise ValueError("n_observations must be non-negative and train_fraction must be between 0 and 1.")
    return int(n_observations * train_fraction)

def build_model(lookback: int = 24):
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for training. Install dependencies with: python -m pip install -r requirements.txt") from exc
    model = keras.Sequential([
        keras.layers.Input(shape=(lookback, 1)),
        keras.layers.LSTM(64, activation="relu", return_sequences=True),
        keras.layers.LSTM(32, activation="relu"),
        keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def train_model(series, lookback: int = 24, max_epochs: int = 50, batch_size: int = 32, patience: int = 5, progress_callback=None) -> TrainingResult:
    """Split chronologically; scaler fit uses training observations only."""
    from tensorflow import keras
    values = np.asarray(series, dtype=np.float32).reshape(-1, 1)
    split = chronological_split_index(len(values), 0.67)
    if lookback < 1 or split <= lookback or len(values) - split < 1:
        raise ValueError(f"Insufficient observations ({len(values)}) for lookback {lookback} and a 67/33 chronological split.")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(values[:split])
    all_scaled = scaler.transform(values)
    x_train, y_train = make_sequences(train_scaled, lookback)
    # Targets before split are training-only. Test inputs use the preceding
    # lookback history, including the final train observations, as at forecast time.
    x_all, y_all = make_sequences(all_scaled, lookback)
    target_positions = np.arange(lookback, len(values))
    train_mask = target_positions < split
    test_mask = target_positions >= split
    x_test, y_test_scaled = x_all[test_mask], y_all[test_mask]
    model = build_model(lookback)
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)]
    if progress_callback:
        callbacks.append(progress_callback)
    history = model.fit(x_train, y_train, validation_split=0.1, epochs=min(max_epochs, 50), batch_size=batch_size, shuffle=False, callbacks=callbacks, verbose=0)
    pred_scaled = model.predict(x_test, verbose=0)
    actual = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(-1)
    predicted = scaler.inverse_transform(pred_scaled).reshape(-1)
    test_index = series.index[split:]
    return TrainingResult(model, scaler, test_index, actual, predicted, history, split, lookback)
