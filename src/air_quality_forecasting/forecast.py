"""Recursive multi-step forecasting."""
import numpy as np

def recursive_forecast(model, history, steps: int = 5, lookback: int = 24, scaler=None) -> np.ndarray:
    """Predict recursively; history and output are in original units."""
    if steps < 1:
        raise ValueError("steps must be positive")
    values = np.asarray(history, dtype=float).reshape(-1)
    if len(values) < lookback:
        raise ValueError(f"Need at least {lookback} history values, received {len(values)}.")
    transformed = scaler is not None
    work = scaler.transform(values.reshape(-1, 1)).reshape(-1) if transformed else values.copy()
    result = []
    for _ in range(steps):
        x = work[-lookback:].reshape(1, lookback, 1).astype(np.float32)
        prediction = float(np.asarray(model.predict(x, verbose=0)).reshape(-1)[0])
        work = np.append(work, prediction)
        result.append(float(scaler.inverse_transform([[prediction]])[0, 0]) if transformed else prediction)
    return np.asarray(result)
