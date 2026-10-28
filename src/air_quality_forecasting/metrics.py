"""Evaluation metrics kept in original pollutant units."""
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("Actual and predicted arrays must be non-empty and have equal lengths.")
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    maximum = float(np.max(actual))
    pct = float(100 * rmse / maximum) if maximum != 0 else float("nan")
    return {"rmse": rmse, "percentage_rmse": pct}
