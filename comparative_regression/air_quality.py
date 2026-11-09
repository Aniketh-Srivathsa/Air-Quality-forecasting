"""Train and compare the four regressors used by Ameer et al. (2019).

The paper's target is hourly PM2.5. This runnable adaptation trains on recent
Open-Meteo hourly history and recursively forecasts up to 5 days ahead. The API
forecast values are modelled estimates, not regulatory monitor observations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
HISTORY_DAYS = 90
MAX_FORECAST_HOURS = 119

CITY_COORDINATES: dict[str, tuple[float, float, str]] = {
    "beijing": (39.9042, 116.4074, "Asia/Shanghai"),
    "chengdu": (30.5728, 104.0668, "Asia/Shanghai"),
    "guangzhou": (23.1291, 113.2644, "Asia/Shanghai"),
    "shanghai": (31.2304, 121.4737, "Asia/Shanghai"),
    "shenyang": (41.8057, 123.4315, "Asia/Shanghai"),
}

WEATHER_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]
FEATURES = [
    "pm25",
    *WEATHER_FIELDS,
    "hour",
    "day_of_week",
    "month",
    "pm25_lag_1",
    "pm25_lag_3",
    "pm25_lag_24",
    "pm25_mean_24",
]
TARGET = "target_pm25"


@dataclass
class TrainedCity:
    city: str
    model_name: str
    model: Any
    scores: pd.DataFrame
    training_rows: int


def fetch_hourly(city: str, past_days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Fetch aligned hourly PM2.5 and weather series from Open-Meteo."""
    key = city.lower()
    if key not in CITY_COORDINATES:
        raise ValueError(f"Unknown city {city!r}; choose from {', '.join(CITY_COORDINATES)}")
    latitude, longitude, timezone = CITY_COORDINATES[key]
    common = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "past_days": past_days,
        "forecast_days": 5,
    }
    session = requests.Session()
    weather = session.get(
        "https://api.open-meteo.com/v1/forecast",
        params={**common, "hourly": ",".join(WEATHER_FIELDS)},
        timeout=30,
    )
    air = session.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={**common, "hourly": "pm2_5"},
        timeout=30,
    )
    weather.raise_for_status()
    air.raise_for_status()
    weather_json, air_json = weather.json(), air.json()
    w = pd.DataFrame(weather_json["hourly"])
    a = pd.DataFrame(air_json["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    a["time"] = pd.to_datetime(a["time"])
    data = w.merge(a[["time", "pm2_5"]], on="time", how="inner")
    data = data.rename(columns={"pm2_5": "pm25"})
    # Open-Meteo returns local wall-clock timestamps when a timezone is set.
    # Localize them explicitly so comparisons to the current city time are sound.
    if data["time"].dt.tz is None:
        data["time"] = data["time"].dt.tz_localize(timezone)
    data["city"] = key
    data = data.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if data.empty:
        raise RuntimeError(f"Open-Meteo returned no aligned hourly data for {city}")
    return data


def build_supervised(data: pd.DataFrame) -> pd.DataFrame:
    """Create one-hour-ahead samples using only information available at hour t."""
    frame = data.copy().sort_values("time").reset_index(drop=True)
    frame["time"] = pd.to_datetime(frame["time"])
    frame["hour"] = frame["time"].dt.hour
    frame["day_of_week"] = frame["time"].dt.dayofweek
    frame["month"] = frame["time"].dt.month
    frame["pm25_lag_1"] = frame["pm25"].shift(1)
    frame["pm25_lag_3"] = frame["pm25"].shift(3)
    frame["pm25_lag_24"] = frame["pm25"].shift(24)
    frame["pm25_mean_24"] = frame["pm25"].shift(1).rolling(24, min_periods=12).mean()
    frame[TARGET] = frame["pm25"].shift(-1)
    return frame


def make_models() -> dict[str, Any]:
    """Construct Decision Tree, Random Forest, MLP and Gradient Boosting models."""
    return {
        "decision_tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeRegressor(max_depth=12, min_samples_leaf=3, random_state=42),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=250, min_samples_leaf=2, n_jobs=-1, random_state=42),
        ),
        "mlp": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True,
                         max_iter=500, random_state=42),
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
        ),
    }


def train_city(data: pd.DataFrame, city: str) -> TrainedCity:
    """Chronologically evaluate each model, then refit the MAE winner."""
    frame = build_supervised(data)
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)
    if len(frame) < 72:
        raise ValueError(f"Need at least 72 aligned hourly samples; got {len(frame)}")
    split = int(len(frame) * 0.8)
    train, validation = frame.iloc[:split], frame.iloc[split:]
    scores: list[dict[str, float | str]] = []
    for name, model in make_models().items():
        model.fit(train[FEATURES], train[TARGET])
        prediction = np.maximum(0, model.predict(validation[FEATURES]))
        scores.append({
            "model": name,
            "mae": float(mean_absolute_error(validation[TARGET], prediction)),
            "rmse": float(np.sqrt(mean_squared_error(validation[TARGET], prediction))),
            "r2": float(r2_score(validation[TARGET], prediction)),
        })
    score_frame = pd.DataFrame(scores).sort_values(["mae", "rmse"]).reset_index(drop=True)
    winner = str(score_frame.iloc[0]["model"])
    final_model = make_models()[winner]
    final_model.fit(frame[FEATURES], frame[TARGET])
    return TrainedCity(city.lower(), winner, final_model, score_frame, len(frame))


def _feature_row(
    timestamp: pd.Timestamp,
    pm_values: list[float],
    weather_row: pd.Series,
) -> dict[str, float]:
    last24 = pm_values[-24:]
    return {
        "pm25": pm_values[-1],
        **{field: float(weather_row[field]) if pd.notna(weather_row[field]) else np.nan
           for field in WEATHER_FIELDS},
        "hour": float(timestamp.hour),
        "day_of_week": float(timestamp.dayofweek),
        "month": float(timestamp.month),
        "pm25_lag_1": pm_values[-2] if len(pm_values) >= 2 else np.nan,
        "pm25_lag_3": pm_values[-4] if len(pm_values) >= 4 else np.nan,
        "pm25_lag_24": pm_values[-25] if len(pm_values) >= 25 else np.nan,
        "pm25_mean_24": float(np.mean(last24)) if len(last24) >= 12 else np.nan,
    }


def forecast(trained: TrainedCity, data: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Recursively forecast PM2.5, using API weather forecasts for future hours."""
    if not 1 <= hours <= MAX_FORECAST_HOURS:
        raise ValueError(f"Forecast horizon must be from 1 to {MAX_FORECAST_HOURS} hours")
    frame = data.sort_values("time").reset_index(drop=True).copy()
    frame["time"] = pd.to_datetime(frame["time"])
    timezone = CITY_COORDINATES[trained.city][2]
    now = pd.Timestamp.now(tz=timezone)
    past = frame.loc[frame["time"] <= now].dropna(subset=["pm25"])
    future = frame.loc[frame["time"] > now].head(hours)
    if past.empty:
        raise ValueError("No past PM2.5 values are available to initialize the forecast")
    if len(future) < hours:
        raise ValueError(f"Only {len(future)} future weather/air rows available; requested {hours}")
    pm_values = past["pm25"].astype(float).tolist()
    results: list[dict[str, Any]] = []
    for _, row in future.iterrows():
        timestamp = pd.Timestamp(row["time"])
        x = pd.DataFrame([_feature_row(timestamp, pm_values, row)], columns=FEATURES)
        predicted = max(0.0, float(trained.model.predict(x)[0]))
        pm_values.append(predicted)
        results.append({"time": timestamp.isoformat(), "city": trained.city,
                        "predicted_pm25": round(predicted, 3), "unit": "µg/m³",
                        "model": trained.model_name})
    return pd.DataFrame(results)


def run(city: str, hours: int, save_model: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = fetch_hourly(city)
    # Exclude future API forecast rows from model evaluation and fitting.
    now = pd.Timestamp.now(tz=CITY_COORDINATES[city.lower()][2])
    training_data = data.loc[(data["time"] <= now) & data["pm25"].notna()].copy()
    trained = train_city(training_data, city)
    predictions = forecast(trained, data, hours)
    if save_model:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": trained.model, "features": FEATURES, "city": trained.city,
                     "model_name": trained.model_name}, MODEL_DIR / f"{city.lower()}.joblib")
    REPORT_DIR.mkdir(exist_ok=True)
    score_path = REPORT_DIR / f"{city.lower()}_comparison.csv"
    prediction_path = REPORT_DIR / f"{city.lower()}_forecast.csv"
    trained.scores.to_csv(score_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    return trained.scores, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=sorted(CITY_COORDINATES), default="beijing")
    parser.add_argument("--hours", type=int, default=24, help="forecast horizon (1-119 hours)")
    args = parser.parse_args()
    scores, predictions = run(args.city, args.hours)
    print("Chronological validation scores (lower MAE/RMSE is better):")
    print(scores.to_string(index=False))
    print(f"\nSelected model: {scores.iloc[0]['model']}")
    print("Forecast:")
    print(predictions.to_string(index=False))
    print(f"\nSaved outputs under {REPORT_DIR}")


if __name__ == "__main__":
    main()

