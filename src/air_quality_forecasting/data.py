"""Download and load the public substitute dataset without modifying its raw copy."""
from pathlib import Path
import requests
import pandas as pd

DATA_URL = "https://data.opencity.in/dataset/79fa909a-38f1-4657-b369-6d0e2a6f500e/resource/3251915c-9455-4784-b7db-5fad4963310d/download/blr-jayanagar-5th-block-kspcb-2024-25.csv"
RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "blr-jayanagar-5th-block-kspcb-2024-25.csv"

def download_dataset(path: str | Path = RAW_PATH, url: str = DATA_URL, timeout: int = 45) -> Path:
    """Download once to path. A failed transfer never replaces an existing file."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with temp.open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        out.write(chunk)
        if not temp.stat().st_size:
            raise OSError("The server returned an empty CSV file.")
        temp.replace(path)
    except Exception as exc:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download the required substitute dataset from {url}. Check your internet connection or download it manually into {path}. Details: {exc}") from exc
    return path

def load_csv(path: str | Path = RAW_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        path = download_dataset(path)
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not read CSV at {path}: {exc}") from exc


OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPEN_METEO_COORDINATES = {"latitude": 12.925, "longitude": 77.583}
OPEN_METEO_FIELDS = {
    "pm2_5": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "nitrogen_dioxide": "NO2 (µg/m³)",
    "nitrogen_monoxide": "NO (µg/m³)",
    "sulphur_dioxide": "SO2 (µg/m³)",
    "ozone": "O3 (µg/m³)",
    "carbon_monoxide": "CO (µg/m³)",
    "ammonia": "NH3 (µg/m³)",
}

def load_current_estimates(timeout: int = 30) -> pd.DataFrame:
    """Fetch up to 92 days of recent CAMS gridded hourly estimates near Jayanagar.

    The API is model output, not a ground-station observation. Times are normalized
    to UTC and future forecast hours are excluded before they reach preprocessing.
    """
    params = {
        **OPEN_METEO_COORDINATES,
        "hourly": ",".join(OPEN_METEO_FIELDS),
        "past_days": 92,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    try:
        response = requests.get(OPEN_METEO_AIR_QUALITY_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            "Could not retrieve current air-quality estimates from Open-Meteo. "
            "Check your internet connection and retry. Details: " + str(exc)
        ) from exc
    hourly = payload.get("hourly") or {}
    if not hourly.get("time"):
        raise ValueError("Open-Meteo returned no hourly timestamp data.")
    frame = pd.DataFrame({"Timestamp": pd.to_datetime(hourly["time"], utc=True, errors="coerce")})
    for api_name, display_name in OPEN_METEO_FIELDS.items():
        values = hourly.get(api_name)
        if values is not None:
            frame[display_name] = pd.to_numeric(pd.Series(values), errors="coerce")
    pollutants = [name for name in OPEN_METEO_FIELDS.values() if name in frame]
    if not pollutants:
        raise ValueError("Open-Meteo did not return any supported pollutant columns.")
    # Use only completed past/current hours; never score the API's forward forecast
    # as if it were observed history. Keep a small time margin for API update delays.
    latest_completed = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    earliest = latest_completed - pd.Timedelta(days=92)
    frame = frame.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    frame = frame[(frame["Timestamp"] >= earliest) & (frame["Timestamp"] <= latest_completed)]
    frame = frame.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    if len(frame) < 24 * 3:
        raise ValueError(
            f"Open-Meteo returned only {len(frame)} completed hourly rows; at least 72 are "
            "needed for a safe chronological training/test split and 24-hour lookback."
        )
    if frame[pollutants].notna().sum().sum() == 0:
        raise ValueError("Open-Meteo returned timestamps but no numeric pollutant values.")
    return frame
