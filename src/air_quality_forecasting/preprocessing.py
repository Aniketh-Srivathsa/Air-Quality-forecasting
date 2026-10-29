"""Timestamp normalization, hourly aggregation, outlier filtering and weekly imputation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

TIME_CANDIDATES = ("timestamp", "datetime", "date_time", "date", "time", "from date")
STATION_CANDIDATES = ("station", "station name", "location", "site", "monitoring station")
NON_POLLUTANT_TOKENS = ("temp", "humidity", "wind", "weather", "rain", "pressure", "latitude", "longitude", "aqi", "index", "station id", "site id", "serial", "record id")
NON_POLLUTANT_ABBREVIATIONS = {"at", "rh", "ws", "wd", "rf", "tot-rf", "sr", "bp", "vws"}

@dataclass
class PreparedData:
    hourly: pd.DataFrame
    observed_hourly: pd.DataFrame
    imputation_details: pd.DataFrame
    pollutants: list[str]
    timestamp_column: str
    station_column: str | None
    station: str | None
    summary: dict

def _column_match(columns, candidates):
    normalized = {str(c).strip().lower(): c for c in columns}
    return next((normalized[x] for x in candidates if x in normalized), None)

def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse ISO timestamps without day-first reinterpretation; support D/M/Y exports too."""
    sample = values.astype("string").dropna().head(100)
    iso_ratio = float(sample.str.match(r"^\s*\d{4}-\d{2}-\d{2}(?:$|[T\s])").mean()) if len(sample) else 0.0
    return pd.to_datetime(values, format="mixed", errors="coerce", utc=True, dayfirst=iso_ratio < 0.8)


def detect_columns(df: pd.DataFrame):
    time_col = _column_match(df.columns, TIME_CANDIDATES)
    station_col = _column_match(df.columns, STATION_CANDIDATES)
    if time_col is None:
        # Common CPCB exports use From Date or Timestamp. If names differ, accept
        # a column only when most non-null entries parse as timestamps.
        for c in df.columns:
            parsed = parse_timestamps(df[c])
            if parsed.notna().mean() >= 0.8:
                time_col = c
                break
    if time_col is None:
        raise ValueError("No timestamp column found. Expected a date/time field such as Timestamp or From Date.")
    return time_col, station_col

def infer_pollutants(df: pd.DataFrame, time_col: str, station_col: str | None) -> list[str]:
    result = []
    for col in df.columns:
        if col in (time_col, station_col):
            continue
        name = str(col).strip().lower()
        base_name = name.split("(", 1)[0].strip().split()[0]
        if base_name in NON_POLLUTANT_ABBREVIATIONS or any(token in name for token in NON_POLLUTANT_TOKENS):
            continue
        values = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False).str.strip().replace({"": np.nan, "-": np.nan, "NA": np.nan, "N/A": np.nan}), errors="coerce")
        if values.notna().sum() > 0:
            result.append(col)
    return result

def remove_zscore_outliers(series: pd.Series, threshold: float = 3.0) -> tuple[pd.Series, int]:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    mean, std = numeric.mean(), numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return numeric, 0
    mask = ((numeric - mean).abs() / std) > threshold
    count = int(mask.sum())
    return numeric.mask(mask), count

def impute_weekly(series: pd.Series, lag: int = 168) -> pd.Series:
    """Fill missing hourly points from one week earlier, then the series mean."""
    values = pd.to_numeric(series, errors="coerce").astype(float).copy()
    mean = values.mean()
    if pd.isna(mean):
        raise ValueError("Cannot impute a pollutant series with no observed numeric values.")
    weekly = values.shift(lag)
    values = values.fillna(weekly)
    return values.fillna(mean)

def prepare_data(raw: pd.DataFrame, station: str | None = None) -> PreparedData:
    time_col, station_col = detect_columns(raw)
    frame = raw.copy()
    frame[time_col] = parse_timestamps(frame[time_col])
    valid_timestamp_rows = int(frame[time_col].notna().sum())
    frame = frame.dropna(subset=[time_col]).sort_values(time_col)
    selected_station = None
    if station_col and station is not None:
        frame = frame[frame[station_col].astype(str) == str(station)]
        selected_station = station
    elif station_col:
        choices = frame[station_col].dropna().astype(str).unique()
        if len(choices) == 1:
            selected_station = choices[0]
            frame = frame[frame[station_col].astype(str) == selected_station]
    pollutants = infer_pollutants(frame, time_col, station_col)
    if not pollutants:
        raise ValueError("No numeric pollutant measurement columns were found in the CSV.")
    for col in pollutants:
        frame[col] = pd.to_numeric(frame[col].astype(str).str.replace(",", "", regex=False).str.strip().replace({"": np.nan, "-": np.nan, "NA": np.nan, "N/A": np.nan}), errors="coerce")
    duplicate_count = int(frame.duplicated(subset=[time_col], keep="last").sum())
    frame = frame.drop_duplicates(subset=[time_col], keep="last").set_index(time_col).sort_index()
    # Remove outlier measurements before hourly aggregation so a spike in a
    # 15-minute source row cannot distort its clock-hour mean.
    outlier_counts = {}
    for col in pollutants:
        frame[col], outlier_counts[col] = remove_zscore_outliers(frame[col])
    # Retain pre-imputation means to show measured gaps honestly in the UI.
    observed_hourly = frame[pollutants].resample("h").mean()
    hourly_missing = {col: int(observed_hourly[col].isna().sum()) for col in pollutants}
    hourly = observed_hourly.copy()
    imputation_records = []
    for col in pollutants:
        original = hourly[col].copy()
        missing_mask = original.isna()
        weekly_source = original.shift(168)
        weekly_fill = missing_mask & weekly_source.notna()
        mean_fill = missing_mask & weekly_source.isna()
        for timestamp in original.index[weekly_fill]:
            source_timestamp = timestamp - pd.Timedelta(hours=168)
            imputation_records.append({
                "Timestamp": timestamp,
                "Pollutant": col,
                "Imputed value": float(weekly_source.loc[timestamp]),
                "Method": "Same hour 168 hours earlier",
                "Source timestamp": source_timestamp,
                "Source value": float(weekly_source.loc[timestamp]),
            })
        fallback_mean = float(original.mean()) if original.notna().any() else np.nan
        for timestamp in original.index[mean_fill]:
            imputation_records.append({
                "Timestamp": timestamp,
                "Pollutant": col,
                "Imputed value": fallback_mean,
                "Method": "Series mean (no weekly source)",
                "Source timestamp": pd.NaT,
                "Source value": fallback_mean,
            })
        hourly[col] = impute_weekly(original)
    if hourly.empty:
        raise ValueError("No hourly observations remain after parsing timestamps.")
    summary = {
        "raw_rows": int(len(raw)), "valid_timestamp_rows": valid_timestamp_rows,
        "duplicate_timestamps_removed": duplicate_count,
        "hourly_rows": int(len(hourly)), "outliers_removed": outlier_counts,
        "hourly_missing_before_imputation": hourly_missing,
        "hourly_values_imputed": hourly_missing.copy(),
        "missing_after_imputation": {c: int(hourly[c].isna().sum()) for c in pollutants},
    }
    imputation_details = pd.DataFrame(imputation_records, columns=[
        "Timestamp", "Pollutant", "Imputed value", "Method", "Source timestamp", "Source value"
    ])
    return PreparedData(hourly, observed_hourly, imputation_details, pollutants, time_col, station_col, selected_station, summary)

