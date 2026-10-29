import numpy as np
import pandas as pd
import pytest

from air_quality_forecasting.preprocessing import impute_weekly, parse_timestamps
from air_quality_forecasting.model import make_sequences, chronological_split_index
from air_quality_forecasting.forecast import recursive_forecast
from air_quality_forecasting.metrics import evaluate

def test_iso_timestamps_after_day_12_are_not_discarded():
    parsed = parse_timestamps(pd.Series([
        "2025-12-13T00:00:00.000000+0000",
        "2025-12-31T23:45:00.000000+0000",
    ]))
    assert parsed.notna().all()
    assert parsed.iloc[0].day == 13
    assert parsed.iloc[1].day == 31


def test_day_first_slash_dates_remain_supported():
    parsed = parse_timestamps(pd.Series(["31/12/2025", "13/12/2025"]))
    assert parsed.notna().all()
    assert parsed.iloc[0].month == 12 and parsed.iloc[0].day == 31


def test_weekly_imputation_then_mean_fallback():
    s = pd.Series([1.0] * 170)
    s.iloc[168] = np.nan
    s.iloc[3] = np.nan
    result = impute_weekly(s)
    assert result.iloc[168] == 1.0
    assert result.iloc[3] == 1.0
    assert result.isna().sum() == 0

def test_mean_fallback_uses_original_observed_values():
    result = impute_weekly(pd.Series([np.nan, np.nan, 2.0, 8.0, np.nan, np.nan]), lag=2)
    assert result.iloc[0] == 5.0
    assert result.iloc[1] == 5.0
    assert result.iloc[4] == 2.0
    assert result.iloc[5] == 8.0

def test_chronological_67_33_split_boundaries():
    values = np.arange(300, dtype=np.float32).reshape(-1, 1)
    split = chronological_split_index(len(values))
    x, y = make_sequences(values, 24)
    target_times = np.arange(24, len(values))
    train_targets = y[target_times < split]
    test_targets = y[target_times >= split]
    assert train_targets[-1] == split - 1
    assert test_targets[0] == split
    assert len(test_targets) == len(values) - split

class IncrementModel:
    def predict(self, x, verbose=0):
        return np.array([[x[0, -1, 0] + 1]])

def test_recursive_forecast_feeds_predictions_forward():
    result = recursive_forecast(IncrementModel(), np.arange(5), steps=3, lookback=2)
    np.testing.assert_allclose(result, [5, 6, 7])

def test_metrics_rmse_and_percentage_rmse():
    scores = evaluate([2, 4], [1, 2])
    assert scores["rmse"] == pytest.approx(np.sqrt(2.5))
    assert scores["percentage_rmse"] == pytest.approx(100 * np.sqrt(2.5) / 4)
    assert scores["mae"] == pytest.approx(1.5)
    assert scores["r2"] == pytest.approx(-1.5)

def test_percentage_rmse_defined_for_zero_maximum():
    assert np.isnan(evaluate([0, 0], [1, 1])["percentage_rmse"])


def test_current_estimate_loader_excludes_future_and_maps_pollutants(monkeypatch):
    from air_quality_forecasting import data

    latest_completed = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
    times = pd.date_range(latest_completed - pd.Timedelta(hours=80), periods=105, freq="h")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"hourly": {
                "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                "pm2_5": [10.0] * len(times),
                "pm10": [20.0] * len(times),
            }}

    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: FakeResponse())
    frame = data.load_current_estimates()
    assert "PM2.5 (µg/m³)" in frame
    assert "PM10 (µg/m³)" in frame
    assert frame["Timestamp"].max() <= latest_completed
    assert frame["Timestamp"].min() >= latest_completed - pd.Timedelta(days=92)
    assert len(frame) >= 72


def test_current_estimate_loader_reports_short_history(monkeypatch):
    from air_quality_forecasting import data

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            times = pd.date_range(pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=5), periods=6, freq="h")
            return {"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times], "pm2_5": [4.0] * len(times)}}

    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: FakeResponse())
    with pytest.raises(ValueError, match="only .* completed hourly rows"):
        data.load_current_estimates()
