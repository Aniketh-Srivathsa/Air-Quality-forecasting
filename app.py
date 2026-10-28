"""Streamlit dashboard for the Belavadi et al. paper-informed reproduction."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta

from air_quality_forecasting.data import DATA_URL, RAW_PATH, load_csv, load_current_estimates
from air_quality_forecasting.preprocessing import prepare_data
from air_quality_forecasting.model import train_model
from air_quality_forecasting.forecast import recursive_forecast
from air_quality_forecasting.metrics import evaluate

st.set_page_config(page_title="Air quality forecasting | LSTM", page_icon="🌫️", layout="wide")
st.title("Hourly air quality forecasting")
st.caption("Paper-informed LSTM reproduction · Belavadi et al. (2020) · 5-hour recursive forecast")
source_mode = st.sidebar.radio(
    "Data source",
    ["Recent CAMS estimates (near Jayanagar)", "Historical Jayanagar CSV (2024–25)"],
    help="CAMS is a gridded atmospheric model product, not a local sensor reading.",
)

@st.cache_data(ttl=1800, show_spinner=False)
def get_current_estimates():
    return load_current_estimates()

if source_mode.startswith("Recent CAMS"):
    st.warning("Current-date mode uses Open-Meteo/CAMS gridded model estimates near Jayanagar, Bengaluru. These are not measurements from the Jayanagar station or a physical sensor node, and spatial resolution may be coarse. Results evaluate against this model-derived history, not ground truth.")
else:
    st.warning("Historical mode uses substitute Jayanagar 5th Block, Bengaluru 2024–25 data, not the paper’s 2019 Amaravati/Bengaluru experimental data. Its forecast ends at the latest timestamp in that CSV, and results are not a numerical reproduction of the paper’s tables.")

@st.cache_data(show_spinner="Loading CSV…")
def get_raw(path: str):
    return load_csv(path)

try:
    if source_mode.startswith("Recent CAMS"):
        if st.sidebar.button("Refresh current estimates"):
            get_current_estimates.clear()
        raw = get_current_estimates()
    else:
        raw = get_raw(str(RAW_PATH))
    initial = prepare_data(raw)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if initial.station_column:
    stations = sorted(raw[initial.station_column].dropna().astype(str).unique())
    selected_station = st.sidebar.selectbox("Station", stations, index=stations.index(initial.station) if initial.station in stations else 0)
else:
    selected_station = None
prepared = prepare_data(raw, selected_station)

left, right = st.columns([2, 1])
with left:
    st.subheader("Dataset preview")
    st.dataframe(raw.head(12), use_container_width=True)
with right:
    st.subheader("Timestamp coverage")
    st.metric("Raw rows", f"{len(raw):,}")
    st.metric("Hourly observations", f"{len(prepared.hourly):,}")
    st.write(f"**From:** {prepared.hourly.index.min()}")
    st.write(f"**To:** {prepared.hourly.index.max()}")
    if source_mode.startswith("Recent CAMS"):
        st.caption("Source: Open-Meteo Air Quality API · CAMS model estimates · [API documentation](https://open-meteo.com/en/docs/air-quality-api)")
        st.caption("All timestamps are UTC. Only completed hours are used; API forecast hours are excluded from model history.")
    else:
        st.caption(f"Source: [OpenCity CSV]({DATA_URL})")

st.subheader("Preprocessing summary")
st.json(prepared.summary)
st.caption("Duplicate timestamps keep the last record; source pollutant readings with absolute Z-score above 3 are removed before hourly averaging. Missing hourly values are filled from 168 hours earlier or the series mean. Weather variables are excluded from model inputs.")

pollutant = st.sidebar.selectbox("Pollutant", prepared.pollutants)
lookback = 24  # Paper omits lookback length; keep the documented 24-hour assumption.
epochs = st.sidebar.slider("Maximum epochs", min_value=1, max_value=50, value=50)
batch_size = st.sidebar.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=32)
patience = st.sidebar.slider("Early-stopping patience", min_value=2, max_value=12, value=5)
st.subheader(f"Hourly series · {pollutant}")
first_date = prepared.hourly.index.min().date()
last_date = prepared.hourly.index.max().date()
default_start = max(first_date, last_date - timedelta(days=29))
selected_dates = st.sidebar.date_input(
    "Series date range", value=(default_start, last_date),
    min_value=first_date, max_value=last_date,
)
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    range_start, range_end = selected_dates
    if range_start and range_end:
        in_range = (prepared.hourly.index.date >= range_start) & (prepared.hourly.index.date <= range_end)
        observed_view = prepared.observed_hourly.loc[in_range, pollutant]
        imputed_view = prepared.hourly.loc[in_range, pollutant].where(observed_view.isna())
        series_fig = go.Figure()
        observed_label = "Hourly CAMS estimate" if source_mode.startswith("Recent CAMS") else "Observed hourly mean"
        series_fig.add_trace(go.Scatter(
            x=observed_view.index, y=observed_view.values, name=observed_label,
            mode="lines", connectgaps=False,
        ))
        series_fig.add_trace(go.Scatter(
            x=imputed_view.index, y=imputed_view.values, name="Imputed values used by model",
            mode="markers", marker={"size": 5, "color": "#f59e0b"},
        ))
        series_fig.update_layout(
            title=f"{range_start:%d %b %Y} – {range_end:%d %b %Y}",
            xaxis_title="Hour", yaxis_title=pollutant, height=330,
            legend_title="Hourly values",
        )
        st.plotly_chart(series_fig, use_container_width=True)
        total_imputed = prepared.summary["hourly_values_imputed"][pollutant]
        value_description = "CAMS model estimates" if source_mode.startswith("Recent CAMS") else "measured hourly means"
        st.caption(
            f"Showing {value_description} and imputed values separately. "
            f"{total_imputed:,} of {len(prepared.hourly):,} hours ({100 * total_imputed / len(prepared.hourly):.1f}%) "
            f"were imputed for this pollutant across the full dataset."
        )
    else:
        st.info("Choose both a start and end date to view the hourly series.")
else:
    st.info("Choose a start and end date to view the hourly series.")

if st.button("Train LSTM and forecast five hours", type="primary"):
    progress = st.progress(0, text="Preparing chronological split…")
    status = st.empty()
    from tensorflow import keras
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            progress.progress(min((epoch + 1) / epochs, 1.0), text=f"Epoch {epoch+1}/{epochs} · loss {logs.get('loss', float('nan')):.5g} · validation {logs.get('val_loss', float('nan')):.5g}")
    try:
        with st.spinner("Training TensorFlow model…"):
            result = train_model(prepared.hourly[pollutant], lookback=int(lookback), max_epochs=epochs, batch_size=int(batch_size), patience=patience, progress_callback=ProgressCallback())
        progress.progress(1.0, text=f"Training complete · {len(result.history.epoch)} epoch(s)")
        status.success(f"Chronological split: {result.split_index:,} training hours and {len(result.actual):,} test targets.")
        scores = evaluate(result.actual, result.predicted)
        m1, m2, m3 = st.columns(3)
        m1.metric("Test RMSE", f"{scores['rmse']:.4g}")
        m2.metric("Percentage RMSE", "undefined (max actual is zero)" if pd.isna(scores["percentage_rmse"]) else f"{scores['percentage_rmse']:.2f}%")
        m3.metric("Test observations", f"{len(result.actual):,}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.test_index, y=result.actual, name="Actual", mode="lines"))
        fig.add_trace(go.Scatter(x=result.test_index, y=result.predicted, name="Predicted", mode="lines"))
        fig.update_layout(title="Test set: actual vs predicted", xaxis_title="Hour", yaxis_title=pollutant, legend_title="Series", height=420)
        st.plotly_chart(fig, use_container_width=True)
        values = recursive_forecast(result.model, prepared.hourly[pollutant].values, steps=5, lookback=int(lookback), scaler=result.scaler)
        future_index = pd.date_range(prepared.hourly.index[-1] + pd.Timedelta(hours=1), periods=5, freq="h")
        forecast = pd.DataFrame({"Timestamp": future_index, pollutant: values})
        st.subheader("Next five hours · recursive forecast")
        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(x=prepared.hourly.index[-48:], y=prepared.hourly[pollutant].iloc[-48:], name="Recent actuals"))
        forecast_fig.add_trace(go.Scatter(x=future_index, y=values, name="Forecast", mode="lines+markers"))
        forecast_fig.update_layout(xaxis_title="Hour", yaxis_title=pollutant, height=360)
        st.plotly_chart(forecast_fig, use_container_width=True)
        st.dataframe(forecast, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Training or forecasting failed: {exc}")

with st.expander("Model architecture and methodology"):
    st.code("Input: (24 hourly observations, 1 pollutant)\nLSTM(64, activation='relu', return_sequences=True)\nLSTM(32, activation='relu')\nDense(1, activation='linear')\nLoss: mean squared error · Optimizer: Adam")
    st.markdown("The paper describes univariate hourly pollutant forecasting; weather is not directly used. This implementation uses a 24-hour lookback because the paper does not publish the sequence length. Training/test data are split in chronological order (67%/33%). Scaling is fit on training observations only. Early stopping uses a tail validation slice from the training sequences; the test set is held back until final evaluation. Five future hours are generated recursively.")
with st.expander("Provenance, assumptions and limitations"):
    st.markdown(f"""
**Historical data:** [Jayanagar 5th Block KSPCB 2024–25 CSV]({DATA_URL}) from OpenCity. It is substitute data and differs in location/time and collection from the paper's 2019 experimental data (Amaravati APPCB and Bengaluru WSN/government observations).

**Current-date source:** Open-Meteo's Air Quality API supplies recent hourly CAMS estimates for approximate Jayanagar coordinates (12.925, 77.583). The model trains on up to 92 days of these recent completed hourly estimates and forecasts five hours beyond the latest returned completed hour. CAMS is atmospheric model output on a gridded domain, not station-level sensor data; resolution can be coarse. Give attribution to [Open-Meteo](https://open-meteo.com/en/docs/air-quality-api) and [CAMS](https://atmosphere.copernicus.eu/). Live-mode RMSE measures agreement with held-out CAMS estimates, not verified ground measurements. A refresh button clears the 30-minute cache. The historical CSV remains available in the source selector.

For live mode, pollutant columns use API units µg/m³; this can differ from the historical CSV (for example, its CO unit). The source selector never mixes the CSV and CAMS values in a single training series.

**Assumptions:** one-hour aggregation by timestamp; duplicate timestamps keep the last row; Z-scores use the available hourly values with population standard deviation; missing values use the same pollutant's value at lag 168, otherwise the series mean; 24-hour input window; 10% tail of training sequences used for validation; early-stopping patience defaults to 5.

**Limitations:** current estimates are model-derived, coordinate-based, and may be spatially coarse; they are not a real-time Jayanagar station sensor feed. API availability, model updates and missing pollutant fields affect coverage. The historical CSV is stale for today's forecast. No Arduino Uno, MQ-2, MQ-135, Sharp dust sensor, GSM link, EC2 ingestion/API, or physical WSN is implemented. Sensor calibration, connectivity, and geographic variation cannot be evaluated. Dataset coverage, pollutants and units differ from the paper. Percentage RMSE is undefined if the maximum actual test value is zero.
""")
