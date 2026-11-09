"""Streamlit dashboard for the 2019 comparative-regression adaptation."""

import pandas as pd
import plotly.express as px
import streamlit as st

from air_quality import (
    CITY_COORDINATES,
    MAX_FORECAST_HOURS,
    fetch_hourly,
    forecast,
    train_city,
)

st.set_page_config(
    page_title="Air quality regression comparison",
    page_icon="🌫️",
    layout="wide",
)
st.title("Hourly PM₂.₅ prediction")
st.caption(
    "Ameer et al. (2019) model comparison · Decision Tree · Random Forest · "
    "MLP · Gradient Boosting"
)
st.info(
    "This dashboard uses Open-Meteo/CAMS gridded model estimates, not local "
    "monitor readings. It adapts the paper's workflow; it does not reproduce "
    "the paper's 2010–2015 experiment or published scores."
)

@st.cache_data(ttl=1800, show_spinner=False)
def get_hourly(city: str) -> pd.DataFrame:
    return fetch_hourly(city)

with st.sidebar:
    st.header("Forecast settings")
    city = st.selectbox(
        "City",
        options=sorted(CITY_COORDINATES),
        index=0,
        format_func=lambda name: name.title(),
    )
    horizon = st.slider(
        "Forecast horizon (hours)",
        min_value=1,
        max_value=MAX_FORECAST_HOURS,
        value=24,
    )
    if st.button("Refresh API data"):
        get_hourly.clear()
    run_button = st.button("Train models and forecast", type="primary")

if run_button:
    try:
        with st.spinner("Fetching recent data and comparing four regression models…"):
            data = get_hourly(city)
            now = pd.Timestamp.now(tz=CITY_COORDINATES[city][2])
            historical = data.loc[
                (data["time"] <= now) & data["pm25"].notna()
            ].copy()
            trained = train_city(historical, city)
            future = forecast(trained, data, horizon)
            st.session_state["regression_result"] = {
                "city": city,
                "scores": trained.scores,
                "forecast": future,
                "training_rows": trained.training_rows,
                "model_name": trained.model_name,
                "data_end": historical["time"].max(),
            }
    except Exception as exc:
        st.error(f"Could not train or forecast: {exc}")

result = st.session_state.get("regression_result")
if result is None:
    st.write(
        "Choose a city and horizon, then click **Train models and forecast**. "
        "The app evaluates the models on a chronological holdout and charts "
        "the selected model's forecast."
    )
    st.stop()

scores = result["scores"].copy()
predictions = result["forecast"].copy()
scores["model_label"] = scores["model"].str.replace("_", " ").str.title()

st.subheader(f"Results for {result['city'].title()}")
m1, m2, m3 = st.columns(3)
m1.metric("Selected model", result["model_name"].replace("_", " ").title())
m2.metric("Training samples", f"{result['training_rows']:,}")
m3.metric("Forecast hours", f"{len(predictions):,}")
st.caption(
    f"Historical input through {pd.Timestamp(result['data_end']).strftime('%Y-%m-%d %H:%M %Z')}. "
    "Metrics are from the final 20% of the recent hourly series."
)

st.subheader("Model comparison")
left, right = st.columns(2)
with left:
    mae_chart = px.bar(
        scores,
        x="model_label",
        y="mae",
        color="model_label",
        title="Validation MAE (lower is better)",
        labels={"model_label": "Model", "mae": "MAE (µg/m³)"},
        text_auto=".2f",
    )
    mae_chart.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(mae_chart, width="stretch")
with right:
    rmse_chart = px.bar(
        scores,
        x="model_label",
        y="rmse",
        color="model_label",
        title="Validation RMSE (lower is better)",
        labels={"model_label": "Model", "rmse": "RMSE (µg/m³)"},
        text_auto=".2f",
    )
    rmse_chart.update_layout(showlegend=False, xaxis_title=None)
    st.plotly_chart(rmse_chart, width="stretch")

r2_chart = px.bar(
    scores,
    x="model_label",
    y="r2",
    color="model_label",
    title="Validation R²",
    labels={"model_label": "Model", "r2": "R²"},
    text_auto=".3f",
)
r2_chart.update_layout(showlegend=False, xaxis_title=None)
st.plotly_chart(r2_chart, width="stretch")
st.dataframe(
    scores[["model_label", "mae", "rmse", "r2"]].rename(
        columns={
            "model_label": "Model",
            "mae": "MAE (µg/m³)",
            "rmse": "RMSE (µg/m³)",
            "r2": "R²",
        }
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("PM₂.₅ forecast")
predictions["time"] = pd.to_datetime(predictions["time"])
forecast_chart = px.line(
    predictions,
    x="time",
    y="predicted_pm25",
    markers=True,
    title=f"Recursive {result['city'].title()} forecast · {result['model_name'].replace('_', ' ').title()}",
    labels={"time": "Forecast hour", "predicted_pm25": "PM₂.₅ (µg/m³)"},
)
forecast_chart.update_traces(line={"width": 3})
st.plotly_chart(forecast_chart, width="stretch")
st.dataframe(
    predictions[["time", "predicted_pm25"]].rename(
        columns={"time": "Forecast hour", "predicted_pm25": "PM₂.₅ (µg/m³)"}
    ),
    hide_index=True,
    width="stretch",
)
st.caption(
    "Forecasts after the first hour use the model's previous PM₂.₅ predictions "
    "recursively. Uncertainty increases with the forecast horizon. Open-Meteo "
    "CAMS values and outputs are estimates, not verified station measurements."
)
