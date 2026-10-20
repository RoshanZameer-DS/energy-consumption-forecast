"""Streamlit dashboard: explore hourly PJM West load and see a 30-day calendar-based forecast."""
import sys
from datetime import timedelta
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURES, calendar_features  # noqa: E402

st.set_page_config(page_title="Energy Consumption Forecast", page_icon="⚡", layout="wide")


@st.cache_resource
def load_assets():
    model_path, data_path = ROOT / "models/model.joblib", ROOT / "data/data_sample.csv"
    if not (model_path.exists() and data_path.exists()):
        return None, None
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    return joblib.load(model_path), data


model, df = load_assets()
if model is None:
    st.error("Model or data sample not found. Run  `python src/train.py`  first.")
    st.stop()

page = st.sidebar.radio("Page", ["Overview", "Exploratory analysis", "30-day forecast"])

if page == "Overview":
    st.title("⚡ Hourly Energy Consumption Forecast")
    st.markdown(
        """
Hourly electricity load for **PJM West** (2002-2018, 143,206 readings) and a model that predicts it from the calendar
(hour, weekday, month, season).

**Honest result:** on the held-out 2018 data the Random Forest scores MAE 511 MW (about 8.7% of the mean load),
exactly the same as a plain "average for this month / weekday / hour" lookup. Load here is driven by the
calendar, so the forecast below is a seasonal profile, not a weather-aware prediction.
        """
    )

elif page == "Exploratory analysis":
    st.title("📊 Exploratory analysis")
    st.caption("Last 5,000 hourly readings")
    col1, col2 = st.columns(2)
    hourly = df.groupby(df.index.hour)["PJMW_MW"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.lineplot(x=hourly.index, y=hourly.values, marker="o", ax=ax)
    ax.set(xlabel="Hour of day", ylabel="Average MW", title="Daily load shape")
    col1.pyplot(fig)
    weekly = df.groupby(df.index.dayofweek)["PJMW_MW"].mean()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.barplot(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=weekly.values, ax=ax, color="steelblue")
    ax.set(ylabel="Average MW", title="Weekday vs weekend")
    col2.pyplot(fig)
    st.info("Load is lowest in the early morning, peaks in the afternoon and evening, and is lower at weekends.")

else:
    st.title("📈 30-day forecast")
    future_index = pd.date_range(df.index.max() + timedelta(hours=1), periods=30 * 24, freq="h")
    forecast = pd.Series(model.predict(calendar_features(future_index)[FEATURES]), index=future_index)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    recent = df["PJMW_MW"].tail(24 * 7)
    ax.plot(recent.index, recent.values, label="Last 7 days (actual)")
    ax.plot(forecast.index, forecast.values, "--", label="Next 30 days (forecast)")
    ax.set(ylabel="Load (MW)")
    ax.legend()
    st.pyplot(fig)
    st.caption("The model only sees calendar information, so this is the expected shape of demand, not a weather-driven forecast.")
    st.dataframe(forecast.head(24).rename("Forecast (MW)").round(0))
