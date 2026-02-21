import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import os

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set page config
st.set_page_config(page_title="Energy Consumption Forecasting", layout="wide")

# Load model and data
@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'model.joblib')
    data_path = os.path.join(BASE_DIR, 'data_sample.csv')
    
    model = joblib.load(model_path)
    data = pd.read_csv(data_path, index_col=0)
    data.index = pd.to_datetime(data.index)
    return model, data

try:
    model, df = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Exploratory Data Analysis", "Forecast Consumption"])

if page == "Project Overview":
    st.title("⚡ Energy Consumption Forecasting Analysis")
    st.markdown("""
    ### Objective
    The goal of this project is to analyze the historical hourly energy consumption of the **PJMW (PJM West)** region and build a predictive model to forecast future demand.
    
    ### Dataset
    - **Source**: PJMW_MW_Hourly.xlsx
    - **Records**: 143,206 hourly observations.
    - **Features**: Datetime, MW (Power Consumption).
    
    ### Methodology
    1. **Data Cleaning**: Handling missing values and ensuring datetime consistency.
    2. **Feature Engineering**: Extracting hour, day of week, month, and seasonal indicators.
    3. **Modeling**: Using a Random Forest Regressor to capture non-linear patterns.
    """)

elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    st.write("Analyzing patterns in energy consumption across different time scales.")
    
    # 1. Hourly Pattern
    st.subheader("Hourly Consumption Pattern")
    hourly_avg = df.groupby(df.index.hour)['PJMW_MW'].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, ax=ax, marker='o')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average MW")
    st.pyplot(fig)
    st.info("Insights: Energy consumption typically peaks twice daily—once in the morning (~8 AM) and again in the evening (~6-8 PM).")

    # 2. Weekly Pattern
    st.subheader("Weekly Consumption Pattern")
    weekly_avg = df.groupby(df.index.dayofweek)['PJMW_MW'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=days, y=weekly_avg.values, ax=ax, palette="viridis")
    ax.set_ylabel("Average MW")
    st.pyplot(fig)
    st.info("Insights: Consumption is significantly higher during weekdays compared to weekends, reflecting industrial activity.")

elif page == "Forecast Consumption":
    st.title("📈 30-Day Energy Forecast")
    
    st.write("Generating future energy demand based on the trained Random Forest model.")
    
    # Generate future structure
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + timedelta(hours=1), periods=30*24, freq='h')
    future_df = pd.DataFrame(index=future_dates)
    
    future_df['hour'] = future_df.index.hour
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['month'] = future_df.index.month
    future_df['year'] = future_df.index.year
    future_df['quarter'] = future_df.index.quarter
    future_df['is_weekend'] = (future_df.index.dayofweek == 5) | (future_df.index.dayofweek == 6)
    future_df['is_summer'] = (future_df.index.month >= 6) & (future_df.index.month <= 8)
    future_df['is_winter'] = (future_df.index.month == 12) | (future_df.index.month == 1) | (future_df.index.month == 2)
    
    features = ['hour', 'dayofweek', 'month', 'year', 'quarter', 'is_weekend', 'is_summer', 'is_winter']
    forecast = model.predict(future_df[features])
    future_df['Forecast'] = forecast
    
    # Visualization
    st.subheader("30-Day Forecast Visualization")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot last 7 days of historical and 30 days of forecast
    past_7_days = df.tail(24*7)
    sns.lineplot(x=past_7_days.index, y=past_7_days['PJMW_MW'], label='Historical (Last 7 Days)', ax=ax, color='blue')
    sns.lineplot(x=future_df.index, y=future_df['Forecast'], label='30-Day Forecast', ax=ax, color='green', linestyle='--')
    
    ax.set_ylabel("Energy Consumption (MW)")
    ax.legend()
    st.pyplot(fig)
    
    st.success("Forecast generated successfully for 30 days ahead.")
    st.write(future_df[['Forecast']].head(24))
