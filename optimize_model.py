import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load data
print("Loading data...")
df = pd.read_excel('PJMW_MW_Hourly.xlsx')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Feature Engineering
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['year'] = df.index.year
df['quarter'] = df.index.quarter
df['is_weekend'] = (df.index.dayofweek == 5) | (df.index.dayofweek == 6)
df['is_summer'] = (df.index.month >= 6) & (df.index.month <= 8)
df['is_winter'] = (df.index.month == 12) | (df.index.month == 1) | (df.index.month == 2)

# Train Model with reduced complexity for smaller file size
print("Training optimized model...")
features = ['hour', 'dayofweek', 'month', 'year', 'quarter', 'is_weekend', 'is_summer', 'is_winter']
X = df[features]
y = df['PJMW_MW']

# Reduced n_estimators and max_depth to shrink file size significantly
model = RandomForestRegressor(n_estimators=30, max_depth=15, random_state=42)
model.fit(X, y)

# Export Model with compression
print("Exporting compressed model...")
joblib.dump(model, 'model.joblib', compress=3)

size_mb = os.path.getsize('model.joblib') / (1024 * 1024)
print(f"Optimized model size: {size_mb:.2f} MB")

# Export Data Sample
df.tail(5000).to_csv('data_sample.csv')
print("Assets updated successfully.")
