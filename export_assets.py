import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
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

# Train Model (using same parameters as notebook)
features = ['hour', 'dayofweek', 'month', 'year', 'quarter', 'is_weekend', 'is_summer', 'is_winter']
X = df[features]
y = df['PJMW_MW']

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Export Model
joblib.dump(model, 'model.joblib')

# Export Data Sample (last 5000 rows for EDA demo)
df.tail(5000).to_csv('data_sample.csv')

print("Model and Data Sample exported successfully.")
