"""Train the hourly-load model, compare it with a simple seasonal-average baseline, save everything.

    python src/train.py

Test set = every row from 2018 onwards (Jan-Aug 2018); training = 2002-2017.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import FEATURES, calendar_features

ROOT = Path(__file__).resolve().parents[1]
TARGET = "PJMW_MW"


def load() -> pd.DataFrame:
    df = pd.read_excel(ROOT / "data/PJMW_MW_Hourly.xlsx")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()
    return df[~df.index.duplicated()]  # 4 duplicated timestamps (clock changes)


def scores(y, pred) -> dict:
    return {"MAE": round(float(mean_absolute_error(y, pred)), 1),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 1)}


def main() -> dict:
    df = load()
    X = calendar_features(df.index)
    train, test = df.index.year < 2018, df.index.year >= 2018

    # baseline: average load for the same month, weekday and hour in the training years
    profile = df[train].groupby([X[train].month, X[train].dayofweek, X[train].hour])[TARGET].mean()
    keys = pd.MultiIndex.from_arrays([X[test].month, X[test].dayofweek, X[test].hour])
    baseline = profile.reindex(keys).to_numpy()

    model = RandomForestRegressor(n_estimators=60, max_depth=18, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X[train][FEATURES], df.loc[train, TARGET])
    pred = model.predict(X[test][FEATURES])

    results = {
        "train_rows": int(train.sum()), "test_rows": int(test.sum()),
        "test_period": [str(df.index[test].min().date()), str(df.index[test].max().date())],
        "mean_load_mw_in_test": round(float(df.loc[test, TARGET].mean()), 0),
        "seasonal_average_baseline": scores(df.loc[test, TARGET], baseline),
        "random_forest": scores(df.loc[test, TARGET], pred),
    }
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(model, ROOT / "models/model.joblib", compress=3)
    df.tail(5000).to_csv(ROOT / "data/data_sample.csv")  # recent history shown by the app
    (ROOT / "models/metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
