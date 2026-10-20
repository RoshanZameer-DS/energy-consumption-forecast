"""Calendar features for hourly load. `year` is deliberately excluded: a tree model cannot extrapolate
to years it has not seen, and including it made the test error worse (see README)."""
import pandas as pd

FEATURES = ["hour", "dayofweek", "month", "quarter", "is_weekend", "is_summer", "is_winter"]


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    out["hour"] = index.hour
    out["dayofweek"] = index.dayofweek
    out["month"] = index.month
    out["quarter"] = index.quarter
    out["is_weekend"] = index.dayofweek >= 5
    out["is_summer"] = index.month.isin([6, 7, 8])
    out["is_winter"] = index.month.isin([12, 1, 2])
    return out[FEATURES]
