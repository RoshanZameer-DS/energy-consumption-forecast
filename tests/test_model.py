import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import train  # noqa: E402
from features import FEATURES, calendar_features  # noqa: E402


def test_calendar_features_have_no_year():
    out = calendar_features(pd.date_range("2018-07-04 17:00", periods=3, freq="h"))
    assert list(out.columns) == FEATURES and "year" not in out.columns


def test_model_is_not_worse_than_seasonal_baseline():
    r = train.main()
    assert r["random_forest"]["MAE"] <= r["seasonal_average_baseline"]["MAE"] * 1.02


def test_app_runs_on_every_page():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()
    for page in ["Overview", "Exploratory analysis", "30-day forecast"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
