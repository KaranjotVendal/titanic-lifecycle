"""Tests for titanic.monitor — the drift -> retrain signal."""

import pandas as pd
import pytest

from titanic.monitor import MONITOR_COLS, check_drift

TRAIN_CSV = "data/raw/train.csv"


def _titanic_frame() -> pd.DataFrame:
    return pd.read_csv(TRAIN_CSV)[MONITOR_COLS].dropna()


@pytest.fixture()
def quiet_csv(tmp_path):
    """A same-distribution draw: real Titanic rows, sampled."""
    p = tmp_path / "quiet.csv"
    _titanic_frame().sample(frac=0.5, random_state=7).to_csv(p, index=False)
    return p


@pytest.fixture()
def drifted_csv(tmp_path):
    """Real Titanic rows with drift planted ONLY in Age and Fare."""
    drifted = _titanic_frame()
    drifted["Age"] = drifted["Age"] + 15
    drifted["Fare"] = drifted["Fare"] * 3
    p = tmp_path / "drifted.csv"
    drifted.to_csv(p, index=False)
    return p


def test_monitor_columns_exist_in_train_schema():
    assert set(MONITOR_COLS) <= set(pd.read_csv(TRAIN_CSV, nrows=1).columns)


def test_quiet_data_produces_no_signal(quiet_csv):
    result = check_drift(TRAIN_CSV, quiet_csv)
    assert result["drift_detected"] is False
    assert result["drifted_columns"] == []


def test_drifted_data_fires_signal(drifted_csv):
    result = check_drift(TRAIN_CSV, drifted_csv)
    assert result["drift_detected"] is True
    assert set(result["drifted_columns"]) == {"Age", "Fare"}


def test_signal_shape_is_stable(quiet_csv):
    result = check_drift(TRAIN_CSV, quiet_csv)
    assert set(result) == {"drift_detected", "drifted_columns"}
    assert isinstance(result["drifted_columns"], list)
