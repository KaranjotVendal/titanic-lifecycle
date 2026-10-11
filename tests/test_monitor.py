"""Tests for titanic.monitor — the drift -> retrain signal.

Fully self-contained: the reference and current CSVs are built from the same
synthetic generator, so the suite needs no repo data files (CI-safe).
"""

import numpy as np
import pandas as pd
import pytest

from titanic.monitor import MONITOR_COLS, check_drift


def _batch(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Age": rng.normal(30, 10, n).clip(0, 80),
            "Fare": rng.lognormal(3, 0.8, n),
            "Pclass": rng.choice([1, 2, 3], n),
            "Sex": rng.choice(["male", "female"], n),
        }
    )[MONITOR_COLS]


@pytest.fixture()
def reference_csv(tmp_path):
    p = tmp_path / "reference.csv"
    _batch(400, seed=7).to_csv(p, index=False)
    return p


@pytest.fixture()
def quiet_csv(tmp_path):
    """A same-distribution draw from a different seed: must NOT drift."""
    p = tmp_path / "quiet.csv"
    _batch(400, seed=99).to_csv(p, index=False)
    return p


@pytest.fixture()
def drifted_csv(tmp_path, reference_csv):
    """Reference batch with drift planted ONLY in Age and Fare."""
    drifted = pd.read_csv(reference_csv)
    drifted["Age"] = drifted["Age"] + 15
    drifted["Fare"] = drifted["Fare"] * 3
    p = tmp_path / "drifted.csv"
    drifted.to_csv(p, index=False)
    return p


def test_monitor_columns_cover_key_schema():
    assert set(MONITOR_COLS) == {"Age", "Fare", "Pclass", "Sex"}


def test_quiet_data_produces_no_signal(reference_csv, quiet_csv):
    result = check_drift(reference_csv, quiet_csv)
    assert result["drift_detected"] is False
    assert result["drifted_columns"] == []


def test_drifted_data_fires_signal(reference_csv, drifted_csv):
    result = check_drift(reference_csv, drifted_csv)
    assert result["drift_detected"] is True
    assert set(result["drifted_columns"]) == {"Age", "Fare"}


def test_signal_shape_is_stable(reference_csv, quiet_csv):
    result = check_drift(reference_csv, quiet_csv)
    assert set(result) == {"drift_detected", "drifted_columns"}
    assert isinstance(result["drifted_columns"], list)
