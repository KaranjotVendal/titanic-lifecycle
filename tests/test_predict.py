"""Tests for titanic.predict — loading the champion and predicting one passenger."""

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from titanic.features import CATEGORICAL, NUMERIC, add_features, build_preprocessor
from titanic.predict import load_champion, predict_one

PASSENGER = {
    "Pclass": 1,
    "Name": "Cumings, Mrs. John Bradley",
    "Sex": "female",
    "Age": 38.0,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "PC 17599",
    "Fare": 71.2833,
    "Cabin": "C85",
    "Embarked": "C",
    "PassengerId": 2,
}


@pytest.fixture()
def champion_path(tmp_path):
    """Train a tiny real pipeline on synthetic data so tests never depend on models/."""
    rng_rows = [
        {**PASSENGER, "Survived": 1},
        {**PASSENGER, "Survived": 1, "Sex": "male", "Age": 60.0},
        {
            **PASSENGER,
            "Survived": 0,
            "Sex": "male",
            "Age": 60.0,
            "Pclass": 3,
            "Fare": 7.0,
        },
        {**PASSENGER, "Survived": 1, "Sex": "female", "Age": 5.0},
        {**PASSENGER, "Survived": 0, "Sex": "male", "Age": 55.0, "Fare": 8.0},
        {**PASSENGER, "Survived": 0, "Sex": "male", "Age": 50.0, "Pclass": 3},
    ]
    df = add_features(pd.DataFrame(rng_rows))
    X = df[NUMERIC + CATEGORICAL]
    y = df["Survived"]
    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X, y)
    path = tmp_path / "champion.joblib"
    joblib.dump(pipe, path)
    return path


def test_load_champion_returns_pipeline(champion_path):
    model = load_champion(champion_path)
    assert hasattr(model, "predict") and hasattr(model, "predict_proba")


def test_predict_one_returns_probability_and_label(champion_path):
    model = load_champion(champion_path)
    result = predict_one(model, PASSENGER)
    assert set(result) == {"survived", "probability"}
    assert 0.0 <= result["probability"] <= 1.0
    assert result["survived"] in (True, False)


def test_predict_one_is_deterministic(champion_path):
    model = load_champion(champion_path)
    a = predict_one(model, PASSENGER)
    b = predict_one(model, PASSENGER)
    assert a == b


def test_predict_one_handles_missing_optional_fields(champion_path):
    model = load_champion(champion_path)
    sparse = {
        "Pclass": 3,
        "Sex": "male",
        "Age": None,
        "SibSp": 0,
        "Parch": 0,
        "Fare": None,
        "Embarked": "S",
        "Name": "Doe, Mr. John",
    }
    result = predict_one(model, sparse)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_one_rejects_unknown_columns(champion_path):
    model = load_champion(champion_path)
    with pytest.raises(ValueError):
        predict_one(model, {**PASSENGER, "hacker_field": "x"})
