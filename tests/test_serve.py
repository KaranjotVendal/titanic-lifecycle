"""Contract tests for the FastAPI serving layer."""


import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from titanic.features import CATEGORICAL, NUMERIC, add_features, build_preprocessor

VALID_PASSENGER = {
    "Pclass": 1,
    "Name": "Cumings, Mrs. John Bradley",
    "Sex": "female",
    "Age": 38.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 71.2833,
    "Embarked": "C",
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A test API backed by a tiny locally-trained artifact (never models/)."""
    rows = [
        {**VALID_PASSENGER, "Survived": 1},
        {**VALID_PASSENGER, "Survived": 0, "Sex": "male", "Age": 60.0, "Pclass": 3, "Fare": 7.0},
        {**VALID_PASSENGER, "Survived": 1, "Sex": "female", "Age": 5.0},
        {**VALID_PASSENGER, "Survived": 0, "Sex": "male", "Age": 55.0},
    ]
    df = add_features(pd.DataFrame(rows))
    pipe = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]
    )
    pipe.fit(df[NUMERIC + CATEGORICAL], df["Survived"])
    artifact = tmp_path / "champion.joblib"
    joblib.dump(pipe, artifact)

    monkeypatch.setenv("TITANIC_MODEL_PATH", str(artifact))

    from titanic.serve import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_passenger(client):
    resp = client.post("/predict", json=VALID_PASSENGER)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"survived", "probability"}
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["survived"], bool)


def test_predict_missing_required_field_is_422(client):
    payload = {k: v for k, v in VALID_PASSENGER.items() if k != "Sex"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_predict_unknown_field_is_422(client):
    resp = client.post("/predict", json={**VALID_PASSENGER, "hacker_field": "x"})
    assert resp.status_code == 422


def test_predict_survives_missing_optionals(client):
    sparse = {"Pclass": 3, "Sex": "male", "Name": "Doe, Mr. John", "SibSp": 0, "Parch": 0}
    resp = client.post("/predict", json=sparse)
    assert resp.status_code == 200
    assert 0.0 <= resp.json()["probability"] <= 1.0
