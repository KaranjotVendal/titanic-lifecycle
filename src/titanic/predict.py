"""Champion inference — load the artifact, predict one passenger.

This is the production surface: the API (serve.py) calls predict_one().
"""

from pathlib import Path

import joblib
import pandas as pd

from titanic.features import CATEGORICAL, NUMERIC, add_features

# every column a raw passenger record may carry (mirrors train.csv/test.csv)
RAW_COLUMNS = {
    "PassengerId",
    "Survived",
    "Name",
    "Ticket",
    "Cabin",
    *NUMERIC,
    *CATEGORICAL,
}


def load_champion(path) -> object:
    """Load a saved pipeline artifact (joblib)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"champion artifact not found: {path}")
    return joblib.load(path)


def predict_one(model, passenger: dict) -> dict:
    """Predict survival for one raw passenger dict.

    Returns {"survived": bool, "probability": float}.
    Raises ValueError on unexpected input columns.
    """
    unknown = set(passenger) - RAW_COLUMNS
    if unknown:
        raise ValueError(f"unexpected passenger fields: {sorted(unknown)}")

    engineered = add_features(pd.DataFrame([passenger]))
    X = engineered[NUMERIC + CATEGORICAL]

    probability = float(model.predict_proba(X)[0, 1])
    survived = bool(probability >= 0.5)
    return {"survived": survived, "probability": round(probability, 4)}
