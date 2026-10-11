"""Training entrypoint — reproduce the champion artifact from raw data.

Wednesday's winning configuration, frozen as code:
gradient boosting, lr=0.1, max_depth=2, n_estimators=200.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from titanic.features import CATEGORICAL, NUMERIC, add_features, build_preprocessor

CHAMPION_PARAMS = {"learning_rate": 0.1, "max_depth": 2, "n_estimators": 200}
RANDOM_STATE = 42


def train_champion(raw_csv, out_path, target: str = "Survived") -> Path:
    """Train the champion pipeline on a raw train.csv and save the artifact."""
    df = add_features(pd.read_csv(raw_csv))
    X = df[NUMERIC + CATEGORICAL]
    y = df[target]

    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE, **CHAMPION_PARAMS
                ),
            ),
        ]
    )
    pipe.fit(X, y)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_path)
    return out_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    out = train_champion(
        root / "data" / "raw" / "train.csv", root / "models" / "champion.joblib"
    )
    print(f"champion trained -> {out}")
