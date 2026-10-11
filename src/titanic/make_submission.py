"""Kaggle submission pipeline — champion predictions, format-validated.

The #1 beginner failure mode is a malformed submission CSV. This module
hard-fails with a clear message BEFORE the file can be uploaded.
"""

from pathlib import Path

import joblib
import pandas as pd

from titanic.features import CATEGORICAL, NUMERIC, add_features

SUBMISSION_COLUMNS = ["PassengerId", "Survived"]


def validate_submission(sub: pd.DataFrame, expected_rows: int) -> None:
    """Kaggle's contract: exact columns, exact row count, binary predictions."""
    if list(sub.columns) != SUBMISSION_COLUMNS:
        raise ValueError(
            f"submission columns must be {SUBMISSION_COLUMNS}, got {list(sub.columns)}"
        )
    if len(sub) != expected_rows:
        raise ValueError(
            f"submission must have {expected_rows} rows (one per test passenger), got {len(sub)}"
        )
    if not sub["PassengerId"].is_unique:
        raise ValueError("PassengerId must be unique")
    if not set(sub["Survived"].unique()) <= {0, 1}:
        raise ValueError("Survived must be binary (0 or 1)")


def make_submission(test_csv, model_path, out_path) -> Path:
    """Predict all test passengers with the champion and write a validated CSV."""
    test = pd.read_csv(test_csv)
    model = joblib.load(model_path)

    engineered = add_features(test)
    predictions = model.predict(engineered[NUMERIC + CATEGORICAL]).astype(int)

    sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
    validate_submission(sub, expected_rows=len(test))

    out_path = Path(out_path)
    sub.to_csv(out_path, index=False)
    print(f"submission written: {out_path.resolve()}  ({len(sub)} rows)")
    return out_path


if __name__ == "__main__":
    from titanic.train import CHAMPION_PARAMS  # noqa: F401  (documents the config)

    root = Path(__file__).resolve().parents[2]
    make_submission(
        root / "data" / "raw" / "test.csv",
        root / "models" / "champion.joblib",
        root / "submission.csv",
    )
