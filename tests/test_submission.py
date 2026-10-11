"""Tests for titanic.make_submission — the Kaggle submission file, validated."""

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from titanic.features import CATEGORICAL, NUMERIC, add_features, build_preprocessor
from titanic.make_submission import make_submission, validate_submission

TEST_ROWS = [
    {"PassengerId": 892, "Pclass": 3, "Name": "Kelly, Mr. James", "Sex": "male",
     "Age": 34.5, "SibSp": 0, "Parch": 0, "Ticket": "330911", "Fare": 7.8292,
     "Cabin": None, "Embarked": "Q"},
    {"PassengerId": 893, "Pclass": 3, "Name": "Wilkes, Mrs. James", "Sex": "female",
     "Age": 47.0, "SibSp": 1, "Parch": 0, "Ticket": "363272", "Fare": 7.0,
     "Cabin": None, "Embarked": "S"},
    {"PassengerId": 894, "Pclass": 2, "Name": "Myles, Mr. Thomas", "Sex": "male",
     "Age": 62.0, "SibSp": 0, "Parch": 0, "Ticket": "240276", "Fare": 9.6875,
     "Cabin": None, "Embarked": "Q"},
]


@pytest.fixture()
def test_csv(tmp_path):
    p = tmp_path / "test.csv"
    pd.DataFrame(TEST_ROWS).to_csv(p, index=False)
    return p


@pytest.fixture()
def model_path(tmp_path):
    rows = [{**r, "Survived": 0 if r["Sex"] == "male" else 1} for r in TEST_ROWS]
    df = add_features(pd.DataFrame(rows))
    pipe = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))]
    )
    pipe.fit(df[NUMERIC + CATEGORICAL], df["Survived"])
    p = tmp_path / "champion.joblib"
    joblib.dump(pipe, p)
    return p


def test_make_submission_format(test_csv, model_path, tmp_path):
    out = tmp_path / "submission.csv"
    make_submission(test_csv, model_path, out)

    sub = pd.read_csv(out)
    assert list(sub.columns) == ["PassengerId", "Survived"]
    assert len(sub) == len(TEST_ROWS)
    assert set(sub["Survived"].unique()) <= {0, 1}
    assert sub["PassengerId"].is_unique


def test_validate_submission_rejects_bad_values():
    bad = pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 7]})
    with pytest.raises(ValueError):
        validate_submission(bad, expected_rows=2)


def test_validate_submission_rejects_wrong_columns():
    bad = pd.DataFrame({"PassengerId": [1, 2], "Prediction": [0, 1]})
    with pytest.raises(ValueError):
        validate_submission(bad, expected_rows=2)


def test_validate_submission_rejects_row_count():
    bad = pd.DataFrame({"PassengerId": [1], "Survived": [1]})
    with pytest.raises(ValueError):
        validate_submission(bad, expected_rows=2)
