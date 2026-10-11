"""Tests for titanic.features — the promoted notebook-02 code."""

import pandas as pd

from titanic.features import (
    CATEGORICAL,
    NUMERIC,
    add_features,
    build_preprocessor,
    extract_title,
)

RAW_ROW = {
    "PassengerId": 1,
    "Survived": 0,
    "Pclass": 3,
    "Name": "Braund, Mr. Owen Harris",
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "A/5 21171",
    "Fare": 7.25,
    "Cabin": None,
    "Embarked": "S",
}


def test_extract_title_basic():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"


def test_extract_title_malformed():
    assert extract_title("NoCommaHere") == "Unknown"
    assert extract_title("Doe, . John") == "Unknown"


def test_extract_title_rare_titles():
    assert extract_title("Smith, Rev. John") == "Rev"
    assert extract_title("Peel, Mrs. Jacques Heath") == "Mrs"


def test_add_features_creates_expected_columns():
    out = add_features(pd.DataFrame([RAW_ROW]))
    assert out.loc[0, "Title"] == "Mr"
    assert out.loc[0, "FamilySize"] == 2
    assert out.loc[0, "IsAlone"] == 0
    assert out.loc[0, "HasCabin"] == 0
    # raw/dropped columns gone
    for col in ("PassengerId", "Name", "Ticket", "Cabin"):
        assert col not in out.columns


def test_add_features_rare_title_grouping():
    row = dict(RAW_ROW, Name="Smith, Rev. John")
    assert add_features(pd.DataFrame([row])).loc[0, "Title"] == "Rare"


def test_add_features_is_alone_and_cabin():
    row = dict(RAW_ROW, SibSp=0, Parch=0, Cabin="C85")
    out = add_features(pd.DataFrame([row]))
    assert out.loc[0, "IsAlone"] == 1
    assert out.loc[0, "HasCabin"] == 1


def test_build_preprocessor_transforms_single_row():
    fe = add_features(pd.DataFrame([RAW_ROW]))
    X = fe[NUMERIC + CATEGORICAL]
    pre = build_preprocessor()
    vec = pre.fit_transform(X)
    assert vec.shape[0] == 1
    assert vec.shape[1] == len(pre.get_feature_names_out())


def test_preprocessor_handles_missing_values():
    row = dict(RAW_ROW, Age=float("nan"), Fare=float("nan"), Embarked=None)
    fe = add_features(pd.DataFrame([row]))
    pre = build_preprocessor()
    vec = pre.fit_transform(fe[NUMERIC + CATEGORICAL])
    assert vec.shape[0] == 1  # no crash; imputation applied
