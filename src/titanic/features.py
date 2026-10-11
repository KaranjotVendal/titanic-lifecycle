"""Feature engineering + preprocessing — promoted from notebooks/02.

Everything the model needs to turn a raw passenger record into a
model-ready vector lives here, exactly as the training notebooks defined it.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# columns the preprocessor expects (engineered frame, target excluded)
NUMERIC = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "HasCabin",
]
CATEGORICAL = ["Sex", "Embarked", "Title"]

RARE_TITLES = {
    "Dr",
    "Rev",
    "Col",
    "Major",
    "Capt",
    "Sir",
    "Lady",
    "Don",
    "Dona",
    "Countess",
    "Jonkheer",
    "Mme",
    "Ms",
    "Mlle",
}


def extract_title(name: str) -> str:
    """Pull the title out of 'Braund, Mr. Owen Harris' -> 'Mr'.

    Falls back to 'Unknown' when the name doesn't have the expected shape.
    """
    try:
        after_comma = name.split(",", 1)[1]
        title = after_comma.split(".", 1)[0].strip()
        return title if title.isalpha() else "Unknown"
    except IndexError:
        return "Unknown"


def add_features(df):
    """Raw passenger data -> engineered features. EDA findings, as code.

    Adds Title, FamilySize, IsAlone, HasCabin; drops identifier/high-cardinality text.
    """
    df = df.copy()
    df["Title"] = df["Name"].map(extract_title).replace(list(RARE_TITLES), "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # Cabin (and the dropped text columns) may be absent from partial payloads
    df["HasCabin"] = df["Cabin"].notna().astype(int) if "Cabin" in df.columns else 0
    return df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")


def build_preprocessor() -> ColumnTransformer:
    """The preprocessing half of the champion pipeline (impute + one-hot).

    Fit on training data only; then transform anything with the same rules.
    """
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ]
    )


def feature_columns() -> list[str]:
    """Column order the preprocessor expects."""
    return NUMERIC + CATEGORICAL
