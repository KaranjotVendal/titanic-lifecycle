"""Drift monitoring — is the incoming data still the data we trained on?

check_drift() wraps the Evidently report validated in notebooks/04 and
returns a plain dict a pipeline (or human) can act on.
"""

import warnings

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

# the schema columns we monitor (must exist in both reference and current CSVs)
MONITOR_COLS = ["Age", "Fare", "Pclass", "Sex"]


def check_drift(reference_csv, current_csv, columns=MONITOR_COLS) -> dict:
    """Compare current data against the training reference.

    Returns {"drift_detected": bool, "drifted_columns": [...]}.
    """
    ref = pd.read_csv(reference_csv)[list(columns)].dropna()
    cur = pd.read_csv(current_csv)[list(columns)].dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)

    drifted_columns = []
    for metric in report.as_dict()["metrics"]:
        result = metric.get("result", {})
        if "drift_by_columns" in result:
            for name, col in result["drift_by_columns"].items():
                if col["drift_detected"]:
                    drifted_columns.append(name)

    return {
        "drift_detected": len(drifted_columns) > 0,
        "drifted_columns": sorted(drifted_columns),
    }


if __name__ == "__main__":
    import json
    import sys

    signal = check_drift(
        sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv",
        sys.argv[2] if len(sys.argv) > 2 else "data/processed/drifted.csv",
    )
    print(json.dumps(signal))
    sys.exit(1 if signal["drift_detected"] else 0)
