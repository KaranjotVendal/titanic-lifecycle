"""Retraining — refit the champion configuration on refreshed data.

Every retrain is logged to MLflow and registered as a new version of the
"titanic-survival-champion" model, so v1 and v2 are comparable and
roll-back-able.
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from titanic.train import train_champion

REGISTERED_NAME = "titanic-survival-champion"
EXPERIMENT = "titanic-baselines"


def retrain(raw_csv, out_path=None, model_path=None) -> str:
    """Refit on raw_csv, log the run, register the new model version.

    Returns the registered version string (e.g. "titanic-survival-champion/2").
    """
    project_root = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri("sqlite:///" + str(project_root / "mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)

    out_path = out_path or (project_root / "models" / "champion_retrained.joblib")
    artifact = train_champion(raw_csv, out_path)
    model = joblib.load(artifact)

    with mlflow.start_run(run_name="retrain_v_next") as run:
        mlflow.log_param("trigger", os.environ.get("RETRAIN_TRIGGER", "manual"))
        mlflow.log_param("source_csv", str(raw_csv))
        mlflow.sklearn.log_model(
            model, artifact_path="model", serialization_format="cloudpickle"
        )
        registered = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", REGISTERED_NAME
        )

    print(f"retrained -> {artifact}")
    print(f"registered: {REGISTERED_NAME} v{registered.version}")
    return f"{REGISTERED_NAME}/{registered.version}"


if __name__ == "__main__":
    import sys

    retrain(sys.argv[1] if len(sys.argv) > 1 else "data/raw/train.csv")
