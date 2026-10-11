"""FastAPI serving layer — HTTP front door for the champion.

Run locally:  uv run uvicorn titanic.serve:app --reload
The champion artifact loads ONCE at startup (lifespan), then every request
reuses it — load once, predict many.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from titanic.predict import load_champion, predict_one

MODEL_PATH = Path(os.environ.get("TITANIC_MODEL_PATH", "models/champion.joblib"))


class Passenger(BaseModel):
    """A raw passenger record, as the API receives it.

    Only Pclass/Sex/Name/SibSp/Parch are required; everything else may be
    absent or null (the pipeline imputes). Extra fields are rejected (422).
    """

    model_config = ConfigDict(extra="forbid")

    Pclass: int
    Sex: str
    Name: str
    SibSp: int
    Parch: int
    Age: float | None = None
    Fare: float | None = None
    Cabin: str | None = None
    Embarked: str | None = None
    Ticket: str | None = None
    PassengerId: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_champion(MODEL_PATH)
    yield


app = FastAPI(title="Titanic Survival API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}


@app.post("/predict")
def predict(passenger: Passenger, request: Request) -> dict:
    return predict_one(request.app.state.model, passenger.model_dump())
