import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("life_expectancy_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

MODEL_PATH = os.getenv("MODEL_PATH", "models/lightgbm_life_expectancy.pkl")


class PredictRequest(BaseModel):
    data: dict[str, float] = Field(..., description="Feature-name to numeric-value mapping")


class PredictResponse(BaseModel):
    prediction: float
    model_name: str = "lightgbm"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class ModelState:
    model: Any | None = None
    feature_columns: list[str] = []
    metadata: dict[str, Any] = {}


state = ModelState()


def _predict_with_model(model: Any, features: pd.DataFrame) -> float:
    if hasattr(model, "best_iteration") and hasattr(model, "predict"):
        preds = model.predict(features, num_iteration=model.best_iteration)
        return float(preds[0])
    preds = model.predict(features)
    return float(preds[0])


def load_model(path: str) -> tuple[Any, list[str], dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    payload = joblib.load(path)
    if "model" not in payload or "features" not in payload:
        raise ValueError("Model payload must contain 'model' and 'features'.")
    return payload["model"], payload["features"], payload.get("metadata", {})


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        model, feature_columns, metadata = load_model(MODEL_PATH)
        state.model = model
        state.feature_columns = feature_columns
        state.metadata = metadata
        logger.info("Model loaded from %s with %d features", MODEL_PATH, len(feature_columns))
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to load model: %s", exc)
    yield


app = FastAPI(
    title="Life Expectancy Prediction API",
    version="1.0.0",
    description="Production-ready API for LightGBM life expectancy regression.",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        model_path=MODEL_PATH,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return root()


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"features": state.feature_columns, "metadata": state.metadata}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    missing = [col for col in state.feature_columns if col not in req.data]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")

    ordered_values = [req.data[col] for col in state.feature_columns]
    features = pd.DataFrame([ordered_values], columns=state.feature_columns)
    try:
        prediction = _predict_with_model(state.model, features)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed")
    return PredictResponse(prediction=prediction)
