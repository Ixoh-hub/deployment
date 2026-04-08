import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Life Expectancy API", version="1.0.0")

MODEL_PATH = os.getenv("MODEL_PATH", "models/lightgbm_life_expectancy.pkl")

model = None
feature_columns: list = []


class PredictRequest(BaseModel):
    data: dict


def load_model():
    global model, feature_columns
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    feature_columns = payload["features"]


@app.on_event("startup")
def startup_event():
    try:
        load_model()
    except FileNotFoundError as e:
        # Service still starts; health will show not loaded
        print(e)


@app.get("/")
def root():
    return {
        "status": "ok" if model is not None else "degraded",
        "model": "lightgbm",
        "model_loaded": model is not None,
        "features": feature_columns,
    }


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "degraded", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    record = req.data
    row = []
    for col in feature_columns:
        if col not in record:
            raise HTTPException(status_code=400, detail=f"Missing feature: {col}")
        row.append(record[col])

    X = pd.DataFrame([row], columns=feature_columns)
    pred = model.predict(X, num_iteration=model.best_iteration)
    return {"prediction": float(pred[0])}
