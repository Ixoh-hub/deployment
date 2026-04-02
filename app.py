import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

MODEL_PATH = os.getenv('MODEL_PATH', 'models/lightgbm_life_expectancy.pkl')

class PredictRequest(BaseModel):
    data: dict


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    payload = joblib.load(MODEL_PATH)
    return payload['model'], payload['features']


model, feature_columns = None, None

@app.on_event('startup')
def startup_event():
    global model, feature_columns
    model, feature_columns = load_model()


@app.get('/')
def root():
    return {'status': 'ok', 'model': 'lightgbm', 'features': feature_columns}


@app.post('/predict')
def predict(req: PredictRequest):
    global model, feature_columns
    if model is None:
        raise HTTPException(status_code=500, detail='Model not loaded')

    record = req.data
    row = []
    for col in feature_columns:
        if col not in record:
            raise HTTPException(status_code=400, detail=f"Missing feature: {col}")
        row.append(record[col])

    X = pd.DataFrame([row], columns=feature_columns)
    pred = model.predict(X, num_iteration=model.best_iteration)
    return {'prediction': float(pred[0])}
