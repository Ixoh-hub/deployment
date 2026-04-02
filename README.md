# Life Expectancy LightGBM Deployment

This repo contains code to train a LightGBM regression model on `Life Expectancy Data.csv`, save it, and serve it via FastAPI.

## 1) Train & save model

1. Place `Life Expectancy Data.csv` in the repository root (same level as `train_save_model.py`) or set `DATA_PATH` env variable.
2. Run:

```bash
python train_save_model.py
```

3. The model will be saved to `models/lightgbm_life_expectancy.pkl`.

## 2) Run locally

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- `GET /` returns model status and feature set
- `POST /predict` with JSON `{ "data": { ... } }` returns prediction

Example:

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"data": {"Year": 2000, "Status": 0, ...}}'
```

## 3) Deploy to Render

1. Connect repository in Render dashboard.
2. Set build command:

```bash
pip install -r requirements.txt
python train_save_model.py
```

3. Set start command:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

4. Set any environment variables as needed: `MODEL_PATH` etc.

## 4) Git push

```bash
cd deployment-repo
git init
git add .
git commit -m "Add LightGBM training and Render deployment"
git branch -M main
git remote add origin https://github.com/Ixoh-hub/deployment.git
git push -u origin main
```
