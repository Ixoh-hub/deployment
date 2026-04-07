# Life Expectancy Deployment (Production-Ready Baseline)

This project trains a LightGBM model, serves it via FastAPI, and provides a Streamlit user interface.

## Highlights

- FastAPI with typed request/response models and health endpoints (`/` and `/health`)
- Streamlit UI with cleaner layout, metrics cards, and median-based default inputs
- Hardened Docker image (non-root user + healthcheck + slim runtime)
- GitHub Actions CI pipeline for syntax and import smoke tests
- Training script exports model metadata and feature ranges for safer inference/UI defaults

## 1) Train and save model artifact

1. Place `Life Expectancy Data.csv` in repo root (or set `DATA_PATH`)
2. Run:

```bash
python train_save_model.py
```

The model is saved to `models/lightgbm_life_expectancy.pkl` with:

- model object
- feature list
- metadata (RMSE, R2, feature ranges)

## 2) Run locally

Install deps:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Run Streamlit UI:

```bash
streamlit run streamlit_app.py --server.port 8501
```

## 3) API usage

- `GET /` or `GET /health` -> service and model status
- `GET /metadata` -> required feature list and model metadata
- `POST /predict` -> inference endpoint

Example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data": {"Year": 2010, "Status": 1, "Adult Mortality": 200}}'
```

> Include all required features from `GET /metadata`.

## 4) Docker deployment

### Option A (recommended): run API + UI together

```bash
docker compose up --build
```

- API: `http://localhost:8000` (health: `GET /health`, docs: `GET /docs`)
- UI: `http://localhost:8501`

### Option B: Streamlit-only container (UI demo)

Build image:

```bash
docker build -t life-exp-app .
```

Run:

```bash
docker run -p 8501:8501 -e PORT=8501 life-exp-app
```

The container has a healthcheck against Streamlit's health endpoint.

## 6) Hosted page: am I auto-updating it?

No. I changed files on your computer only.

To update your hosted deployment, you must **commit + push** your updated `deployment-repo` to the GitHub repo that your host (Render / Streamlit Cloud / etc.) is connected to. Most hosts redeploy automatically after a push to `main`.

If you tell me which host you’re using (Render vs Streamlit Community Cloud vs something else), I’ll give you the exact click-by-click redeploy settings for it.

## 5) CI pipeline

GitHub Actions workflow at `.github/workflows/ci.yml` runs on push/PR:

- dependency installation
- Python syntax compile checks
- FastAPI and Streamlit import smoke tests

This provides a baseline deployment gate before release.
