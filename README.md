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

## 5) Streamlit Community Cloud (when Render already hosts the API)

Your logs show three separate problems. Fix **all** of them or the app stays on “in the oven” or fails after a long build.

### A) Wrong entry file

Streamlit is starting **`app.py`**. In this repo, `app.py` is **FastAPI**, not Streamlit.

In Streamlit Community Cloud → your app → **Settings** (or the redeploy wizard): set **Main file** to **`streamlit_app.py`**.

### B) `seaborn==0.14.1` breaks the resolver

`uv` reports **no solution** for `seaborn==0.14.1` on their environment (worse on **Python 3.14**).

- If the UI does **not** import seaborn: **remove seaborn** from `requirements.txt`.
- If it does: use a range such as `seaborn>=0.13.2` instead of `==0.14.1`, or use **`requirements-streamlit-cloud.txt`** from this repo as your Cloud `requirements.txt`.

### C) Python 3.14 + pandas = slow / stuck builds

Logs show **Python 3.14.3** and **pandas** building from **source** (`pandas-2.2.2.tar.gz`), which can take ages and look stuck.

Per [Streamlit’s Python version docs](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python), you **cannot** change Python after deploy: **delete the Cloud app**, redeploy, and in **Advanced settings** choose **Python 3.12** or **3.11** if offered.

### D) Branch name

Your log shows branch **`master`**. Push fixes to the branch Streamlit is tracking, or change the app to use **`main`**.

## 6) Hosted page: am I auto-updating it?

No. I changed files on your computer only.

To update your hosted deployment, you must **commit + push** your updated `deployment-repo` to the GitHub repo that your host (Render / Streamlit Cloud / etc.) is connected to. Most hosts redeploy automatically after a push to `main`.

## 7) CI pipeline

GitHub Actions workflow at `.github/workflows/ci.yml` runs on push/PR:

- dependency installation
- Python syntax compile checks
- FastAPI and Streamlit import smoke tests

This provides a baseline deployment gate before release.
