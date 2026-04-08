# Life Expectancy — train, API (FastAPI), UI (Streamlit)

No scikit-learn. Training uses **LightGBM** + **pandas** only.

## Train

Put `Life Expectancy Data.csv` next to `train_save_model.py` (or set `DATA_PATH`).

```bash
python train_save_model.py
```

Saves `models/lightgbm_life_expectancy.pkl`.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Other terminal:

```bash
streamlit run streamlit_app.py --server.port 8501
```

- API: `http://localhost:8000` — `GET /health`, `POST /predict` with `{"data": { ...all features... }}`
- UI: `http://localhost:8501`

## Docker (Streamlit only)

```bash
docker build -t life-exp .
docker run -p 8501:8501 life-exp
```

## Streamlit Community Cloud

1. **Main file:** `streamlit_app.py` (not `app.py`).
2. **`requirements.txt`:** use `requirements-streamlit-cloud.txt` as your Cloud requirements (no seaborn, no sklearn).
3. **Python:** pick **3.12** or **3.11** in Advanced settings when you create the app (avoids broken builds on 3.14).
4. Commit **`models/lightgbm_life_expectancy.pkl`** with the repo.

## Render (API)

Build: `pip install -r requirements.txt` (and train if you need a fresh model).  
Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## CI

`.github/workflows/ci.yml` runs compile checks on push/PR to `main`.
