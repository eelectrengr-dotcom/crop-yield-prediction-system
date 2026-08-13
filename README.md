# 🌾 Crop Yield Prediction System

**Week 7 — Advanced**
Predict expected crop production using weather and soil information.

- **Models:** XGBoost Regressor, Random Forest Regressor, LightGBM (best one auto-selected by RMSE)
- **Evaluation:** RMSE, MAE, R² Score
- **Deployment:** Streamlit (frontend) + FastAPI (backend)
- **Skills:** Advanced Regression, Forecasting, Feature Engineering

---

## 1. Project structure

```
crop-yield-app/
├── config.py              # Shared feature schema, ranges, and file paths
├── train.py                # Generates data, trains 3 models, saves the best pipeline
├── api.py                  # FastAPI backend that serves predictions
├── app.py                  # Streamlit frontend UI
├── requirements.txt
├── .gitignore
├── data/
│   └── crop_yield_dataset.csv     # created by train.py
└── models/
    ├── crop_yield_pipeline.joblib # created by train.py (preprocessing + best model)
    ├── metrics.json               # created by train.py (RMSE/MAE/R² per model)
    └── feature_importance.json    # created by train.py
```

## 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train the models

```bash
python train.py
```

This will:
1. Generate a synthetic 5,000-row dataset relating weather + soil conditions to yield (tons/hectare), saved to `data/crop_yield_dataset.csv`. To use your own data instead, run `python train.py --data path/to/your.csv` — the CSV needs the same columns as listed in `config.ALL_FEATURES` plus a `yield_tons_per_ha` target column.
2. Train **XGBoost**, **Random Forest**, and **LightGBM**, each wrapped in a full preprocessing pipeline (scaling + one-hot encoding).
3. Evaluate all three on a held-out test set with **RMSE**, **MAE**, and **R²**.
4. Save the best-performing pipeline (lowest RMSE) to `models/crop_yield_pipeline.joblib`, plus `metrics.json` and `feature_importance.json`.

Useful flags:
```bash
python train.py --n-samples 8000     # generate more synthetic rows
python train.py --test-size 0.25     # change train/test split
```

## 4. Run the API

```bash
uvicorn api:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

| Method | Endpoint              | Description                                   |
|--------|------------------------|------------------------------------------------|
| GET    | `/health`              | Liveness check + whether a model is loaded     |
| GET    | `/schema`              | Valid categorical values and numeric ranges    |
| GET    | `/model-info`          | Metrics for all trained models + best model    |
| GET    | `/feature-importance`  | Feature importances of the deployed model      |
| POST   | `/predict`             | Predict yield for a single set of inputs       |
| POST   | `/predict/batch`       | Predict yield for a list of records            |

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "temperature_avg_c": 24.5,
        "rainfall_mm": 950,
        "humidity_pct": 65,
        "sunlight_hours": 7.2,
        "soil_ph": 6.5,
        "nitrogen_kg_ha": 120,
        "phosphorus_kg_ha": 60,
        "potassium_kg_ha": 80,
        "soil_moisture_pct": 55,
        "crop_type": "Wheat",
        "region": "North",
        "season": "Rabi"
      }'
```

## 5. Run the frontend

In a second terminal (with the API already running):

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501). The app has three tabs:
- **Single Prediction** — sliders/dropdowns for one field, returns a predicted yield.
- **Batch Prediction (CSV)** — upload a CSV of many fields, get predictions + a downloadable CSV.
- **Model Info** — compares RMSE/MAE/R² across all three trained models and shows feature importance.

If the API isn't on `localhost:8000`, set `API_URL` before launching:
```bash
API_URL=http://your-api-host:8000 streamlit run app.py
```

## 6. Feature schema

**Numeric (weather + soil):**
`temperature_avg_c`, `rainfall_mm`, `humidity_pct`, `sunlight_hours`, `soil_ph`, `nitrogen_kg_ha`, `phosphorus_kg_ha`, `potassium_kg_ha`, `soil_moisture_pct`

**Categorical:**
- `crop_type`: Wheat, Rice, Maize, Soybean, Cotton
- `region`: North, South, East, West, Central
- `season`: Kharif, Rabi, Zaid

**Target:** `yield_tons_per_ha`

All of this lives in one place — `config.py` — so `train.py`, `api.py`, and `app.py` never drift out of sync.

## 7. Notes on the synthetic data

`train.py` ships with a synthetic data generator so the whole pipeline runs end-to-end with zero setup. The relationships are hand-crafted to be agronomically plausible (yield peaks at a crop-specific optimal temperature, responds to rainfall/nutrients with diminishing returns, etc.) rather than pure noise, so the models have a real signal to learn. **Swap in real agricultural data** (e.g., from your local agriculture department or a Kaggle crop-yield dataset) by passing `--data your_file.csv` with matching columns for production use.

## 9. Deploying: Streamlit + FastAPI (both live, as intended)

`app.py` supports two modes, chosen automatically:
- **API mode** — if an `API_URL` is configured, it calls your deployed FastAPI backend for every prediction (this is the "Streamlit + FastAPI" setup).
- **Standalone mode** — if `API_URL` isn't set (or the API is unreachable), it loads/trains the model directly, so the app still works with zero backend.

To get the real two-service deployment running:

### 9.1 Push to GitHub

```bash
cd "Crop Yield Prediction System"
git init
git add .
git commit -m "Initial commit: crop yield prediction system"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

### 9.2 Deploy the FastAPI backend on Render (free tier)

1. Go to **https://render.com**, sign in with GitHub.
2. Click **New +** → **Web Service** → select your repo.
3. Set:
   - **Build Command:** `pip install -r requirements-api.txt`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Click **Create Web Service** and wait for it to deploy.
5. Once live, note the URL Render gives you, e.g. `https://your-app.onrender.com`. Visit `https://your-app.onrender.com/health` to confirm it responds — the first request may take ~30-90s since `api.py` trains a model on first startup (no `models/` folder is committed to git).

**Note:** Render's free tier spins the service down after ~15 minutes of no traffic, and cold-starts take 30-60s on the next request. That's normal for a free-tier demo.

### 9.3 Deploy the Streamlit frontend

1. Go to **https://share.streamlit.io**, sign in with GitHub.
2. Click **New app**, select your repo/branch, set **Main file path** to `app.py`, and deploy.
3. Once deployed, open **Manage app** → **Settings** → **Secrets**, and add:
   ```
   API_URL = "https://your-app.onrender.com"
   ```
4. Save — the app restarts automatically and will now show **"Connected to FastAPI backend at ..."** and route every prediction through your live API.

### 9.4 Verify

- Deployed Streamlit app header should say *"Connected to FastAPI backend at `https://your-app.onrender.com`"*.
- Make a prediction — it's now served by the deployed FastAPI, not computed in-process.
- If Render's free instance has gone to sleep, you may briefly see a fallback warning and standalone mode kick in — refresh after a few seconds once Render wakes back up.

## 10. Troubleshooting

- **Streamlit can't load a model in standalone mode** → run `python train.py` first, or let it auto-train on first load.
- **`api.py` says "Model is not loaded" locally** → run `python train.py` first, or just start the API — it now auto-trains if no model file exists.
- **LightGBM install issues on macOS** → you may need `brew install libomp` first.
- **pip fails building pandas/numpy from source on Windows** → use the `>=` version ranges already in `requirements.txt` (not exact pins) so pip picks a version with a prebuilt wheel for your Python version; avoid Python 3.13+ if wheels aren't yet available for it.
- **Streamlit app stuck showing the API warning** → Render's free tier sleeps after inactivity; give it ~30-60s and refresh, or check the Render dashboard to confirm the service is running.
