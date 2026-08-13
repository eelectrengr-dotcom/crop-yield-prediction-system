"""
app.py
Streamlit frontend for the Crop Yield Prediction System.

Two modes, controlled automatically by whether API_URL is configured:

1. API MODE (Streamlit + FastAPI, matches the assignment spec):
   Set an API_URL secret/env var pointing at your deployed api.py
   (e.g. a Render URL). The app calls it over HTTP for every prediction.

2. STANDALONE MODE (fallback, no separate backend needed):
   If API_URL isn't set, the app loads/trains the model directly in-process.
   This keeps the app usable even without a hosted API.

Local dev:
    uvicorn api:app --reload --port 8000      # terminal 1
    API_URL=http://localhost:8000 streamlit run app.py   # terminal 2 (API mode)
    streamlit run app.py                                 # or just this (standalone mode)
"""

import json
import os

import joblib
import pandas as pd
import requests
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import config
from train import (
    build_models,
    build_preprocessor,
    evaluate,
    generate_synthetic_dataset,
    get_feature_importance,
)

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")

# API_URL can come from an environment variable (local dev) or Streamlit secrets (cloud deploy).
API_URL = os.environ.get("API_URL") or st.secrets.get("API_URL", None) if hasattr(st, "secrets") else os.environ.get("API_URL")
try:
    if not API_URL and "API_URL" in st.secrets:
        API_URL = st.secrets["API_URL"]
except Exception:
    pass

USE_API = bool(API_URL)


# ---------------------------------------------------------------------------
# STANDALONE MODE: load or train the model directly in-process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing model (first run trains it, ~30-90s)...")
def load_or_train():
    if os.path.exists(config.PIPELINE_PATH) and os.path.exists(config.METRICS_PATH):
        pipeline = joblib.load(config.PIPELINE_PATH)
        with open(config.METRICS_PATH) as f:
            metrics = json.load(f)
        return pipeline, metrics

    df = generate_synthetic_dataset(n_samples=5000)
    X, y = df[config.ALL_FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    results, fitted = {}, {}
    for name, model in build_models().items():
        pipe = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])
        pipe.fit(X_train, y_train)
        results[name] = evaluate(y_test, pipe.predict(X_test))
        fitted[name] = pipe

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_pipeline = fitted[best_name]

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipeline, config.PIPELINE_PATH)

    metrics = {
        "best_model": best_name,
        "all_models": results,
        "trained_on_rows": len(df),
        "features": config.ALL_FEATURES,
        "target": config.TARGET,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    importance = get_feature_importance(best_pipeline)
    with open(config.FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(importance, f, indent=2)

    return best_pipeline, metrics


@st.cache_resource
def load_feature_importance_local():
    if os.path.exists(config.FEATURE_IMPORTANCE_PATH):
        with open(config.FEATURE_IMPORTANCE_PATH) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# API MODE: talk to the deployed FastAPI backend
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=60)
def api_model_info():
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=60)
def api_feature_importance():
    try:
        r = requests.get(f"{API_URL}/feature-importance", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_predict(payload: dict) -> dict:
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def api_predict_batch(records: list) -> dict:
    r = requests.post(f"{API_URL}/predict/batch", json={"records": records}, timeout=60)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Unified interface used by the UI below
# ---------------------------------------------------------------------------
if USE_API:
    health = api_health()
    if health is None or not health.get("model_loaded"):
        st.warning(
            f"Configured to use the API at `{API_URL}` but it isn't responding yet "
            "(free hosts can take ~30-60s to wake up from a cold start). Retry in a moment, "
            "or the app will fall back to standalone mode below."
        )
        USE_API = False

if USE_API:
    metrics_info = api_model_info() or {}
    importance_info = api_feature_importance() or {}

    def predict_one(payload: dict) -> float:
        return api_predict(payload)["predicted_yield_tons_per_ha"]

    def predict_many(df: pd.DataFrame) -> list:
        records = df[config.ALL_FEATURES].to_dict(orient="records")
        return api_predict_batch(records)["predictions"]

else:
    pipeline, metrics_info = load_or_train()
    importance_info = load_feature_importance_local()

    def predict_one(payload: dict) -> float:
        df = pd.DataFrame([payload])[config.ALL_FEATURES]
        return round(float(pipeline.predict(df)[0]), 3)

    def predict_many(df: pd.DataFrame) -> list:
        return [round(float(p), 3) for p in pipeline.predict(df[config.ALL_FEATURES])]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌾 Crop Yield Prediction System")
st.caption("Predict expected crop production from weather and soil information.")

if USE_API:
    st.success(f"Connected to FastAPI backend at `{API_URL}` — using **{metrics_info.get('best_model', '?')}**")
else:
    st.success(f"Standalone mode — using **{metrics_info.get('best_model', '?')}**")

tab_single, tab_batch, tab_model = st.tabs(["Single Prediction", "Batch Prediction (CSV)", "Model Info"])

# ---------------------------------------------------------------------------
# Tab 1: Single prediction
# ---------------------------------------------------------------------------
with tab_single:
    col_weather, col_soil, col_meta = st.columns(3)

    with col_weather:
        st.subheader("Weather")
        temperature_avg_c = st.slider("Avg Temperature (°C)", *config.FEATURE_RANGES["temperature_avg_c"], value=24.0)
        rainfall_mm = st.slider("Rainfall (mm)", *config.FEATURE_RANGES["rainfall_mm"], value=950.0)
        humidity_pct = st.slider("Humidity (%)", *config.FEATURE_RANGES["humidity_pct"], value=65.0)
        sunlight_hours = st.slider("Sunlight (hrs/day)", *config.FEATURE_RANGES["sunlight_hours"], value=7.0)

    with col_soil:
        st.subheader("Soil")
        soil_ph = st.slider("Soil pH", *config.FEATURE_RANGES["soil_ph"], value=6.5)
        nitrogen_kg_ha = st.slider("Nitrogen (kg/ha)", *config.FEATURE_RANGES["nitrogen_kg_ha"], value=120.0)
        phosphorus_kg_ha = st.slider("Phosphorus (kg/ha)", *config.FEATURE_RANGES["phosphorus_kg_ha"], value=60.0)
        potassium_kg_ha = st.slider("Potassium (kg/ha)", *config.FEATURE_RANGES["potassium_kg_ha"], value=80.0)
        soil_moisture_pct = st.slider("Soil Moisture (%)", *config.FEATURE_RANGES["soil_moisture_pct"], value=55.0)

    with col_meta:
        st.subheader("Crop & Context")
        crop_type = st.selectbox("Crop Type", config.CROP_TYPES)
        region = st.selectbox("Region", config.REGIONS)
        season = st.selectbox("Season", config.SEASONS)

    st.divider()

    if st.button("🔮 Predict Yield", type="primary", use_container_width=True):
        payload = {
            "temperature_avg_c": temperature_avg_c,
            "rainfall_mm": rainfall_mm,
            "humidity_pct": humidity_pct,
            "sunlight_hours": sunlight_hours,
            "soil_ph": soil_ph,
            "nitrogen_kg_ha": nitrogen_kg_ha,
            "phosphorus_kg_ha": phosphorus_kg_ha,
            "potassium_kg_ha": potassium_kg_ha,
            "soil_moisture_pct": soil_moisture_pct,
            "crop_type": crop_type,
            "region": region,
            "season": season,
        }
        try:
            pred = predict_one(payload)
            st.metric(
                label=f"Predicted Yield — {crop_type} ({region}, {season})",
                value=f"{pred} t/ha",
            )
            st.caption(f"Model used: {metrics_info.get('best_model', '?')}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# Tab 2: Batch prediction
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.write(f"Upload a CSV with these columns: `{', '.join(config.ALL_FEATURES)}`")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = set(config.ALL_FEATURES) - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            st.dataframe(df.head(), use_container_width=True)
            if st.button("Run batch prediction", type="primary"):
                try:
                    df_out = df.copy()
                    df_out["predicted_yield_tons_per_ha"] = predict_many(df)
                    st.success(f"Predicted {len(df_out)} rows using {metrics_info.get('best_model', '?')}.")
                    st.dataframe(df_out, use_container_width=True)
                    st.download_button(
                        "Download results as CSV",
                        df_out.to_csv(index=False).encode("utf-8"),
                        file_name="crop_yield_predictions.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

# ---------------------------------------------------------------------------
# Tab 3: Model info
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Model comparison & metrics")
    if metrics_info:
        st.write(f"**Best model in production:** {metrics_info.get('best_model', '?')}")
        if "all_models" in metrics_info:
            metrics_df = pd.DataFrame(metrics_info["all_models"]).T
            st.dataframe(metrics_df, use_container_width=True)
        st.caption(f"Trained on {metrics_info.get('trained_on_rows', '?')} rows.")
    else:
        st.info("No model metrics available yet.")

    st.subheader("Feature importance")
    if importance_info:
        imp_df = pd.DataFrame(list(importance_info.items()), columns=["feature", "importance"]).head(15)
        st.bar_chart(imp_df.set_index("feature"))
    else:
        st.info("No feature importance available yet.")
