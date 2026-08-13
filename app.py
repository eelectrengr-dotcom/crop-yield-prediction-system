"""
app.py
Streamlit frontend for the Crop Yield Prediction System.

Talks to the FastAPI backend (api.py) over HTTP. Start the API first:
    uvicorn api:app --reload --port 8000
Then run this app:
    streamlit run app.py

Set the API_URL environment variable if the API is not on localhost:8000.
"""

import os

import pandas as pd
import requests
import streamlit as st

import config

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Crop Yield Prediction", page_icon="🌾", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30)
def get_model_info():
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30)
def get_feature_importance():
    try:
        r = requests.get(f"{API_URL}/feature-importance", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def call_predict(payload: dict):
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def call_predict_batch(records: list):
    r = requests.post(f"{API_URL}/predict/batch", json={"records": records}, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Header + connection status
# ---------------------------------------------------------------------------
st.title("🌾 Crop Yield Prediction System")
st.caption("Predict expected crop production from weather and soil information.")

health = check_health()
if health is None:
    st.error(
        f"Can't reach the API at `{API_URL}`. Start it with `uvicorn api:app --reload --port 8000`, "
        "or set the API_URL environment variable."
    )
    st.stop()
elif not health.get("model_loaded"):
    st.warning("API is running but no trained model is loaded. Run `python train.py` first, then restart the API.")
    st.stop()
else:
    st.success(f"Connected to API at `{API_URL}`")

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
            result = call_predict(payload)
            st.metric(
                label=f"Predicted Yield — {crop_type} ({region}, {season})",
                value=f"{result['predicted_yield_tons_per_ha']} t/ha",
            )
            st.caption(f"Model used: {result['model_used']}")
        except requests.HTTPError as e:
            st.error(f"Prediction failed: {e.response.text}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# Tab 2: Batch prediction
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.write(
        "Upload a CSV with these columns: "
        f"`{', '.join(config.ALL_FEATURES)}`"
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = set(config.ALL_FEATURES) - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            st.dataframe(df.head(), use_container_width=True)
            if st.button("Run batch prediction", type="primary"):
                records = df[config.ALL_FEATURES].to_dict(orient="records")
                try:
                    result = call_predict_batch(records)
                    df_out = df.copy()
                    df_out["predicted_yield_tons_per_ha"] = result["predictions"]
                    st.success(f"Predicted {len(df_out)} rows using {result['model_used']}.")
                    st.dataframe(df_out, use_container_width=True)
                    st.download_button(
                        "Download results as CSV",
                        df_out.to_csv(index=False).encode("utf-8"),
                        file_name="crop_yield_predictions.csv",
                        mime="text/csv",
                    )
                except requests.HTTPError as e:
                    st.error(f"Batch prediction failed: {e.response.text}")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

# ---------------------------------------------------------------------------
# Tab 3: Model info
# ---------------------------------------------------------------------------
with tab_model:
    st.subheader("Model comparison & metrics")
    info = get_model_info()
    if info is None:
        st.info("No model metrics available yet. Run `python train.py`.")
    else:
        st.write(f"**Best model in production:** {info['best_model']}")
        metrics_df = pd.DataFrame(info["all_models"]).T
        st.dataframe(metrics_df, use_container_width=True)
        st.caption(f"Trained on {info['trained_on_rows']} rows.")

    st.subheader("Feature importance")
    importance = get_feature_importance()
    if importance:
        imp_df = pd.DataFrame(list(importance.items()), columns=["feature", "importance"]).head(15)
        st.bar_chart(imp_df.set_index("feature"))
    else:
        st.info("No feature importance available yet.")