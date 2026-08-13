"""
config.py
Central place for feature definitions, valid ranges, and file paths.
Imported by train.py, api.py, and app.py so all three always agree
on what a "valid input" looks like.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATASET_PATH = os.path.join(DATA_DIR, "crop_yield_dataset.csv")
PIPELINE_PATH = os.path.join(MODEL_DIR, "crop_yield_pipeline.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, "feature_importance.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
# Numeric weather + soil features
NUMERIC_FEATURES = [
    "temperature_avg_c",
    "rainfall_mm",
    "humidity_pct",
    "sunlight_hours",
    "soil_ph",
    "nitrogen_kg_ha",
    "phosphorus_kg_ha",
    "potassium_kg_ha",
    "soil_moisture_pct",
]

# Categorical features
CATEGORICAL_FEATURES = ["crop_type", "region", "season"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "yield_tons_per_ha"

# Allowed categorical values (also used to build Streamlit dropdowns)
CROP_TYPES = ["Wheat", "Rice", "Maize", "Soybean", "Cotton"]
REGIONS = ["North", "South", "East", "West", "Central"]
SEASONS = ["Kharif", "Rabi", "Zaid"]

# Reasonable real-world ranges, used both for synthetic data generation
# and for API/Streamlit input validation.
FEATURE_RANGES = {
    "temperature_avg_c": (5.0, 45.0),
    "rainfall_mm": (0.0, 3000.0),
    "humidity_pct": (10.0, 100.0),
    "sunlight_hours": (2.0, 13.0),
    "soil_ph": (3.5, 9.5),
    "nitrogen_kg_ha": (0.0, 300.0),
    "phosphorus_kg_ha": (0.0, 200.0),
    "potassium_kg_ha": (0.0, 300.0),
    "soil_moisture_pct": (0.0, 100.0),
}

RANDOM_STATE = 42
