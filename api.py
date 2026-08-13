"""
api.py
FastAPI backend for the Crop Yield Prediction System.

Loads the pipeline saved by train.py and exposes:
    GET  /health              -> liveness + whether a model is loaded
    GET  /model-info          -> metrics for all trained models + which one is in use
    GET  /feature-importance  -> feature importances of the deployed model
    GET  /schema              -> valid categorical values + numeric ranges (for building forms)
    POST /predict              -> single prediction
    POST /predict/batch        -> batch prediction from a list of records

Run with:
    uvicorn api:app --reload --port 8000
"""

import json
import os
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

import config

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
ml_artifacts = {"pipeline": None, "metrics": None, "importance": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(config.PIPELINE_PATH):
        print(
            f"WARNING: no trained pipeline found at {config.PIPELINE_PATH}. "
            "Run `python train.py` first. The API will start but /predict will fail."
        )
    else:
        ml_artifacts["pipeline"] = joblib.load(config.PIPELINE_PATH)
        print(f"Loaded pipeline from {config.PIPELINE_PATH}")

    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as f:
            ml_artifacts["metrics"] = json.load(f)

    if os.path.exists(config.FEATURE_IMPORTANCE_PATH):
        with open(config.FEATURE_IMPORTANCE_PATH) as f:
            ml_artifacts["importance"] = json.load(f)

    yield
    ml_artifacts.clear()


app = FastAPI(
    title="Crop Yield Prediction API",
    description="Predicts expected crop yield (tons/hectare) from weather and soil inputs.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Streamlit frontend (any origin, since this is a local demo app) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
def _range_validator(field_name: str):
    lo, hi = config.FEATURE_RANGES[field_name]

    def _validate(v):
        if not (lo <= v <= hi):
            raise ValueError(f"{field_name} must be between {lo} and {hi}, got {v}")
        return v

    return _validate


class CropInput(BaseModel):
    temperature_avg_c: float = Field(..., description="Average temperature in Celsius")
    rainfall_mm: float = Field(..., description="Total rainfall in millimeters")
    humidity_pct: float = Field(..., description="Average relative humidity (%)")
    sunlight_hours: float = Field(..., description="Average daily sunlight hours")
    soil_ph: float = Field(..., description="Soil pH")
    nitrogen_kg_ha: float = Field(..., description="Nitrogen content (kg/ha)")
    phosphorus_kg_ha: float = Field(..., description="Phosphorus content (kg/ha)")
    potassium_kg_ha: float = Field(..., description="Potassium content (kg/ha)")
    soil_moisture_pct: float = Field(..., description="Soil moisture (%)")
    crop_type: str = Field(..., description=f"One of {config.CROP_TYPES}")
    region: str = Field(..., description=f"One of {config.REGIONS}")
    season: str = Field(..., description=f"One of {config.SEASONS}")

    @field_validator("crop_type")
    @classmethod
    def check_crop(cls, v):
        if v not in config.CROP_TYPES:
            raise ValueError(f"crop_type must be one of {config.CROP_TYPES}")
        return v

    @field_validator("region")
    @classmethod
    def check_region(cls, v):
        if v not in config.REGIONS:
            raise ValueError(f"region must be one of {config.REGIONS}")
        return v

    @field_validator("season")
    @classmethod
    def check_season(cls, v):
        if v not in config.SEASONS:
            raise ValueError(f"season must be one of {config.SEASONS}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "temperature_avg_c": 24.5,
                "rainfall_mm": 950.0,
                "humidity_pct": 65.0,
                "sunlight_hours": 7.2,
                "soil_ph": 6.5,
                "nitrogen_kg_ha": 120.0,
                "phosphorus_kg_ha": 60.0,
                "potassium_kg_ha": 80.0,
                "soil_moisture_pct": 55.0,
                "crop_type": "Wheat",
                "region": "North",
                "season": "Rabi",
            }
        }


class PredictionResponse(BaseModel):
    predicted_yield_tons_per_ha: float
    model_used: str


class BatchPredictionRequest(BaseModel):
    records: List[CropInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_used: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_model():
    if ml_artifacts["pipeline"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run `python train.py` to train and save a pipeline, then restart the API.",
        )


def _to_dataframe(records: List[CropInput]) -> pd.DataFrame:
    rows = [r.model_dump() for r in records]
    return pd.DataFrame(rows)[config.ALL_FEATURES]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": ml_artifacts["pipeline"] is not None,
    }


@app.get("/schema")
def schema():
    return {
        "numeric_features": config.NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "feature_ranges": config.FEATURE_RANGES,
        "crop_types": config.CROP_TYPES,
        "regions": config.REGIONS,
        "seasons": config.SEASONS,
        "target": config.TARGET,
    }


@app.get("/model-info")
def model_info():
    if ml_artifacts["metrics"] is None:
        raise HTTPException(status_code=404, detail="No metrics.json found. Run train.py first.")
    return ml_artifacts["metrics"]


@app.get("/feature-importance")
def feature_importance():
    if ml_artifacts["importance"] is None:
        raise HTTPException(status_code=404, detail="No feature_importance.json found. Run train.py first.")
    return ml_artifacts["importance"]


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CropInput):
    _require_model()
    df = _to_dataframe([payload])
    pred = float(ml_artifacts["pipeline"].predict(df)[0])
    model_used = (ml_artifacts["metrics"] or {}).get("best_model", "unknown")
    return PredictionResponse(predicted_yield_tons_per_ha=round(pred, 3), model_used=model_used)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    _require_model()
    if not payload.records:
        raise HTTPException(status_code=400, detail="records list is empty")
    df = _to_dataframe(payload.records)
    preds = ml_artifacts["pipeline"].predict(df)
    model_used = (ml_artifacts["metrics"] or {}).get("best_model", "unknown")
    return BatchPredictionResponse(
        predictions=[round(float(p), 3) for p in preds], model_used=model_used
    )


@app.get("/")
def root():
    return {
        "message": "Crop Yield Prediction API",
        "docs": "/docs",
        "health": "/health",
    }