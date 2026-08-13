"""
train.py
Crop Yield Prediction System — Week 7 (Advanced)

Generates a synthetic weather + soil dataset (swap in real data by
replacing `generate_synthetic_dataset` or pointing DATASET_PATH at
your own CSV with the same columns), trains three regressors
(XGBoost, Random Forest, LightGBM) inside sklearn Pipelines, evaluates
them with RMSE / MAE / R^2, and saves the best-performing pipeline
for the API to serve.

Usage:
    python train.py                 # generate data + train + save
    python train.py --data my.csv   # train on your own CSV instead
    python train.py --n-samples 8000
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import config


# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(n_samples: int = 5000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """
    Creates a synthetic but agronomically-plausible dataset relating
    weather + soil conditions to crop yield (tons/hectare).

    The relationships are hand-crafted (not random noise) so trained
    models actually learn a real signal:
      - Yield rises with rainfall and nitrogen up to a point, then
        plateaus/declines (too much water or N hurts).
      - Extreme temperatures reduce yield (quadratic penalty around
        an optimal temperature per crop).
      - Soil pH close to neutral (6.0-7.5) is best.
      - Each crop has a different base yield potential.
    """
    rng = np.random.default_rng(seed)

    crop_type = rng.choice(config.CROP_TYPES, size=n_samples)
    region = rng.choice(config.REGIONS, size=n_samples)
    season = rng.choice(config.SEASONS, size=n_samples)

    def sample_range(name, size):
        lo, hi = config.FEATURE_RANGES[name]
        return rng.uniform(lo, hi, size=size)

    temperature_avg_c = sample_range("temperature_avg_c", n_samples)
    rainfall_mm = sample_range("rainfall_mm", n_samples)
    humidity_pct = sample_range("humidity_pct", n_samples)
    sunlight_hours = sample_range("sunlight_hours", n_samples)
    soil_ph = sample_range("soil_ph", n_samples)
    nitrogen_kg_ha = sample_range("nitrogen_kg_ha", n_samples)
    phosphorus_kg_ha = sample_range("phosphorus_kg_ha", n_samples)
    potassium_kg_ha = sample_range("potassium_kg_ha", n_samples)
    soil_moisture_pct = sample_range("soil_moisture_pct", n_samples)

    # Base yield potential per crop (tons/ha, roughly realistic averages)
    base_yield_map = {"Wheat": 3.2, "Rice": 4.0, "Maize": 5.5, "Soybean": 2.8, "Cotton": 2.2}
    optimal_temp_map = {"Wheat": 22.0, "Rice": 27.0, "Maize": 25.0, "Soybean": 24.0, "Cotton": 30.0}

    base_yield = np.array([base_yield_map[c] for c in crop_type])
    optimal_temp = np.array([optimal_temp_map[c] for c in crop_type])

    # --- Component effects -------------------------------------------------
    temp_effect = -0.01 * (temperature_avg_c - optimal_temp) ** 2

    rainfall_opt = 1000.0
    rainfall_effect = 1.4 * (1 - ((rainfall_mm - rainfall_opt) / 1400.0) ** 2)
    rainfall_effect = np.clip(rainfall_effect, -1.0, 1.4)

    ph_effect = -0.35 * (soil_ph - 6.6) ** 2

    n_effect = 1.6 * (1 - np.exp(-nitrogen_kg_ha / 90.0)) - 0.002 * np.maximum(nitrogen_kg_ha - 220, 0)
    p_effect = 0.5 * (1 - np.exp(-phosphorus_kg_ha / 70.0))
    k_effect = 0.5 * (1 - np.exp(-potassium_kg_ha / 90.0))

    moisture_effect = -0.0004 * (soil_moisture_pct - 55.0) ** 2
    humidity_effect = -0.0006 * (humidity_pct - 60.0) ** 2
    sunlight_effect = 0.12 * (sunlight_hours - 6.0)

    season_bonus_map = {"Kharif": 0.15, "Rabi": 0.25, "Zaid": -0.10}
    season_effect = np.array([season_bonus_map[s] for s in season])

    region_bonus_map = {"North": 0.1, "South": 0.0, "East": 0.2, "West": -0.1, "Central": 0.05}
    region_effect = np.array([region_bonus_map[r] for r in region])

    noise = rng.normal(0, 0.35, size=n_samples)

    yield_tons_per_ha = (
        base_yield
        + temp_effect
        + rainfall_effect
        + ph_effect
        + n_effect
        + p_effect
        + k_effect
        + moisture_effect
        + humidity_effect
        + sunlight_effect
        + season_effect
        + region_effect
        + noise
    )
    yield_tons_per_ha = np.clip(yield_tons_per_ha, 0.1, None)

    df = pd.DataFrame(
        {
            "temperature_avg_c": temperature_avg_c.round(2),
            "rainfall_mm": rainfall_mm.round(1),
            "humidity_pct": humidity_pct.round(1),
            "sunlight_hours": sunlight_hours.round(2),
            "soil_ph": soil_ph.round(2),
            "nitrogen_kg_ha": nitrogen_kg_ha.round(1),
            "phosphorus_kg_ha": phosphorus_kg_ha.round(1),
            "potassium_kg_ha": potassium_kg_ha.round(1),
            "soil_moisture_pct": soil_moisture_pct.round(1),
            "crop_type": crop_type,
            "region": region,
            "season": season,
            config.TARGET: yield_tons_per_ha.round(3),
        }
    )
    return df


# ---------------------------------------------------------------------------
# 2. Pipeline builders
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict:
    return {
        "XGBoost": XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=3,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


# ---------------------------------------------------------------------------
# 3. Train + evaluate
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def get_feature_importance(pipeline: Pipeline) -> dict:
    """Extract feature importance from a fitted pipeline, mapped back to
    human-readable feature names (numeric features keep their name,
    one-hot categorical columns become `column=value`)."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return {}

    preprocessor = pipeline.named_steps["preprocessor"]
    cat_names = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(config.CATEGORICAL_FEATURES)
    )
    feature_names = config.NUMERIC_FEATURES + cat_names

    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return {name: round(float(val), 5) for name, val in pairs}


def main():
    parser = argparse.ArgumentParser(description="Train the Crop Yield Prediction models.")
    parser.add_argument("--data", type=str, default=None, help="Path to a CSV with the expected columns. If omitted, a synthetic dataset is generated.")
    parser.add_argument("--n-samples", type=int, default=5000, help="Number of synthetic rows to generate (ignored if --data is given).")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 70)
    print("CROP YIELD PREDICTION SYSTEM — TRAINING")
    print("=" * 70)

    if args.data:
        print(f"Loading dataset from {args.data} ...")
        df = pd.read_csv(args.data)
        missing = set(config.ALL_FEATURES + [config.TARGET]) - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
    else:
        print(f"Generating synthetic dataset ({args.n_samples} rows) ...")
        df = generate_synthetic_dataset(n_samples=args.n_samples)
        df.to_csv(config.DATASET_PATH, index=False)
        print(f"Saved synthetic dataset -> {config.DATASET_PATH}")

    X = df[config.ALL_FEATURES]
    y = df[config.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=config.RANDOM_STATE
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    models = build_models()
    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        start = time.time()
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - start

        preds = pipeline.predict(X_test)
        metrics = evaluate(y_test, preds)
        metrics["train_seconds"] = round(elapsed, 2)

        results[name] = metrics
        fitted_pipelines[name] = pipeline
        print(f"  RMSE={metrics['rmse']}  MAE={metrics['mae']}  R2={metrics['r2']}  ({elapsed:.1f}s)")

    # --- Pick the best model by lowest RMSE --------------------------------
    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_pipeline = fitted_pipelines[best_name]

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    summary_df = pd.DataFrame(results).T.sort_values("rmse")
    print(summary_df.to_string())
    print(f"\nBest model: {best_name} (lowest RMSE)")

    # --- Save artifacts ------------------------------------------------------
    joblib.dump(best_pipeline, config.PIPELINE_PATH)
    print(f"\nSaved best pipeline -> {config.PIPELINE_PATH}")

    metrics_payload = {
        "best_model": best_name,
        "all_models": results,
        "trained_on_rows": len(df),
        "features": config.ALL_FEATURES,
        "target": config.TARGET,
    }
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved metrics -> {config.METRICS_PATH}")

    importance = get_feature_importance(best_pipeline)
    with open(config.FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(importance, f, indent=2)
    print(f"Saved feature importance -> {config.FEATURE_IMPORTANCE_PATH}")

    print("\nDone. Start the API with:  uvicorn api:app --reload")


if __name__ == "__main__":
    main()