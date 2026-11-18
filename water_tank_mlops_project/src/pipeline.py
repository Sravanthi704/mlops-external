"""Preprocessing, feature engineering and reusable feature-store export."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CAPACITY = 100_000
NUMERIC = [
    "tank_level_8pm_litres", "connected_households",
    "avg_hourly_consumption_litres", "expected_pumping_duration_hours",
    "expected_pumping_rate_lph", "leakage_estimate_lph",
    "expected_overnight_demand", "expected_pump_supply",
    "projected_water_buffer"
]
CATEGORICAL = ["day_type"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Eight-hour planning window from 8 PM to 4 AM plus a two-hour safety allowance.
    out["expected_overnight_demand"] = (
        out["avg_hourly_consumption_litres"] * 10
        + out["leakage_estimate_lph"] * 10
    )
    out["expected_pump_supply"] = (
        out["expected_pumping_duration_hours"] * out["expected_pumping_rate_lph"]
    )
    out["projected_water_buffer"] = (
        out["tank_level_8pm_litres"]
        + out["expected_pump_supply"]
        - out["expected_overnight_demand"]
        - 15_000
    )
    return out

def validate_data(df: pd.DataFrame):
    required = set(NUMERIC[:6] + CATEGORICAL + ["morning_water_shortage"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not df["tank_level_8pm_litres"].between(0, CAPACITY).all():
        raise ValueError("Tank levels must be between 0 and 100000 litres.")
    if not df["connected_households"].ge(1).all():
        raise ValueError("Connected households must be at least 1.")
    if not df["morning_water_shortage"].isin([0, 1]).all():
        raise ValueError("Target must be binary.")

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL)
    ])

def run_pipeline(version="v1"):
    source = ROOT / "data" / "raw" / ("water_tank_v1.csv" if version == "v1" else "water_tank_v2_summer.csv")
    df = pd.read_csv(source)
    validate_data(df)
    engineered = engineer_features(df)
    (ROOT / "features").mkdir(exist_ok=True)
    engineered.to_csv(ROOT / "features" / f"water_tank_feature_store_{version}.csv", index=False)
    # Fit the reusable transformer on the selected version and persist the matrix.
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(engineered[NUMERIC + CATEGORICAL])
    names = preprocessor.get_feature_names_out()
    matrix = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X, columns=names)
    matrix["morning_water_shortage"] = engineered["morning_water_shortage"].values
    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(processed / f"model_matrix_{version}.csv", index=False)
    import joblib
    joblib.dump(preprocessor, processed / f"preprocessor_{version}.joblib")
    metadata = {
        "version": version, "rows": len(engineered),
        "feature_names": list(NUMERIC + CATEGORICAL),
        "engineered_features": ["expected_overnight_demand", "expected_pump_supply", "projected_water_buffer"]
    }
    (ROOT / "features" / f"feature_store_{version}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Pipeline complete for {version}: {X.shape[0]} rows, {X.shape[1]} transformed columns")
    return engineered

if __name__ == "__main__":
    run_pipeline("v1")
    run_pipeline("v2")
