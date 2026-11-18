"""FastAPI model serving for evening water-tank conditions."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "selected_model.joblib"
app = FastAPI(title="Overnight Water Shortage Predictor", version="1.0.0")
_model = None

class EveningConditions(BaseModel):
    tank_level_8pm_litres: float = Field(ge=0, le=100000)
    connected_households: int = Field(ge=1, le=1000)
    avg_hourly_consumption_litres: float = Field(gt=0)
    expected_pumping_duration_hours: float = Field(ge=0, le=12)
    expected_pumping_rate_lph: float = Field(ge=0)
    leakage_estimate_lph: float = Field(ge=0)
    day_type: Literal["weekday", "weekend", "holiday"]

@app.on_event("startup")
def load_model():
    global _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict")
def predict(data: EveningConditions):
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            _model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model not found. Run python src/train.py first.")
    row = pd.DataFrame([data.model_dump()])
    row["expected_overnight_demand"] = (
        row["avg_hourly_consumption_litres"] * 10 + row["leakage_estimate_lph"] * 10
    )
    row["expected_pump_supply"] = row["expected_pumping_duration_hours"] * row["expected_pumping_rate_lph"]
    row["projected_water_buffer"] = (
        row["tank_level_8pm_litres"] + row["expected_pump_supply"]
        - row["expected_overnight_demand"] - 15000
    )
    prediction = int(_model.predict(row)[0])
    probability = float(_model.predict_proba(row)[0][1])
    return {
        "prediction": "Shortage Risk" if prediction == 1 else "Adequate Reserve",
        "shortage_probability": round(probability, 4),
        "threshold_litres": 15000
    }
