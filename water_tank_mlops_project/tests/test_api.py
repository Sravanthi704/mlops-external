import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_rejects_over_capacity():
    payload = {
        "tank_level_8pm_litres": 100001,
        "connected_households": 60,
        "avg_hourly_consumption_litres": 300,
        "expected_pumping_duration_hours": 1,
        "expected_pumping_rate_lph": 10000,
        "leakage_estimate_lph": 100,
        "day_type": "weekday"
    }
    assert client.post("/predict", json=payload).status_code == 422
