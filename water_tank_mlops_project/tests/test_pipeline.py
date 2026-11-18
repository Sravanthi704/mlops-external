import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import pytest
from pipeline import validate_data, engineer_features

def sample():
    return pd.DataFrame([{
        "tank_level_8pm_litres": 50000, "connected_households": 60,
        "avg_hourly_consumption_litres": 300, "expected_pumping_duration_hours": 1,
        "expected_pumping_rate_lph": 10000, "leakage_estimate_lph": 100,
        "day_type": "weekday", "morning_water_shortage": 0
    }])

def test_valid_data_passes():
    validate_data(sample())

def test_tank_level_cannot_exceed_capacity():
    df = sample()
    df.loc[0, "tank_level_8pm_litres"] = 100001
    with pytest.raises(ValueError):
        validate_data(df)

def test_feature_engineering_columns_exist():
    out = engineer_features(sample())
    for col in ["expected_overnight_demand", "expected_pump_supply", "projected_water_buffer"]:
        assert col in out.columns

def test_expected_demand_calculation():
    out = engineer_features(sample())
    assert out.loc[0, "expected_overnight_demand"] == 4000
