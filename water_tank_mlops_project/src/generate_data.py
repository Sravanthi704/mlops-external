"""Create versioned synthetic nightly water-tank datasets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CAPACITY = 100_000
THRESHOLD = 0.15 * CAPACITY
SEED = 42

def make_version(n=250, version=1, seed=SEED):
    rng = np.random.default_rng(seed + version * 100)
    rows = []
    day_types = np.array(["weekday", "weekend", "holiday"])
    probs = [0.68, 0.25, 0.07]
    for i in range(n):
        day = rng.choice(day_types, p=probs)
        households = int(rng.integers(35, 121))
        # Per-household demand varies by day type and season.
        per_household_hour = rng.normal(34.0, 6.0)
        if day == "weekend":
            per_household_hour *= 1.08
        elif day == "holiday":
            per_household_hour *= 1.15
        if version == 2:  # summer scenario: ~35% more household demand
            per_household_hour *= 1.35
        avg_hourly = max(80.0, households * per_household_hour * rng.normal(1.0, 0.08))
        tank_level = float(rng.uniform(22_000, 96_000))
        pump_hours = float(rng.choice([0, 0.5, 1, 1.5, 2, 2.5, 3], p=[.10,.08,.16,.22,.20,.14,.10]))
        pump_rate = float(rng.normal(11_500, 1_500))
        pump_rate = float(np.clip(pump_rate, 7_000, 15_000))
        leakage = float(np.clip(rng.gamma(2.0, 75.0), 10, 600))
        # Eight overnight hours, with demand uncertainty and occasional high-use events.
        demand_multiplier = rng.lognormal(mean=0.0, sigma=0.13)
        if rng.random() < 0.08:
            demand_multiplier *= rng.uniform(1.25, 1.65)
        actual_demand = avg_hourly * 10 * demand_multiplier
        actual_leakage = leakage * 10 * rng.uniform(0.85, 1.2)
        actual_pump = pump_hours * pump_rate * rng.uniform(0.85, 1.05)
        morning_level = tank_level + actual_pump - actual_demand - actual_leakage
        # Small operational uncertainty (unmetered draw/measurement error).
        morning_level += rng.normal(0, 900)
        shortage = int(morning_level < THRESHOLD)
        rows.append({
            "record_id": f"V{version}-{i+1:03d}",
            "tank_level_8pm_litres": round(tank_level, 2),
            "connected_households": households,
            "avg_hourly_consumption_litres": round(avg_hourly, 2),
            "expected_pumping_duration_hours": pump_hours,
            "expected_pumping_rate_lph": round(pump_rate, 2),
            "leakage_estimate_lph": round(leakage, 2),
            "day_type": day,
            "morning_water_shortage": shortage
        })
    return pd.DataFrame(rows)

def main():
    RAW.mkdir(parents=True, exist_ok=True)
    v1 = make_version(version=1)
    v2 = make_version(version=2)
    v1.to_csv(RAW / "water_tank_v1.csv", index=False)
    v2.to_csv(RAW / "water_tank_v2_summer.csv", index=False)
    manifest = {
        "tank_capacity_litres": CAPACITY,
        "shortage_threshold_litres": THRESHOLD,
        "versions": {
            "V1": {"file": "water_tank_v1.csv", "rows": len(v1), "description": "Baseline mixed-demand synthetic nights."},
            "V2": {"file": "water_tank_v2_summer.csv", "rows": len(v2), "description": "Summer scenario; per-household demand multiplier increased by 35%."}
        },
        "target_definition": "1 when simulated 6:00 AM tank level is below 15,000 litres; otherwise 0.",
        "seed": SEED
    }
    (RAW / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Created V1 and V2 datasets:", len(v1), len(v2))
    print("V1 shortage rate:", round(v1.morning_water_shortage.mean(), 3))
    print("V2 shortage rate:", round(v2.morning_water_shortage.mean(), 3))

if __name__ == "__main__":
    main()
