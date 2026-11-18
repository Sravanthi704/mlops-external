# Overnight Residential Water Shortage Prediction — MLOps Project

A complete, reproducible MLOps implementation for predicting whether a 100,000-litre residential overhead tank will fall below **15% capacity (15,000 L) by 6:00 AM**, using information available at 8:00 PM.

## 1. Project contents

```text
water_tank_mlops_project/
├── data/raw/                 # Version 1 and Version 2 synthetic datasets
├── data/processed/           # Cleaned/encoded/scaled pipeline outputs
├── features/                 # Reusable feature-store CSV and metadata
├── models/                   # Trained model artifacts and metrics
├── reports/                  # Implementation PDF report
├── evidence/                 # Prediction request/response and local run evidence
├── src/
│   ├── generate_data.py      # Dataset creation and versioning
│   ├── pipeline.py           # Preprocessing and feature engineering
│   ├── train.py              # Model training, evaluation, MLflow registry
│   └── api.py                # FastAPI serving application
├── tests/                    # Unit and API tests
├── .github/workflows/        # GitHub Actions CI
├── requirements.txt
└── README.md
```

## 2. Quick start (Windows PowerShell)

Open PowerShell in this folder.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/generate_data.py
python src/pipeline.py
python src/train.py
pytest -q
uvicorn src.api:app --reload
```

Open Swagger UI at `http://127.0.0.1:8000/docs`. Swagger is optional; the `/predict` endpoint can also be called directly.

Example request:

```json
{
  "tank_level_8pm_litres": 26000,
  "connected_households": 80,
  "avg_hourly_consumption_litres": 420,
  "expected_pumping_duration_hours": 1.5,
  "expected_pumping_rate_lph": 12000,
  "leakage_estimate_lph": 250,
  "day_type": "weekday"
}
```

Example response shape:

```json
{
  "prediction": "Shortage Risk",
  "shortage_probability": 0.82,
  "threshold_litres": 15000
}
```

The probability is model-dependent; run the trained model to obtain the actual value.

## 3. Dataset versions

- **V1:** 250 synthetic nightly records with a mixed seasonal demand profile.
- **V2:** 250 records generated with summer household demand approximately 35% higher than V1's base demand. This is a deliberate data-drift scenario.
- Both datasets include the original evening conditions and `morning_water_shortage` target.
- `data/raw/dataset_manifest.json` documents row counts, schema and version differences.

The synthetic data is for education and demonstration, not for operational water-system decisions.

## 4. Feature engineering and feature store

The pipeline:
1. Validates physical ranges and the 100,000 L tank capacity.
2. Handles missing numeric values with median imputation and categorical values with most-frequent imputation.
3. One-hot encodes `day_type` and standard-scales numeric inputs.
4. Creates:
   - `expected_overnight_demand`
   - `expected_pump_supply`
   - `projected_water_buffer`
5. Writes reusable engineered features to `features/water_tank_feature_store.csv` and the transformed model matrix to `data/processed/`.

The feature store is a versioned, file-based teaching implementation. In a production system, this could be replaced by Feast or another online/offline feature store.

## 5. Model training and MLflow

`src/train.py` trains Decision Tree and Random Forest classifiers with a stratified train/test split. It reports precision, recall, F1-score and confusion matrices. The selection rule prioritizes shortage recall, then F1-score, because missing a real shortage is more costly than raising an extra warning.

MLflow tracking and model registration are implemented in the script. The selected model is logged and registered as `WaterShortageClassifier`. To inspect runs:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

Then open `http://127.0.0.1:5000`. If the MLflow server is not running, the API still loads the exported local model from `models/`.

## 6. API

- `GET /health` — health check
- `POST /predict` — predicts `Shortage Risk` or `Adequate Reserve`

The API validates that the tank level is between 0 and 100,000 L. It accepts the evening conditions only; it does not accept a morning reading.

## 7. CI/CD

GitHub Actions runs on every push and pull request:
- installs dependencies
- checks that generated tank levels do not exceed capacity
- runs preprocessing/training smoke tests and API tests

GitHub Actions evidence is generated only after you push this project to your own GitHub repository. Do not represent a local test run as a GitHub-hosted Actions run.

## 8. Important interpretation

This is a synthetic demonstration. The label is generated from a simulated overnight water-balance calculation with realistic variation in household demand, pumping, leakage and random demand shocks. It is not a calibrated forecast for a real colony. For real deployment, collect measured tank levels, flow-meter readings, pump schedules and verified shortage outcomes.
