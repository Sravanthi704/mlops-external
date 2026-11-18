"""Train/evaluate classifiers and register the selected model with MLflow."""
from pathlib import Path
import json
import warnings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from pipeline import NUMERIC, CATEGORICAL, build_preprocessor, engineer_features, validate_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_NAME = "WaterShortageClassifier"

def train():
    data_path = ROOT / "data/raw/water_tank_v1.csv"
    df = pd.read_csv(data_path)
    validate_data(df)
    df = engineer_features(df)
    X = df[NUMERIC + CATEGORICAL]
    y = df["morning_water_shortage"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    candidates = {
        "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=250, max_depth=7, min_samples_leaf=3, class_weight="balanced", random_state=42)
    }
    results, fitted = {}, {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, pred, labels=[0,1]).tolist(),
            "classification_report": classification_report(y_test, pred, zero_division=0, output_dict=True)
        }
        fitted[name] = pipe
    # Safety-oriented selection: prioritize shortage recall, then F1, then precision.
    selected = max(results, key=lambda n: (results[n]["recall"], results[n]["f1_score"], results[n]["precision"]))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted[selected], MODEL_DIR / "selected_model.joblib")
    joblib.dump(fitted, MODEL_DIR / "all_models.joblib")
    summary = {"selected_model": selected, "selection_rule": "Highest shortage recall, then F1, then precision.", "models": results}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: {m: v[m] for m in ["precision","recall","f1_score","confusion_matrix"]} for k,v in results.items()}, indent=2))
    print("Selected:", selected)

    # MLflow tracking + model registry. If MLflow is not installed/configured, local artifacts remain usable.
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Overnight-Water-Shortage")
        with mlflow.start_run(run_name=selected) as run:
            mlflow.log_param("selected_model", selected)
            mlflow.log_param("dataset_version", "V1")
            mlflow.log_param("selection_rule", "recall_then_f1_then_precision")
            for model_name, metrics in results.items():
                for metric in ["precision", "recall", "f1_score"]:
                    mlflow.log_metric(f"{model_name}_{metric}", metrics[metric])
            mlflow.sklearn.log_model(
                sk_model=fitted[selected],
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )
            mlflow.log_artifact(str(MODEL_DIR / "metrics.json"))
            evidence = {"tracking_uri": mlflow.get_tracking_uri(), "run_id": run.info.run_id, "registered_model": MODEL_NAME, "selected_model": selected}
            (ROOT / "evidence/mlflow_run.json").write_text(json.dumps(evidence, indent=2))
            print("MLflow model registered:", MODEL_NAME, "run:", run.info.run_id)
    except Exception as exc:
        warnings.warn(f"MLflow registration was not completed in this environment: {exc}. Run `python src/train.py` after installing requirements.")
        (ROOT / "evidence/mlflow_run.json").write_text(json.dumps({
            "status": "not_registered_in_build_environment",
            "instruction": "Install requirements and rerun python src/train.py to create MLflow tracking run and registered model.",
            "reason": str(exc)
        }, indent=2))
    return summary

if __name__ == "__main__":
    train()
