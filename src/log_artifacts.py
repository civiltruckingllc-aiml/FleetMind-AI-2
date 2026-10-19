
"""Log existing FleetMind AI model artifacts to MLflow."""

import json
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "best_model.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

# Verify that the files exist.
for file_path in [MODEL_FILE, METADATA_FILE]:
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing artifact: {file_path}")

with METADATA_FILE.open("r", encoding="utf-8") as file:
    metadata = json.load(file)

# Use the same SQLite database as our training pipeline.
mlflow.set_tracking_uri(
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("FleetMind_Dispatch_Risk")

with mlflow.start_run(run_name="best_model_artifacts"):
    mlflow.set_tag("purpose", "Post-training artifact registration")
    mlflow.set_tag("source_model", metadata["best_model"])

    mlflow.log_param("selected_model", metadata["best_model"])

    mlflow.log_metric(
        "selected_validation_f1",
        float(metadata["validation_f1"])
    )

    for metric, value in metadata["test_metrics"].items():
        mlflow.log_metric(f"test_{metric}", float(value))

    mlflow.log_artifact(str(MODEL_FILE), artifact_path="model")
    mlflow.log_artifact(str(METADATA_FILE), artifact_path="metadata")

    print("SUCCESS: Model artifacts logged to MLflow!")
    print(f"Selected model: {metadata['best_model']}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
