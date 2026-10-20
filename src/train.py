
"""FleetMind AI: group-aware training and MLflow tracking."""

import argparse
import uuid
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import APPROVED_FEATURES, select_model_features


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    """Load training settings from the YAML configuration."""
    with open(CONFIG_PATH, "r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)
DEFAULT_CSV = PROJECT_ROOT / "data" / "raw" / "DataCoSupplyChainDataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
TARGET = "Late_delivery_risk"
GROUP_COLUMN = "Order Id"

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
]

CATEGORICAL_FEATURES = [
    column for column in APPROVED_FEATURES
    if column not in NUMERIC_FEATURES
]


def build_pipeline(model):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

def build_model_from_config(settings, random_state):
    """Create a machine learning model using YAML settings."""
    settings = settings.copy()
    algorithm = settings.pop("algorithm")

    if algorithm == "logistic_regression":
        return LogisticRegression(
            **settings,
            random_state=random_state
        )

    elif algorithm == "random_forest":
        return RandomForestClassifier(
            **settings,
            random_state=random_state
        )

    elif algorithm == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            **settings,
            random_state=random_state
        )

    else:
        raise ValueError(
            f"Unsupported model algorithm: {algorithm}"
        )
def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def train(csv_path):
    config = load_config()
    random_state = config["project"]["random_state"]
    model_configs = config["training"]["models"]

    print(f"Loaded {len(model_configs)} model configurations from YAML.")
    print("\nLoading FleetMind AI dataset...")
    df = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)

    required = set(APPROVED_FEATURES + [TARGET, GROUP_COLUMN])
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = select_model_features(df)
    y = df[TARGET].astype(int)
    groups = df[GROUP_COLUMN]

    # Hold out 20% of orders for final testing.
    outer_split = GroupShuffleSplit(
        n_splits=1,
        test_size=config["data"]["test_size"],
        random_state=random_state,
    )
    train_val_idx, test_idx = next(
        outer_split.split(X, y, groups)
    )

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # Split remaining orders into training and validation.
    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=config["data"]["validation_size"],
        random_state=random_state + 1,
    )
    train_idx, val_idx = next(
        inner_split.split(X_train_val, y_train_val, groups_train_val)
    )

    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]
    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]

    print(f"Training records: {len(X_train):,}")
    print(f"Validation records: {len(X_val):,}")
    print(f"Testing records: {len(X_test):,}")

    experiments = {
        name: build_model_from_config(settings, random_state)
        for name, settings in model_configs.items()
    }


    mlflow.set_tracking_uri(
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


    training_session_id = uuid.uuid4().hex
    best_name = None
    best_score = -1.0
    best_model = None
    results = []

    for name, estimator in experiments.items():
        print(f"\nTraining: {name}")

        pipeline = build_pipeline(estimator)

        with mlflow.start_run(run_name=name):
            mlflow.set_tag("training_session_id", training_session_id)
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("feature_count", len(APPROVED_FEATURES))
            mlflow.log_param("dataset_name", "DataCo Smart Supply Chain")
            mlflow.log_param("target_column", TARGET)
            mlflow.log_param("group_column", GROUP_COLUMN)
            mlflow.log_param("split_strategy", "GroupShuffleSplit_by_Order_Id")
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("validation_rows", len(X_val))
            mlflow.log_param("testing_rows", len(X_test))
            mlflow.log_param("test_split", config["data"]["test_size"])
            mlflow.log_param("validation_split", config["data"]["validation_size"])

            for parameter, value in estimator.get_params().items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    mlflow.log_param(parameter, str(value))

            pipeline.fit(X_train, y_train)
            metrics = calculate_metrics(pipeline, X_val, y_val)

            mlflow.log_metrics({
                f"val_{key}": float(value)
                for key, value in metrics.items()
            })

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_artifact = MODEL_DIR / f"{name}.joblib"
            joblib.dump(pipeline, model_artifact)
            mlflow.log_artifact(str(model_artifact), artifact_path="trained_model")

            result = {"model": name, **metrics}
            results.append(result)

            print(
                f"F1: {metrics['f1']:.4f} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f}"
            )

            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                best_name = name
                best_model = pipeline

    # Use MLflow's experiment records to verify the selected run.
    runs = mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]],
        order_by=["metrics.val_f1 DESC"],
    )
    print("\nTop tracked MLflow runs:")
    print(runs[["tags.mlflow.runName", "metrics.val_f1"]].head())

    # Select the best configuration from this training session.
    current_runs = runs[
        runs["tags.training_session_id"] == training_session_id
    ]

    mlflow_best_name = current_runs.iloc[0]["tags.mlflow.runName"]
    mlflow_best_score = float(current_runs.iloc[0]["metrics.val_f1"])

    if mlflow_best_name != best_name:
        raise RuntimeError("MLflow ranking disagrees with training selection.")

    best_name = mlflow_best_name
    best_score = mlflow_best_score

    print(f"\nBest model: {best_name}")
    print(f"Validation F1: {best_score:.4f}")

    # Final evaluation on untouched test data.
    test_metrics = calculate_metrics(best_model, X_test, y_test)

    print("\nFinal test metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    metadata = {
        "best_model": best_name,
        "validation_f1": float(best_score),
        "test_metrics": {
            key: float(value) for key, value in test_metrics.items()
        },
        "features": APPROVED_FEATURES,
        "experiments": results,
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nModel saved successfully!")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
    )
    args = parser.parse_args()
    train(args.csv)
