
"""FleetMind AI: group-aware training and MLflow tracking."""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

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
        n_splits=1, test_size=0.20, random_state=42
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
        n_splits=1, test_size=0.20, random_state=43
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
        "logistic_C_0.1": LogisticRegression(
            C=0.1, max_iter=1000, random_state=42
        ),
        "logistic_C_1.0": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "random_forest_100": RandomForestClassifier(
            n_estimators=100, max_depth=12,
            random_state=42, n_jobs=-1
        ),
        "random_forest_200": RandomForestClassifier(
            n_estimators=200, max_depth=18,
            random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=8, random_state=42
        ),
    }


    mlflow.set_tracking_uri(
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
    )
    mlflow.set_experiment("FleetMind_Dispatch_Risk")


    best_name = None
    best_score = -1.0
    best_model = None
    results = []

    for name, estimator in experiments.items():
        print(f"\nTraining: {name}")

        pipeline = build_pipeline(estimator)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("feature_count", len(APPROVED_FEATURES))

            for parameter, value in estimator.get_params().items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    mlflow.log_param(parameter, str(value))

            pipeline.fit(X_train, y_train)
            metrics = calculate_metrics(pipeline, X_val, y_val)

            mlflow.log_metrics({
                f"val_{key}": float(value)
                for key, value in metrics.items()
            })

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
        experiment_names=["FleetMind_Dispatch_Risk"],
        order_by=["metrics.val_f1 DESC"],
    )
    print("\nTop tracked MLflow runs:")
    print(runs[["tags.mlflow.runName", "metrics.val_f1"]].head())

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
