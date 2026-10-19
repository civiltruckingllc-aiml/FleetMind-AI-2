"""Tests for FleetMind AI model predictions."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

FEATURES = [
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Market",
    "Customer Segment",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
]

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Market",
    "Customer Segment",
]


@pytest.fixture
def sample_shipment():
    return {
        "Days for shipment (scheduled)": 4,
        "Shipping Mode": "Standard Class",
        "Order Region": "Western Europe",
        "Order Country": "France",
        "Market": "Europe",
        "Customer Segment": "Consumer",
        "Order Item Quantity": 1,
        "Order Item Product Price": 100.0,
        "Order Item Discount Rate": 0.0,
    }


@pytest.fixture(scope="module")
def model_and_features():
    """Load production artifacts or create a temporary CI test model."""

    if MODEL_PATH.exists() and METADATA_PATH.exists():
        model = joblib.load(MODEL_PATH)

        with open(METADATA_PATH, encoding="utf-8") as file:
            metadata = json.load(file)

        return model, metadata["features"]

    # GitHub Actions fallback: small synthetic dataset.
    examples = []

    for i in range(20):
        examples.append({
            "Days for shipment (scheduled)": 2 + (i % 4),
            "Shipping Mode": "Standard Class",
            "Order Region": "Western Europe",
            "Order Country": "France",
            "Market": "Europe",
            "Customer Segment": "Consumer",
            "Order Item Quantity": 1 + (i % 3),
            "Order Item Product Price": 50.0 + i * 10,
            "Order Item Discount Rate": 0.0,
        })

    X = pd.DataFrame(examples)[FEATURES]
    y = [i % 2 for i in range(20)]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=10,
            random_state=42,
        )),
    ])

    model.fit(X, y)

    return model, FEATURES


def test_model_returns_valid_class(model_and_features, sample_shipment):
    model, features = model_and_features
    shipment = pd.DataFrame([sample_shipment])[features]

    prediction = int(model.predict(shipment)[0])

    assert prediction in (0, 1)


def test_model_returns_valid_probability(model_and_features, sample_shipment):
    model, features = model_and_features
    shipment = pd.DataFrame([sample_shipment])[features]

    probabilities = model.predict_proba(shipment)[0]

    assert len(probabilities) == 2
    assert all(0.0 <= float(p) <= 1.0 for p in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)
