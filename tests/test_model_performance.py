"""Minimum performance requirement for the trained FleetMind AI model."""

import json
from pathlib import Path

import pytest


METADATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model_metadata.json"
)


def test_model_meets_minimum_f1():
    if not METADATA_PATH.exists():
        pytest.skip(
            "Trained model metadata is unavailable in this environment."
        )

    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    test_f1 = metadata["test_metrics"]["f1"]

    assert 0.0 <= test_f1 <= 1.0
    assert test_f1 >= 0.60, (
        f"Model F1 score {test_f1:.4f} is below the 0.60 threshold."
    )
