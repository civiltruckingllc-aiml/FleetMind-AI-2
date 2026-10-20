"""Tests for FleetMind AI's fitted preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from src.train import (
    build_pipeline,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


@pytest.fixture
def fitted_preprocessor():
    data = {}

    for feature in NUMERIC_FEATURES:
        data[feature] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    for feature in CATEGORICAL_FEATURES:
        data[feature] = ["A", "B", "A", "B", "A", "B"]

    df = pd.DataFrame(data)
    df.loc[0, NUMERIC_FEATURES[0]] = np.nan
    df.loc[1, CATEGORICAL_FEATURES[0]] = np.nan

    y = [0, 1, 0, 1, 0, 1]

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(df, y)

    return pipeline.named_steps["preprocessor"], df


def test_missing_values_are_imputed(fitted_preprocessor):
    preprocessor, df = fitted_preprocessor

    transformed = preprocessor.transform(df)

    assert np.isfinite(transformed).all()


def test_categorical_features_are_encoded(fitted_preprocessor):
    preprocessor, df = fitted_preprocessor

    transformed = preprocessor.transform(df)

    assert transformed.shape[1] > len(NUMERIC_FEATURES)


def test_numerical_features_are_scaled(fitted_preprocessor):
    preprocessor, df = fitted_preprocessor

    transformed = preprocessor.transform(df)

    numeric_values = transformed[:, :len(NUMERIC_FEATURES)]

    assert np.allclose(numeric_values.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(numeric_values.std(axis=0), 1, atol=1e-7)


