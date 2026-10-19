
"""Automated preprocessing tests for FleetMind AI."""

import pandas as pd
import pytest

from src.preprocess import (
    APPROVED_FEATURES,
    exclude_forbidden_predictors,
    select_model_features,
)


def test_excludes_leaky_columns():
    """Target and post-outcome columns must be excluded."""
    df = pd.DataFrame({
        "Late_delivery_risk": [1],
        "Delivery Status": ["Late"],
        "Days for shipping (real)": [6],
        "Order Status": ["COMPLETE"],
        "shipping date (DateOrders)": ["2026-01-01"],
        "Type": ["DEBIT"],
    })

    result = exclude_forbidden_predictors(df)

    assert list(result.columns) == ["Type"]


def test_original_dataframe_unchanged():
    """Preprocessing must not modify the original dataset."""
    df = pd.DataFrame({
        "Late_delivery_risk": [1],
        "Type": ["DEBIT"],
    })

    before = df.copy(deep=True)

    exclude_forbidden_predictors(df)

    pd.testing.assert_frame_equal(df, before)


def test_selects_only_approved_features():
    """Model input must contain exactly the approved features."""
    data = {
        feature: [1]
        for feature in APPROVED_FEATURES
    }
    data["Late_delivery_risk"] = [1]
    data["Delivery Status"] = ["Late"]

    df = pd.DataFrame(data)

    result = select_model_features(df)

    assert list(result.columns) == APPROVED_FEATURES
    assert "Late_delivery_risk" not in result.columns
    assert "Delivery Status" not in result.columns


def test_missing_required_feature_raises_error():
    """Missing required features must produce a clear error."""
    data = {
        feature: [1]
        for feature in APPROVED_FEATURES
        if feature != "Shipping Mode"
    }

    df = pd.DataFrame(data)

    with pytest.raises(
        ValueError,
        match="Missing required model features",
    ):
        select_model_features(df)
