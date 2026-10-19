
"""Feature selection and leakage prevention for FleetMind AI."""

import pandas as pd


# Features available when a shipment risk prediction is requested.
APPROVED_FEATURES = [
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


# These columns must never be used as model predictors.
# They contain the target or information known after dispatch.
FORBIDDEN_PREDICTORS = [
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Delivery Status",
    "Order Status",
    "shipping date (DateOrders)",
]


def exclude_forbidden_predictors(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Remove target and post-outcome columns without changing input."""

    return frame.drop(
        columns=FORBIDDEN_PREDICTORS,
        errors="ignore",
    ).copy()


def select_model_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Select only approved, prediction-time features."""

    missing = [
        column
        for column in APPROVED_FEATURES
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required model features: {missing}"
        )

    clean_frame = exclude_forbidden_predictors(frame)

    return clean_frame.loc[:, APPROVED_FEATURES].copy()
