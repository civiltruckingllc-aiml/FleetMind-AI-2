"""Tests for FleetMind AI's Nebius language-model interface."""

from types import SimpleNamespace

import pytest

from src import llm


def mock_nebius_response(content):
    """Build a simulated Nebius chat response."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content)
            )
        ]
    )


class MockClient:
    """Simulate the Nebius chat completions API."""

    def __init__(self, content):
        self.content = content
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        return mock_nebius_response(self.content)


def test_extracts_valid_shipment(monkeypatch):
    """A complete shipment description should be parsed correctly."""

    response_json = """
    {
        "intent": "predict",
        "shipping_mode": "Standard Class",
        "order_region": "Western Europe",
        "order_country": "France",
        "market": "Europe",
        "customer_segment": "Consumer",
        "scheduled_days": 4,
        "quantity": 1,
        "product_price": 100.0,
        "discount_rate": 0.0
    }
    """

    monkeypatch.setattr(
        llm, "get_client", lambda: MockClient(response_json)
    )
    monkeypatch.setattr(
        llm, "get_model_id", lambda: "mock-model"
    )

    result = llm.extract_shipment_request(
        "Predict dispatch risk for my shipment."
    )

    assert result.intent == "predict"
    assert result.shipping_mode == "Standard Class"
    assert result.scheduled_days == 4
    assert result.product_price == 100.0


def test_rejects_missing_shipment_information(monkeypatch):
    """Incomplete shipment information should request clarification."""

    response_json = """
    {
        "intent": "predict",
        "shipping_mode": "Standard Class",
        "order_region": null,
        "order_country": null,
        "market": null,
        "customer_segment": null,
        "scheduled_days": null,
        "quantity": null,
        "product_price": null,
        "discount_rate": null
    }
    """

    monkeypatch.setattr(
        llm, "get_client", lambda: MockClient(response_json)
    )
    monkeypatch.setattr(
        llm, "get_model_id", lambda: "mock-model"
    )

    with pytest.raises(ValueError, match="missing shipment details"):
        llm.extract_shipment_request(
            "I have a Standard Class shipment."
        )
