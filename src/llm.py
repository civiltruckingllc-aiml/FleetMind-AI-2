
"""Nebius LLM integration for FleetMind AI."""

import json
import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

SHIPPING_MODES = (
    "Standard Class", "First Class", "Second Class", "Same Day"
)
MARKETS = ("Europe", "USCA", "LATAM", "Pacific Asia", "Africa")
SEGMENTS = ("Consumer", "Corporate", "Home Office")


class ShipmentRequest(BaseModel):
    intent: Literal["predict", "out_of_scope"]
    shipping_mode: str | None = None
    order_region: str | None = None
    order_country: str | None = None
    market: str | None = None
    customer_segment: str | None = None
    scheduled_days: int | None = Field(default=None, ge=0, le=30)
    quantity: int | None = Field(default=None, ge=1, le=100)
    product_price: float | None = Field(default=None, ge=0)
    discount_rate: float | None = Field(default=None, ge=0, le=1)


def get_client() -> OpenAI:
    api_key = os.getenv("NEBIUS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is missing. Add it to your .env file."
        )

    return OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1/",
        api_key=api_key,
        timeout=30.0,
        max_retries=1,
    )


def get_model_id() -> str:
    model_id = os.getenv("NEBIUS_MODEL_ID")
    if not model_id:
        raise RuntimeError(
            "NEBIUS_MODEL_ID is missing. Configure it in .env."
        )
    return model_id


def extract_shipment_request(message: str) -> ShipmentRequest:
    """Convert natural language into validated shipment fields."""

    if not message.strip():
        raise ValueError("Please enter a shipment description.")

    system_prompt = f"""
You extract shipment details for FleetMind AI.

Return ONLY a valid JSON object with these keys:
intent, shipping_mode, order_region, order_country,
market, customer_segment, scheduled_days, quantity,
product_price, discount_rate.

Set intent to "predict" for shipment dispatch risk requests.
Set intent to "out_of_scope" for unrelated requests.

Use null for information that was not provided.
Never invent missing shipment details.

Allowed shipping modes: {list(SHIPPING_MODES)}
Allowed markets: {list(MARKETS)}
Allowed customer segments: {list(SEGMENTS)}

Convert percentages such as 10% into decimal rates such as 0.10.
Do not calculate dispatch risk. Only extract information.
"""

    response = get_client().chat.completions.create(
        model=get_model_id(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Nebius returned an empty response.")

    try:
        parsed = ShipmentRequest.model_validate_json(content)
    except ValidationError as exc:
        raise ValueError(
            "The shipment information could not be validated."
        ) from exc

    if parsed.intent == "out_of_scope":
        return parsed

    required = (
        "shipping_mode", "order_region", "order_country",
        "market", "customer_segment", "scheduled_days",
        "quantity", "product_price", "discount_rate",
    )

    missing = [
        field for field in required
        if getattr(parsed, field) is None
    ]

    if missing:
        raise ValueError(
            "Please provide the missing shipment details: "
            + ", ".join(missing)
        )

    if parsed.shipping_mode not in SHIPPING_MODES:
        raise ValueError("Invalid shipping mode.")

    if parsed.market not in MARKETS:
        raise ValueError("Invalid market.")

    if parsed.customer_segment not in SEGMENTS:
        raise ValueError("Invalid customer segment.")

    return parsed


def explain_prediction(
    shipment: dict,
    probability: float,
    predicted_class: int,
) -> str:
    """Explain the model result without changing its prediction."""

    if not 0 <= probability <= 1:
        raise ValueError("Prediction probability must be between 0 and 1.")

    payload = {
        "shipment": shipment,
        "late_dispatch_probability": round(probability, 4),
        "predicted_class": (
            "Late Dispatch" if predicted_class == 1
            else "On-Time Dispatch"
        ),
    }

    response = get_client().chat.completions.create(
        model=get_model_id(),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are FleetMind AI, a shipment dispatch risk advisor. "
                    "Explain the provided machine learning result clearly "
                    "in 2-4 sentences. Do not change the probability or "
                    "predicted class. Do not claim to know actual delivery "
                    "times, live traffic, weather, or tracking information. "
                    "Do not invent causes, feature importance, or certainty. "
                    "Offer practical monitoring suggestions when appropriate."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(payload),
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content or (
        "An explanation was not available."
    )
