
"""FleetMind AI - Intelligent Shipment Dispatch Risk Advisor."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from llm import extract_shipment_request, explain_prediction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="FleetMind AI",
    page_icon="🚚",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def create_shipment_dataframe(details, expected_features):
    """Convert validated shipment details to model input."""

    shipment = pd.DataFrame([{
        "Days for shipment (scheduled)": details["scheduled_days"],
        "Shipping Mode": details["shipping_mode"],
        "Order Region": details["order_region"],
        "Order Country": details["order_country"],
        "Market": details["market"],
        "Customer Segment": details["customer_segment"],
        "Order Item Quantity": details["quantity"],
        "Order Item Product Price": details["product_price"],
        "Order Item Discount Rate": details["discount_rate"],
    }])

    return shipment[expected_features]


def show_prediction(details, model, metadata, use_ai=False):
    """Run the ML model and display its prediction."""

    shipment = create_shipment_dataframe(
        details,
        metadata["features"],
    )

    prediction = int(model.predict(shipment)[0])
    probability = float(model.predict_proba(shipment)[0, 1])

    st.divider()
    st.header("Dispatch Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Late-Dispatch Risk",
            f"{probability:.1%}",
        )

    with col2:
        st.metric(
            "Predicted Class",
            "Late Dispatch" if prediction == 1
            else "On-Time Dispatch",
        )

    if prediction == 1:
        st.warning(
            "The model predicts elevated late-dispatch risk. "
            "Review the shipment schedule and operational constraints."
        )
    else:
        st.success(
            "The model predicts a lower likelihood of late dispatch. "
            "Continue standard monitoring."
        )

    st.caption(
        "This is a historical-data-based estimate of late dispatch, "
        "not actual delivery arrival time or live shipment tracking."
    )

    if use_ai:
        st.subheader("AI-Generated Explanation")

        try:
            with st.spinner("Nebius is preparing an explanation..."):
                explanation = explain_prediction(
                    shipment=details,
                    probability=probability,
                    predicted_class=prediction,
                )

            st.write(explanation)

        except Exception:
            st.warning(
                "The prediction succeeded, but the AI explanation "
                "is temporarily unavailable. Please try again."
            )


st.title("🚚 FleetMind AI")
st.subheader("Intelligent Shipment Dispatch Risk Advisor")

st.write(
    "Predict late shipment dispatch risk using machine learning "
    "and explain results using Nebius AI."
)

st.info(
    "FleetMind AI predicts late dispatch risk, not actual "
    "customer delivery arrival time."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error(
        "Trained model or metadata file is missing. "
        "Complete model training before starting the app."
    )
    st.stop()

try:
    model = load_model()
    metadata = load_metadata()
except Exception:
    st.error("Unable to load the trained model or metadata.")
    st.stop()

st.divider()

manual_tab, ai_tab = st.tabs([
    "Manual Shipment Prediction",
    "AI-Powered Shipment Prediction",
])


with manual_tab:

    st.header("Enter Shipment Information")

    with st.form("manual_shipment_form"):

        col1, col2 = st.columns(2)

        with col1:
            shipping_mode = st.selectbox(
                "Shipping Mode",
                ["Standard Class", "First Class",
                 "Second Class", "Same Day"],
            )

            order_region = st.text_input(
                "Order Region",
                value="Western Europe",
            )

            order_country = st.text_input(
                "Order Country",
                value="France",
            )

            market = st.selectbox(
                "Market",
                ["Europe", "USCA", "LATAM",
                 "Pacific Asia", "Africa"],
            )

            customer_segment = st.selectbox(
                "Customer Segment",
                ["Consumer", "Corporate", "Home Office"],
            )

        with col2:
            scheduled_days = st.number_input(
                "Scheduled Shipping Days",
                min_value=0,
                max_value=30,
                value=4,
            )

            quantity = st.number_input(
                "Order Item Quantity",
                min_value=1,
                max_value=100,
                value=1,
            )

            product_price = st.number_input(
                "Product Price ($)",
                min_value=0.0,
                value=100.0,
                step=1.0,
            )

            discount_rate = st.number_input(
                "Discount Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
            )

        manual_submitted = st.form_submit_button(
            "Predict Dispatch Risk",
            use_container_width=True,
        )

    if manual_submitted:

        manual_details = {
            "shipping_mode": shipping_mode,
            "order_region": order_region.strip(),
            "order_country": order_country.strip(),
            "market": market,
            "customer_segment": customer_segment,
            "scheduled_days": int(scheduled_days),
            "quantity": int(quantity),
            "product_price": float(product_price),
            "discount_rate": float(discount_rate),
        }

        if not manual_details["order_region"] or not manual_details["order_country"]:
            st.error("Order region and country cannot be empty.")
        else:
            show_prediction(
                manual_details,
                model,
                metadata,
                use_ai=False,
            )


with ai_tab:

    st.header("Describe Your Shipment")

    st.write(
        "Enter shipment details in everyday language. "
        "Nebius AI will extract the information and pass "
        "it to our trained machine learning model."
    )

    example = (
        "Predict the late dispatch risk for a shipment using "
        "Standard Class shipping to France in Western Europe, "
        "market Europe, for a Consumer. The scheduled shipping "
        "time is 4 days, quantity is 1, product price is $100, "
        "and the discount rate is 0%."
    )

    with st.form("ai_shipment_form"):

        user_description = st.text_area(
            "Shipment Description",
            value=example,
            height=180,
        )

        ai_submitted = st.form_submit_button(
            "Analyze Shipment with AI",
            use_container_width=True,
        )

    if ai_submitted:

        if not user_description.strip():
            st.error("Please enter a shipment description.")

        else:
            try:
                with st.spinner(
                    "Nebius is extracting shipment details..."
                ):
                    extracted = extract_shipment_request(
                        user_description
                    )

                if extracted.intent == "out_of_scope":
                    st.warning(
                        "This request is outside FleetMind AI's "
                        "shipment dispatch risk assessment scope."
                    )

                else:
                    details = extracted.model_dump(
                        exclude={"intent"}
                    )

                    st.success(
                        "Shipment information extracted successfully."
                    )

                    with st.expander(
                        "View Extracted Shipment Information",
                        expanded=True,
                    ):
                        st.json(details)

                    show_prediction(
                        details,
                        model,
                        metadata,
                        use_ai=True,
                    )

            except ValueError as error:
                st.warning(str(error))

            except Exception:
                st.error(
                    "The AI service could not process this request. "
                    "Check your Nebius configuration and try again."
                )


st.divider()

with st.expander("Model Information"):
    st.write("Selected Model:", metadata["best_model"])
    st.write(
        "Validation F1:",
        round(metadata["validation_f1"], 4),
    )
    st.write("Input Features:", metadata["features"])
    st.caption(
        "The machine learning model generates the risk estimate. "
        "The language model extracts input details and explains "
        "the result; it does not calculate the risk."
    )
