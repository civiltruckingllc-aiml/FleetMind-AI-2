# FleetMind AI — Intelligent Shipment Dispatch Risk Advisor

An end-to-end machine learning application developed for the TripleTen AI and Machine Learning Engineering capstone.

FleetMind AI predicts whether a newly placed shipment order is at risk of being dispatched late. It combines a trained scikit-learn classification model, MLflow experiment tracking, a Streamlit application, and a Nebius-powered natural-language interface.

## Project Overview

FleetMind AI helps demonstrate how machine learning can support shipment planning and dispatch-risk assessment.

Users can enter shipment details manually or describe a shipment in natural language.

The application:
1. Collects shipment information.
2. Validates the required input fields.
3. Predicts late-dispatch risk using a trained Random Forest model.
4. Displays the predicted classification and probability.
5. Uses Nebius to generate a natural-language explanation.

The language model does not calculate the risk score. The trained classification model performs the prediction.

## Dataset

Dataset: DataCo Smart Supply Chain

Source: Mendeley Data, DataCo Global

License: CC BY 4.0

The original dataset contains approximately 180,519 order-item records and 53 columns.

The dataset is not included in this repository. Obtain it from its original publisher before retraining.

### Prediction Target

`Late_delivery_risk`

This project interprets the available outcome as late shipment dispatch risk, not actual delivery-arrival delay.

### Leakage Prevention

The following information is excluded from model predictors:

- Late_delivery_risk
- Days for shipping (real)
- Delivery Status
- Order Status
- shipping date (DateOrders)

Only approved shipment information available at prediction time is used.

### Model Features

- Days for shipment (scheduled)
- Shipping Mode
- Order Region
- Order Country
- Market
- Customer Segment
- Order Item Quantity
- Order Item Product Price
- Order Item Discount Rate

## Machine Learning

Framework: scikit-learn

Selected model: Random Forest classifier

Data splitting uses Order Id grouping to reduce leakage between order-item records belonging to the same order.

Multiple model configurations were evaluated and tracked using MLflow.

The primary model-selection metric was validation F1 score.

### Model Performance

| Metric | Held-out test result |
|---|---:|
| Accuracy | 0.6875 |
| Precision | 0.8022 |
| Recall | 0.5709 |
| F1 score | 0.6671 |
| ROC-AUC | 0.7249 |

These metrics describe evaluation on the held-out test data, not guaranteed performance on future shipments.

## Application Architecture

Streamlit interface
    |
    +-- Manual shipment entry
    |
    +-- Natural-language shipment entry
            |
            v
       Nebius LLM
            |
            v
       Structured extraction
            |
            v
       Pydantic validation
            |
            v
       scikit-learn pipeline
            |
            v
       Risk classification and probability
            |
            v
       Streamlit results and Nebius explanation

## Technology Stack

- Python 3.12
- pandas and NumPy
- scikit-learn
- MLflow
- Streamlit
- Pydantic
- Nebius Token Factory
- OpenAI-compatible API client
- pytest
- Git and GitHub

## Local Installation

### 1. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

Idris Roberts

TripleTen AI and Machine Learning Engineering Capstone

## Reproducing the Trained Model

The original DataCo dataset is not included in this repository.

1. Download the DataCo Smart Supply Chain dataset from Mendeley Data.
2. Save the CSV as data/raw/DataCoSupplyChainDataset.csv.
3. Install the Python dependencies.
4. Run the training script from the project root.

Training command:

    python src/train.py

The training process evaluates five model configurations using group-aware data splitting and MLflow experiment tracking.

The training script generates:

- models/best_model.joblib
- models/model_metadata.json

Both files are required to run the prediction application.

Start the Streamlit application:

    python -m streamlit run src/app.py

Run automated tests:

    python -m pytest tests/ -v
