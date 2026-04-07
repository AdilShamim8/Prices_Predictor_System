"""
predictor.py — ZenML Step: Run Inference Against the Deployed Model
====================================================================

PURPOSE:
    Takes a JSON string of input data and sends it to the deployed MLflow prediction
    service for batch inference. Steps:
    1. Starts the MLflow service (no-op if already running).
    2. Parses the JSON string into a DataFrame with the expected feature columns.
    3. Converts the DataFrame to the format MLflow expects (array of dicts).
    4. Calls service.predict() and returns the prediction array.

USED BY:
    pipelines/deployment_pipeline.py  →  inference_pipeline() calls this as its final step.

INPUT:
    service (MLFlowDeploymentService) — handle to the running prediction server
    input_data (str)                  — JSON string from dynamic_importer step

OUTPUT:
    np.ndarray — array of predicted SalePrice values
"""

import json

import numpy as np
import pandas as pd
from zenml import step
from zenml.integrations.mlflow.services import MLFlowDeploymentService


@step(enable_cache=False)
def predictor(
    service: MLFlowDeploymentService,
    input_data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service.

    Args:
        service (MLFlowDeploymentService): The deployed MLFlow service for prediction.
        input_data (str): The input data as a JSON string.

    Returns:
        np.ndarray: The model's prediction.
    """

    # Start the service (should be a NOP if already started)
    service.start(timeout=10)

    # Load the input data from JSON string
    data = json.loads(input_data)

    # Extract the actual data and expected columns
    data.pop("columns", None)  # Remove 'columns' if it's present
    data.pop("index", None)  # Remove 'index' if it's present

    # Define the columns the model expects
    expected_columns = [
        "Order",
        "PID",
        "MS SubClass",
        "Lot Frontage",
        "Lot Area",
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Year Remod/Add",
        "Mas Vnr Area",
        "BsmtFin SF 1",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF",
        "1st Flr SF",
        "2nd Flr SF",
        "Low Qual Fin SF",
        "Gr Liv Area",
        "Bsmt Full Bath",
        "Bsmt Half Bath",
        "Full Bath",
        "Half Bath",
        "Bedroom AbvGr",
        "Kitchen AbvGr",
        "TotRms AbvGrd",
        "Fireplaces",
        "Garage Yr Blt",
        "Garage Cars",
        "Garage Area",
        "Wood Deck SF",
        "Open Porch SF",
        "Enclosed Porch",
        "3Ssn Porch",
        "Screen Porch",
        "Pool Area",
        "Misc Val",
        "Mo Sold",
        "Yr Sold",
    ]

    # Convert the data into a DataFrame with the correct columns
    df = pd.DataFrame(data["data"], columns=expected_columns)

    # Convert DataFrame to JSON list for prediction
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data_array = np.array(json_list)

    # Run the prediction
    prediction = service.predict(data_array)

    return prediction
